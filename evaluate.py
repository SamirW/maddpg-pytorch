import argparse
import torch
import time
import os
import pickle
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.logging import set_log
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = False  # torch.cuda.is_available()


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    run_dir = model_dir / "run{}".format(config.run_num)
    model_file = run_dir / "model.pt"
    log_dir = run_dir / 'logs'

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    maddpg = MADDPG.init_from_save(str(model_file))
    t = 0
    flip = False

    print("********Starting training********")
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):

        # Flip environment
        if ep_i == config.flip_ep:
            print("********Flipping********")
            # Reset replay buffer
            flip = True

        obs = env.reset(flip=flip)
        maddpg.prep_rollouts(device='cpu')

        maddpg.scale_noise(0)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):

            # rearrange observations to be per agent, and convert to torch
            # Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]

            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions]
                       for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            if et_i == (config.episode_length - 1):
                dones = dones + 1

            time.sleep(0.015)
            env.render()

            obs = next_obs
            t += config.n_rollout_threads

        if (ep_i + 1) == config.hard_distill_ep:
            print("************Distilling***********")

            maddpg.prep_rollouts(device='cpu')
            with open(str(run_dir / "replay_buffer.pkl"), 'rb') as input:
                distill_replay_buffer = pickle.load(input)
            maddpg.distill(config.num_distills, 1024, distill_replay_buffer, hard=True,
                           pass_actor=config.distill_pass_actor, pass_critic=config.distill_pass_critic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("run_num", help="Number of run")
    parser.add_argument("--log_comment",
                        default="", type=str,
                        help="Log file comment")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--display_every",
                        default=999999, type=int,
                        help="Display frequency")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=20, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--flip_ep",
                        default=10, type=int,
                        help="Episode at which to flip")
    parser.add_argument("--eval_ep",
                        default=999999, type=int,
                        help="Episode at which to start evaluating")
    parser.add_argument("--hard_distill_ep",
                        default=999, type=int,
                        help="Episode at which to hard distill")
    parser.add_argument("--num_distills",
                        default=1024, type=int,
                        help="Number of times to distill")
    parser.add_argument("--distill_pass_actor",
                        action="store_true",
                        default=False,
                        help="How long to skip actor updates")
    parser.add_argument("--distill_pass_critic",
                        action="store_true",
                        default=False,
                        help="How long to skip actor updates")
    parser.add_argument("--n_exploration_eps", default=3000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true',
                        default=True)
    parser.add_argument("--save_buffer",
                        action="store_true",
                        default=False)

    config = parser.parse_args()

    config.log_name = \
        "env::%s_seed::%s_comment::%s_log" % (
            config.env_id, str(config.seed), config.log_comment)

    run(config)
