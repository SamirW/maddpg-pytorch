import argparse
import torch
import os
import pickle
import numpy as np
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.logging import set_log
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = False  # torch.cuda.is_available()


def make_parallel_env(env_id, n_rollout_threads, seed, mode):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, mode)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    # Create directory
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    log = set_log(config, model_dir)
    logger = SummaryWriter(str(log_dir))

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # Set env
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.mode)

    # Set maddpg
    maddpg = MADDPG.init_from_save(
        filename1="models/complex_push/mode0/run24/incremental/model_ep" + str(config.load_iter + 1) + ".pt",
        filename2="models/complex_push/mode1/run1/incremental/model_ep" + str(config.load_iter + 1) + ".pt")

    # Set buffer
    replay_buffer_mode0 = pickle.load(open(
        "models/complex_push/mode0/run24/replay_buffer" + str(config.load_iter) + ".pkl", "rb"))

    replay_buffer_mode1 = pickle.load(open(
        "models/complex_push/mode1/run1/replay_buffer" + str(config.load_iter) + ".pkl", "rb"))

    replay_buffer = ReplayBuffer(
        config.buffer_length, maddpg.nagents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space])

    replay_buffer.combine_two_memory(replay_buffer_mode0, replay_buffer_mode1)

    # Distillation
    print("[ INFO ] Distillation on progress ...")
    maddpg.prep_rollouts(device='cpu')
    maddpg.distill(
        config.num_distills, 
        batch_size=1024, 
        replay_buffer=replay_buffer, 
        hard=True,
        pass_actor=config.distill_pass_actor, 
        pass_critic=config.distill_pass_critic)

    print("********Starting training********")
    t = 0
    replay_buffer.reset()  # NOTE Reset memory after distillation

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = env.reset()
        maddpg.prep_rollouts(device='cpu')
        ep_rew = 0.

        # Reset noise
        explr_pct_remaining = max(
            0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        noise = config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining
        maddpg.scale_noise(noise)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            if ep_i % 200 == 0:
                env.render()

            # rearrange observations to be per agent, and convert to torch variable
            torch_obs = [
                Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                for i in range(maddpg.nagents)]

            # get actions
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions]
                       for i in range(config.n_rollout_threads)]

            # Take action
            next_obs, rewards, dones, infos = env.step(actions)
            if et_i == (config.episode_length - 1):
                dones = dones + 1

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            # For next timestep
            obs = next_obs
            t += config.n_rollout_threads
            ep_rew += rewards[0][0]

            # Update policy
            if ep_i < config.eval_ep:  # do not update after evaluation phase starts
                if len(replay_buffer) >= config.batch_size:
                    if (t % config.steps_per_update) < config.n_rollout_threads:
                        if USE_CUDA:
                            maddpg.prep_training(device='gpu')
                        else:
                            maddpg.prep_training(device='cpu')

                        for u_i in range(config.n_rollout_threads):
                            for a_i in range(maddpg.nagents):
                                sample = replay_buffer.sample(config.batch_size, to_gpu=False, norm_rews=True)
                                maddpg.update(sample, a_i, logger=logger)
                            maddpg.update_all_targets()
                        maddpg.prep_rollouts(device='cpu')

        # Log performance
        log[config.log_name].info(
            "Train episode reward {:0.5f} at episode {}".format(ep_rew, ep_i))
        logger.add_scalar('reward/episode_rewards', ep_rew, ep_i)
        logger.add_scalar('misc/noise', noise, ep_i)

        # # Save maddpg model and replay buffer time to time
        # if ep_i % 10000 == 0:
        #     print("*******Saving model and replay Buffer******")
        #     os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
        #     maddpg.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
        #     maddpg.save(str(run_dir / 'model.pt'))

        #     with open(str(run_dir / ('replay_buffer' + str(ep_i) + '.pkl')), 'wb') as output:
        #         pickle.dump(replay_buffer, output, -1)

    # print("********Saving and Closing*******")
    # maddpg.save(str(run_dir / 'model.pt'))
    # env.close()
    # logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    # logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id", help="Name of environment")
    parser.add_argument(
        "--model_name", help="Name of directory to store")
    parser.add_argument(
        "--mode", type=int, help="Mode for pushbox")
    parser.add_argument(
        "--load_iter", type=int, help="Iteration number to load")
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
    parser.add_argument("--n_episodes", default=2000, type=int)
    parser.add_argument(
        "--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument(
        "--batch_size", default=1024, type=int,
        help="Batch size for model training")
    parser.add_argument("--flip_ep",
                        default=999999, type=int,
                        help="Episode at which to flip")
    parser.add_argument("--eval_ep",
                        default=999999, type=int,
                        help="Episode at which to start evaluating")
    parser.add_argument("--hard_distill_ep",
                        default=999999, type=int,
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
    parser.add_argument(
        "--n_exploration_eps", default=25000, type=int)
    parser.add_argument(
        "--init_noise_scale", default=0.3, type=float)
    parser.add_argument(
        "--final_noise_scale", default=0.0, type=float)
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
