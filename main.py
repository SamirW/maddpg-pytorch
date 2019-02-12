import argparse
import torch
import os
import numpy as np
from gym.spaces import Box
from pathlib import Path
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.logging import set_log
from utils.buffer import ReplayBuffer
from utils.env_wrappers import DummyVecEnv
from algorithms.maddpg import MADDPG
from train import train


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
        raise NotImplementedError()


def run(config):
    # Directories
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [
            int(str(folder.name).split('run')[1]) 
            for folder in model_dir.iterdir() 
            if str(folder.name).startswith('run')]

        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))

    # Set log
    log = set_log(config, model_dir)
    logger = SummaryWriter(str(log_dir))

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Set env
    env = make_parallel_env(
        config.env_id, 
        config.n_rollout_threads, 
        config.seed,
        config.discrete_action)

    # Set maddpg
    maddpg = MADDPG.init_from_env(
        env, 
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim)

    # Set memory
    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        maddpg.nagents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space])

    train(
        maddpg=maddpg,
        env=env,
        replay_buffer=replay_buffer,
        config=config,
        log=log,
        logger=logger,
        run_dir=run_dir,
        log_dir=log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_id", 
        help="Name of environment")
    parser.add_argument(
        "model_name",
        help="Name of directory to store " + "model/training contents")
    parser.add_argument(
        "--log_comment",
        default="", type=str,
        help="Log file comment")
    parser.add_argument(
        "--seed",
        default=1, type=int,
        help="Random seed")
    parser.add_argument(
        "--display_every",
        default=999999, type=int,
        help="Display frequency")
    parser.add_argument(
        "--n_rollout_threads", 
        default=1, type=int)
    parser.add_argument(
        "--n_training_threads", 
        default=6, 
        type=int)
    parser.add_argument(
        "--buffer_length", 
        default=int(1e6), 
        type=int)
    parser.add_argument(
        "--n_episodes", 
        default=2000, 
        type=int)
    parser.add_argument(
        "--episode_length", 
        default=25, 
        type=int)
    parser.add_argument(
        "--steps_per_update", 
        default=100, 
        type=int)
    parser.add_argument(
        "--batch_size",
        default=1024, type=int,
        help="Batch size for model training")
    parser.add_argument(
        "--flip_ep",
        default=999999, type=int,
        help="Episode at which to flip")
    parser.add_argument(
        "--eval_ep",
        default=999999, type=int,
        help="Episode at which to start evaluating")
    parser.add_argument(
        "--hard_distill_ep",
        default=999999, type=int,
        help="Episode at which to hard distill")
    parser.add_argument(
        "--distill_freq",
        default=999999, type=int,
        help="Distilling frequency")
    parser.add_argument(
        "--model_save_freq",
        default=999999, type=int,
        help="Model save freq")
    parser.add_argument(
        "--skip_actor_length",
        default=0, type=int,
        help="How long to skip actor updates")
    parser.add_argument(
        "--n_exploration_eps", 
        default=25000, 
        type=int)
    parser.add_argument(
        "--init_noise_scale", 
        default=0.3, 
        type=float)
    parser.add_argument(
        "--final_noise_scale", 
        default=0.0, 
        type=float)
    parser.add_argument(
        "--save_interval", 
        default=1000, 
        type=int)
    parser.add_argument(
        "--hidden_dim", 
        default=64, 
        type=int)
    parser.add_argument(
        "--lr", 
        default=0.01, 
        type=float)
    parser.add_argument(
        "--tau", 
        default=0.01, 
        type=float)
    parser.add_argument(
        "--agent_alg",
        default="MADDPG", type=str,
        choices=['MADDPG', 'DDPG'])
    parser.add_argument(
        "--adversary_alg",
        default="MADDPG", type=str,
        choices=['MADDPG', 'DDPG'])
    parser.add_argument(
        "--discrete_action",
        action='store_true',
        default=True)
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        default=False)

    config = parser.parse_args()

    config.log_name = \
        "env::%s_seed::%s_comment::%s_log" % (
            config.env_id, str(config.seed), config.log_comment)  

    run(config)
