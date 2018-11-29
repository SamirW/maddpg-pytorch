import argparse
import torch
import time
import pickle
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG


def run(config):
    model_dir = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_dir / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_dir / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(str(model_path))
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.distill:
        print("************Distilling***********")
        with open(str(model_dir / "replay_buffer.pkl"), 'rb') as input:
            replay_buffer = pickle.load(input)
        maddpg.distill(512, 1024, replay_buffer, hard=True)

    frames = []
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env._reset(flip=config.flip)
        if config.save_gifs:
            # frames = []
            frames.append(env._render('rgb_array')[0])
        env._render('human')
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # print([torch.argmax(a).numpy().item()==0 for a in torch_actions])
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env._step(actions)
            if config.save_gifs:
                frames.append(env._render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env._render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--distill", action="store_true")

    config = parser.parse_args()

    # config.flip = False
    run(config)

    # config.flip = True
    # run(config)