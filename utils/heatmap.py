import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from torch.autograd import Variable
from algorithms.maddpg import MADDPG

lndmrk_poses = np.array([[-0.75, -0.75], [0.75, 0.75]])

def get_observations(agent_poses):
    obs_n = []
    for i, agent_pos in enumerate(agent_poses):
        entity_pos = []
        for lndmrk_pos in lndmrk_poses:
            entity_pos.append(lndmrk_pos - agent_pos)
        other_pos = []
        comm = []
        for j, agent_pos_2 in enumerate(agent_poses):
            if i == j: continue
            comm.append(np.array([0,0]))
            other_pos.append(agent_pos_2 - agent_pos)
        obs_n.append(np.concatenate([np.array([0,0])] + [agent_pos] + entity_pos + other_pos + comm))
    return np.array([obs_n])

def add_arrow(pos, dir):
    [x, y] = pos
    [dx, dy] = [0, 0]

    if dir == 1:
        dx += 0.06
    elif dir == 2:
        dx -= 0.06
    elif dir == 3:
        dy += 0.06
    elif dir == 4:
        dy -= 0.06

    plt.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.02)

def heatmap(model_file):
    maddpg = MADDPG.init_from_save(model_file)
    agent_2_pos = [0.5, 0.5]

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    for agent_1_x in np.linspace(-1, 1, 21):
        for agent_1_y in np.linspace(-1, 1, 21):
            agent_1_pos = [agent_1_x, agent_1_y]

            agent_poses = np.array([agent_1_pos, agent_2_pos])

            obs = get_observations(agent_poses)
            maddpg.prep_rollouts(device='cpu')

            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            torch_agent_actions = maddpg.step(torch_obs, explore=False)

            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            agent_1_dir = np.argmax(agent_actions[0])
            add_arrow(agent_1_pos, agent_1_dir)

    ax.add_artist(plt.Circle(agent_2_pos, 0.1, color='r'))
    ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
    ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    plt.show()
    fig.savefig('Agent_1_Policy.png')