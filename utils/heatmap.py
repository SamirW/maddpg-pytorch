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

def add_arrow(axes, pos, action):

    delta = [action[1] - action[2], action[3] - action[4]]
    delta = delta/np.linalg.norm(delta)
    delta *= 0.06

    axes.arrow(pos[0], pos[1], delta[0], delta[1], length_includes_head=True, head_width=0.02)

def heatmap(model_file):
    maddpg = MADDPG.init_from_save(model_file)
    fig, axes = plt.subplots(2, 2)
    
    ### Agent 1, Non-Flipped Policy" ###
    agent_2_pos = [0.5, 0.5]

    ax = axes[0, 0]
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Agent 1, Non-Flipped Policy")

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
            torch_agent_logits = maddpg.action_logits(torch_obs)

            agent_logits = [ac.data.numpy() for ac in torch_agent_logits]
            agent_1_logits = agent_logits[0][0]

            add_arrow(ax, agent_1_pos, agent_1_logits)

    ax.add_artist(plt.Circle(agent_2_pos, 0.1, color='r'))
    ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
    ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    ### Agent 1, Flipped Policy" ###
    agent_2_pos = [-0.5, -0.5]

    ax = axes[0, 1]
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Agent 1, Flipped Policy")

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
            torch_agent_logits = maddpg.action_logits(torch_obs)

            agent_logits = [ac.data.numpy() for ac in torch_agent_logits]
            agent_1_logits = agent_logits[0][0]

            add_arrow(ax, agent_1_pos, agent_1_logits)

    ax.add_artist(plt.Circle(agent_2_pos, 0.1, color='r'))
    ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
    ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    ### Agent 2, Non-Flipped Policy" ###
    agent_1_pos = [-0.5, -0.5]

    ax = axes[1, 0]
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Agent 2, Non-Flipped Policy")

    for agent_2_x in np.linspace(-1, 1, 21):
        for agent_2_y in np.linspace(-1, 1, 21):
            agent_2_pos = [agent_2_x, agent_2_y]

            agent_poses = np.array([agent_2_pos, agent_2_pos])

            obs = get_observations(agent_poses)
            maddpg.prep_rollouts(device='cpu')

            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            torch_agent_logits = maddpg.action_logits(torch_obs)

            agent_logits = [ac.data.numpy() for ac in torch_agent_logits]
            agent_2_logits = agent_logits[1][0]

            add_arrow(ax, agent_2_pos, agent_2_logits)

    ax.add_artist(plt.Circle(agent_1_pos, 0.1, color='b'))
    ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
    ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    ### Agent 2, Flipped Policy" ###
    agent_1_pos = [0.5, 0.5]

    ax = axes[1, 1]
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Agent 2, Flipped Policy")

    for agent_2_x in np.linspace(-1, 1, 21):
        for agent_2_y in np.linspace(-1, 1, 21):
            agent_2_pos = [agent_2_x, agent_2_y]

            agent_poses = np.array([agent_2_pos, agent_2_pos])

            obs = get_observations(agent_poses)
            maddpg.prep_rollouts(device='cpu')

            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            torch_agent_logits = maddpg.action_logits(torch_obs)

            agent_logits = [ac.data.numpy() for ac in torch_agent_logits]
            agent_2_logits = agent_logits[1][0]

            add_arrow(ax, agent_2_pos, agent_2_logits)

    ax.add_artist(plt.Circle(agent_1_pos, 0.1, color='b'))
    ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
    ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    ### Show ###
    plt.show()
