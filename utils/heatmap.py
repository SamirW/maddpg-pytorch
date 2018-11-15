import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from torch.autograd import Variable

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

def add_arrows(axes, delta_dict, q_vals = None, rescale=False):
    max_delta = max(delta_dict.values(), key=(lambda key: np.linalg.norm(key)))
    max_delta_size = np.linalg.norm(max_delta)


    for pos, delta in delta_dict.items():
        if rescale:
            delta = delta/max_delta_size*0.15
        else:
            delta = delta/np.linalg.norm(delta)*0.06

        axes.arrow(pos[0], pos[1], delta[0], delta[1], length_includes_head=True, head_width=0.018)

    if q_vals is not None:
        axes.imshow(q_vals)
        # print(q_vals)

def heatmap(maddpg, title="Agent Policies", save=False):
    fig, axes = plt.subplots(2, 2)

    num_arrows = 21
    other_poses = [[[0.5, 0.5], [-0.5, -0.5]], [[-0.5, -0.5], [0.5, 0.5]]]
    titles = [["Blue Agent, State 1", "Blue Agent, State 2"], ["Red Agent, State 1", "Red Agent, State 2"]]
    
    for i in range(len(axes)):
        for j in range(len(axes[i])):

            ax = axes[i, j]
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(titles[i][j])

            delta_dict = dict()
            other_pos = other_poses[i][j]
            for x in np.linspace(-1, 1, num_arrows):
                for y in np.linspace(-1, 1, num_arrows):
                    agent_pos = [x, y]

                    if i == 0:
                        agent_poses = np.array([agent_pos, other_pos])
                    else:
                        agent_poses = np.array([other_pos, agent_pos])

                    obs = get_observations(agent_poses)
                    maddpg.prep_rollouts(device='cpu')

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, k])),
                                          requires_grad=False)
                                 for k in range(maddpg.nagents)]
                    torch_agent_logits = maddpg.action_logits(torch_obs)
                    agent_logits = [ac.data.numpy() for ac in torch_agent_logits]
                    action = agent_logits[i][0]

                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]

            add_arrows(ax, delta_dict, rescale=False)

            if i==0:
                color = 'r'
            else:
                color = 'b'
            ax.add_artist(plt.Circle(other_pos, 0.1, color=color))
            ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
            ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    fig.suptitle(title)

    if save:
        plt.savefig("{}.png".format(title), bbox_inches="tight", dpi=300) 

def distilled_heatmap(maddpg, save=False):
    fig, axes = plt.subplots(2, 2)

    num_arrows = 21
    other_poses = [[[0.5, 0.5], [-0.5, -0.5]], [[-0.5, -0.5], [0.5, 0.5]]]
    titles = [["Blue Agent, State 1", "Blue Agent, State 2"], ["Red Agent, State 1", "Red Agent, State 2"]]
    
    for i in range(len(axes)):
        for j in range(len(axes[i])):

            ax = axes[i, j]
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(titles[i][j])

            delta_dict = dict()
            other_pos = other_poses[i][j]
            for x in np.linspace(-1, 1, num_arrows):
                for y in np.linspace(-1, 1, num_arrows):
                    agent_pos = [x, y]

                    if i == 0:
                        agent_poses = np.array([agent_pos, other_pos])
                    else:
                        agent_poses = np.array([other_pos, agent_pos])

                    obs = get_observations(agent_poses)
                    maddpg.prep_rollouts(device='cpu')

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, k])),
                                          requires_grad=False)
                                 for k in range(maddpg.nagents)]

                    torch_agent_i_logits = maddpg.distilled_agent.policy(torch_obs[i])
                    action = torch_agent_i_logits.data.numpy()[0]

                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]

            add_arrows(ax, delta_dict, rescale=False)

            if i==0:
                color = 'r'
            else:
                color = 'b'
            ax.add_artist(plt.Circle(other_pos, 0.1, color=color))
            ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
            ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    fig.suptitle("Distilled Policy")

    if save:
        plt.savefig("Distilled Policy.png", bbox_inches="tight", dpi=300) 

def distilled_heatmap2(maddpg):
    fig, axes = plt.subplots(1, 2)

    num_arrows = 21
    other_poses = [[[0.5, 0.5], [-0.5, -0.5]], [[-0.5, -0.5], [0.5, 0.5]]]
    titles = ["Distilled Agent, State 1", "Distilled Agent, State 2"]
    
    for j in range(len(axes)):

        ax = axes[j]
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(titles[j])

        delta_dict = dict()
        other_pos = other_poses[0][j]
        for x in np.linspace(-1, 1, num_arrows):
            for y in np.linspace(-1, 1, num_arrows):
                agent_pos = [x, y]

                agent_poses = np.array([agent_pos, other_pos])

                obs = get_observations(agent_poses)
                maddpg.prep_rollouts(device='cpu')

                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, k])),
                                      requires_grad=False)
                             for k in range(maddpg.nagents)]

                torch_agent_i_logits = maddpg.distilled_agent.policy(torch_obs[0])
                action = torch_agent_i_logits.data.numpy()[0]

                delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]

        add_arrows(ax, delta_dict, rescale=False)

        ax.add_artist(plt.Circle(other_pos, 0.1, color='grey'))
        ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='black'))
        ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='black'))

    fig.suptitle("Distilled Policy")