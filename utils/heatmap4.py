import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from torch.autograd import Variable

lndmrk_poses = np.array([[-0.75, 0.75], [0.75, 0.75], [0.75, -0.75], [-0.75, -0.75]])
default_agent_poses = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
flipped_agent_poses = [[0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
color = {0: 'b', 1: 'r', 2: 'g', 3: [0.65, 0.65, 0.65]}

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

def add_arrows(axes, delta_dict, arrow_color="black", q_vals = None, rescale=False):
    max_delta = max(delta_dict.values(), key=(lambda key: np.linalg.norm(key)))
    max_delta_size = np.linalg.norm(max_delta)


    for pos, delta in delta_dict.items():
        if rescale:
            delta = delta/max_delta_size*0.15
        else:
            delta = delta/np.linalg.norm(delta)*0.06

        axes.arrow(pos[0], pos[1], delta[0], delta[1], length_includes_head=True, head_width=0.018, color=arrow_color)

    if q_vals is not None:
        axes.imshow(q_vals)
        # print(q_vals)

def heatmap(maddpg, title="Agent Policies", save=False):
    fig, axes = plt.subplots(4, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    num_arrows = 21
    titles = [["Blue Agent, State 1", "Blue Agent, State 2"], ["Red Agent, State 1", "Red Agent, State 2"], ["Green Agent, State 1", "Green Agent, State 2"], ["Silver Agent, State 1", "Silver Agent, State 2"]]
    
    for i in range(len(axes)):
        for j in range(len(axes[i])):

            ax = axes[i, j]
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(titles[i][j])

            delta_dict = dict()
            for x in np.linspace(-1, 1, num_arrows):
                for y in np.linspace(-1, 1, num_arrows):
                    agent_pos = [x, y]

                    if j == 0:
                        agent_poses = np.copy(default_agent_poses)
                    else:
                        agent_poses = np.copy(flipped_agent_poses)
                    agent_poses[i] = agent_pos

                    obs = get_observations(agent_poses)
                    maddpg.prep_rollouts(device='cpu')

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, k])),
                                          requires_grad=False)
                                 for k in range(maddpg.nagents)]
                    torch_agent_logits = maddpg.action_logits(torch_obs)
                    agent_logits = [ac.data.numpy() for ac in torch_agent_logits]
                    action = agent_logits[i][0]

                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]

            add_arrows(ax, delta_dict, arrow_color=color[i], rescale=False)
            for l in range(len(agent_poses)):
                if i == l: continue
                ax.add_artist(plt.Circle(agent_poses[l], 0.1, color=color[l]))
            for lndmrk_pos in lndmrk_poses:
                ax.add_artist(plt.Circle(lndmrk_pos, 0.05, color=[0.25, 0.25, 0.25]))

    fig.suptitle(title)

    if save:
        plt.savefig("{}.png".format(title), bbox_inches="tight", dpi=300) 

def distilled_heatmap(maddpg, save=False):
    fig, axes = plt.subplots(4, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    num_arrows = 21
    titles = [["Blue Agent, State 1", "Blue Agent, State 2"], ["Red Agent, State 1", "Red Agent, State 2"], ["Green Agent, State 1", "Green Agent, State 2"], ["Silver Agent, State 1", "Silver Agent, State 2"]]
    
    for i in range(len(axes)):
        for j in range(len(axes[i])):

            ax = axes[i, j]
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(titles[i][j])

            delta_dict = dict()
            for x in np.linspace(-1, 1, num_arrows):
                for y in np.linspace(-1, 1, num_arrows):
                    agent_pos = [x, y]

                    if j == 0:
                        agent_poses = np.copy(default_agent_poses)
                    else:
                        agent_poses = np.copy(flipped_agent_poses)
                    agent_poses[i] = agent_pos

                    obs = get_observations(agent_poses)
                    maddpg.prep_rollouts(device='cpu')

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, k])),
                                          requires_grad=False)
                                 for k in range(maddpg.nagents)]

                    torch_agent_i_logits = maddpg.distilled_agent.policy(torch_obs[i])
                    action = torch_agent_i_logits.data.numpy()[0]
                    print(action)
                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]

            add_arrows(ax, delta_dict, arrow_color=color[i], rescale=False)
            for l in range(len(agent_poses)):
                if i == l: continue
                ax.add_artist(plt.Circle(agent_poses[l], 0.1, color=color[l]))
            for lndmrk_pos in lndmrk_poses:
                ax.add_artist(plt.Circle(lndmrk_pos, 0.05, color=[0.25, 0.25, 0.25]))

    fig.suptitle("Distilled Policy")

    if save:
        plt.savefig("Distilled Policy.png", bbox_inches="tight", dpi=300)

def test():

    fig, ax = plt.subplots(1, 1)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.add_artist(plt.Circle([-0.5, -0.5], 0.1, color="b"))
    ax.add_artist(plt.Circle([0.5, 0.5], 0.1, color='r'))
    ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
    ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    plt.savefig("Example.png", bbox_inches="tight", dpi=300)