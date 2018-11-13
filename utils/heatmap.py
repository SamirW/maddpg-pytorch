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

def heatmap(maddpg):
    fig, axes = plt.subplots(2, 2)

    num_arrows = 17
    other_poses = [[[0.5, 0.5], [-0.5, -0.5]], [[-0.5, -0.5], [0.5, 0.5]]]
    titles = [["Agent 1, State 1", "Agent 1, State 2"], ["Agent 2, State 1", "Agent 2, State 2"]]
    
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

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                          requires_grad=False)
                                 for i in range(maddpg.nagents)]
                    torch_agent_logits = maddpg.action_logits(torch_obs)
                    agent_logits = [ac.data.numpy() for ac in torch_agent_logits]
                    action = agent_logits[i][0]

                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]

            add_arrows(ax, delta_dict, rescale=True)

            if i==0:
                color = 'r'
            else:
                color = 'b'
            ax.add_artist(plt.Circle(other_pos, 0.1, color=color))
            ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
            ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

def distilled_heatmap(maddpg):
    fig, axes = plt.subplots(2, 2)

    num_arrows = 17
    other_poses = [[[0.5, 0.5], [-0.5, -0.5]], [[-0.5, -0.5], [0.5, 0.5]]]
    titles = [["Agent 1, State 1", "Agent 1, State 2"], ["Agent 2, State 1", "Agent 2, State 2"]]
    
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

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                          requires_grad=False)
                                 for i in range(maddpg.nagents)]

                    torch_agent_i_logits = maddpg.distilled_agent.policy(torch_obs[i])
                    action = torch_agent_i_logits.data.numpy()[0]
                    print(action)

                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]

            add_arrows(ax, delta_dict, rescale=True)

            if i==0:
                color = 'r'
            else:
                color = 'b'
            ax.add_artist(plt.Circle(other_pos, 0.1, color=color))
            ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
            ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

def heatmap2(model_file):
    maddpg = MADDPG.init_from_save(model_file)
    fig, axes = plt.subplots(2, 2)

    num_arrows = 11
    other_poses = [[[0.5, 0.5], [-0.5, -0.5]], [[-0.5, -0.5], [0.5, 0.5]]]
    titles = [["Agent 1, State 1", "Agent 1, State 2"], ["Agent 2, State 1", "Agent 2, State 2"]]
    
    for i in range(len(axes)):
        for j in range(len(axes[i])):

            ax = axes[i, j]
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(titles[i][j])

            delta_dict = dict()
            q_vals = np.zeros((num_arrows, num_arrows))
            other_pos = other_poses[i][j]

            x_iter = 0
            for x in np.linspace(-1, 1, num_arrows):
                y_iter = 0
                for y in np.linspace(-1, 1, num_arrows):
                    agent_pos = [x, y]

                    if i == 0:
                        agent_poses = np.array([agent_pos, other_pos])
                    else:
                        agent_poses = np.array([other_pos, agent_pos])

                    obs = get_observations(agent_poses)
                    maddpg.prep_rollouts(device='cpu')

                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                          requires_grad=False)
                                 for i in range(maddpg.nagents)]
                    
                    torch_agent_actions = maddpg.step(torch_obs, explore=True)
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                    action = agent_actions[i][0]

                    agent_q_vals = maddpg.get_critic_vals(torch_obs, torch_agent_actions)
                    q_val = agent_q_vals[i][0].data.numpy()
                    q_vals[x_iter, y_iter] = q_val

                    delta_dict[tuple(agent_pos)] = [action[1] - action[2], action[3] - action[4]]
                    
                    y_iter += 1
                x_iter += 1

            add_arrows(ax, delta_dict, q_vals=q_vals)

            if i==0:
                color = 'r'
            else:
                color = 'b'
            ax.add_artist(plt.Circle(other_pos, 0.1, color=color))
            ax.add_artist(plt.Circle(lndmrk_poses[0], 0.05, color='grey'))
            ax.add_artist(plt.Circle(lndmrk_poses[1], 0.05, color='grey'))

    plt.tight_layout()
    plt.show()