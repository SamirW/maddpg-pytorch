from torch import Tensor
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(
            num_in_pol, num_out_pol,
            hidden_dim=hidden_dim,
            constrain_out=True,
            discrete_action=discrete_action)

        self.critic = MLPNetwork(
            num_in_critic, 1,
            hidden_dim=hidden_dim,
            constrain_out=False)

        self.target_policy = MLPNetwork(
            num_in_pol, num_out_pol,
            hidden_dim=hidden_dim,
            constrain_out=True,
            discrete_action=discrete_action)

        self.target_critic = MLPNetwork(
            num_in_critic, 1,
            hidden_dim=hidden_dim,
            constrain_out=False)

        hard_update(target=self.target_policy, source=self.policy)
        hard_update(target=self.target_critic, source=self.critic)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        assert discrete_action is True
        self.max_entropy = Categorical(logits=Tensor(np.ones((1, num_out_pol)))).entropy()

    def reset_noise(self):
        if not self.discrete_action:
            raise ValueError()
            self.exploration.reset()  # For OU Noise

    def reset(self):
        self.policy.randomize()
        self.critic.randomize()
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        assert self.discrete_action is True

        action = self.policy(obs)
        if explore:
            action = gumbel_softmax(action, hard=True)
        else:
            action = onehot_from_logits(action)

        return action

    def action_logits(self, obs):
        return(self.policy(obs))

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
