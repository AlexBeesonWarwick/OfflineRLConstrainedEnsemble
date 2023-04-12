# Import modules
import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.independent import Independent
from torch.distributions.transforms import AffineTransform

'''
Original IQL code - https://github.com/ikostrikov/implicit_q_learning
PyTorch implementation - https://github.com/rail-berkeley/rlkit/tree/master/examples/iql
'''

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], dim=-1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return torch.squeeze(q1, dim=-1), torch.squeeze(q2, dim=-1)

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(Value, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        v = self.l3(v)

        return torch.squeeze(v, dim=-1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action,
                 log_std_min=-5.0, log_std_max=2.0, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

        self.log_std_min = log_std_min              # for stability
        self.log_std_max = log_std_max              # for stability
        self.min_action = min_action
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mu = torch.tanh(self.mean(a))
        log_sigma = self.log_std.clamp(self.log_std_min, self.log_std_max)

        sigma = log_sigma.exp()

        return mu, sigma

    def distribution(self, state):
        mu, sigma = self.forward(state)
        distribution = Independent(Normal(mu, sigma), 1)

        return distribution

    def log_probs_bc(self, state, action):
        dist = self.distribution(state)
        log_probs_bc = dist.log_prob(action)

        return log_probs_bc


# IQL
class Agent():
    def __init__(self, state_dim, action_dim, min_action, max_action,
                 batch_size=256, quantile=1.0, beta=1.0, lr_critic=3e-4, lr_actor=3e-4, gamma=0.99,
                 tau=0.005, device="cpu"):

        # Initialise networks
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, int(1e6))

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.value = Value(state_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_critic)

        # Record losses
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.value_loss_history = []

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.quantile = quantile
        self.beta = beta
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action

    def choose_action(self, state, mean_action=True):
        # Allows use of mean or sampled actions #
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if mean_action == True:
                action, _ = self.actor(state)
            else:
                dist = self.actor.distribution(state)
                action = dist.sample().clamp(self.min_action, self.max_action)

        return action.cpu().numpy().flatten()

    def train_offline(self, replay_buffer, iterations):

        for it in range(iterations):
            # Sample batch from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)
            done = torch.index_select(replay_buffer[4], 0, indices).to(self.device)

            ### Value network ###
            with torch.no_grad():
                qt1, qt2 = self.critic_target(state, action)
                qt = torch.min(qt1, qt2)
            vf_pred = self.value(state)
            vf_error = qt - vf_pred
            vf_sign = (vf_error > 0).float()
            vf_weight = vf_sign * self.quantile + (1 - vf_sign) * (1 - self.quantile)
            value_loss = (vf_weight * vf_error.pow(2)).mean()

            self.value_loss_history.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            ### Actor network ###
            log_probs_bc = self.actor.log_probs_bc(state, action)
            with torch.no_grad():
                qt1, qt2 = self.critic_target(state, action)
                qt = torch.min(qt1, qt2)
                vf_pred = self.value(state)
                adv = qt - vf_pred
                exp_adv = torch.exp(adv * self.beta)
                weights = exp_adv.clamp(min=None, max=100.0)
            actor_loss = -(log_probs_bc * weights).mean()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_scheduler.step()

            ### Critic network ###
            q1, q2 = self.critic(state, action)
            with torch.no_grad():
                target_vf_pred = self.value(next_state)
                q_target = reward + (1 - done) * self.gamma * target_vf_pred
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Soft-update Target Networks #
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
