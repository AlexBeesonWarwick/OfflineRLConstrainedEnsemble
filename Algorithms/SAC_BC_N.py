# Import modules
import numpy as np
import random
import copy
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.independent import Independent
from torch.distributions.transforms import AffineTransform

'''
Original SAC code - https://github.com/rail-berkeley/softlearning/
Vectorised Linear code - https://github.com/tinkoff-ai/CORL/blob/main/algorithms/sac_n.py
'''

class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias

class VectorizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_critics=2, hidden_dim=256):
        super(VectorizedCritic, self).__init__()

        self.l1 = VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics)
        self.l2 = VectorizedLinear(hidden_dim, hidden_dim, num_critics)
        self.l3 = VectorizedLinear(hidden_dim, hidden_dim, num_critics)
        self.qs = VectorizedLinear(hidden_dim, 1, num_critics)

        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        q_values = F.relu(self.l1(state_action))
        q_values = F.relu(self.l2(q_values))
        q_values = F.relu(self.l3(q_values))
        q_values = self.qs(q_values)

        return q_values.squeeze(-1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, hidden_dim=256, log_std_min=-20.0, log_std_max=2.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.log_std_min = log_std_min                      # for numerical stability
        self.log_std_max = log_std_max                      # for numerical stability
        self.action_mean = (max_action + min_action) / 2    # to allow for action domains other than [-1, 1]
        self.action_scale = (max_action - min_action) / 2   # to allow for action domains other than [-1, 1]
        self.min_action = min_action
        self.max_action = max_action
        self.eps = 1e-6

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        mean = self.mean(a)
        std = self.log_std(a).clamp(self.log_std_min, self.log_std_max).exp()

        return mean, std

    def sample_normal(self, state, action=None):
        mu, sigma = self.forward(state)
        dist = TransformedDistribution(Independent(Normal(mu, sigma), 1), [
            TanhTransform(cache_size=1), AffineTransform(self.action_mean, self.action_scale, cache_size=1)])
        actions = dist.rsample()                    # For repam trick
        log_prob_action = dist.log_prob(actions)

        if action is None:
            return actions, log_prob_action
        else:
            action = action.clamp(self.min_action + self.eps, self.max_action - self.eps)
            log_prob_bc = dist.log_prob(action)
            return actions, log_prob_action, log_prob_bc


class Agent():
    def __init__(self, state_dim, action_dim, min_action, max_action, min_ent=0, num_critics=2, online_ft=None,
                 batch_size=256, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, device="cpu"):

        # Initialisation
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = VectorizedCritic(state_dim, action_dim, num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        if online_ft is not None:
            self.log_alpha = torch.tensor([online_ft], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        # Record losses
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_loss_history = []

        # Set remaining parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.max_action = max_action
        self.min_ent = min_ent
        self.num_critics = num_critics

    def choose_action(self, state, mean=True):
        # Take mean/greedy action by default, but also allows sampling from policy
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if mean == True:
                action = self.max_action * torch.tanh(self.actor(state)[0])
            else:
                action = self.actor.sample_normal(state)[0]

        return action.cpu().numpy().flatten()

    def train_offline(self, replay_buffer, iterations=1, beta=0.0, dep_targets=True):

        for it in range(iterations):
            # Sample batch from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)
            done = torch.index_select(replay_buffer[4], 0, indices).to(self.device)

            alpha = self.log_alpha.exp().detach()

            # Critic loss #
            with torch.no_grad():
                next_action, log_next_action = self.actor.sample_normal(next_state)
                q_target = self.critic_target(next_state, next_action)
                if dep_targets:
                    q_target = q_target.min(0)[0]
                q_target -= (alpha * log_next_action)
                q_hat = reward + self.gamma * (1 - done) * q_target
            q_values_all = self.critic(state, action)
            critic_loss = F.mse_loss(q_values_all, q_hat)  # Ignore warning if using dependent targets - we want broadcasting

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor and alpha loss #
            actions, log_actions, log_prob_bc = self.actor.sample_normal(state, action)
            critic_value = self.critic(state, actions).min(0)[0]
            critic_value -= (alpha * log_actions)
            critic_value /= critic_value.abs().mean().detach()
            actor_loss = -(critic_value + beta * log_prob_bc).mean()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha * (log_actions.detach() + self.min_ent)).mean()
            self.alpha_loss_history.append(alpha_loss.item())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ## Polyak target network updates
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def train_online_ft(self, replay_buffer, iterations=1, beta=0.0, dep_targets=True):

        for it in range(iterations):
            # Sample batch from replay buffer
            minibatch = random.sample(replay_buffer, self.batch_size)
            state = torch.Tensor(tuple(d[0] for d in minibatch)).to(self.device)
            action = torch.Tensor(tuple(d[1] for d in minibatch)).to(self.device)
            reward = torch.Tensor(tuple(d[2] for d in minibatch)).to(self.device)
            next_state = torch.Tensor(tuple(d[3] for d in minibatch)).to(self.device)
            done = torch.Tensor(tuple(d[4] for d in minibatch)).to(self.device)

            alpha = self.log_alpha.exp().detach()

            # Critic loss #
            with torch.no_grad():
                next_action, log_next_action = self.actor.sample_normal(next_state)
                q_target = self.critic_target(next_state, next_action)
                if dep_targets:
                    q_target = q_target.min(0)[0]
                q_target -= (alpha * log_next_action)
                q_hat = reward + self.gamma * (1 - done) * q_target
            q_values_all = self.critic(state, action)
            critic_loss = F.mse_loss(q_values_all, q_hat)  # Ignore warning if using shared targets - we want broadcasting

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor and alpha loss #
            actions, log_actions, log_prob_bc = self.actor.sample_normal(state, action)
            critic_value = self.critic(state, actions).min(0)[0]
            critic_value -= (alpha * log_actions)
            critic_value /= critic_value.abs().mean().detach()
            actor_loss = -(critic_value + beta * log_prob_bc).mean()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha * (log_actions.detach() + self.min_ent)).mean()
            self.alpha_loss_history.append(alpha_loss.item())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ## Polyak target network updates
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
