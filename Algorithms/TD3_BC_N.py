# Imports
import copy
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Original TD3-BC code - https://github.com/sfujim/TD3_BC
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
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2b = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l2b(a))
        a = self.max_action * torch.tanh(self.l3(a))

        return a


### TD3-BC-N Agent ###
class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, num_critics=2,
                 batch_size=256, gamma=0.99, tau=0.005, lr=3e-4,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cpu"):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = VectorizedCritic(state_dim, action_dim, num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.batch_size = batch_size
        self.num_critics = num_critics

        self.critic_loss_history = []
        self.actor_loss_history = []

        self.total_it = 0

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

        return action.cpu().numpy().flatten()


    def train_offline(self, replay_buffer, iterations=1, beta=0.4, dep_targets=True):

        for it in range(iterations):
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)
            done = torch.index_select(replay_buffer[4], 0, indices).to(self.device)

            # Critic #
            self.total_it += 1
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                q_values = self.critic_target(next_state, next_action)
                if dep_targets:
                    q_values = q_values.min(0)[0]
                targetQ = reward + (1 - done) * self.gamma * q_values
            q_values_all = self.critic(state, action)
            critic_loss = F.mse_loss(q_values_all, targetQ)  # Ignore warning if using shared targets - we want broadcasting

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor #
            if self.total_it % self.policy_freq == 0:
                policy_actions = self.actor(state)
                Q = self.critic(state, policy_actions).min(0)[0]
                Q /= Q.abs().mean().detach()
                actor_loss = -Q.mean() + beta * F.mse_loss(policy_actions, action)

                self.actor_loss_history.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_online_ft(self, replay_buffer, iterations=1, beta=0.4, dep_targets=True):

        for it in range(iterations):
            minibatch = random.sample(replay_buffer, self.batch_size)
            state = torch.Tensor(tuple(d[0] for d in minibatch)).to(self.device)
            action = torch.Tensor(tuple(d[1] for d in minibatch)).to(self.device)
            reward = torch.Tensor(tuple(d[2] for d in minibatch)).to(self.device)
            next_state = torch.Tensor(tuple(d[3] for d in minibatch)).to(self.device)
            done = torch.Tensor(tuple(d[4] for d in minibatch)).to(self.device)

            # Critic #
            self.total_it += 1
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                q_values = self.critic_target(next_state, next_action)
                if dep_targets:
                    q_values = q_values.min(0)[0]
                targetQ = reward + (1 - done) * self.gamma * q_values
            q_values_all = self.critic(state, action)
            critic_loss = F.mse_loss(q_values_all, targetQ)  # Ignore warning - we want broadcasting

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor #
            if self.total_it % self.policy_freq == 0:
                policy_actions = self.actor(state)
                Q = self.critic(state, policy_actions).min(0)[0]
                Q /= Q.abs().mean().detach()
                actor_loss = -Q.mean() + beta * F.mse_loss(policy_actions, action)

                self.actor_loss_history.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
