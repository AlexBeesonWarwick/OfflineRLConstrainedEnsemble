# Import modules
import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.transforms import AffineTransform

'''
Original CQL code - https://github.com/aviralkumar2907/CQL
Alternative CQL code - https://github.com/young-geng/CQL

Watch out for use of alpha and alpha_prime.  Alpha is the entropy regulariser from SAC,
Alpha_prime is the conservative coefficient for CQL (but this is denoted alpha in the CQL paper)
Alpha_prime is often denoted as min_q_weight in GitHubs
'''

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.q1(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], dim=-1)))
        q2 = F.relu(self.l5(q2))
        q2 = F.relu(self.l6(q2))
        q2 = self.q2(q2)

        return torch.squeeze(q1, dim=-1), torch.squeeze(q2, dim=-1)

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

    def sample_normal(self, state):
        mu, sigma = self.forward(state)
        dist = TransformedDistribution(Independent(Normal(mu, sigma), 1), [
            TanhTransform(cache_size=1), AffineTransform(self.action_mean, self.action_scale, cache_size=1)])
        actions = dist.rsample()                    # For repam trick
        log_prob_action = dist.log_prob(actions)

        return actions, log_prob_action


# CQL_SAC with manual alpha prime tuning #
class Agent():
    def __init__(self, state_dim, action_dim, min_action, max_action, min_ent=0,
                 batch_size=256, alpha_prime=1.0, lr_critic=3e-4, lr_actor=3e-4, gamma=0.99,
                 tau=0.005, repeats=10, device="cpu"):

        # Initialise networks
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # This is the entropy regulariser coefficient for SAC
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        # Record losses
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_loss_history = []

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.alpha_prime = alpha_prime
        self.min_ent = min_ent
        self.action_dim = action_dim
        self.repeats = repeats
        self.min_action = min_action
        self.max_action = max_action

    def choose_action(self, state, mean=True):
        # Take mean/greedy action by default, but also allows sampling from policy
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if mean == True:
                action = self.max_action * torch.tanh(self.actor(state)[0])
            else:
                action = self.actor.sample_normal(state)[0]

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

            actions, log_probs = self.actor.sample_normal(state)
            alpha = self.log_alpha.detach().exp()

            ### Entropy loss ###
            alpha_loss = -(self.log_alpha * (log_probs + self.min_ent).detach()).mean()

            ### Actor loss ###
            q1_policy, q2_policy = self.critic(state, actions)
            actor_loss = (alpha * log_probs - torch.min(q1_policy, q2_policy)).mean()

            ### Critic and CQL loss ##
            # Critic
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample_normal(next_state)
                q1_target, q2_target = self.critic_target(next_state, next_actions)
                q_target = torch.min(q1_target, q2_target) - alpha * next_log_probs
                q_hat = reward + self.gamma * (1 - done) * q_target
            q1, q2 = self.critic(state, action)

            # CQL
            state_rep = torch.repeat_interleave(state, self.repeats, 0)
            next_state_rep = torch.repeat_interleave(next_state, self.repeats, 0)
            with torch.no_grad():
                actions_rep, log_probs_rep = self.actor.sample_normal(state_rep)
                next_actions_rep, next_log_probs_rep = self.actor.sample_normal(next_state_rep)
            rand_actions = torch.Tensor(np.random.uniform(-1, 1, size=actions_rep.shape)).to(self.device)
            rand_probs = np.log(0.5 ** self.action_dim)

            out_dist_a, out_dist_b = self.critic(state_rep, actions_rep)
            out_dist_a = (out_dist_a - log_probs_rep).reshape(self.batch_size, self.repeats)
            out_dist_b = (out_dist_b - log_probs_rep).reshape(self.batch_size, self.repeats)
            next_out_dist_a, next_out_dist_b = self.critic(state_rep, next_actions_rep)
            next_out_dist_a = (next_out_dist_a - next_log_probs_rep).reshape(self.batch_size, self.repeats)
            next_out_dist_b = (next_out_dist_b - next_log_probs_rep).reshape(self.batch_size, self.repeats)
            rand_dist_a, rand_dist_b = self.critic(state_rep, rand_actions)
            rand_dist_a = (rand_dist_a - rand_probs).reshape(self.batch_size, self.repeats)
            rand_dist_b = (rand_dist_b - rand_probs).reshape(self.batch_size, self.repeats)

            catq1 = torch.cat([rand_dist_a, next_out_dist_a, out_dist_a], 1)
            catq2 = torch.cat([rand_dist_b, next_out_dist_b, out_dist_b], 1)
            logsumexp1 = torch.logsumexp(catq1, 1)
            logsumexp2 = torch.logsumexp(catq2, 1)
            p1 = logsumexp1.mean() - q1.mean()
            p2 = logsumexp2.mean() - q2.mean()

            critic_loss = self.alpha_prime * (p1 + p2) + F.mse_loss(q1, q_hat) + F.mse_loss(q2, q_hat)

            ### Backprop ###
            self.alpha_loss_history.append(alpha_loss.item())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Soft-update Target Networks #
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
