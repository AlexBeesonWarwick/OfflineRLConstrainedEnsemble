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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, hidden_dim=256, log_std_min=-20.0, log_std_max=2.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
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
        mean = self.mean(a)
        std = self.log_std(a).clamp(self.log_std_min, self.log_std_max).exp()

        return mean, std

    def log_prob_bc(self, state, action=None):
        mu, sigma = self.forward(state)
        dist = TransformedDistribution(Independent(Normal(mu, sigma), 1), [
            TanhTransform(cache_size=1), AffineTransform(self.action_mean, self.action_scale, cache_size=1)])

        action = action.clamp(self.min_action + self.eps, self.max_action - self.eps)
        log_prob_bc = dist.log_prob(action)
        return log_prob_bc


class Agent():
    def __init__(self, state_dim, action_dim, min_action, max_action, batch_size=256, lr_actor=3e-4, device="cpu"):

        # Initialisation
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Record losses
        self.actor_loss_history = []

        # Set remaining parameters
        self.batch_size = batch_size
        self.device = device
        self.max_action = max_action

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            action = self.max_action * torch.tanh(self.actor(state)[0])

        return action.cpu().numpy().flatten()

    def train_bc(self, replay_buffer, iterations=1):

        for it in range(iterations):
            # Sample batch from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)

            actor_loss = -self.actor.log_prob_bc(state, action).mean()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
