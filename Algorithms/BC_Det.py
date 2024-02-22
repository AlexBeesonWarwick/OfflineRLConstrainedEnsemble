# Import modules
import numpy as np
import random
import copy
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))

        return a


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, batch_size=256, lr=3e-4, device="cpu"):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.device = device
        self.batch_size = batch_size
        self.actor_loss_history = []

    def choose_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def train_bc(self, replay_buffer, iterations=1):
        self.actor.train()
        for it in range(iterations):
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)

            actor_loss = F.mse_loss(self.actor(state), action)

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
