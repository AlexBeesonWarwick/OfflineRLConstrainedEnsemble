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
Original MSG code - https://github.com/google-research/google-research/tree/master/jrl
Vectorised Linear code - https://github.com/tinkoff-ai/CORL/blob/main/algorithms/sac_n.py

There are three versions that differ only in how they calculate the cql component of the critic loss:
    Agent: This version uses the difference between policy actions and data actions.  This is the most computationally
           efficient version and the one used for efficiency comparisons in the paper.
    Agent_IS: This version uses importance sampling with only one sample.  This is optimized for a single sample by 
              reusing the action/log probability used in the actor loss.  A sample size of one is what is used in the 
              MSG paper (https://github.com/google-research/google-research/tree/master/jrl/agents/msg)
    Agent_IS_Rep: This version uses importance sampling with user specified number of samples (reps).  Using a sample
                  size of one is equivalent to Agent_IS but slightly less computationally efficient as an additional
                  action/log probability is sampled from the policy.  Obviously as the number of samples increases
                  so does the computation time.
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
        self.l2b = VectorizedLinear(hidden_dim, hidden_dim, num_critics)
        self.l3 = VectorizedLinear(hidden_dim, 1, num_critics)

        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        q_values = F.relu(self.l1(state_action))
        q_values = F.relu(self.l2(q_values))
        q_values = F.relu(self.l2b(q_values))
        q_values = self.l3(q_values)

        return q_values.squeeze(-1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, hidden_dim=256, log_std_min=-20.0, log_std_max=2.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2b = nn.Linear(hidden_dim, hidden_dim)
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
        a = F.relu(self.l2b(a))
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


class Agent():
    def __init__(self, state_dim, action_dim, min_action, max_action, min_ent=0, num_critics=2, alpha_prime=1, beta=-1,
                 batch_size=256, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, device="cpu"):

        # Initialisation
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = VectorizedCritic(state_dim, action_dim, num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
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
        self.alpha_prime = alpha_prime
        self.beta = beta

    def choose_action(self, state, mean=True):
        # Take mean/greedy action by default, but also allows sampling from policy
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if mean == True:
                action = self.max_action * torch.tanh(self.actor(state)[0])
            else:
                action = self.actor.sample_normal(state)[0]

        return action.cpu().numpy().flatten()

    def train_offline(self, replay_buffer, iterations=1):

        for it in range(iterations):
            # Sample batch from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)
            done = torch.index_select(replay_buffer[4], 0, indices).to(self.device)

            alpha = self.log_alpha.exp().detach()
            actions, log_actions = self.actor.sample_normal(state)

            # Critic loss #
            with torch.no_grad():
                next_action, log_next_action = self.actor.sample_normal(next_state)
                q_target = self.critic_target(next_state, next_action)
                q_target -= (alpha * log_next_action)
                q_hat = reward + self.gamma * (1 - done) * q_target
            q_values_data = self.critic(state, action)
            q_values_policy = self.critic(state, actions.detach())

            critic_loss = F.mse_loss(q_values_data, q_hat) + self.alpha_prime * (q_values_policy - q_values_data).mean()
            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor and alpha loss #
            critic_values = self.critic(state, actions) - alpha * log_actions
            actor_loss = -(critic_values.mean(0) + self.beta * critic_values.std(0)).mean()
            alpha_loss = -(self.log_alpha * (log_actions.detach() + self.min_ent)).mean()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.alpha_loss_history.append(alpha_loss.item())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ## Polyak target network updates
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


class Agent_IS():
    def __init__(self, state_dim, action_dim, min_action, max_action, min_ent=0, num_critics=2, alpha_prime=1, beta=-4,
                 batch_size=256, lr_actor=3e-5, lr_critic=3e-4, gamma=0.99, tau=0.005, device="cpu"):

        # Initialisation
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = VectorizedCritic(state_dim, action_dim, num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
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
        self.alpha_prime = alpha_prime
        self.beta = beta
        self.action_dim = action_dim

    def choose_action(self, state, mean=True):
        # Take mean/greedy action by default, but also allows sampling from policy
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if mean == True:
                action = self.max_action * torch.tanh(self.actor(state)[0])
            else:
                action = self.actor.sample_normal(state)[0]

        return action.cpu().numpy().flatten()

    def train_offline(self, replay_buffer, iterations=1):

        for it in range(iterations):
            # Sample batch from replay buffer
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            state = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state = torch.index_select(replay_buffer[3], 0, indices).to(self.device)
            done = torch.index_select(replay_buffer[4], 0, indices).to(self.device)

            alpha = self.log_alpha.exp().detach()
            actions, log_actions = self.actor.sample_normal(state)

            # Critic loss #
            with torch.no_grad():
                next_action, log_next_action = self.actor.sample_normal(next_state)
                q_target = self.critic_target(next_state, next_action)
                q_target -= (alpha * log_next_action)
                q_hat = reward + self.gamma * (1 - done) * q_target
            q_values_data = self.critic(state, action)
            critic_loss = F.mse_loss(q_values_data, q_hat)

            ## CQL
            rand_actions = torch.Tensor(np.random.uniform(-1, 1, size=actions.shape)).to(self.device)
            rand_probs = np.log(0.5 ** self.action_dim)

            out_dist = self.critic(state, actions.detach())
            out_dist -= (alpha * log_actions.detach())
            out_dist = out_dist.unsqueeze(2)
            rand_dist = self.critic(state, rand_actions)
            rand_dist -= (alpha * rand_probs)
            rand_dist = rand_dist.unsqueeze(2)

            catq = torch.cat([rand_dist, out_dist], 2)
            logsumexp = torch.logsumexp(catq, 2)
            cql_loss = (logsumexp - q_values_data).mean()
            critic_loss += (self.alpha_prime * cql_loss)

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor and alpha loss #
            critic_values = self.critic(state, actions) - alpha * log_actions
            actor_loss = -(critic_values.mean(0) + self.beta * critic_values.std(0)).mean()
            alpha_loss = -(self.log_alpha * (log_actions.detach() + self.min_ent)).mean()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.alpha_loss_history.append(alpha_loss.item())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ## Polyak target network updates
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


class Agent_IS_Reps():
    def __init__(self, state_dim, action_dim, min_action, max_action, min_ent=0, num_critics=2, alpha_prime=1, beta=-1,
                 batch_size=256, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, reps=1, device="cpu"):

        # Initialisation
        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = VectorizedCritic(state_dim, action_dim, num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
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
        self.alpha_prime = alpha_prime
        self.beta = beta
        self.reps = reps
        self.action_dim = action_dim

    def choose_action(self, state, mean=True):
        # Take mean/greedy action by default, but also allows sampling from policy
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if mean == True:
                action = self.max_action * torch.tanh(self.actor(state)[0])
            else:
                action = self.actor.sample_normal(state)[0]

        return action.cpu().numpy().flatten()

    def train_offline(self, replay_buffer, iterations=1):

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
                q_target -= (alpha * log_next_action)
                q_hat = reward + self.gamma * (1 - done) * q_target
            q_values_data = self.critic(state, action)
            critic_loss = F.mse_loss(q_values_data, q_hat)

            ## CQL
            state_rep = torch.repeat_interleave(state, self.reps, 0)
            with torch.no_grad():
                actions_rep, log_probs_rep = self.actor.sample_normal(state_rep)
            rand_actions = torch.Tensor(np.random.uniform(-1, 1, size=actions_rep.shape)).to(self.device)
            rand_probs = np.log(0.5 ** self.action_dim)

            out_dist = self.critic(state_rep, actions_rep)
            out_dist -= (alpha * log_probs_rep)
            out_dist = out_dist.reshape(self.num_critics, self.batch_size, self.reps)
            rand_dist = self.critic(state_rep, rand_actions)
            rand_dist -= (alpha * rand_probs)
            rand_dist = rand_dist.reshape(self.num_critics, self.batch_size, self.reps)

            catq = torch.cat([rand_dist, out_dist], 2)
            logsumexp = torch.logsumexp(catq, 2)
            cql_loss = (logsumexp - q_values_data).mean()
            critic_loss += (self.alpha_prime * cql_loss)

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor and alpha loss #
            actions, log_actions = self.actor.sample_normal(state)
            critic_values = self.critic(state, actions) - alpha * log_actions
            actor_loss = -(critic_values.mean(0) + self.beta * critic_values.std(0)).mean()
            alpha_loss = -(self.log_alpha * (log_actions.detach() + self.min_ent)).mean()

            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.alpha_loss_history.append(alpha_loss.item())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            ## Polyak target network updates
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
