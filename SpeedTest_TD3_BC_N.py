# Imports
import gym
import random
import numpy as np
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from Algorithms import TD3_BC_N

import d4rl

# Load environment
env = gym.make('hopper-medium-v2')
dataset = d4rl.qlearning_dataset(env)

# Convert D4RL to replay buffer
print("Converting data...")
mean = np.mean(dataset["observations"], 0)
std = np.std(dataset["observations"], 0) + 1e-3
states = torch.Tensor((dataset["observations"] - mean) / std)
actions = torch.Tensor(dataset["actions"])
rewards = torch.Tensor(dataset["rewards"])
next_states = torch.Tensor((dataset["next_observations"] - mean) / std)
dones = torch.Tensor(dataset["terminals"])
replay_buffer = [states, actions, rewards, next_states, dones]
print("...data conversion complete")

# Hyperparameters and initialisation
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_ent = -action_dim
num_critics = 10
device = "cuda:0"

agent = TD3_BC_N.Agent(state_dim, action_dim, max_action, num_critics, device=device)

# Speed test
grad_steps = 0
training_time = 0
epochs = 10
iterations = 1000

for epoch in range(epochs):
    train_start = time.time()
    agent.train_offline(replay_buffer, iterations)
    training_time += (time.time() - train_start)
    grad_steps += iterations

    print("Iterations", grad_steps, "Time(sec) %.1f" % training_time)

print("Total training time for", grad_steps, "gradient updates = %.1f" % training_time, "seconds")
