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

from Algorithms import SAC_BC_N

import d4rl

# Load environment
env = gym.make('hopper-medium-v2')
dataset = d4rl.qlearning_dataset(env)

# Set seed
seed = 19636
offset = 100
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]
min_ent = -action_dim
dep_targets = True
higher_bc_period = 50000
beta = 0.0025
num_critics = 10
device = "cuda:0"

agent = SAC_BC_N.Agent(state_dim, action_dim, min_action, max_action, min_ent, num_critics, device=device)

# Training SAC-BC-N #
epochs = 100
iterations = 10000
grad_steps = 0
evals = 10

for epoch in range(epochs):
    if grad_steps < higher_bc_period:
        agent.train_offline(replay_buffer, iterations, 10 * beta, dep_targets)
    else:
        agent.train_offline(replay_buffer, iterations, beta, dep_targets)
    grad_steps += iterations

    # Evaluation (mean) #
    env.seed(seed + offset)
    scores_norm_mean = []
    for eval in range(evals):
        done = False
        state = env.reset()
        score = 0
        while not done:
            state = (state - mean) / std
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            score += reward
        score_norm = 100 * env.get_normalized_score(score)
        scores_norm_mean.append(score_norm)

    print("Epoch", epoch, "Grad steps", grad_steps, "Score Norm (Mean) %.2f" % np.mean(scores_norm_mean))
