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

from Algorithms import SAC_BC_N_MSE

import d4rl

# Load environment
env = gym.make('maze2d-medium-v1')
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
states = torch.Tensor(dataset["observations"])
rewards = torch.Tensor(dataset["rewards"])
'''
Set terminals as per maze2d_set_terminals function found here:
https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/preprocessing.py
For the Maze2d tasks the target goal is fixed.  This pre-processing step sets the timeout field to True iff:
    the agent is at the goal at time t and 
    not at the goal at time t + 1    
'''
goal = np.array(env.get_target())
threshold = 0.5
xy = states[:, :2]
distances = np.linalg.norm(xy - goal, axis=-1)
at_goal = distances < threshold
timeouts = np.zeros_like(rewards)
timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

mean = np.mean(dataset["observations"], 0)
std = np.std(dataset["observations"], 0) + 1e-3
states = torch.Tensor((dataset["observations"] - mean) / std)
actions = torch.Tensor(dataset["actions"])
rewards = torch.Tensor(dataset["rewards"])
next_states = torch.Tensor((dataset["next_observations"] - mean) / std)
dones = torch.Tensor(timeouts)
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
beta = 0.05
num_critics = 10
device = "cuda:0"

agent = SAC_BC_N_MSE.Agent(state_dim, action_dim, min_action, max_action, min_ent, num_critics, device=device)

# Training SAC-BC-N #
epochs = 100
iterations = 10000
grad_steps = 0
evals = 100

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
