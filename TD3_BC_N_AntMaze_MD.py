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
env = gym.make('antmaze-medium-diverse-v0')
dataset = d4rl.qlearning_dataset(env)

# Set seeds
seed = 19636
offset = 100
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Convert D4RL to replay buffer
print("Converting data...")
'''
In v0 of antmaze data sets, the timeout flags are not synced with actual trajectory ends
see - https://github.com/Farama-Foundation/D4RL/issues/77
These end transitions can be identified as having a state/next-state L2-norm of > 0.5
These transitions are removed prior to training
Note this is similar to MSG, which removes the final transition from each trajectory
 - https://github.com/google-research/google-research/blob/master/jrl/data/d4rl_get_dataset.py
'''
states = dataset["observations"]
next_states = dataset["next_observations"]
distance = np.linalg.norm(states[:, :2] - next_states[:, :2], axis=-1)

mean = np.mean(dataset["observations"][distance <= 0.5], 0)
std = np.std(dataset["observations"][distance <= 0.5], 0) + 1e-3
states = torch.Tensor((dataset["observations"][distance <= 0.5] - mean) / std)
actions = torch.Tensor(dataset["actions"][distance <= 0.5])
rewards = dataset["rewards"][distance <= 0.5]
rewards = 4 * (rewards - 0.5)
rewards = torch.Tensor(rewards)
next_states = torch.Tensor((dataset["next_observations"][distance <= 0.5] - mean) / std)
dones = torch.Tensor(dataset["terminals"][distance <= 0.5])
replay_buffer = [states, actions, rewards, next_states, dones]
print("...data conversion complete")

# Hyperparameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
beta = 0.02
higher_bc_period = 50000
num_critics = 10
dep_targets = False
device = "cuda:0"

agent = TD3_BC_N.Agent(state_dim, action_dim, max_action, num_critics, device=device)

# Training #
'''
Reset goal each episode to evaluate as per https://github.com/Farama-Foundation/D4RL/pull/128
and lines 189-198 of https://github.com/Farama-Foundation/D4RL/pull/128/commits/724c37483a3ff9d8106107344742566eda4a11d6
'''
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
        goal = env.goal_sampler(np.random)
        env.set_target_goal(goal)
        while not done:
            state = (state - mean) / std
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            score += reward
        score_norm = 100 * env.get_normalized_score(score)
        scores_norm_mean.append(score_norm)

    print("Epoch", epoch, "Grad steps", grad_steps, "Score Norm (Mean) %.2f" % np.mean(scores_norm_mean))
