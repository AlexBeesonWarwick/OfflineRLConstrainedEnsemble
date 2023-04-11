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
env_eval = gym.make('hopper-medium-v2')
dataset = d4rl.qlearning_dataset(env)

seed = 19636
offset = 100
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Convert D4RL to replay buffer
print("Converting data...")
states = dataset["observations"]
actions = dataset["actions"]
rewards = dataset["rewards"]
next_states = dataset["next_observations"]
dones = dataset["terminals"]
mean = np.mean(dataset["observations"], 0)
std = np.std(dataset["observations"], 0) + 1e-3

states = (states - mean) / std
next_states = (next_states - mean) / std

replay_buffer = []

for j in range(len(states)):
    replay_buffer.append((states[j], actions[j], rewards[j], next_states[j], dones[j]))
replay_buffer = replay_buffer[-2500:]

print("...data conversion complete")

# Hyperparameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]
min_ent = -action_dim
num_critics = 10
dep_targets = True
device = "cuda:0"
max_steps = 250e3
memory_size = max_steps
train_starts = 5000
log_alpha = torch.load("YourPathHere", map_location=device).item()

beta_start = 2.5e-3
beta_end = 1e-3
decay_steps = 50e3
beta_decay = np.exp(np.log(beta_end / beta_start) / decay_steps)
beta = beta_start

# Initial and load pre-train agent
agent = SAC_BC_N.Agent(state_dim, action_dim, min_action, max_action, min_ent, num_critics, online_ft=log_alpha, device=device)
agent.actor.load_state_dict(torch.load("YourPathHere", map_location=device))
agent.critic.load_state_dict(torch.load("YourPathHere", map_location=device))
agent.critic_target.load_state_dict(torch.load("YourPathHere", map_location=device))

# Training FT #
env_steps = 0
episodes = 0
eval_every = 5000
evals = 10

while env_steps < max_steps + 1:
    done = False
    state = env.reset()
    state = (state - mean) / std
    step_env = 0
    while not done:
        action = agent.choose_action(state, mean=False)
        next_state, reward, done, info = env.step(action)
        next_state = (next_state - mean) / std
        env_steps += 1
        step_env += 1
        if step_env == env._max_episode_steps:
            done_rb = False
            print("Max env steps reached")
        else:
            done_rb = done
        replay_buffer.append((state, action, reward, next_state, done_rb))
        state = next_state

        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)

        if len(replay_buffer) >= train_starts:
            agent.train_online_ft(replay_buffer, iterations=1, beta=beta, dep_targets=dep_targets)
            beta *= beta_decay
            beta = np.max((beta, beta_end))

        # Evaluation (every step_eval steps)
        if env_steps % eval_every == 0 and len(replay_buffer) >= train_starts:
            env_eval.seed(seed + offset)
            scores_mean = []
            scores_norm_mean = []
            for e in range(evals):
                done_eval = False
                state_eval = env_eval.reset()
                score_eval = 0
                while not done_eval:
                    with torch.no_grad():
                        state_eval = (state_eval - mean) / std
                        action_eval = agent.choose_action(state_eval)
                        state_eval, reward_eval, done_eval, info_eval = env_eval.step(action_eval)
                        score_eval += reward_eval
                score_norm = 100 * env.get_normalized_score(score_eval)
                scores_mean.append(score_eval)
                scores_norm_mean.append(score_norm)

            print("Environment steps", env_steps,
                  "Score (Mean) %.2f" % np.mean(scores_mean), "Score Norm (Mean) %.2f" % np.mean(scores_norm_mean),
                  "Beta %.6f" % beta)
