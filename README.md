# Balancing policy constraint and ensemble size in uncertainty-based offline reinforcement learning.

This repository provides the code used to produce the results in our paper - https://arxiv.org/abs/2303.14716.

The algorithms used in the work can be found in the folder "Algorithms".

Our work makes use of the D4RL benchmarking suite.  Installation instructions can be found here - https://github.com/Farama-Foundation/D4RL.  Note that D4RL is based on OpenAI's MuJoCo - https://github.com/openai/mujoco-py (not Deepmind's).

## Offline reinforcement learning
We provide individual examples trained on the D4RL datasets, one for each domain (MuJoCo, Maze2d, AntMaze, Adroit).  To train on a different dataset, simply replace the dataset name under the "Load environment" section of the code.

## Online fine-tuning
We provide individual examples, one for each approach (TD3-BC-N and SAC-BC-N).  To fine-tune on a different dataset, replace the dataset names under the "Load environment" section of the code.  Remember to apply any data transformations to newly aquired interactions (e.g. state normalisation, reward scaling).

## Computational efficiency
We provide examples of calculating computation time for 10,000 gradient updates for TD3-BC-N and SAC-BC-N.  Other algorithms can be tested by simply amending the import

If you experience any problems, please raise an issue or pull request.
