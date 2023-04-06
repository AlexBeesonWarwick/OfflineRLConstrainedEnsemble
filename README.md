# Balancing policy constraint and ensemble size in uncertainty-based offline reinforcement learning

This repository provides the code used to produce the results in our paper - https://arxiv.org/abs/2303.14716

The algorithms used in the work can be found in the folder "Algorithms"

We provide individual examples trained on the D4RL datasets, one for each domain (MuJoCo, Maze2d, AntMaze, Adroit).  To train on a different dataset, simply replace the dataset name under the "Load environment" section of the code.

Installation instructions for D4RL can be found here - https://github.com/Farama-Foundation/D4RL.  Note that D4RL is based on OpenAI's MuJoCo - https://github.com/openai/mujoco-py (not Deepmind's)

If you experience any problems, please raise an issue or pull request.
