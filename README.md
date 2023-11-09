# Balancing policy constraint and ensemble size in uncertainty-based offline reinforcement learning.

This repository contains the code used to produce the results in our paper - https://link.springer.com/article/10.1007/s10994-023-06458-y.

The algorithms used in the work can be found in the folder "Algorithms".

Our work makes use of the D4RL benchmarking suite.  Installation instructions can be found here - https://github.com/Farama-Foundation/D4RL.  Note that D4RL is based on OpenAI's MuJoCo - https://github.com/openai/mujoco-py (not Deepmind's).

## Offline reinforcement learning
We provide individual examples trained on the D4RL datasets, one for each domain (MuJoCo, Maze2d, AntMaze, Adroit).  To train on a different dataset, simply replace the dataset name under the "Load environment" section of the code.  Note that agents are evaluated every 10,000 gradient updates only as a means of tracking progress and to check the code runs correctly.  In our paper, all agents are trained for 1M gradient updates and the policy at the last iteration used for evaluation.

## Online fine-tuning
We provide individual examples, one for each approach (TD3-BC-N and SAC-BC-N).  To fine-tune on a different dataset, replace the dataset names under the "Load environment" section of the code.  Remember to apply any data transformations to newly aquired interactions (e.g. state normalisation, reward scaling).

## Computational efficiency
We provide examples of calculating computation time for 10,000 gradient updates for TD3-BC-N and SAC-BC-N.  Other algorithms can be tested by simply amending the import.

## Feedback
If you experience any problems or have any queries, please raise an issue or pull request.
