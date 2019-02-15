#!/bin/bash

FOLDER_NAME="maddpg-pytorch"

# pytorch
pip install torch

# baseline
cd /home/ubuntu/$FOLDER_NAME/thirdparty/baselines
pip install -e .

# gym
cd /home/ubuntu/$FOLDER_NAME/thirdparty/gym
pip install -e .

# multiagent
cd /home/ubuntu/$FOLDER_NAME/thirdparty/multiagent-particle-envs
pip install -e .

# etc
pip install tensorboardX