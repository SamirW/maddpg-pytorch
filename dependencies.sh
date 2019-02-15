#!/bin/bash

FOLDER_NAME="maddpy-pytorch"

# pytorch
pip install torch

# gym
cd /home/ubuntu/$FOLDER_NAME/thirdparty/gym
pip install -e .

# baseline
cd /home/ubuntu/$FOLDER_NAME/thirdparty/baselines
pip install -e .

# multiagent
cd /home/ubuntu/$FOLDER_NAME/thirdparty/multiagent-particle-envs
pip install -e .

# etc
pip install tensorboardX
pip install coloredlogs
pip install pydot
