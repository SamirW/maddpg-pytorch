#!/bin/bash

# Directory of this script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# Virtualenv
cd $DIR
virtualenv venv
source $DIR/venv/bin/activate
pip3 install -r requirements.txt

# thirdparty
# cd ~/dev/acl/baselines
# pip3 install -e .

cd ~/multiagent-particle-envs
pip3 install -e .

# Tensorboard
# pkill tensorboard
# rm -rf logs/tb*
# cd $DIR
# tensorboard --logdir models/simple_spread/test &

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Train tf 
echo "Training network"
cd $DIR

# Begin experiment
python3 main.py \
	"simple_spread" \
	"test" \
	--display_every 100 \
	--n_episodes 1000