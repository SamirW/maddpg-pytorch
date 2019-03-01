#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Tensorboard
pkill tensorboard

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Add baseline package to path
export PYTHONPATH=$DIR/thirdparty/multiagent-particle-envs:$PYTHONPATH

# Train tf 
print_header "Training network"
cd $DIR

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

python3.6 main_distill.py complex_push mode2 \
--seed 0 \
--n_episodes 5000 \
--init_noise_scale 0.3 \
--final_noise_scale 0 \
--n_exploration_eps 3000 \
--hidden_dim 256 \
--episode_length 100 \
--log_comment "no_distill"
