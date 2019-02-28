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

seed=1

python3.6 main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--log_comment "no_distill" \

python3.6 main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--log_comment "distill_all" \

python3.6 main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--log_comment "critic_only" \

python3.6 main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--distill_pass_critic \
--log_comment "actor_only" \

python3.6 main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--distill_pass_critic \
--flip_critic \
--log_comment "ours" \

python3.6 main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--log_comment "critic_flip_only" \
--flip_critic \
