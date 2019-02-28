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

for seed in {0..10..1} 
    do
        python3.6 main.py simple_spread_flip_3 eval_graph_relative \
        --seed $seed \
        --n_episodes 10000 \
        --flip_ep 4000 \
        --init_noise_scale 0.3 \
        --n_exploration_eps 3000 \
        --hidden_dim 256 \
        --log_comment "no_distill"

        python3.6 main.py simple_spread_flip_3 eval_graph_relative \
        --seed $seed \
        --n_episodes 10000 \
        --flip_ep 4000 \
        --init_noise_scale 0.3 \
        --n_exploration_eps 3000 \
        --hidden_dim 256 \
        --distill_ep 4000 \
        --log_comment "distill_all"
        
        python3.6 main.py simple_spread_flip_3 eval_graph_relative \
        --seed $seed \
        --n_episodes 10000 \
        --flip_ep 4000 \
        --init_noise_scale 0.3 \
        --n_exploration_eps 3000 \
        --hidden_dim 256 \
        --distill_ep 4000 \
        --distill_pass_actor \
        --log_comment "critic_only"
        
        python3.6 main.py simple_spread_flip_3 eval_graph_relative \
        --seed $seed \
        --n_episodes 10000 \
        --flip_ep 4000 \
        --init_noise_scale 0.3 \
        --n_exploration_eps 3000 \
        --hidden_dim 256 \
        --distill_ep 4000 \
        --distill_pass_critic \
        --log_comment "actor_only"
    done
