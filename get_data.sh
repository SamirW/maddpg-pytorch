#!/bin/bash
seed=1

python main.py complex_push test --episode_length 100 --seed $seed --init_noise_scale 0.9 --n_episodes 10000 --log_comment "test" --display_every 50

# while [ $seed -le 3 ];
# do
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --log_comment "no_distill" 
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --log_comment "distill"
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --distill_pass_actor --log_comment "distill_pass_actor"
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --distill_pass_critic --log_comment "distill_pass_critic"
# 	((seed++))
# done
