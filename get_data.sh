#!/bin/bash
seed=1

# python main.py simple_spread_hard test --seed $seed --n_episodes 7500 --init_noise_scale 0.7 --episode_length 50 --log_comment "test" 

while [ $seed -le 2 ];
do
	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 25000 --flip_ep 7500  --log_comment "no_distill"
	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 25000 --flip_ep 7500 --hard_distill_ep 7500 --log_comment "distill"
	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 25000 --flip_ep 7500 --hard_distill_ep 7500 --distill_pass_actor --log_comment "distill_pass_actor"
	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 25000 --flip_ep 7500 --hard_distill_ep 7500 --distill_pass_critic --log_comment "distill_pass_critic"
	((seed++))
done
