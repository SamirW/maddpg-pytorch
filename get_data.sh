#!/bin/bash
seed=1

python main.py simple_spread_flip_3 eval_graph --seed $seed --n_episodes 3000 --log_comment "heatmap_data" --save_buffer

while [ $seed -le 10 ];
do
	python main.py simple_spread_flip_3 eval_graph --seed $seed --n_episodes 12000 --model_save_freq 1000 --flip_ep 3000 --log_comment "no_distill" 
	python main.py simple_spread_flip_3 eval_graph --seed $seed --n_episodes 12000 --model_save_freq 1000 --flip_ep 3000 --hard_distill_ep 3000 --log_comment "distill"
	python main.py simple_spread_flip_3 eval_graph --seed $seed --n_episodes 12000 --model_save_freq 1000 --flip_ep 3000 --hard_distill_ep 3000 --distill_pass_actor --log_comment "distill_pass_actor"
	python main.py simple_spread_flip_3 eval_graph --seed $seed --n_episodes 12000 --model_save_freq 1000 --flip_ep 3000 --hard_distill_ep 3000 --distill_pass_critic --log_comment "distill_pass_critic"
	((seed++))
done
