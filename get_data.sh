#!/bin/bash
# python main.py simple_spread_flip test_intelligent_flip --seed 3 --n_episodes 1500 --hard_distill_ep 1500 --log_comment "heatmap_data" --save_buffer

seed=3
while [ $seed -lt 4 ];
do
	python main.py simple_spread_flip eval_graph --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 1500 --log_comment "no_distill" 
	# python main.py simple_spread_flip test_intelligent_flip --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 1500 --hard_distill_ep 1500 --log_comment "distill"
# 	python main.py simple_spread_flip eval_graph --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 1500 --hard_distill_ep 1500 --distill_pass_actor --log_comment "distill_pass_actor"
	python main.py simple_spread_flip eval_graph --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 1500 --hard_distill_ep 1500 --distill_pass_critic --log_comment "distill_pass_critic"
	((seed++))
done
