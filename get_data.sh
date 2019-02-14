#!/bin/bash
# python main.py simple_spread_flip_3 test --seed 3 --n_episodes 3000 --log_comment "heatmap_data" --save_buffer

seed=3
while [ $seed -lt 9 ];
do
	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 5000 --model_save_freq 500 --flip_ep 3000 --log_comment "no_distill"
	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 5000 --model_save_freq 500 --flip_ep 3000 --hard_distill_ep 3000 --log_comment "distill"
	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 5000 --model_save_freq 500 --flip_ep 3000 --hard_distill_ep 3000 --distill_pass_actor --log_comment "distill_pass_actor"
	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 5000 --model_save_freq 500 --flip_ep 3000 --hard_distill_ep 3000 --distill_pass_critic --log_comment "distill_pass_critic"
	((seed++))
done
