#!/bin/bash
python main.py simple_spread_flip_6 test --seed 3 --n_episodes 5000 --log_comment "heatmap_data" --save_buffer

# seed=3
# while [ $seed -lt 4 ];
# do
# 	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 2500 --log_comment "no_distill" --display_every 100 
# 	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 2500 --hard_distill_ep 2500 --log_comment "distill" --display_every 100
# 	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 2500 --hard_distill_ep 2500 --distill_pass_actor --log_comment "distill_pass_actor"
# 	python main.py simple_spread_flip_3 test --seed $seed --n_episodes 4000 --model_save_freq 100 --flip_ep 2500 --hard_distill_ep 2500 --distill_pass_critic --log_comment "distill_pass_critic"
# 	((seed++))
# done
