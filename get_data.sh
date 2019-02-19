#!/bin/bash
seed=1

python main.py simple_spread_flip_4 heatmap_relative --seed $seed --n_episodes 8000 --log_comment "heatmap_data" --save_buffer

# while [ $seed -le 3 ];
# do
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --log_comment "no_distill" 
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --log_comment "distill"
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --distill_pass_actor --log_comment "distill_pass_actor"
# 	python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --distill_pass_critic --log_comment "distill_pass_critic"
# 	((seed++))
# done
