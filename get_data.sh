#!/bin/bash
seed=1

python main.py simple_spread_flip heatmap --seed $seed --n_episodes 2000 --log_comment "heatmap"

while [ $seed -le 3 ];
do
	python main.py simple_spread_flip eval_graph_relative --seed $seed --n_episodes 10000 --flip_ep 2000 --log_comment "no_distill" 
	python main.py simple_spread_flip eval_graph_relative --seed $seed --n_episodes 10000 --flip_ep 2000 --hard_distill_ep 2000 --log_comment "distill"
	python main.py simple_spread_flip eval_graph_relative --seed $seed --n_episodes 10000 --flip_ep 2000 --hard_distill_ep 2000 --distill_pass_actor --log_comment "distill_pass_actor"
	python main.py simple_spread_flip eval_graph_relative --seed $seed --n_episodes 10000 --flip_ep 2000 --hard_distill_ep 2000 --distill_pass_critic --log_comment "distill_pass_critic"
	((seed++))
done
