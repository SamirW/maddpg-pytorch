#!/bin/bash
seed=1

while [ $seed -le 3 ];
do
	python main.py simple_spread_flip_4 training_set_size_2 --seed $seed --n_episodes 30000 --flip_ep 15000 --log_comment "no_distill" --display_every 250
	python main.py simple_spread_flip_4 training_set_size_2 --seed $seed --n_episodes 30000 --flip_ep 15000 --hard_distill_ep 15000 --log_comment "distill"
	python main.py simple_spread_flip_4 training_set_size_2 --seed $seed --n_episodes 30000 --flip_ep 15000 --hard_distill_ep 15000 --distill_pass_actor --log_comment "distill_pass_actor"
	python main.py simple_spread_flip_4 training_set_size_2 --seed $seed --n_episodes 30000 --flip_ep 15000 --hard_distill_ep 15000 --distill_pass_critic --log_comment "distill_pass_critic"
	((seed++))
done
