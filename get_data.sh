#!/bin/bash
seed=1

while [ $seed -le 3 ];
do
	python main.py simple_spread deepset --seed $seed --n_episodes 20000 --log_comment "baseline" 
	python main.py simple_spread deepset --seed $seed --n_episodes 20000 --log_comment "deepset" --deepset

	python main.py simple_spread_3 deepset --seed $seed --n_episodes 20000 --log_comment "baseline" 
	python main.py simple_spread_3 deepset --seed $seed --n_episodes 20000 --log_comment "deepset" --deepset

	python main.py simple_spread_4 deepset --seed $seed --n_episodes 20000 --log_comment "baseline" 
	python main.py simple_spread_4 deepset --seed $seed --n_episodes 20000 --log_comment "deepset" --deepset

	# python main.py simple_spread_5 deepset --seed $seed --n_episodes 30000 --log_comment "baseline" 
	# python main.py simple_spread_5 deepset --seed $seed --n_episodes 30000 --log_comment "deepset" --deepset
	
	((seed++))
done
