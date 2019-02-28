#!/bin/bash
seed=1

python main.py simple_spread_flip test --seed $seed --n_episodes 1000 --log_comment "test" --flip_ep 500 --display_every 100 --flip_critic

# while [ $seed -le 3 ];
# do
# 	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 15000 --flip_ep 7500  --log_comment "no_distill"
# 	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 15000 --flip_ep 7500  --eval_ep 7500 --log_comment "no_distill_eval"
# 	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 15000 --flip_ep 7500 --hard_distill_ep 7500 --log_comment "distill"
# 	python main.py simple_spread_hard eval_graph_relative --seed $seed --init_noise_scale 0.7 --episode_length 50 --n_episodes 15000 --flip_ep 7500 --hard_distill_ep 7500 --eval_ep 7500 --log_comment "distill_eval"
# 	# python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --distill_pass_actor --log_comment "distill_pass_actor"
# 	# python main.py simple_spread_flip_4 eval_graph_relative --seed $seed --n_episodes 15000 --flip_ep 4000 --hard_distill_ep 4000 --distill_pass_critic --log_comment "distill_pass_critic"
# 	((seed++))
# done
