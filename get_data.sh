#!/bin/bash
seed=1

python main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--log_comment "no_distill" \

python main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--log_comment "distill_all" \

python main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--log_comment "critic_only" \

python main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--distill_pass_critic \
--log_comment "actor_only" \

python main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--distill_ep 6000 \
--distill_pass_critic \
--flip_critic \
--log_comment "ours" \

python main.py simple_spread_hard eval_graph_relative \
--seed $seed \
--init_noise_scale 0.7 \
--episode_length 50  \
--n_episodes 15000 \
--flip_ep 6000 \
--log_comment "critic_flip_only" \
--flip_critic \