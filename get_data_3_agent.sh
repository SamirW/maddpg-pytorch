#!/bin/bash
seed=1

python main.py simple_spread_flip_3 eval_graph_relative \
--seed $seed \
--n_episodes 10000 \
--flip_ep 4000 \
--log_comment "no_distill" \

python main.py simple_spread_flip_3 eval_graph_relative \
--seed $seed \
--n_episodes 10000 \
--flip_ep 4000 \
--distill_ep 4000 \
--log_comment "distill_all" \

python main.py simple_spread_flip_3 eval_graph_relative \
--seed $seed \
--n_episodes 10000 \
--flip_ep 4000 \
--distill_ep 4000 \
--log_comment "critic_only" \

python main.py simple_spread_flip_3 eval_graph_relative \
--seed $seed \
--n_episodes 10000 \
--flip_ep 4000 \
--distill_ep 4000 \
--distill_pass_critic \
--log_comment "actor_only" \

python main.py simple_spread_flip_3 eval_graph_relative \
--seed $seed \
--n_episodes 10000 \
--flip_ep 4000 \
--distill_ep 4000 \
--distill_pass_critic \
--flip_critic \
--log_comment "ours" \

python main.py simple_spread_flip_3 eval_graph_relative \
--seed $seed \
--n_episodes 10000 \
--flip_ep 4000 \
--log_comment "critic_flip_only" \
--flip_critic \