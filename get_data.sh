#!/usr/bin/env bash
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 6
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 7
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 8
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 9
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 10