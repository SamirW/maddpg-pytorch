#!/usr/bin/env bash
python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "distill" --hard_distill_ep 2500 --seed 3
python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "no_distill" --seed 3

python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "distill" --hard_distill_ep 2500 --seed 4
python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "no_distill" --seed 4

python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "distill" --hard_distill_ep 2500 --seed 5
python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "no_distill" --seed 5

python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "distill" --hard_distill_ep 2500 --seed 7
python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "no_distill" --seed 7

python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "distill" --hard_distill_ep 2500 --seed 8
python main.py simple_spread_flip_4 eval_graph --n_episodes 2500 --log_comment "no_distill" --seed 8


git add .
git commit -m "Eval graph data 4 agent"
git push origin analyzing_flip