#!/usr/bin/env bash
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "distill" --hard_distill_ep 5000 --seed 3
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "distill" --hard_distill_ep 5000 --seed 4
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "distill" --hard_distill_ep 5000 --seed 5
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "distill" --hard_distill_ep 5000 --seed 6
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "distill" --hard_distill_ep 5000 --seed 7
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "distill" --hard_distill_ep 5000 --seed 8

python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 3
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 4
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 5
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 6
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 7
python main.py simple_spread_flip eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 8

git add .
git commit -m "Eval graph data"
git push origin analyzing_flip