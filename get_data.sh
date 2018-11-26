#!/usr/bin/env bash
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 1
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 2
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 3
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 4
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 5
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 6
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 7
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "distill" --hard_distill_ep 1500 --seed 8

python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 1
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 2
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 3
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 4
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 5
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 6
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 7
python main.py simple_spread_flip eval_graph_random --n_episodes 1500 --log_comment "no_distill" --seed 8

git add .
git commit -m "Random eval graph data"
git push origin analyzing_flip