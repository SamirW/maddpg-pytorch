#!/usr/bin/env bash
python main.py simple_spread eval_graph --n_episodes 10000 --hard_distill_ep 10000 --log_comment "distill" --seed 3
python main.py simple_spread eval_graph --n_episodes 10000 --hard_distill_ep 10000 --log_comment "distill" --seed 4
python main.py simple_spread eval_graph --n_episodes 10000 --hard_distill_ep 10000 --log_comment "distill" --seed 5
python main.py simple_spread eval_graph --n_episodes 10000 --hard_distill_ep 10000 --log_comment "distill" --seed 7
python main.py simple_spread eval_graph --n_episodes 10000 --hard_distill_ep 10000 --log_comment "distill" --seed 8


# git add .
# git commit -m "Eval graph data 4 agent"
# git push origin analyzing_flip