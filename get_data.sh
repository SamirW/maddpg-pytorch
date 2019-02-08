#!/usr/bin/env bash
# python main.py simple_spread_flip_4 eval_graph --n_episodes 1500 --hard_distill_ep 1500 --log_comment "distill" --seed 3
# python main.py simple_spread_flip_4 eval_graph --n_episodes 1500 --hard_distill_ep 1500 --log_comment "distill" --seed 4
python main.py simple_spread_flip_4 eval_graph --n_episodes 1500 --hard_distill_ep 1500 --log_comment "distill" --seed 5 --save_buffer

# python main.py simple_spread_flip_4 eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 3
# python main.py simple_spread_flip_4 eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 4
# python main.py simple_spread_flip_4 eval_graph --n_episodes 1500 --log_comment "no_distill" --seed 5

# python create_graph.py 

# git add .
# git commit -m "Eval naive sharing"
# git push origin analyzing_reset