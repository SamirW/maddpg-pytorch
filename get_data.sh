#!/usr/bin/env bash
python main.py simple_spread eval_graph --n_episodes 6000 --hard_distill_ep 6000 --log_comment "entropy_distill_sharpened" --seed 3
python main.py simple_spread eval_graph --n_episodes 6000 --hard_distill_ep 6000 --log_comment "entropy_distill_sharpened" --seed 4
python main.py simple_spread eval_graph --n_episodes 6000 --hard_distill_ep 6000 --log_comment "entropy_distill_sharpened" --seed 5
python main.py simple_spread eval_graph --n_episodes 6000 --hard_distill_ep 6000 --log_comment "entropy_distill_sharpened" --seed 7
python main.py simple_spread eval_graph --n_episodes 6000 --hard_distill_ep 6000 --log_comment "entropy_distill_sharpened" --seed 8

# python main.py simple_spread eval_graph --n_episodes 6000 --log_comment "no_distill" --seed 3
# python main.py simple_spread eval_graph --n_episodes 6000 --log_comment "no_distill" --seed 4
# python main.py simple_spread eval_graph --n_episodes 6000 --log_comment "no_distill" --seed 5
# python main.py simple_spread eval_graph --n_episodes 6000 --log_comment "no_distill" --seed 7
# python main.py simple_spread eval_graph --n_episodes 6000 --log_comment "no_distill" --seed 8

# python create_graph.py 

# git add .
# git commit -m "Eval naive sharing"
# git push origin analyzing_reset