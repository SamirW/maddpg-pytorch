#!/usr/bin/env bash
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 1
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 2
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 3
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 4
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 5
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 6
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 7
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 8
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 9
python main.py simple_spread_flip init_graph --n_episodes 6000 --flip_ep 1500 --log_comment "distill_256" --hard_distill_ep 1500 --seed 10
git add .
git commit -m "Hard distill data"
git push origin analyzing_flip