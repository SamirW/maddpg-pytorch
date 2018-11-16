#!/usr/bin/env bash
python main.py simple_spread_flip simple_network --n_episodes 15000 --flip_ep 5000 --log_comment "distill" --hard_distill_ep 5000 --seed 3
python main.py simple_spread_flip simple_network --n_episodes 15000 --flip_ep 5000 --log_comment "distill" --hard_distill_ep 5000 --seed 4
python main.py simple_spread_flip simple_network --n_episodes 15000 --flip_ep 5000 --log_comment "distill" --hard_distill_ep 5000 --seed 5
python main.py simple_spread_flip simple_network --n_episodes 15000 --flip_ep 5000 --log_comment "distill" --hard_distill_ep 5000 --seed 6
python main.py simple_spread_flip simple_network --n_episodes 15000 --flip_ep 5000 --log_comment "distill" --hard_distill_ep 5000 --seed 7
python main.py simple_spread_flip simple_network --n_episodes 15000 --flip_ep 5000 --log_comment "distill" --hard_distill_ep 5000 --seed 8
# git add .
# git commit -m "Hard distill data"
# git push origin analyzing_flip