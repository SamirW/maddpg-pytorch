#!/usr/bin/env bash
python main.py simple_spread_flip analyze_vf_flipped_input --seed 5 --n_episodes 1500 --hard_distill_ep 1500 --log_comment "heatmap_data" --save_buffer --display_every 100

# python main.py simple_spread_flip analyze_vf --seed 3 --n_episodes 2500 --model_save_freq 250 --flip_ep 0 --log_comment "baseline" 
# python main.py simple_spread_flip analyze_vf --seed 3 --n_episodes 2500 --model_save_freq 250 --flip_ep 750 --hard_distill_ep 750 --eval_ep 750 --log_comment "distill_eval"
# python main.py simple_spread_flip analyze_vf --seed 3 --n_episodes 2500 --model_save_freq 250 --flip_ep 750 --eval_ep 750 --log_comment "no_distill_eval" 
# python main.py simple_spread_flip_4 analyze_vf --seed 3 --n_episodes 2500 --model_save_freq 500 --flip_ep 1500 --hard_distill_ep 1500 --log_comment "distill_learn"
# python main.py simple_spread_flip analyze_vf --seed 3 --n_episodes 2500 --model_save_freq 250 --flip_ep 750 --hard_distill_ep 750 --skip_actor_length 250 --log_comment "distill_learn_skip_1000"

# python main.py simple_spread_flip analyze_vf --seed 4 --n_episodes 2500 --flip_ep 0 --log_comment "baseline" 
# python main.py simple_spread_flip analyze_vf --seed 4 --n_episodes 2500 --flip_ep 750 --eval_ep 750 --log_comment "no_distill_eval" 
# python main.py simple_spread_flip analyze_vf --seed 4 --n_episodes 2500 --flip_ep 750 --hard_distill_ep 750 --eval_ep 750 --log_comment "distill_eval"
# python main.py simple_spread_flip analyze_vf --seed 4 --n_episodes 2500 --flip_ep 750 --hard_distill_ep 750 --log_comment "distill_learn"
# python main.py simple_spread_flip analyze_vf --seed 4 --n_episodes 2500 --flip_ep 750 --hard_distill_ep 750 --skip_actor_length 250 --log_comment "distill_learn_skip_1000"

# python main.py simple_spread_flip analyze_vf --seed 5 --n_episodes 2500 --flip_ep 0 --log_comment "baseline" 
# python main.py simple_spread_flip analyze_vf --seed 5 --n_episodes 2500 --flip_ep 750 --eval_ep 750 --log_comment "no_distill_eval" 
# python main.py simple_spread_flip analyze_vf --seed 5 --n_episodes 2500 --flip_ep 750 --hard_distill_ep 750 --eval_ep 750 --log_comment "distill_eval"
# python main.py simple_spread_flip analyze_vf --seed 5 --n_episodes 2500 --flip_ep 750 --hard_distill_ep 750 --log_comment "distill_learn"
# python main.py simple_spread_flip analyze_vf --seed 5 --n_episodes 2500 --flip_ep 750 --hard_distill_ep 750 --skip_actor_length 250 --log_comment "distill_learn_skip_1000"

# python main.py simple_spread_flip_6 eval_graph --seed 3 --n_episodes 4000 --log_comment "heatmap_data" --save_buffer --display_every 100
# python main.py simple_spread_flip_6 eval_graph --seed 3 --n_episodes 10000 --flip_ep 0 --log_comment "baseline"
# python main.py simple_spread_flip_6 eval_graph --seed 3 --n_episodes 10000 --flip_ep 4000 --hard_distill_ep 4000 --eval_ep 4000 --log_comment "distill_eval" --display_every 250
# python main.py simple_spread_flip_6 eval_graph --seed 3 --n_episodes 10000 --flip_ep 4000 --hard_distill_ep 4000 --log_comment "distill_learn"
# python main.py simple_spread_flip_6 eval_graph --seed 3 --n_episodes 10000 --flip_ep 4000 --hard_distill_ep 4000 --skip_actor_length 1000 --log_comment "distill_learn_skip_1000"