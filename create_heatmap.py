import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from algorithms.maddpg import MADDPG
from utils.heatmap import *
from utils.buffer import ReplayBuffer

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name / "run{}".format(config.run)
    model_file = str(model_dir / "model.pt")

    if config.env_id == "simple_spread_flip_4":
        heatmap_fn = heatmap_4 
        distilled_heatmap_fn = distilled_heatmap_4
    else:
        heatmap_fn = heatmap
        distilled_heatmap_fn = distilled_heatmap

    maddpg = MADDPG.init_from_save(model_file)
    heatmap_fn(maddpg, title="Agent Policies Before Distillation", save=config.save)

    with open(str(model_dir / "replay_buffer.pkl"), 'rb') as input:
        replay_buffer = pickle.load(input)

    maddpg.distill(256, 1024, replay_buffer, hard=True)
    distilled_heatmap_fn(maddpg, save=config.save)

    # test()

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("run", help="Run number")
    parser.add_argument("--save",
                        action="store_true",
                        default=False)
    config = parser.parse_args()

    run(config)