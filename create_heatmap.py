import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from algorithms.maddpg import MADDPG
from utils.heatmap import *
from utils.buffer import ReplayBuffer

plots = [0, 500, 600, 800, 1000, 1500, 2000, 2500]
names = ["before_distillation"]

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name / "run{}".format(config.run)
    # for i in range(1000):
    for name in names:
        try:
            model_dir_folder = model_dir / "models"
            # model_file = str(model_dir_folder / "model{}.pt".format(i))
            model_file = str(model_dir_folder / "{}.pt".format(name))

            maddpg = MADDPG.init_from_save(model_file)
            print("Creating heatmap")
            heatmap(maddpg, title="Agent Policies Before Distillation", save=config.save)

            print("Distilling")
            with open(str(model_dir / "replay_buffer.pkl"), 'rb') as input:
                replay_buffer = pickle.load(input)
            maddpg.distill(256, 1024, replay_buffer, hard=True)

            print("Creating distilled heatmap")
            distilled_heatmap(maddpg, save=config.save)
        except:
            pass

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