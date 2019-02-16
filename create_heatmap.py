import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from algorithms.maddpg import MADDPG
from utils.heatmap import heatmap
from utils.heatmap3 import heatmap3
from utils.heatmap4 import heatmap4
from utils.buffer import ReplayBuffer

plots = [0, 200, 400, 600, 800, 1000]
names = ["before_distillation"]

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name / "run{}".format(config.run)
    # for i in plots:
    for name in names:
        try:
            # model_file = str(model_dir / "models" / "model{}.pt".format(i))
            # model_file = str(model_dir / "models" / "{}.pt".format(name))
            model_file = str(model_dir / "model.pt")

            maddpg = MADDPG.init_from_save(model_file)
            nagents = maddpg.nagents
            if nagents == 2:
                heatmap_fn = heatmap 
            elif nagents == 3:
                heatmap_fn = heatmap3
            else: # 4 agents
                heatmap_fn = heatmap4

            print("Creating heatmap")
            heatmap_fn(maddpg, title="{} Agent Policies Before Distillation".format(nagents), save=config.save)

            print("Distilling")
            with open(str(model_dir / "replay_buffer.pkl"), 'rb') as input:
                replay_buffer = pickle.load(input)
            maddpg.distill(256, 1024, replay_buffer, hard=True, pass_critic=False)

            print("Creating distilled heatmap")
            heatmap_fn(maddpg, title="{} Agent Policies After Distillation".format(nagents), save=config.save)
        except Exception as e:
            print(e)

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
