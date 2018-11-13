import argparse
import pickle
from pathlib import Path
from algorithms.maddpg import MADDPG
from utils.heatmap import *
from utils.buffer import ReplayBuffer

def run(model_dir):
    model_file = str(model_dir / "model.pt")
    init_maddpg = MADDPG.init_from_save(model_file)
    heatmap(init_maddpg)

    with open(str(model_dir / "replay_buffer.pkl"), 'rb') as input:
        replay_buffer = pickle.load(input)

    print(replay_buffer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("run", help="Run number")
    config = parser.parse_args()

    model_dir = Path('./models') / config.env_id / config.model_name / "run{}".format(config.run)
    run(model_dir)