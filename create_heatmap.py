import argparse
from pathlib import Path
from utils.heatmap import heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("run", help="Run number")
    config = parser.parse_args()

    model_file = str(Path('./models') / config.env_id / config.model_name / "run{}".format(config.run) / "model.pt")
    heatmap(model_file)
