import seaborn as sns
import pylab as plot
from utils.logging import *

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)

ENV = "simple_spread"
BASE_DIR = "/home/samir/maddpg-pytorch/models/" + ENV + "/eval_graph/"
FIGURE_NAME = "figures/reset_sharing_separate.png"

CONV_SIZE = 50
NUM_SEEDS = 10

SHOW = False
SAVE = True

def moving_average(data_set, periods=10):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


if __name__ == "__main__":
    datas = []
    datas_x = []
    legends = []

    """
        Baseline
    """
    no_distill_data = []
    no_distill_data_ep = []

    for i in range(NUM_SEEDS):
        path = \
            BASE_DIR + \
            "env::{}_seed::{}_comment::no_distill_log".format(ENV, i+1)
        
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

            no_distill_data.append(data)
            no_distill_data_ep.append(data_x)
        except:
            continue

    datas.append(no_distill_data)  
    datas_x.append(no_distill_data_ep)
    legends.append(r'No Distillation')

    """
        Distilled
    """
    distill_data = []
    distill_data_ep = []

    for i in range(NUM_SEEDS):
        path = \
            BASE_DIR + \
            "env::{}_seed::{}_comment::distill_log".format(ENV, i+1)
    
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

            distill_data.append(data)
            distill_data_ep.append(data_x)
        except:
            continue

    datas.append(distill_data)
    datas_x.append(distill_data_ep)
    legends.append(r'Naive Distillation')

    """
        Entropy Distillation
    """
    entropy_distill_data = []
    entropy_distill_data_ep = []

    for i in range(NUM_SEEDS):
        path = \
            BASE_DIR + \
            "env::{}_seed::{}_comment::entropy_distill_log".format(ENV, i+1)
    
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

            entropy_distill_data.append(data)
            entropy_distill_data_ep.append(data_x)
        except:
            continue

    datas.append(entropy_distill_data)
    datas_x.append(entropy_distill_data_ep)
    legends.append(r'Entropy-Weighted Distillation')

    """
        Separate ER Distillation
    """
    separate_replay = []
    separate_replay_ep = []

    for i in range(NUM_SEEDS):
        path = \
            BASE_DIR + \
            "env::{}_seed::{}_comment::separate_replay_distilled_log".format(ENV, i+1)
    
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

            separate_replay.append(data)
            separate_replay_ep.append(data_x)
        except:
            continue

    datas.append(separate_replay)
    datas_x.append(separate_replay_ep)
    legends.append(r'Separate ER Distillation')

    """
        Plot data
    """
    fig, ax = plt.subplots()
    sns.set_style("ticks")
    
    for i_data, data in enumerate(datas):
        x = datas_x[i_data][0]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        error = (mean - std, mean + std)

        ax.fill_between(x, error[0], error[1], alpha=0.2)
        ax.plot(x, mean, label=legends[i_data])
        ax.margins(x=0)

    plt.xlabel(r'\textbf{Train Episode}', size=14)
    plt.ylabel(r'\textbf{Training Reward}', size=14)
    plt.title(r'\textbf{Distillation Comparison (No Training after Reset)}', size=15)

    legend = plt.legend(
        bbox_to_anchor=(0., 1.07, 1., .102), 
        loc=3, 
        ncol=2, 
        mode="expand", 
        borderaxespad=0.)

    if SAVE:
        plt.savefig(FIGURE_NAME, bbox_inches="tight", dpi=300) 

    if SHOW:
        plt.show()
