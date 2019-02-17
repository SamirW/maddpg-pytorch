import seaborn as sns
import pylab as plot
from utils.logging import *

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)

ENV = "simple_spread_flip_4"
BASE_DIR = "/home/samir/maddpg-pytorch/models/" + ENV + "/training_set_size_2/"
FIGURE_NAME = "figures/2_agent/flip_without_prior.png"

CONV_SIZE = 50
NUM_SEEDS = 3

SHOW = True
SAVE = False

def moving_average(data_set, periods=10):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


if __name__ == "__main__":
    datas = []
    datas_x = []
    legends = []

    """
        no_distill
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
    legends.append(r'Distill All')

    """
        Distill Pass Actor
    """
    distill_pass_actor = []
    distill_pass_actor_ep = []

    for i in range(NUM_SEEDS):
        path = \
            BASE_DIR + \
            "env::{}_seed::{}_comment::distill_pass_actor_log".format(ENV, i+1)
    
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

            distill_pass_actor.append(data)
            distill_pass_actor_ep.append(data_x)
        except:
            continue

    datas.append(distill_pass_actor)
    datas_x.append(distill_pass_actor_ep)
    legends.append(r'Critic-Only Distill')

    """
        Separate ER Distillation
    """
    distill_pass_critic = []
    distill_pass_critic_ep = []

    for i in range(NUM_SEEDS):
        path = \
            BASE_DIR + \
            "env::{}_seed::{}_comment::distill_pass_critic_log".format(ENV, i+1)
    
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

            distill_pass_critic.append(data)
            distill_pass_critic_ep.append(data_x)
        except:
            continue

    datas.append(distill_pass_critic)
    datas_x.append(distill_pass_critic_ep)
    legends.append(r'Actor-Only Distill')

    """
        Plot data
    """
    fig, ax = plt.subplots()
    sns.set_style("ticks")

    for i_data, data in enumerate(datas):
        x = datas_x[i_data][0]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        error = (mean - 0.5*std, mean + 0.5*std)

        ax.fill_between(x, error[0], error[1], alpha=0.2)
        ax.plot(x, mean, label=legends[i_data])
        ax.margins(x=0)

    plt.xlabel(r'\textbf{Train Episode}', size=14)
    plt.ylabel(r'\textbf{Training Reward}', size=14)
    plt.title(r'\textbf{4-Agent Distillation (Training Set Size 2)}', size=15)

    plt.legend()

    if SAVE:
        plt.savefig(FIGURE_NAME, bbox_inches="tight", dpi=600) 

    if SHOW:
        plt.show()
