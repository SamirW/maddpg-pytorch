import seaborn as sns
import pylab as plot
from utils.logging import *

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)

ENV = "simple_spread_hard"
BASE_DIR = "/home/samir/maddpg-pytorch/models/" + ENV + "/eval_graph_relative/"
FIGURE_TITLE = "3-Agent Distillation (Hard Environment)"
FIGURE_DIR = "figures/relative_obs/"
FIGURE_SAVE_NAME = ""

CONV_SIZE = 250
NUM_SEEDS = 11

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
        No Distill Eval
    """
    if False: 
        no_distill_eval = []
        no_distill_eval_ep = []

        for i in range(NUM_SEEDS):
            path = \
                BASE_DIR + \
                "env::{}_seed::{}_comment::no_distill_eval_log".format(ENV, i+1)
        
            try:
                data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
                data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

                no_distill_eval.append(data)
                no_distill_eval_ep.append(data_x)
            except:
                continue

        datas.append(no_distill_eval)
        datas_x.append(no_distill_eval_ep)
        legends.append(r'No Distillation and Eval')
    """
        Distill Eval
    """
    if False: 
        distill_eval = []
        distill_eval_ep = []

        for i in range(NUM_SEEDS):
            path = \
                BASE_DIR + \
                "env::{}_seed::{}_comment::distill_eval_log".format(ENV, i+1)
        
            try:
                data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
                data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

                distill_eval.append(data)
                distill_eval_ep.append(data_x)
            except:
                continue

        datas.append(distill_eval)
        datas_x.append(distill_eval_ep)
        legends.append(r'Distill and Eval')

    """
        no_distill
    """
    if True:
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
        Distill Pass Actor
    """
    if True:
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
        Pass critic
    """
    if True:
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
        Distilled
    """
    if True:
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
        Plot data
    """
    fig, ax = plt.subplots()
    sns.set_style("ticks")
    for i_data, data in enumerate(datas):
        x = datas_x[i_data][0]

        # x = x[:18000]
        # data = [d[:18000] for d in data]

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        error = (mean - std, mean + std)

        ax.fill_between(x, error[0], error[1], alpha=0.2)
        ax.plot(x, mean, label=legends[i_data])
        ax.margins(x=0)

    plt.xlabel(r'\textbf{Train Episode}', size=14)
    plt.ylabel(r'\textbf{Training Reward}', size=14)
    plt.title(FIGURE_TITLE, size=15)

    plt.legend()

    if SAVE:
        plt.savefig(FIGURE_DIR + FIGURE_SAVE_NAME, bbox_inches="tight", dpi=600) 

    if SHOW:
        plt.show()
