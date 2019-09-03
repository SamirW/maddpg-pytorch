import seaborn as sns
import pylab as plot
from utils.logging import *

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)

ENV = "simple_spread_5"
RUN = "deepset"
BASE_DIR = "/home/samir/dev/acl/maddpg-pytorch-sharing/models/" + ENV + "/" + RUN + "/"
FIGURE_TITLE = "5-Agent Simple Spread (Relative Obs)"
FIGURE_DIR = "figures/deepset/"
FIGURE_SAVE_NAME = "five-agent-deepset.png"

CONV_SIZE = 100
NUM_SEEDS = 11

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
        no_distill
    """
    if False:
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
    if False:
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

                distill_eval.append(data[:10000])
                distill_eval_ep.append(data_x[:10000])
            except:
                continue

        datas.append(distill_eval)
        datas_x.append(distill_eval_ep)
        legends.append(r'Distill All')

    """
        Distill Pass Actor
    """
    if False:
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
    if False:
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
        Baseline
    """
    if True:
        distill_pass_critic = []
        distill_pass_critic_ep = []

        for i in range(NUM_SEEDS):
            path = \
                BASE_DIR + \
                "env::{}_seed::{}_comment::baseline_log".format(ENV, i+1)
        
            try:
                data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
                data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

                distill_pass_critic.append(data)
                distill_pass_critic_ep.append(data_x)
            except:
                continue

        datas.append(distill_pass_critic)
        datas_x.append(distill_pass_critic_ep)
        legends.append(r'MADDPG')
    """
        Deepset
    """
    if True:
        distill_pass_critic = []
        distill_pass_critic_ep = []

        for i in range(NUM_SEEDS):
            path = \
                BASE_DIR + \
                "env::{}_seed::{}_comment::deepset_log".format(ENV, i+1)
        
            try:
                data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), CONV_SIZE)
                data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), CONV_SIZE)

                distill_pass_critic.append(data)
                distill_pass_critic_ep.append(data_x)
            except:
                continue

        datas.append(distill_pass_critic)
        datas_x.append(distill_pass_critic_ep)
        legends.append(r'MADDPG w/ Deepsets')

    """
        Plot data
    """
    fig, ax = plt.subplots()
    sns.set_style("ticks")
    for i_data, data in enumerate(datas):
        x = datas_x[i_data][0]

        x = x[:30000]
        data = [d[:30000] for d in data]

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        error = (mean - 0.5*std, mean + 0.5*std)

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
