import seaborn as sns
import pylab as plot
from utils.logging import *

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)
BASE_DIR = "/home/samir/maddpg-pytorch/models/simple_spread_flip/eval_graph_random/"
figure_name = "figures/eval_graph_random_no_training.png"
num_seeds = 10
conv_size = 20

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

    for i in range(num_seeds):
        path = \
            BASE_DIR + \
            "env::simple_spread_flip_seed::{}_comment::no_distill_log".format(i+1)
        
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), conv_size)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), conv_size)

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

    for i in range(num_seeds):
        path = \
            BASE_DIR + \
            "env::simple_spread_flip_seed::{}_comment::distill_log".format(i+1)
    
        try:
            data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), conv_size)
            data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), conv_size)

            distill_data.append(data)
            distill_data_ep.append(data_x)
        except:
            continue

    datas.append(distill_data)
    datas_x.append(distill_data_ep)
    legends.append(r'Single Hard Distillation')

    """
        Distilled_256
    """
    # distill_256_data = []
    # distill_256_data_ep = []

    # for i in range(num_seeds):
    #     path = \
    #         BASE_DIR + \
    #         "env::simple_spread_flip_seed::{}_comment::distill_256_log".format(i+1)
    
    #     data = moving_average(read_key_from_log(path, key="Train episode reward", index=6), conv_size)
    #     data_x = moving_average(read_key_from_log(path, key="Train episode reward", index=-1), conv_size)

    #     distill_256_data.append(data)
    #     distill_256_data_ep.append(data_x)

    # datas.append(distill_256_data)
    # datas_x.append(distill_256_data_ep)
    # legends.append(r'Single Hard Distillation (256)')

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
    plt.title(r'\textbf{Single Hard Distillation Comparison (No Training after Flip)}', size=15)

    legend = plt.legend(
        bbox_to_anchor=(0., 1.07, 1., .102), 
        loc=3, 
        ncol=2, 
        mode="expand", 
        borderaxespad=0.)

    # plt.show()
    
    plt.savefig(figure_name, bbox_inches="tight", dpi=300) 
