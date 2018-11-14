import seaborn as sns
import pylab as plot
from utils import *

sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'legend.fontsize': 12}
plot.rcParams.update(params)
BASE_DIR = "/home/samirw/maddpg-pytorch/models/simple_spread_flip/"


def moving_average(data_set, periods=10):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


if __name__ == "__main__":
    """
        Process teach
    """
    path = \
        BASE_DIR + \
        "testing/run8/" + \
        "env::simple_spread_flip_seed::1_comment::_log"

    session_freq = read_key_from_log(path, key="session", index=-1)[0]
    ours = read_key_from_log(path, key="Evaluation Reward", index=5)
    ours_time = read_key_from_log(path, key="Evaluation Reward", index=-1)

    time_from = 5400 - 5 * session_freq
    time_to = 5400
    i_from = ours_time.index(time_from)
    i_to = ours_time.index(time_to)
    ours = ours[i_from:i_to]

    ours = np.split(np.asarray(ours), 5)
    n_ours = []
    for session in ours:
        n_ours.append(moving_average(session))
    n_ours = np.transpose(np.stack(n_ours, axis=1))
    ours_time = np.linspace(1, session_freq, num=n_ours.shape[1])

    """
        Process teach (difference reward)
    """
    path = \
        BASE_DIR + \
        "push_one_direction/ours_difference_reward/" + \
        "env::complex_push_seed::7_tau::0.01_n_eval::2_n_traj::1_start::2000_noise_std::0.1_batch_size::50_discount::0.99_teacher_discount::0.99_manager_done::True_teacher_reward_type::improvement_prefix::teach_moreIter_log"

    difference = read_key_from_log(path, key="Evaluation Reward", index=5)
    difference_time = read_key_from_log(path, key="Evaluation Reward", index=-1)
    
    i_from = difference_time.index(time_from)
    i_to = difference_time.index(time_to)
    difference = difference[i_from:i_to]

    difference = np.split(np.asarray(difference), 5)
    n_difference = []
    for session in difference:
        n_difference.append(moving_average(session))
    n_difference = np.transpose(np.stack(n_difference, axis=1))
    difference_time = np.linspace(1, session_freq, num=n_difference.shape[1])

    """
        Process HRL baseline
    """
    path = \
        BASE_DIR + \
        "push_one_direction/hrl_baseline/" + \
        "env::complex_push_seed::7_tau::0.01_n_eval::1_n_traj::1_start::2000_noise_std::0.1_batch_size::50_discount::0.99_teacher_discount::0.99_manager_done::True_teacher_reward_type::progress_prefix::BASELINE-FINAL-LOG_log"

    hrl = read_key_from_log(path, key="Evaluation Reward", index=5)
    hrl_time = read_key_from_log(path, key="Evaluation Reward", index=-1)

    i_from = hrl_time.index(time_from)
    i_to = hrl_time.index(time_to)
    hrl = hrl[i_from:i_to]

    hrl = np.split(np.asarray(hrl), 5)
    n_hrl = []
    for session in hrl:
        n_hrl.append(moving_average(session))
    n_hrl = np.transpose(np.stack(n_hrl, axis=1))
    hrl_time = np.linspace(1, session_freq, num=n_hrl.shape[1])

    """
        Process RL baseline
    """
    path = \
        BASE_DIR + \
        "push_one_direction/premitive_baseline/" + \
        "env::complex_push_seed::7_tau::0.01_n_eval::10_prefix::BASELINE-CENTRALIZED-CRITIC_log"

    session_freq = read_key_from_log(path, key="session", index=-1)[0]
    rl = read_key_from_log(path, key="Evaluation Reward", index=5)
    rl_time = read_key_from_log(path, key="Evaluation Reward", index=-1)

    i_from = rl_time.index(time_from)
    i_to = rl_time.index(time_to)
    rl = rl[i_from:i_to]

    rl = np.split(np.asarray(rl), 5)
    n_rl = []
    for session in rl:
        n_rl.append(moving_average(session))
    n_rl = np.transpose(np.stack(n_rl, axis=1))
    rl_time = np.linspace(1, session_freq, num=n_rl.shape[1])

    """
        Process lectr baseline
    """
    path = \
        BASE_DIR + \
        "push_one_direction/lectr_baseline/task_batch50_teacher_iter50/" + \
        "env::complex_push_seed::7_tau::0.01_n_eval::10_n_traj::10_start::0_noise_std::0.1_batch_size::50_discount::0.99_teacher_discount::0.99_prefix::teach-batch50_log"
    # path = \
    #     BASE_DIR + \
    #     "push_one_direction/lectr_baseline/task_batch50_teacher_iter50/" + \
    #     "env::complex_push_seed::7_tau::0.01_n_eval::10_n_traj::10_start::0_noise_std::0.1_batch_size::50_discount::0.99_teacher_discount::0.99_prefix::teach_log"

    session_freq = read_key_from_log(path, key="session", index=-1)[0]
    lectr = read_key_from_log(path, key="Evaluation Reward", index=5)
    lectr_time = read_key_from_log(path, key="Evaluation Reward", index=-1)

    i_from = lectr_time.index(time_from)
    i_to = lectr_time.index(time_to)
    lectr = lectr[i_from:i_to]

    lectr = np.split(np.asarray(lectr), 5)
    n_lectr = []
    for session in lectr:
        n_lectr.append(moving_average(session))
    n_lectr = np.transpose(np.stack(n_lectr, axis=1))
    lectr_time = np.linspace(1, session_freq, num=n_lectr.shape[1])

    """
        Oracle
    """
    path = \
        BASE_DIR + \
        "push_one_direction/oracle/" + \
        "env::complex_push_seed::7_tau::0.01_n_eval::10_n_traj::1_start::2000_noise_std::0.1_batch_size::50_discount::0.99_manager_done::True_prefix::BASELINE-HIERARCHCICAL_log"

    oracle = read_key_from_log(path, key="Evaluation Reward", index=5)
    oracle = moving_average(oracle, 20)

    oracle_time = ours_time
    oracle = np.zeros(len(oracle_time)) + np.max(oracle)

    datas = [
        n_ours,
        n_hrl,
        n_rl]
    datas_x = [
        ours_time,
        hrl_time,
        rl_time]

    legends = [
        r'HMAT',
        r'MATD3 (Hierarchical)',
        r'MATD3 (Primitive)']
    
    fig, ax = plt.subplots()
    sns.set_style("ticks")
    
    for i_data, data in enumerate(datas):
        x = datas_x[i_data]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        error = (mean - std, mean + std)
    
        ax.fill_between(x, error[0], error[1], alpha=0.2)
        ax.plot(x, mean, label=legends[i_data])
        ax.margins(x=0)

    plt.plot(oracle_time, oracle, "k--", label=r'Oracle')
    plt.ylim([-18.2,-9.8])
    plt.xlabel(r'\textbf{Train Episode}', size=14)
    plt.ylabel(r'\textbf{Average Evaluation Reward}', size=14)
    plt.title(r'\textbf{Comparisons in One-Box Push Domain}', size=15)

    legend = plt.legend(
        bbox_to_anchor=(0., 1.07, 1., .102), 
        loc=3, 
        ncol=2, 
        mode="expand", 
        borderaxespad=0.)
    
    plt.savefig("one_direction_transfer.png", bbox_inches="tight") 
