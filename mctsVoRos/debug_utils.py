import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from MCTS_VO.experiment_utils import create_animation_tree_trajectory, plot_frame2


WINDOW_SIZE = 5


def plot_distribution(times, mean=None, std=None):
    import matplotlib.pyplot as plt
    plt.hist(times, color='b', bins=100, alpha=0.5 )
    if mean is not None:
        plt.axvline(mean, color='r', linestyle='solid', linewidth=2, label='mean')
    if std is not None:
        plt.axvline(mean + std, color='r', linestyle='dashed', linewidth=2, label='std')
        plt.axvline(mean - std, color='r', linestyle='dashed', linewidth=2)
    
    plt.legend()    
    plt.savefig('debug/distribution.png')

def plot_times_rolling_mean(times):
    import matplotlib.pyplot as plt
    plt.cla()
    plt.clf()
    plt.plot(times)
    plt.plot(np.convolve(times, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid'))
    plt.ylim([0, 0.3])
    plt.savefig('times_rolling_mean.png')

def debug_plots_and_animations(loopHandler, exp_num, algorithm):
    # print("Average time: ", np.mean(loopHandler.times))
    # print("Std time: ", np.std(loopHandler.times))
    # plot_times_distribution(loopHandler.times, np.mean(loopHandler.times), np.std(loopHandler.times))
    # plot_times_rolling_mean(loopHandler.times)
    print("Creating Gif...")
    suffix = f'{algorithm}_{exp_num}'

    goal = loopHandler.s0.goal
    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        plot_frame2,
        fargs=(goal, loopHandler.config, loopHandler.obstacles, loopHandler.trajectory, ax, (loopHandler.gt_obs_pos, loopHandler.gt_obs_rad), loopHandler.points_list),
        frames=tqdm(range(len(loopHandler.trajectory)), file=sys.stdout),
        save_count=None,
        cache_frame_data=False,
        # interval=1000
    )
    ani.save(f"debug/trajectory_{suffix}.gif",)
    plt.close(fig)
    
    if algorithm != 'VO-PLANNER':
        print("Creating animation")
        infos = loopHandler.infos
        trajectories = [i["trajectories"] for i in infos]
        rollout_values = [i["rollout_values"] for i in infos]

        create_animation_tree_trajectory(
            goal, 
            loopHandler.config, 
            loopHandler.obstacles, 
            exp_num, 
            suffix, 
            rollout_values, 
            trajectories
        )