import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from MCTS_VO.experiment_utils import plot_frame2, plot_frame_tree_traj


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


def create_tree_animation(goal, config, obs, values, trajectories, out_path):
    """
    Create the animation of the rollout trajectories explored by the tree.

    This is a local copy of MCTS_VO.experiment_utils.create_animation_tree_trajectory
    with the output path as a parameter. The upstream helper hardcodes
    `./debug/rollout_{exp_name}.mp4`: since the prefix ends in the middle of the
    file name, the destination directory cannot be injected through `exp_name`
    (passing "sinusoidal/animations/VO-TREE_0" would ask matplotlib to write into
    a directory literally called "rollout_sinusoidal").

    Args:
        goal: Goal position of the robot.
        config: Environment configuration (plot limits, robot radius, ...).
        obs: Obstacles at each step.
        values: Rollout values of each step.
        trajectories: Rollout trajectories of each step.
        out_path (str): Full path of the .mp4 file to write.
    """
    fig, _ = plt.subplots()
    ani = FuncAnimation(
        fig,
        plot_frame_tree_traj,
        fargs=(goal, config, obs, trajectories, values, fig),
        frames=len(trajectories),
        save_count=None,
        cache_frame_data=False,
    )
    ani.save(out_path, dpi=300)
    plt.close(fig)


def debug_plots_and_animations(loopHandler, exp_num, algorithm, out_dir='debug'):
    """
    Create the debug animations of a run.

    Args:
        loopHandler: The LoopHandler holding the data of the finished run.
        exp_num (int): Experiment number.
        algorithm (str): Name of the algorithm used for the run.
        out_dir (str): Output directory of the run (e.g. debug/sinusoidal).
                       Animations are written to <out_dir>/animations.
    """
    print("Creating Gif...")
    suffix = f'{algorithm}_{exp_num}'

    # All the animations of the run live in <out_dir>/animations
    anim_dir = os.path.join(out_dir, 'animations')
    os.makedirs(anim_dir, exist_ok=True)

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
    ani.save(os.path.join(anim_dir, f"trajectory_{suffix}.gif"))
    plt.close(fig)
    
    if algorithm != 'VO-PLANNER':
        print("Creating animation")
        infos = loopHandler.infos
        trajectories = [i["trajectories"] for i in infos]
        rollout_values = [i["rollout_values"] for i in infos]

        create_tree_animation(
            goal,
            loopHandler.config,
            loopHandler.obstacles,
            rollout_values,
            trajectories,
            os.path.join(anim_dir, f"rollout_{suffix}.mp4"),
        )