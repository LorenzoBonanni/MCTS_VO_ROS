

from functools import partial
import random
import os
from matplotlib import pyplot as plt
import numpy as np
from numpy import array
from numba import jit

from MCTS_VO.bettergym.agents.planner_mcts import Mcts
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.agents.utils.vo import get_radii
from MCTS_VO.bettergym.compiled_utils import robot_dynamics
from MCTS_VO.environment_creator import create_pedestrian_env

def plot():
    fig, ax = plt.subplots()
    # radius = 
    r1, r0 = get_radii(obstacles_pos, obstacles_rad, dt, robot_radius, 0.3)
    for o, rad in zip(obstacles_pos, r1+r0):
        # plot the obstacle
        circle = plt.Circle((o[0], o[1]), rad, color='k', fill=False)
        ax.add_artist(circle)
        circle2 = plt.Circle((o[0], o[1]), 0.1, color='k',linestyle='dashed', fill=False)
        ax.add_artist(circle2)
    
    plt.plot(s0.goal[0], s0.goal[1], 'rx')
    plt.plot(robot_state[0], robot_state[1], 'bx')
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([-3.0, 1.0])
    plt.ylim([-3.0, 1.0])
    plt.show()

@jit(nopython=True, cache=True)
def set_seed(value):
    np.random.seed(value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    set_seed(seed_value)

dt = 0.3
robot_radius = 0.2
robot_state = array([-0.69678324, -0.97870925, -2.57,       -0.1])
obstacles_pos = array([
            [-1.127, -0.833, 0.0, 0.0],
            [-0.92, -1.651, 0.0, 0.0],
            # [-1.82, -0.306, 0.0, 0.0]
        ])
obstacles_rad = array([0.1, 0.1])
_, sim_env = create_pedestrian_env(
    discrete=True,
    rwrd_in_sim=True,
    out_boundaries_rwrd=-100,
    n_vel=5,
    n_angles=11,
    vo=True,
    obs_pos=None,
    n_obs=None,
    dt_real=dt,
)
planner = Mcts(
    num_sim=100,
    c=10,
    environment=sim_env,
    computational_budget=300,
    rollout_policy=partial(
        epsilon_uniform_uniform,
        std_angle_rollout=0.38,
        eps=0.2
    ),
    discount=0.7,
    logger=None
)
s0, _ = sim_env.reset()
s0.goal = array([0.22631, -0.9885721])
s0.x = robot_state
s0.obstacles = (obstacles_pos, obstacles_rad)

seed_everything(0)
action, _ = planner.plan(s0)
print(action)
plot()

s1 = robot_dynamics(
    robot_state,
    action,
    dt
)