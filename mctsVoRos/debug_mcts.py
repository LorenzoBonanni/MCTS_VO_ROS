from functools import partial
import pickle

import numpy as np

from MCTS_VO.bettergym.agents.planner_mcts import Mcts
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.environment_creator import create_pedestrian_env
from loopHandler import seed_everything


trajectory = pickle.load(open("debug/ps_VO-TREE_0.pkl", "rb"))
obs = pickle.load(open("debug/obs_VO-TREE_0.pkl", "rb"))
acts = pickle.load(open("debug/acts_VO-TREE_0.pkl", "rb"))
times = pickle.load(open("debug/times_VO-TREE_0.pkl", "rb"))

DISCOUNT = 0.9
DEPTH = 200
dt = 0.3
idx = 66
robot_state = trajectory[idx]
obs_x, obs_rad = obs[idx]
print(acts[idx])

_, sim_env = create_pedestrian_env(
                discrete=True,
                rwrd_in_sim=True,
                out_boundaries_rwrd=-100,
                n_vel=4,
                n_angles=6,
                vo=True,
                obs_pos=None,
                n_obs=None,
                dt_real=dt,
            )
config = sim_env.config
planner = Mcts(
    num_sim=100,
    c=10,
    environment=sim_env,
    computational_budget=DEPTH,
    rollout_policy=partial(
        epsilon_uniform_uniform,
        std_angle_rollout=2.84*dt,
        eps=0.2
    ),
    discount=DISCOUNT,
    logger=None
)
s0, _ = sim_env.reset()
s0.goal = np.array([-2.783, -0.72])
s0.x = robot_state
s0.obstacles = (obs_x, obs_rad)
seed_everything(0)
a, _ = planner.plan(s0, times[idx])

print(a)