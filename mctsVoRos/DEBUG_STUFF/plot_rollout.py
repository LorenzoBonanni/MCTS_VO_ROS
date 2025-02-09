
from functools import partial
from matplotlib import pyplot as plt
import numpy as np

from MCTS_VO.bettergym.agents.planner_mcts import RolloutStateNode
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.environments.env import State
from MCTS_VO.environment_creator import create_pedestrian_env
from loopHandler import seed_everything

seed_everything(0)
class RolloutPlanner:
    def __init__(self, rollout_policy, environment):
        self.rollout_policy = rollout_policy
        self.environment = environment

    def plan(self, state):
        return self.rollout_policy(RolloutStateNode(state), self), None
    
goal = np.array([-2.783, -0.993])
robot_state = np.array([0.07297561, -1.13600409,  3.14133167,  0.        ])
dt = 1
MAX_LEN = 50
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
s0, _ = sim_env.reset()
s0.x = robot_state
s0.goal = goal
s0.obstacles = (np.array([]), np.array([]))

config = sim_env.config



eps = 0.2
planner = RolloutPlanner(
    partial(
        epsilon_uniform_uniform,
        std_angle_rollout=1,
        eps=eps
    ),
    sim_env
)
step = []
for _ in range(10):
    trajectory = []
    s = s0
    for i in range(MAX_LEN):
        act, _ = planner.plan(s)
        s, r, terminal, _, _= sim_env.step(s, act)
        trajectory.append(s.x)
    trajectory = np.array(trajectory)
    step.append(trajectory)
    
fig, ax = plt.subplots()

last_points = np.array([trj[-1][:2] for trj in step])
x0 = robot_state
ax.set_xlim([-4, 2])
ax.set_ylim([-4, 2])
ax.grid(True)


# OBSTACLES
# obs_x, obs_rad = obs[i]
# for idx in range(len(obs_x)):
#     ob_x = obs_x[idx]
#     ob_rad = obs_rad[idx]
#     circle = plt.Circle((ob_x[0], ob_x[1]), ob_rad, color="k")
#     ax.add_artist(circle)

for trj in step:
    # last_points_trj = trj[:-1][:, :2]
    ax.plot(trj[:, 0], trj[:, 1], "r--", alpha=0.5)
cmap = ax.scatter(last_points[:, 0], last_points[:, 1], marker="x")
# ROBOT POSITION
ax.plot(x0[0], x0[1], "xk")
# GOAL POSITION
ax.plot(goal[0], goal[1], "xb")
# plt.show()
plt.savefig(f'debug/rollout_{eps}.png')
plt.close(fig)