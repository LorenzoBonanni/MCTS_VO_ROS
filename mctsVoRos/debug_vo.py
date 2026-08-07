"""Render the velocity-obstacle geometry of one finished run, step by step.

Run from `mctsVoRos/` (the imports below and the `debug/` paths both assume it):

    python debug_vo.py

Reads the artifacts written by `loopHandler_copy.py` into
`debug/<trajectories>/` and writes one PNG per planning step into `debug/pics/`.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # headless by construction; a local shell has no MPLBACKEND

from matplotlib import patches, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from MCTS_VO.bettergym.agents.utils.utils import get_robot_angles
from MCTS_VO.bettergym.agents.utils.vo import get_radii, get_unsafe_angles
from MCTS_VO.mcts_utils import get_intersections_vectorized

# The run to render. Artifacts live in debug/<trajectories>/, as built by
# `out_dir` in loopHandler_copy.py.
RUN_DIR = "debug/intention_complex"
SUFFIX = "VO-TREE_6"
OUT_DIR = "debug/pics"

# These mirror the configuration the experiments actually run with; drawing them
# from anywhere else silently draws the wrong picture.
#
# dt: the --ts default in loopHandler_copy.py. No docker script overrides it, so
# every run launched by docker/run_all_experiment.sh is at 0.1.
# ROBOT_RADIUS, VMAX: the EnvConfig overrides in MCTS_VO/environment_creator.py,
# not the stale dataclass defaults in bettergym/environments/env.py.
# THINK_MARGIN: loopHandler_copy.py sets it to dt when --plan-budget is unset,
# which is the case for the whole campaign.
dt = 0.1
ROBOT_RADIUS = 0.15
VMAX = 0.22
THINK_MARGIN = dt
MAX_ANGLE_CHANGE = 2.84 * dt
# get_radii's r0, constant across obstacles. Taken from the formula rather than
# from its return value, which is per obstacle and empty when nothing is detected.
R0 = VMAX * dt
# Factor the code applies to the VO radii before deciding an obstacle is too far
# to matter: `d > 1.6 * (r0 + r1)` in get_intersections_vectorized, and the same
# literal in compiled_utils.vo_forbidden_ranges.
VO_SCALE = 1.6

# The goal is a constant in loopHandler_copy.py and never written to disk.
GOAL = np.array([-2.783, -0.720])

# One colour per concept, used for every artist that expresses it:
#   blue   the robot - its VO ball, its position, its heading, the headings it can
#          reach in one step
#   red    the obstacles' VO balls, the tangents to them, and the headings they
#          forbid
#   green  the goal
C_ROBOT = 'tab:blue'
C_VO_FULL = 'tab:red'
C_GOAL = 'tab:green'

# Drawing length of the heading wedges. Arbitrary - only the angles mean anything.
WEDGE_R = 0.22

LEGEND_HANDLES = [
    Line2D([], [], color=C_ROBOT, lw=1.3,
           label=r'robot VO ball  $r_0 = v_{max}\Delta t$'),
    Line2D([], [], color=C_ROBOT, marker='x', ls='none', label='robot centre'),
    Patch(facecolor=C_ROBOT, alpha=0.35, label='headings reachable in one step'),
    Line2D([], [], color=C_VO_FULL, lw=1.3,
           label=f'obstacle VO ball  ${VO_SCALE}\\,r_1$'),
    Patch(facecolor=C_VO_FULL, alpha=0.45, label='headings forbidden by VO'),
    Line2D([], [], color=C_GOAL, marker='x', ls='none', label='goal'),
]

os.makedirs(OUT_DIR, exist_ok=True)

# trj_, not ps_. loopHandler_copy.py appends the trajectory and the obstacles
# before the termination check but the planning states after it, so ps_ is one
# entry short and the frame it misses is the terminal one - the collision, if the
# run ended in one. trj_[i] == ps_[i] for every i that ps_ has, so the pairing
# with obs_ is unchanged; this only adds the last frame.
trajectory = pickle.load(open(f"{RUN_DIR}/trj_{SUFFIX}.pkl", "rb"))
obs = pickle.load(open(f"{RUN_DIR}/obs_{SUFFIX}.pkl", "rb"))

for idx in range(len(trajectory)):
    fig, ax = plt.subplots()

    robot_state = trajectory[idx]  # [x, y, theta, v]
    # obs_rad already carries the RADIUS_SCALE inflation applied when the
    # obstacles were fitted, so it must not be scaled again here.
    obs_x, obs_rad = obs[idx]
    yaw = robot_state[2]

    obs_r, r = get_radii(
        circle_obs_x=obs_x,
        circle_obs_rad=obs_rad,
        dt=dt,
        robot_radius=ROBOT_RADIUS,
        vmax=VMAX,
        think_margin=THINK_MARGIN
    )
    robot_angles = np.array(get_robot_angles(robot_state, MAX_ANGLE_CHANGE))
    intersections, dist, mask = get_intersections_vectorized(
        x=robot_state,
        obs_x=obs_x,
        r0=r,
        r1=obs_r
    )
    forbidden_angles = get_unsafe_angles(
        intersection_points=intersections,
        robot_angles=robot_angles,
        x=robot_state
    )

    # Tangent lines from the robot to the VO circle, and their touch points. They
    # belong to the velocity obstacle, so they take the VO colour.
    for p in intersections:
        x1, y1, x2, y2 = p
        ax.plot([x1, x2], [y1, y2], '+', color=C_VO_FULL, ms=5)
        ax.plot([robot_state[0], x1], [robot_state[1], y1], color=C_VO_FULL, lw=0.6)
        ax.plot([robot_state[0], x2], [robot_state[1], y2], color=C_VO_FULL, lw=0.6)

    # One circle per obstacle: r1 straight out of get_radii, scaled by the same
    # factor the code applies before deciding an obstacle is out of range. r1
    # already contains the obstacle's own radius, how far it can travel in a whole
    # cycle, and the robot's radius - which is why the robot needs only r0.
    for o, r1 in zip(obs_x, obs_r):
        ax.add_artist(plt.Circle((o[0], o[1]), VO_SCALE * r1,
                                 color=C_VO_FULL, fill=False, lw=1.3))

    # Headings reachable in one step (robot colour), then the subset the velocity
    # obstacle forbids (VO colour). What is left is what the planner may pick.
    for angle_range in robot_angles:
        ax.add_patch(patches.Wedge(
            (robot_state[0], robot_state[1]), WEDGE_R,
            np.degrees(angle_range[0]), np.degrees(angle_range[1]),
            facecolor=C_ROBOT, alpha=0.35, edgecolor='none'))

    for angle_range in forbidden_angles:
        ax.add_patch(patches.Wedge(
            (robot_state[0], robot_state[1]), WEDGE_R,
            np.degrees(angle_range[0]), np.degrees(angle_range[1]),
            facecolor=C_VO_FULL, alpha=0.45, edgecolor='none'))

    ax.plot(GOAL[0], GOAL[1], 'x', color=C_GOAL, ms=9, mew=2)
    ax.plot(robot_state[0], robot_state[1], 'x', color=C_ROBOT, ms=7, mew=1.5)
    # One circle for the robot: r0, how far it travels in one step. Its overlap
    # with an obstacle's circle is the intersection to look for.
    ax.add_artist(plt.Circle((robot_state[0], robot_state[1]), R0,
                             color=C_ROBOT, fill=False, lw=1.3))
    ax.arrow(robot_state[0], robot_state[1],
             np.cos(yaw) * 0.3, np.sin(yaw) * 0.3,
             head_width=0.08, head_length=0.08, fc=C_ROBOT, ec=C_ROBOT, lw=1.0)

    ax.legend(handles=LEGEND_HANDLES, loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    ax.set_title(f"{SUFFIX}  step {idx}", fontsize=9)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-4, 2])
    ax.set_ylim([-4, 2])
    # Vector output: the circles sit within a couple of centimetres of each other
    # at this axis scale, which is a pixel or two in a raster, so the picture is
    # only readable if it can be zoomed.
    # bbox_inches='tight' so the legend, which sits outside the axes, is not
    # cropped away.
    plt.savefig(f"{OUT_DIR}/intersection_{idx}.pdf", bbox_inches='tight')
    plt.close(fig)
