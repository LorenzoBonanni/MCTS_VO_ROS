import pickle

from matplotlib import patches, pyplot as plt
import numpy as np

from MCTS_VO.bettergym.agents.utils.utils import get_robot_angles
from MCTS_VO.bettergym.agents.utils.vo import get_radii, get_unsafe_angles
from MCTS_VO.mcts_utils import get_intersections_vectorized


trajectory = pickle.load(open("debug/ps_VO-TREE_0.pkl", "rb"))
obs = pickle.load(open("debug/obs_VO-TREE_0.pkl", "rb"))
acts = pickle.load(open("debug/acts_VO-TREE_0.pkl", "rb"))
for idx in range(len(trajectory)):
    dt = 0.3
    fig, ax = plt.subplots()

    robot_state = trajectory[idx]  # [x, y, theta, v]
    obs_x, obs_rad = obs[idx]
    goal = np.array([-2.783, -0.72])
    yaw = robot_state[2]

    obs_r, r = get_radii(
        circle_obs_x=obs_x,
        circle_obs_rad=obs_rad,
        dt=dt,
        robot_radius=0.105,
        vmax=0.3
    )
    robot_angles = np.array(get_robot_angles(robot_state, 2.84 * dt))
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

    for p in intersections:
        x1, y1, x2, y2 = p
        # plot a cross on the intersection point
        plt.plot(x1, y1, 'k+')
        plt.plot(x2, y2, 'k+')
        # plot the tangent line
        plt.plot([robot_state[0], x1], [robot_state[1], y1], 'k')
        plt.plot([robot_state[0], x2], [robot_state[1], y2], 'k')

    for o, radius, aug_rad in zip(obs_x, obs_rad, obs_r+r):
        # plot the obstacle
        circle = plt.Circle((o[0], o[1]), radius, color='k', fill=False)
        ax.add_artist(circle)
        circle2 = plt.Circle((o[0], o[1]), aug_rad, color='b', fill=False)
        ax.add_artist(circle2)
        circle3 = plt.Circle((o[0], o[1]), 1.6*aug_rad, color='r', fill=False)
        ax.add_artist(circle3)
    
    for angle_range in robot_angles:
        theta1 = np.degrees(angle_range[0])
        theta2 = np.degrees(angle_range[1])
        wedge = patches.Wedge((robot_state[0], robot_state[1]), 0.22, theta1, theta2, facecolor='green', alpha=0.3)
        ax.add_patch(wedge)

    for angle_range in forbidden_angles:
        theta1 = np.degrees(angle_range[0])
        theta2 = np.degrees(angle_range[1])
        wedge = patches.Wedge((robot_state[0], robot_state[1]), 0.22, theta1, theta2, facecolor='red', alpha=0.3)
        ax.add_patch(wedge)


    plt.plot(goal[0], goal[1], 'rx')
    plt.plot(robot_state[0], robot_state[1], 'bx')
    circle = plt.Circle((robot_state[0], robot_state[1]), 0.105, color='k', fill=False)
    ax.add_artist(circle)
    # out_x, out_y = (
    #         np.array([robot_state[0], robot_state[1]]) + np.array([np.cos(yaw), np.sin(yaw)]) * 0.3
    # )
    plt.arrow(robot_state[0], robot_state[1], np.cos(yaw) * 0.3, np.sin(yaw) * 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-4, 2])
    ax.set_ylim([-4, 2])
    plt.savefig(f"debug/pics/intersection_{idx}.png")
    plt.close(fig)
