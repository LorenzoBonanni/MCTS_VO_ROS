import os
import random
import numpy as np
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
import matplotlib.pyplot as plt
from numba import jit
from numpy import float32

@jit(nopython=True, cache=True)
def set_seed(value):
    np.random.seed(value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    set_seed(seed_value)

seed_everything(0)

def group_matrix(M, I):
    unique_indices = np.unique(I)
    return {idx: M[I == idx] for idx in unique_indices}

RADIUS_SCALE = 2.2
dist = np.array([1.3188971, 1.3047571, 1.3043324, 1.317075 , 1.3770177, 1.3516467, 1.347925 , 1.357903 ], dtype=float32)
angles = np.array([0.48869219, 0.50614548, 0.52359877, 0.54105206, 6.14355892, 6.16101221, 6.1784655 , 6.1959188 ])
# X, Y, Theta, V
# state = np.array([0.22631, -0.9885721, 3.059037272847974+np.deg2rad(90), 0.0])
state = np.array([0.22591971, -0.98864764, 3.075198563626026, 0.0])
goal_x, goal_y = -2.783, -0.993
# 3.1379432773966265+np.deg2rad(90)

points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
points_copy = points
points_copy = np.empty_like(points)
points_copy[:, 0] = -points[:, 0]
points_copy[:, 1] = points[:, 1]
points_copy = state[:2] + points_copy
points_copy = np.hstack((points_copy, np.zeros(points.shape[0])[:, None]))

# hdbscan = HDBSCAN(n_jobs=-1)
hdbscan = DBSCAN(eps=0.1, min_samples=2, n_jobs=-1)
clusters = hdbscan.fit_predict(points_copy)
groups = group_matrix(points_copy, clusters)
obs_pos = np.empty((0, 4))
obs_rad = np.array([])


for key, group in groups.items():
    sph = pyrsc.Circle()
    center, _ , radius, _ = sph.fit(group)
    radius *= RADIUS_SCALE
    obs_pos = np.vstack((obs_pos, np.array([center[0], center[1], 0.0, 0.0])))
    obs_rad = np.append(obs_rad, radius)

    plt.scatter(group[:, 0], group[:, 1], label='Group Points {}'.format(key))
    circle = plt.Circle((center[0], center[1]), radius, color='r', fill=False, label='Fitted Circle')
    plt.gca().add_patch(circle)
    plt.scatter(center[0], center[1], color='r', label='Circle Center')
    arrow_length = 0.2
    arrow_dx = arrow_length * np.cos(state[2])
    arrow_dy = arrow_length * np.sin(state[2])
    plt.arrow(state[0], state[1], arrow_dx, arrow_dy, head_width=0.03, head_length=0.03, fc='k', ec='k', label='Heading')


print(obs_pos)
print(obs_rad)
plt.plot(goal_x, goal_y, 'gx', label='Goal Position')
plt.scatter(state[0], state[1], color='k', label='Robot Position')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Group Points and Fitted Circle')
plt.axis('equal')
plt.show()