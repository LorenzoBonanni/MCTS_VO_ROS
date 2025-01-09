import os
import random
import numpy as np
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from numba import jit

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
dist = np.array([1.3536634, 1.3352884, 1.3337014, 1.3462055, 1.3077849, 1.2888747, 1.2855871, 1.293689 , 1.3536634], dtype=np.float32)
angles = np.array([0.        , 0.01745329, 0.03490658, 0.05235988, 0.62831853, 0.64577182, 0.66322511, 0.6806784 , 6.28318526])
# X, Y, Theta, V
state = np.array([0.22631, -0.9885721, -3.14, 0.0])

points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
points_copy = np.empty_like(points)
points_copy[:, 0] = points[:, 1]
points_copy[:, 1] = -points[:, 0]
# points_copy = state[:2] + points_copy
points_copy = np.hstack((points_copy, np.zeros(points.shape[0])[:, None]))

hdbscan = DBSCAN(n_jobs=-1)
clusters = hdbscan.fit_predict(points_copy)
groups = group_matrix(points_copy, clusters)

obs_pos = np.empty((0, 4))
obs_rad = np.array([])


for group in groups.values():
    sph = pyrsc.Circle()
    center, _ , radius, _ = sph.fit(group)
    radius *= RADIUS_SCALE
    obs_pos = np.vstack((obs_pos, np.array([center[0], center[1], 0.0, 0.0])))
    obs_rad = np.append(obs_rad, radius)

    plt.scatter(group[:, 0], group[:, 1], label='Group Points')
    circle = plt.Circle((center[0], center[1]), radius, color='r', fill=False, label='Fitted Circle')
    plt.gca().add_patch(circle)
    plt.scatter(center[0], center[1], color='r', label='Circle Center')


print(obs_pos)
print(obs_rad)
    
plt.scatter(state[0], state[1], color='k', label='Robot Position')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Group Points and Fitted Circle')
plt.axis('equal')
plt.show()