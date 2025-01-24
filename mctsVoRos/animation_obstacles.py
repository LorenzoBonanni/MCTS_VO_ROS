
import pickle
import sys

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from skimage.measure import CircleModel, ransac


RADIUS_SCALE = 3.

def group_matrix(M, I):
    unique_indices = np.unique(I)
    return {idx: M[I == idx] for idx in unique_indices}

def estimate_obstacles(points):
    clusters = DBSCAN(eps=0.1, min_samples=3, n_jobs=-1).fit_predict(points)
    groups = group_matrix(points, clusters)
    est_obs = np.empty((0, 2))
    est_rad = np.array([])
    for group in groups.values():
        if len(group) >= 4:
            ransac_model, _ = ransac(group, CircleModel, min_samples=4, residual_threshold=0.05, rng=0)
            center = ransac_model.params[0:2]
            radius = ransac_model.params[2]
            radius *= RADIUS_SCALE
            est_obs = np.vstack((est_obs, center))
            est_rad = np.append(est_rad, radius)
    
    return est_obs, est_rad

def plot_obstacles(i, points_list, ax):
    ax.clear()
    points = points_list[i]
    est_obs, est_rad = estimate_obstacles(points)
    
    for point in points:
        ax.plot(point[0], point[1], 'go', label='Point')
        
    for o, rad in zip(est_obs, est_rad):
        # plot the obstacle
        ax.plot(o[0], o[1], 'bx')
        circle = plt.Circle((o[0], o[1]), rad, color='b', fill=False, label='Estimated Obstacle')
        ax.add_artist(circle)
    
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([-3.0, 3.0])
    plt.ylim([-3.0, 3.0])
    
    

points_list = pickle.load(open("debug/points.pkl", 'rb'))

fig, ax = plt.subplots()
ani = FuncAnimation(
    fig,
    plot_obstacles,
    fargs=(points_list, ax),
    frames=tqdm(range(len(points_list)), file=sys.stdout),
    save_count=None,
    cache_frame_data=False,
    interval=1000
)
ani.save(f"debug/obs_ani.gif")