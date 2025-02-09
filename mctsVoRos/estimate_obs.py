from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from skimage.measure import CircleModel, ransac
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def group_matrix(M, I):
    unique_indices = np.unique(I)
    return {idx: M[I == idx] for idx in unique_indices}

fig, ax = plt.subplots()
RADIUS_SCALE = 4.

robot_state = np.array([0.07291288, -1.13601339,  3.1406439,   0.])
goal = np.array([-2.783, -0.993])
points = np.array([[-0.90405641, -1.13575706],
       [-0.91192374, -1.15294649],
       [-0.93592147, -1.17098019],
       [-2.23359116, -1.37784466],
       [-2.23833081, -1.41920392],
       [-1.54382752, -1.36281429],
       [-1.53034695, -1.38953011],
       [-1.53738441, -1.41953443],
       [-1.90091784, -1.81511233],
       [-1.87862213, -1.84576925],
       [-2.53161771,  1.21037543],
       [-1.50402516,  0.23556252],
       [-1.53077153,  0.21039124],
       [-2.01094137,  0.43516669],
       [-2.01093934,  0.37885436],
       [-2.01093712,  0.3239531 ],
       [-2.01093541,  0.27037879],
       [-2.08010481,  0.26299716],
       [-1.4299961 , -0.19631421],
       [-1.44298444, -0.22460121],
       [-1.48388304, -0.236626  ],
       [-1.46189185, -0.80935215],
       [-1.4618908 , -0.83725338],
       [-1.46188996, -0.86496558],
       [-1.46188891, -0.89250748],
       [-1.46188791, -0.91989723],
       [-1.4618871 , -0.94715261],
       [-1.46188607, -0.97429117],
       [-1.46188517, -1.00133005],
       [-2.96730464, -0.92263457],
       [-0.92340321, -1.08353338],
       [-0.90801586, -1.1014987 ],
       [-0.90292766, -1.11872275],
       [-0.90405641, -1.13575701]])

dist = np.linalg.norm(points - robot_state[0:2], axis=1)

points = points[dist < 3.0]

clusters = HDBSCAN(allow_single_cluster= True, alpha= 0.5, cluster_selection_epsilon= 0.01, min_cluster_size=2, min_samples=1, n_jobs=-1).fit_predict(points)
groups = group_matrix(points, clusters)
est_obs = np.empty((0, 2))
est_rad = np.array([])
colors = plt.cm.get_cmap('tab10', len(groups))

for idx, (group, color) in enumerate(zip(groups.values(), colors.colors)):
    for point in group:
        ax.plot(point[0], point[1], 'o', color=color, label=f'Cluster {idx}')

for group in groups.values():
    if len(group) < 3:
        continue
    ransac_model, _ = ransac(group, 
                             CircleModel, 
                             max_trials=100, 
                             min_samples=3, 
                             residual_threshold= 0.1, 
                             stop_probability= 0.99
                            )
    if ransac_model is None:
        continue
    center = ransac_model.params[0:2]
    radius = ransac_model.params[2]
    radius *= RADIUS_SCALE
    est_obs = np.vstack((est_obs, center))
    est_rad = np.append(est_rad, radius)

for o, rad in zip(est_obs, est_rad):
    # plot the obstacle
    ax.plot(o[0], o[1], 'bx')
    circle = plt.Circle((o[0], o[1]), rad, color='b', fill=False, label='Estimated Obstacle')
    ax.add_artist(circle)

# plot the robot
ax.plot(robot_state[0], robot_state[1], 'kx', label='Robot')
# plot the robot heading
heading_length = 0.2  # length of the heading line
heading_x = robot_state[0] + heading_length * np.cos(robot_state[2])
heading_y = robot_state[1] + heading_length * np.sin(robot_state[2])
ax.arrow(robot_state[0], robot_state[1], heading_length * np.cos(robot_state[2]), heading_length * np.sin(robot_state[2]), head_width=0.1, head_length=0.1, fc='k', ec='k')
# plot the goal
ax.plot(goal[0], goal[1], 'x', color='gold', label='Goal')

ax.set_aspect('equal', adjustable='box')
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax.legend(*zip(*unique))


# plt.show()
i = 0
print(est_obs)
print(est_rad)
plt.savefig(f'debug/estimated_obs_{i}.png')
ax.clear()