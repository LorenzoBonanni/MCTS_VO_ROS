from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.measure import CircleModel, ransac


def group_matrix(M, I):
    unique_indices = np.unique(I)
    return {idx: M[I == idx] for idx in unique_indices}

def get_points_from_lidar(dist, angles, robot_pos, heading):
    points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
    points_copy = np.empty_like(points)
    points_copy[:, 0] = points[:, 0]
    points_copy[:, 1] = points[:, 1]
    # R = np.array([[np.cos(heading), -np.sin(heading)], 
    #               [np.sin(heading), np.cos(heading)]])
    # points_copy = points_copy @ R.T
    points_copy = robot_pos + points_copy

    return points_copy

fig, ax = plt.subplots()
RADIUS_SCALE = 3.
robot_pos = np.array([-0.05297427, -1.18251967])
# X python = Unity Z
# Z python = Unity Y
# Y python = Unity -X
gt_obs_pos = np.array([
            [-1.127, -0.833, 0.0, 0.0],
            [-0.92, -1.651, 0.0, 0.0],
            # [-1.82, -0.306, 0.0, 0.0]
        ])
gt_obs_rad = np.array([0.105, 0.105])

est_obs = np.array([[-0.95244857, -1.65063049,  0.        ,  0.        ],
                    [-1.15946449, -0.83263667,  0.        ,  0.        ]])

est_rad = np.array([0.04865198, 0.04648517]) * 3.0

points = None

heading = -2.9030833378332828
dist = np.array([1.0373137, 1.0249155, 1.0217499, 1.0256535, 1.0394408, 1.1853805, 1.1655198, 1.1603134, 1.1641103, 1.1805027])
angles = np.array([0.03490658, 0.05235988, 0.06981317, 0.08726646, 0.10471975, 5.56760027, 5.58505356, 5.60250686, 5.61996015, 5.63741344])
angles += heading
angles = (angles + np.pi) % (2 * np.pi) - np.pi
points = get_points_from_lidar(dist, angles, robot_pos, heading)
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


# Plot the robot position
ax.plot(robot_pos[0], robot_pos[1], 'gx', label='Robot Position')
# Plot the robot heading
heading_arrow_length = 0.2
heading_arrow = np.array([np.cos(heading), np.sin(heading)]) * heading_arrow_length
ax.arrow(robot_pos[0], robot_pos[1], heading_arrow[0], heading_arrow[1], head_width=0.1, head_length=0.1, fc='g', ec='g', label='Heading')

for o, rad in zip(gt_obs_pos, gt_obs_rad):
    # plot the obstacle
    ax.plot(o[0], o[1], 'kx')
    circle = plt.Circle((o[0], o[1]), rad, color='k', fill=False, linestyle='dashed', label='Ground Truth Obstacle')
    ax.add_artist(circle)

for o, rad in zip(est_obs, est_rad):
    # plot the obstacle
    ax.plot(o[0], o[1], 'bx')
    circle = plt.Circle((o[0], o[1]), rad, color='b', fill=False, label='Estimated Obstacle')
    ax.add_artist(circle)

for point in points:
    ax.plot(point[0], point[1], 'ro', label='Point')


ax.set_aspect('equal', adjustable='box')
plt.xlim([-3.0, 3.0])
plt.ylim([-3.0, 3.0])
handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax.legend(*zip(*unique))


plt.show()