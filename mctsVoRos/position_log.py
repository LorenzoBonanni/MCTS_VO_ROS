"""
Parser and animation helper for the ground-truth position log produced by
Unity's PositionLogger (mcts_vo_Turtlebot3UnityROS2/Assets/PositionLogger.cs)
when a run is launched with --log-positions.

Usage - generate the GIFs for one run:
    1. Run the experiment with logging on, e.g.:
         python3 loopHandler_copy.py --exp_num 0 --algorithm VO-TREE \
             --trajectories sinusoidal --log-positions
    2. From the same mctsVoRos/ directory (same venv):
         from position_log import animate_run_positions, animate_run_positions_with_estimates
         animate_run_positions('sinusoidal', 'VO-TREE', 0)
         animate_run_positions_with_estimates('sinusoidal', 'VO-TREE', 0)
       (pass the --suffix string as a 4th argument if the run used one)
    They land in debug/<trajectories>/animations/positions_<algorithm>_<exp_num>.gif
    and .../positions_estimates_<algorithm>_<exp_num>.gif. See
    ~/Desktop/generate-position-gifs.txt for the full walkthrough including
    environment setup.

File format (little-endian throughout):
    header:  4 bytes magic b"MVPL" + 1 byte format version (currently 2)
    then a stream of tagged records:
      tag 0 "define object": ushort object_id, byte name_len, name_len bytes
                              of UTF-8 name
      tag 1 "data row":      int32 step, float32 time, ushort object_id,
                              float32 x, float32 z, float32 speed_instant,
                              float32 speed_max, float32 radius
                              (30 bytes after the tag)

radius is the object's true world-space radius (transform.localScale.x / 2
for a unit sphere primitive), used to draw real-size circles instead of dots
so collisions are visible. The robot's own "robot"-named rows carry radius 0
- ROBOT_RADIUS below (the planner's fixed constant) is used for the robot's
circle instead, matching what every existing debug animation already draws
the robot's collision circle as.

Unity writes raw world-frame X/Z (Unity's left-handed, Y-up axes), for both
the robot and every obstacle - NOT the ROS frame used by the robot's executed
trajectory (trj_<suffix>.pkl, columns [x, y, heading, v]). The two frames are
related by the same fixed axis remap Unity's own Unity2Ros(Vector3) uses
(Assets/Scripts/Extensions/RosMessageExtensions.cs): x_ros = z_unity,
y_ros = -x_unity. unity_xz_to_ros_xy() below applies it.

The process producing this file is always killed with SIGTERM/SIGKILL to its
whole process group (loopHandler_copy.py never calls process.wait()), so the
last in-flight record can be cut off mid-write. parse_position_log() detects
and silently drops an incomplete trailing record instead of raising.

PositionLogger.cs batches its flushes (every 50 writes, not every row): a
synchronous flush is a real disk I/O stall, and flushing every row measurably
slowed Unity's frame time enough to change experiment outcomes, since the
Python-side control loop is itself wall-clock-timed. This trades a slightly
larger truncation window on an abrupt kill (up to 49 trailing records instead
of 1) for not perturbing the run it's observing - parse_position_log()'s
truncation handling covers this the same way either way.
"""
import os
import pickle
import struct
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

MAGIC = b'MVPL'
FORMAT_VERSION = 2

_DEFINE_TAG = 0
_ROW_TAG = 1
_ROW_STRUCT = struct.Struct('<i f H f f f f f')  # step, time, id, x, z, speed_instant, speed_max, radius

# The robot's goal position, in ROS frame. Copied from the constant
# LoopHandler.__init__ sets at loopHandler_copy.py:353 (self.goal) - it is
# not persisted per-run anywhere on disk, since debug_plots_and_animations
# only needs it because it runs in-process right after the experiment, with
# live access to loopHandler.s0.goal. Same value for every run today (set
# unconditionally, not per-scene). Keep this in sync with loopHandler_copy.py
# if that constant ever changes.
GOAL = np.array([-2.783, -0.720])

# The planner's fixed robot collision radius, in metres. Copied from
# MCTS_VO/environment_creator.py:25 (EnvConfig(robot_radius=0.15, ...)),
# unconditional for every algorithm/scene - there is no --robot-radius CLI
# flag. Matches what MCTS_VO.experiment_utils.plot_robot already draws the
# robot's circle as in the existing trajectory_<suffix>.gif.
ROBOT_RADIUS = 0.15

# The six static obstacles, in ROS frame, with their fixed radius. Copied
# from LoopHandler.__init__ (loopHandler_copy.py:383-391, self.gt_obs_pos /
# self.gt_obs_rad) - read once off the scene transforms and never persisted
# per-run, same situation as GOAL above. Only the (x, y) columns matter here;
# the source array's trailing two columns are unrelated state-vector padding.
STATIC_OBSTACLE_POS = np.array([
    [-0.399,  0.420],
    [-1.542, -1.790],
    [-1.539,  0.360],
    [-2.640, -1.310],
    [-0.317, -1.820],
    [-3.020,  0.363],
])
STATIC_OBSTACLE_RAD = np.full(len(STATIC_OBSTACLE_POS), 0.100)

# Fixed arena bounds, matching the [-4, 2] x [-4, 2] convention used by
# MCTS_VO.experiment_utils.plot_frame2 for the existing trajectory_*.gif.
_AXIS_LIMITS = (-4, 2)


def parse_position_log(path):
    """
    Parse a positions_<suffix>.bin file into a DataFrame.

    Args:
        path (str): Path to the .bin file written by PositionLogger.

    Returns:
        pandas.DataFrame with columns
            step (int), time (float), object (str),
            x (float), z (float), speed_instant (float), speed_max (float),
            radius (float)
        in raw Unity world-frame coordinates, one row per logged sample,
        in the order they were written. Returns an empty DataFrame with
        those columns if the file is missing or has no complete rows.
    """
    columns = ['step', 'time', 'object', 'x', 'z', 'speed_instant', 'speed_max', 'radius']
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)

    with open(path, 'rb') as f:
        data = f.read()

    if len(data) < 5 or data[:4] != MAGIC:
        raise ValueError(f'{path}: missing or bad MVPL header')
    version = data[4]
    if version != FORMAT_VERSION:
        raise ValueError(f'{path}: unsupported format version {version}')

    names = {}
    rows = []
    offset = 5
    n = len(data)
    while offset < n:
        tag = data[offset]
        offset += 1
        if tag == _DEFINE_TAG:
            if offset + 3 > n:
                break  # truncated define header
            object_id, name_len = struct.unpack_from('<H B', data, offset)
            offset += 3
            if offset + name_len > n:
                break  # truncated name
            name = data[offset:offset + name_len].decode('utf-8')
            offset += name_len
            names[object_id] = name
        elif tag == _ROW_TAG:
            if offset + _ROW_STRUCT.size > n:
                break  # truncated trailing row - abrupt-kill artefact, expected
            step, time_, object_id, x, z, speed_instant, speed_max, radius = \
                _ROW_STRUCT.unpack_from(data, offset)
            offset += _ROW_STRUCT.size
            rows.append((step, time_, names.get(object_id, f'<unknown:{object_id}>'),
                         x, z, speed_instant, speed_max, radius))
        else:
            raise ValueError(f'{path}: unknown record tag {tag} at offset {offset - 1}')

    return pd.DataFrame(rows, columns=columns)


def unity_xz_to_ros_xy(x_unity, z_unity):
    """
    Convert raw Unity world (X, Z) to the ROS (x, y) frame used by
    trj_<suffix>.pkl, matching Assets/Scripts/Extensions/RosMessageExtensions.cs
    Unity2Ros(Vector3): x_ros = z_unity, y_ros = -x_unity.
    """
    return z_unity, -x_unity


def load_run_trajectories(scene, algorithm, exp_num, suffix_tag='', debug_dir='debug'):
    """
    Load the robot's executed trajectory and the ground-truth position log
    for one run.

    Args:
        scene (str): --trajectories value, e.g. 'sinusoidal'.
        algorithm (str): --algorithm value, e.g. 'MCTS'.
        exp_num (int): --exp_num value (the run's seed).
        suffix_tag (str): --suffix value used for the run, if any.
        debug_dir (str): Root debug directory, matching DEBUG_DIR in
                          loopHandler_copy.py.

    Returns:
        (robot_trajectory, positions) where robot_trajectory is the
        [x, y, heading, v] ndarray unpickled from trj_<suffix>.pkl (ROS
        frame), and positions is the DataFrame from parse_position_log (Unity
        frame).

    Raises:
        FileNotFoundError: if the run was not launched with --log-positions,
            so no position log exists for it - this is a real, expected
            boundary condition, not something to silently degrade past.
    """
    out_dir = os.path.join(debug_dir, scene)
    suffix = f'{algorithm}_{exp_num}{suffix_tag}'

    with open(os.path.join(out_dir, f'trj_{suffix}.pkl'), 'rb') as f:
        robot_trajectory = pickle.load(f)

    pos_log_path = os.path.join(out_dir, f'positions_{suffix}.bin')
    if not os.path.exists(pos_log_path):
        raise FileNotFoundError(
            f'{pos_log_path} not found - was this run launched with --log-positions?')
    positions = parse_position_log(pos_log_path)
    return robot_trajectory, positions


def load_obstacle_estimates(scene, algorithm, exp_num, suffix_tag='', debug_dir='debug'):
    """
    Load the planner's own per-step obstacle estimate, obs_<suffix>.pkl -
    saved unconditionally by save_data() (loopHandler_copy.py), a list of
    (obs_pos, obs_rad) tuples one entry per control step, indexed the same
    way as trj_<suffix>.pkl (both appended together, once per step, in
    LoopHandler.control_loop - see loopHandler_copy.py:997/1005).

    Returns:
        list[(obs_pos, obs_rad)]: obs_pos is an (N, 4) array (columns 0/1 are
        x, y in ROS frame; obs_rad is the matching (N,) radius array.
    """
    out_dir = os.path.join(debug_dir, scene)
    suffix = f'{algorithm}_{exp_num}{suffix_tag}'
    with open(os.path.join(out_dir, f'obs_{suffix}.pkl'), 'rb') as f:
        return pickle.load(f)


def _object_palette(object_names):
    cmap = plt.get_cmap('tab10')
    return {name: cmap(i % 10) for i, name in enumerate(sorted(object_names))}


def _hold_last_sample(positions_by_object, name, t):
    """Zero-order hold: the most recent sample of `name` at or before time t."""
    group = positions_by_object.get(name)
    if group is None or len(group) == 0:
        return None
    idx = group['time'].searchsorted(t, side='right') - 1
    if idx < 0:
        return None
    return group.iloc[idx]


def _frame_times_from_robot_gt(robot_trajectory, positions_by_object):
    """
    Real Unity-clock time for each robot_trajectory frame, found by matching
    each frame's (x, y) to the closest position in the position log's own
    "robot" ground-truth series - NOT by assuming any clock model.

    Why this is needed: trj_<suffix>.pkl (Python's executed trajectory) and
    the position log's "robot" rows (Unity's GroundTruthOdometry) describe
    the same physical robot, logged by two independently-clocked processes
    with no shared timestamp. A first version of this module assumed a flat
    `dt` seconds per trajectory frame, which is wrong in two ways: (1) rate
    - the control loop is itself wall-clock-timed and frequently holds
    position / repeats the last action when sensor data is stale (see
    loopHandler_copy.py's MAX_SENSOR_AGE gate and its "stale sensor data...
    holding position" warning), so real elapsed time per step is not
    constant; and (2) origin - Unity starts ticking as soon as the player
    launches, ~2s (loopHandler_copy.py's time.sleep(2)) plus ROS discovery
    time before Python's control loop starts its own clock at frame 0, an
    offset this module has no direct way to measure. Both errors shift
    where an obstacle is drawn relative to the robot at a given frame, and
    were the actual cause of an obstacle appearing to overlap the robot
    more than the (LIDAR-based, unaffected by any of this) real collision
    check implies.

    Since /odom currently carries no injected noise (loopHandler_copy.py's
    GroundTruthOdometry has a "TODO: add optional drift and noise" that is
    not implemented), the two robot traces coincide almost exactly in
    space once converted to the same frame. Matching by position instead of
    by time sidesteps needing any clock model at all: it directly asks
    "when, on Unity's own clock, was the robot at this position" and uses
    that Unity time to look up every other object's position log entry, so
    obstacles line up against the robot the way they actually did.

    Returns:
        np.ndarray of shape (len(robot_trajectory),): Unity-clock time for
        each frame, non-decreasing (a rare position-match ambiguity, e.g.
        while the robot is holding still, is clamped forward rather than
        left to jump backward).
    """
    robot_gt = positions_by_object.get('robot')
    if robot_gt is None or len(robot_gt) == 0:
        raise ValueError(
            "position log has no 'robot' entries - was GroundTruthOdometry logging?")

    gt_x, gt_y = unity_xz_to_ros_xy(robot_gt['x'].to_numpy(), robot_gt['z'].to_numpy())
    gt_xy = np.stack([gt_x, gt_y], axis=1)
    gt_time = robot_gt['time'].to_numpy()

    traj_xy = robot_trajectory[:, :2]
    dist = np.linalg.norm(gt_xy[None, :, :] - traj_xy[:, None, :], axis=2)
    nearest = np.argmin(dist, axis=1)
    return np.maximum.accumulate(gt_time[nearest])


def _draw_base_frame(ax, i, robot_trajectory, positions_by_object, times, palette):
    """
    Draws the goal, the robot (real-size circle + heading, from the executed
    trajectory) and every ground-truth obstacle (real-size circle, from the
    position log) plus the six fixed static obstacles. Shared by both
    animate_run_positions and animate_run_positions_with_estimates so the two
    GIFs are visually identical apart from the estimate overlay.
    """
    ax.clear()

    # GOAL POSITION
    ax.plot(GOAL[0], GOAL[1], "xb", label="Goal")

    # STATIC OBSTACLES (fixed every frame)
    for pos, rad in zip(STATIC_OBSTACLE_POS, STATIC_OBSTACLE_RAD):
        ax.add_patch(plt.Circle(pos, rad, color="dimgray"))

    # ROBOT: real-size circle + heading tick, from the executed trajectory
    # (ROS frame), matching MCTS_VO.experiment_utils.plot_robot.
    x, y, heading = robot_trajectory[i, 0], robot_trajectory[i, 1], robot_trajectory[i, 2]
    ax.add_patch(plt.Circle((x, y), ROBOT_RADIUS, color="b"))
    heading_tip = np.array([x, y]) + np.array([np.cos(heading), np.sin(heading)]) * ROBOT_RADIUS
    ax.plot([x, heading_tip[0]], [y, heading_tip[1]], "-k")
    sub_traj = robot_trajectory[:i]
    ax.plot(sub_traj[:, 0], sub_traj[:, 1], "--r", label="Robot (executed)")

    # GROUND-TRUTH OBSTACLES: real-size circles, held at the last known
    # sample <= this frame's time (obstacles log at their own, slower,
    # asynchronous rate). "robot"'s own ground-truth rows are skipped here -
    # already represented above via the executed trajectory + ROBOT_RADIUS.
    t = times[i]
    for name, group in positions_by_object.items():
        if name == 'robot':
            continue
        row = _hold_last_sample(positions_by_object, name, t)
        if row is None:
            continue
        x_ros, y_ros = unity_xz_to_ros_xy(row['x'], row['z'])
        ax.add_patch(plt.Circle((x_ros, y_ros), row['radius'], color=palette[name]))
        ax.plot([], [], 'o', color=palette[name], label=name)  # legend entry only

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(_AXIS_LIMITS)
    ax.set_ylim(_AXIS_LIMITS)


def _plot_position_frame(i, robot_trajectory, positions_by_object, times, ax, palette):
    _draw_base_frame(ax, i, robot_trajectory, positions_by_object, times, palette)
    ax.legend(loc='upper right', fontsize='small')


def _plot_position_frame_with_estimates(i, robot_trajectory, positions_by_object, times,
                                         obstacle_estimates, ax, palette):
    _draw_base_frame(ax, i, robot_trajectory, positions_by_object, times, palette)

    # ESTIMATED OBSTACLES (the planner's own per-step LIDAR-based estimate,
    # obs_<suffix>.pkl, indexed by the same step i as robot_trajectory).
    obs_pos, obs_rad = obstacle_estimates[i]
    for pos, rad in zip(obs_pos, obs_rad):
        ax.add_patch(plt.Circle((pos[0], pos[1]), rad, fill=False, color="k", linestyle='--'))
    ax.plot([], [], linestyle='--', color='k', marker='none', label='Obstacle estimate')

    ax.legend(loc='upper right', fontsize='small')


def _load_frame_data(scene, algorithm, exp_num, suffix_tag, debug_dir):
    robot_trajectory, positions = load_run_trajectories(
        scene, algorithm, exp_num, suffix_tag, debug_dir)

    palette = _object_palette(positions['object'].unique())
    positions_by_object = {
        name: group.sort_values('time').reset_index(drop=True)
        for name, group in positions.groupby('object')
    }
    times = _frame_times_from_robot_gt(robot_trajectory, positions_by_object)
    return robot_trajectory, positions_by_object, palette, times


def animate_run_positions(scene, algorithm, exp_num, suffix_tag='', debug_dir='debug'):
    """
    Build a GIF animating the robot's executed trajectory together with the
    ground-truth trajectory of every logged object (robot + every obstacle),
    matching the visual style of debug_utils.debug_plots_and_animations'
    trajectory_<suffix>.gif (goal marker, robot marker + trail, fixed
    [-4, 2] x [-4, 2] arena) - but with the robot and every obstacle drawn at
    their real size (so collisions are visible) and the robot's heading
    shown, instead of dots.

    Args:
        scene, algorithm, exp_num, suffix_tag, debug_dir: identify the run,
            same convention as load_run_trajectories.

    Returns:
        str: path to the written GIF, <out_dir>/animations/positions_<suffix>.gif
    """
    robot_trajectory, positions_by_object, palette, times = _load_frame_data(
        scene, algorithm, exp_num, suffix_tag, debug_dir)

    n_frames = len(robot_trajectory)

    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        _plot_position_frame,
        fargs=(robot_trajectory, positions_by_object, times, ax, palette),
        frames=tqdm(range(n_frames), file=sys.stdout),
        save_count=None,
        cache_frame_data=False,
    )

    out_dir = os.path.join(debug_dir, scene)
    anim_dir = os.path.join(out_dir, 'animations')
    os.makedirs(anim_dir, exist_ok=True)
    suffix = f'{algorithm}_{exp_num}{suffix_tag}'
    out_path = os.path.join(anim_dir, f'positions_{suffix}.gif')

    ani.save(out_path)
    plt.close(fig)
    return out_path


def animate_run_positions_with_estimates(scene, algorithm, exp_num, suffix_tag='',
                                          debug_dir='debug'):
    """
    Same as animate_run_positions, but additionally overlays the planner's
    own per-step obstacle estimate (obs_<suffix>.pkl, dashed black circles)
    on top of the ground truth, so estimation error is directly visible.

    Returns:
        str: path to the written GIF,
             <out_dir>/animations/positions_estimates_<suffix>.gif
    """
    robot_trajectory, positions_by_object, palette, times = _load_frame_data(
        scene, algorithm, exp_num, suffix_tag, debug_dir)
    obstacle_estimates = load_obstacle_estimates(scene, algorithm, exp_num, suffix_tag, debug_dir)

    n_frames = len(robot_trajectory)

    fig, ax = plt.subplots()
    ani = FuncAnimation(
        fig,
        _plot_position_frame_with_estimates,
        fargs=(robot_trajectory, positions_by_object, times, obstacle_estimates, ax, palette),
        frames=tqdm(range(n_frames), file=sys.stdout),
        save_count=None,
        cache_frame_data=False,
    )

    out_dir = os.path.join(debug_dir, scene)
    anim_dir = os.path.join(out_dir, 'animations')
    os.makedirs(anim_dir, exist_ok=True)
    suffix = f'{algorithm}_{exp_num}{suffix_tag}'
    out_path = os.path.join(anim_dir, f'positions_estimates_{suffix}.gif')

    ani.save(out_path)
    plt.close(fig)
    return out_path
