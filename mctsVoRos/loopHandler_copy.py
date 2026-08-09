import gc
import math
import os
import pickle
import random
import signal
import subprocess
import argparse
import pandas as pd
import rclpy
import rclpy.qos
import numpy as np
import time
import tf_transformations

from sklearn.cluster import HDBSCAN
from debug_utils import debug_plots_and_animations
from MCTS_VO.bettergym.agents.planner_mcts import Mcts, RolloutStateNode
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.compiled_utils import dist_to_goal, get_points_from_lidar
from MCTS_VO.environment_creator import create_pedestrian_env
from geometry_msgs.msg import Twist
from rclpy.node import Node
from numba import jit
from copy import deepcopy
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from skimage.measure import CircleModel, ransac
from functools import partial
from MCTS_VO.bettergym.agents.utils.vo import epsilon_uniform_uniform_vo, set_legacy_vo
from rclpy.time import Time


parser = argparse.ArgumentParser()
parser.add_argument('--exp_num', default=0, type=int)
parser.add_argument('--algorithm', default='VO-PLANNER', type=str,
                    choices=['MCTS', 'VO-TREE', 'VO-PLANNER'])
parser.add_argument('--trajectories', default='sinusoidal', type=str,
                    choices=['sinusoidal', 'intention', 'sinusoidal_complex', 'intention_complex', 'sinusoidal_complex_speed', 'intention_complex_speed'],
                    help='Type of obstacle trajectories, i.e. which Unity '
                         'environment to launch. Also determines the output '
                         'directory (debug/<trajectories>).')
parser.add_argument('--max-obs-vel', default=0.1, type=float,
                    help='Maximum obstacle speed the velocity obstacles are '
                         'sized for. This MUST be >= the fastest obstacle in '
                         'the scene or the safety guarantee does not hold: '
                         'move_1.cs and move_2.cs (Obstacle_7/8_MOVING in the '
                         'sinusoidal scene) draw Random.Range(0.10, 0.15), so '
                         'the true maximum is 0.15 m/s while the code assumed '
                         '0.10.')
parser.add_argument('--exploration-c', default=1.0, type=float,
                    help='UCB exploration constant. The paper uses 10, but the '
                         'Q-values of the root actions span only ~0.06, while '
                         'the UCB bonus at c=10 and ~20 visits is ~5.3 - the '
                         'bonus swamps the signal and action selection becomes '
                         'close to random. Values around 1 or below make it '
                         'discriminate.')
parser.add_argument('--gamma-per-second', default=0.81, type=float,
                    help='Discount per SECOND. The paper discounts by 0.9 per '
                         'step at ts=0.1, i.e. 0.9**10 = 0.349 per second, an '
                         'effective horizon of about 1 s - shorter than the '
                         'time needed to reach the goal, so the first action '
                         'barely changes the return. 0.81 is a ~5 s horizon. '
                         'Pass 0.34867844 to reproduce the paper exactly.')
parser.add_argument('--radius-scale', default=1.8, type=float,
                    help='Factor applied to every RANSAC-fitted obstacle '
                         'radius before it reaches the velocity obstacles, '
                         'compensating for a front-facing arc under-estimating '
                         'the true radius. Too large and VO prunes away every '
                         'forward heading, leaving the robot stopped.')
parser.add_argument('--vo-geometry', default='paper', type=str,
                    choices=['paper', 'legacy'],
                    help='Which velocity-obstacle geometry to plan with. '
                         '"paper" follows Algorithm 4: tangents to the ball of '
                         'radius r1, obstacles beyond r0 + r1 ignored, trapped '
                         'below r1. "legacy" restores the geometry used up to '
                         'and including the 180-run campaign, which took '
                         'tangents to r0 + r1 and ignored beyond 1.6*(r0 + r1); '
                         'pass it to A/B the correction on identical scenes.')
parser.add_argument('--max-obs-radius', default=0.5, type=float,
                    help='Largest RANSAC-fitted obstacle radius accepted, in '
                         'metres. Applied to the *measured* radius, before '
                         '--radius-scale inflates it, so that the two knobs are '
                         'independent: raising the scale widens the safety '
                         'margin without also narrowing what counts as an '
                         'obstacle. Fits above this are usually walls.')
parser.add_argument('--rollout-collision', default='check', type=str,
                    choices=['check', 'none'],
                    help="Whether a rollout terminates on collision. 'check' "
                         'reproduces step_check_coll (an obstacle centre within '
                         "robot_radius ends the rollout with -100); 'none' "
                         'reproduces step_no_check_coll, leaving collision '
                         'avoidance entirely to VO pruning in the tree.')
parser.add_argument('--collect-trajectories', action='store_true',
                    help='Record every simulated state so that the rollout '
                         'tree animation can be produced. Costs about a tenth '
                         'of the planning budget and forces the uncompiled '
                         'rollout, so it is off by default.')
parser.add_argument('--no-plots', action='store_true',
                    help='Skip the debug plots and animations at the end of a '
                         'run. Rendering the trajectory GIF takes far longer '
                         'than the run itself, so sweeps want this.')
parser.add_argument('--ts', default=0.1, type=float,
                    help='Control and simulation time step in seconds. The '
                         'robot is commanded for ts after each planning step, '
                         'and MCTS simulates with the same step. Default 0.1 is '
                         'the configuration of the paper.')
parser.add_argument('--plan-budget', default=None, type=float,
                    help='Wall-clock seconds given to the planner each step. '
                         'The planner always spends its whole budget, so this - '
                         'not how fast the planner is - is what sets the cycle '
                         'time; a faster planner buys more simulations per '
                         'budget. Default keeps the paper behaviour: whatever '
                         'is left of one ts after sensing, with the cycle '
                         'running at 2*ts.')
parser.add_argument('--env-build', default=None, type=str,
                    help='Path to the Unity build to launch, overriding the '
                         'default for --trajectories. Use it to select a build '
                         'with different sensor publish rates, e.g. '
                         '../env_build/sin_env_50hz/env.x86_64')
parser.add_argument('--suffix', default='', type=str,
                    help='Extra tag appended to every output filename, so that '
                         'sweeps do not overwrite each other.')
parser.add_argument('--env-render', default='headless', type=str,
                    choices=['headless', 'low', 'full'],
                    help='How the Unity environment renders, which is what '
                         'actually sets the sensor publish rate: LaserScanner2D '
                         'triggers from a coroutine and scans in Update(), both '
                         'of which run once per frame, so ScanningFrequency is '
                         'silently clamped to the frame rate and rounded down to '
                         'a multiple of it. Measured on a scene configured for '
                         "50 Hz: 'full' (Ultra + vSync, the old behaviour) gives "
                         "16.8 Hz, 'low' gives 39.4 Hz and is still watchable, "
                         "'headless' gives 49.7 Hz. Since control_loop skips a "
                         'tick whenever the sensor data is older than one cycle, '
                         'that rate is the hard floor on the cycle time: '
                         'cycle >= 2/rate. NOTE the scenes currently in '
                         'env_build/ are NOT configured for 50 Hz: '
                         'turtlebot3_burger.prefab sets ScanningFrequency and '
                         'OdometryPublishingFrequency to 10, and both topics were '
                         'measured at 10.0 Hz under headless.')
parser.add_argument('--max-sensor-age', default=None, type=float,
                    help='Reject sensor data older than this many seconds and '
                         'hold position instead of planning on it. Defaults to '
                         'one control cycle, ts + think_margin, which is the '
                         'horizon the velocity obstacles are sized for: data '
                         'older than that breaks the assumption the safety '
                         'argument rests on. Measured fresh data is 27-94 ms '
                         'old, so the default leaves a factor of two.')
parser.add_argument('--diag', action='store_true',
                    help='Record where each cycle got its sensor data from: the '
                         'sequence number of the synchronised pair, the age of '
                         'each message, and raw per-topic arrival counters read '
                         'independently of the message filter. Tells apart "the '
                         'simulator stopped publishing" from "the synchroniser '
                         'stopped matching". Adds two subscriptions, so it is '
                         'off unless asked for.')

# Unity build associated to each type of obstacle trajectory
ENV_BUILDS = {
    'intention': '../env_build/INT_EASY/INT_EASY.x86_64',
    'sinusoidal': '../env_build/SIN_EASY/SIN_EASY.x86_64',
    'intention_complex': '../env_build/INT_COMPLEX/INT_COMPLEX.x86_64',
    'sinusoidal_complex': '../env_build/SIN_COMPLEX/SIN_COMPLEX.x86_64',
    'intention_complex_speed': '../env_build/INT_COMPLEX_SPEED/INT_COMPLEX_SPEED.x86_64',
    'sinusoidal_complex_speed': '../env_build/SIN_COMPLEX_SPEED/SIN_COMPLEX_SPEED.x86_64',
}
# The pre-fix builds, kept reachable by path through --env-build. Their
# obstacles step on the frame clock, so they run at about half speed whenever
# the frame rate does not divide the 0.1 s step - 0.050 m/s windowed against
# 0.101 m/s headless. The published results were produced on those.
#   ../env_build/{sin,int}_env/env.x86_64        pre-fix, 7.5 Hz sensors
#   ../env_build/{sin,int}_env_50hz/env.x86_64   pre-fix, 50 Hz sensors

# Command line arguments handed to the Unity player per --env-render. The
# project's default quality level is 5 (Ultra) with vSync on, which is what held
# the frame rate - and therefore the sensor publish rate - down to 16.8 Hz on a
# scene configured for 50 Hz.
ENV_RENDER_ARGS = {
    'headless': ['-batchmode', '-nographics'],
    'low': ['-screen-quality', 'Low'],
    'full': [],
}

# Root directory of every artifact produced by the experiments
DEBUG_DIR = 'debug'

cli_args = parser.parse_args()
dt = cli_args.ts
PLAN_BUDGET = cli_args.plan_budget

# Time the robot stands still between commands while it senses and plans. In the
# default configuration that is one whole ts (the loop runs at 2*ts, half of it
# stopped); with an explicit --plan-budget it collapses to the budget plus a
# small allowance for sensing. Velocity obstacles are enlarged by it, since the
# obstacles keep moving throughout, so shrinking it is what makes VO less
# conservative and gives the planner back its manoeuvring room.
SENSE_ALLOWANCE = 0.005
if PLAN_BUDGET is None:
    THINK_MARGIN = dt
else:
    THINK_MARGIN = PLAN_BUDGET + SENSE_ALLOWANCE

# Upper bound on the planning horizon, in seconds rather than in simulation
# steps so it stays put when the control period changes: as a fixed depth of
# 200 it would silently halve whenever dt did. The depth actually used is
# derived from the discount below and is normally far smaller than this; the
# cap only bites for discounts close to 1.
HORIZON_S = 20.0

# Episode length as a time budget rather than a step count. As a fixed 350
# steps it silently shrank with dt: at dt = 0.05 an episode allowed 17.5 s of
# motion against 35 s at dt = 0.1, i.e. 3.85 m of travel for a goal 3.30 m
# away, so a faster control loop was being scored on half the distance budget.
EPISODE_S = 35.0
MAX_STEPS = int(round(EPISODE_S / dt))

# Options for the planner
# MCTS
# VO-TREE
# VO-PLANNER
exp_num = cli_args.exp_num
algorithm = cli_args.algorithm
trajectories = cli_args.trajectories
EXPLORATION_C = cli_args.exploration_c
RADIUS_SCALE = cli_args.radius_scale
MAX_OBS_RADIUS = cli_args.max_obs_radius
LEGACY_VO = cli_args.vo_geometry == 'legacy'
# Applied at import time, i.e. before any environment or planner is built, so
# that no code path can observe the default first.
set_legacy_vo(LEGACY_VO)
ROLLOUT_COLLISION_CHECK = cli_args.rollout_collision == 'check'
collect_trajectories = cli_args.collect_trajectories
DIAG = cli_args.diag
# One control cycle by default: the robot is stopped for think_margin and moves
# for ts, and the VO radii cover obstacle motion over exactly that span.
MAX_SENSOR_AGE = (dt + THINK_MARGIN if cli_args.max_sensor_age is None
                  else cli_args.max_sensor_age)

# The discount is specified per second and converted to per step, so that the
# effective horizon is a property of the problem rather than of the control
# period: as a bare per-step constant it silently halved whenever dt did.
GAMMA_S = cli_args.gamma_per_second
DISCOUNT = GAMMA_S ** dt

# Rollout depth, derived from the discount instead of fixed. A reward arriving
# at step k is worth DISCOUNT**k, so simulating past the point where that falls
# below TAIL_WEIGHT cannot change any decision. DEPTH was 200 for years, which
# is what 1e-4 asks for at the old gamma of 0.65/s (0.958/step) - so the
# constant was silently tracking a discount nobody had chosen deliberately, and
# stayed put when the discount moved. At 0.04/s (0.725/step) 22 steps carry the
# same 1e-4 of weight and DISCOUNT**200 is 1e-28.
#
# This is about not computing meaningless numbers, not about speed: a 200-step
# fused_rollout is ~11 us against ~277 us per simulation, so the saving is a few
# per cent. Note the rollout can then no longer physically reach a distant goal
# (22 steps is 0.48 m of travel) - but at these discounts a terminal reward that
# far out is weighted below 1e-14 and was never visible to the decision anyway.
TAIL_WEIGHT = 1e-4
DEPTH = min(int(math.ceil(math.log(TAIL_WEIGHT) / math.log(DISCOUNT))),
            int(round(HORIZON_S / dt)))

# Environment executable and output directory for the selected trajectories
env_build = cli_args.env_build or ENV_BUILDS[trajectories]
out_dir = os.path.join(DEBUG_DIR, trajectories)
os.makedirs(out_dir, exist_ok=True)

@jit(nopython=True, cache=True)
def set_seed(value):
    np.random.seed(value)


def seed_everything(seed_value: int):
    """
    Sets the seed for various random number generators to ensure reproducibility.
    This function seeds the following:
    - Python's built-in `random` module
    - NumPy's random number generator
    - The environment variable `PYTHONHASHSEED` for Python's hash-based operations
    - Numba library using the `set_seed` function
    Args:
        seed_value (int): The seed value to use for all random number generators.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    set_seed(seed_value)


class RolloutPlanner:
    def __init__(self, rollout_policy, environment):
        self.rollout_policy = rollout_policy
        self.environment = environment

    def plan(self, state, time_budget):
        return self.rollout_policy(RolloutStateNode(state), self), None



class LoopHandler(Node):

    def __init__(self, dt):
        super().__init__('loopHandler')
        self.dt = dt
        self.logger = self.get_logger()

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # Position of the `Goal` object in the Unity scene, converted with the
        # frame mapping below. It used to read [-3.26, -1.61], which is 0.95 m
        # away from the marker the scene actually draws: the robot was steering
        # at a point no one could see, and "goal reached" was judged against it.
        # Verified by applying the same conversion to the robot's spawn
        # transform, unity (1.136, 0, 0.490) -> (0.490, -1.136), which matches
        # the first odometry reading exactly.
        self.goal = np.array([-2.783, -0.720])

        self.i = 0
        
        # X python = Unity Z
        # Y python = Unity -X
        #
        # The six static obstacles of the scene, read off the transforms in
        # Assets/Scenes/turtlebot3_COPY.unity and converted with the mapping
        # above. Radius 0.100 m is their sphere scale of 0.2 in the scene.
        #
        # The four Obstacle_*_MOVING spheres are deliberately not listed: they
        # drift at up to 0.15 m/s, so a fixed position would be ground truth
        # only at t = 0. Nothing here reaches the planner, which navigates on
        # the LIDAR estimates; these serve the debug plots and the warm-up
        # state alone.
        self.gt_obs_pos = np.array([
            [-0.399,  0.420, 0.0, 0.0],
            [-1.542, -1.790, 0.0, 0.0],
            [-1.539,  0.360, 0.0, 0.0],
            [-2.640, -1.310, 0.0, 0.0],
            [-0.317, -1.820, 0.0, 0.0],
            [-3.020,  0.363, 0.0, 0.0],
        ])
        self.gt_obs_rad = np.array([0.100 for _ in range(len(self.gt_obs_pos))])
        
        # (obs_pos, obs_rad)
        self.obstacles = []
        self.obstacles_pred = []

        if algorithm == 'MCTS':
            _, self.sim_env = create_pedestrian_env(
                discrete=True,
                rwrd_in_sim=True,
                out_boundaries_rwrd=-100,
                n_vel=4,
                n_angles=6,
                vo=False,
                obs_pos=None,
                n_obs=None,
                dt_real=self.dt,
                think_margin=THINK_MARGIN,
            )
            self.config = self.sim_env.config
            self.planner = Mcts(
                num_sim=100,
                c=EXPLORATION_C,
                environment=self.sim_env,
                computational_budget=DEPTH,
                rollout_policy=partial(
                    epsilon_uniform_uniform,
                    std_angle_rollout=2.84*self.dt,
                    eps=0.4
                ),
                discount=DISCOUNT,
                logger=self.logger,
                # must match the eps of rollout_policy above
                rollout_eps=0.4,
                rollout_collision_check=ROLLOUT_COLLISION_CHECK,
                collect_trajectories=collect_trajectories,
            )
        elif algorithm == 'VO-TREE':
            _, self.sim_env = create_pedestrian_env(
                discrete=True,
                rwrd_in_sim=True,
                out_boundaries_rwrd=-100,
                n_vel=4,
                n_angles=6,
                vo=True,
                obs_pos=None,
                n_obs=None,
                dt_real=self.dt,
                think_margin=THINK_MARGIN,
            )
            self.config = self.sim_env.config
            self.planner = Mcts(
                num_sim=100,
                c=EXPLORATION_C,
                environment=self.sim_env,
                computational_budget=DEPTH,
                rollout_policy=partial(
                    epsilon_uniform_uniform,
                    std_angle_rollout=2.84*self.dt,
                    eps=0.2
                ),
                discount=DISCOUNT,
                logger=self.logger,
                # must match the eps of rollout_policy above
                rollout_eps=0.2,
                rollout_collision_check=ROLLOUT_COLLISION_CHECK,
                collect_trajectories=collect_trajectories,
            )
        elif algorithm == 'VO-PLANNER':
            _, self.sim_env = create_pedestrian_env(
                discrete=True,
                rwrd_in_sim=True,
                out_boundaries_rwrd=-100,
                n_vel=4,
                n_angles=6,
                vo=True,
                obs_pos=None,
                n_obs=None,
                dt_real=self.dt,
                think_margin=THINK_MARGIN,
            )
            self.config = self.sim_env.config
            self.planner = RolloutPlanner(
                rollout_policy=partial(
                    epsilon_uniform_uniform_vo,
                    std_angle_rollout=2.84*self.dt,
                    eps=0
                ),
                environment=self.sim_env
            )
        else:
            raise Exception('Invalid Algorithm')

        self.initialize()
        self.i = 0
        self.infos = []
        self.times = []
        # Per-step breakdown of the plan-sense-control cycle, one dict per step:
        # t_sense (obstacle estimation), t_plan (planner), t_budget (what the
        # planner was given), n_sims, n_obs, and t_cycle, the wall-clock gap
        # between two consecutive commands. `times` above keeps holding just the
        # planning budget, so save_data and debug_utils are unaffected.
        self.step_stats = []
        self.prev_move_time = None
        self.actions = []
        self.actions_executed = []
        self.sim_env.gym_env.max_eudist = dist_to_goal(self.s0.goal, self.s0.x[:2])
        # One cycle is: stop, sense and plan, then command the robot for ts. The
        # robot is deliberately stopped while it thinks - that is what keeps the
        # VO guarantee sound while the obstacles keep moving - so the period is
        # ts plus the thinking time, not ts. In the paper configuration thinking
        # is given a whole ts, hence the original hard-coded 2*dt.
        self.t_timer = self.dt + THINK_MARGIN
        if algorithm != 'VO-PLANNER':
            self.timer = self.create_timer(self.t_timer, self.control_loop)
        else:
            self.timer = self.create_timer(self.t_timer, self.control_loop_vo_planner)
        self.logger.info('Loop Handler initialized')
        self.time = 0
        self.obs_pos = None
        self.obs_rad = None
        self.heading_copy = None
        self.pos_copy = None
        self.latest_odom = None
        self.latest_scan = None
        # Two independent subscriptions, each keeping only the newest message.
        #
        # This replaces an ApproximateTimeSynchronizer with slop=0.01, which was
        # measured emitting no pair at all for 11-13 s at a stretch, three times in
        # a 69 s episode, while both topics kept delivering without dropping a
        # single message. The two publishers are stamped from different Unity
        # callbacks - the scanner from Update, odometry from FixedUpdate - each
        # driven by its own WaitForSeconds coroutine, so their relative phase
        # drifts freely. Pairing them within 10 ms of a 100 ms period therefore
        # succeeds only while the drift happens to sit inside the window, which is
        # what produced the blackouts. Any slop is a bet on that drift; not pairing
        # at all removes the failure instead of making it rarer.
        #
        # The loop never needed simultaneity, only recency: it reads one pose and
        # one scan per cycle and treats them as the current world state. Recency is
        # checked below, per topic, against each message's own stamp - which is
        # immune to the drift, being a comparison against the clock rather than
        # against the other topic.
        #
        # qos_profile_sensor_data, i.e. best-effort, as the original subscriptions
        # used: for a sensor stream the newest sample is what matters, and a
        # reliable reader would have DDS retransmit and apply backpressure while
        # control_loop spends its 75 ms planning.
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self._on_odom,
            rclpy.qos.qos_profile_sensor_data
        )
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self._on_scan,
            rclpy.qos.qos_profile_sensor_data
        )

        # Diagnostics. pair_seq counts what the synchroniser emits; raw_odom and
        # raw_scan count what arrives on the topics at all. The two together say
        # whether a cycle went stale because nothing was published or because
        # nothing matched, which is not decidable from the recorded runs.
        self.pair_seq = 0
        self.raw_odom = 0
        self.raw_scan = 0
        self.prev_pair_seq = -1
        self.stale_skips = 0
        if DIAG:
            self.diag_odom_sub = self.create_subscription(
                Odometry, '/odom', self._count_odom, rclpy.qos.qos_profile_sensor_data)
            self.diag_scan_sub = self.create_subscription(
                LaserScan, '/scan', self._count_scan, rclpy.qos.qos_profile_sensor_data)

        self.clusting_algo = HDBSCAN(allow_single_cluster= True, alpha=0.5, cluster_selection_epsilon=0.01, min_cluster_size=2, min_samples=1, n_jobs=-1)
        self.last_action = np.array([0., self.s0.x[2]])
        self.max_obs_vel = cli_args.max_obs_vel
        self.robot_position = None
        self.heading = None
        self.obs_rad = None
        self.obs_pos = None
        self.lidar_msg = None
        self.odom_msg = None
        self.points_list = []
        self.update_odom = False
        self.update_lidar = False
        self.prev_odom = None
        self.prev_lidar = None
        self.reached_goal = False
        self.collision = False
        self.obs_collision = False
        self.max_steps = False
        self.distances = None
        self.angles = None

    def _on_odom(self, msg):
        """Keep the newest odometry message."""
        self.latest_odom = msg
        self.pair_seq += 1

    def _on_scan(self, msg):
        """Keep the newest scan."""
        self.latest_scan = msg

    def _count_odom(self, _msg):
        """Raw arrival counter, deliberately bypassing the message filter."""
        self.raw_odom += 1

    def _count_scan(self, _msg):
        self.raw_scan += 1

    @staticmethod
    def _stamp_age(now_ns, msg):
        """Seconds between a message's header stamp and now, NaN if unavailable."""
        if msg is None:
            return float('nan')
        st = msg.header.stamp
        return (now_ns - (st.sec * 1_000_000_000 + st.nanosec)) / 1e9

        
    def SetLaser(self, msg):
        """
        Sets the lidar message.
        This method assigns the provided message to the `lidar_msg` attribute.
        Args:
            msg: The message containing lidar data to be set.
        """
        self.lidar_msg = msg

    def SetOdom(self, msg):
        """
        Sets the odometry message.
        This method assigns the provided odometry message to the `odom_msg` attribute.
        Args:
            msg: The odometry message to be set. The type of `msg` depends on the 
                 specific implementation or ROS message type being used.
        """
        self.odom_msg = msg
        
    def get_odom(self):
        """
        Retrieves the current odometry information, including the position and heading.
        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: A 2D array representing the (x, y) position.
                - float: The heading (yaw) angle in radians, normalized to the range [-π, π].
        """

        point = self.odom_msg.pose.pose.position
        rot = self.odom_msg.pose.pose.orientation
        self.rot_ = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
        heading = ((self.rot_[2]) + np.pi) % (2 * np.pi) - np.pi
        return np.array([point.x, point.y]), heading

    def get_scan(self):
        """
        Retrieves and processes LiDAR scan data to extract valid distances and their corresponding angles.
        Returns:
            tuple: A tuple containing:
                - distances (numpy.ndarray): An array of valid distance measurements from the LiDAR scan.
                - angles (numpy.ndarray): An array of angles corresponding to the valid distance measurements.
        """

        scan = self.lidar_msg.ranges
        angle_min = self.lidar_msg.angle_min
        angle_increment = self.lidar_msg.angle_increment

        mask = np.where(~np.logical_or(np.isnan(scan), np.isinf(scan)))[0]
        scan = np.array(scan)

        distances = scan[mask.astype(int)].copy()
        angles = mask * angle_increment + angle_min

        return distances, angles

    

    def group_matrix(self, M, I):
        unique_indices = np.unique(I)
        return {idx: M[I == idx] for idx in unique_indices}


    def estimate_obstacles(self, pos, heading, dist, angles):
        """
        Estimates the positions and radii of obstacles based on LiDAR data.
        Processes LiDAR distance and angle data to identify circular obstacles
        using clustering and RANSAC, and returns their positions and radii.
        Args:
            pos (np.ndarray): Current robot position (x, y).
            heading (float): Current robot heading in radians.
            dist (np.ndarray): LiDAR-measured distances.
            angles (np.ndarray): Angles corresponding to LiDAR distances.
        Returns:
            tuple:
            - obs_pos (np.ndarray): Array of shape (N, 4) with obstacle positions (x, y),
              heading (set to 0), and maximum velocity.
            - obs_rad (np.ndarray): Array of shape (N,) with obstacle radii.
        Notes:
            - Clusters with fewer than 3 points are ignored.
            - Obstacles whose measured radius exceeds MAX_OBS_RADIUS are
              filtered out, before any scaling.
            - Surviving radii are then scaled by RADIUS_SCALE.
        """

        # If no distance data is available, return empty arrays
        if len(dist) == 0:
            return np.empty((0, 4)), np.array([])

        # Convert LiDAR distances and angles into Cartesian coordinates
        points = get_points_from_lidar(dist, angles, pos, heading)
        self.points_list.append(points)

        # Perform clustering on the points to group potential obstacles
        clusters = self.clusting_algo.fit_predict(points)
        groups = self.group_matrix(points, clusters)

        # Initialize arrays to store obstacle positions and radii
        obs_pos = np.empty((0, 2))
        obs_rad = np.array([])

        # Iterate through each cluster group
        for group in groups.values():
            # Ignore clusters with fewer than 3 points
            if len(group) < 3:
                continue

            # Use RANSAC to fit a circle model to the cluster points
            ransac_model, _ = ransac(
                group, 
                CircleModel, 
                max_trials=100, 
                min_samples=3, 
                residual_threshold=0.1, 
                stop_probability=0.99, 
                rng=0
            )
            if ransac_model is None:
                continue

            # Extract the circle's center and radius from the RANSAC model
            center = ransac_model.params[0:2]
            radius = ransac_model.params[2]

            # Append the center and the *measured* radius to the respective
            # arrays. The RADIUS_SCALE inflation is applied after the filter
            # below, not here, so that the two knobs stay independent.
            obs_pos = np.vstack((obs_pos, center))
            obs_rad = np.append(obs_rad, radius)

        # Discard implausibly large fits - walls, mostly - on the measured
        # radius. Filtering the scaled radius instead would make the effective
        # cutoff MAX_OBS_RADIUS / RADIUS_SCALE, so raising the scale to widen
        # the safety margin would also silently stop the largest obstacles from
        # being detected at all.
        mask = obs_rad <= MAX_OBS_RADIUS
        obs_rad = obs_rad[mask] * RADIUS_SCALE
        obs_pos = obs_pos[mask]

        # Add heading (set to 0) and maximum velocity to the obstacle positions
        obs_pos = np.hstack((obs_pos, np.tile([0, self.max_obs_vel], (len(obs_pos), 1))))

        # Return the estimated obstacle positions and radii
        return obs_pos, obs_rad
    
    
    def callback_lidar(self, msg):
        """
        Callback function for processing incoming LiDAR messages.
        This function is triggered whenever a new LiDAR message is received. It updates
        the robot's LiDAR data, checks for changes in the LiDAR readings, and determines
        if there is a potential collision based on the minimum distance to obstacles.
        Args:
            msg (sensor_msgs.msg.LaserScan): The incoming LiDAR message containing
                range and angle data.
        Returns:
            None: If the robot's position is not initialized.
        Attributes Updated:
            self.prev_lidar (sensor_msgs.msg.LaserScan): Stores the previous LiDAR message.
            self.update_lidar (bool): Indicates whether the LiDAR readings have changed.
            self.distances (list[float]): List of distances to obstacles from the LiDAR scan.
            self.collision (bool): True if the minimum distance to an obstacle is less than
                or equal to the robot's radius, indicating a potential collision.
            self.angles (list[float]): List of angles corresponding to the LiDAR scan distances.
        """
        # If the robot's position is not initialized, exit the callback
        if self.robot_position is None:
            return

        # Check if a previous LiDAR message exists
        if self.lidar_msg is not None:
            # Store the previous LiDAR message for comparison
            self.prev_lidar = deepcopy(self.lidar_msg)
            # Update the flag if the current LiDAR ranges differ from the previous ones
            self.update_lidar = self.prev_lidar.ranges != msg.ranges

        # Set the current LiDAR message
        self.SetLaser(msg)
        # Retrieve distances and angles from the LiDAR scan
        dist, angles = self.get_scan()
        # Update the distances and angles attributes
        self.distances = dist
        # Check for potential collisions based on the minimum distance to obstacles
        self.collision = min(dist) <= self.config.robot_radius
        self.angles = angles


    def callback_odom(self, msg):
        """
        Callback function for handling odometry messages.
        This function is triggered whenever a new odometry message is received.
        It updates the previous odometry data, checks if the robot's position has
        changed, and updates the current odometry information.
        Args:
            msg (nav_msgs.msg.Odometry): The incoming odometry message containing
                the robot's current position and orientation.
        Side Effects:
            - Updates `self.prev_odom` with the previous odometry message.
            - Sets `self.update_odom` to True if the robot's position has changed.
            - Updates the current odometry information using `self.SetOdom`.
            - Updates `self.robot_position` and `self.heading` with the robot's
              current position and heading.
        """

        # Check if a previous odometry message exists
        if self.odom_msg is not None:
            # Store the previous odometry message
            self.prev_odom = deepcopy(self.odom_msg)
            # Update the flag if the current position differs from the previous one
            self.update_odom = self.prev_odom.pose.pose.position != msg.pose.pose.position

        # Set the current odometry message
        self.SetOdom(msg)
        # Update the robot's position and heading based on the current odometry
        self.robot_position, self.heading = self.get_odom()

    
    def initialize(self):
        # state, dist, angles, pos = self.get_state()
        state = np.array([0.49, -1.136, -3.14, 0.0])
        # state = np.array([0., 0., 0., 0.0])
        self.s0, _ = self.sim_env.reset()
        self.s0.goal = self.goal
        obs = (
            self.gt_obs_pos,
            self.gt_obs_rad
        )
        self.s0.obstacles = obs
        self.s0.x = state
        self.trajectory = np.empty((0, 4))
        self.planning_states = np.empty((0, 4))
        self.planner.plan(self.s0, 0.2)

        self.pub.publish(Twist())
    
    def move(self, state, action, pub):
        """
        Executes a movement action by publishing a Twist message to a ROS topic.
        Args:
            state (list): The current state of the robot, where state[2] represents
                          the robot's current orientation (theta) in radians.
            action (list): The desired action to execute, where action[0] is the
                           linear velocity (m/s) and action[1] is the target orientation (theta) in radians.
            pub (rospy.Publisher): A ROS publisher object used to publish Twist messages.
        Behavior:
            - Computes the angular velocity required to achieve the desired orientation
              by calculating the shortest angular distance (d_theta) between the current
              and target orientations.
            - Publishes a Twist message containing the linear and angular velocities.
            - Appends the executed action (linear velocity and angular velocity) to the
              `self.actions_executed` list for tracking.
        Note:
            - The angular velocity is calculated based on the time step `self.dt`.
        """
        # Record the current time for tracking
        curr_time = time.time()
        self.time = curr_time
        
        # Create a Twist message to define the robot's movement
        twist = Twist()
        twist.linear.x = action[0]  # Set the linear velocity from the action
        twist.linear.y = 0.0  # No movement in the y-direction
        twist.linear.z = 0.0  # No movement in the z-direction (2D motion)
                
        twist.angular.x = 0.0  # No rotation around the x-axis
        twist.angular.y = 0.0  # No rotation around the y-axis

        # Calculate the angular velocity required to achieve the desired orientation
        d_theta = (action[1] - state[2] + np.pi) % (2 * np.pi) - np.pi
        twist.angular.z = d_theta / self.dt

        # Publish the Twist message to the robot's command velocity topic
        pub.publish(twist)

        # Append the executed action (linear and angular velocities) to the actions_executed list
        self.actions_executed.append([twist.linear.x, twist.angular.z])

        
    def control_loop(self):
        """
        Executes the main control loop for the robot.
        This method handles the robot's movement, state updates, obstacle estimation, 
        and planning. It publishes commands to the robot, updates its trajectory, 
        and checks for termination conditions such as reaching the goal, collisions, 
        or exceeding the maximum number of steps.
        Raises:
            Exception: If the goal is reached, a collision occurs, or the maximum 
                       number of steps is exceeded.
        Steps:
            1. Publishes an initial stop command to the robot.
            2. Checks if odometry and lidar updates are available; exits if not.
            3. Logs the current step and updates the robot's state and trajectory.
            4. Estimates obstacles based on sensor data and updates the obstacle list.
            5. Checks for termination conditions (goal reached, collision, max steps).
            6. If not terminated, plans the next action using the planner and executes it.
        Note:
            - The method uses a fixed random seed for reproducibility during planning.
            - The robot's movement is controlled by publishing Twist messages.
        """

        # 1. Stop the robot at the beginning of the thinking phase
        self.pub.publish(Twist())

        # 2. Wait until both topics have produced something
        if self.latest_odom is None or self.latest_scan is None:
            self.get_logger().debug("control_loop: no sensor data yet, skipping")
            return

        # 2b. Refuse to plan on stale data.
        #
        # Everything below treats the stored pose and scan as the current state of
        # the world, and the velocity obstacles are sized for exactly one control
        # cycle of obstacle motion. Sensor data older than that breaks the
        # assumption the safety argument rests on, so the only sound response is
        # not to move: the stop command was already published above, and the tick
        # is abandoned.
        #
        # This restores a gate the loop used to have and lost when the message
        # filter was introduced - the --env-render help still describes it - except
        # that freshness is now judged from each message's own stamp rather than by
        # comparing consecutive values. A value comparison cannot tell "the reading
        # did not change" from "no reading arrived", and those differ precisely
        # when VO has commanded a stop, i.e. when the robot is stationary and its
        # pose legitimately repeats.
        now_ns = self.get_clock().now().nanoseconds
        age_odom = self._stamp_age(now_ns, self.latest_odom)
        age_scan = self._stamp_age(now_ns, self.latest_scan)
        pair_seq = self.pair_seq
        stale = max(age_odom, age_scan) > MAX_SENSOR_AGE
        self.prev_pair_seq = pair_seq
        diag = {
            'pair_seq': pair_seq,
            'stale': stale,
            'raw_odom': self.raw_odom,
            'raw_scan': self.raw_scan,
            'age_odom': age_odom,
            'age_scan': age_scan,
        }
        if DIAG:
            self.logger.info(
                f"diag step={self.i} pair_seq={pair_seq} stale={stale} "
                f"raw_odom={self.raw_odom} raw_scan={self.raw_scan} "
                f"age_odom={age_odom:.3f} age_scan={age_scan:.3f}"
            )
        if stale:
            self.stale_skips += 1
            # Loud, because a robot that stops for seconds on end is a symptom
            # worth seeing rather than a state to sit in quietly.
            self.logger.warn(
                f"stale sensor data at step {self.i}: odom {age_odom:.3f} s, "
                f"scan {age_scan:.3f} s, limit {MAX_SENSOR_AGE:.3f} s - "
                f"holding position (skip {self.stale_skips})"
            )
            return

        # 3. Use the latest message from each topic as the current world state

        self.odom_msg = self.latest_odom
        self.lidar_msg = self.latest_scan

        # 4. Extract robot state 
        self.robot_position, self.heading = self.get_odom()

        # 5. Extract LiDAR data and compute collision (was in callback_lidar)
        dist, angles = self.get_scan()
        self.distances = dist
        self.angles = angles
        self.collision = min(dist) <= self.config.robot_radius  # exactly as before

        # Log the current step number
        self.logger.info(f"Step: {self.i}")

        # Retrieve the robot's current position and heading
        position, heading = self.robot_position.copy(), deepcopy(self.heading)

        # Retrieve and copy lidar distances and angles
        dist, angles = self.distances.copy(), self.angles.copy()
        dist = dist.copy()
        angles = angles.copy()

        # Update the robot's state with the current position, heading, and velocity
        robot_state = np.array([position[0], position[1], heading, self.s0.x[3]])
        self.s0.x = robot_state

        # Append the current state to the trajectory for tracking
        self.trajectory = np.vstack((self.trajectory, self.s0.x))

        # Start timing for obstacle estimation and planning
        start_time = time.time()

        # Estimate obstacles based on lidar data and update the environment's obstacle list
        self.obs_pos, self.obs_rad = self.estimate_obstacles(position, heading, dist, angles)
        self.s0.obstacles = (self.obs_pos, self.obs_rad)
        self.obstacles.append(self.s0.obstacles)

        # Check the distance to the goal
        d = dist_to_goal(self.s0.goal, position)

        # Determine if the goal has been reached. The tolerance matches the
        # simulator's own terminal test (Env.step_*: dist_goal <= robot_radius),
        # so that a state the planner treats as terminal is also one the loop
        # scores as a success.
        self.reached_goal = d <= self.config.robot_radius

        # Check termination conditions: maximum steps, goal reached, or collision
        if self.i == MAX_STEPS or self.reached_goal or self.collision:
            # Publish a stop command to halt the robot
            self.pub.publish(Twist())

            # Determine if the collision was with an obstacle or due to other reasons
            self.obs_collision = self.collision and self.last_action[0] == 0
            self.collision = self.collision and self.last_action[0] != 0

            # Check if the maximum number of steps has been reached
            self.max_steps = self.i == MAX_STEPS

            # Log the termination condition
            self.logger.info(f"Goal Reached: {self.reached_goal} Collision: {self.collision} Obs Collision: {self.obs_collision}")

            # Raise an exception to terminate the loop
            raise Exception("Finished")
        
        # Calculate the time taken for obstacle estimation
        t1 = time.time() - start_time

        # Start timing for planning
        initial_time = time.time()

        # Append the current obstacles to the predicted obstacle list
        self.obstacles_pred.append(self.s0.obstacles)

        # Append the current state to the planning states for tracking
        self.planning_states = np.vstack((self.planning_states, self.s0.x))

        # Time given to the planner. By default it is whatever is left of one ts
        # after sensing, which is the paper's configuration; --plan-budget sets
        # it directly, which is what decouples the cycle time from ts.
        if PLAN_BUDGET is None:
            budget = self.dt - t1 - SENSE_ALLOWANCE
        else:
            budget = PLAN_BUDGET
        self.times.append(budget)

        if budget <= 0:
            # Sensing alone overran the step. Planning now would push the cycle
            # past the period the velocity obstacles were sized for, so re-issue
            # the last action, which VO already certified against the obstacles
            # as they were one step ago, and give up the step.
            self.logger.warn(
                f"sensing took {t1 * 1e3:.1f} ms, leaving no planning budget; "
                f"repeating the last action"
            )
            action, info = self.last_action, None
        else:
            # Plan the next action using the planner
            action, info = self.planner.plan(self.s0, budget)

        # Append the planning information and action to their respective lists
        self.infos.append(info)
        self.actions.append(action)

        # Update the last action taken
        self.last_action = action

        # Calculate the time taken for planning
        t2 = time.time() - initial_time

        # Reset the odometry and lidar update flags
        self.update_odom = False
        self.update_lidar = False

        # Increment the step counter
        self.i += 1

        # Record the cycle breakdown. t_cycle is measured command to command,
        # so it is the period the robot actually ran at - which is not
        # necessarily t_timer: control_loop returns early when no new sensor
        # data has arrived, and then the whole tick is skipped.
        now = time.time()
        self.step_stats.append({
            "t_sense": t1,
            "t_plan": t2,
            "t_budget": budget,
            "n_sims": info["simulations"] if info is not None else np.nan,
            "n_obs": len(self.obs_rad),
            "t_cycle": np.nan if self.prev_move_time is None
                       else now - self.prev_move_time,
            **diag,
        })
        self.prev_move_time = now

        # Execute the planned action by moving the robot
        self.move(self.s0.x, self.last_action, self.pub)

    def control_loop_vo_planner(self):
        """
        Executes the control loop for the VO (Velocity Obstacle) planner, ensuring
        that the loop adheres to the desired time step interval (dt).
        This method measures the execution time of the control loop and, if the
        execution completes faster than the specified time step (dt), it introduces
        a delay to maintain a consistent loop frequency.
        Steps:
            1. Records the initial time before executing the control loop.
            2. Executes the control loop logic via `self.control_loop()`.
            3. Calculates the elapsed time for the control loop execution.
            4. If the elapsed time is less than `self.dt`, sleeps for the remaining
               time to maintain the desired loop frequency.
        """

        initial_time = time.time()
        self.control_loop()
        final_time = time.time() - initial_time
        if final_time < self.dt:
            time.sleep(self.dt - final_time)
        
def save_data(loopHandler, exp_num):
    """
    Save various data attributes and experiment results of the loopHandler object 
    to files for debugging and analysis.
    Args:
        loopHandler (object): An object containing simulation data and attributes 
                              related to the experiment.
        exp_num (int): The experiment number used to create unique file suffixes.
    Saves:
        - Pickle files containing:
            - Number of simulations (`sim_num`)
            - Actions (`actions`)
            - Trajectory (`trajectory`)
            - Planning states (`planning_states`)
            - Obstacles (`obstacles`)
            - Predicted obstacles (`obstacles_pred`)
            - Time steps (`times`)
            - Executed actions (`actions_executed`)
        - A CSV file containing:
            - Algorithm name
            - Whether the goal was reached
            - Collision status
            - Obstacle collision status
            - Maximum steps allowed
            - Number of steps taken
            - Discounted return
            - Undiscounted return
            - Mean and standard deviation of simulation numbers
    """

    # Define a suffix for file names based on the algorithm and experiment number
    suffix = f'{algorithm}_{exp_num}{cli_args.suffix}'

    # Check if the loopHandler's infos attribute contains valid data
    if None not in loopHandler.infos and len(loopHandler.infos) != 0:
        # Extract the number of simulations from the infos and save it to a pickle file
        sim_num = [i["simulations"] for i in loopHandler.infos]
        pickle.dump(sim_num, open(f"{out_dir}/sim_num_{suffix}.pkl", 'wb'))

        # Depth statistics of the search, one value per planning step.
        # Only tree-based planners (MCTS, VO-TREE) produce them: VO-PLANNER
        # returns None as info, so the columns below stay empty for it.
        tree_depths = [i["max_tree_depth"] for i in loopHandler.infos]
        rollout_depths = [i["max_rollout_depth"] for i in loopHandler.infos]
        total_depths = [i["max_total_depth"] for i in loopHandler.infos]
        # The mean rollout length, as opposed to the per-plan maximum above.
        # The maximum is pinned to the budget whenever any rollout runs to the
        # end, which is why it read as a constant 199 in every campaign so far;
        # the mean is what shows how often rollouts terminate early.
        mean_rollout_depths = [i["mean_rollout_depth"] for i in loopHandler.infos]
        pickle.dump(
            {
                "max_tree_depth": tree_depths,
                "max_rollout_depth": rollout_depths,
                "max_total_depth": total_depths,
                "mean_rollout_depth": mean_rollout_depths,
            },
            open(f"{out_dir}/depths_{suffix}.pkl", 'wb')
        )
    else:
        # If infos contains None, initialize sim_num as an empty list
        sim_num = []
        tree_depths = []
        rollout_depths = []
        total_depths = []
        mean_rollout_depths = []

    # Save various data attributes of the loopHandler to pickle files for debugging
    pickle.dump(loopHandler.actions, open(f"{out_dir}/acts_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.trajectory, open(f"{out_dir}/trj_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.planning_states, open(f"{out_dir}/ps_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.obstacles, open(f"{out_dir}/obs_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.obstacles_pred, open(f"{out_dir}/obsPred_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.times, open(f"{out_dir}/times_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.actions_executed, open(f"{out_dir}/actions_executed_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.step_stats, open(f"{out_dir}/step_stats_{suffix}.pkl", 'wb'))

    # Per-phase timings, summarised into the CSV. The median says what a step
    # normally costs, p99 says whether the budget is ever blown - and it is the
    # p99, not the mean, that decides whether t_sense + t_plan < ts holds, which
    # is the precondition for the VO guarantee.
    def _phase(field):
        vals = np.array([s[field] for s in loopHandler.step_stats], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.median(vals)), float(np.percentile(vals, 99))

    t_sense_med, t_sense_p99 = _phase("t_sense")
    t_plan_med, t_plan_p99 = _phase("t_plan")
    t_cycle_med, t_cycle_p99 = _phase("t_cycle")

    # Sensor freshness, summarised so the answer is readable straight off the CSV.
    # staleFrac is the share of cycles that replanned on the previous cycle's data;
    # staleRun is the worst uninterrupted stretch of them. pairsPerCycle against
    # rawScanPerCycle is the discriminator: both near zero means the simulator
    # stopped publishing, the first near zero while the second does not means the
    # synchroniser stopped matching.
    ss = loopHandler.step_stats
    n_cycles = len(ss)
    if n_cycles:
        # A stale tick now returns before step_stats is appended, so the flag is
        # False on every recorded cycle by construction. What counts is how many
        # ticks were refused: staleFrac is skips as a share of all ticks, refused
        # plus completed.
        n_skips = loopHandler.stale_skips
        stale_frac = n_skips / (n_skips + n_cycles)
        worst = n_skips
        pairs_per_cycle = ss[-1].get("pair_seq", np.nan) / n_cycles
        # The raw counters only advance when --diag installed the extra
        # subscriptions. Report NaN rather than 0 otherwise, so an ordinary run
        # cannot be misread as "the simulator published nothing".
        if DIAG:
            raw_scan_per_cycle = ss[-1].get("raw_scan", np.nan) / n_cycles
            raw_odom_per_cycle = ss[-1].get("raw_odom", np.nan) / n_cycles
        else:
            raw_scan_per_cycle = raw_odom_per_cycle = np.nan
        age_scan_med, age_scan_p99 = _phase("age_scan")
    else:
        stale_frac = worst = pairs_per_cycle = np.nan
        raw_scan_per_cycle = raw_odom_per_cycle = np.nan
        age_scan_med = age_scan_p99 = np.nan

    # Calculate normalized distances to the goal
    max_eudist = loopHandler.sim_env.gym_env.max_eudist
    goal = loopHandler.s0.goal
    distances = np.linalg.norm(loopHandler.trajectory[:, :2] - goal, axis=1) / max_eudist

    # Adjust the last distance value based on the termination condition
    if loopHandler.reached_goal:
        distances[-1] += 100  # Add a large positive value if the goal is reached
    elif loopHandler.collision or loopHandler.obs_collision:
        distances[-1] -= 100  # Subtract a large value if a collision occurred

    # Compute discounted and undiscounted returns
    discounts = DISCOUNT ** np.arange(len(distances))
    discounted_return = np.sum(distances * discounts)
    undiscounted_return = np.sum(distances)

    # Create a dictionary to store experiment results
    data = {
        "algorithm": algorithm,
        "trajectories": trajectories,
        # Which simulator produced this run. Obstacle speed differs by a factor
        # of two between the pre-fix builds and the fixed ones, and on the
        # pre-fix builds also with the render mode, so a result is not
        # interpretable without them.
        "envBuild": env_build,
        "envRender": cli_args.env_render,
        "expNum": exp_num,
        "reachGoal": loopHandler.reached_goal,
        "collision": loopHandler.collision,
        "Obscollision": loopHandler.obs_collision,
        "maxSteps": loopHandler.max_steps,
        "nSteps": loopHandler.i + 1,
        "discountedReturn": discounted_return,
        "undiscountedReturn": undiscounted_return,
        "simNum": np.mean(sim_num),
        "simNumStd": np.std(sim_num),
        # Depth statistics: "max" is over the whole run, "mean" is the average
        # of the per-planning-step maxima. NaN for VO-PLANNER, which has no tree.
        "maxTreeDepth": np.max(tree_depths) if tree_depths else np.nan,
        "meanTreeDepth": np.mean(tree_depths) if tree_depths else np.nan,
        "maxRolloutDepth": np.max(rollout_depths) if rollout_depths else np.nan,
        "meanRolloutDepth": np.mean(rollout_depths) if rollout_depths else np.nan,
        # Mean over rollouts, not over per-plan maxima: the length a typical
        # rollout actually ran before reaching the goal, colliding or
        # exhausting the budget.
        "rolloutLen": np.mean(mean_rollout_depths) if mean_rollout_depths else np.nan,
        "maxTotalDepth": np.max(total_depths) if total_depths else np.nan,
        "meanTotalDepth": np.mean(total_depths) if total_depths else np.nan,
        # Per-phase timings, in seconds.
        "tSenseMed": t_sense_med,
        "tSenseP99": t_sense_p99,
        "tPlanMed": t_plan_med,
        "tPlanP99": t_plan_p99,
        "tCycleMed": t_cycle_med,
        "tCycleP99": t_cycle_p99,
        "staleFrac": stale_frac,
        "staleSkips": worst,
        "maxSensorAge": MAX_SENSOR_AGE,
        "pairsPerCycle": pairs_per_cycle,
        "rawScanPerCycle": raw_scan_per_cycle,
        "rawOdomPerCycle": raw_odom_per_cycle,
        "ageScanMed": age_scan_med,
        "ageScanP99": age_scan_p99,
        # Configuration, so a sweep's CSVs are self-describing.
        "ts": dt,
        "planBudget": PLAN_BUDGET if PLAN_BUDGET is not None else np.nan,
        "thinkMargin": THINK_MARGIN,
        "horizonS": HORIZON_S,
        "depth": DEPTH,
        "radiusScale": RADIUS_SCALE,
        "maxObsRadius": MAX_OBS_RADIUS,
        "voGeometry": cli_args.vo_geometry,
        "maxObsVel": cli_args.max_obs_vel,
        "explorationC": EXPLORATION_C,
        "gammaPerSecond": GAMMA_S,
    }

    # Save the results to a CSV file for analysis
    df = pd.DataFrame([data])
    df.to_csv(f"{out_dir}/data_{suffix}.csv")
        

def main(args=None):
    """
    The main entry point for the application.
    This function initializes the ROS 2 client library, disables garbage collection,
    and sets up the main loop for the application. It launches an external process,
    manages the ROS 2 executor, and handles cleanup in case of exceptions.
    Args:
        args (list, optional): Command-line arguments passed to the ROS 2 client library. Defaults to None.
    Behavior:
        - Initializes ROS 2 with the provided arguments.
        - Disables Python's garbage collection to improve performance.
        - Prints the experiment number.
        - Creates an instance of the `LoopHandler` class with a specified time step.
        - Launches an external process (e.g., a simulation environment).
        - Sets up a single-threaded ROS 2 executor and adds the `LoopHandler` node to it.
        - Spins the executor to process ROS 2 callbacks.
        - Handles exceptions by:
            - Destroying the `LoopHandler` node.
            - Terminating the external process.
            - Saving data and generating debug plots/animations.
            - Collecting garbage to free resources.
    """

    rclpy.init(args=args)

    # Set a fixed random seed for reproducibility
    seed_everything(exp_num)

    gc.disable()
    print(f"Experiment: {exp_num} | Algorithm: {algorithm} | Trajectories: {trajectories}")
    print(f"Environment: {env_build} | Output directory: {out_dir}")
    print(f"Render mode: {cli_args.env_render} ({' '.join(ENV_RENDER_ARGS[cli_args.env_render]) or 'default'})")
    loopHandler = LoopHandler(dt)
    process = subprocess.Popen([env_build] + ENV_RENDER_ARGS[cli_args.env_render],
                               preexec_fn=os.setpgrp)
    time.sleep(2)
    try:
        executor = SingleThreadedExecutor()
        executor.add_node(loopHandler)
        executor.spin()
    except Exception as e:
        loopHandler.destroy_node()
        # kill the environment process
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        save_data(loopHandler, exp_num)
        if not cli_args.no_plots:
            debug_plots_and_animations(loopHandler, exp_num, algorithm=algorithm,
                                       out_dir=out_dir, suffix_tag=cli_args.suffix)
        gc.collect()


if __name__ == '__main__':
    main()
