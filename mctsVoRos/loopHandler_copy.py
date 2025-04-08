import gc
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
from MCTS_VO.bettergym.agents.utils.vo import epsilon_uniform_uniform_vo


parser = argparse.ArgumentParser()
parser.add_argument('--exp_num', default=0, type=int)
parser.add_argument('--algorithm', default='VO-PLANNER', type=str)

MAX_STEPS = 350
RADIUS_SCALE = 3
DISCOUNT = 0.9
DEPTH = 200
dt = 0.1

# Options for the planner
# MCTS
# VO-TREE
# VO-PLANNER
exp_num = parser.parse_args().exp_num
algorithm = parser.parse_args().algorithm

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
        self.goal = np.array([-3.26, -1.61])
        
        self.i = 0
        
        # X python = Unity Z
        # Y python = Unity -X
        self.gt_obs_pos = np.array([
            [-2.221, -2.04, 0.0, 0.0],
            [-0.92, -1.651, 0.0, 0.0],
            [-1.127, -0.833, 0.0, 0.0],
            [-1.739,-1.395, 0.0, 0.0],
            [-2.003,-0.696, 0.0, 0.0],
            # [-2.357,-1.027, 0.0, 0.0],
            [-1.127,-0.146, 0.0, 0.0],
            
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
            )
            self.config = self.sim_env.config
            self.planner = Mcts(
                num_sim=100,
                c=10,
                environment=self.sim_env,
                computational_budget=DEPTH,
                rollout_policy=partial(
                    epsilon_uniform_uniform,
                    std_angle_rollout=2.84*self.dt,
                    eps=0.4
                ),
                discount=DISCOUNT,
                logger=self.logger
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
            )
            self.config = self.sim_env.config
            self.planner = Mcts(
                num_sim=100,
                c=10,
                environment=self.sim_env,
                computational_budget=DEPTH,
                rollout_policy=partial(
                    epsilon_uniform_uniform,
                    std_angle_rollout=2.84*self.dt,
                    eps=0.2
                ),
                discount=DISCOUNT,
                logger=self.logger
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
        self.actions = []
        self.actions_executed = []
        self.sim_env.gym_env.max_eudist = dist_to_goal(self.s0.goal, self.s0.x[:2])
        # t_timer is double the dt
        self.t_timer = 0.2
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
        
        self.odom_subscriber = self.create_subscription(
            Odometry, 
            'odom', 
            self.callback_odom, 
            rclpy.qos.qos_profile_sensor_data
        )
        self.lidar_subscriber = self.create_subscription(
            LaserScan, 
            'scan', 
            self.callback_lidar, 
            rclpy.qos.qos_profile_sensor_data
        )
        self.clusting_algo = HDBSCAN(allow_single_cluster= True, alpha=0.5, cluster_selection_epsilon=0.01, min_cluster_size=2, min_samples=1, n_jobs=-1)
        self.last_action = np.array([0., self.s0.x[2]])
        self.max_obs_vel = 0.1
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

        distances = []
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
            - Obstacles with radii > 0.5 are filtered out.
            - Detected radii are scaled by RADIUS_SCALE.
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
            ransac_model, _ = ransac(group, CircleModel, max_trials=100, min_samples=3, residual_threshold=0.1, stop_probability=0.99)
            if ransac_model is None:
                continue

            # Extract the circle's center and radius from the RANSAC model
            center = ransac_model.params[0:2]
            radius = ransac_model.params[2]

            # Scale the radius by a predefined factor
            radius *= RADIUS_SCALE

            # Append the center and radius to the respective arrays
            obs_pos = np.vstack((obs_pos, center))
            obs_rad = np.append(obs_rad, radius)

        # Filter out obstacles with radii greater than 0.5
        mask = obs_rad <= 0.5
        obs_rad = obs_rad[mask]
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

        # Publish a stop command to ensure the robot starts from a stationary state
        self.pub.publish(Twist())

        # Check if odometry and lidar updates are available; if not, exit the loop
        if not self.update_odom or not self.update_lidar:
            return
        
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

        # Set a fixed random seed for reproducibility
        seed_everything(0)

        # Estimate obstacles based on lidar data and update the environment's obstacle list
        self.obs_pos, self.obs_rad = self.estimate_obstacles(position, heading, dist, angles)
        self.s0.obstacles = (self.obs_pos, self.obs_rad)
        self.obstacles.append(self.s0.obstacles)

        # Check the distance to the goal
        d = dist_to_goal(self.s0.goal, position)

        # Determine if the goal has been reached
        self.reached_goal = d <= 0.2

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

        # Set a fixed random seed for reproducibility
        seed_everything(0)

        # Start timing for planning
        initial_time = time.time()

        # Append the current obstacles to the predicted obstacle list
        self.obstacles_pred.append(self.s0.obstacles)

        # Append the current state to the planning states for tracking
        self.planning_states = np.vstack((self.planning_states, self.s0.x))

        # Record the remaining time for planning
        self.times.append(self.dt - t1 - 0.005)

        # Plan the next action using the planner
        action, info = self.planner.plan(self.s0, self.dt - t1 - 0.005)

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
    suffix = f'{algorithm}_{exp_num}'

    # Check if the loopHandler's infos attribute contains valid data
    if None not in loopHandler.infos:
        # Extract the number of simulations from the infos and save it to a pickle file
        sim_num = [i["simulations"] for i in loopHandler.infos]
        pickle.dump(sim_num, open(f"debug/sim_num_{suffix}.pkl", 'wb'))
    else:
        # If infos contains None, initialize sim_num as an empty list
        sim_num = []

    # Save various data attributes of the loopHandler to pickle files for debugging
    pickle.dump(loopHandler.actions, open(f"debug/acts_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.trajectory, open(f"debug/trj_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.planning_states, open(f"debug/ps_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.obstacles, open(f"debug/obs_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.obstacles_pred, open(f"debug/obsPred_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.times, open(f"debug/times_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.actions_executed, open(f"debug/actions_executed_{suffix}.pkl", 'wb'))

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
        "reachGoal": loopHandler.reached_goal,
        "collision": loopHandler.collision,
        "Obscollision": loopHandler.obs_collision,
        "maxSteps": loopHandler.max_steps,
        "nSteps": loopHandler.i + 1,
        "discountedReturn": discounted_return,
        "undiscountedReturn": undiscounted_return,
        "simNum": np.mean(sim_num),
        "simNumStd": np.std(sim_num),
    }

    # Save the results to a CSV file for analysis
    df = pd.DataFrame([data])
    df.to_csv(f"debug/data_{suffix}.csv")
        

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

    gc.disable()
    print(f"Experiment: {exp_num}")
    loopHandler = LoopHandler(dt)
    process = subprocess.Popen(["../env_build/sin_env/env.x86_64"], preexec_fn=os.setpgrp)
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
        debug_plots_and_animations(loopHandler, exp_num, algorithm=algorithm)
        gc.collect()


if __name__ == '__main__':
    main()
    

