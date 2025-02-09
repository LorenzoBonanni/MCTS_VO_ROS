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

from sklearn.cluster import DBSCAN, HDBSCAN
from debug_utils import debug_plots_and_animations
from MCTS_VO.bettergym.agents.planner_mcts import Mcts, RolloutStateNode
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.compiled_utils import check_coll_vectorized, dist_to_goal, get_points_from_lidar, predict_obstacles, robot_dynamics
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
parser.add_argument('--algorithm', default='VO-TREE', type=str)

MAX_STEPS = 250
RADIUS_SCALE = 3
DISCOUNT = 0.9
DEPTH = 200
dt = 0.3

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
        # self.sensorHandler = SensorHandler(dt, self.logger)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # Subscribers for obstacles and position
        self.goal = np.array([-2.783, -0.72])
        
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
                    std_angle_rollout=1.84*self.dt,
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
        self.sim_env.gym_env.max_eudist = dist_to_goal(self.s0.goal, self.s0.x[:2])
        self.t_timer = 0.3
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

    
    def SetLaser(self, msg):
        self.lidar_msg = msg

    def SetOdom(self, msg):
        self.odom_msg = msg
        
    def get_odom(self):
        # read odometry pose from self.odom_msg (for domuentation check http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)
        point = self.odom_msg.pose.pose.position
        rot = self.odom_msg.pose.pose.orientation
        self.rot_ = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
        heading = ((self.rot_[2]) + np.pi) % (2 * np.pi) - np.pi
        return np.array([point.x, point.y]), heading

    def get_scan(self):
        distances = []
        scan = self.lidar_msg.ranges
        angle_min = self.lidar_msg.angle_min
        angle_increment = self.lidar_msg.angle_increment

        mask = np.where(~np.logical_or(np.isnan(scan), np.isinf(scan)))[0]
        # mask = np.where(~np.isinf(scan))[0]
        scan = np.array(scan)

        distances = scan[mask.astype(int)].copy()
        angles = mask * angle_increment + angle_min
        
        return distances, angles
    
        
    def group_matrix(self, M, I):
        unique_indices = np.unique(I)
        return {idx: M[I == idx] for idx in unique_indices}

    
    def estimate_obstacles(self, pos, heading, dist, angles):
        # dist, angles = self.get_scan()
        
        if len(dist) == 0:
            return np.empty((0, 4)), np.array([])

        points = get_points_from_lidar(dist, angles, pos, heading)
        self.points_list.append(points)


        # self.logger.info(f'Points: {points}')
        clusters = self.clusting_algo.fit_predict(points)
        groups = self.group_matrix(points, clusters)
        obs_pos = np.empty((0, 2))
        obs_rad = np.array([])
        for group in groups.values():
            if len(group) < 3:
                continue
            
            ransac_model, _ = ransac(group, CircleModel, max_trials=100, min_samples=3, residual_threshold=0.1, stop_probability=0.99)
            if ransac_model is None:
                continue
            center = ransac_model.params[0:2]
            radius = ransac_model.params[2]
            radius *= RADIUS_SCALE
            obs_pos = np.vstack((obs_pos, center))
            obs_rad = np.append(obs_rad, radius)

        mask = obs_rad <= 0.5
        obs_rad = obs_rad[mask]
        obs_pos = obs_pos[mask]
        obs_pos = np.hstack((obs_pos, np.tile([0, self.max_obs_vel], (len(obs_pos), 1))))
        return obs_pos, obs_rad
    
    def callback_lidar(self, msg):
        if self.robot_position is None:
            return
        
        if self.lidar_msg is not None:
            self.prev_lidar = deepcopy(self.lidar_msg)
            self.update_lidar = self.prev_lidar.ranges != msg.ranges
            
        self.SetLaser(msg)


    def callback_odom(self, msg):
        if self.odom_msg is not None:
            self.prev_odom = deepcopy(self.odom_msg)
            self.update_odom = self.prev_odom.pose.pose.position != msg.pose.pose.position

        self.SetOdom(msg)
        self.robot_position, self.heading = self.get_odom()

    def initialize(self):
        # state, dist, angles, pos = self.get_state()
        state = np.array([0.073, -1.136, -3.14, 0.0])
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
        curr_time = time.time()
        # self.logger.info(f"Time MOVE: {curr_time-self.time}")
        self.time = curr_time
        
        twist = Twist()
        twist.linear.x = action[0]
        twist.linear.y = 0.0
        twist.linear.z = 0.0
                        
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        d_theta = (action[1] - state[2] + np.pi) % (2 * np.pi) - np.pi
        twist.angular.z = d_theta/self.dt
        pub.publish(twist)
        # self.logger.info(f"State: {state}")
        # self.logger.info(f"Action: {action}")
        # self.logger.info(f"Linear: {twist.linear.x} Anglular: {twist.angular.z}")

        
    def control_loop(self):
        if not self.update_odom or not self.update_lidar:
            return
        
        self.logger.info(f"Step: {self.i}")
        position, heading = self.robot_position.copy(), deepcopy(self.heading)
        dist, angles = self.get_scan()
        dist = dist.copy()
        angles = angles.copy()
        self.move(self.s0.x, self.last_action, self.pub)

        robot_state = np.array([position[0], position[1], heading, self.s0.x[3]])
        self.s0.x = robot_state
        self.trajectory = np.vstack((self.trajectory, self.s0.x))
        start_time = time.time()
        seed_everything(0)
        self.obs_pos, self.obs_rad = self.estimate_obstacles(position, heading, dist, angles)
        self.s0.obstacles = (self.obs_pos, self.obs_rad)
        self.obstacles.append(self.s0.obstacles)

        d = dist_to_goal(self.s0.goal, position)
        # collision = False
        self.reached_goal = d <= 0.2
        self.collision = min(dist) <= self.config.robot_radius
        if self.i == MAX_STEPS or self.reached_goal or self.collision:
            self.pub.publish(Twist())
            self.obs_collision = self.collision and self.last_action[0] == 0
            self.collision = self.collision and self.last_action[0] != 0
            self.max_steps = self.i == MAX_STEPS
            self.logger.info(f"Goal Reached: {self.reached_goal} Collision: {self.collision}")
            raise Exception("Finished")
        
        t1 = time.time() - start_time
        seed_everything(0)
        initial_time = time.time()
        self.s0.x = robot_dynamics(
            state_x=robot_state,
            u=self.last_action,
            dt=self.t_timer
        )
        new_obs = predict_obstacles(
            robot_position=position,
            obstacles=self.s0.obstacles[0],
            dt=self.t_timer
        )
        self.s0.obstacles = (new_obs, self.s0.obstacles[1])
        self.obstacles_pred.append(self.s0.obstacles)
        self.planning_states = np.vstack((self.planning_states, self.s0.x))
        self.times.append(self.dt-t1-0.005)
        action, info = self.planner.plan(self.s0, self.dt-t1-0.005)
        self.infos.append(info)
        self.actions.append(action)
        self.last_action = action
        t2 = time.time() - initial_time
        self.update_odom = False
        self.update_lidar = False
        self.pub.publish(Twist())
        self.i += 1

    # def control_loop_vo_planner(self):
    #     if not self.update_odom or not self.update_lidar:
    #         return
        
    #     position, heading = self.robot_position.copy(), deepcopy(self.heading)
    #     dist, angles = self.get_scan()
    #     dist = dist.copy()
    #     angles = angles.copy()
    #     self.move(self.s0.x, self.last_action, self.pub)

    #     robot_state = np.array([position[0], position[1], heading, self.s0.x[3]])
    #     self.s0.x = robot_state
    #     d = dist_to_goal(self.s0.goal, position)
    #     collision = check_coll_vectorized(position, self.s0.obstacles[0], self.config.robot_radius, self.s0.obstacles[1])
    #     if self.i == MAX_STEPS or self.reached_goal or collision:
    #         self.pub.publish(Twist())
    #         self.reached_goal = d<=0.2
    #         self.collision = collision
    #         self.max_steps = self.i == MAX_STEPS
    #         self.logger.info(f"Goal Reached: {self.reached_goal} Collision: {collision}")
    #         raise Exception("Finished")
        

    #     self.trajectory = np.vstack((self.trajectory, self.s0.x))
    #     start_time = time.time()
    #     seed_everything(0)
    #     self.obs_pos, self.obs_rad = self.estimate_obstacles(position, heading, dist, angles)
    #     self.s0.obstacles = (self.obs_pos, self.obs_rad)
    #     self.obstacles.append(self.s0.obstacles)
    #     t1 = time.time() - start_time
    #     seed_everything(0)
    #     initial_time = time.time()
    #     self.s0.x = robot_dynamics(
    #         state_x=robot_state,
    #         u=self.last_action,
    #         dt=t1
    #     )
    #     action, info = self.planner.plan(self.s0, self.dt-t1-0.005)
    #     self.last_action = np.array([action[0], action[1]])
    #     t2 = time.time() - initial_time
    #     time.sleep(self.dt-t1-t2-0.015)
    #     self.update_odom = False
    #     self.update_lidar = False
    #     self.pub.publish(Twist())
    #     self.i += 1

def save_data(loopHandler, exp_num):
    suffix = f'{algorithm}_{exp_num}'
    pickle.dump(loopHandler.actions, open(f"debug/acts_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.trajectory, open(f"debug/trj_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.planning_states, open(f"debug/ps_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.obstacles, open(f"debug/obs_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.obstacles_pred, open(f"debug/obsPred_{suffix}.pkl", 'wb'))
    pickle.dump(loopHandler.times, open(f"debug/times_{suffix}.pkl", 'wb'))
    data = {
        "algorithm": algorithm,
        "reachGoal": loopHandler.reached_goal,
        "collision": loopHandler.collision,
        "Obscollision": loopHandler.obs_collision,
        "maxSteps": loopHandler.max_steps,
        "nSteps": loopHandler.i+1,
    }
    df = pd.DataFrame([data])
    df.to_csv(f"debug/data_{suffix}.csv")
        

def main(args=None):
    rclpy.init(args=args)

    gc.disable()
    print(f"Experiment: {exp_num}")
    loopHandler = LoopHandler(dt)
    process = subprocess.Popen(["../env_build/env.x86_64"], preexec_fn=os.setpgrp)
    time.sleep(2)
    try:
        executor = SingleThreadedExecutor()
        executor.add_node(loopHandler)
        executor.spin()
    except Exception as e:
        # raise e
        loopHandler.destroy_node()
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Send the signal to all the process groups
        save_data(loopHandler, exp_num)
        debug_plots_and_animations(loopHandler, exp_num, algorithm=algorithm)
        gc.collect()
        


        


if __name__ == '__main__':
    main()
    

