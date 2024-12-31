from functools import partial
import os
import pickle
import random

from MCTS_VO.bettergym.agents.planner_mcts import Mcts, RolloutStateNode
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.agents.utils.vo import epsilon_uniform_uniform_vo
from MCTS_VO.bettergym.environments.env import State, robot_dynamics
from MCTS_VO.environment_creator import create_pedestrian_env
from MCTS_VO.experiment_utils import create_animation_tree_trajectory
from turtlebot3 import TurtleBot3
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
import rclpy.qos
import numpy as np
import time
from numba import jit

# X python = Unity Z
# Z python = Unity Y 
# Y python = Unity -X
@jit(nopython=True, cache=True)
def get_points_from_lidar(dist, angles):
    points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
    points_copy = np.empty_like(points)
    points_copy[:, 0] = points[:, 1]
    points_copy[:, 1] = -points[:, 0]
    return np.hstack((points_copy, np.zeros(points.shape[0])[:, None]))


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

class RolloutPlanner:
    def __init__(self, rollout_policy, environment):
        self.rollout_policy = rollout_policy
        self.environment = environment

    def plan(self, state):
        return self.rollout_policy(RolloutStateNode(state), self), None

class LoopHandler(Node):

    def __init__(self):
        super().__init__('loopHandler')

        
        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.callback_lidar, rclpy.qos.qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.callback_odom, 1)
        self.i = 0
        self.verbose = True
        
        # number of attempts, the network could be fail if you set a number greater than 1 the robot try again to reach the goal
        self.n_episode_test = 1
        self.dt = 0.2
        # TODO parametrize
        _, self.sim_env = create_pedestrian_env(
            discrete=True,
            rwrd_in_sim=True,
            out_boundaries_rwrd=-100,
            n_vel=5,
            n_angles=11,
            vo=True,
            obs_pos=None,
            n_obs=None,
            dt_real=self.dt,
        )
        self.s0 = None
        self.config = self.sim_env.config
        # X python = Unity Z
        # Z python = Unity Y 
        # Y python = Unity -X
        self.obstacles = [
            [-1.127, -0.833, 0.0, 0.0],
        ]

        self.obstacles = [
            State(
                x=np.array(self.obstacles[i]),
                goal=None,
                obs_type='circle',
                radius=0.1,
                obstacles=None
            )
            for i in range(len(self.obstacles))
        ]
        self.planner = Mcts(
            num_sim=100,
            c=10,
            environment=self.sim_env,
            computational_budget=200,
            rollout_policy=partial(
                epsilon_uniform_uniform,
                std_angle_rollout=2.0,
                eps=0.2
            ),
            discount=0.95
        )
        # self.planner = RolloutPlanner(
        #     partial(
        #         epsilon_uniform_uniform_vo,
        #         std_angle_rollout=1.0,
        #         eps=0.2
        #     ),
        #     self.sim_env
        # )
        self.logger = self.get_logger()
        self.turtlebot3 = TurtleBot3(self.logger, self.dt)
        self.initialize()
        self.i = 0
        self.infos = []

        self.timer = self.create_timer(0.25, self.control_loop)


    def initialize(self):
        # state, dist, angles, pos = self.get_state()
        seed_everything(42)
        state = np.array([0.22631, -0.9885721, -3.14, 0.0])
        self.s0, _ = self.sim_env.reset()
        self.s0.goal = np.array([self.turtlebot3.goal_x, self.turtlebot3.goal_y])
        self.trajectory = np.array(self.s0.x)
        self.s0.obstacles = self.obstacles
        self.s0.x = state
        self.planner.plan(self.s0)
        self.turtlebot3.move(state, [0.0, state[2]], self.pub)

        
    
    def callback_lidar(self, msg):
        self.turtlebot3.SetLaser(msg)

    def callback_odom(self, msg):
        self.turtlebot3.SetOdom(msg)

    def get_state(self):
        pos, heading = self.turtlebot3.get_odom()
        
        dist, angles = self.turtlebot3.get_scan()        
        # x, y, angle ,vel_lin
        state = np.array([pos[0], pos[1], heading, 0.0])

        return state, np.array(dist), np.array(angles), pos

    # def zero_vel(self):
    #     twist = Twist() 
    #     twist.linear.x = 0.0
    #     twist.linear.y = 0.0
    #     twist.linear.z = 0.0
        
    #     twist.angular.x = 0.0
    #     twist.angular.y = 0.0
    #     twist.angular.z = 0.0
    #     self.pub.publish(twist)
        
    # def one_vel(self):
    #     twist = Twist() 
    #     twist.linear.x = 1.0
    #     twist.linear.y = 0.0
    #     twist.linear.z = 0.0
        
    #     twist.angular.x = 0.0
    #     twist.angular.y = 0.0
    #     twist.angular.z = 0.0
    #     self.pub.publish(twist)

    def control_loop(self):
        initial_time = time.time()        
        # points = get_points_from_lidar(dist, angles)
        # sph = pyrsc.Circle()
        # center, axis, radius, inliers = sph.fit(points)
        action, info = self.planner.plan(self.s0)
        self.turtlebot3.move(self.s0.x, action, self.pub)
        self.infos.append(info)

        self.s0.x = robot_dynamics(
            state_x=self.s0.x,
            u=action,
            dt=self.dt
        )
        self.logger.info(f"Time elapsed: {time.time() - initial_time}")
        self.i += 1
        if self.i == 300:
            raise Exception
        




def main(args=None):
    rclpy.init(args=args)
    try:
        loopHandler = LoopHandler()
        rclpy.spin(loopHandler)
    except Exception as e:
        print(e)
        print("Creating animation")
        infos = loopHandler.infos
        trajectories = [i["trajectories"] for i in infos]
        rollout_values = [i["rollout_values"] for i in infos]
        obs = [loopHandler.obstacles for _ in range(len(infos))]
        create_animation_tree_trajectory(
            [loopHandler.turtlebot3.goal_x, loopHandler.turtlebot3.goal_y], 
            loopHandler.config, 
            obs, 
            0, 
            'test', 
            rollout_values, 
            trajectories
        )
        
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    


    

