from functools import partial
import os
from time import sleep

from MCTS_VO.bettergym.agents.planner_mcts import Mcts
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.environments.env import State
from MCTS_VO.environment_creator import create_pedestrian_env
from turtlebot3 import TurtleBot3
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
import rclpy.qos
import numpy as np
import time
import pyransac3d as pyrsc
from numba import njit

@njit
def get_points_from_lidar(dist, angles):
    points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
    return np.hstack((points, np.zeros(points.shape[0])[:, None]))


class LoopHandler(Node):

    def __init__(self):
        super().__init__('loopHandler')

        
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.turtlebot3 = TurtleBot3()
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.callback_lidar, rclpy.qos.qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.callback_odom, 10)

        self.verbose = True
        # self.agent = Agent(self.verbose)
        
        # number of attempts, the network could be fail if you set a number greater than 1 the robot try again to reach the goal
        self.n_episode_test = 1
        timer_period = 0.8  # 1 second
        time.sleep(1)
        self.timer = self.create_timer(timer_period, self.control_loop)
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
        )
        self.s0 = None
        self.config = self.sim_env.config
        self.obstacles = [
            [-1.602, -0.744, 0.0, 0.0],
            [-0.920, -1.433, 0.0, 0.0],
            [-0.944, -0.497, 0.0, 0.0],
            [-1.365, -1.292, 0.0, 0.0]
        ]
        self.obstacles = [
            State(
                x=np.array(self.obstacles[i]),
                goal=None,
                obs_type='circle',
                radius=0.3,
                obstacles=None
            )
            for i in range(len(self.obstacles))
        ]
        self.planner = Mcts(
            num_sim=200,
            c=10,
            environment=self.sim_env,
            computational_budget=100,
            rollout_policy=partial(
                epsilon_uniform_uniform,
                std_angle_rollout=1.0,
                eps=0.2
            ),
            discount=0.7
        )
        

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


    def control_loop(self):
        initial_time = time.time()
        state, dist, angles, pos = self.get_state()
        if self.s0 is None:
            self.s0, _ = self.sim_env.reset()
            self.s0.goal = np.array([self.turtlebot3.goal_x, self.turtlebot3.goal_y])
            self.trajectory = np.array(self.s0.x)
            self.s0.obstacles = self.obstacles
        
        self.s0.x = state
        # points = get_points_from_lidar(dist, angles)
        # sph = pyrsc.Circle()
        # center, axis, radius, inliers = sph.fit(points)

        action, _ = self.planner.plan(self.s0)
        time_elapsed = time.time() - initial_time
        print(f"Time elapsed: {time_elapsed}")
        self.turtlebot3.move(state, action, self.pub)




def main(args=None):
    rclpy.init(args=args)

    loopHandler = LoopHandler()
    
    rclpy.spin(loopHandler)

    loopHandler.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    


    

