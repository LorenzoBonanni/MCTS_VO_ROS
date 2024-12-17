from functools import partial
import os
import random

from MCTS_VO.bettergym.agents.planner_mcts import Mcts, RolloutStateNode
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.agents.utils.vo import epsilon_uniform_uniform_vo
from MCTS_VO.bettergym.environments.env import State
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
from numba import njit

@njit
def get_points_from_lidar(dist, angles):
    points = dist[:, None] * np.vstack((np.cos(angles), np.sin(angles))).transpose()
    return np.hstack((points, np.zeros(points.shape[0])[:, None]))


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
        self.turtlebot3 = TurtleBot3()
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.callback_lidar, rclpy.qos.qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.callback_odom, 1)

        self.verbose = True
        # self.agent = Agent(self.verbose)
        
        # number of attempts, the network could be fail if you set a number greater than 1 the robot try again to reach the goal
        self.n_episode_test = 1
        timer_period = 1.0
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
            [-1.82, -0.306, 0.0, 0.0],
            [-0.92, -1.651, 0.0, 0.0],
            [-1.127, -0.833, 0.0, 0.0],
            [-1.724, -1.647, 0.0, 0.0]
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
        # self.planner = Mcts(
        #     num_sim=100,
        #     c=10,
        #     environment=self.sim_env,
        #     computational_budget=100,
        #     rollout_policy=partial(
        #         epsilon_uniform_uniform,
        #         std_angle_rollout=1.0,
        #         eps=0.2
        #     ),
        #     discount=0.95
        # )
        self.planner = RolloutPlanner(
            partial(
                epsilon_uniform_uniform_vo,
                std_angle_rollout=1.0,
                eps=0.2
            ),
            self.sim_env
        )
        self.initialize()
        self.i = 0
        self.infos = []
        self.timer = self.create_timer(timer_period, self.control_loop)


    def initialize(self):
        state, dist, angles, pos = self.get_state()
        self.s0, _ = self.sim_env.reset()
        self.s0.goal = np.array([self.turtlebot3.goal_x, self.turtlebot3.goal_y])
        self.trajectory = np.array(self.s0.x)
        self.s0.obstacles = self.obstacles
        self.s0.x = state
        self.planner.plan(self.s0)
        self.turtlebot3.move(state, [0.3, state[2]], self.pub)
        
    
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
        pass
        # initial_time = time.time()
        # # print(self.i)
        # seed_everything(42)
        # state, dist, angles, pos = self.get_state()

        # self.s0.x = state
        # # points = get_points_from_lidar(dist, angles)
        # # sph = pyrsc.Circle()
        # # center, axis, radius, inliers = sph.fit(points)
        # print(f"State: {state}")
        # action, info = self.planner.plan(self.s0)
        # # print(f"Action: {action}")
        # print(f"Time elapsed: {time.time() - initial_time}")

        
        # self.turtlebot3.move(state, action, self.pub)
        # self.i+=1
        # self.infos.append(info)
        # # if self.i == 20:
        # #     # stop node and exit
        # #     self.destroy_node()
        # #     raise Exception




def main(args=None):
    rclpy.init(args=args)
    try:
        loopHandler = LoopHandler()
        rclpy.spin(loopHandler)
    except Exception:
        print("Creating animation")
        # infos = loopHandler.infos
        # trajectories = [i["trajectories"] for i in infos]
        # rollout_values = [i["rollout_values"] for i in infos]
        # obs = [loopHandler.obstacles for _ in range(len(infos))]
        # create_animation_tree_trajectory(
        #     [loopHandler.turtlebot3.goal_x, loopHandler.turtlebot3.goal_y], 
        #     loopHandler.config, 
        #     obs, 
        #     0, 
        #     'test', 
        #     rollout_values, 
        #     trajectories
        # )
        
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    


    

