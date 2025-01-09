from functools import partial
import os
import pickle
import random

from MCTS_VO.bettergym.agents.planner_mcts import Mcts, RolloutStateNode
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.compiled_utils import dist_to_goal, get_points_from_lidar, robot_dynamics
from MCTS_VO.environment_creator import create_pedestrian_env
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
from sklearn.cluster import DBSCAN
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from debug_utils import debug_plots_and_animations
import pyransac3d as pyrsc

MAX_STEPS = 300


# X python = Unity Z
# Z python = Unity Y 
# Y python = Unity -X
@jit(nopython=True, cache=True)
def set_seed(value):
    np.random.seed(value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    set_seed(seed_value)

def group_matrix(M, I):
    unique_indices = np.unique(I)
    return {idx: M[I == idx] for idx in unique_indices}

# class RolloutPlanner:
#     def __init__(self, rollout_policy, environment):
#         self.rollout_policy = rollout_policy
#         self.environment = environment

#     def plan(self, state):
#         return self.rollout_policy(RolloutStateNode(state), self), None

class LoopHandler(Node):

    def __init__(self):
        super().__init__('loopHandler')

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # Subscribers for odometry and lidar
        self.reentrant_callback_group = ReentrantCallbackGroup()
        self.odom_subscriber = self.create_subscription(
            Odometry, 
            'odom', 
            self.callback_odom, 
            1, 
            callback_group=self.reentrant_callback_group
        )
        self.lidar_subscriber = self.create_subscription(
            LaserScan, 
            'scan', 
            self.callback_lidar, 
            rclpy.qos.qos_profile_sensor_data, 
            callback_group=self.reentrant_callback_group
        )
        self.i = 0
        self.dt = 0.3
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
        self.config = self.sim_env.config
        # X python = Unity Z
        # Z python = Unity Y 
        # Y python = Unity -X
        
        obs_pos = np.array([
            [-1.127, -0.833, 0.0, 0.0],
            [-0.92, -1.651, 0.0, 0.0],
            # [-1.82, -0.306, 0.0, 0.0]
        ])
        # obs_rad = np.array([0.1, 0.1])
        self.obs_pos = None
        self.obs_rad = None
        
        # (obs_pos, obs_rad)
        self.obstacles = (np.array([]), np.array([]))

        self.planner = Mcts(
            num_sim=100,
            c=10,
            environment=self.sim_env,
            computational_budget=300,
            rollout_policy=partial(
                epsilon_uniform_uniform,
                std_angle_rollout=0.38,
                eps=0.2
            ),
            discount=0.7
        )

        self.logger = self.get_logger()
        self.turtlebot3 = TurtleBot3(self.logger, self.dt)
        self.initialize()
        self.i = 0
        self.infos = []
        self.times = []
        self.actions = []
        self.sim_env.gym_env.max_eudist = dist_to_goal(self.s0.goal, self.s0.x[:2])
        self.clusting_algo = DBSCAN(n_jobs=-1)
        self.robot_position = None
        self.max_obs_vel = 0.
        self.timer = self.create_timer(0., self.control_loop)
        


    def initialize(self):
        # state, dist, angles, pos = self.get_state()
        seed_everything(0)
        state = np.array([0.22631, -0.9885721, -3.14, 0.0])
        self.s0, _ = self.sim_env.reset()
        self.s0.goal = np.array([self.turtlebot3.goal_x, self.turtlebot3.goal_y])
        self.s0.obstacles = self.obstacles
        self.s0.x = state
        self.trajectory = np.array(self.s0.x)
        self.planner.plan(self.s0)
        self.turtlebot3.move(state, [0.0, state[2]], self.pub)
    
    def estimate_obstacles(self):
        dist, angles = self.turtlebot3.get_scan()
        pos, _ = self.turtlebot3.get_odom()
        points = get_points_from_lidar(dist, angles, pos)
        clusters = self.clusting_algo.fit_predict(points)
        groups = group_matrix(points, clusters)
        obs_pos = np.empty((0, 4))
        obs_rad = np.array([])
        for group in groups.values():
            sph = pyrsc.Circle()
            center, _ , radius, _ = sph.fit(group)
            obs_pos = np.vstack((obs_pos, np.array([center[0], center[1], 0.0, self.max_obs_vel])))
            obs_rad = np.append(obs_rad, radius)
        
        self.s0.obstacles = (obs_pos, obs_rad)
        self.obs_pos = obs_pos
        self.obs_rad = obs_rad
    
    def callback_lidar(self, msg):
        self.turtlebot3.SetLaser(msg)
        self.estimate_obstacles()

    def callback_odom(self, msg):
        self.turtlebot3.SetOdom(msg)
    
    def get_state(self):
        pos, heading = self.turtlebot3.get_odom()
        
        dist, angles = self.turtlebot3.get_scan()
        # x, y, angle ,vel_lin
        state = np.array([pos[0], pos[1], heading, 0.0])

        return state, np.array(dist), np.array(angles)

    def zero_vel(self):
        twist = Twist() 
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.pub.publish(twist)
        
    def control_loop(self):
        while self.obs_pos is None and self.robot_position is None:
            time.sleep(0.2)
        self.logger.info(f"Step {self.i}")
        position, heading = self.turtlebot3.get_odom()

        initial_time = time.time()
        action, info = self.planner.plan(self.s0)
        t = time.time() - initial_time
        self.logger.info(f"Time Elapsed: {t}")
        time.sleep(self.dt - t)
        
        # DEBUG STUFF
        self.actions.append(action)
        self.infos.append(info)
        self.trajectory = np.vstack((self.trajectory, self.s0.x))
        # END DEBUG STUFF
        
        # FORECAST NEXT STATE BASED ON ODOMETRY AND ACTION
        self.turtlebot3.move(self.s0.x, action, self.pub)
        position, heading = self.turtlebot3.get_odom()
        robot_state = np.array([position[0], position[1], heading, action[0]])
        self.s0.x = robot_dynamics(
            state_x=robot_state,
            u=action,
            dt=self.dt
        )
        self.times.append(t)
        self.i += 1
        d = dist_to_goal(self.s0.goal, self.s0.x[:2])
        if self.i == MAX_STEPS or d<=0.3:
            pickle.dump(self.actions, open("debug/acts.pkl", 'wb'))
            raise Exception
        

def main(args=None):
    rclpy.init(args=args)
    try:
        loopHandler = LoopHandler()
        executor = MultiThreadedExecutor(num_threads=4)  # Adjust the number of threads as needed
        executor.add_node(loopHandler)
        executor.spin()

    except Exception as e:
        raise e
        debug_plots_and_animations(loopHandler)
        # loopHandler.zero_vel()
        # rclpy.shutdown()
        # exit(0)

        


if __name__ == '__main__':
    main()