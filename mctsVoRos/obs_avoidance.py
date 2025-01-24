
from functools import partial
import gc
import os
import random
import time

from matplotlib import pyplot as plt
from MCTS_VO.bettergym.agents.planner_mcts import Mcts
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.compiled_utils import dist_to_goal, robot_dynamics
from MCTS_VO.environment_creator import create_pedestrian_env
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.measure import CircleModel, ransac
from numba import jit
import tf_transformations
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from MCTS_VO.experiment_utils import plot_frame2


RADIUS_SCALE = 3.

@jit(nopython=True, cache=True)
def set_seed(value):
    np.random.seed(value)


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    set_seed(seed_value)
    

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')
        self.dt = 0.3
        self.logger = self.get_logger()
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, 
            '/odom', 
            self.callback_odom, 
            1
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.obstacle_centers = []
        self.goal = np.array([-2.783, -0.993])
        
        self.i = 0
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
        
        self.gt_obs_pos = np.array([
            [-1.127, -0.833, 0.0, 0.0],
            [-0.92, -1.651, 0.0, 0.0],
            # [-1.82, -0.306, 0.0, 0.0]
        ])
        self.gt_obs_rad = np.array([0.100, 0.100])
        
        # (obs_pos, obs_rad)
        self.obstacles = []


        self.planner = Mcts(
            num_sim=100,
            c=10,
            environment=self.sim_env,
            computational_budget=80,
            rollout_policy=partial(
                epsilon_uniform_uniform,
                std_angle_rollout=2.84*self.dt,
                eps=0.2
            ),
            discount=0.7,
            logger=self.logger
        )

        self.initialize()
        self.i = 0
        self.infos = []
        self.times = []
        self.actions = []
        self.sim_env.gym_env.max_eudist = dist_to_goal(self.s0.goal, self.s0.x[:2])
        self.robot_position = None
        self.heading = None
        self.last_action = np.array([0., self.s0.x[2]])
        self.max_obs_vel = 0.0
        self.time = 0
        self.obstacles = []
    
    def get_odom(self):
        # read odometry pose from self.odom_msg (for domuentation check http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)
        point = self.odom_msg.pose.pose.position
        rot = self.odom_msg.pose.pose.orientation
        self.rot_ = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
        heading = ((self.rot_[2]) + np.pi) % (2 * np.pi) - np.pi
        return np.array([point.x, point.y]), heading
    
    def callback_odom(self, msg):
        self.odom_msg = msg
        self.robot_position, self.heading = self.get_odom()
    
    def initialize(self):
        # state, dist, angles, pos = self.get_state()
        state = np.array([0.073, -1.136, -3.14, 0.0])
        self.s0, _ = self.sim_env.reset()
        self.s0.goal = self.goal - [0.073, -1.136]
        obs = (
            self.gt_obs_pos,
            self.gt_obs_rad
        )
        self.s0.obstacles = obs
        self.s0.x = state
        self.trajectory = np.empty((0, 4))
        self.planner.plan(self.s0, 0.2)

        self.cmd_vel_pub.publish(Twist())
    
    
    def scan_callback(self, msg):
        if self.robot_position is None:
            return 
        points = self.lidar_to_points(msg)
        clusters = self.cluster_points(points)
        initial_time = time.time()
        self.estimate_obstacle_centers(clusters)
        curr_time = time.time() - initial_time
        
        self.logger.info(str(self.robot_position))
        self.logger.info(str(self.s0.obstacles[0]))
        
        self.publish_velocity_command(curr_time)
        
        
    def lidar_to_points(self, scan):
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        points = np.array([(r * np.cos(a), r * np.sin(a)) for r, a in zip(scan.ranges, angles) if r < scan.range_max])
        R = np.array([[np.cos(self.heading), -np.sin(self.heading)], 
                      [np.sin(self.heading), np.cos(self.heading)]])
        points = np.array([R @ p for p in points])
        points = points + self.robot_position
        return points
    
    def cluster_points(self, points):
        # clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
        clustering = DBSCAN(eps=0.1, min_samples=3, n_jobs=-1).fit(points)
        clusters = [points[clustering.labels_ == i] for i in range(max(clustering.labels_) + 1)]
        return clusters
    
    def estimate_obstacle_centers(self, clusters):
        obs_pos = np.empty((0, 2))
        obs_rad = np.array([])
        for cluster in clusters:
            if len(cluster) >= 4:
                ransac_model, _ = ransac(cluster, CircleModel, min_samples=4, residual_threshold=0.05, rng=0)
                center = ransac_model.params[0:2]
                radius = ransac_model.params[2]
                radius *= RADIUS_SCALE
                obs_pos = np.vstack((obs_pos, center))
                obs_rad = np.append(obs_rad, radius)
        
        obs_pos = np.hstack((obs_pos, np.tile([0, self.max_obs_vel], (len(obs_pos), 1))))
        self.s0.obstacles = (obs_pos, obs_rad)
    
    def publish_velocity_command(self, used_time):
        self.i += 1
        
        if self.i == 50:
            raise Exception("Finished")

        
        twist = Twist()
        # Implement your planner logic here to set twist.linear.x and twist.angular.z
        self.trajectory = np.vstack((self.trajectory, self.s0.x))
        self.obstacles.append(self.s0.obstacles)
        position, heading = self.robot_position, self.heading
        robot_state = np.array([position[0], position[1], heading, self.s0.x[3]])
        self.s0.x = robot_dynamics(
            state_x=robot_state,
            u=self.last_action,
            dt=self.dt
        )
        seed_everything(0)
        action, info = self.planner.plan(self.s0, self.dt-used_time-0.012)
        self.last_action = action
        d_theta = (action[1] - self.s0.x[2] + np.pi) % (2 * np.pi) - np.pi
        omega = d_theta/self.dt
        twist.linear.x = action[0]
        twist.angular.z = omega
        
        curr_time = time.time()
        self.logger.info(f"Time MOVE: {curr_time-self.time}")
        self.time = curr_time
        self.cmd_vel_pub.publish(twist)
        
        
        
def main(args=None):
    gc.disable()

    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame2,
            fargs=(node.goal, node.config, node.obstacles, node.trajectory, ax, (node.gt_obs_pos, node.gt_obs_rad)),
            frames=tqdm(range(len(node.trajectory)), file=sys.stdout),
            save_count=None,
            cache_frame_data=False,
            interval=1000
        )
        ani.save(f"debug/trajectory.gif")
        plt.close(fig)
        node.destroy_node()
        rclpy.shutdown()


    
if __name__ == '__main__':
    main()
 