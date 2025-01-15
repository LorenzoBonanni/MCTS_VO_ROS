import time
import numpy as np
from MCTS_VO.bettergym.compiled_utils import get_points_from_lidar
import rclpy
from sklearn.cluster import DBSCAN
from rclpy.node import Node
from turtlebot3 import TurtleBot3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mcts_vo_ros_msgs.msg import Obstacles, RobotState
import pyransac3d as pyrsc


RADIUS_SCALE = 2.2


class SensorHandler(Node):
    def __init__(self, dt):
        super().__init__('sensorHandler')
        self.dt = dt
        self.turtlebot3 = TurtleBot3(dt)
        self.clusting_algo = DBSCAN(eps=0.1, min_samples=2, n_jobs=-1)

        self.reentrant_callback_group = MutuallyExclusiveCallbackGroup()
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
        self.state_pub = self.create_publisher(RobotState, 'state', 1)
        self.obstacles_pub = self.create_publisher(Obstacles, 'obstacle', 1)
        self.logger = self.get_logger()
        self.logger.info('Sensor Handler initialized')
        self.max_obs_vel = 0.
        self.position = None
        self.heading = None
        

        
    def group_matrix(self, M, I):
        unique_indices = np.unique(I)
        return {idx: M[I == idx] for idx in unique_indices}

    
    def estimate_obstacles(self):
        dist, angles = self.turtlebot3.get_scan()
        pos = self.position
        points = get_points_from_lidar(dist, angles, pos)
        clusters = self.clusting_algo.fit_predict(points)
        groups = self.group_matrix(points, clusters)
        obs_pos = np.empty((0, 4))
        obs_rad = np.array([])
        for group in groups.values():
            sph = pyrsc.Circle()
            center, _ , radius, _ = sph.fit(group)
            radius *= RADIUS_SCALE
            obs_pos = np.vstack((obs_pos, np.array([center[0], center[1], 0.0, self.max_obs_vel])))
            obs_rad = np.append(obs_rad, radius)
        
        
        return obs_pos, obs_rad
    
    def callback_lidar(self, msg):
        if self.position is None:
            return
            
        self.turtlebot3.SetLaser(msg)
        obs_pos, obs_rad = self.estimate_obstacles()

        obs_msg = Obstacles(
            positions=obs_pos.flatten(),
            width=obs_pos.shape[0],
            height=obs_pos.shape[1],
            radii=obs_rad
        )
        
        self.obstacles_pub.publish(obs_msg)

    def callback_odom(self, msg):
        self.turtlebot3.SetOdom(msg)
        self.position, self.heading = self.turtlebot3.get_odom()
        self.state_pub.publish(
            RobotState(
                position=self.position,
                heading=self.heading
            )
        )
        