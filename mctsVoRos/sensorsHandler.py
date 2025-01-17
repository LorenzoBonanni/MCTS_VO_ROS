import numpy as np
from MCTS_VO.bettergym.compiled_utils import get_points_from_lidar
from sklearn.cluster import DBSCAN
import rclpy
import tf_transformations
import pyransac3d as pyrsc
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.node import Node


RADIUS_SCALE = 2.2


class SensorHandler():
    def __init__(self, dt, logger):
        # super().__init__('SensorHandler')
        self.dt = dt
        self.clusting_algo = DBSCAN(eps=0.1, min_samples=3, n_jobs=-1)
        self.max_obs_vel = 0.
        self.robot_position = None
        self.heading = None
        self.obs_rad = None
        self.obs_pos = None
        self.lidar_msg = LaserScan()
        self.odom_msg = Odometry()
        # self.odom_subscriber = self.create_subscription(
        #     Odometry, 
        #     'odom', 
        #     self.callback_odom, 
        #     1
        # )
        # self.lidar_subscriber = self.create_subscription(
        #     LaserScan, 
        #     'scan', 
        #     self.callback_lidar, 
        #     rclpy.qos.qos_profile_sensor_data
        # )
        # self.state_pub = self.create_publisher(RobotState, 'state', 1)
        # self.obstacles_pub = self.create_publisher(Obstacles, 'obstacle', 1)
        # self.logger = self.get_logger()
        self.logger = logger
        self.logger.info('Sensor Handler initialized')
        
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

        mask = np.where(~np.isinf(scan))[0]
        scan = np.array(scan)

        distances = scan[mask.astype(int)].copy()
        angles = mask * angle_increment + angle_min

        return distances, angles
    
        
    def group_matrix(self, M, I):
        unique_indices = np.unique(I)
        return {idx: M[I == idx] for idx in unique_indices}

    
    def estimate_obstacles(self):
        dist, angles = self.get_scan()
        if len(dist) == 0:
            return np.empty((0, 4)), np.array([])
        pos = self.robot_position
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
        self.logger.info('Lidar callback')
        if self.robot_position is None:
            return
            
        self.SetLaser(msg)
        self.obs_pos, self.obs_rad = self.estimate_obstacles()
        
        # obs_msg = Obstacles(
        #     positions=obs_pos.flatten(),
        #     width=obs_pos.shape[0],
        #     height=obs_pos.shape[1],
        #     radii=obs_rad
        # )
        # self.logger.info(f'Obstacles: {obs_pos.shape[0]}')
        # self.obstacles_pub.publish(obs_msg)

    def callback_odom(self, msg):
        self.logger.info('Odom callback')
        self.SetOdom(msg)
        self.robot_position, self.heading = self.get_odom()
        # self.state_pub.publish(
        #     RobotState(
        #         position=robot_position,
        #         heading=heading
        #     )
        # )