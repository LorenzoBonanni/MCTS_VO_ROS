import math

import numpy as np


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import rclpy.qos
import tf_transformations


class TurtleBot3():

    def __init__(self, logger, dt):
        self.lidar_msg = LaserScan()
        self.odom_msg = Odometry()
        # set your desired goal: 
        # X python = Unity Z
        # Z python = Unity Y 
        # Y python = Unity -X
                
        self.goal_x, self.goal_y = -2.783, -0.993 # this is for simulation change for real robot
        self.logger = logger
        self.dt = dt
        

    def SetLaser(self, msg):
        self.lidar_msg = msg

    def SetOdom(self, msg):
        self.odom_msg = msg

    def stop_tb(self):
        self.pub.publish(Twist())

    def get_odom(self):
        # read odometry pose from self.odom_msg (for domuentation check http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)
        point = self.odom_msg.pose.pose.position
        rot = self.odom_msg.pose.pose.orientation
        self.rot_ = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
        new_pos = [-point.y, point.x]
        # add constant 90 degrees to the heading
        heading = ((self.rot_[2] + 1.5708) + np.pi) % (2 * np.pi) - np.pi
        return new_pos, heading

    def get_scan(self):
        distances = []
        scan = self.lidar_msg.ranges
        angle_min = self.lidar_msg.angle_min
        angle_increment = self.lidar_msg.angle_increment
        # read lidar msg from self.lidar_msg and save in scan variable

        mask = np.where(~np.isinf(scan))[0]
        scan = np.array(scan)

        distances = np.copy(scan[mask.astype(int)])
        angles = mask * angle_increment + angle_min

        return distances, angles

        
    def move(self, state, action, pub):
        # check action 0: move forward 1: turn left 2: turn right
        # save the linear velocity in target_linear_velocity
        # save the angular velocity in target_angular_velocity
        
        twist = Twist() 
        twist.linear.x = action[0]
        twist.linear.y = 0.0
        twist.linear.z = 0.0
                        
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = ((action[1] - state[2]) / self.dt + np.pi) % (2 * np.pi) - np.pi
        pub.publish(twist)
        self.logger.info(f"Linear: {twist.linear.x} Anglular: {twist.angular.z}")