import math

import numpy as np


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import rclpy.qos
import tf_transformations


class TurtleBot3():

    def __init__(self):
        
        
        #qos = QoSProfile(depth=10)
        # self.node = rclpy.create_node('turtlebot3_DDQN_node')
        self.lidar_msg = LaserScan()
        self.odom_msg = Odometry()
        # set your desired goal: 
        self.goal_x, self.goal_y = -1.795, -1.431 # this is for simulation change for real robot


        # self.r = rclpy.spin_once(self.node,timeout_sec=0.25)

        
        print("Robot initialized")

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
        heading = (self.rot_[2] + np.pi) % (2 * np.pi) - np.pi
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

    def get_goal_info(self, tb3_pos):

        # compute distance euclidean distance use self.goal_x/y pose and tb3_pose.x/y
        # compute the heading using atan2 of delta y and x
        # subctract the actual robot rotation to heading
        # save in distance and heading the value
        

        
        # we round the distance dividing by 2.8 under the assumption that the max distance between 
        # two points in the environment is approximately 3.3 meters, e.g. 3m
        # return heading in deg
        return distance/2.8, np.rad2deg(heading) / 180     
        
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
        twist.angular.z = action[1] - state[2] / 1.0
        print(f"Action: {twist.linear.x}, {twist.angular.z}")
        pub.publish(twist)