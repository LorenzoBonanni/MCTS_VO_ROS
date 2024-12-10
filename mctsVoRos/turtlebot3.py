import math
import time

import numpy as np


from nav_msgs.msg import Odometry
import rclpy
from geometry_msgs.msg import Twist, Point
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
import rclpy.qos
import tf_transformations
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from math import pi

class TurtleBot3():

    def __init__(self):
        
        
        #qos = QoSProfile(depth=10)
        # self.node = rclpy.create_node('turtlebot3_DDQN_node')
        self.lidar_msg = LaserScan()
        self.odom_msg = Odometry()
        # set your desired goal: 
        self.goal_x, self.goal_y = -0.777, 1.176 # this is for simulation change for real robot

        # linear velocity is costant set your value
        self.linear_velocity = 0.2  # to comment
        
        # ang_vel is in rad/s, so we rotate 5 deg/s [action0, action1, action2]
        self.angular_velocity = [0.0, -2.20, 2.20]# to comment 


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
        
        return point, self.rot_[2]

    def get_scan(self):
        distances = []
        angles = []
        scan = self.lidar_msg.ranges
        angle_min = self.lidar_msg.angle_min
        angle_increment = self.lidar_msg.angle_increment
        # read lidar msg from self.lidar_msg and save in scan variable

        for i in range(len(scan)):       # cast limit values (0, Inf) to usable floats
            if scan[i] == float('Inf'):
                scan[i] = 3.5
            elif math.isnan(scan[i]):
                scan[i] = 0
            
            distances.append(scan[i])
            angle_val = angle_min + i * angle_increment
            if scan[i] < 0:
                angle_val -= 6.28
            angles.append(angle_val)

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
        
    def move(self, action, pub):
        # stop robot
        if action == -1:
            pub.publish(Twist())
        else:
            # check action 0: move forward 1: turn left 2: turn right
            # save the linear velocity in target_linear_velocity
            # save the angular velocity in target_angular_velocity
            
            twist = Twist() 
            twist.linear.x = target_linear_velocity
            twist.linear.y = 0.0
            twist.linear.z = 0.0
                            
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = target_angular_velocity

            pub.publish(twist)
