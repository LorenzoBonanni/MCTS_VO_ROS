from functools import partial
import gc
import os
import pickle
import random
import threading
import rclpy
import rclpy.qos
import numpy as np
import time

from MCTS_VO.bettergym.agents.planner_mcts import Mcts
from MCTS_VO.bettergym.agents.utils.utils import epsilon_uniform_uniform
from MCTS_VO.bettergym.compiled_utils import check_coll_vectorized, dist_to_goal, robot_dynamics
from MCTS_VO.environment_creator import create_pedestrian_env
from sensorsHandler import SensorHandler
from geometry_msgs.msg import Twist
from rclpy.node import Node
from numba import jit
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


MAX_STEPS = 50

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


# class RolloutPlanner:
#     def __init__(self, rollout_policy, environment):
#         self.rollout_policy = rollout_policy
#         self.environment = environment

#     def plan(self, state):
#         return self.rollout_policy(RolloutStateNode(state), self), None

class LoopHandler(Node):

    def __init__(self, dt):
        super().__init__('loopHandler')
        self.dt = dt
        self.logger = self.get_logger()
        self.sensorHandler = SensorHandler(dt, self.logger)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # Subscribers for obstacles and position
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
        self.gt_obs_rad = np.array([0.1, 0.1])
        
        # (obs_pos, obs_rad)
        self.obstacles = []


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
            discount=0.7,
            logger=self.logger
        )

        self.initialize()
        self.i = 0
        self.infos = []
        self.times = []
        self.actions = []
        self.sim_env.gym_env.max_eudist = dist_to_goal(self.s0.goal, self.s0.x[:2])
        self.timer = self.create_timer(0., self.control_loop)
        self.logger.info('Loop Handler initialized')
        self.time = 0
        self.obs_pos = None
        self.obs_rad = None
        
        self.odom_subscriber = self.create_subscription(
            Odometry, 
            'odom', 
            self.sensorHandler.callback_odom, 
            1
        )
        self.lidar_subscriber = self.create_subscription(
            LaserScan, 
            'scan', 
            self.sensorHandler.callback_lidar, 
            rclpy.qos.qos_profile_sensor_data
        )
        
        seed_everything(0)

    
    def initialize(self):
        # state, dist, angles, pos = self.get_state()
        state = np.array([0.22631, -0.9885721, -3.14, 0.0])
        self.s0, _ = self.sim_env.reset()
        self.s0.goal = self.goal
        obs = (
            self.gt_obs_pos,
            self.gt_obs_rad
        )
        self.s0.obstacles = obs
        self.s0.x = state
        self.trajectory = np.empty((0, 4))
        self.planner.plan(self.s0)

        self.pub.publish(Twist())
    
    def callback_state(self, msg):
        self.logger.info("Callback State")
        self.robot_position = msg.position
        self.heading = msg.heading
    
    def callback_obs(self, msg):
        self.logger.info("Callback Obs")
        self.obs_pos = np.array(msg.positions).reshape(msg.width, msg.height)
        self.obs_rad = np.array(msg.radii)

    def move(self, state, action, pub):
        curr_time = time.time()
        self.logger.info(f"Time MOVE: {curr_time-self.time}")
        self.time = curr_time
        
        twist = Twist()
        twist.linear.x = action[0]
        twist.linear.y = 0.0
        twist.linear.z = 0.0
                        
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        d_theta = (action[1] - state[2] + np.pi) % (2 * np.pi) - np.pi
        twist.angular.z = d_theta/self.dt
        pub.publish(twist)
        # self.logger.info(f"State: {state}")
        # self.logger.info(f"Action: {action}")
        # self.logger.info(f"Linear: {twist.linear.x} Anglular: {twist.angular.z}")

        
    def control_loop(self):
        initial_time = time.time()

        if self.sensorHandler.obs_pos is None or self.sensorHandler.robot_position is None:
            return
        
        if self.i == 0:
            # self.s0.obstacles = (self.sensorHandler.obs_pos, self.sensorHandler.obs_rad)
            self.s0.obstacles = (self.gt_obs_pos, self.gt_obs_rad)  
            
        self.logger.info(f"Step {self.i}")

        self.trajectory = np.vstack((self.trajectory, self.s0.x))

        # self.logger.info(f"State: {self.s0.x}")
        # self.logger.info(f"RANDOM: {random.getstate()}\n\n")
        # self.logger.info(f"NP: {np.random.get_state()}\n\n")
        # self.logger.info("Starting Planning")
        action, info = self.planner.plan(self.s0)
        
        t = time.time() - initial_time
        # self.logger.info(f"Action: {action}")
        self.logger.info(f"Time Elapsed: {t}")
        self.move(self.s0.x, action, self.pub)
        # time.sleep(self.dt - t)
        
        # DEBUG STUFF
        # self.actions.append(action)
        # self.infos.append(info)
        # END DEBUG STUFF
        
        # FORECAST NEXT STATE BASED ON ODOMETRY AND ACTION
        # position, heading = self.sensorHandler.robot_position, self.sensorHandler.heading
        
        position = self.s0.x[:2]
        heading = self.s0.x[2]
        robot_state = np.array([position[0], position[1], heading, action[0]])
        
        self.s0.x = robot_dynamics(
            state_x=robot_state,
            u=action,
            dt=self.dt
        )
        # self.s0.obstacles = (self.sensorHandler.obs_pos, self.sensorHandler.obs_rad)
        self.s0.obstacles = (self.gt_obs_pos, self.gt_obs_rad)
        self.obstacles.append(self.s0.obstacles)
        self.times.append(t)
        
        self.i += 1
        d = dist_to_goal(self.s0.goal, self.s0.x[:2])
        collision = check_coll_vectorized(self.s0.x[:2], self.gt_obs_pos[:, :2], 0.2, self.s0.obstacles[1])
        if self.i == MAX_STEPS or d<=0.3 or collision:
            self.logger.info(f"Goal Reached: {d<=0.3} Collision: {collision}")
            pickle.dump(self.actions, open("debug/acts.pkl", 'wb'))
            pickle.dump(self.trajectory, open("debug/trj.pkl", 'wb'))
            # self.destroy_node()
            raise Exception("Finished")

def main(args=None):
    dt = 0.3
    gc.disable()
    rclpy.init(args=args)
    try:
        executor2 = SingleThreadedExecutor()
        executor2.add_node(LoopHandler(dt=dt))
        executor2.spin()
        
    except Exception as e:
        # debug_plots_and_animations(loopHandler)
        raise e
        # loopHandler.zero_vel()
        # rclpy.shutdown()
        # exit(0)

        


if __name__ == '__main__':
    main()
    

