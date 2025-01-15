from functools import partial
import os
import pickle
import random
import threading
from debug_utils import debug_plots_and_animations
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
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from mcts_vo_ros_msgs.msg import Obstacles, RobotState
from rclpy.node import Node
from numba import jit
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

MAX_STEPS = 10

# X python = Unity Z
# Z python = Unity Y 
# Y python = Unity -X
@jit(nopython=True, cache=True)
def set_seed(value):
    np.random.seed(value)

def get_state(logger):
    logger.info(f"RANDOM: {random.getstate()}\n\n")
    logger.info(f"NP: {np.random.get_state()}\n\n")

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

        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        # Subscribers for obstacles and position
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.state_sub = self.create_subscription(
            RobotState, 
            'state', 
            self.callback_state, 
            1, 
            callback_group=self.reentrant_callback_group
        )
        self.lidar_subscriber = self.create_subscription(
            Obstacles, 
            'obstacle', 
            self.callback_obs, 
            1, 
            callback_group=self.reentrant_callback_group
        )
        
        self.i = 0
        self.dt = dt
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

        self.logger = self.get_logger()

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
        self.robot_position = None
        self.heading = None
        self.obs_pos = None
        self.obs_rad = None
        self.timer = self.create_timer(0., self.control_loop)
        self.logger.info('Loop Handler initialized')
    
    def callback_state(self, msg):
        self.robot_position = msg.position
        self.heading = msg.heading
    
    def callback_obs(self, msg):
        self.obs_pos = np.array(msg.positions).reshape(msg.width, msg.height)
        self.obs_rad = np.array(msg.radii)

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
        d_theta = (action[1] - state[2] + np.pi) % (2 * np.pi) - np.pi
        twist.angular.z = d_theta/self.dt
        pub.publish(twist)
        self.logger.info(f"Linear: {twist.linear.x} Anglular: {twist.angular.z}")

    def initialize(self):
        # state, dist, angles, pos = self.get_state()
        seed_everything(0)
        state = np.array([0.22631, -0.9885721, -3.14, 0.0])
        self.s0, _ = self.sim_env.reset()
        self.s0.goal = np.array([-2.783, -0.993])
        obs = (
            np.array([
                [-1.127, -0.833, 0.0, 0.0],
                [-0.92, -1.651, 0.0, 0.0],
                # [-1.82, -0.306, 0.0, 0.0]
            ]), 
            np.array([0.1, 0.1])
        )
        self.s0.obstacles = obs
        self.s0.x = state
        self.trajectory = np.empty((0, 4))
        self.planner.plan(self.s0)
        self.pub.publish(Twist())

        
    def control_loop(self):
        if self.obs_pos is None or self.robot_position is None:
            return
        self.logger.info(f"Step {self.i}")

        self.trajectory = np.vstack((self.trajectory, self.s0.x))
        self.s0.obstacles = (self.obs_pos, self.obs_rad)
        self.obstacles.append(self.s0.obstacles)

        initial_time = time.time()
        self.logger.info(f"State: {self.s0.x}")
        self.logger.info(f"Obstacles: {self.s0.obstacles}")
        # get_state(self.logger)
        action, info = self.planner.plan(self.s0)
        t = time.time() - initial_time
        self.logger.info(f"Time Elapsed: {t}")
        time.sleep(self.dt - t)
        
        # DEBUG STUFF
        self.actions.append(action)
        self.infos.append(info)
        # END DEBUG STUFF
        
        # FORECAST NEXT STATE BASED ON ODOMETRY AND ACTION
        position, heading = self.robot_position, self.heading
        robot_state = np.array([position[0], position[1], heading, action[0]])

        self.move(robot_state, action, self.pub)

        self.s0.x = robot_dynamics(
            state_x=robot_state,
            u=action,
            dt=self.dt
        )
        self.times.append(t)
        self.i += 1
        d = dist_to_goal(self.s0.goal, self.s0.x[:2])
        collision = check_coll_vectorized(self.s0.x[:2], self.gt_obs_pos[:, :2], 0.3, self.s0.obstacles[1])
        if self.i == MAX_STEPS or d<=0.3 or collision:
            self.logger.info(f"Goal Reached: {d<=0.3} Collision: {collision}")
            # pickle.dump(self.actions, open("debug/acts.pkl", 'wb'))
            # self.destroy_node()
            raise Exception("Goal Reached")
        

def main(args=None):
    dt = 0.3
    rclpy.init(args=args)
    try:
        loopHandler = LoopHandler(dt=dt)
        sensorHandler = SensorHandler(dt=dt)
        # executor = MultiThreadedExecutor(num_threads=6)  # Adjust the number of threads as needed
        executor = SingleThreadedExecutor()  # Adjust the number of threads as needed
        executor.add_node(sensorHandler)
        executor.add_node(loopHandler)
        # executor_thread = threading.Thread(target=executor.spin, daemon=True)
        # executor_thread.start()
        # executor_thread.join()
        executor.spin()
        
    except Exception as e:
        debug_plots_and_animations(loopHandler)
        raise e
        # loopHandler.zero_vel()
        # rclpy.shutdown()
        # exit(0)

        


if __name__ == '__main__':
    main()