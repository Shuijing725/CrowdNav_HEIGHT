import gym
import numpy as np
from numpy.linalg import norm
import os
import time

# prevent import error if other code is run in conda env
try:
	# import ROS related packages
	import rospy
	import tf2_ros
	from geometry_msgs.msg import Twist, TransformStamped, PoseArray, PoseStamped
	import tf
	from sensor_msgs.msg import JointState
	from threading import Lock
	from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
	import actionlib
	from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
	from nav_msgs.msg import Odometry
except:
	pass


import copy
import sys

from crowd_sim.envs.crowd_sim_tb2 import CrowdSim3DTB
from crowd_sim.envs.crowd_sim_tb2_sim2real import CrowdSim3DTB_Sim2real

class rosTurtlebot2iEnv(CrowdSim3DTB):
	'''
	Environment for testing a simulated policy on a real Turtlebot2i
	To use it, change the env_name in arguments.py in the tested model folder to 'rosTurtlebot2iEnv-v0'
	'''
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(CrowdSim3DTB, self).__init__()

		# subscriber callback function will change these two variables
		self.robotMsg=None # robot state message
		self.humanMsg=None # human state message
		self.jointMsg=None # joint state message
		self.odomMsg = None # odometry message

		self.currentTime=0.0
		self.lastTime=0.0 # store time for calculating time interval
		self.prevTime = time.time()

		self.current_human_states = None  # (px,py)
		self.detectedHumanNum=0
		# self.real_max_human_num = self.max_human_num if self.max_human_num > 0 else self.real_max_human_num = 1
		self.real_max_human_num = None

		# goal positions will be set manually in self.reset()
		self.goal_x = 0.0
		self.goal_y = 0.0

		self.last_left = 0.
		self.last_right = 0.
		self.last_w = 0.0
		self.jointVel=None

		# to calculate vx, vy
		self.last_v = 0.0
		self.desiredVelocity=[0.0,0.0]

		self.mutex = Lock()

		self.fake_pc_env = CrowdSim3DTB_Sim2real()

		self.goal_reach_dist = 0.8
		self.intrusion_timesteps = 0
		self.dist_intrusion = []

		# given a goal pose (x, y) in T265 frame (direction faces front), convert it to map frame of nav stack (translation x, y, z, rotation x, y, z, w)
		self.pose_lookup_table = {(0., 6.): [5.993, 1.488, 0.0102, 0, 0, 0.101, 0.995],
								  (0., 4.): [3.851, 0.500, 0.0102, 0, 0, 0.197, 0.980]}

		self.spatial_edges_log = []

	def configure(self, config):
		super().configure(config)

		# whether we're testing ros navigation stack (sets goal and record evaluation metrics)
		# or RL policy (
		try:
			if self.config.sim2real.test_nav_stack:
				self.publish_actions = False
			else:
				self.publish_actions = True
		except:
			self.publish_actions = True

		# increase the time limit to count for delays in real world
		if self.load_act:
			self.time_limit = (len(self.episodeRecoder.v_list)-3) * self.config.sim2real.fixed_time_interval
		else:
			self.time_limit = self.time_limit * 2.5


		self.real_max_human_num = max(self.max_human_num, 1)
		print('self.real_max_human_num', self.real_max_human_num)

		# zed or lidar
		self.human_detect_method = config.sim2real.human_detector
		print('self.human_detect_method', self.human_detect_method)

		self.robot_localizer = config.sim2real.robot_localization

		# define ob space and action space
		self.set_ob_act_space()

		# ROS
		rospy.init_node('ros_turtlebot2i_env_node', anonymous=True)

		if self.publish_actions:
			if self.config.action_space.kinematics == 'turtlebot':
				self.actionPublisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=1)
			else:
				self.actionPublisher = rospy.Publisher('/navigation/cmd_vel', Twist, queue_size=1)
		self.tfBuffer = tf2_ros.Buffer()
		self.transformListener = tf2_ros.TransformListener(self.tfBuffer)

		# ROS subscribers
		# to obtain robot left & right wheel velocity (raw, noisy)
		jointStateSub = Subscriber("/joint_states", JointState)

		# to obtain robot (v, w) velocity (smooth, filtered)
		# tb base odometry
		# odomSub = Subscriber('/odom', Odometry)
		if self.robot_localizer == 'zed':
			# zed odometry
			if self.config.action_space.kinematics == 'turtlebot':
				odomSub = Subscriber('/odom', Odometry)
			else:
				odomSub = Subscriber('/odometry/filtered', Odometry)

		else:
			odomSub = Subscriber('/odom', Odometry)

		# to obtain human poses
		if self.human_detect_method == 'lidar':
			humanStatesSub = Subscriber('/dr_spaam_detections', PoseArray)  # human px, py, visible
		elif self.human_detect_method == 'zed':
			# need to source the catkin_ws where the zed ros wrapper is installed
			from zed_interfaces.msg import Object, ObjectsStamped
			if self.config.action_space.kinematics == 'turtlebot':
				humanStatesSub = Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped)
			else:
				humanStatesSub = Subscriber('/zed2i/zed_node/obj_det/objects', ObjectsStamped)
		else:
			# need to source the catkin_ws where the zed ros wrapper is installed
			from zed_interfaces.msg import Object, ObjectsStamped
			if self.config.action_space.kinematics == 'turtlebot':
				zedHumanStatesSub = Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped)
			else:
				zedHumanStatesSub = Subscriber('/zed2i/zed_node/obj_det/objects', ObjectsStamped)
			lidarHumanStatesSub = Subscriber('/dr_spaam_detections', PoseArray)  # human px, py, visible

		if self.use_dummy_detect:
			subList = [odomSub, jointStateSub] # if use T265 for robot pose
		elif self.human_detect_method in ['zed', 'lidar']:
			subList = [odomSub, jointStateSub, humanStatesSub] # if use T265 for robot pose
		else:
			# subList = [jointStateSub, zedHumanStatesSub, lidarHumanStatesSub, robotPoseSub]
			subList = [odomSub, jointStateSub, zedHumanStatesSub, lidarHumanStatesSub]

		# to obtain robot location
		if self.robot_localizer == 'zed':
			print('use zed to obtain robot pose')
			if self.config.action_space.kinematics == 'turtlebot':
				robotPoseSub = Subscriber('/zed2/zed_node/pose', PoseStamped)
			else:
				robotPoseSub = Subscriber('/zed2i/zed_node/pose', PoseStamped)
			subList.append(robotPoseSub)
		elif self.robot_localizer == 'enml' and self.config.action_space.kinematics == 'jackal':
			print('use enml to obtain robot pose')
			robotPoseSub = Subscriber('/localization', Localization2DMsg)


		# print(subList)
		# synchronize the robot base joint states and humnan detections with at most 1 seconds of difference
		self.ats = ApproximateTimeSynchronizer(subList, queue_size=100, slop=5)

		# if ignore sensor inputs and use fake human detections
		if self.use_dummy_detect:
			if self.robot_localizer == 't265':
				self.ats.registerCallback(self.state_cb_dummy)
			else:
				self.ats.registerCallback(self.state_cb_dummy_zed)
		# use t265 for robot localization
		elif self.robot_localizer == 't265':
			self.ats.registerCallback(self.state_cb)
			# print('registered state_cb')
		# use zed for robot localization
		elif self.human_detect_method in ['zed', 'lidar']:
			self.ats.registerCallback(self.state_cb_zed)
		else:
			self.ats.registerCallback(self.state_cb_fusion)

		rospy.on_shutdown(self.shutdown)

		self.lidar_ang_res = config.lidar.angular_res
		# total number of rays
		self.ray_num = int(360. / self.lidar_ang_res)

		# DRL-VO only: for OM observation
		self.ob_grid_size = self.config.planner.ob_grid_resolution
		self.ob_om_boundary = 6.0
		self.ob_grid_num = int(np.ceil(self.ob_om_boundary * 2 / self.ob_grid_size))

		self.grid_size = self.config.planner.grid_resolution
		# make sure the robot can circulate around large obstacles
		self.om_boundary = self.config.sim.robot_circle_radius + 1
		# generate the matrix for 2D grids
		self.grid_num = int(np.ceil(self.om_boundary * 2 / self.grid_size))

		self.fake_pc_env.configure(config)
		self.fake_pc_env.reset()


	def set_robot(self, robot):
		self.robot = robot

	def set_ob_act_space(self):
		# set observation space and action space
		# we set the max and min of action/observation space as inf
		# clip the action and observation as you need

		d = {}
		# robot node: num_visible_humans, px, py, gx, gy, theta

		d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5,), dtype=np.float32)
		# only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
		d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)

		# lidar cannot detect human velocity
		if self.human_detect_method in ['lidar', 'fusion']:
			self.human_state_size = 2
		# zed can, we can choose whether to use human velocity or not
		else:
			if self.config.ob_space.add_human_vel:
				self.human_state_size = 4
			else:
				self.human_state_size = 2
		print('self.real_max_human_num', self.real_max_human_num)
		d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
											shape=(self.real_max_human_num, self.human_state_size),
											dtype=np.float32)

		# number of humans detected at each timestep
		d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

		# real/fake lidar point cloud
		d['point_clouds'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.ray_num,), dtype=np.float32)

		if self.config.robot.policy == 'om_gru':
			self.ob_om_boundary = 4.5 + self.config.sim.human_pos_noise_range - 0.5
			d['om'] = gym.spaces.Box(low=-np.inf, high=np.inf,
									 shape=(3, int(np.ceil(
										 self.ob_om_boundary * 2 / self.config.planner.ob_grid_resolution)),
											int(np.ceil(
												self.ob_om_boundary * 2 / self.config.planner.ob_grid_resolution))),
									 dtype=np.float32)

		self.observation_space = gym.spaces.Dict(d)

		if self.config.env.action_space == 'continuous':
			high = np.inf * np.ones([2, ])
			self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
		elif self.config.env.action_space == 'discrete':
			# action is change of v and w, for sim2real
			# translational velocity change: +0.1, 0, -0.1 m/s
			# rotational velocity change: +0.5, 0, -0.5 rad/s
			delta_v_max = self.config.robot.acc_v_max * self.time_step
			delta_w_max = self.config.robot.acc_w_max * self.time_step
			self.action_convert = {0: [delta_v_max, delta_w_max], 1: [delta_v_max, 0], 2: [delta_v_max, -delta_w_max],
								3: [0, delta_w_max], 4: [0, 0], 5: [0, -delta_w_max],
								6: [-delta_v_max, delta_w_max], 7: [-delta_v_max, 0], 8: [-delta_v_max, -delta_w_max]}

			self.action_space = gym.spaces.Discrete(len(self.action_convert))

	# if use T265 for robot pose
	# (used if self.use_dummy_detect is False)
	# callback function to store the realtime messages from the robot to this env
	def state_cb(self, odomMsg, jointStateMsg, humanArrayMsg):
		# print('state_cb')
		if self.human_detect_method == 'lidar':
			self.humanMsg=humanArrayMsg.poses
		else:
			self.humanMsg = humanArrayMsg.objects
		self.jointMsg=jointStateMsg
		self.odomMsg = odomMsg.twist.twist

	# if use zed for robot pose
	def state_cb_zed(self, odomMsg, jointStateMsg, humanArrayMsg, robotPoseMsg):
		# print('state_cb')
		if self.human_detect_method == 'lidar':
			self.humanMsg=humanArrayMsg.poses
		else:
			self.humanMsg = humanArrayMsg.objects
		self.jointMsg=jointStateMsg
		self.robotMsg = robotPoseMsg
		self.odomMsg = odomMsg.twist.twist

	# todo: change this
	def state_cb_fusion(self, odomMsg, jointStateMsg, humanArrayMsg, robotPoseMsg):
		# print('state_cb')
		if self.human_detect_method == 'lidar':
			self.humanMsg=humanArrayMsg.poses
		else:
			self.humanMsg = humanArrayMsg.objects
		self.jointMsg=jointStateMsg
		self.robotMsg = robotPoseMsg
		self.odomMsg = odomMsg.twist.twist

	# if use T265 for robot pose
	# (used if self.use_dummy_detect is True)
	# callback function to store the realtime messages from the robot to this env
	# no need to real human message
	def state_cb_dummy(self, odomMsg, jointStateMsg):
		# print('state cb dummy', jointStateMsg)
		self.jointMsg = jointStateMsg
		self.odomMsg = odomMsg.twist.twist

	def state_cb_dummy_zed(self, odomMsg, jointStateMsg, robotPoseMsg):
		self.jointMsg = jointStateMsg
		self.robotMsg = robotPoseMsg
		self.odomMsg = odomMsg.twist.twist

	def coordinate_transform(self, ego_coord, target_coord):
		# Create the transformation matrix from ego frame
		transform_matrix = np.array([
			[np.cos(ego_coord[2]), -np.sin(ego_coord[2]), ego_coord[0]],
			[np.sin(ego_coord[2]), np.cos(ego_coord[2]), ego_coord[1]],
			[0, 0, 1]
		])
		    
		# Convert target coordinates to homogeneous coordinates
		target_coord_homogeneous = np.array([target_coord[0], target_coord[1], 1])
		
		# Transform target coordinates to the ego frame
		target_in_ego_frame = np.linalg.inv(transform_matrix) @ target_coord_homogeneous
		
		# Compute the new orientation (theta) in the ego frame
		new_theta = target_coord[2] - ego_coord[2]
		
		# Normalize new_theta to be within [-pi, pi]
		new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
		
		return [target_in_ego_frame[0], target_in_ego_frame[1], new_theta]


	def readMsg(self):
		"""
		read messages passed through ROS & prepare for generating obervations
		this function should be called right before generate_ob() is called
		"""
		self.mutex.acquire()
		# get time
		# print(self.jointMsg.header.stamp.secs, self.jointMsg.header.stamp.nsecs)
		if not self.use_fixed_time_interval:
			self.currentTime = self.jointMsg.header.stamp.secs + self.jointMsg.header.stamp.nsecs / 1e9

		# get robot pose from T265 SLAM camera
		if self.robot_localizer == 't265':
			try:
				self.robotMsg = self.tfBuffer.lookup_transform('t265_odom_frame', 't265_pose_frame', rospy.Time(0), rospy.Duration(1.0))
				# print('got robot msg from t265')
			except:
				print("did not get robot msg from t265, problem in getting transform")

		# get robot wheel velocity from the base
		try:
			self.jointVel=self.jointMsg.velocity
		except:
			print("problem in getting joint velocity")

		# get robot (v, w) velocity from the odometry
		try:
			self.odomVel=self.odomMsg
		except:
			print("problem in getting odom velocity")

		# print(self.robotMsg, "ROBOT mSG")
		# if use T265 for robot pose
		if self.robot_localizer == 't265':
		# store the robot pose and robot base velocity in self variables
			try:
				self.robot.px = -self.robotMsg.transform.translation.y
				self.robot.py = self.robotMsg.transform.translation.x
			except:
				print('Cannot get robot pose from T265, is T265 launched without error?')
			quaternion = (
				self.robotMsg.transform.rotation.x,
				self.robotMsg.transform.rotation.y,
				self.robotMsg.transform.rotation.z,
				self.robotMsg.transform.rotation.w
			)
		elif self.robot_localizer == 'enml':
			# read the robot pose from enml
			if self.step_counter == 1:
				self.start_pose = self.robotMsg.data
			curr_pose = [self.robotMsg.pose.x, self.robotMsg.pose.y, self.robotMsg.pose.theta] 
			self.robot.px, self.robot.py, self.robot.theta = self.coordinate_transform(self.start_pos, curr_pose)
			print('robot pose from enml:', self.robot.px, self.robot.py, self.robot.theta)
		# zed for robot pose
		else:
			self.robot.px = -self.robotMsg.pose.position.y
			self.robot.py = self.robotMsg.pose.position.x
			quaternion = (
				self.robotMsg.pose.orientation.x,
				self.robotMsg.pose.orientation.y,
				self.robotMsg.pose.orientation.z,
				self.robotMsg.pose.orientation.w
			)


		if self.use_dummy_detect:
			self.detectedHumanNum = 1

		else:
			# read human states
			if self.human_detect_method == 'lidar':
				self.detectedHumanNum=min(len(self.humanMsg), self.real_max_human_num)
				self.current_human_states_raw = np.ones((self.detectedHumanNum, 2)) * 15

				for i in range(self.detectedHumanNum):
					self.current_human_states_raw[i,0]=self.humanMsg[i].position.x
					self.current_human_states_raw[i,1] = self.humanMsg[i].position.y
			else:
				if len(self.humanMsg) == 0:
					self.detectedHumanNum = 0
					self.current_human_states_raw = np.ones((self.real_max_human_num, self.human_state_size)) * 15
				else:
					self.detectedHumanNum=min(len(self.humanMsg), self.real_max_human_num)
					self.current_human_states_raw = np.ones((self.detectedHumanNum, self.human_state_size+1)) * 15
					# read all detections
					for i, obj in enumerate(self.humanMsg):
						# if we detected more humans than the max number of humans, just drop the rest of humans
						if i >= self.detectedHumanNum:
							break
						if obj.label_id == -1:
							continue

						if self.config.ob_space.add_human_vel:
							# px, py, vx, vy, confidence (in camera frame)
							self.current_human_states_raw[i] = np.array(
								[obj.position[0], obj.position[1], obj.velocity[0], obj.velocity[1], obj.confidence])
						else:
							# px, py, confidence (in camera frame)
							self.current_human_states_raw[i] = np.array([obj.position[0], obj.position[1], obj.confidence])
					# print(self.current_human_states_raw)
					# if number of detected humans > self.real_max_human_num, take the top self.real_max_human_num humans with highest confidence
					self.current_human_states_raw = np.array(sorted(self.current_human_states_raw, key=lambda x: x[-1], reverse=True))
					# print(self.current_human_states_raw)
					self.current_human_states_raw = self.current_human_states_raw[:self.real_max_human_num, :-1]

		self.mutex.release()

		# robot orientation (+pi/2 to transform from T265 frame to simulated robot frame)
		# print('raw theta:', tf.transformations.euler_from_quaternion(quaternion)[2])
		self.robot.theta = tf.transformations.euler_from_quaternion(quaternion)[2] + np.pi / 2

		if self.robot.theta < 0:
			self.robot.theta = self.robot.theta + 2 * np.pi
		# print('current_human_states_raw:', self.current_human_states_raw)
		# add 180 degrees because of the transform from lidar frame to t265 camera frame
		hMatrix = np.array([[np.cos(self.robot.theta+np.pi), -np.sin(self.robot.theta+np.pi), 0, 0],
							  [np.sin(self.robot.theta+np.pi), np.cos(self.robot.theta+np.pi), 0, 0],
							 [0,0,1,0], [0,0,0,1]])

		# if we detected at least one person
		self.current_human_states = np.ones((self.real_max_human_num, self.human_state_size)) * 15

		if not self.use_dummy_detect:
			if self.human_detect_method == 'lidar':
				# transform human detections from lidar frame to world frame (not needed actually)
				for j in range(self.detectedHumanNum):
					xy=np.matmul(hMatrix,np.array([[self.current_human_states_raw[j,0],
													self.current_human_states_raw[j,1],
													0,
													1]]).T)

					self.current_human_states[j]=xy[:2,0]
			else:

				# transform human detections from camera frame to robot frame
				# print('self.current_human_states', self.current_human_states)
				# print('self.current_human_states_raw', self.current_human_states_raw)
				# x_cam = y_robot, y_cam = -x_robot
				self.current_human_states[:self.detectedHumanNum, 0] = -self.current_human_states_raw[:self.detectedHumanNum, 1]
				self.current_human_states[:self.detectedHumanNum, 1] = self.current_human_states_raw[:self.detectedHumanNum, 0]
				
				if self.config.ob_space.add_human_vel:
					self.current_human_states[:self.detectedHumanNum, 2] = -self.current_human_states_raw[:self.detectedHumanNum, 3]
					self.current_human_states[:self.detectedHumanNum, 3] = self.current_human_states_raw[:self.detectedHumanNum, 2]


		else:
			# self.current_human_states[0] = np.array([0, 1 - 0.5 * 0.1 * self.step_counter- self.robot.py])
			self.current_human_states[0] = np.array([15] * self.human_state_size)

		# this is desired velocity, not actual velocity!
		# self.robot.vx = self.last_v * np.cos(self.robot.theta)
		# self.robot.vy = self.last_v * np.sin(self.robot.theta)
		self.actual_v, self.actual_w = self.odomVel.linear.x, self.odomVel.angular.z
		self.robot.vx = self.actual_v * np.cos(self.robot.theta)
		self.robot.vy = self.actual_v * np.sin(self.robot.theta)


		print('robot state: px:', self.robot.px, ', py:', self.robot.py, ', vx:', self.robot.vx, ', vy:', self.robot.vy, ', theta:', self.robot.theta)
		# print(', theta:', self.robot.theta)

	@staticmethod
	def list_to_move_base_goal(goal_pose_list):
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = "map"
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = goal_pose_list[0]
		goal.target_pose.pose.position.y = goal_pose_list[1]
		goal.target_pose.pose.position.z = goal_pose_list[2]
		goal.target_pose.pose.orientation.x = goal_pose_list[3]
		goal.target_pose.pose.orientation.y = goal_pose_list[4]
		goal.target_pose.pose.orientation.z = goal_pose_list[5]
		goal.target_pose.pose.orientation.w = goal_pose_list[6]
		return goal

	def init_and_set_goal(self):
		# stop the turtlebot
		self.smoothStop()
		self.step_counter = 0
		self.currentTime = 0.0
		self.lastTime = 0.0
		self.global_time = 0.

		self.detectedHumanNum = 0
		self.current_human_states = np.ones((self.real_max_human_num, 2)) * 15
		self.desiredVelocity = [0.0, 0.0]
		self.last_left = 0.
		self.last_right = 0.
		self.last_w = 0.0

		self.last_v = 0.0

		while True:
			a = input("Press y for the next episode \t")
			if a == "y":
				self.robot.gx = float(input("Input goal location in x-axis\t"))
				self.robot.gy = float(input("Input goal location in y-axis\t"))
				for i in range(5):
					print(5-i, "seconds until the robot starts moving")
					time.sleep(1)
				break
			else:
				sys.exit()

		# send goal to nav stack
		if not self.publish_actions:
			try:
				goal = self.pose_lookup_table[(self.robot.gx, self.robot.gy)]
			except KeyError:
				print(
					'Goal pose not found in dictionary, please use the script in CrowdNav_sim2real_learning_dynamics/ to find the pose')
				sys.exit()
			self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
			move_base_goal = self.list_to_move_base_goal(goal)
			self.client.send_goal(move_base_goal)
			print('goal sent to nav stack')

		# to evaluate intrusion time ratio and average social distance during intrusion
		self.intrusion_timesteps = 0
		self.dist_intrusion = []

		if self.record:
			self.episodeRecoder.robot_goal.append([self.robot.gx, self.robot.gy])

	def reset(self):
		"""
		Reset function
		"""

		self.init_and_set_goal()

		self.readMsg()

		ob=self.generate_ob(reset=True) # generate initial obs

		return ob

	# input: v, w
	# output: v, w
	def smooth(self, v, w):
		beta = 0.2
		v_smooth = (1.-beta) * self.last_v + beta * v
		w_smooth = (1.-beta) * self.last_w + beta * w

		self.last_w = w

		return v_smooth, w_smooth

	def generate_ob(self, reset):
		ob = {}
		if self.config.ob_space.robot_state == 'absolute':
			ob['robot_node'] = np.array([[self.robot.px, self.robot.py, self.robot.gx, self.robot.gy, self.robot.theta]])
		else:
			ob['robot_node'] = np.array([[self.robot.gx - self.robot.px, self.robot.gy - self.robot.py, self.robot.theta]])
		ob['temporal_edges']=np.array([[self.robot.vx, self.robot.vy]])

		# Transform only detected human positions from robot to world frame
		for i in range(self.detectedHumanNum):
			pos = self.current_human_states[i, :2]
			if not np.allclose(pos, [15, 15]):
				world_pos = self.robot_to_world(pos)
				self.current_human_states[i, 0] = world_pos[0]
				self.current_human_states[i, 1] = world_pos[1]
			if self.config.ob_space.add_human_vel:
				vel = self.current_human_states[i, 2:4]
				if not np.allclose(vel, [15, 15]):
					world_vel = self.robot_to_world(vel)
					self.current_human_states[i, 2] = world_vel[0]
					self.current_human_states[i, 3] = world_vel[1]

		# print(self.current_human_states.shape)
		
		# Transform only detected human positions from robot to world frame
		for i in range(self.detectedHumanNum):
			pos = self.current_human_states[i, :2]
			if not np.allclose(pos, [15, 15]):
				# world_pos = self.robot_to_world(pos)
				world_pos = pos
				self.current_human_states[i, 0] = world_pos[0]
				self.current_human_states[i, 1] = world_pos[1]
		spatial_edges=self.current_human_states

		# sort humans by distance to robot
		spatial_edges = np.array(sorted(spatial_edges, key=lambda x: np.linalg.norm(x[:2])))
		# todo: uncomment later
		print('spatial edges:', spatial_edges[:self.detectedHumanNum])
		print('detected human num:', self.detectedHumanNum)
		ob['spatial_edges'] = spatial_edges

		self.spatial_edges_log.append(ob['spatial_edges'].copy())

		ob['detected_human_num'] = self.detectedHumanNum
		if ob['detected_human_num'] == 0:
			ob['detected_human_num'] = 1

		# generate fake point cloud
		self.fake_pc_env.set_robot_pose(px=self.robot.px, py=self.robot.py, theta=self.robot.theta)
		point_cloud, _, _, _ = self.fake_pc_env.step(0)  # shape [1, ray_num]
		ob['point_clouds'] = point_cloud

		if self.config.robot.policy == 'om_gru':
			# compute om channels (vx, vy)
			om_base = self.update_ob_om()

			# normalize point cloud and tile to match OM shape
			pc = (point_cloud - np.min(point_cloud)) / (np.max(point_cloud) - np.min(point_cloud) + 1e-8)
			pc_repeat = np.tile(pc[:, np.newaxis, :], [1, self.ob_grid_num, 1])  # shape [1, H, W]

			# combine into final OM representation
			ob['om'] = np.concatenate([om_base, pc_repeat], axis=0)  # shape [3, H, W]
		return ob
		

	def step(self, action, update=True):
		""" Step function """
		print("Step", self.step_counter)

		current_time = time.time()  # this gives the current Unix timestamp in seconds

		if self.step_counter > 0:
			actual_delta_t = current_time - self.prevTime
			# print('timestep:', actual_delta_t)
		else:
			actual_delta_t = 0.06
			# print('timestep: (first step)')
		self.prevTime = current_time

		# make sure the timestep in sim and real are consistent
		rospy.sleep(0.09)  # act as frame skip
		# if self.step_counter > 0 and actual_delta_t < self.delta_t:
		# 	print(self.fixed_time_interval - actual_delta_t)
		# 	rospy.sleep(self.fixed_time_interval - actual_delta_t)

		if self.publish_actions:
			# process action
			realAction = Twist()
			if isinstance(action, np.ndarray):
				action = action[0]
			delta_v, delta_w = self.action_convert[action]
			# print('delta_v:', delta_v, 'delta_w:', delta_w)

			if self.load_act: # load action from file for robot dynamics checking
				v_unsmooth= self.episodeRecoder.v_list[self.step_counter]
				w_unsmooth = self.episodeRecoder.delta_theta_list[self.step_counter]
				v_smooth, w_smooth = v_unsmooth, w_unsmooth
				# print('v_smooth:', v_smooth, 'w_smooth:', w_smooth)
			else:
				if self.config.env.action_space == 'continuous':
					action = self.robot.policy.clip_action(action, None)

					self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
					self.desiredVelocity[1] = action.r / self.fixed_time_interval # TODO: dynamic time step is not supported now

				else:

					self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + delta_v, self.config.robot.v_min, self.config.robot.v_max)
					self.desiredVelocity[1] = np.clip(self.desiredVelocity[1] + delta_w, self.config.robot.w_min, self.config.robot.w_max)

				# v_smooth, w_smooth = self.smooth(self.desiredVelocity[0], self.desiredVelocity[1])
				v_smooth, w_smooth = self.desiredVelocity[0], self.desiredVelocity[1]


			self.last_v = v_smooth

			realAction.linear.x = v_smooth
			realAction.angular.z = w_smooth

			self.actionPublisher.publish(realAction)

		# todo: why do we need this?
		rospy.sleep(self.ROSStepInterval)  # act as frame skip

		# get the latest states

		self.readMsg()


		# update time
		if self.step_counter==0: # if it is the first step of the episode
			self.delta_t = np.inf
		else:
			# time interval between two steps
			if self.use_fixed_time_interval:
				self.delta_t=self.fixed_time_interval
				# print('delta_t', self.delta_t)
			else:
				self.delta_t = self.currentTime - self.lastTime
				# print('delta_t:', self.currentTime - self.lastTime)
			#print('actual delta t:', currentTime - self.baseEnv.lastTime)
			self.global_time = self.global_time + self.delta_t
		self.step_counter=self.step_counter+1
		self.lastTime = self.currentTime

		if self.publish_actions:
			# process action
			realAction = Twist()
			if isinstance(action, np.ndarray):
				action = action[0]
				# action = 4
				delta_v, delta_w = self.action_convert[action]
				print('delta_v:', delta_v, 'delta_w:', delta_w)

			if self.load_act: # load action from file for robot dynamics checking
				# # before:
				# v_unsmooth= self.episodeRecoder.v_list[self.step_counter]
				# # in the simulator we use and recrod delta theta. We convert it to omega by dividing it by the time interval
				# w_unsmooth = self.episodeRecoder.delta_theta_list[self.step_counter] / self.delta_t
				# # v_smooth, w_smooth = self.desiredVelocity[0], self.desiredVelocity[1]
				# v_smooth, w_smooth = self.smooth(v_unsmooth, w_unsmooth)
				v_unsmooth= self.episodeRecoder.v_list[self.step_counter]
				w_unsmooth = self.episodeRecoder.delta_theta_list[self.step_counter]
				v_smooth, w_smooth = v_unsmooth, w_unsmooth
				print('v_smooth:', v_smooth, 'w_smooth:', w_smooth)
			else:
				if self.config.env.action_space == 'continuous':
					action = self.robot.policy.clip_action(action, None)

					self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
					self.desiredVelocity[1] = action.r / self.fixed_time_interval # TODO: dynamic time step is not supported now

				else:

					self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + delta_v, self.config.robot.v_min, self.config.robot.v_max)
					self.desiredVelocity[1] = np.clip(self.desiredVelocity[1] + delta_w, self.config.robot.w_min, self.config.robot.w_max)

				# v_smooth, w_smooth = self.smooth(self.desiredVelocity[0], self.desiredVelocity[1])
				v_smooth, w_smooth = self.desiredVelocity[0], self.desiredVelocity[1]
				# print('v_smooth:', v_smooth, 'w_smooth:', w_smooth)

			self.last_v = v_smooth

			realAction.linear.x = v_smooth
			realAction.angular.z = w_smooth
			# print('realAction:', realAction.linear.x, realAction.angular.z)
			self.actionPublisher.publish(realAction)

		# get the latest states

		self.readMsg()
		# print('self.odomMsg', self.odomMsg)

		# check for intrusion and if true, social distance during intrusion
		dist_RH = np.linalg.norm(self.current_human_states, axis=1)
		if np.any(dist_RH < self.discomfort_dist):
			self.intrusion_timesteps = self.intrusion_timesteps + 1
			self.dist_intrusion.append(np.min(dist_RH[dist_RH < self.discomfort_dist]))


		# generate new observation
		ob=self.generate_ob(reset=False)

		# calculate reward
		reward = 0

		# determine if the episode ends
		done=False
		reaching_goal = norm(np.array([self.robot.gx, self.robot.gy]) - np.array([self.robot.px, self.robot.py]))  < self.goal_reach_dist
		print(self.global_time, self.time_limit)
		if self.global_time >= self.time_limit:
			done = True
			print("Timeout")
		elif reaching_goal:
			done = True
			print("Goal Achieved")
		elif self.load_act and self.record:
			if self.step_counter >= len(self.episodeRecoder.v_list):
				done = True
		else:
			done = False


		info = {'info': None}

		if self.record:
			self.episodeRecoder.wheelVelList.append([self.actual_v, self.actual_w]) # it is the calculated wheel velocity, not the measured

			self.episodeRecoder.actionList.append([v_smooth, w_smooth])
			self.episodeRecoder.positionList.append([self.robot.px, self.robot.py])
			self.episodeRecoder.orientationList.append(self.robot.theta)

		if done:
			print('Done!')

			# record intrusion ratio and social distance during intrusion
			intrusion_ratio = self.intrusion_timesteps / self.step_counter
			if len(self.dist_intrusion) > 0:
				ave_dist_intrusion = sum(self.dist_intrusion) / len(self.dist_intrusion)
			else:
				ave_dist_intrusion = 0.0  # or np.nan or some sentinel value
			output_file = os.path.join(self.config.training.output_dir, 'real_world.txt')
			file_mode = 'a' if os.path.exists(output_file) else 'w'
			with open('output.txt', file_mode) as file:
				# Write the values of the variables to the file
				file.write(f'intrusion_ratio: {intrusion_ratio:.2f}, ave social dist during intrusion: {ave_dist_intrusion:.2f}\n')

			if self.record:
				self.episodeRecoder.saveEpisode(self.case_counter['test'])
				# Save spatial_edges_log to .npz file
				os.makedirs(self.episodeRecoder.savePath, exist_ok=True)
				output_path = os.path.join(self.episodeRecoder.savePath, f"spatial_edges_{int(time.time())}.npz")
				np.savez_compressed(output_path, spatial_edges=np.array(self.spatial_edges_log))
				print(f"Saved spatial_edges to {output_path}")


		return ob, reward, done, info

	def robot_to_world(self, vec):
		x, y = vec
		# Invert the rotation used in world_to_robot
		rot_angle = self.robot.theta - np.pi / 2
		R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
					[np.sin(rot_angle),  np.cos(rot_angle)]])
		vec_trans = np.matmul(R, np.array([[x], [y]]))
		return np.array([vec_trans[0, 0], vec_trans[1, 0]])

	def shutdown(self):
		self.smoothStop()
		print("You are stopping the robot!")
		self.reset()
		

	def smoothStop(self):
		if self.publish_actions:
			realAction = Twist()
			self.actionPublisher.publish(Twist())


	'''
	For OM in DRL-VO only
	'''
	def point_to_grid(self, px, py):
		# Normalize point coordinates to grid indices
		grid_x = int(np.floor((px + self.om_boundary) / self.grid_size))
		grid_y = int(np.floor((py + self.om_boundary) / self.grid_size))

		# Ensure indices are within bounds
		grid_x = min(max(grid_x, 0), self.grid_num - 1)
		grid_y = min(max(grid_y, 0), self.grid_num - 1)

		return grid_x, grid_y


	def update_ob_om(self):
		om = -np.ones([2, self.ob_grid_num, self.ob_grid_num])

		human_radius = self.config.humans.radius
		robot_radius = self.config.robot.radius

		# Mark humans
		for i in range(self.detectedHumanNum):
			human_x, human_y = self.current_human_states[i, 0], self.current_human_states[i, 1]

			# Transform velocity from robot frame to world frame and add robot's motion
			vx, vy = 0.0, 0.0
			if self.human_state_size >= 4:
				vx_r = self.current_human_states[i, 2]
				vy_r = self.current_human_states[i, 3]
				vx_world_rel, vy_world_rel = self.robot_to_world([vx_r, vy_r])
				vx = vx_world_rel + self.robot.vx
				vy = vy_world_rel + self.robot.vy

			grid_x, grid_y = self.point_to_grid(human_x, human_y)
			radius_in_cells = int(np.ceil(human_radius / self.ob_grid_size))

			for dx in range(-radius_in_cells, radius_in_cells + 1):
				for dy in range(-radius_in_cells, radius_in_cells + 1):
					x_idx = grid_x + dx
					y_idx = grid_y + dy
					if 0 <= x_idx < self.ob_grid_num and 0 <= y_idx < self.ob_grid_num:
						if (dx ** 2 + dy ** 2) * (self.ob_grid_size ** 2) <= human_radius ** 2:
							om[0, y_idx, x_idx] = vx
							om[1, y_idx, x_idx] = vy

		# Mark robot velocity in OM
		robot_grid_x, robot_grid_y = self.point_to_grid(self.robot.px, self.robot.py)
		robot_radius_cells = int(np.ceil(robot_radius / self.ob_grid_size))

		for dx in range(-robot_radius_cells, robot_radius_cells + 1):
			for dy in range(-robot_radius_cells, robot_radius_cells + 1):
				x_idx = robot_grid_x + dx
				y_idx = robot_grid_y + dy
				if 0 <= x_idx < self.ob_grid_num and 0 <= y_idx < self.ob_grid_num:
					if (dx ** 2 + dy ** 2) * (self.ob_grid_size ** 2) <= robot_radius ** 2:
						om[0, y_idx, x_idx] = self.robot.vx
						om[1, y_idx, x_idx] = self.robot.vy

		return om

	def robot_to_world(self, vec):
		x, y = vec
		# Invert the rotation used in world_to_robot
		rot_angle = self.robot.theta - np.pi / 2
		R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
					  [np.sin(rot_angle), np.cos(rot_angle)]])
		vec_trans = np.matmul(R, np.array([[x], [y]]))
		return np.array([vec_trans[0, 0], vec_trans[1, 0]])