import gym
import numpy as np
from numpy.linalg import norm
import os

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

		self.currentTime=0.0
		self.lastTime=0.0 # store time for calculating time interval

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

		# self.goal_reach_dist = 1.0
		self.intrusion_timesteps = 0
		self.dist_intrusion = []

		# given a goal pose (x, y) in T265 frame (direction faces front), convert it to map frame of nav stack (translation x, y, z, rotation x, y, z, w)
		self.pose_lookup_table = {(0., 6.): [5.993, 1.488, 0.0102, 0, 0, 0.101, 0.995],
								  (0., 4.): [3.851, 0.500, 0.0102, 0, 0, 0.197, 0.980]}


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
			self.actionPublisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=1)
		self.tfBuffer = tf2_ros.Buffer()
		self.transformListener = tf2_ros.TransformListener(self.tfBuffer)

		# ROS subscribers
		# to obtain robot velocity
		jointStateSub = Subscriber("/joint_states", JointState)

		# to obtain human poses
		if self.human_detect_method == 'lidar':
			humanStatesSub = Subscriber('/dr_spaam_detections', PoseArray)  # human px, py, visible
		elif self.human_detect_method == 'zed':
			# need to source the catkin_ws where the zed ros wrapper is installed
			from zed_interfaces.msg import Object, ObjectsStamped
			humanStatesSub = Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped)
		else:
			# need to source the catkin_ws where the zed ros wrapper is installed
			from zed_interfaces.msg import Object, ObjectsStamped
			zedHumanStatesSub = Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped)
			lidarHumanStatesSub = Subscriber('/dr_spaam_detections', PoseArray)  # human px, py, visible

		if self.use_dummy_detect:
			subList = [jointStateSub] # if use T265 for robot pose
		elif self.human_detect_method in ['zed', 'lidar']:
			subList = [jointStateSub, humanStatesSub] # if use T265 for robot pose
		else:
			# subList = [jointStateSub, zedHumanStatesSub, lidarHumanStatesSub, robotPoseSub]
			subList = [jointStateSub, zedHumanStatesSub, lidarHumanStatesSub]

		# to obtain robot location
		if self.robot_localizer == 'zed':
			print('use zed to obtain robot pose')
			robotPoseSub = Subscriber('/zed2/zed_node/pose', PoseStamped)
			subList.append(robotPoseSub)

		# print(subList)
		# synchronize the robot base joint states and humnan detections with at most 1 seconds of difference
		self.ats = ApproximateTimeSynchronizer(subList, queue_size=100, slop=5)

		# if ignore sensor inputs and use fake human detections
		if self.use_dummy_detect:
			if self.robot_localizer == 't265':
				self.ats.registerCallback(self.state_cb_dummy)
			else:
				self.ats.registerCallback(self.state_cb_dummy_zed)
		elif self.robot_localizer == 't265':
			self.ats.registerCallback(self.state_cb)
			print('registered state_cb')
		elif self.human_detect_method in ['zed', 'lidar']:
			self.ats.registerCallback(self.state_cb_zed)
		else:
			self.ats.registerCallback(self.state_cb_fusion)

		rospy.on_shutdown(self.shutdown)

		self.lidar_ang_res = config.lidar.angular_res
		# total number of rays
		self.ray_num = int(360. / self.lidar_ang_res)

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

		self.observation_space = gym.spaces.Dict(d)

		if self.config.env.action_space == 'continuous':
			high = np.inf * np.ones([2, ])
			self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
		elif self.config.env.action_space == 'discrete':
			# action is change of v and w, for sim2real
			# translational velocity change: +0.1, 0, -0.1 m/s
			# rotational velocity change: +0.5, 0, -0.5 rad/s
			self.action_convert = {0: [0.05, 0.1], 1: [0.05, 0], 2: [0.05, -0.1],
								   3: [0, 0.1], 4: [0, 0], 5: [0, -0.1],
								   6: [-0.05, 0.1], 7: [-0.05, 0], 8: [-0.05, -0.1]}

			self.action_space = gym.spaces.Discrete(len(self.action_convert))

	# if use T265 for robot pose
	# (used if self.use_dummy_detect is False)
	# callback function to store the realtime messages from the robot to this env
	def state_cb(self, jointStateMsg, humanArrayMsg):
		# print('state_cb')
		self.humanMsg=humanArrayMsg.poses
		self.jointMsg=jointStateMsg

	def state_cb_zed(self, jointStateMsg, humanArrayMsg, robotPoseMsg):
		# print('state_cb')
		if self.human_detect_method == 'lidar':
			self.humanMsg=humanArrayMsg.poses
		else:
			self.humanMsg = humanArrayMsg.objects
		self.jointMsg=jointStateMsg
		self.robotMsg = robotPoseMsg

	# todo: change this
	def state_cb_fusion(self, jointStateMsg, humanArrayMsg, robotPoseMsg):
		# print('state_cb')
		if self.human_detect_method == 'lidar':
			self.humanMsg=humanArrayMsg.poses
		else:
			self.humanMsg = humanArrayMsg.objects
		self.jointMsg=jointStateMsg
		self.robotMsg = robotPoseMsg

	# if use T265 for robot pose
	# (used if self.use_dummy_detect is True)
	# callback function to store the realtime messages from the robot to this env
	# no need to real human message
	def state_cb_dummy(self, jointStateMsg):
		# print('state cb dummy', jointStateMsg)
		self.jointMsg = jointStateMsg

	def state_cb_dummy_zed(self, jointStateMsg, robotPoseMsg):
		self.jointMsg = jointStateMsg
		self.robotMsg = robotPoseMsg

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
				self.robotMsg = self.tfBuffer.lookup_transform('t265_odom_frame', 't265_pose_frame', rospy.Time.now(), rospy.Duration(1.0))
				# print('got robot msg from t265')
			except:
				print("did not get robot msg from t265, problem in getting transform")

		# get robot base velocity from the base
		try:
			self.jointVel=self.jointMsg.velocity
		except:
			print("problem in getting joint velocity")

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
		print('robot pos:', self.robot.px, self.robot.py)

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
				self.detectedHumanNum = len(self.humanMsg)
				self.current_human_states_raw = np.ones((max(self.max_human_num, len(self.humanMsg)), self.human_state_size+1)) * 15
				# read all detections
				for i, obj in enumerate(self.humanMsg):
					if obj.label_id == -1:
						continue
					# px, py, vx, vy, confidence (in camera frame)
					self.current_human_states_raw[i] = np.array([obj.position[0], obj.position[1], obj.velocity[0], obj.velocity[1], obj.confidence])
				# print(self.current_human_states_raw)
				# if number of detected humans > self.real_max_human_num, take the top self.real_max_human_num humans with highest confidence
				self.current_human_states_raw = np.array(sorted(self.current_human_states_raw, key=lambda x: x[-1], reverse=True))
				# print(self.current_human_states_raw)
				self.current_human_states_raw = self.current_human_states_raw[:self.real_max_human_num, :-1]

		print(self.current_human_states_raw)
		self.mutex.release()

		# robot orientation (+pi/2 to transform from T265 frame to simulated robot frame)
		self.robot.theta = tf.transformations.euler_from_quaternion(quaternion)[2] + np.pi / 2

		if self.robot.theta < 0:
			self.robot.theta = self.robot.theta + 2 * np.pi

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

		self.robot.vx = self.last_v * np.cos(self.robot.theta)
		self.robot.vy = self.last_v * np.sin(self.robot.theta)

		# print('robot velocity:', self.robot.vx, self.robot.vy, 'robot theta:', self.robot.theta)

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

		# get fake point cloud
		self.fake_pc_env.set_robot_pose(px=self.robot.px, py=self.robot.py, theta=self.robot.theta)
		point_cloud, _, _, _ = self.fake_pc_env.step(0)  # 0 is the dummy action
		ob['point_clouds'] = point_cloud

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
		# print(self.current_human_states.shape)
		spatial_edges=self.current_human_states

		# sort humans by distance to robot
		spatial_edges = np.array(sorted(spatial_edges, key=lambda x: np.linalg.norm(x[:2])))
			# print(spatial_edges)
		print('spatial edges:', spatial_edges)
		print('detected human num:', self.detectedHumanNum)
		ob['spatial_edges'] = spatial_edges

		ob['detected_human_num'] = self.detectedHumanNum
		if ob['detected_human_num'] == 0:
			ob['detected_human_num'] = 1
			
		return ob
		

	def step(self, action, update=True):
		""" Step function """
		print("Step", self.step_counter)
		if self.publish_actions:
			# process action
			realAction = Twist()

			if self.load_act: # load action from file for robot dynamics checking
				v_unsmooth= self.episodeRecoder.v_list[self.step_counter]
				# in the simulator we use and recrod delta theta. We convert it to omega by dividing it by the time interval
				w_unsmooth = self.episodeRecoder.delta_theta_list[self.step_counter] / self.delta_t
				# v_smooth, w_smooth = self.desiredVelocity[0], self.desiredVelocity[1]
				v_smooth, w_smooth = self.smooth(v_unsmooth, w_unsmooth)
			else:
				if self.config.env.action_space == 'continuous':
					action = self.robot.policy.clip_action(action, None)

					self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
					self.desiredVelocity[1] = action.r / self.fixed_time_interval # TODO: dynamic time step is not supported now

				else:
					if isinstance(action, np.ndarray):
						action = action[0]
					delta_v, delta_w = self.action_convert[action]
					self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + delta_v, 0, 0.5)
					self.desiredVelocity[1] = np.clip(self.desiredVelocity[1] + delta_w, -1, 1)

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
			else:
				self.delta_t = self.currentTime - self.lastTime
				print('delta_t:', self.currentTime - self.lastTime)
			#print('actual delta t:', currentTime - self.baseEnv.lastTime)
			self.global_time = self.global_time + self.delta_t
		self.step_counter=self.step_counter+1
		self.lastTime = self.currentTime

		# check for intrusion and if true, social distance during intrusion
		dist_RH = np.linalg.norm(self.current_human_states, axis=1)
		if np.any(dist_RH < self.discomfort_dist):
			self.intrusion_timesteps = self.intrusion_timesteps + 1
			self.dist_intrusion.append(np.min(dist_RH[dist_RH < self.discomfort_dist]))


		# generate new observation
		ob=self.generate_ob(reset=False)

		# get fake point cloud
		self.fake_pc_env.set_robot_pose(px=self.robot.px, py=self.robot.py, theta=self.robot.theta)
		point_cloud, _, _, _ = self.fake_pc_env.step(0)  # 0 is the dummy action
		ob['point_clouds'] = point_cloud


		# calculate reward
		reward = 0

		# determine if the episode ends
		done=False
		reaching_goal = norm(np.array([self.robot.gx, self.robot.gy]) - np.array([self.robot.px, self.robot.py]))  < self.goal_reach_dist
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
			self.episodeRecoder.wheelVelList.append(self.jointVel) # it is the calculated wheel velocity, not the measured
			self.episodeRecoder.actionList.append([v_smooth, w_smooth])
			self.episodeRecoder.positionList.append([self.robot.px, self.robot.py])
			self.episodeRecoder.orientationList.append(self.robot.theta)

		if done:
			print('Done!')

			# record intrusion ratio and social distance during intrusion
			intrusion_ratio = self.intrusion_timesteps / self.step_counter
			ave_dist_intrusion = sum(self.dist_intrusion) / len(self.dist_intrusion)
			output_file = os.path.join(self.config.training.output_dir, 'real_world.txt')
			file_mode = 'a' if os.path.exists(output_file) else 'w'
			with open('output.txt', file_mode) as file:
				# Write the values of the variables to the file
				file.write(f'intrusion_ratio: {intrusion_ratio:.2f}, ave social dist during intrusion: {ave_dist_intrusion:.2f}\n')

			if self.record:
				self.episodeRecoder.saveEpisode(self.case_counter['test'])


		return ob, reward, done, info


	def shutdown(self):
		self.smoothStop()
		print("You are stopping the robot!")
		self.reset()
		

	def smoothStop(self):
		if self.publish_actions:
			realAction = Twist()
			self.actionPublisher.publish(Twist())



