import gym
import numpy as np
from numpy.linalg import norm
import copy
import os
from collections import defaultdict

import pybullet as p
from pybullet_utils import bullet_client

try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass


from crowd_sim.envs.crowd_sim_tb2 import CrowdSim3DTB

'''
Used to generate fake lidar point cloud for observation space during sim2real. 
This class is the same as CrowdSim3DTB except observation space
'''

class CrowdSim3DTB_Sim2real(CrowdSim3DTB):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        config.sim.human_num = 0
        config.sim.human_num_range = 0
        super().configure(config)


    def set_robot(self, robot):
        self.robot = robot

        # set observation space and action space
        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        # 3. raw lidar point cloud
        self.observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.ray_num,), dtype=np.float32)

        self.action_convert = {0: [0.05, 0.1], 1: [0.05, 0], 2: [0.05, -0.1],
                               3: [0, 0.1], 4: [0, 0], 5: [0, -0.1],
                               6: [-0.05, 0.1], 7: [-0.05, 0], 8: [-0.05, -0.1]}

        self.action_space = gym.spaces.Discrete(len(self.action_convert))



    def generate_ob(self, reset):
        self.ray_test()
        ob = np.expand_dims(self.closest_hit_dist, axis=0)
        return ob

    # set the robot in a dummy pose, used in reset only
    def generate_robot(self):
        self.robot.set(0, 0, 0, 0, 0, 0, 0)

    def reset(self, phase='train', test_case=None):

        ############################ 2. create pybullet simulator for the robot and humans #############################
        # After gym.make, we call env.reset(). This function is then called.
        # this function starts pybullet client and create_single_player_scene

        # if it is the first run, setup Pybullet client and set GUI camera
        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True

            if self.config.sim.render:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

            # optionally enable EGL for faster headless rendering
            try:
                if os.environ["PYBULLET_EGL"]:
                    con_mode = self._p.getConnectionInfo()['connectionMethod']
                    if con_mode == self._p.DIRECT:
                        egl = pkgutil.get_loader('eglRenderer')
                        if (egl):
                            self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                        else:
                            self._p.loadPlugin("eglRendererPlugin")
            except:
                pass

            self.physicsClientId = self._p._client

            assert self._p.isNumpyEnabled() == 1
            p.setRealTimeSimulation(1)

            self.used_human_uids = []  # list of objectIDs of cylinders that have ALREADY been assigned to a human

            # load tb2 here
            # robot height must be < self.lidar.height!!!
            self.robot.uid = p.loadURDF("crowd_sim/pybullet/media/turtlebot2/turtlebot.urdf",[0, 0, 0])

            self.free_obs_uids = []
            self.used_obs_uids = []

            self.obs_sizes = self.config.fixed_obs.sizes
            self.obs_pos = self.config.fixed_obs.positions_lower_left
            max_obs_num = len(self.obs_sizes)
            for i in range(max_obs_num):
                # create cylinders, only for fixed obs environments
                if not self.config.sim.random_obs and self.config.fixed_obs.shapes[i] == 0:
                    new_uid = self.create_object(50, 50, radius=self.config.fixed_obs.cylinder_radius,
                                                         height=self.config.fixed_obs.cylinder_height,
                                                        shape=p.GEOM_CYLINDER,
                                                        color=[0.3, 0.3, 0.3, 1])
                    self.free_obs_uids.append(new_uid)
                    self.cylinder_obs_uids.append(new_uid)
                # create rectangles
                else:
                    self.free_obs_uids.append(self.create_object(50, 50, radius=None, height=0.7,
                                                                 shape=p.GEOM_BOX,
                                                                 color=[0.65, 0.65, 0.65, 1],
                                                                 # assume all objects' true heights are 1, will reset offset in z-axis later
                                                                 halfExtents=[self.obs_sizes[i, 0] / 2,
                                                                              self.obs_sizes[i, 1] / 2, 1]))
            # the matching between self.all_obs_uids and self.obs_sizes is always the same!!!
            self.all_obs_uids = copy.deepcopy(self.free_obs_uids)

            # update self.obstacles (stores ALL obstacles with pybullet object IDs)
            self.generate_rectangle_obstacle(max_obs_num, fixed_sizes=self.obs_sizes, fixed_pos=self.obs_pos,
                                             uids=self.all_obs_uids)

        # if it is the first run, build scene and setup simulation physics
        if self.scene is None:
            # this function will call episode_restart()->clean_everything() in
            # class World in scene_bases.py (self.cpp_world)
            # set gravity, set physics engine parameters, etc.
            self.scene = self.create_single_player_scene(self._p)

            # load arena floor
            self.loadArena()

        # # if it is not the first run
        if self.ownsPhysicsClient:
            self.scene.episode_restart()


        self.envStepCounter = 0

        # assign uids to static obs, and reset pos here
        self.obs_num = len(self.obs_pos)
        # array to store the (x, y, w, h, uid) of all present obs in this episode
        self.cur_obstacles = np.zeros((self.obs_num, 5))
        for i in range(self.obs_num):
            cur_uid = self.free_obs_uids.pop()
            self.used_obs_uids.append(cur_uid)
            idx = np.where(self.obstacles[:, -1] == cur_uid)
            idx = idx[0][0]
            center_x = self.obstacles[idx][0] + self.obstacles[idx][2] / 2
            center_y = self.obstacles[idx][1] + self.obstacles[idx][3] / 2
            # randomize the heights of obs
            if self.config.sim.random_static_obs_height:
                center_z = np.random.uniform(-0.6, 0.5)
            # fix the heights of obs
            else:
                center_z = 0.
            self._p.resetBasePositionAndOrientation(cur_uid,
                                                    [center_x, center_y, center_z],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))
            # update the present obs for this episode
            self.cur_obstacles[i] = self.obstacles[idx]

        # span the robot and humans in the arena
        self.generate_robot_humans(phase)

        p.resetBasePositionAndOrientation(self.robot.uid, [self.robot.px, self.robot.py, 0], p.getQuaternionFromEuler([0, 0, self.robot.theta]))

        self._p.stepSimulation()  # refresh the simulator. Needed for the ray test

        # compute the observation
        ob = self.generate_ob(reset=True)

        return ob

    # reset the pose of robot in simulator
    def set_robot_pose(self, px, py, theta):
        self.robot.px = px
        self.robot.py = py
        self.robot.theta = theta
        p.resetBasePositionAndOrientation(self.robot.uid, [self.robot.px, self.robot.py, 0], p.getQuaternionFromEuler([0, 0, self.robot.theta]))


    # given a robot pose, obtain a fake lidar scan by ray_test in an empty environment
    def step(self, action, update=True):
        reward, done, episode_info = 0, False, 'none'

        info={'info':episode_info}

        # for _ in range(self.config.pybullet.frameSkip):
        self.scene.global_step()

        # compute the observation
        ob = self.generate_ob(reset=False)

        return ob, reward, done, info