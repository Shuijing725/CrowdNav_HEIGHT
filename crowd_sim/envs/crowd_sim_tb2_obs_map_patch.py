import gym
import numpy as np
from numpy.linalg import norm
import os
import copy
import pybullet as p
import matplotlib.pyplot as plt

try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass

from crowd_sim.envs.utils.info import *
from crowd_sim.envs.crowd_sim_tb2_obs_hierarchy import CrowdSim3DTbObsHie
from crowd_sim.envs.crowd_sim_tb2_obs import CrowdSim3DTbObs
from crowd_sim.envs.planning_utils.Astar_with_clearance import generate_astar_path

'''
The obstacles are represented by OM instead of point cloud, 
inherents from CrowdSim3DTbObsHie for OM as part of ob space
'''

class CrowdSim3DTbObsHieOMPatch(CrowdSim3DTbObsHie):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        self.grid_size = config.planner.grid_resolution
        # make sure the robot can circulate around large obstacles
        self.om_boundary = config.sim.robot_circle_radius + 1

        # generate the matrix for 2D grids
        self.grid_num = int(np.ceil(self.om_boundary * 2 / self.grid_size))
        # we divide the entire map into 4x4 patches
        self.num_patches_per_edge = config.planner.om_patch_num
        self.patch_size = self.grid_num // self.num_patches_per_edge
        super().configure(config)
        # force this variable to be False, so that self.om only contains static obstacles
        self.config.planner.om_inludes_human = False


    def set_observation_space(self):
        d = {}
        # robot px, py, gx, gy, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5,), dtype=np.float32)

        # robot vx, vy
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # make sure there's at least one human
        if self.config.ob_space.add_human_vel:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 4),
                                                dtype=np.float32)
        else:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 2),
                                                dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # 16 (4x4) patches, each patch's size is (self.patch_size, self.patch_size)
        # d['om'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_patches_per_edge**2, self.patch_size, self.patch_size), dtype=np.float32)
        d['om'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(1, self.grid_num, self.grid_num),
                                 dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)


    def generate_ob(self, reset):
        ob = super(CrowdSim3DTbObs, self).generate_ob(reset=reset)

        ob['om'] = self.om

        return ob

    def reset(self, phase='train', test_case=None):
        self.create_scenario(phase=phase, test_case=test_case)

        # create occupancy map and plan a path
        # in random env, need to reset self.ob_om in the beginning of every episode
        if self.config.env.scenario == 'circle_crossing':
            self.update_om()
            # calculate the om in ob because it won't change until the obstacle layout changes
            # 2. Reshape into 4x4 grid of 10x10 patches → shape [4, 4, 10, 10]
            # self.ob_patches = self.om.reshape(self.num_patches_per_edge, self.patch_size, self.num_patches_per_edge, self.patch_size).transpose(0, 2, 1, 3)
            # 3. Flatten the 4x4 grid into 16 patches → shape [16, 10, 10]
            # self.ob_patches = self.ob_patches.reshape(-1, self.patch_size, self.patch_size)
        # otherwise, since the obstacle number and poses do not change, only need to do it once when the python program begins
        else:
            if self.first_epi:
                self.update_om()
                # calculate the om in ob because it won't change until the obstacle layout changes
                # 2. Reshape into 4x4 grid of 10x10 patches → shape [4, 4, 10, 10]
                # self.ob_patches = self.om.reshape(self.num_patches_per_edge, self.patch_size, self.num_patches_per_edge,
                #                                   self.patch_size).swapaxis(1, 2)
                # # 3. Flatten the 4x4 grid into 16 patches → shape [16, 10, 10]
                # self.ob_patches = self.ob_patches.reshape(-1, self.patch_size, self.patch_size)
        # create sphere shapes to visualize waypoints and the goal
        self.create_goal_object()

        ob = self.generate_ob(reset=True)
        return ob

    def step(self, action, update=True):
        ob, reward, done, info = super(CrowdSim3DTbObs, self).step(action, update=update)
        return ob, reward, done, info

    def calc_reward(self, action, danger_zone='circle'):
        reward, done, episode_info = super(CrowdSim3DTbObs, self).calc_reward(action, danger_zone=danger_zone)
        return reward, done, episode_info