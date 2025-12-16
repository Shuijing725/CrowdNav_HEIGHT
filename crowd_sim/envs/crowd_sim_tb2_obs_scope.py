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

class CrowdSim3DTbObsHieOM(CrowdSim3DTbObsHie):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        super().configure(config)
        self.ob_grid_size = self.config.planner.ob_grid_resolution

        # make sure the robot can circulate around large obstacles
        self.ob_om_boundary = 6.

        # generate the matrix for 2D grids
        self.ob_grid_num = int(np.ceil(self.ob_om_boundary * 2 / self.ob_grid_size))

    def set_observation_space(self):
        d = {}
        # robot px, py, gx, gy, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5,), dtype=np.float32)

        # robot vx, vy
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        if self.config.planner.om_type == 'global':
            self.ob_om_boundary = 4.5 + self.config.sim.human_pos_noise_range - 0.5
            d['om'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                     shape=(3, int(np.ceil(self.ob_om_boundary * 2 / self.config.planner.ob_grid_resolution)), int(np.ceil(self.ob_om_boundary * 2 / self.config.planner.ob_grid_resolution))),
                                     dtype=np.float32)
        else:
            # 3. 64x64 OM, centered at robot, grid size = 0.1
            d['om'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.config.planner.ob_grid_num,self.config.planner.ob_grid_num), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

    def update_ob_om(self):
        """
        Extracts a local occupancy map based on the configured mode:
        - Local: 64x64 map centered at the robot
        - Global: full environment occupancy map centered at the arena

        Pads with 1s if the robot is near the boundary in local mode.
        Marks humans as 1, the robot as 2, and the goal (if desired) as 3.
        """
        self.ob_om = -np.ones([2, self.ob_grid_num, self.ob_grid_num])

        # mark humans (vx, vy)
        for i, human in enumerate(self.humans):
            if not self.human_visibility[i]:
                continue

            human_grid_x, human_grid_y = self.point_to_grid(human.px, human.py)
            radius_in_cells = int(np.ceil(human.radius / self.ob_grid_size))

            for dx in range(-radius_in_cells, radius_in_cells + 1):
                for dy in range(-radius_in_cells, radius_in_cells + 1):
                    grid_x = human_grid_x + dx
                    grid_y = human_grid_y + dy
                    if 0 <= grid_x < self.ob_om.shape[1] and 0 <= grid_y < self.ob_om.shape[2]:
                        if (dx ** 2 + dy ** 2) * (self.ob_grid_size ** 2) <= human.radius ** 2:
                            self.ob_om[0, grid_y, grid_x] = human.vx
                            self.ob_om[1, grid_y, grid_x] = human.vy

        robot_grid_x, robot_grid_y = self.point_to_grid(self.robot.px, self.robot.py)
        robot_radius_in_cells = int(np.ceil(self.robot.radius / self.ob_grid_size))

        # mark the robot speed
        for dx in range(-robot_radius_in_cells, robot_radius_in_cells + 1):
            for dy in range(-robot_radius_in_cells, robot_radius_in_cells + 1):
                grid_x = robot_grid_x + dx
                grid_y = robot_grid_y + dy
                if 0 <= grid_x < self.ob_om.shape[1] and 0 <= grid_y < self.ob_om.shape[2]:
                    if (dx ** 2 + dy ** 2) * (self.ob_grid_size ** 2) <= self.robot.radius ** 2:
                        self.ob_om[0, grid_y, grid_x] = self.robot.vx
                        self.ob_om[1, grid_y, grid_x] = self.robot.vy


        # import os
        # import matplotlib.pyplot as plt
        # # Create a folder for storing images
        # folder_name = os.path.join("local_om_images", str(self.rand_seed))
        # os.makedirs(folder_name, exist_ok=True)
        # # Save local_om as an image inside the folder with step_counter in filename
        # filename_local_om = os.path.join(folder_name, f"vx_{self.step_counter}.png")
        # plt.imshow(self.ob_om[0], cmap='gray', origin='lower')
        # plt.colorbar()
        # plt.savefig(filename_local_om)
        # plt.close()
        # filename_local_om = os.path.join(folder_name, f"vy_{self.step_counter}.png")
        # plt.imshow(self.ob_om[1], cmap='gray', origin='lower')
        # plt.colorbar()
        # plt.savefig(filename_local_om)
        # plt.close()


    def generate_ob(self, reset):
        ob = {}

        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        # robot states
        if self.config.ob_space.robot_state == 'absolute':
            ob['robot_node'] = self.robot.get_changing_state_list()
        else:
            ob['robot_node'] = self.robot.get_changing_state_list_goal_offset()

        self.update_last_human_states(self.human_visibility, reset=reset)

        ob['temporal_edges'] = np.array([self.robot.v, self.robot.w])

        self.update_ob_om()
        # 3. raw lidar point cloud
        # include everything (humans and obs) in lidar pc scan
        if self.config.ob_space.lidar_pc_include_humans:
            self.ray_test()
            pc = np.expand_dims(self.closest_hit_dist, axis=0)
        # only include obs
        # a. all obs in one lidar scan
        # b. seperate each obs for robot-obstacle attn
        else:
            self.ray_test_no_humans()
            pc = np.expand_dims(self.closest_hit_dist, axis=0)
        # normalize pc values to [0, 1]
        pc = (pc - np.min(pc)) / (np.max(pc) - np.min(pc) + 1e-8)
        # tile it from [1, self.ob_grid_num] to [1, self.ob_grid_num, self.ob_grid_num]
        pc_repeat = np.tile(pc[:, np.newaxis, :], [1, self.ob_grid_num, 1])
        ob['om'] = np.concatenate([self.ob_om, pc_repeat], axis=0)

        # update self.observed_human_ids
        self.observed_human_ids = np.where(self.human_visibility)[0]
        self.ob = ob

        return ob

    def reset(self, phase='train', test_case=None):
        self.create_scenario(phase=phase, test_case=test_case)

        if self.config.planner.om_inludes_human:
            self.om_human_mask = np.ones([self.ob_grid_num, self.ob_grid_num], dtype=int)
        # create occupancy map and plan a path
        # in random env, need to reset self.ob_om in the beginning of every episode
        if self.config.env.scenario == 'circle_crossing':
            self.update_ob_om()
            if self.config.planner.om_inludes_human:
                self.update_om_humans()
        # otherwise, since the obstacle number and poses do not change, only need to do it once when the python program begins
        else:
            if self.first_epi:
                self.update_ob_om()
                if self.config.planner.om_inludes_human:
                    self.update_om_humans()

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