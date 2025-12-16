import gym
import numpy as np
from numpy.linalg import norm
import os
from collections import deque
import matplotlib.pyplot as plt
try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass

from crowd_sim.envs.crowd_sim_tb2 import CrowdSim3DTB

'''
Everything is the same as CrowdSimVarNum, except the obstacle vertices are in observation space
'''

# This class is the same as CrowdSim3DTB except observation space
class CrowdSim3DTbObs(CrowdSim3DTB):
    def __init__(self):
        super().__init__()
        # number of steps before we consider a human is stuck
        self.human_timeout_len = 10

    def set_observation_space(self):
        d={}
        # we use 'absolute' here
        if self.config.ob_space.robot_state == 'absolute':
            # robot px, py (in world frame), and theta (heading angle in z axis)
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5,), dtype=np.float32)
        else:
            # gx-px, gy-py, theta
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,3,), dtype = np.float32)
        # robot vx, vy (in world frame)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # make sure there's at least one human
        # add_human_vel is True
        if self.config.ob_space.add_human_vel:
            # [maximum number of humans, human state], where each human state = [human px - robot px, human py - robot py, human vx, human vy]
            # the frame of relative position can be changed in config.ob_space.human_state_frame in configs/config.py
            # the frame of velocity can be changed in in config.ob_space.human_vel in configs/config.py
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 4),
                                                dtype=np.float32)
        else:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 2),
                                                dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

        # obstacle representations methods:
        # Relative coordinates of 4 vertices w.r.t. the robot
        # # [lower left, lower_right, upper_right, upper_left]
        # where lower left = [lower left x  - robot.px, lower left y - robot.py], and same for others
        d['obstacle_vertices'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_obs_num), 8,), dtype=np.float32)

        # number of obstacles
        d['obstacle_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # 3. raw lidar point cloud from robot's 2D lidar
        d['point_clouds'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.ray_num,), dtype=np.float32)

        self.observation_space=gym.spaces.Dict(d)


    def generate_circle_crossing_human(self, static=False, route_type='independent'):
        human = super().generate_circle_crossing_human(static=static, route_type=route_type)

        # use a deque to store the recent speed of each human
        human.recent_speeds = deque(maxlen=self.human_timeout_len*2)
        human.recent_speeds.append(norm([human.vx, human.vy]))
        return human

    def step(self, action, update=True):
        if self.use_replay_spatial_edges and self.replay_step >= len(self.replay_spatial_edges):
            print("[Replay] End of replay log reached.")
            done = True
            ob = self.ob  # repeat last observation
            reward = 0
            info = {'info': 'End of replay'}
            return ob, reward, done, info

        ob, reward, done, info = super().step(action, update=update)
        for human in self.humans:
            human.recent_speeds.append(norm([human.vx, human.vy]))
        # For every self.human_timeout_len steps, check if a human is stuck for a while
        # if so, change its goal
        # Don't do this in csl_workspace scenarios because human routes are designed
        if self.config.env.scenario == 'circle_crossing':
            if self.step_counter % self.human_timeout_len == 0:
                for i in range(len(self.humans)):
                    if sum(self.humans[i].recent_speeds) / len(self.humans[i].recent_speeds) < 0.1:
                        self.update_human_goal(self.humans[i])
                        # print('new goal', self.humans[i].gx, self.humans[i].gy)

        return ob, reward, done, info

    def ray_test_no_humans(self):
        """
        perform a lidar ray test on all obstacles in the scene WITHOUT humans
        save the current range readings in self.closest_hit_dist
        """
        # relocate all pybullet human cylinders to somewhere out of lidar range
        for i in range(self.human_num):
            self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                    [30, 30, self.config.humans.height/2],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))
        # lidar ray test
        self.ray_test()

        # relocate humans back based on their original positions
        for i in range(self.human_num):
            self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                    [self.humans[i].px, self.humans[i].py, self.config.humans.height/2],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))

    # rotate a velocity vector [x, y] from world frame to robot frame
    def world_to_robot(self, vec):
        x, y = vec
        # rotation matrix from world to robot frame
        rot_angle = -(self.robot.theta - np.pi/2)
        R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                      [np.sin(rot_angle), np.cos(rot_angle)]])
        vec_trans = np.matmul(R, np.array([[x], [y]]))
        return np.array([vec_trans[0, 0], vec_trans[1, 0]])

    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        if self.config.ob_space.robot_state == 'absolute':
            ob['robot_node'] = self.robot.get_changing_state_list()
        else:
            ob['robot_node'] = self.robot.get_changing_state_list_goal_offset()

        self.update_last_human_states(self.human_visibility, reset=reset)

        # edges
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])


        if self.use_replay_spatial_edges:
            spatial_edges = self.replay_spatial_edges[self.replay_step]
            ob['spatial_edges'] = spatial_edges
            self.replay_step += 1

            # todo: do we need to convert human angle & positions??? (check ros_turtlebot_env.py)
            # # Use robot pose to convert each human's relative pos to world frame
            # theta = self.robot.theta
            # rot = np.array([[np.cos(theta), -np.sin(theta)],
            #                 [np.sin(theta), np.cos(theta)]])

            for i in range(min(self.human_num, spatial_edges.shape[0])):
                # rel_pos = spatial_edges[i, :2]
                # world_pos = np.dot(rot, rel_pos) # + np.array([self.robot.px, self.robot.py])
                #
                # # Move PyBullet cylinder
                # self.humans[i].px, self.humans[i].py = world_pos
                # self._p.resetBasePositionAndOrientation(
                #     self.humans[i].uid,
                #     [world_pos[0], world_pos[1], self.config.humans.height / 2],
                #     self._p.getQuaternionFromEuler([0, 0, 0])
                # )
                self.humans[i].px, self.humans[i].py = spatial_edges[i, :2]
                self._p.resetBasePositionAndOrientation(
                    self.humans[i].uid,
                    [ self.humans[i].px, self.humans[i].py, self.config.humans.height / 2],
                    self._p.getQuaternionFromEuler([0, 0, 0])
                )


            ob['detected_human_num'] = np.sum(np.linalg.norm(spatial_edges[:, :2], axis=1) < 15)

        else:
            # ([relative px, relative py, disp_x, disp_y], human id)
            # make sure there's at least one placeholder
            if self.config.ob_space.add_human_vel:
                all_spatial_edges = np.ones((max(1, self.max_human_num), 4)) * np.inf
            else:
                all_spatial_edges = np.ones((max(1, self.max_human_num), 2)) * np.inf
            # robot.vx and vy are in world frame, transform to robot frame first
            v_robot_robFrame = self.world_to_robot([self.robot.vx, self.robot.vy])
            for i in range(self.human_num):
                if self.human_visibility[i]:
                    # vector pointing from human i to robot (in world frame)
                    relative_pos = np.array(
                        [self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
                    # in self.last_human_states, the human relative positions are in world frame, transform to robot frame for sim2real
                    if self.config.ob_space.human_state_frame == 'robot':
                        all_spatial_edges[self.humans[i].id, :2] = self.world_to_robot(relative_pos)
                    else:
                        all_spatial_edges[self.humans[i].id, :2] = relative_pos
                    if self.config.ob_space.add_human_vel:
                        if self.config.ob_space.human_state_frame == 'robot':
                            # in self.last_human_states, the human velocities are in world frame, transform to robot frame for sim2real
                            v_human = self.world_to_robot(self.last_human_states[i, 2:4])
                            # subtract robot velocity from human velocity, to get relative velocity of this human w.r.t. robot
                            if self.config.ob_space.human_vel == 'relative':
                                all_spatial_edges[self.humans[i].id, 2:] = v_human - v_robot_robFrame
                            else:
                                all_spatial_edges[self.humans[i].id, 2:] = v_human
                        else:
                            all_spatial_edges[self.humans[i].id, 2:] = self.last_human_states[i, 2:4]

            # sort all humans by distance (invisible humans will be in the end automatically)
            ob['spatial_edges'] = np.array(sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x[:2])))
            ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15

            ob['detected_human_num'] = num_visibles
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        # obstacle representations methods:
        # 1. relative coordinates of 4 vertices w.r.t. the robot

        ob['obstacle_vertices'] = np.ones((max(1, self.max_obs_num), 8,)) * 15
        # obstacle vertices in world frame
        if self.config.ob_space.human_state_frame == 'robot':
            cur_obs_vertices = np.array(self.obstacle_coord) - np.array([self.robot.px, self.robot.py])
            # convert obstacle vertics from world to robot frame
            cur_obs_vertices_rob_frame = np.zeros((self.obs_num, 4, 2))
            for i in range(self.obs_num):
                for j in range(4):
                    cur_obs_vertices_rob_frame[i, j] = self.world_to_robot(cur_obs_vertices[i, j])

            ob['obstacle_vertices'][:self.obs_num] = cur_obs_vertices_rob_frame.reshape(self.obs_num, -1)
        else:
            cur_obs_vertices = (np.array(self.obstacle_coord) - np.array([self.robot.px, self.robot.py])).reshape(self.obs_num, -1)
            ob['obstacle_vertices'][:self.obs_num] = cur_obs_vertices

        # 2. coordinates of bounding boxes (aabb)
        ob['obstacle_num'] = self.obs_num

        # 3. raw lidar point cloud
        # include everything (humans and obs) in lidar pc scan
        if self.config.ob_space.lidar_pc_include_humans:
            self.ray_test()
            ob['point_clouds'] = np.expand_dims(self.closest_hit_dist, axis=0)
        # only include obs
        # a. all obs in one lidar scan
        # b. seperate each obs for robot-obstacle attn
        else:
            self.ray_test_no_humans()
            ob['point_clouds'] = np.expand_dims(self.closest_hit_dist, axis=0)

        # update self.observed_human_ids
        self.observed_human_ids = np.where(self.human_visibility)[0]
        self.ob = ob

        return ob
