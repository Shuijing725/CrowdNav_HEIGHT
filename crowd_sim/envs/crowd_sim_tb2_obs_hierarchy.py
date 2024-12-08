import gym
import numpy as np
from numpy.linalg import norm
import os
import pybullet as p

try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass

from crowd_sim.envs.utils.info import *
from crowd_sim.envs.crowd_sim_tb2_obs import CrowdSim3DTbObs
from crowd_sim.envs.planning_utils.Astar_with_clearance import generate_astar_path

'''
Everything is the same as CrowdSim3DTbObs, except an A* planner during training AND testing to generate waypoints
'''

class CrowdSim3DTbObsHie(CrowdSim3DTbObs):
    def __init__(self):
        super().__init__()
        self.robot_waypoint_list = []
        self.om = None
        self.grid_size = None
        self.grid_num = None
        self.astar_succeed = True
        self.waypoint_uid = []

    def configure(self, config):
        super().configure(config)
        self.grid_size = self.config.planner.grid_resolution
        # make sure the robot can circulate around large obstacles
        self.om_boundary = self.config.sim.robot_circle_radius + 1

        # generate the matrix for 2D grids
        self.grid_num = int(np.ceil(self.om_boundary * 2 / self.grid_size))
        self.path_clearance = self.config.planner.path_clearance

        self.waypoint_reward = self.config.reward.waypoint_reward

    def set_observation_space(self):
        d = {}
        # px, py, theta + waypoints + final goal
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3+2*(self.config.planner.num_waypoints+1),), dtype=np.float32)
        # d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
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

        # 3. raw lidar point cloud
        d['point_clouds'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.ray_num,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

    '''
    Check if two rectangles overlap
    lower left of 1st rectangle: x1, y1
    upper right of 1st rectangle: x2, y2
    lower left of 2nd rectangle: ox1, oy1
    upper right of 2nd rectangle: ox2, oy2
    '''
    @staticmethod
    def check_rect_overlap(x1, y1, x2, y2, ox1, oy1, ox2, oy2):
        # Check if two rectangles overlap
        return not (x2 <= ox1 or x1 >= ox2 or y2 <= oy1 or y1 >= oy2)


    '''
    Check if a circle overlaps with a rectangle
    center x, center y, and radius of the circle: cx, cy, r
    lower left and upper right of the rectangle: rx1, ry1, rx2, ry2
    '''
    @staticmethod
    def check_circle_rect_overlap(cx, cy, r, rx1, ry1, rx2, ry2):
        # Check if a circle overlaps with a rectangle
        closest_x = max(rx1, min(cx, rx2))
        closest_y = max(ry1, min(cy, ry2))
        distance_squared = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
        return distance_squared <= r ** 2

    '''
    update self.om, a 2D matrix of 0s and 1s based on the obstacles in the current episode
    '''
    def update_om(self):
        # row: x axis, col: y axis
        self.om = np.zeros([self.grid_num, self.grid_num], dtype=int)

        # update self.om by filling 1s to grids that are occupied by self.cur_obstacles
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                # Calculate the coordinates of the current grid cell
                grid_x1 = -self.om_boundary + i * self.grid_size
                grid_y1 = -self.om_boundary + j * self.grid_size
                grid_x2 = grid_x1 + self.grid_size
                grid_y2 = grid_y1 + self.grid_size

                # Check overlap with each obstacle
                for obstacle in self.cur_obstacles:
                    if obstacle[4] in self.cylinder_obs_uids:
                        # It's a circle
                        obs_x1, obs_y1, obs_width, obs_height, obs_id = obstacle
                        r = obs_width/2
                        cx = obs_x1 + r
                        cy = obs_y1 + r

                        if self.check_circle_rect_overlap(cx, cy, r, grid_x1, grid_y1, grid_x2, grid_y2):
                            self.om[i][j] = 1
                            break  # No need to check other obstacles for this grid cell
                    else:
                        obs_x1, obs_y1, obs_width, obs_height, obs_id = obstacle
                        obs_x2 = obs_x1 + obs_width
                        obs_y2 = obs_y1 + obs_height

                        if self.check_rect_overlap(grid_x1, grid_y1, grid_x2, grid_y2, obs_x1, obs_y1, obs_x2, obs_y2):
                            self.om[i][j] = 1
                            break  # No need to check other obstacles for this grid cell



        self.om = np.transpose(self.om)

    def update_om_humans(self):
        self.om = np.transpose(self.om)
        # clear out previous humans from self.om
        self.om = self.om * self.om_human_mask

        # re-initialize self.om_human_mask
        # 1: the grid is not occupied by humans, 0: the grid is occupied by humans
        # used to clear out all humans from self.om, but keeps the obstacles
        self.om_human_mask = np.ones([self.grid_num, self.grid_num], dtype=int)
        # update self.om by filling 1s to grids that are occupied by self.cur_obstacles
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                # Calculate the coordinates of the current grid cell
                grid_x1 = -self.om_boundary + i * self.grid_size
                grid_y1 = -self.om_boundary + j * self.grid_size
                grid_x2 = grid_x1 + self.grid_size
                grid_y2 = grid_y1 + self.grid_size
                for k in range(len(self.humans)):
                    if self.check_circle_rect_overlap(self.humans[k].px, self.humans[k].py, self.humans[k].radius, grid_x1, grid_y1, grid_x2, grid_y2):
                        self.om[i][j] = 1
                        self.om_human_mask[i][j] = 0
                        break

        self.om = np.transpose(self.om)

    def create_goal_object(self):
        # visualize the final goal
        super().create_goal_object()
        # visualize the waypoints
        for i, waypoint in enumerate(self.robot_waypoint_list):
            if self.first_epi:
                # self.waypoint_uid.append(self.create_object(px=20, py=20, radius=0.15, height=2, shape=p.GEOM_SPHERE,
                #                                    color=[1./self.config.planner.num_waypoints*i, 0.6, 0, 1]))
                self.waypoint_uid.append(self.create_object(px=20, py=20, radius=0.15, height=2, shape=p.GEOM_SPHERE,
                                                            color=[1./3*5, 0.6, 0, 1]))

            if self.config.sim.render or self.config.camera.render_checkpoint:
                self._p.resetBasePositionAndOrientation(self.waypoint_uid[i],
                                                    [waypoint[0], waypoint[1], 2],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))

    def get_neighbors(self, x, y, offsets):
        coords = offsets + np.array([x, y])
        valid_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < self.om.shape[0]) &
                (coords[:, 1] >= 0) & (coords[:, 1] < self.om.shape[1])
        )
        valid_neighbors = coords[valid_mask]
        return valid_neighbors

    def check_neighbors_with_value(self, neighbors, value):
        if neighbors.size == 0:
            return np.array([])
        neighbors_values = self.om[neighbors[:, 0], neighbors[:, 1]]
        valid_value_indices = np.where(neighbors_values == value)[0]
        if valid_value_indices.size > 0:
            return neighbors[valid_value_indices]
        return np.array([])


    '''
    Given a (x, y) coordinate on self.om, check if its upper, lower, left, or right neighbors has a value of 0
    '''
    def check_orthogonal_neighbors(self, x, y):
        # Define the relative coordinates for the 4 orthogonal neighbors
        orthogonal_offsets = np.array([
            [-1, 0],  # top
            [1, 0],  # bottom
            [0, -1],  # left
            [0, 1]  # right
        ])

        # Calculate the absolute coordinates of the orthogonal neighbors
        orthogonal_coords = orthogonal_offsets + np.array([x, y])

        # Filter out neighbors that are out of bounds
        valid_mask = (
                (orthogonal_coords[:, 0] >= 0) & (orthogonal_coords[:, 0] < self.om.shape[0]) &
                (orthogonal_coords[:, 1] >= 0) & (orthogonal_coords[:, 1] < self.om.shape[1])
        )
        valid_neighbors = orthogonal_coords[valid_mask]

        # Check the values of the valid neighbors
        neighbors_values = self.om[valid_neighbors[:, 0], valid_neighbors[:, 1]]

        # Find any neighbor with value 0
        zero_neighbor_indices = np.where(neighbors_values == 0)[0]
        if zero_neighbor_indices.size > 0:
            first_zero_neighbor = valid_neighbors[zero_neighbor_indices[0]]
            return tuple(first_zero_neighbor)

        return None

    '''
    Given a (x, y) coordinate on self.om, check if any of its diagonal neighbor has a value of 0
    '''
    def check_diagonal_neighbors(self, x, y):
        # Define the relative coordinates for the 4 diagonal neighbors
        diagonal_offsets = np.array([
            [-1, -1],  # top-left
            [-1, 1],  # top-right
            [1, -1],  # bottom-left
            [1, 1]  # bottom-right
        ])

        # Calculate the absolute coordinates of the diagonal neighbors
        diagonal_coords = diagonal_offsets + np.array([x, y])

        # Filter out neighbors that are out of bounds
        valid_mask = (
                (diagonal_coords[:, 0] >= 0) & (diagonal_coords[:, 0] < self.om.shape[0]) &
                (diagonal_coords[:, 1] >= 0) & (diagonal_coords[:, 1] < self.om.shape[1])
        )
        valid_neighbors = diagonal_coords[valid_mask]

        # Check the values of the valid neighbors
        neighbors_values = self.om[valid_neighbors[:, 0], valid_neighbors[:, 1]]

        # Find any neighbor with value 0
        zero_neighbor_indices = np.where(neighbors_values == 0)[0]
        if zero_neighbor_indices.size > 0:
            first_zero_neighbor = valid_neighbors[zero_neighbor_indices[0]]
            return tuple(first_zero_neighbor)

        return None


    def check_second_layer_neighbors(self, x, y):
        second_layer_offsets = np.array([
            [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [0, -2], [0, -1], [0, 1], [0, 2],
            [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
            [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]
        ])
        neighbors = self.get_neighbors(x, y, second_layer_offsets)
        result = self.check_neighbors_with_value(neighbors, 0)
        return tuple(result[0]) if result.size > 0 else None


    '''
    Given a coordinate (x, y) on self.om, find a free neighbor from its 8 neighbor grids
    '''
    def find_free_neighbor(self, x, y):
        # First check the orthogonal neighbors
        result = self.check_orthogonal_neighbors(x, y)
        if result is not None:
            return result

        # If no orthogonal neighbor is found, check the diagonal neighbors
        result = self.check_diagonal_neighbors(x, y)
        if result is not None:
            return result
        result = self.check_second_layer_neighbors(x, y)
        if result is not None:
            return result
        # none of the neighbors is free, it's quite unlikely (unless goal/robot is inside obstables), and we just let planning fail here
        else:
            print('none of the neighbors is free')
            return x, y


    # use self.om, robot current position and goal position to plan waypoints
    # updates self.robot_waypoint_list, self.astar_succeed, self.visited_waypoints
    def plan_waypoints(self):
        # find the grid coordinate of robot (px, py) and (gx, gy)
        grid_px, grid_py = self.point_to_grid(self.robot.px, self.robot.py)

        # if robot px, py is occupied, find a closest neighbor to prevent planning failure

        if self.om[grid_py, grid_px] == 1:
            grid_py, grid_px = self.find_free_neighbor(grid_py, grid_px)
        # to deal with unknown error
        try:
            grid_gx, grid_gy = self.point_to_grid(self.robot.gx, self.robot.gy)
        except:
            print('Planning failed')
            self.robot_waypoint_list = np.tile([self.robot.gx, self.robot.gy], [self.config.planner.num_waypoints, 1])
            # print(self.robot_waypoint_list)
            self.astar_succeed = False
        # for the same reason, make sure gx, gy is not occupied
        if self.om[grid_gy, grid_gx] == 1:
            grid_gy, grid_gx = self.find_free_neighbor(grid_gy, grid_gx)

        # print('begin A*')
        # run A* algorithm to get planned path
        # if path is None, reduce self.path_clearance until it reaches 0
        cur_path_clearance = self.path_clearance
        while cur_path_clearance >= 0:
            path = generate_astar_path(
                # in nirrt* code, x is column, and y is row -> need to swap x and y axis
                1 - self.om,  # need to invert because in the A* code, 1 means traversable, 0 means occupied
                (grid_px, grid_py),
                (grid_gx, grid_gy),
                clearance=cur_path_clearance,
            )
            if path:
                break
            cur_path_clearance = cur_path_clearance - 1

        # print('Done A*')
        # if A* planner fails, repeat the [gx, gy] to fill the waypoints
        if path is None:
            print('Planning failed')

            self.robot_waypoint_list = np.tile([self.robot.gx, self.robot.gy], [self.config.planner.num_waypoints, 1])
            # print(self.robot_waypoint_list)
            self.astar_succeed = False
        else:
            # remove first 4 waypoints that are too close and goal position from path
            path = np.array(path)[1:-1]
            if len(path) > 0:
                # convert the (i, j) grids back to points (take the center of grids)
                path = self.waypoints_to_points(path)
            # print(path)
            # sample every "path_resolution" waypoints
            if len(path) > self.config.planner.num_waypoints:
                # print('sample')
                path_resolution = np.ceil(len(path) / (self.config.planner.num_waypoints+1) * 1.0)
                # print(path_resolution, self.config.planner.max_waypoint_resolution)
                path_resolution = min(path_resolution, self.config.planner.max_waypoint_resolution)
                self.robot_waypoint_list = path[int(path_resolution-1)::int(path_resolution)]
                self.robot_waypoint_list = self.robot_waypoint_list[:self.config.planner.num_waypoints]
                if len(self.robot_waypoint_list) < self.config.planner.num_waypoints:
                    self.robot_waypoint_list = np.vstack([self.robot_waypoint_list, path[-1]])
                # if len(self.robot_waypoint_list) == 5:
                #     a= 0
            elif len(path) == self.config.planner.num_waypoints:
                # print('equal')
                self.robot_waypoint_list = path
            else:
                # print('duplicate')
                extra_waypoints = np.tile([self.robot.gx, self.robot.gy], [self.config.planner.num_waypoints - len(path), 1])
                self.robot_waypoint_list = np.concatenate([path, extra_waypoints], axis=0)

            # print(self.robot_waypoint_list)
            self.astar_succeed = True

        # indicate whether the robot visited each waypoint or not, for reward calculation
        self.visited_waypoints = np.zeros(self.config.planner.num_waypoints, dtype=np.bool)

    def reset(self, phase='train', test_case=None):
        self.create_scenario(phase=phase, test_case=test_case)

        if self.config.planner.om_inludes_human:
            self.om_human_mask = np.ones([self.grid_num, self.grid_num], dtype=int)
        # create occupancy map and plan a path
        # in random env, need to reset self.om in the beginning of every episode
        if self.config.env.scenario == 'circle_crossing':
            self.update_om()
            if self.config.planner.om_inludes_human:
                self.update_om_humans()
        # otherwise, since the obstacle number and poses do not change, only need to do it once when the python program begins
        else:
            if self.first_epi:
                self.update_om()
                if self.config.planner.om_inludes_human:
                    self.update_om_humans()

        self.plan_waypoints()

        # create sphere shapes to visualize waypoints and the goal
        self.create_goal_object()

        ob = self.generate_ob(reset=True)
        return ob

    '''
    Given the robot current position and goal position, use A* to plan a new sequence of waypoints
    '''
    def replan(self):
        # replan every 'self.config.planner.replan_freq' timesteps OR when the robot reaches all waypoints
        robot_position = np.array([self.robot.px, self.robot.py])
        goal_position = np.array([self.robot.gx, self.robot.gy])
        # Compute the Euclidean distances
        rob_waypoint_dist = np.linalg.norm(self.robot_waypoint_list[-1] - robot_position)
        rob_goal_dist = np.linalg.norm(goal_position - robot_position)
        if (self.step_counter > 0 and self.step_counter % self.config.planner.replan_freq == 0) or rob_goal_dist < rob_waypoint_dist:
            # update humans on self.om
            if self.config.planner.om_inludes_human:
                self.update_om_humans()
            self.plan_waypoints()

            if self.config.sim.render or self.config.camera.render_checkpoint:
                for i, waypoint in enumerate(self.robot_waypoint_list):
                    self._p.resetBasePositionAndOrientation(self.waypoint_uid[i],
                                                            [waypoint[0], waypoint[1], 2],
                                                            self._p.getQuaternionFromEuler([0, 0, 0]))



    def step(self, action, update=True):
        # replan and update the object positions for waypoint visualization
        if self.config.planner.replan:
            self.replan()

        ob, reward, done, info = super().step(action, update=update)
        return ob, reward, done, info

    '''
    Given a point in the 2D environment, calculate its row and column number in self.om
    '''
    def point_to_grid(self, px, py):
        # Normalize point coordinates to grid indices
        grid_x = int(np.floor((px + self.om_boundary) / self.grid_size))
        grid_y = int(np.floor((py + self.om_boundary) / self.grid_size))

        # Ensure indices are within bounds
        grid_x = min(max(grid_x, 0), self.grid_num - 1)
        grid_y = min(max(grid_y, 0), self.grid_num - 1)

        return grid_x, grid_y

    '''
    Given a (row, col) number of a grid in self.om, calculate a point centered in the grid
    '''
    def grid_to_point(self, grid_x, grid_y):
        # Calculate the center of the grid cell
        px = -self.om_boundary + (grid_x + 0.5) * self.grid_size
        py = -self.om_boundary + (grid_y + 0.5) * self.grid_size
        return px, py

    def waypoints_to_points(self, waypoints):
        # Use np.vectorize to apply grid_to_point to all waypoints
        vectorized_grid_to_point = np.vectorize(self.grid_to_point)
        points_x, points_y = vectorized_grid_to_point(waypoints[:, 0], waypoints[:, 1])
        return np.column_stack((points_x, points_y))

    def generate_ob(self, reset):
        ob = super().generate_ob(reset=reset)
        # original: [px, py, gx, gy, theta]
        # new: [px, py, theta, subgoal1 x, subgoal1 y, ..., subgoaln x, subgoaln y, gx, gy
        ob['robot_node'] = ob['robot_node'][:2] + [ob['robot_node'][4]] + self.robot_waypoint_list.flatten().tolist() + ob['robot_node'][2:4]
        if len(ob['robot_node']) != 17:
            a=0
        return ob

    # finding the index of the last True value in the array and then setting all elements up to and including that index to True.
    @staticmethod
    def fill_up_to_last_true(arr):
        if np.any(arr):  # Check if there is at least one True
            last_true_index = np.where(arr)[0][-1]  # Find the last True index
            arr[:last_true_index + 1] = True  # Set all elements up to and including last True index to True
        return arr


    # added a subgoal reward
    def calc_reward(self, action, danger_zone='circle'):
        reward, done, episode_info = super().calc_reward(action, danger_zone=danger_zone)
        # only calculate waypoint reward if A* planner successfully returns waypoints
        if self.astar_succeed:
            # add waypoint reward
            robot_position = np.array([self.robot.px, self.robot.py])
            goal_position = np.array([self.robot.gx, self.robot.gy])
            # Compute the Euclidean distances
            rob_waypoint_dist = np.linalg.norm(self.robot_waypoint_list - robot_position, axis=1)
            waypoint_goal_dist = np.linalg.norm(self.robot_waypoint_list - goal_position, axis=1)
            rob_goal_dist = np.linalg.norm(goal_position - robot_position)
            arrived_waypoint = np.logical_and(rob_waypoint_dist < self.robot.radius * 2, waypoint_goal_dist < rob_goal_dist)
            new_waypoint = np.logical_and(arrived_waypoint, np.logical_not(self.visited_waypoints))
            if any(new_waypoint):
                reward = reward + self.waypoint_reward

                # the waypoint that robot arrives, as well as all previous waypoints, are considered visited
                # because if the robot skips waypoints, we don't want the robot to go back and collect reward from previous waypoints
                self.visited_waypoints[new_waypoint] = True

        return reward, done, episode_info