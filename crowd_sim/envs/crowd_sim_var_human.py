import gym
import numpy as np
from numpy.linalg import norm
import copy

from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import *
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.orca import ORCA

'''
This environment contains all non-pybullet part, 
mainly the logic to span the ORCA humans and the robot, and stepping the ORCA humans
'''

class CrowdSimVarNum(CrowdSim):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()
        self.desiredVelocity = [0.0, 0.0]

        self.id_counter = None
        self.observed_human_ids = None

        self.static_human_in_hallway = False
        self.human_facing_robot = False


    def set_observation_space(self):
        d = {}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.human_num + self.config.sim.human_num_range, 2),
                                            dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(d)

    def set_action_space(self):
        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def set_robot(self, robot):
        self.robot = robot

        # set observation space and action space
        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need
        self.set_observation_space()
        self.set_action_space()

    def line_intersects_rectangles(self, px, py, gx, gy, rectangles):
        # Calculate the direction of the line
        dx, dy = gx - px, gy - py

        # Rectangle edges
        x1, y1, w, h = rectangles[:, 0], rectangles[:, 1], rectangles[:, 2], rectangles[:, 3]
        x2, y2 = x1 + w, y1 + h

        # Line segment and rectangle intersection parameters
        t1 = (x1 - px) / dx
        t2 = (x2 - px) / dx
        t3 = (y1 - py) / dy
        t4 = (y2 - py) / dy

        # Find intersection points
        tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
        tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))

        # Determine if intersection points are within the segment
        intersects = np.logical_and(tmin <= 1, tmax >= 0)

        # Filter valid intersections
        valid_intersections = np.where(tmin <= tmax, intersects, False)

        # Count the number of intersecting rectangles
        return np.sum(valid_intersections)

    def generate_robot(self):
        theta = np.random.uniform(self.config.robot.initTheta_range[0], self.config.robot.initTheta_range[1])
        num_obs_intersects = 4
        trial = 0
        while True:
            trial = trial + 1
            if trial > 100:
                trial = 0
                num_obs_intersects = min(0, num_obs_intersects - 1)
            if self.config.env.scenario == 'circle_crossing':
                px = np.random.uniform(self.config.robot.initX_range[0], self.config.robot.initX_range[1])
                py = np.random.uniform(self.config.robot.initY_range[0], self.config.robot.initY_range[1])
                gx = np.random.uniform(self.config.robot.goalX_range[0], self.config.robot.goalX_range[1])
                gy = np.random.uniform(self.config.robot.goalY_range[0], self.config.robot.goalY_range[1])

            else:
                route_idx = np.random.choice(len(self.config.robot.routes))
                start_region, goal_region = self.config.robot.routes[route_idx]
                px = np.random.uniform(self.config.robot.regions[start_region][0],
                                       self.config.robot.regions[start_region][1])
                py = np.random.uniform(self.config.robot.regions[start_region][2],
                                       self.config.robot.regions[start_region][3])
                gx = np.random.uniform(self.config.robot.regions[goal_region][0],
                                       self.config.robot.regions[goal_region][1])
                gy = np.random.uniform(self.config.robot.regions[goal_region][2],
                                       self.config.robot.regions[goal_region][3])
            self.goal_dist = np.linalg.norm([px - gx, py - gy])
            if self.add_static_obs:
                if self.config.env.scenario == 'circle_crossing':
                    pass_cond = self.config.robot.min_goal_dist <= self.goal_dist <= self.config.robot.max_goal_dist \
                                and not self.circle_in_obstacles(gx, gy, self.robot.radius * 2) \
                                and not self.circle_in_obstacles(px, py, self.robot.radius * 2) \
                                and self.line_intersects_rectangles(px, py, gx, gy, self.cur_obstacles) >= num_obs_intersects

                else:
                    pass_cond = not self.circle_in_obstacles(gx, gy, self.robot.radius * 2) \
                                and not self.circle_in_obstacles(px, py, self.robot.radius * 2)
            else:
                pass_cond = self.config.robot.min_goal_dist <= self.goal_dist <= self.config.robot.max_goal_dist

            if pass_cond:
                break

        self.robot.set(px, py, gx, gy, 0, 0, theta)

    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        # print('begin generate robot')
        # generate the robot
        self.generate_robot()
        # print('done generate robot')
        # generate humans

        # select number of static humans
        self.static_human_num = np.random.randint(low=self.config.sim.static_human_num - self.config.sim.static_human_range,
                                           high=self.config.sim.static_human_num + self.config.sim.static_human_range + 1)
        # select number of dynamic humans
        # if the routes of all humans are correlated, chosen the routes for all of them here
        if self.config.env.scenario == 'csl_workspace' and self.config.human_flow.route_type == 'correlated':
            route_idx = np.random.choice(len(self.config.human_flow.correlated_routes))
            self.human_routes = copy.deepcopy(self.config.human_flow.correlated_routes[route_idx])
            self.dynanmic_human_num = len(self.config.human_flow.correlated_routes[route_idx])
            # print(self.human_routes)
            self.generate_random_human_position(human_num=self.dynanmic_human_num,
                                                static_human_num=self.static_human_num, route_type='correlated')
        elif self.config.env.scenario == 'csl_workspace' and self.config.human_flow.route_type == 'mixed':
            route_idx = np.random.choice(len(self.config.human_flow.correlated_routes))
            self.human_routes = copy.deepcopy(self.config.human_flow.correlated_routes[route_idx])
            dynanmic_human_num = np.random.randint(low=self.config.sim.human_num - self.human_num_range,
                                               high=self.config.sim.human_num + self.human_num_range + 1)
            self.correlated_human_num = len(self.config.human_flow.correlated_routes[route_idx])
            self.independent_human_num = max(0, dynanmic_human_num-self.correlated_human_num)
            self.dynanmic_human_num = self.correlated_human_num + self.independent_human_num
            self.generate_random_human_position(human_num=self.correlated_human_num,
                                                static_human_num=self.static_human_num, route_type='correlated')
            self.generate_random_human_position(human_num=self.independent_human_num,
                                                static_human_num=0, route_type='independent')
        else:
            self.dynanmic_human_num = np.random.randint(low=self.config.sim.human_num - self.human_num_range,
                                               high=self.config.sim.human_num + self.human_num_range + 1)
            self.generate_random_human_position(human_num=self.dynanmic_human_num,
                                                static_human_num=self.static_human_num, route_type='independent')

        self.human_num = self.dynanmic_human_num + self.static_human_num
        # todo: for mixed (some humans correlated, some independent, we just need to call this function twice (first with human_num=# of correlated humans, then human_num=# of independent humans)
        # todo: add an argument for correlated/independent


        self.last_human_states = np.zeros((self.human_num, 5))
        # set human ids
        for i in range(self.human_num):
            self.humans[i].id = i


    # more uniform circle crossing scenario
    def generate_circle_crossing_human(self, static=False, route_type='independent'):
        # print('begin generate human')
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        noise_range = self.config.sim.human_pos_noise_range
        trial = 0

        # if the routes of all humans are correlated, the routes are already chosen and stored in self.human_routes
        if not static and route_type == 'correlated':
            this_human_route = self.human_routes.pop()

        while True:
            trial = trial + 1
            # print(trial)
            # to prevent the program from stucking in this loop
            if trial > 100:
                trial = 0
                noise_range = noise_range * 1.5

            # designed human flow
            if self.config.env.scenario == 'csl_workspace':
                if static:
                    if self.config.env.mode == 'sim2real' and self.static_human_in_hallway:
                        region_idx = np.random.choice(len(self.config.human_flow.static_regions)-1)
                    else:
                        if self.config.env.mode == 'sim2real' and self.config.env.csl_workspace_type == 'gdc':
                            try:
                                region_idx = np.random.choice(len(self.config.human_flow.static_regions), p=self.config.human_flow.static_regions_prob)
                            except:
                                region_idx = np.random.choice(len(self.config.human_flow.static_regions))
                    # print(self.config.human_flow.static_regions_prob)
                    # print(self.config.human_flow.static_regions)
                    # print(region_idx)
                    # region_idx = np.random.choice(len(self.config.human_flow.static_regions))
                    px = gx = np.random.uniform(self.config.human_flow.static_regions[region_idx, 0],
                                                self.config.human_flow.static_regions[region_idx, 1])
                    py = gy = np.random.uniform(self.config.human_flow.static_regions[region_idx, 2],
                                                self.config.human_flow.static_regions[region_idx, 3])

                else:

                    # choose route for this human
                    if route_type == 'independent':
                        if self.human_facing_robot:
                            route_idx = np.random.choice(len(self.config.human_flow.routes)-4)
                        else:
                            route_idx = np.random.choice(len(self.config.human_flow.routes))

                        human.route = copy.deepcopy(self.config.human_flow.routes[route_idx])
                    # else if self.config.human_flow.route_type == 'correlated', human route is already chosen above
                    else:
                        human.route = copy.deepcopy(this_human_route)
                    # print(human.route)
                    start_region = human.route.pop(0)
                    goal_region = human.route.pop(0)

                    # print('start region', start_region, 'goal_region', goal_region)

                    px = np.random.uniform(self.config.human_flow.regions[start_region][0], self.config.human_flow.regions[start_region][1])
                    py = np.random.uniform(self.config.human_flow.regions[start_region][2],
                                           self.config.human_flow.regions[start_region][3])
                    gx = np.random.uniform(self.config.human_flow.regions[goal_region][0], self.config.human_flow.regions[goal_region][1])
                    gy = np.random.uniform(self.config.human_flow.regions[goal_region][2],
                                           self.config.human_flow.regions[goal_region][3])
            # circle crossing
            else:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = np.random.uniform(-1, 1) * noise_range
                py_noise = np.random.uniform(-1, 1) * noise_range
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                gx = -px
                gy = -py

            collide_obs = False
            # check collision of start and goal position with static obstacles
            if self.config.action_space.kinematics == 'jackal':
                collide_obs = False
            else:
                if self.add_static_obs:
                    start_collision = self.circle_in_obstacles(px, py, human.radius + self.discomfort_dist)
                    goal_collision = self.circle_in_obstacles(gx, gy, human.radius + self.discomfort_dist)
                    if start_collision or goal_collision:
                        collide_obs = True
                        # print('collide obs')

            # check collision with the robot and all other existing humans
            collide_human = self.check_collision((px, py), human.radius, static=static)

            collide_robot_goal = False
            if static:
                if np.linalg.norm([px-self.robot.gx, py-self.robot.gy]) <= (self.robot.radius + human.radius)*2:
                    collide_robot_goal = True
                    # print('collide robot goal')

            if not collide_obs and not collide_human and not collide_robot_goal:
                break

        # adjust scenario difficulty
        if static:
            if self.config.env.scenario == 'csl_workspace' and self.config.env.mode == 'sim2real' and region_idx == 2:
                self.static_human_in_hallway = True
        else:
            if self.config.env.scenario == 'csl_workspace' and self.config.env.csl_workspace_type == 'lounge' and \
                    3 in self.config.human_flow.routes[route_idx] and 1 in self.config.human_flow.routes[route_idx]:
                self.human_facing_robot = True
            if self.config.env.scenario == 'csl_workspace' and self.config.env.csl_workspace_type == 'lounge' and start_region == 7:
                if np.random.rand() > 0.5:
                    human.isObstacle = True
                    # human.isObstacle_period = np.random.uniform(0, self.config.env.time_limit / self.config.env.time_step)
                    human.isObstacle_period = np.random.normal(120, 80)
                    human.isObstacle_period = np.clip(human.isObstacle_period, 0, self.config.env.time_limit / self.config.env.time_step)
                    # print(human.isObstacle_period)
        # print('chosen px, py, gx, gy', px, py, gx, gy)
        # print('human dist to rob:', np.linalg.norm([px - self.robot.px, py - self.robot.py]))
        human.set(px, py, gx, gy, 0, 0, 0)

        # # shift the center of circle up so that the bottom of the circle is at (0, 0)
        if self.robot.kinematics == 'turtlebot':
            if static:
                human.react_to_robot = False
            else:
                if np.random.rand() <= self.config.robot.visible_prob:
                    human.react_to_robot = True
                else:
                    human.react_to_robot = False

        # if we have static obstacles, feed them to the humans' ORCA policy so that the humans can avoid them
        if self.add_static_obs:
            if self.config.humans.policy == 'orca':
                assert isinstance(human.policy, ORCA)
                human.policy.static_obs = self.obstacle_coord

        # print('Done with this human')
        return human

    # calculate the ground truth future trajectory of humans
    # if robot is visible: assume linear motion for robot
    # ret val: [self.predict_steps + 1, self.human_num, 4]
    # method: 'truth' or 'const_vel' or 'inferred'
    def calc_human_future_traj(self, method):
        # if the robot is invisible, it won't affect human motions
        # else it will
        human_num = self.human_num + 1 if self.robot.visible else self.human_num
        # buffer to store predicted future traj of all humans [px, py, vx, vy]
        # [time, human id, features]
        if method == 'truth':
            self.human_future_traj = np.zeros((self.buffer_len + 1, human_num, 4))
        elif method == 'const_vel':
            self.human_future_traj = np.zeros((self.predict_steps + 1, human_num, 4))
        else:
            raise NotImplementedError

        # initialize the 0-th position with current states
        for i in range(self.human_num):
            # use true states for now, to count for invisible humans' influence on visible humans
            # take px, py, vx, vy, remove radius
            self.human_future_traj[0, i] = np.array(self.humans[i].get_observable_state_list()[:-1])

        # if we are using constant velocity model, we need to use displacement to approximate velocity (pos_t - pos_t-1)
        # we shouldn't use true velocity for fair comparison with GST inferred pred
        if method == 'const_vel':
            if self.robot.visible:
                self.human_future_traj[0, :-1, 2:4] = self.prev_human_pos[:, 2:4]
            else:
                self.human_future_traj[0, :, 2:4] = self.prev_human_pos[:, 2:4]

        if self.robot.visible:
            self.human_future_traj[0, -1] = np.array(self.robot.get_observable_state_list()[:-1])

        if method == 'truth':
            for i in range(1, self.buffer_len + 1):
                for j in range(self.human_num):
                    # prepare joint state for all humans
                    full_state = np.concatenate(
                        (self.human_future_traj[i - 1, j], self.humans[j].get_full_state_list()[4:]))
                    observable_states = []
                    for k in range(self.human_num):
                        if j == k:
                            continue
                        observable_states.append(
                            np.concatenate((self.human_future_traj[i - 1, k], [self.humans[k].radius])))

                    # use joint states to get actions from the states in the last step (i-1)
                    action = self.humans[j].act_joint_state(JointState(full_state, observable_states))

                    # step all humans with action
                    self.human_future_traj[i, j] = self.humans[j].one_step_lookahead(
                        self.human_future_traj[i - 1, j, :2], action)

                if self.robot.visible:
                    action = ActionXY(*self.human_future_traj[i - 1, -1, 2:])
                    # update px, py, vx, vy
                    self.human_future_traj[i, -1] = self.robot.one_step_lookahead(self.human_future_traj[i - 1, -1, :2],
                                                                                  action)
            # only take predictions every self.pred_interval steps
            self.human_future_traj = self.human_future_traj[::self.pred_interval]
        # for const vel model
        elif method == 'const_vel':
            # [self.pred_steps+1, human_num, 4]
            self.human_future_traj = np.tile(self.human_future_traj[0].reshape(1, human_num, 4), (self.predict_steps+1, 1, 1))
            # [self.pred_steps+1, human_num, 2]
            pred_timestep = np.tile(np.arange(0, self.predict_steps+1, dtype=float).reshape((self.predict_steps+1, 1, 1)) * self.time_step * self.pred_interval,
                                    [1, human_num, 2])
            pred_disp = pred_timestep * self.human_future_traj[:, :, 2:]
            self.human_future_traj[:, :, :2] = self.human_future_traj[:, :, :2] + pred_disp
        else:
            raise NotImplementedError

        # remove the robot if it is visible
        if self.robot.visible:
            self.human_future_traj = self.human_future_traj[:, :-1]


        # remove invisible humans
        self.human_future_traj[:, np.logical_not(self.human_visibility), :2] = 15
        self.human_future_traj[:, np.logical_not(self.human_visibility), 2:] = 0

        return self.human_future_traj

    def calc_potential_reward(self):
        # potential reward
        potential_cur = np.linalg.norm(
            np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
        reward = self.pot_factor * (-abs(potential_cur) - self.potential)
        # print(reward)
        # clip the reward within the maximum theoretical range
        reward = np.clip(reward, -self.max_abs_pot_reward, self.max_abs_pot_reward)
        self.potential = -abs(potential_cur)
        # print(reward)
        return reward

    # find R(s, a), done or not, and episode information
    def calc_reward(self, action, danger_zone='circle'):

        # collision checking
        collision, dmin, collision_with = self.collision_checker()

        # check if reaching the goal
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.goal_reach_dist

        # use danger_zone to determine the condition for Danger
        if danger_zone == 'circle' or self.phase == 'train':
            danger_cond = dmin < self.discomfort_dist
            min_danger_dist = 0
        else:
            # if the robot collides with future states, give it a collision penalty
            relative_pos = self.human_future_traj[1:, :, :2] - np.array([self.robot.px, self.robot.py])
            relative_dist = np.linalg.norm(relative_pos, axis=-1)

            collision_idx = relative_dist < self.robot.radius + self.config.humans.radius  # [predict_steps, human_num]

            danger_cond = np.any(collision_idx)
            # if robot is dangerously close to any human, calculate the min distance between robot and its closest human
            if danger_cond:
                min_danger_dist = np.amin(relative_dist[collision_idx])
            else:
                min_danger_dist = 0

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            if collision_with == 'human':
                episode_info = CollisionHuman()
            elif collision_with == 'obstacle':
                episode_info = CollisionObs()
            else:
                episode_info = CollisionWall()

        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()

        elif danger_cond:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            # print(dmin)
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            # print('discomfort reward:', reward)
            done = False
            episode_info = Danger(min_danger_dist)

        else:
            reward = self.calc_potential_reward()

            done = False
            episode_info = Nothing()

        # if the robot is near collision/arrival, it should be able to turn a large angle
        if self.robot.kinematics in ['unicycle', 'turtlebot', 'jackal']:
            if self.robot.kinematics == 'unicycle':
                # if action.r is w, factor = -0.02 if w in [-1.5, 1.5], factor = -0.045 if w in [-1, 1];
                # if action.r is delta theta, factor = -2 if r in [-0.15, 0.15], factor = -4.5 if r in [-0.1, 0.1]
                w = action.r
                v = action.v
            else:
                w = self.robot.w
                v = self.robot.v
            # add a rotational penalty

            r_spin = -self.config.reward.spin_factor * w ** 2
            # clip the reward within the maximum theoretical range
            r_spin = np.clip(r_spin, -self.max_abs_rot_penalty, self.max_abs_rot_penalty)

            # prevent jackal from spinning in place
            if self.config.action_space.kinematics == 'jackal':
                if self.desiredVelocity[0] == 0 and self.desiredVelocity[1] != 0:
                    # print('spin in place')
                    r_spin = -0.5

            # add a penalty for going backwards
            # print(v)
            if v < 0:
                r_back = -self.config.reward.back_factor * abs(v)
            else:
                r_back = 0.
            r_back = np.clip(r_back, -self.max_abs_back_penalty, self.max_abs_back_penalty)
            # print('original r:', reward, 'r spin:', r_spin, 'r_back:', r_back)
            reward = reward + r_spin + r_back + self.config.reward.constant_penalty

        # print(reward)
        return reward, done, episode_info

    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        ob['robot_node'] = self.robot.get_full_state_list_noV()

        self.update_last_human_states(self.human_visibility, reset=reset)

        # edges
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        # ([relative px, relative py, disp_x, disp_y], human id)
        all_spatial_edges = np.ones((self.config.sim.human_num + self.config.sim.human_num_range, 2)) * np.inf

        for i in range(self.human_num):
            if self.human_visibility[i]:
                # vector pointing from human i to robot
                relative_pos = np.array(
                    [self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
                all_spatial_edges[self.humans[i].id, :2] = relative_pos

        # sort all humans by distance (invisible humans will be in the end automatically)
        ob['spatial_edges'] = np.array(sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x)))
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15

        ob['detected_human_num'] = num_visibles
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        # update self.observed_human_ids
        self.observed_human_ids = np.where(self.human_visibility)[0]
        self.ob = ob

        return ob



    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0


        self.humans = []
        # self.human_num = self.config.sim.human_num
        # initialize a list to store observed humans' IDs
        self.observed_human_ids = []

        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        # print(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        # np.random.seed(1038)

        # generate static obstacles
        if self.add_static_obs:
            self.obs_num = np.random.randint(self.avg_obs_num - self.obs_range, self.avg_obs_num + self.obs_range+1)
            self.generate_rectangle_obstacle(self.obs_num)

        # generate the robot and humans
        self.generate_robot_humans(phase)
        
        # initialize self.human_visibility
        _, _, self.human_visibility = self.get_num_human_in_fov()

        # record px, py, r of each human, used for crowd_sim_pc env
        self.cur_human_states = np.zeros((self.max_human_num, 3))
        for i in range(self.human_num):
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])



        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()


        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get robot observation
        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        return ob


    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.robot.policy.name == 'ORCA':
            # assemble observation for orca: px, py, vx, vy, r
            human_states = copy.deepcopy(self.last_human_states)
            # get orca action
            action = self.robot.act(human_states.tolist())
        else:
            action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == 'unicycle':
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
            action = ActionRot(self.desiredVelocity[0], action.r)

        human_actions = self.get_human_actions()

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)

        # apply action and update all agents
        # apply robot action
        if self.add_static_obs:
            # if next position is collision free with respect to obstacles, step; otherwise stay here
            next_px, next_py = self.robot.compute_position(action, delta_t=self.time_step)
            if not self.circle_in_obstacles(next_px, next_py, self.robot.radius):
                self.robot.step(action)
        else:
            self.robot.step(action)

        # apply humans' actions
        # record px, py, r of each human, used for crowd_sim_pc env
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter = self.step_counter + 1

        info={'info':episode_info}

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.config.sim.change_human_num_in_episode:
            self.change_human_num_periodically()

        # compute the observation
        ob = self.generate_ob(reset=False)


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.humans[i] = self.generate_circle_crossing_human()
                    self.humans[i].id = i

        return ob, reward, done, info

    def change_human_num_periodically(self):
        if self.human_num_range > 0 and self.global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # print('before:', self.human_num,', self.min_human_num:', self.min_human_num)
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - self.min_human_num
                    # print('max_remove_num, invisible', max_remove_num)
                else:
                    max_remove_num = min(self.human_num - self.min_human_num, (self.human_num - 1) - max(self.observed_human_ids))
                    # print('max_remove_num, visible', max_remove_num)
                remove_num = np.random.randint(low=0, high=max_remove_num + 1)
                for _ in range(remove_num):
                    self.humans.pop()
                self.human_num = self.human_num - remove_num
                # print('after:', self.human_num)
                self.last_human_states = self.last_human_states[:self.human_num]
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    true_add_num = 0
                    for i in range(self.human_num, self.human_num + add_num):
                        if i == self.config.sim.human_num + self.human_num_range:
                            break
                        self.generate_random_human_position(human_num=1)
                        self.humans[i].id = i
                        true_add_num = true_add_num + 1
                    self.human_num = self.human_num + true_add_num
                    if true_add_num > 0:
                        self.last_human_states = np.concatenate((self.last_human_states, np.array([[15, 15, 0, 0, 0.3]]*true_add_num)), axis=0)
        # if not self.min_human_num <= self.human_num <= self.max_human_num:
        #     a = 0
        assert self.min_human_num <= self.human_num <= self.max_human_num

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint



        ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)


        # draw FOV for the robot
        # add robot FOV
        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add an arc of robot's sensor range
        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range, fill=False, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.humans]
        human_belief_circles = [plt.Circle((self.last_human_states[i, 0], self.last_human_states[i, 1]),
                                           self.last_human_states[i, 4], fill=True) for i in range(self.human_num)]

        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            ax.add_artist(human_belief_circles[i])
            artists.append(human_belief_circles[i])

            # green: visible; red: invisible
            if self.detect_visible(self.robot, self.humans[i], robot1=True):
                human_circles[i].set_color(c='g')
            else:
                human_circles[i].set_color(c='r')
            if self.humans[i].id in self.observed_human_ids:
                human_circles[i].set_color(c='b')
            # label numbers on each human
            plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
            plt.text(self.last_human_states[i, 0] - 0.1, self.last_human_states[i, 1] - 0.1, str(i), color='black',
                     fontsize=12)



        plt.pause(0.01)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)
