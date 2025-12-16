import numpy as np
import rvo2
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
import math
from enum import Enum
import matplotlib.pyplot as plt
# import imageio
from os.path import exists
from os import makedirs, remove

class RobotType(Enum):
    circle = 0
    rectangle = 1

class DWA(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.name = "DWA"
        self.max_speed = config.robot.v_max  # [m/s]
        self.min_speed = config.robot.v_min  # [m/s]
        self.max_yaw_rate = config.robot.w_max  # [rad/s]
        self.max_accel = config.dwa.max_accel  # [m/ss]
        self.max_delta_yaw_rate = config.dwa.max_delta_yaw_rate # math.pi # [rad/ss]
        self.v_resolution = config.dwa.v_resolution # [m/s]
        self.yaw_rate_resolution = config.dwa.yaw_rate_resolution # 0.1 * math.pi # [rad/s]
        self.dt = config.env.time_step  # [s] Time tick for motion prediction
        self.predict_time = config.dwa.predict_time # [s]
        self.to_goal_cost_gain = config.dwa.to_goal_cost_gain
        self.speed_cost_gain = config.dwa.speed_cost_gain
        self.obstacle_cost_gain = config.dwa.obstacle_cost_gain
        self.robot_stuck_flag_cons = config.dwa.robot_stuck_flag_cons  # constant to prevent robot stucked
        if config.dwa.robot_type == 0:
            self.robot_type = RobotType.circle
        elif config.dwa.robot_type == 1:
            self.robot_type = RobotType.rectangle
        self.dynamics_weight = config.dwa.dynamics_weight
        self.stuck_action = config.dwa.stuck_action

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = config.robot.radius  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = config.dwa.robot_width  # [m] for collision check
        self.robot_length = config.dwa.robot_length  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = config.dwa.ob
        
        self.boundary = config.dwa.boundary
        self.boundary_width = config.dwa.boundary_width
        self.boundary_height = config.dwa.boundary_height
        
        
    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value
    
    # Add rectangular obstacles with rectangular robot
    def rectangle_obstacle_rectangle_robot(self, x, y, robot_width, robot_length):
        interval = min(robot_width, robot_length)
        obstacle = []
        obstacle.append([x[1], y[1]])
        for i in np.arange(x[0], x[1], interval):
            new_pos_1 = [i, y[0]]
            new_pos_2 = [i, y[1]]
            if new_pos_1 not in obstacle:
                obstacle.append(new_pos_1)
            if new_pos_2 not in obstacle:
                obstacle.append(new_pos_2)
    
        for j in np.arange(y[0], y[1], interval):
            new_pos_1 = [x[0], j]
            new_pos_2 = [x[1], j]
            if new_pos_1 not in obstacle:
                obstacle.append(new_pos_1)
            if new_pos_2 not in obstacle:
                obstacle.append(new_pos_2)
        return np.array(obstacle)
    
    # Add rectangular obstacles with circular robot
    def rectangle_obstacle_circle_robot(self, x, y, robot_radius):
        interval = robot_radius
        obstacle = []
        obstacle.append([x[1], y[1]])
        for i in np.arange(x[0], x[1], interval):
            new_pos_1 = [i, y[0]]
            new_pos_2 = [i, y[1]]
            if new_pos_1 not in obstacle:
                obstacle.append(new_pos_1)
            if new_pos_2 not in obstacle:
                obstacle.append(new_pos_2)
    
        for j in np.arange(y[0], y[1], interval):
            new_pos_1 = [x[0], j]
            new_pos_2 = [x[1], j]
            if new_pos_1 not in obstacle:
                obstacle.append(new_pos_1)
            if new_pos_2 not in obstacle:
                obstacle.append(new_pos_2)
        return np.array(obstacle)
    
    # Add circular obstacles with rectangular robot    
    def circle_obstacle_rectangle_robot(self, center, radius, robot_width, robot_length):    
        min_side = min(robot_width, robot_length)
        theta = np.arccos((radius ** 2 + radius ** 2 - min_side ** 2) / (2*radius*radius))
        n = int(np.floor(2 * np.pi / theta))
        obstacle = []
        curr = [center[0], center[1]+radius]
        obstacle.append(curr)
        for i in range(n):
            curr = [center[0]+np.sin(theta*(i+1))*radius, center[1]+np.cos(theta*(i+1))*radius]
            if curr not in obstacle:
                obstacle.append(curr)
        return np.array(obstacle)
    
    # Add circular robot with circular robots
    def circle_obstacle_circle_robot(self, center, radius, robot_radius):
        min_side = robot_radius
        theta = np.arccos((radius ** 2 + radius ** 2 - min_side ** 2) / (2*radius*radius))
        n = int(np.floor(2 * np.pi / theta))
        obstacle = []
        curr = [center[0], center[1]+radius]
        obstacle.append(curr)
        for i in range(n):
            curr = [center[0]+np.sin(theta*(i+1))*radius, center[1]+np.cos(theta*(i+1))*radius]
            if curr not in obstacle:
                obstacle.append(curr)
        return np.array(obstacle)
    
    # Run dwa
    # x is current state
    # goal is the coordinate of goal position
    # ob is obstacles
    
    
    def set_obstacle(self, obstacles, humans, clean = False):
        if clean:
            self.ob = []
        for i in obstacles:
            x = [i[0], i[0] + i[2]]
            y = [i[1], i[1] + i[3]]
            # if config.robot_type == RobotType.circle:
            if self.ob == []:
                self.ob = self.rectangle_obstacle_circle_robot(x, y, self.robot_radius)
            else:
                self.ob = np.concatenate((self.ob, self.rectangle_obstacle_circle_robot(x, y, self.robot_radius)), axis = 0)
            # else:
            #    config.ob.append(rectangle_obstacle_rectangle_robot(x, y, config.robot_width, config.robot_length))
            
        for i in humans:
            #if config.robot_type == RobotType.circle:
            self.ob = np.concatenate((self.ob, self.circle_obstacle_circle_robot([i.px, i.py], i.radius, self.robot_radius)), axis = 0)    
            #else:
            #    config.ob.append(circle_obstacle_rectangle_robot([i.px, i.py], config.robot_width, config.robot_length))
            
        boundary_x = [self.boundary[0], self.boundary_width + self.boundary[0]]
        boundary_y = [self.boundary[1], self.boundary_height + self.boundary[1]]
        self.ob = np.concatenate((self.ob, self.rectangle_obstacle_circle_robot(boundary_x, boundary_y, self.robot_radius)), axis = 0)
        
    
    def motion(self, x, u, dt):
        """
        motion model
        """
        output = x.copy()
        output[2] += u[1] * dt
        output[0] += u[0] * math.cos(output[2]) * dt
        output[1] += u[0] * math.sin(output[2]) * dt
        output[3] = u[0]
        output[4] = u[1]

        return output


    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
              -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_delta_yaw_rate * self.dt,
              x[4] + self.max_delta_yaw_rate * self.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        # print("dw: ", dw)
        return dw

    # Predict trajectory in the next time period with input transitional velocity v and angular velocity y
    # In general, it gets trajectory by calculating the states in from t to t + dt * n 
    # where (dt * n) <= predict time and n = predict_time/dt
    # and then stack the states to get the trajectory 
    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory


    def calc_control_and_trajectory(self, x, dw, goal, ob):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        # The dynamic window is within velocity space
        # Try all transitional and angular velocity and acceleration within the dynamic window
        # with resolution. The number of transitional velocity to try is (max_dwv - min_dwv)/resolution
        # The number of angular velocity to try is (max_dwy - min_dwy)/yaw_rate_resolution
        # From (x, y, yaw, v, yaw_rate) to (x_new, y_new, yaw_new, v_new, yaw_rate_new)
        v_start = x[3]
        y_start = x[4]
        # weights of turtle robot dynamics
        sigma1 = 3.65
        sigma2 = 4.51
        sigma3 = 0.85
        sigma4 = 0.22
        value = [-1, 0, 1]
        action_sim = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        action_flag = 8
        best_action = 4
        dynamics_weight = self.dynamics_weight
        for i in range(3):
            if (v_start + value[i] * self.v_resolution) >= dw[0] and (v_start + value[i] * self.v_resolution) <= dw[1]:
                v_d = v_start + value[i] * self.v_resolution
                v = v_start + (-(sigma3/sigma1) * v_start + 1/sigma1 * v_d) * self.dt * dynamics_weight
            else:
                action_flag -= 3
                continue
        
            for j in range(3):
                if (y_start + value[j] * self.yaw_rate_resolution) >= dw[2] and (y_start + value[j] * self.yaw_rate_resolution) <= dw[3]:
                    y_d = y_start + value[j] * self.yaw_rate_resolution
                    y = y_start + (-(sigma4/sigma2) * y_start + 1/sigma2 * y_d) * self.dt * dynamics_weight
                else:
                    action_flag -= 1
                    continue
            
                trajectory = self.predict_trajectory(x_init, v, y)
                # calc cost
                # Measure the progress towards the goal location. It is maximal if the robot moves directly towards the target
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                # The forward velocity of the robot and supports fast movements
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                # The distance to the closest obstacle on the trajectory
                ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob)

                final_cost = to_goal_cost + speed_cost + ob_cost
            
                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    best_action = action_sim[action_flag]
                    if abs(best_u[0]) < self.robot_stuck_flag_cons \
                            and abs(x[3]) < self.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.max_delta_yaw_rate
                        best_action = self.stuck_action
                action_flag -= 1
                    
        # best_u: Best (transitional velocity, yaw_rate)
        # best_trajectory (trajectory with best_u)
        # print(action_flag, best_action)
        return best_u, best_trajectory, best_action


    def calc_obstacle_cost(self, trajectory, ob):
        """
        calc obstacle cost inf: collision
        """
        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        if self.robot_type == RobotType.rectangle:
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= self.robot_length / 2
            right_check = local_ob[:, 1] <= self.robot_width / 2
            bottom_check = local_ob[:, 0] >= -self.robot_length / 2
            left_check = local_ob[:, 1] >= -self.robot_width / 2
            if (np.logical_and(np.logical_and(upper_check, right_check),
                               np.logical_and(bottom_check, left_check))).any():
                return float("Inf")
        elif self.robot_type == RobotType.circle:
            if np.array(r <= self.robot_radius).any():
                return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK


    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost


    def plot_arrow(self, x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  head_length=width, head_width=width)
        plt.plot(x, y)


    def plot_robot(self, x, y, yaw):  # pragma: no cover
        if self.robot_type == RobotType.rectangle:
            outline = np.array([[-self.robot_length / 2, self.robot_length / 2,
                                 (self.robot_length / 2), -self.robot_length / 2,
                                 -self.robot_length / 2],
                                [self.robot_width / 2, self.robot_width / 2,
                                 - self.robot_width / 2, -self.robot_width / 2,
                                 self.robot_width / 2]])
            Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                             [-math.sin(yaw), math.cos(yaw)]])
            outline = (outline.T.dot(Rot1)).T
            outline[0, :] += x
            outline[1, :] += y
            plt.plot(np.array(outline[0, :]).flatten(),
                     np.array(outline[1, :]).flatten(), "-k")
        elif self.robot_type == RobotType.circle:
            circle = plt.Circle((x, y), self.robot_radius, color="b")
            plt.gcf().gca().add_artist(circle)
            out_x, out_y = (np.array([x, y]) +
                            np.array([np.cos(yaw), np.sin(yaw)]) * self.robot_radius)
            plt.plot([x, out_x], [y, out_y], "-k")
            
    def dwa_control(self, x, goal, ob):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory, best_action = self.calc_control_and_trajectory(x, dw, goal, ob)

        return u, trajectory, best_action
    
    # 
    def predict(self, env):
        goals = [env.robot.gx, env.robot.gy]
        self.set_obstacle(env.cur_obstacles, env.humans, True)
        ob = self.ob
        curr_state_before = np.array([env.robot.px, env.robot.py, env.robot.theta, env.robot.v, env.robot.w])
        u, predicted_trajectory, action = self.dwa_control(curr_state_before, goals, ob)
        curr_state_after = self.motion(curr_state_before, u, self.dt)
        # print("before: ", curr_state_before)
        # print("after: ", curr_state_after, action)
        return u, predicted_trajectory, curr_state_after, action
    