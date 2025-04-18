import math
import heapq
import random

import numpy as np


class AStar:
    """
    Adopted from Huang et al, Neural Informed RRT* (https://ieeexplore.ieee.org/abstract/document/10611099)
    AStar set the cost + heuristics as the priority
    """

    def __init__(self, s_start, s_goal, binary_mask, clearance, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]  # feasible input set
        self.obs = binary_mask  # position of obstacles
        self.clearance = clearance

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))
        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        neighbors = []
        for u in self.u_set:
            neighbor_candidate = (s[0] + u[0], s[1] + u[1])
            if self.clearance <= neighbor_candidate[0] < self.obs.shape[1] - self.clearance \
                    and self.clearance <= neighbor_candidate[1] < self.obs.shape[0] - self.clearance:
                neighbors.append(neighbor_candidate)
        return neighbors

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        if np.any(self.obs[s_start[1] - self.clearance:s_start[1] + self.clearance + 1, \
                  s_start[0] - self.clearance:s_start[0] + self.clearance + 1] == 0) or \
                np.any(self.obs[s_end[1] - self.clearance:s_end[1] + self.clearance + 1, \
                       s_end[0] - self.clearance:s_end[0] + self.clearance + 1] == 0):
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                # reverse of %, check circles.
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                # %, check circles.
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if self.obs[s1[1], s1[0]] == 0 or self.obs[s2[1], s2[0]] == 0:
                return True

            if np.any(self.obs[s1[1] - self.clearance:s1[1] + self.clearance + 1, \
                      s1[0] - self.clearance:s1[0] + self.clearance + 1] == 0) or \
                    np.any(self.obs[s2[1] - self.clearance:s2[1] + self.clearance + 1, \
                           s2[0] - self.clearance:s2[0] + self.clearance + 1] == 0):
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            if s not in PARENT.keys():
                break
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def get_path_from_start_to_goal(self, path):
        # note the returned path is from goal to start, so we need to reverse the path
        path.reverse()
        return path

    def check_success(self, path):
        return path[0] == self.s_start and path[-1] == self.s_goal


def generate_start_goal_points(binary_mask, clearance=0, distance_lower_limit=50, max_attempt_count=100):
    img_height, img_width = binary_mask.shape
    attempt_count = 0
    while True:
        # random.randint has both ends included, so different from np.random.randint
        xs, ys = random.randint(clearance, img_width - clearance - 1), random.randint(clearance,
                                                                                      img_height - clearance - 1)
        xg, yg = random.randint(clearance, img_width - clearance - 1), random.randint(clearance,
                                                                                      img_height - clearance - 1)
        if abs(xs - xg) >= distance_lower_limit and abs(ys - yg) >= distance_lower_limit and \
                not np.any(binary_mask[ys - clearance:ys + clearance + 1, xs - clearance:xs + clearance + 1] == 0) and \
                not np.any(binary_mask[yg - clearance:yg + clearance + 1, xg - clearance:xg + clearance + 1] == 0):
            return (xs, ys), (xg, yg)
        attempt_count += 1
        if attempt_count > max_attempt_count:
            return None, None

def generate_astar_path(
    binary_env,
    s_start,
    s_goal,
    clearance=3,
):
    astar = AStar(s_start, s_goal, binary_env, clearance, "euclidean")
    path, visited = astar.searching()
    path = astar.get_path_from_start_to_goal(path) # list of tuples
    path_success = astar.check_success(path)
    if path_success:
        return path
    else:
        return None