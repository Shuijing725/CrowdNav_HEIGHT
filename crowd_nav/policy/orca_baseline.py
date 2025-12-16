import numpy as np
import rvo2
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self, config):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__(config)
        self.name = 'ORCA'
        self.time_step = config.env.time_step
        self.max_neighbors = None
        self.radius = None
        self.robot_pref = config.robot.v_max
        self.humans_pref = 1.0
        self.max_speed = 1.0
        self.sim = None
        self.safety_space = self.config.orca.safety_space
        # empty for env that doesn't involve static obstacles, will be overwritten in hallway.py
        self.static_obs = []
        
        self.action_convert = {
            0: [ 0.05,  0.10], 1: [ 0.05,  0.00], 2: [ 0.05, -0.10],
            3: [ 0.00,  0.10], 4: [ 0.00,  0.00], 5: [ 0.00, -0.10],
            6: [-0.05,  0.10], 7: [-0.05,  0.00], 8: [-0.05, -0.10]
        }
        self._cand = np.array(list(self.action_convert.values()))
        
        self._c = config.orca

    
    def _angle_diff(self, a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi
    
    def _discretise(self, vx, vy, robot, env):
        v_des   = np.hypot(vx, vy)
        head_des = np.arctan2(vy, vx)

        v_cur   = env.desiredVelocity[0]
        w_cur   = env.desiredVelocity[1]
        theta   = robot.theta

        w_des   = self._angle_diff(head_des, theta) / self.time_step
        target  = np.array([v_des - v_cur, w_des - w_cur])

        lam = 1.0                                   
        diff = target - self._cand
        idx  = int(np.argmin(diff[:, 0]**2 + lam * diff[:, 1]**2))
        return idx

    def predict(self, env):
        rb        = env.robot
        humans    = env.humans
        obstacles = env.cur_obstacles

        # print("robot: ", rb)
        # for i in humans: 
        #     print("humans: ",i.vx, i.vy)
        # print("obstacles: ", obstacles)

        max_nbrs = len(humans)
        sim = rvo2.PyRVOSimulator(
            self.time_step,
            self._c.neighbor_dist,
            max_nbrs,
            self._c.time_horizon,
            self._c.time_horizon_obst,
            rb.radius + 0.01 + self._c.safety_space,
            self.max_speed
        )

        sim.addAgent(
            (rb.px, rb.py),
            self._c.neighbor_dist, max_nbrs,
            self._c.time_horizon, self._c.time_horizon_obst,
            rb.radius + 0.01 + self._c.safety_space,
            self.robot_pref,
            (rb.v * np.cos(rb.theta), rb.v * np.sin(rb.theta))
        )

        for h in humans:
            sim.addAgent(
                (h.px, h.py),
                self._c.neighbor_dist, max_nbrs,
                self._c.time_horizon, self._c.time_horizon_obst,
                h.radius + 0.01 + self._c.safety_space,
                self.humans_pref,
                (h.vx, h.vy)
            )

        for rect in obstacles:
            x, y, w, h, _ = rect
            sim.addObstacle([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        sim.processObstacles()

        to_goal = np.array([rb.gx - rb.px, rb.gy - rb.py])
        dist    = np.linalg.norm(to_goal)
        pref    = to_goal / dist if dist > 1.0 else to_goal
        sim.setAgentPrefVelocity(0, tuple(pref))

        for i, h in enumerate(humans):
            sim.setAgentPrefVelocity(
                i + 1, (h.vx, h.vy)
            )

        sim.doStep()
        vx, vy = sim.getAgentVelocity(0)

        idx     = self._discretise(vx, vy, rb, env)
        dv, dw  = self.action_convert[idx]
        return idx