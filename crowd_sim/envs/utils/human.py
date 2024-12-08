import numpy as np

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)
        # whether the human is a static obstacle (part of wall) or a moving agent
        self.isObstacle = False
        self.isObstacle_period = np.inf
        # whether the human reacts to the robot
        self.react_to_robot = False
        # route of this human, only exists in constrained env
        self.route = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
