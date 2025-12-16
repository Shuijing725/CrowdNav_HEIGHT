from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config,section):
        super().__init__(config,section)
        self.sensor_range = config.lidar.sensor_range
        # for turtlebot env only
        self.w = 0
        self.v = 0
        # for hierarchical policy only
        self.subgoalx = None
        self.subgoaly = None

    def get_changing_state_list_subgoal(self):
        return [self.px, self.py, self.subgoalx, self.subgoaly, self.theta]

    def get_changing_state_list_goal_offset_subgoal(self):
        return [self.subgoalx - self.px, self.subgoaly - self.py, self.theta]

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action


    def actWithJointState(self,ob):
        action = self.policy.predict(ob)
        return action
