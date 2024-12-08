from collections import namedtuple
import numpy as np

FullState = namedtuple('FullState', ['px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'])
ObservableState = namedtuple('ObservableState', ['px', 'py', 'vx', 'vy', 'radius'])

# JointState has 2 attributes:
# self.self_state is a FullState
# self.human_states is a list of ObservableStates
class JointState(object):
    # self_state: list of length 9
    # human_states: list of length human_num*5 or nested list [human_num, 5]
    def __init__(self, self_state, human_states):
        assert len(self_state) == 9
        human_states_namedtuple = []
        # if human states is a nested list [human_num, 5]
        if len(np.shape(human_states)) == 2:
            for human_state in human_states:
                assert len(human_state) == 5
                human_states_namedtuple.append(ObservableState(*human_state))
        # if human states is a flatten list of length human_num*5
        else:
            assert len(human_states) % 5 == 0
            human_num = len(human_states) // 5
            for i in range(human_num):
                human_states_namedtuple.append(ObservableState(*human_states[int(i*5):(int((i+1)*5))]))

        self.self_state = FullState(*self_state)
        self.human_states = human_states_namedtuple

    # convert a joint state to a flattened list of length 9 + 5 * human_num
    def to_flatten_list(self):
        flatten_list = list(self.self_state)
        for human_state in self.human_states:
            flatten_list.extend(list(human_state))
        return flatten_list

# class FullState(object):
#     def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
#         self.px = px
#         self.py = py
#         self.vx = vx
#         self.vy = vy
#         self.radius = radius
#         self.gx = gx
#         self.gy = gy
#         self.v_pref = v_pref
#         self.theta = theta
#
#         self.position = (self.px, self.py)
#         self.goal_position = (self.gx, self.gy)
#         self.velocity = (self.vx, self.vy)
#
#     def __add__(self, other):
#         return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)
#
#     def __str__(self):
#         return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
#                                           self.v_pref, self.theta]])
#
#
# class ObservableState(object):
#     def __init__(self, px, py, vx, vy, radius):
#         self.px = px
#         self.py = py
#         self.vx = vx
#         self.vy = vy
#         self.radius = radius
#
#         self.position = (self.px, self.py)
#         self.velocity = (self.vx, self.vy)
#
#     def __add__(self, other):
#         return other + (self.px, self.py, self.vx, self.vy, self.radius)
#
#     def __str__(self):
#         return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])
#
#
#
# class JointState(object):
#     def __init__(self, self_state, human_states):
#         assert isinstance(self_state, FullState)
#         for human_state in human_states:
#             assert isinstance(human_state, ObservableState)
#
#         self.self_state = self_state
#         self.human_states = human_states
#
#
# class ObservableState_noV(object):
#     def __init__(self, px, py, radius):
#         self.px = px
#         self.py = py
#         self.radius = radius
#
#         self.position = (self.px, self.py)
#
#
#     def __add__(self, other):
#         return other + (self.px, self.py, self.radius)
#
#     def __str__(self):
#         return ' '.join([str(x) for x in [self.px, self.py, self.radius]])
#
#
# class JointState_noV(object):
#     def __init__(self, self_state, human_states):
#         assert isinstance(self_state, FullState)
#         for human_state in human_states:
#             assert isinstance(human_state, ObservableState_noV)
#
#         self.self_state = self_state
#         self.human_states = human_states