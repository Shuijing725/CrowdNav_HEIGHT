policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.srnn import *

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['srnn'] = SRNN
policy_factory['selfAttn_merge_srnn'] = selfAttn_merge_SRNN
policy_factory['selfAttn_merge_srnn_lidar'] = selfAttn_merge_SRNN_lidar
policy_factory['dsrnn_obs_vertex'] = DSRNN_obs_vertex
policy_factory['homo_transformer_obs'] = Homo_Transformer_Obs
policy_factory['lidar_gru'] = lidar_gru
policy_factory['om_gru'] = om_gru
policy_factory['homo_transformer_om_patch'] = Homo_Transformer_Obs_Patch