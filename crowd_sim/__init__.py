from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSimVarNum-v0',
    entry_point='crowd_sim.envs:CrowdSimVarNum',
)

register(
    id='CrowdSim3DTB-v0',
    entry_point='crowd_sim.envs:CrowdSim3DTB',
)

register(
    id='CrowdSim3DTbObs-v0',
    entry_point='crowd_sim.envs:CrowdSim3DTbObs',
)

register(
    id='CrowdSim3DTbObsHie-v0',
    entry_point='crowd_sim.envs:CrowdSim3DTbObsHie',
)

register(
    id='rosTurtlebot2iEnv-v0',
    entry_point='crowd_sim.envs.ros_turtlebot2i_env:rosTurtlebot2iEnv',
)


