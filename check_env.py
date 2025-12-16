# This script sets up a crowd simulation environment, optionally rendering the environment in real-time.
# It initializes the environment configuration, applies random actions to a simulated agent, and renders
# the environment if display is enabled. The script demonstrates a loop to interact with the environment for 2000 steps.

import matplotlib.pyplot as plt
import numpy as np
from crowd_sim.envs import *
import pybullet as p
from crowd_sim.envs.crowd_sim_tb2_obs_scope import CrowdSim3DTbObsHieOM

if __name__ == '__main__':
    display = True

    from crowd_nav.configs.config import Config
    config = Config()

    # Enable rendering if display mode is active and 'sim' attribute exists in config
    if display and hasattr(config, 'sim'):
        inner_object = getattr(config, 'sim')
        setattr(inner_object, 'render', True)

    # config.robot.initTheta_range = [0, 2 * np.pi]
    # Initialize and configure the crowd simulation environment
    # env = CrowdSim3DTbObsHieOM()
    config.env.time_limit = 15
    env = CrowdSim3DTbObs()
    env.configure(config)
    env.thisSeed = 16     # Set seed for reproducibility
    env.nenv = 1          # Define single environment instance
    env.phase = 'test'    # Set environment phase to 'test'


    # Set up visualization if display mode is active and environment type is CrowdSimVarNum
    if display and type(env) == CrowdSimVarNum:
        fig, ax = plt.subplots(figsize=(9, 9))  # Create figure for plotting environment state
        ax.set_xlim(-10, 10)                    # Define plot boundaries
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x(m)', fontsize=16)      # Label axes
        ax.set_ylabel('y(m)', fontsize=16)
        plt.ion()                               # Enable interactive plotting
        plt.show()
        env.render_axis = ax                    # Link environment to the plot axis

    obs = env.reset()  # Initialize environment and get initial observation

    # Print all joints
    '''
    0 base_link_joint
    1 front_left_wheel
    2 front_right_wheel
    3 rear_left_wheel
    4 rear_right_wheel
    5 front_fender_joint
    6 rear_fender_joint
    7 imu_joint
    8 navsat_joint
    '''
    for i in range(p.getNumJoints(env.robot.uid)):
        info = p.getJointInfo(env.robot.uid, i)
        print(i, info[1].decode('utf-8'))  # index and joint name

    done = False       # Track if an episode has ended

    # # todo: debug the start and end regions here
    # if config.env.scenario == 'csl_workspace':
    #     for key in config.human_flow.regions:
    #         p.addUserDebugLine([config.human_flow.regions[key][0],  config.human_flow.regions[key][2], 2],
    #                            [config.human_flow.regions[key][1], config.human_flow.regions[key][2], 2], [0, 0, 1], lineWidth=3)
    #         p.addUserDebugLine([config.human_flow.regions[key][1], config.human_flow.regions[key][2], 2],
    #                            [config.human_flow.regions[key][1], config.human_flow.regions[key][3], 2], [0, 0, 1],
    #                            lineWidth=3)
    #         p.addUserDebugLine([config.human_flow.regions[key][0], config.human_flow.regions[key][2], 2],
    #                            [config.human_flow.regions[key][0], config.human_flow.regions[key][3], 2], [0, 0, 1],
    #                            lineWidth=3)
    #         p.addUserDebugLine([config.human_flow.regions[key][0], config.human_flow.regions[key][3], 2],
    #                            [config.human_flow.regions[key][1], config.human_flow.regions[key][3], 2], [0, 0, 1],
    #                            lineWidth=3)

    for i in range(2000):
        # action is change of v and w, for sim2real
        # translational velocity change: +0.05, 0, -0.05 m/s
        # rotational velocity change: +0.1, 0, -0.1 rad/s
        # self.action_convert = {0: [0.05, 0.1], 1: [0.05, 0], 2: [0.05, -0.1],
        #                        3: [0, 0.1], 4: [0, 0], 5: [0, -0.1],
        #                        6: [-0.05, 0.1], 7: [-0.05, 0], 8: [-0.05, -0.1]}
        action = 4               # Select random action
        obs, reward, done, info = env.step(action)  # Take action, receive next state, reward, and completion flag
        print(reward)
        # print(obs['robot_node'])
        if display:                                 # Render environment state if display mode is enabled
            env.render()

        if done:                                    # If episode ends, print info and reset environment
            print(str(info))
            env.reset()

    env.close()  # Close the environment after completing the loop
