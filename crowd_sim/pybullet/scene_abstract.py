"""
This file is used for setting up physics and simulation properties
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

import time
import gym


class SingleRobotEmptyScene(object):
    "A base class for single agent scenes"

    def __init__(self, bulletClient, gravity, timestep, frame_skip,render):

        self.timestep = timestep
        self.frame_skip = frame_skip
        self.p=bulletClient
        self.render=render
        self.dt = self.timestep * self.frame_skip # one RL step takes self.dt simulation time
        self.cpp_world = World(bulletClient,gravity, timestep, frame_skip, render)

    def episode_restart(self):
        self.cpp_world.clean_everything()

    def global_step(self):
        self.cpp_world.step()


class World:

    def __init__(self, bulletClient,gravity, timestep, frame_skip,render):
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.p = bulletClient
        self.clean_everything()

        self.render=render


    def clean_everything(self):
        self.p.setGravity(self.gravity[0],self.gravity[1],self.gravity[2])
        # fixedTimeStep:physics engine timestep in fraction of seconds,
        # each time you call 'stepSimulation' simulated time will progress this amount. Notice that it is not
        # the wall clock time step. It is just the simulated time step. To make the simulation run faster than the
        # wall-clock time, this number should be larger than CPU computation time. For example, if fixedTimeStep=1/30,
        # and the computer completes the computation in 1/240s, the simulation is 8 times faster than
        # the wall-clock time. If fixedTimeStep=1/240 and and the computer completes the computation in 1/30s,
        # the simulation is 8 times slower than the wall-clock time.
        # In Pybullet, however, in many cases it is best to leave the fixedTimeStep to default, which is 240Hz, because
        # several parameters are tuned with this value in mind'. The setting
        # self.p.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip, numSolverIterations=50,numSubSteps=(self.frame_skip-1))
        # does not break this rule because of the numSubSteps. E.g. self.timestep=1/240, self.frame_skip=8
        # numSubSteps: Subdivide the physics simulation step further by 'numSubSteps'. This will trade performance over accuracy.
        # so every simulation step still takes 1/240 simulation time. The setting
        # self.p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=50) sees every simulation step as 1 RL step,
        # while the previous setting sees 8 simulation step as 1 RL step, hence the name frame skip.
        # self.timestep*self.frame_skip is dt, which means that one RL step takes dt simulation time
        # Notice that the previous setting is more practical and faster. The bigger the frame_skip, the faster the
        # simulation usually will be.
        # For example, when we have camera sensors, the previous setting will
        # generate 1 image every 8 simulation step. Within these 8 simulation steps, no image is generated. This setting,
        # however, generates an image at every simulation step, which slows down the program and may be not necessary,
        # because there may be no visually important things happened in such short amount of time.
        # When you transfer your system to the real world. The real world has fixedTimeStep=camera frames per second
        # Usually, a camera has 60 fps. Then, fixedTimeStep=1/60 instead of (1/240)
        # and 1s wall-clock time equals 60 RL steps.
        # If you want to match your simulation time with this wall-clock time, you should
        # set the self.frame_skip=4. In this way, self.timestep*self.frame_skip=1/240*4=1/60.


        self.p.setPhysicsEngineParameter(fixedTimeStep=self.timestep * self.frame_skip, numSolverIterations=30,
                                         numSubSteps=(self.frame_skip-1))


    def step(self):
        self.p.stepSimulation()
        if self.render:
            time.sleep(self.timestep*self.frame_skip)
            # time.sleep(self.timestep)






