import gym
import numpy as np
from numpy.linalg import norm
import copy
import os
from collections import defaultdict
from PIL import Image, ImageFilter
import imageio

import pybullet as p
from pybullet_utils import bullet_client

try:
    if os.environ["PYBULLET_EGL"]:
        import pkgutil
except:
    pass


from crowd_nav.policy.policy_factory import policy_factory

from crowd_sim.pybullet.scene_abstract import SingleRobotEmptyScene
from crowd_sim.envs.crowd_sim_var_human import CrowdSimVarNum

'''
This environment contains all pybullet part.
Everything is the same as CrowdSimVarNum, except the original holomonic/unicycle robot is replaced with TurtleBot2 from PyBullet
'''


class CrowdSim3DTB(CrowdSimVarNum):
    def __init__(self):
        super().__init__()
        self.id_counter = None
        self.observed_human_ids = None

        self.lidar_ang_res = None
        self.lidar_range = None
        self.ray_slopes = None

        self.robot_dir_line_id = None

        # store the robot angle at previous timestep, to calculate rotation penalty
        self.rob_prev_theta = np.pi/2
        # for turtlebot dynamics
        self.motor_counter = 0
        self.priority_motor = None

        # when running the code, whether this episode is the first episode or not
        self.first_epi = False

        # record the uids of all cylinder obstacles
        self.cylinder_obs_uids = []

        self.images = []

        self.save_dir = None

    def configure(self, config):
        # ray test
        self.rayIDs = []
        self.collision = False  # does collision happen during an episode
        self.rayHitColor = [1, 0, 0]
        self.rayMissColor = [0, 1, 0]

        self.cam_rayIDs = []
        self.cam_rayHitColor = [1, 0.25, 0]
        self.cam_rayMissColor = [0, 1, 0.25]

        # read lidar config
        self.lidar_ang_res = config.lidar.angular_res
        self.lidar_range = config.lidar.sensor_range
        self.lidar_height = config.lidar.height
        # total number of rays
        self.ray_num = int(360. / self.lidar_ang_res)
        # list of all angles of the rays
        self.ray_angles = np.linspace(0, 2 * np.pi, self.ray_num, endpoint=False)

        # total number of rays for camera
        self.cam_ray_num = int(config.camera.fov / config.camera.ray_angular_res)
        # list of all angles of the rays
        self.cam_ray_angles = np.linspace(-config.camera.fov * np.pi/180./2, config.camera.fov * np.pi/180./2, self.cam_ray_num, endpoint=False)

        # read camera config
        # camera config for observations
        self.camera_fov = config.camera.fov
        self.camera_height = config.camera.height

        # camera config for visualization
        self.render_fov = config.camera.render_cam_fov
        self.render_img_w = config.camera.render_cam_img_width
        self.render_img_h = config.camera.render_cam_img_height

        self.cam_pc_ids = []
        self.removed_cam_pc_id = None

        super().configure(config)

        # Pybullet related
        self.scene = None
        self.physicsClientId = -1  # at the first run, we do not own physics client
        self.ownsPhysicsClient = False
        self._p = None

        self.obs_color = [0.65, 0.65, 0.65, 1]

        # 3D models
        self.arena_floor = None
        self.arenaWallInner = None
        self.arenaWallOuter = None
        self.wallTexPath = os.path.join(self.config.pybullet.mediaPath, 'arena', 'wallAppearance', 'texture')
        self.wallTextureList = []

        self.objTexDict = {}
        self.objTexPath = os.path.join(self.config.pybullet.mediaPath, 'objects', 'texture')
        self.objTexList = []

        self.objUidList = []  # PyBullet Uid for each object

        # each spot associates with a list [x,y,yaw]. The order will be same with self.objList
        # [[x,y,yaw],[x,y,yaw],...]
        self.objPoseList = []  # objects position and orientation information
        # current obj in the scene. e.g. [2,3] means cone and cylinder in scene
        self.objInScene = np.array([])

        self.goalObjIdx = None
        self.ground_truth = None
        self.goal_audio = None


        self.goal_area_count = 0

        # to prevent reward explosion when turtlebot drifts or flies after hitting obstacles
        # maximum absolute value of potential reward
        self.max_abs_pot_reward = self.time_step * self.config.robot.v_max * self.pot_factor
        # maximum absolute value of rotation penalty
        self.max_abs_rot_penalty = self.config.reward.spin_factor * max(abs(self.config.robot.w_min), self.config.robot.w_max) ** 2
        self.max_abs_back_penalty = self.config.reward.back_factor * max(abs(self.config.robot.v_min), self.config.robot.v_max)

    def set_observation_space(self):
        d = {}
        # robot node: gx-px, gy-py, theta
        if self.config.ob_space.robot_state == 'absolute':
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5,), dtype=np.float32)
        else:
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 3,), dtype=np.float32)
        # d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # make sure there's at least one human
        if self.config.ob_space.add_human_vel:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 4),
                                                dtype=np.float32)
        else:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 2),
                                                dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(d)

    def set_action_space(self):
        if self.config.env.action_space == 'continuous':
            high = np.inf * np.ones([2, ])
            self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        elif self.config.env.action_space == 'discrete':
            # naive action space, does not consider acceleration limits
            # translational velocity: 0, 1
            # rotational velocity: 0.5 (left turn), 0, -0.5 (right turn)
            # tables to convert actions from index to value
            # 0: left rotation in place, 1: stop, 2: right rotation in place
            # 3: left circling, 4: going straight, 5: right circling
            # self.action_convert = {0: [0, 0.5], 1: [0, 0], 2: [0, -0.5], 3: [1, 0.5], 4: [1, 0], 5: [1, -0.5]}

            # action is change of v and w, for sim2real
            # translational velocity change: +0.05, 0, -0.05 m/s
            # rotational velocity change: +0.1, 0, -0.1 rad/s
            self.action_convert = {0: [0.05, 0.1], 1: [0.05, 0], 2: [0.05, -0.1],
                                   3: [0, 0.1], 4: [0, 0], 5: [0, -0.1],
                                   6: [-0.05, 0.1], 7: [-0.05, 0], 8: [-0.05, -0.1]}

            self.action_space = gym.spaces.Discrete(len(self.action_convert))


    def create_single_player_scene(self, bullet_client):
        """
		Setup physics engine and simulation
		:param bullet_client:
		:return: a scene
		"""
        # use large frame skip such that motor has time to follow the desired position
        return SingleRobotEmptyScene(bullet_client, gravity=(0, 0, -9.8),
                                     timestep=self.config.pybullet.sim_timestep, frame_skip=self.config.pybullet.frameSkip,
                                     render=self.config.sim.render)

    def loadTex(self):
        wallTexList = os.listdir(self.wallTexPath)
        idx = np.arange(len(wallTexList))
        self.np_random.shuffle(idx)

        # load texture for walls
        for i in range(self.config.pybullet.numTexture):
            # key=fileName, val=textureID. If we have already loaded the texture, no need to reload and drain the memory
            texID = self._p.loadTexture(os.path.join(self.wallTexPath, wallTexList[idx[i]]))
            self.wallTextureList.append(texID)

        # load texture for objects
        objTextureList = os.listdir(self.objTexPath)
        for i in range(len(objTextureList)):
            texID = self._p.loadTexture(os.path.join(self.objTexPath, objTextureList[i]))
            self.objTexDict[objTextureList[i]] = texID
        self.objTexList = list(self.objTexDict.keys())

    def loadArena(self):
        # load arena: inner wall
        if self.config.sim.borders:
            if self.config.env.scenario == 'csl_workspace':
                wall_pos = [0, 0, -0.02]
            else: # for circle_crossing
                wall_pos = [0, 0, -0.02]
            self.arenaWallInner = self._p.loadURDF(
                os.path.join(self.config.pybullet.mediaPath, 'arena', 'arena_wall', 'arena_wall.urdf'),
                wall_pos, [0., 0., 0.0, 1.0],
                flags=self._p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                # make sure the humans start and end inside the walls
                globalScaling=self.arena_size + self.config.sim.human_pos_noise_range)
            self._p.changeVisualShape(self.arenaWallInner, -1, rgbaColor=[0.45, 0.45, 0.45, 1])

        # create a simple floor
        self.floor_id = p.loadURDF("crowd_sim/pybullet/media/arena/floor/plane.urdf")



    def create_object(self, px, py, radius, height, shape, color=None, halfExtents=None):
        """
        Create a 3D object with given params, returns its object ID
            pose of object: [px, py, height]
            radius of object: radius
            shape: self._p.GEOM_SPHERE, GEOM_CYLINDER, GEOM_BOX, GEOM_CYLINDER
            color: [r, g, b, alpha], each of which is in [0, 1]
            halfExtents: only for BOX
        """
        # object's appearance
        if color is None:
            color = [np.random.rand(), np.random.rand(), np.random.rand(), 1]

        # objects with radius
        if shape in [p.GEOM_SPHERE, p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
            visualID = self._p.createVisualShape(shapeType=shape,
                                                 visualFramePosition=[0, 0, 0],
                                                 visualFrameOrientation=self._p.getQuaternionFromEuler([0, 0, 0]),
                                                 radius=radius,
                                                 length=height,
                                                 rgbaColor=color  # r, g, b, alpha
                                                 )

            # create collision shape for each object so that the robot will not get closer than
            # objectsRadius+objectsExpandDistance to the object. It will be physically stopped by this collision shape
            # object's collision dynamics
            collisionID = self._p.createCollisionShape(shapeType=shape,
                                                       radius=radius,
                                                       height=height,
                                                       collisionFramePosition=[0, 0, 0],
                                                       collisionFrameOrientation=self._p.getQuaternionFromEuler([0, 0, 0]))
        # box: no radius, has halfExtents
        elif shape == p.GEOM_BOX:
            assert halfExtents is not None

            visualID = self._p.createVisualShape(shapeType=shape,
                                                 visualFramePosition=[0, 0, 0],
                                                 visualFrameOrientation=self._p.getQuaternionFromEuler([0, 0, 0]),
                                                 halfExtents=halfExtents,
                                                 length=height,
                                                 rgbaColor=color  # r, g, b, alpha
                                                 )

            # create collision shape for each object so that the robot will not get closer than
            # objectsRadius+objectsExpandDistance to the object. It will be physically stopped by this collision shape
            # object's collision dynamics
            collisionID = self._p.createCollisionShape(shapeType=shape,
                                                       halfExtents=halfExtents,
                                                       height=height,
                                                       collisionFramePosition=[0, 0, 0],
                                                       collisionFrameOrientation=self._p.getQuaternionFromEuler(
                                                           [0, 0, 0]))
        # plane: no radius, no halfExtents, has planeNormal
        else:
            visualID = self._p.createVisualShape(shapeType=shape,
                                                 visualFramePosition=[0, 0, 0],
                                                 visualFrameOrientation=self._p.getQuaternionFromEuler([0, 0, 0]),
                                                 planeNormal=[0, 0, 1],
                                                 length=height,
                                                 rgbaColor=color  # r, g, b, alpha
                                                 )

            # create collision shape for each object so that the robot will not get closer than
            # objectsRadius+objectsExpandDistance to the object. It will be physically stopped by this collision shape
            # object's collision dynamics
            collisionID = self._p.createCollisionShape(shapeType=shape,
                                                       planeNormal=[0, 0, 1],
                                                       height=height,
                                                       collisionFramePosition=[0, 0, 0],
                                                       collisionFrameOrientation=self._p.getQuaternionFromEuler(
                                                           [0, 0, 0]))

        # create multibody from visual ID and collision ID
        objID = self._p.createMultiBody(baseMass=0,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseCollisionShapeIndex=collisionID,
                                        baseVisualShapeIndex=visualID,
                                        basePosition=[px, py, height / 2],
                                        baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0]))
        self._p.resetBasePositionAndOrientation(objID,
                                                [px, py, height],
                                                self._p.getQuaternionFromEuler([0, 0, 0]))

        return objID

    # if camera_fov is True, use the camera FOV to perform ray test (to get human detections, used in self.get_num_human_in_fov())
    # otherwise, use the lidar with 360 FOV to perform ray test (to get raw lidar pc including obstacles, as ob['point_clouds'])
    def ray_test(self, camera_fov=False):
        """
        perform a lidar ray test on all humans and obstacles in the scene
        save the current range readings in self.closest_hit_dist, the object IDs in self.closest_hit_id, the uids of all detected humans in self.visible_human_uids

        """
        if camera_fov:
            ray_num = self.cam_ray_num
            ray_angles = self.cam_ray_angles
            height = self.camera_height
        else:
            ray_num = self.ray_num
            ray_angles = self.ray_angles
            height = self.lidar_height
        # start xyz pos of all rays in world frame
        startPoint = np.array([[self.robot.px, self.robot.py, height]])
        self.rayFrom = np.repeat(startPoint, ray_num, axis=0) # [ray_num, 3]
        # end xyz pos of all rays
        # robot frame
        rayTo = np.array([self.robot.sensor_range * np.cos(self.robot.theta + ray_angles),
                          self.robot.sensor_range * np.sin(self.robot.theta + ray_angles),
                          np.zeros([ray_num])]) # [3, ray_num]
        # world frame
        # rayTo = np.array([self.robot.sensor_range * np.cos(self.ray_angles),
        #                   self.robot.sensor_range * np.sin(self.ray_angles),
        #                   np.zeros([self.ray_num])])  # [3, ray_num]
        # end xyz pos of all rays in world frame
        self.rayTo = rayTo.T + self.rayFrom # [ray_num, 3]

        # do the ray test, store the results
        # result is a list of tuple that contains [objectUniqueId, linkIndex, hit fraction, hit position, hit normal]
        results = p.rayTestBatch(self.rayFrom, self.rayTo)

        # store the distances between robot and hit objects by lidar
        self.closest_hit_dist = np.zeros(ray_num)
        # store the ground truth hit object ids
        self.closest_hit_id = -np.ones(ray_num)

        # uids of human that can be detected by robot's Lidar
        self.visible_human_uids = set()

        # parse the results and visualize lidar rays

        for i in range(ray_num):
            self.closest_hit_dist[i] = results[i][2] * self.robot.sensor_range
            # only add human ids, assume other objects are undetectable with 2D lidar
            if results[i][0] in self.used_human_uids:
                self.closest_hit_id[i] = results[i][0]
                self.visible_human_uids.add(results[i][0]) # set can avoid duplicates
            if self.config.lidar.visualize_rays:
                if camera_fov:
                    pass
                    # if len(self.cam_rayIDs) < ray_num:  # draw these rays out
                    #     self.cam_rayIDs.append(p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.cam_rayMissColor))
                    #
                    # # this ray didn't hit anything
                    # if results[i][0] == -1:
                    #     p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.cam_rayMissColor,
                    #                        replaceItemUniqueId=self.cam_rayIDs[i])
                    # # this ray hit an object
                    # else:
                    #     p.addUserDebugLine(self.rayFrom[i], results[i][3], self.cam_rayHitColor,
                    #                        replaceItemUniqueId=self.cam_rayIDs[i])
                else:
                    # pass
                    if len(self.rayIDs) < ray_num:  # draw these rays out
                        self.rayIDs.append(p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor))

                    # this ray didn't hit anything
                    if results[i][0] == -1:
                        p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor,
                                           replaceItemUniqueId=self.rayIDs[i])
                    # this ray hit an object
                    else:
                        p.addUserDebugLine(self.rayFrom[i], results[i][3], self.rayHitColor,
                                           replaceItemUniqueId=self.rayIDs[i])




    def render_lidar_pc(self):
        '''
        plot the lidar point clouds as black dots
        '''
        pc_list = np.ones((self.ray_num, 3)) * self.lidar_height
        # convert from polar coordinate to Eulidean coordinate
        pc_list[:, 0] = self.closest_hit_dist * np.cos(self.ray_angles) + self.robot.px
        pc_list[:, 1] = self.closest_hit_dist * np.sin(self.ray_angles) + self.robot.py
        color = np.zeros((self.ray_num, 3))
        if len(self.cam_pc_ids) == 0:
            self.cam_pc_ids.append(p.addUserDebugPoints(pc_list, color.tolist(), pointSize=2))
        else:
            p.addUserDebugPoints(pc_list, color.tolist(), replaceItemUniqueId=self.cam_pc_ids[0], pointSize=2)


    def get_num_human_in_fov(self):
        # use ray test to check visibility of each human
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        # perform ray test, update self.visible_human_uids
        # we use zed camera to detect humans, so we need to add a camera FOV here
        self.ray_test(camera_fov=True)

        for i in range(self.human_num):
            visible = True if self.humans[i].uid in self.visible_human_uids else False
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids

    # randomly change the shape of pybullet obstacles, the number of changed obstacle = obs_num
    # Can only be called in reset function!!!
    # Note: please don't call it in the middle of an episode, outside of reset function! Otherwise, the new object may overlap with existing objects
    def change_obs_shape_randomly(self, obs_num):
        # determine the size of all new obs
        obs_sizes = np.clip(np.random.normal(self.config.sim.obs_size_mean, self.config.sim.obs_size_std,
                                                  size=(obs_num, 2)), a_min=self.config.sim.obs_min_size,
                                 a_max=self.config.sim.obs_max_size)
        for i in range(obs_num):
            # randomly delete old obs
            objID_delete = np.random.choice(self.all_obs_uids)
            # delete pybullet object
            self._p.removeBody(objID_delete)

            # self.free_obs_uids, self.all_obs_uids (deepcopy of self.free_obs_uids), self.used_obs_uids
            self.all_obs_uids.remove(objID_delete)

            # self.obstacles
            for j in range(len(self.obstacles)):
                if self.obstacles[j, -1] == objID_delete:
                    # print('deleted obj width, height:', self.obstacles[j, 2:4])
                    # print('new obj width, height:', obs_sizes[i])
                    self.obstacles = np.delete(self.obstacles, j, 0)
                    break

            # insert new obstacle
            # generate pybullet object
            objID_add = self.create_object(50, 50, radius=None, height=1,
                                                         shape=p.GEOM_BOX,
                                                         color=self.obs_color,
                                                         # assume all objects' true heights are 1, will reset offset in z-axis later
                                                         halfExtents=[obs_sizes[i, 0] / 2, obs_sizes[i, 1] / 2, 1])
            # update self.free_obs_uids, self.all_obs_uids, self.used_obs_uids
            self.all_obs_uids.append(objID_add)

            # update self.obstacles (list of [lower left x, lower left y, width, height, uid])
            # todo: didn't check collision with other objects!!!!!!
            # subtract self.config.sim.obs_size_mean so that the obstacles are approximately centered in the arena
            x, y = np.random.uniform(-self.arena_size - self.config.sim.obs_size_mean, self.arena_size - self.config.sim.obs_size_mean, size=2)
            self.obstacles = np.vstack([self.obstacles, [[x, y, obs_sizes[i, 0], obs_sizes[i, 1], objID_add]]])
            # print('all obj width, height:', self.obstacles[:, 2:4])

    def generate_ob(self, reset):
        # print("check robot node inside 1: ", ob["robot_node"])
        ob = {}

        # nodes
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        if self.config.ob_space.robot_state == 'absolute':
            ob['robot_node'] = self.robot.get_changing_state_list()
        else:
            ob['robot_node'] = self.robot.get_changing_state_list_goal_offset()

        # print("check robot node inside 2: ", ob["robot_node"])

        self.update_last_human_states(self.human_visibility, reset=reset)

        # edges
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        # ([relative px, relative py, disp_x, disp_y], human id)
        # make sure there's at least one placeholder
        if self.config.ob_space.add_human_vel:
            all_spatial_edges = np.ones((max(1, self.max_human_num), 4)) * np.inf
        else:
            all_spatial_edges = np.ones((max(1, self.max_human_num), 2)) * np.inf

        for i in range(self.human_num):
            if self.human_visibility[i]:
                # vector pointing from human i to robot
                relative_pos = np.array(
                    [self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
                all_spatial_edges[self.humans[i].id, :2] = relative_pos
                if self.config.ob_space.add_human_vel:
                    # todo: for now the human velocities are in world frame, check zed2 for frame transformation
                    all_spatial_edges[self.humans[i].id, 2:] = self.last_human_states[i, 2:4]
        # sort all humans by distance (invisible humans will be in the end automatically)
        ob['spatial_edges'] = np.array(sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x[:2])))
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15

        ob['detected_human_num'] = num_visibles
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        # update self.observed_human_ids
        self.observed_human_ids = np.where(self.human_visibility)[0]
        self.ob = ob

        return ob

    def render_scene(self):
        """
        Read rgbd image from camera, save them in self.rgb_img & self.depth_img
        """
        if self.config.env.scenario == 'csl_workspace' and self.config.env.csl_workspace_type == 'lounge':
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0, 5, 12],
                cameraTargetPosition=[0, 5, 0],
                cameraUpVector=[1, 0, 0]
            )
        else:
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0.5, 0.5, 12],
                cameraTargetPosition=[0.5, 0.5, 0],
                cameraUpVector=[1, 0, 0]
            )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.render_fov, aspect=self.render_img_w/self.render_img_h, nearVal=0.05, farVal=100)

        # returns [width, height, rgbPixels, depthPixels, segmentationMaskBuffer]
        # perfect segmentation is not very realistic for floors and walls, not including it
        _, _, self.rgb_img, _, _ = p.getCameraImage(self.render_img_w, self.render_img_h,
                                                                 view_matrix,
                                                                 projection_matrix,
                                                                 shadow=False,
                                                                 flags=self._p.ER_NO_SEGMENTATION_MASK,
                                                                 renderer=self._p.ER_TINY_RENDERER)

        # only keep (r, g, b), remove alpha
        self.rgb_img = self.rgb_img[:, :, :-1]

        # for save_slides use
        rgbim = Image.fromarray(self.rgb_img)
        rgbim = rgbim.crop((322/900*self.render_img_w, 322/900*self.render_img_h, 600/900*self.render_img_w, 600/900*self.render_img_h))
        rgbim = rgbim.resize((self.render_img_w, self.render_img_h), Image.LANCZOS)
        rgbim = rgbim.filter(ImageFilter.SHARPEN)
        save_dir = os.path.join(self.config.training.output_dir, 'test_slideshows',
                                str(self.min_human_num) + 'to' + str(self.max_human_num) + 'humans_' +
                                str(self.min_obs_num) + 'to' + str(self.max_obs_num) + 'obs',
                                self.config.camera.render_checkpoint,
                                str(self.rand_seed) + '_' + str(self.case_counter['test'] - 1))
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        rgbim.save(os.path.join(save_dir, str(self.step_counter) +'.png'))
        self.images.append(rgbim)


    def create_scenario(self, phase='train', test_case=None):
        ############################ 1. init the starting pose of robot and humans #############################
        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case  # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0

        self.desiredVelocity = [0.0, 0.0]  # desired v and w of the robot
        self.humans = []
        # initialize a list to store observed humans' IDs
        self.observed_human_ids = []

        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        # print(counter_offset[phase], self.case_counter[phase], self.thisSeed)
        # np.random.seed(1162)
        # self.rand_seed = 1162

        # for sim2real use (make sure at most 1 static human is in the hallway, otherwise too many static humans in hallway will block the robot and other humans)
        self.static_human_in_hallway = False
        self.human_facing_robot = False

        ############################ 2. create pybullet simulator for the robot and humans #############################
        # After gym.make, we call env.reset(). This function is then called.
        # this function starts pybullet client and create_single_player_scene

        # if it is the first run, setup Pybullet client and set GUI camera
        if self.physicsClientId < 0:
            self.first_epi = True
            self.ownsPhysicsClient = True

            if self.config.sim.render:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

            # optionally enable EGL for faster headless rendering
            try:
                if os.environ["PYBULLET_EGL"]:
                    con_mode = self._p.getConnectionInfo()['connectionMethod']
                    if con_mode == self._p.DIRECT:
                        egl = pkgutil.get_loader('eglRenderer')
                        if (egl):
                            self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                        else:
                            self._p.loadPlugin("eglRendererPlugin")
            except:
                pass

            self.physicsClientId = self._p._client

            assert self._p.isNumpyEnabled() == 1
            p.setRealTimeSimulation(1)

            # add cylinders to represent robot and humans in the arena
            self.free_human_uids = []  # list of objectIDs of cylinders that have NOT been assigned to a human
            self.used_human_uids = []  # list of objectIDs of cylinders that have ALREADY been assigned to a human
            for i in range(self.max_human_num):
                self.free_human_uids.append(
                    self.create_object(px=20, py=20, radius=0.3, height=self.config.humans.height,
                                       shape=p.GEOM_CYLINDER,
                                       color=self.config.human_flow.colors[
                                           i] if self.config.human_flow.colors is not None else None))
                # print('uid:', self.free_human_uids[-1], 'color:', self.config.human_flow.colors[i])
            self.all_human_uids = copy.deepcopy(self.free_human_uids)

            # todo: add tb2 here
            # robot height must be < self.lidar.height!!!
            self.robot.uid = p.loadURDF("crowd_sim/pybullet/media/turtlebot2/turtlebot.urdf", [0, 0, 0])

            # todo: add "sim.static_obs_num+sim.static_obs_num_range" static obs
            # todo: fix the obs width and height, only vary (px, py, pz, theta) here, theta \in {0, 90}
            self.free_obs_uids = []
            self.used_obs_uids = []
            if self.add_static_obs:
                if self.config.sim.random_obs:
                    max_obs_num = self.avg_obs_num + self.obs_range
                    self.obs_sizes = np.clip(
                        np.random.normal(self.config.sim.obs_size_mean, self.config.sim.obs_size_std,
                                         size=(max_obs_num, 2)), a_min=self.config.sim.obs_min_size,
                        a_max=self.config.sim.obs_max_size)
                else:
                    self.obs_sizes = self.config.fixed_obs.sizes
                    self.obs_pos = self.config.fixed_obs.positions_lower_left
                    max_obs_num = len(self.obs_sizes)
                for i in range(max_obs_num):
                    # create cylinders, only for fixed obs environments
                    if not self.config.sim.random_obs and self.config.fixed_obs.shapes[i] == 0:
                        new_uid = self.create_object(50, 50, radius=self.config.fixed_obs.cylinder_radius,
                                                             height=self.config.fixed_obs.cylinder_height,
                                                             shape=p.GEOM_CYLINDER,
                                                             color=self.obs_color)

                        self.free_obs_uids.append(new_uid)
                        self.cylinder_obs_uids.append(new_uid)
                    # create rectangles
                    else:
                        self.free_obs_uids.append(self.create_object(50, 50, radius=None, height=0.7,
                                                                     shape=p.GEOM_BOX,
                                                                     color=self.obs_color,
                                                                     # assume all objects' true heights are 1, will reset offset in z-axis later
                                                                     halfExtents=[self.obs_sizes[i, 0] / 2,
                                                                                  self.obs_sizes[i, 1] / 2, 1]))

                # the matching between self.all_obs_uids and self.obs_sizes is always the same!!!
                self.all_obs_uids = copy.deepcopy(self.free_obs_uids)

                # update self.obstacles (stores ALL obstacles with pybullet object IDs)
                if self.config.sim.random_obs:
                    self.generate_rectangle_obstacle(max_obs_num, fixed_sizes=self.obs_sizes, uids=self.all_obs_uids)
                else:
                    self.generate_rectangle_obstacle(max_obs_num, fixed_sizes=self.obs_sizes, fixed_pos=self.obs_pos,
                                                     uids=self.all_obs_uids)

        else:
            self.first_epi = False

        # if it is the first run, build scene and setup simulation physics
        if self.scene is None:
            # this function will call episode_restart()->clean_everything() in
            # class World in scene_bases.py (self.cpp_world)
            # set gravity, set physics engine parameters, etc.
            self.scene = self.create_single_player_scene(self._p)

            # load arena floor
            self.loadArena()

        # # if it is not the first run
        if self.ownsPhysicsClient:
            self.scene.episode_restart()

        self.envStepCounter = 0

        # assign uids to static obs, and reset pos here
        if self.add_static_obs:
            if self.config.sim.random_obs:
                self.obs_num = np.random.randint(self.avg_obs_num - self.obs_range,
                                                 self.avg_obs_num + self.obs_range + 1)
                # randomly change the shape of some pybullet obstacles
                if np.random.rand() < 1:
                    self.change_obs_shape_randomly(obs_num=1)
                self.reset_obstacle_pos()
                # randomly shuffle the idx of all obs
                np.random.shuffle(self.obstacles)
            else:
                self.obs_num = len(self.obs_pos)
            # array to store the (x, y, w, h, uid) of all present obs in this episode
            # self.cur_obstacles: obstacles present in this episode
            # self.obstacles: all obstacles created when the code begins
            self.cur_obstacles = np.zeros((self.obs_num, 5))
            for i in range(self.obs_num):
                cur_uid = self.free_obs_uids.pop()
                self.used_obs_uids.append(cur_uid)
                idx = np.where(self.obstacles[:, -1] == cur_uid)
                idx = idx[0][0]

                # self.obstacles[idx][0, 1] is the lower left corner of the rectangle/cylinders
                center_x = self.obstacles[idx][0] + self.obstacles[idx][2] / 2
                center_y = self.obstacles[idx][1] + self.obstacles[idx][3] / 2

                # randomize the heights of obs
                if self.config.sim.random_static_obs_height:
                    center_z = np.random.uniform(-0.6, 0.5)
                # fix the heights of obs
                else:
                    center_z = 0.

                # randomize the table positions a bit
                if cur_uid in self.cylinder_obs_uids:
                    x_noise, y_noise = np.random.uniform(-0.2, 0.2, size=2)
                    center_x = center_x + x_noise
                    center_y = center_y + y_noise
                # we only change obstacle positions
                # 1. for all obstacles, in circle_crossing scenario, in every reset
                # 2. for all obstacles, in csl workspace, in first reset
                # 3. for cylinders, in csl workspace, in every reset
                if self.config.env.scenario == 'circle_crossing' or \
                   (self.config.env.scenario == 'csl_workspace' and self.first_epi) or \
                   (self.config.env.scenario == 'csl_workspace' and cur_uid in self.cylinder_obs_uids):
                    self._p.resetBasePositionAndOrientation(cur_uid,
                                                            [center_x, center_y, center_z],
                                                            self._p.getQuaternionFromEuler([0, 0, 0]))
                # update the present obs for this episode
                self.cur_obstacles[i] = self.obstacles[idx]

        # span the robot and humans in the arena
        self.generate_robot_humans(phase)
        # since self.human_num is changed in generate_robot_humans, need to redefine self.last_human_states here
        self.last_human_states = np.zeros((self.human_num, 5))

        # assign uids to humans and robot, and reset human & rob pos here
        for i in range(self.human_num):
            # find a free uid and assign it to human i
            cur_uid = self.free_human_uids.pop()
            self.humans[i].uid = cur_uid
            # print('human', i, 'uid:', cur_uid)
            self.used_human_uids.append(cur_uid)
            # height = np.random.uniform(0.2, 1.5)
            self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                    [self.humans[i].px, self.humans[i].py,
                                                     self.config.humans.height / 2],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))

        p.resetBasePositionAndOrientation(self.robot.uid, [self.robot.px, self.robot.py, 0],
                                          p.getQuaternionFromEuler([0, 0, self.robot.theta]))

        # initialize self.human_visibility
        _, _, self.human_visibility = self.get_num_human_in_fov()

        # record px, py, r of each human, used for crowd_sim_pc env
        self.cur_human_states = np.zeros((self.max_human_num, 3))
        for i in range(self.human_num):
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1 * self.nenv)) % self.case_size[phase]

        # initialize potential
        self.potential = -abs(
            np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        self._p.stepSimulation()  # refresh the simulator. Needed for the ray test

        # for IL data collection only
        self.orca_action = np.zeros(2)


    def create_goal_object(self):
        if self.first_epi:
            # add a cone to represent robot goal
            self.goal_uid = self.create_object(px=20, py=20, radius=0.5, height=1.5, shape=p.GEOM_SPHERE,
                                               color=[1, 0.84, 0, 1])

        self._p.resetBasePositionAndOrientation(self.goal_uid,
                                                [self.robot.gx, self.robot.gy, 2],
                                                self._p.getQuaternionFromEuler([0, 0, 0]))


    def reset(self, phase='train', test_case=None):
        # create robot, humans, and obstacles
        self.create_scenario(phase=phase, test_case=test_case)
        self.create_goal_object()

        # compute the observation
        ob = self.generate_ob(reset=True)
        # self.rob_prev_px, self.rob_prev_py = 0, 0

        if self.record:
            self.episodeRecoder.robot_goal.append([self.robot.gx, self.robot.gy])

        return ob



    # given desired left and right wheel velocity in sim,
    # return the actual left and right wheel velocity approximated from real trajectories
    def tb2_dynamics(self, left, right):
        left = np.clip(left, -11.5, 11.5)
        right = np.clip(right, -11.5, 11.5)

        # when an episode begins, the turtlebot needs sometime to gain speed from stationary
        if self.step_counter < 2:
            return 0, 0

        new_left, new_right = left, right
        left_noise, right_noise = np.random.normal(0, 0.15, size=2)
        return new_left + left_noise, new_right + right_noise

    def step(self, action, update=True):
        # print('Step', self.envStepCounter)

        human_actions = self.get_human_actions()

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)

        if self.config.env.action_space == 'continuous':
            left, right = action
        else:
            # to make it work with vec env wrapper and without the wrapper
            if isinstance(action, np.ndarray):
                action = action[0]

            delta_v, delta_w = self.action_convert[action]
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + delta_v, self.config.robot.v_min, self.config.robot.v_max)
            self.desiredVelocity[1] = np.clip(self.desiredVelocity[1] + delta_w, self.config.robot.w_min, self.config.robot.w_max)
            left = (2. * self.desiredVelocity[0] - 0.23 * self.desiredVelocity[1]) / (2. * 0.035)
            right = (2. * self.desiredVelocity[0] + 0.23 * self.desiredVelocity[1]) / (2. * 0.035)
        # print(self.desiredVelocity)

        # simulate the turtlebot dynamics
        left, right = self.tb2_dynamics(left, right)

        #
        # print(left, right)
        # todo: what should be the value of force (maximum motor force)?
        p.setJointMotorControl2(self.robot.uid, 0, p.VELOCITY_CONTROL, targetVelocity=left, force=10)
        p.setJointMotorControl2(self.robot.uid, 1, p.VELOCITY_CONTROL, targetVelocity=right, force=10)

        # get the new robot state from PyBullet, then set it to the robot instance (to keep consistency with the rest of repo)
        [robot_px, robot_py, _], quaternion_angle = p.getBasePositionAndOrientation(self.robot.uid)
        _, _, robot_yaw = p.getEulerFromQuaternion(quaternion_angle)
        [robot_vx, robot_vy, _,], [_, _, robot_wz] = p.getBaseVelocity(self.robot.uid)

        # print('robot vx:', robot_vx, 'robot vy:', robot_vy)
        # print('time', self.global_time, 'actual vx:', (robot_px - self.rob_prev_px)/self.time_step, 'vy:', (robot_py - self.rob_prev_py)/self.time_step)
        # self.rob_prev_px = robot_px
        # self.rob_prev_py = robot_py
        self.robot.set(robot_px, robot_py, self.robot.gx, self.robot.gy, robot_vx, robot_vy, robot_yaw)
        self.robot.w = robot_wz
        # self.robot.v = 0.035 / 2 * (left + right) # 0.035 is the wheel radius

        self.robot.v = np.sqrt(robot_vx ** 2 + robot_vy ** 2)
        # determine the robot's direction
        # Compute the forward direction vector in x, y coordinates
        forward_direction = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        # Convert the linear velocity to a numpy array
        linear_velocity_vector = np.array([robot_vx, robot_vy])  # Only x, y components
        # Compute the dot product between forward direction and linear velocity
        if np.dot(forward_direction, linear_velocity_vector) < 0:
            self.robot.v = - self.robot.v

        # print('v:', self.robot.v, 'w:', self.robot.w)

        # apply humans' actions
        # record px, py, r of each human, used for crowd_sim_pc env
        for i, human_action in enumerate(human_actions):
            if self.humans[i].isObstacle:
                if self.humans[i].isObstacle_period == np.inf:
                    continue
                elif self.step_counter >= self.humans[i].isObstacle_period:
                    self.humans[i].isObstacle = False
                    self.humans[i].isObstacle_period = np.inf
            self.humans[i].step(human_action)
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])
            # change the position of each human cylinder
            if self.humans[i].uid is None:
                print('human', i, 'uid = None')
            self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                    [self.humans[i].px, self.humans[i].py, self.config.humans.height/2],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter = self.step_counter + 1

        info={'info':episode_info}

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.config.sim.change_human_num_in_episode:
            self.change_human_num_periodically()

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if human.isObstacle:
                    continue
                # in csl_workspace scenarios, prevent the human from freezing near the goal or take unnecessary trajectories in tight spaces
                if self.config.env.scenario == 'circle_crossing':
                    cond = norm((human.gx - human.px, human.gy - human.py)) < human.radius
                else:
                    cond = norm((human.gx - human.px, human.gy - human.py)) < human.radius * 2
                if cond:
                    # print('update the goal of human', i)
                    # save the uid of the bullet cylinder
                    cur_uid = self.humans[i].uid
                    if self.robot.kinematics == 'holonomic':
                        self.humans[i] = self.generate_circle_crossing_human()
                        self.humans[i].id = i
                        # reset the old cylinder's pos for the new human
                        self.humans[i].uid = cur_uid
                        self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                                [self.humans[i].px, self.humans[i].py,
                                                                 self.config.humans.height / 2],
                                                                self._p.getQuaternionFromEuler([0, 0, 0]))
                    else:
                        need_to_remove = self.update_human_goal(human)
                        if need_to_remove:
                            # todo: this if statement will make human stay at goal once it arrives at the goal
                            if self.config.env.scenario == 'csl_workspace' and self.config.env.mode == 'sim2real':
                                human.isObstacle = True
                            else:
                                # remove this human and add a new human
                                self.remove_human(i)
                                self.add_human()

        # for _ in range(self.config.pybullet.frameSkip):
        self.scene.global_step()
        self.envStepCounter = self.envStepCounter + 1


        # compute the observation
        ob = self.generate_ob(reset=False)

        # if test.py requires to save slides, save top-down camera image to disk
        # only save the first 20 episodes
        if self.config.camera.render_checkpoint is not None and self.case_counter['test']<self.config.env.test_size:
            self.render_scene()

        if done:
            # move all cylinders to a dummy pos if an episode ends
            for i in range(self.human_num):
                self._p.resetBasePositionAndOrientation(self.humans[i].uid,
                                                        [20, 20, 0],
                                                        self._p.getQuaternionFromEuler([0, 0, 0]))
            if self.config.env.scenario == 'circle_crossing':
                for i in range(len(self.used_obs_uids)):
                    self._p.resetBasePositionAndOrientation(self.used_obs_uids[i],
                                                            [50, 50, 0],
                                                            self._p.getQuaternionFromEuler([0, 0, 0]))
            self._p.resetBasePositionAndOrientation(self.robot.uid,
                                                    [20, 20, 0],
                                                    self._p.getQuaternionFromEuler([0, 0, 0]))
            # free all used_human_uids
            self.free_human_uids.extend(copy.deepcopy(self.used_human_uids))
            self.used_human_uids.clear()
            # free all used_obs_uids
            self.free_obs_uids.extend(copy.deepcopy(self.used_obs_uids))
            self.used_obs_uids.clear()

        assert set(self.free_human_uids + self.used_human_uids) == set(self.all_human_uids)


        if self.record:
            self.episodeRecoder.wheelVelList.append([left, right])  # it is the calculated wheel velocity, not the measured
            self.episodeRecoder.actionList.append([self.desiredVelocity[0], self.desiredVelocity[1]])
            self.episodeRecoder.positionList.append([self.robot.px, self.robot.py])
            self.episodeRecoder.orientationList.append(self.robot.theta)
        if done:
            if len(self.images) != 0:
                imageio.mimsave(f"{self.save_dir}/output.gif", self.images)
                self.images.clear()
            if self.record:
                self.episodeRecoder.saveEpisode(self.case_counter['test'])

        return ob, reward, done, info

    # randomly select a new goal in the next region on the human's route
    # returns: False if the human finds its next goal, True if the human completes the route and cannot find next goal
    def update_human_goal(self, human):
        # don't need to do anything for static human
        if human.isObstacle or human.v_pref == 0:
            return False
        if self.config.env.scenario == 'csl_workspace':
            if human.route and len(human.route) > 0:
                goal_region = human.route.pop(0)
                human.gx = np.random.uniform(self.config.human_flow.regions[goal_region][0],
                                       self.config.human_flow.regions[goal_region][1])
                human.gy = np.random.uniform(self.config.human_flow.regions[goal_region][2],
                                       self.config.human_flow.regions[goal_region][3])
                # print('next goal:', goal_region, human.gx, human.gy)
                return False
            else:
                # need to remove this human
                return True
        elif self.config.env.scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                noise_range = self.config.sim.human_pos_noise_range
                gx_noise = np.random.uniform(0, 1) * noise_range
                gy_noise = np.random.uniform(0, 1) * noise_range
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                # check collision of start and goal position with static obstacles
                if self.add_static_obs:
                    if self.circle_in_obstacles(gx, gy, human.radius + self.discomfort_dist):
                        collide = True
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy

            # for now, never removes a human in circle_crossing scenario
            return False



    # remove a human with given index from the environment
    # idx: the index of the human in self.humans
    def remove_human(self, idx):
        # recycle the pybullet cylinder
        self._p.resetBasePositionAndOrientation(self.humans[idx].uid,
                                                [20, 20, 0],
                                                self._p.getQuaternionFromEuler([0, 0, 0]))
        self.used_human_uids.remove(self.humans[idx].uid)
        self.free_human_uids.append(self.humans[idx].uid)
        # remove the Human instance
        self.humans.pop(idx)
        # update variables
        self.human_num = self.human_num - 1
        self.last_human_states = np.delete(self.last_human_states, idx, axis=0) # todo: check this
        return

    # remove a human with given index from the environment
    def add_human(self):
        # generate a new human instance, it will be appended to the end of self.humans list
        self.generate_random_human_position(human_num=1)
        self.humans[-1].id = self.human_num
        # assign a cylinder to the new human
        self.humans[-1].uid = self.free_human_uids.pop()
        self.used_human_uids.append(self.humans[-1].uid)
        self._p.resetBasePositionAndOrientation(self.humans[-1].uid,
                                                [self.humans[-1].px, self.humans[-1].py, self.config.humans.height / 2],
                                                self._p.getQuaternionFromEuler([0, 0, 0]))
        # update variables
        self.human_num = self.human_num + 1
        self.last_human_states = np.concatenate((self.last_human_states, np.array([[15, 15, 0, 0, 0.3]])), axis=0)
        return

    def change_human_num_periodically(self):
        if self.human_num_range > 0 and self.global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - self.min_human_num
                else:
                    max_remove_num = min(self.human_num - self.min_human_num, (self.human_num - 1) - max(self.observed_human_ids))
                remove_num = np.random.randint(low=0, high=max_remove_num + 1)
                for _ in range(remove_num):
                    self.remove_human(-1)
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    for i in range(self.human_num, self.human_num + add_num):
                        if i == self.config.sim.human_num + self.human_num_range:
                            break
                        self.add_human()

        assert self.min_human_num <= self.human_num <= self.max_human_num

    # use pybullet instead of geometry to check collision and find the closest distance
    def collision_checker(self):
        '''
        check whether robot collides with other objects
        returns:(collision, dmin)
                collision: True if there is a collision, False if there's no collision
                dmin: the distance between the robot and its closest human
        '''
        # collision detection
        dmin = float('inf')

        collision = False
        collision_with = None
        # check collision with humans
        for i, human in enumerate(self.humans):
            contact_points = p.getContactPoints(self.robot.uid, human.uid)
            if contact_points:
                collision = True
                collision_with = 'human'
                break
            else:
                closest_points = p.getClosestPoints(self.robot.uid, human.uid, distance=1000.0, linkIndexA=-1,
                                                    linkIndexB=-1)
                if closest_points:
                    # The closest points are typically in closest_points[0], check its distance
                    closest_dist = closest_points[0][8]
                    if closest_dist < dmin:
                        dmin = closest_dist

        # check collision with obstacles
        # if collision is already True, we don't overwrite or double count the collision with obstacles
        if not collision and self.add_static_obs:
            for i in range(self.obs_num):
                if p.getContactPoints(self.robot.uid, int(self.cur_obstacles[i, -1])):
                    collision = True
                    collision_with = 'obstacle'

        # check collision with walls
        if not collision and self.config.sim.borders:
            # -0.5: count for the thickness of wall
            arena_size = self.arena_size + self.config.sim.human_pos_noise_range - 0.5
            if self.robot.px + self.robot.radius > arena_size or self.robot.px - self.robot.radius < -arena_size or \
                    self.robot.py + self.robot.radius > arena_size or self.robot.py - self.robot.radius < -arena_size:
                collision = True
                collision_with = 'wall'
        return collision, dmin, collision_with

    def close(self):
        if self.ownsPhysicsClient:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1


    def render(self, mode='human'):
        # no need to implement this function
        pass