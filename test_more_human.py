import logging
import argparse
import os
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


from training.networks.envs import make_vec_envs
from training.evaluation import evaluate
from crowd_sim import *
from training.networks.model import Policy
from crowd_nav.policy.dwa import DWA


def main():
	# the following parameters will be determined for each test run
	parser = argparse.ArgumentParser('Parse configuration file')
	# the model directory that we are testing
	parser.add_argument('--model_dir', type=str, default='data/ours_RH_HH_hallwayEnv_new')
	# we are testing an IL policy or RL policy
	parser.add_argument('--test_il', default=False, action='store_true')
	parser.add_argument('--visualize', default=False, action='store_true')
	# if -1, it will run 500 different cases; if >=0, it will run the specified test case repeatedly
	parser.add_argument('--test_case', type=int, default=-1)
	# dwa: True, others: False
	parser.add_argument('--dwa', default=False, action='store_true')
	# use cpu if do not have GPU or CUDA version does not match
	# No need to change if the computer has a GPU
	# otherwise: set to True
	parser.add_argument('--cpu', default=False, action='store_true')
	# model weight file you want to test
	parser.add_argument('--test_model', type=str, default='237800.pt')
	# parser.add_argument('--test_model', type=str, default='82000.pt')

	# display lidar rays or not
	parser.add_argument('--visualize_lidar_rays', default=False, action='store_true')
	# save slideshow
	parser.add_argument('--save_slides', default=False, action='store_true')

	test_args = parser.parse_args()

	from importlib import import_module
	model_dir_temp = test_args.model_dir
	if model_dir_temp.endswith('/'):
		model_dir_temp = model_dir_temp[:-1]
	# import config class from saved directory
	# if not found, import from the default directory
	try:
		model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
		model_arguments = import_module(model_dir_string)
		Config = getattr(model_arguments, 'Config')
	except:
		print('Failed to get Config function from ', test_args.model_dir, '/config.py')
		from crowd_nav.configs.config import Config


	config = Config()
	config.sim.human_num = 12
	config.sim.human_num_range = 2

	if test_args.visualize and test_args.visualize_lidar_rays:
		config.lidar.visualize_rays = True
	else:
		config.lidar.visualize_rays = False

	if test_args.save_slides:
		# save image slides to disk, remove .pt in path
		config.camera.render_checkpoint = test_args.test_model[:-3]
		# don't render because it will result in a different testing result
		test_args.visualize = False
		# only test for 20 episodes
		config.env.test_size = 20

	# configure logging and device
	# print test result in log file
	log_file = os.path.join(test_args.model_dir,'test')
	if not os.path.exists(log_file):
		os.mkdir(log_file)
	if test_args.visualize or test_args.save_slides:
		log_file = os.path.join(test_args.model_dir, 'test', test_args.test_model+'test_visual_more_human.log')
	else:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_'+test_args.test_model+config.env.scenario+'_more_human.log')



	file_handler = logging.FileHandler(log_file, mode='w')
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

	# logging.info(f"Test parameters: \ngoal weight: {config.dwa.to_goal_cost_gain} \nspeed weight: {config.dwa.speed_cost_gain} \npredict time: {config.dwa.predict_time} \ndynamics weight: {config.dwa.dynamics_weight} \nrobot stuck flag: {config.dwa.robot_stuck_flag_cons} \nstuck action: {config.dwa.stuck_action}\n")
	logging.info('robot FOV %f', config.robot.FOV)
	logging.info('humans FOV %f', config.humans.FOV)

	torch.manual_seed(config.env.seed)
	torch.cuda.manual_seed_all(config.env.seed)
	if config.training.cuda:
		if config.training.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False


	torch.set_num_threads(1)
	device = torch.device("cuda" if config.training.cuda else "cpu")
	if test_args.cpu:
		device = torch.device("cpu")
	print(device)
	logging.info('Create other envs with new settings')


	if test_args.visualize:
		# for pybullet env
		if 'CrowdSim3D' in config.env.env_name:
			config.sim.render = True
			ax = None
		# for the old 2D envs
		else:
			fig, ax = plt.subplots(figsize=(7, 7))
			ax.set_xlim(-6, 6)
			ax.set_ylim(-6, 6)
			ax.set_xlabel('x(m)', fontsize=16)
			ax.set_ylabel('y(m)', fontsize=16)
			plt.ion()
			plt.show()
	else:
		ax = None


	load_path=os.path.join(test_args.model_dir,'checkpoints', test_args.test_model)
	print(load_path)


	env_name = config.env.env_name

	eval_dir = os.path.join(test_args.model_dir,'eval')
	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)

	envs = make_vec_envs(env_name, config.env.seed, 1,
						 config.reward.gamma, eval_dir, device, allow_early_resets=True,
						 config=config, ax=ax, test_case=test_args.test_case)

	# if dwa, skip this part by adding "if"
	if test_args.dwa:
		actor_critic = DWA(config)
	else:
		actor_critic = Policy(
			envs.observation_space.spaces,  # pass the Dict into policy to parse
			envs.action_space,
			base_kwargs=config,
			base=config.robot.policy)

		actor_critic.load_state_dict(torch.load(load_path, map_location=device))
		actor_critic.base.nenv = 1

	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
		nn.DataParallel(actor_critic).to(device)


	evaluate(actor_critic, envs, 1, device, config, logging, test_args)



if __name__ == '__main__':
	main()
