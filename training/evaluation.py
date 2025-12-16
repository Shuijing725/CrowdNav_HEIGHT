import numpy as np
import torch

from crowd_sim.envs.utils.info import *
from training.networks import utils

import time


def evaluate(actor_critic, eval_envs, num_processes, device, config, logging, test_args):
    """Evaluate the policy model (actor_critic) in multiple testing episodes.

        Parameters:
        actor_critic : torch.nn.Module
            The policy model to evaluate.
        eval_envs : VecEnv
            The vectorized environments for evaluation.
        num_processes : int
            Number of parallel environments to run.
        device : torch.device
            Device for running evaluation (CPU or CUDA).
        config : Config
            Configuration object with environment and training settings.
        logging : logging.Logger
            Logger for evaluation information.
        test_args : argparse.Namespace
            Additional testing arguments like visualization options.
        """

    
    test_size = config.env.test_size

    eval_episode_rewards = []

    # initialize the RNN hidden states
    eval_recurrent_hidden_states = {}
    if config.robot.policy in ['srnn', 'dsrnn_obs_pc', 'dsrnn_obs_vertex']:
        node_num = 1
        edge_num = actor_critic.base.human_num + 1 + actor_critic.base.obs_num
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num,
                                                                     config.SRNN.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                           config.SRNN.human_node_rnn_size,
                                                                           device=device)

    else:
        eval_recurrent_hidden_states['rnn'] = torch.zeros(num_processes, 1, config.SRNN.human_node_rnn_size,
                                                          device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    # initialize testing metrics
    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []

    success = 0
    collision = 0
    collision_human = 0
    collision_obs = 0
    collision_wall = 0

    timeout = 0
    too_close_ratios = []
    min_dist = []
    cumulative_rewards = []

    collision_cases = []
    collision_human_cases = []
    collision_obs_cases = []
    collision_wall_cases = []

    timeout_cases = []
    runtime = 0.0
    gamma = 0.99
    baseEnv = eval_envs.venv.envs[0].env

    t = 0

    obs = eval_envs.reset()

    runtime_counter = 0

    # the main testing loop
    for k in range(test_size):
        t += 1
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0

        global_time = 0.0
        path = 0.0
        too_close = 0.

        last_pos = obs['robot_node'][0, 0, 0:2].cpu().numpy()  # robot px, py

        # episode_runtime = 0

        while not done:
            stepCounter = stepCounter + 1
            runtime_counter += 1
            
            t0 = time.perf_counter()
            
            # given observation, forward the robot policy to get action
            if not test_args.dwa and not test_args.orca:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            elif test_args.orca:
                action = actor_critic.predict(eval_envs.venv.envs[0].env)
            else: # for DWA
                u, predicted_trajectory, curr_state, action = actor_critic.predict(eval_envs.venv.envs[0].env)

            dt = time.perf_counter() - t0

            runtime += dt
            
            # print(f"actor_critic.act took {dt*1e3:.3f} ms")

            if not done:
                global_time = baseEnv.global_time
            if test_args.visualize:
                eval_envs.render()

            # step the environment to get reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            path = path + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)

            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            rewards.append(rew)

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

            if runtime_counter >= 500 and test_args.test_runtime:
                break
        
        if runtime_counter >= 500 and test_args.test_runtime:
            print(f"Avg runtime for policy: {runtime/runtime_counter:.4f}")
            break

        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        path_lengths.append(path)
        too_close_ratios.append(too_close / stepCounter * 100)
        # episode_runtime /= stepCounter
        # runtime[k] = episode_runtime

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['info'], CollisionHuman):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            collision_human += 1
            collision_human_cases.append(k)
            print('Collision with human')
        elif isinstance(infos[0]['info'], CollisionObs):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            collision_obs += 1
            collision_obs_cases.append(k)
            print('Collision with obstacle')
        elif isinstance(infos[0]['info'], CollisionWall):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            collision_wall += 1
            collision_wall_cases.append(k)
            print('Collision with wall')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            print('Time out')
        else:
            raise ValueError('Invalid end signal from environment')

        cumulative_rewards.append(sum([pow(gamma, t * baseEnv.robot.time_step * baseEnv.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]))

    # if test_args.test_runtime:
        # print("Avg runtime for each episode:")
        # for episode, i in runtime.items():
        #     print(f"  Episode {episode}: {i:.4f} seconds")
        # print(f"Avg runtime for all episodes: {np.mean(list(runtime.values())):.4f} seconds")


    # after all testing episodes are done,
    # calculate and log results
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size

    collision_human_rate = collision_human / test_size
    collision_obs_rate = collision_obs / test_size
    collision_wall_rate = collision_wall / test_size
    print(success, collision, timeout)
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else baseEnv.time_limit  # baseEnv.env.time_limit

    extra_info = ''
    phase = 'test'
    logging.info(
        '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}'.
        format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, avg_nav_time))
    logging.info(
        'collision rate with humans: {:.2f}, with obstacles: {:.2f}, with walls: {:.2f}, '.
        format(collision_human_rate, collision_obs_rate, collision_wall_rate))
    if phase in ['val', 'test']:
        total_time = sum(success_times + collision_times + timeout_times)
        if min_dist:
            avg_min_dist = np.average(min_dist)
        else:
            avg_min_dist = float("nan")
        logging.info('average intrusion ratio: %.2f and average minimal distance during intrusions: %.2f',
                     np.mean(too_close_ratios), avg_min_dist)

    logging.info(
        '{:<5} {}has average path length: {:.2f}'.
        format(phase.upper(), extra_info, sum(path_lengths) / test_size))
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Collision with Human cases: ' + ' '.join([str(x) for x in collision_human_cases]))
    logging.info('Collision with Obstacle cases: ' + ' '.join([str(x) for x in collision_obs_cases]))
    logging.info('Collision with Wall cases: ' + ' '.join([str(x) for x in collision_wall_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
