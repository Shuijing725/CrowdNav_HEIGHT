# This script is a training loop for a reinforcement learning navigation policy  in a crowd simulation environment.
# It initializes configurations, sets up the environment, defines the policy, and executes the training loop
# with logging and model checkpointing.

import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from training import algo
from training.networks import utils
from training.networks.envs import make_vec_envs
from training.networks.model import Policy
from training.networks.storage import RolloutStorage

from crowd_nav.configs.config import Config
from crowd_sim import *


def main():
    config = Config()

    # Create directories for saving logs and weights
    if not os.path.exists(config.training.output_dir):
        os.makedirs(config.training.output_dir)
    elif not config.training.overwrite:
        raise ValueError('output_dir already exists!')

    save_config_dir = os.path.join(config.training.output_dir, 'configs')
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)
    shutil.copy('crowd_nav/configs/config.py', save_config_dir)
    shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)

    # Set random seeds for reproducibility and configure device
    torch.manual_seed(config.env.seed)
    torch.cuda.manual_seed_all(config.env.seed)
    if config.training.cuda and torch.cuda.is_available():
        if config.training.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    torch.set_num_threads(config.training.num_threads)
    device = torch.device("cuda" if config.training.cuda and torch.cuda.is_available() else "cpu")


    # Define environment and visualization setup
    env_name = config.env.env_name
    ax = None
    if env_name != 'CrowdSim3D-v0' and config.sim.render:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        plt.ion()
        plt.show()

    if config.sim.render:
        config.training.num_processes = 1
        config.ppo.num_mini_batch = 1

    # Create the environment manager
    envs = make_vec_envs(env_name, config.env.seed, config.training.num_processes,
                         config.reward.gamma, None, device, False, config=config, ax=ax)

    # Initialize policy and count parameters
    actor_critic = Policy(envs.observation_space.spaces, envs.action_space, base_kwargs=config,
                          base=config.robot.policy)
    pytorch_total_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
    print('total num of parameters', pytorch_total_params)

    # Define training method and initialize rollout storage
    method = 'baseline' if config.robot.policy in ['srnn', 'dsrnn_obs_pc', 'dsrnn_obs_vertex'] else 'ours'
    rollouts = RolloutStorage(config.ppo.num_steps, config.training.num_processes,
                              envs.observation_space.spaces, envs.action_space,
                              config.SRNN.human_node_rnn_size, method=method)

    # Load model checkpoint if resuming training
    if config.training.resume != 'none':
        load_path = config.training.load_path
        # actor_critic.load_state_dict(torch.load(load_path), strict=False)

        prev_checkpoint = torch.load(load_path)  # Load the checkpoint
        model_dict = actor_critic.state_dict()  # Get the current model's state_dict
        # Filter out mismatched keys
        filtered_state_dict = {k: v for k, v in prev_checkpoint.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

        # Update the model's state_dict with the filtered parameters
        model_dict.update(filtered_state_dict)

        # Load the updated state_dict into the model
        actor_critic.load_state_dict(model_dict)

        print("Loaded the following checkpoint:", load_path)

    # Allow multi-GPU usage for larger batch sizes
    nn.DataParallel(actor_critic).to(device)

    # Initialize PPO algorithm
    agent = algo.PPO(
        actor_critic,
        config.ppo.clip_param,
        config.ppo.epoch,
        config.ppo.num_mini_batch,
        config.ppo.value_loss_coef,
        config.ppo.entropy_coef,
        lr=config.training.lr,
        eps=config.training.eps,
        max_grad_norm=config.training.max_grad_norm
    )

    # Start training loop
    obs = envs.reset()
    if isinstance(obs, dict):
        for key in obs:
            rollouts.obs[key][0].copy_(obs[key])
    else:
        rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    start = time.time()
    num_updates = int(config.training.num_env_steps) // config.ppo.num_steps // config.training.num_processes
    start_epoch = 0
    time_diff = 0

    # Resume optimizer and epoch if resuming training from checkpoint
    if config.training.resume != 'none' and os.path.exists(config.training.load_path.split(".")[0] + "_checkpoint.pt"):
        load_path = config.training.load_path.split(".")[0] + "_checkpoint.pt"
        load_checkpoint = torch.load(load_path)
        agent.optimizer.load_state_dict(load_checkpoint["optimizer_data_dict"])
        start_epoch = load_checkpoint["epoch"]
        time_diff = load_checkpoint["time_diff"]

        print("Loaded checkpoint (optimizer and epoch):", load_path)

    for j in range(start_epoch, num_updates+start_epoch):
        # Adjust learning rate if linear decay is enabled
        if config.training.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates, config.training.lr)

        # Loop through each step
        for step in range(config.ppo.num_steps):
            with torch.no_grad():
                # Retrieve actions from policy
                rollouts_obs = {key: rollouts.obs[key][step] for key in rollouts.obs}
                rollouts_hidden_s = {key: rollouts.recurrent_hidden_states[key][step] for key in
                                     rollouts.recurrent_hidden_states}
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts_obs, rollouts_hidden_s, rollouts.masks[step])

            if config.sim.render:
                envs.render()

            # Execute action in environment and record results
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info:
                    episode_rewards.append(info['episode']['r'])

            # Prepare masks for reset environments
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        # Compute returns for updates
        with torch.no_grad():
            rollouts_obs = {key: rollouts.obs[key][-1] for key in rollouts.obs}
            rollouts_hidden_s = {key: rollouts.recurrent_hidden_states[key][-1] for key in
                                 rollouts.recurrent_hidden_states}
            next_value = actor_critic.get_value(rollouts_obs, rollouts_hidden_s, rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, config.ppo.use_gae, config.reward.gamma, config.ppo.gae_lambda,
                                 config.training.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Save model checkpoints
        if j % config.training.save_interval == 0 or j == num_updates - 1:
            save_path = os.path.join(config.training.output_dir, 'checkpoints')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            end = time.time()
            checkpoint = {'epoch': j, 'optimizer_data_dict': agent.optimizer.state_dict(),
                          'time_diff': (end - start + time_diff)}
            torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i' % j + ".pt"))
            torch.save(checkpoint, os.path.join(save_path, '%.5i' % j + "_checkpoint.pt"))

        # Logging updates
        if j % config.training.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config.training.num_processes * config.ppo.num_steps
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {}\n Last {} training episodes: mean/median reward "
                  "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                j, total_num_steps, int(total_num_steps / (end - start + time_diff)), len(episode_rewards),
                np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))

            # Log to CSV
            df = pd.DataFrame({
                'misc/nupdates': [j],
                'misc/total_timesteps': [total_num_steps],
                'fps': int(total_num_steps / (end - start)),
                'eprewmean': [np.mean(episode_rewards)],
                'loss/policy_entropy': dist_entropy,
                'loss/policy_loss': action_loss,
                'loss/value_loss': value_loss
            })

            # Append or create new CSV based on file existence
            csv_path = os.path.join(config.training.output_dir, 'progress.csv')
            if os.path.exists(csv_path) and j > 20:
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False)

    envs.close()


if __name__ == '__main__':
    main()
