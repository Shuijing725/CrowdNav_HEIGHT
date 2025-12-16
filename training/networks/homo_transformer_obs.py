import torch.nn as nn
import torch
import numpy as np

from training.networks.utils import init
from training.networks.network_utils import Flatten, reshapeT, EndRNNLidar
from .dsrnn_obs_vertex import HumanHumanEdgeRNN
from .selfAttn_srnn_merge import SpatialEdgeSelfAttn

class Homo_Transformer_Obs(nn.Module):
    """
    Class representing the SRNN model
    """
    def __init__(self, obs_space_dict, config):
        """
        Initializer function
        params:
        obs_space_dict : The observation space dictionary of the gym environment being used to generate data
        config : Training arguments
        """
        super(Homo_Transformer_Obs, self).__init__()

        # initialize variables
        self.is_recurrent = True
        self.config = config

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = config.ppo.num_steps
        self.nenv = config.training.num_processes
        self.nminibatch = config.ppo.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = config.SRNN.human_node_rnn_size
        self.human_human_edge_rnn_size = config.SRNN.human_human_edge_rnn_size
        self.output_size = config.SRNN.human_node_output_size

        self.init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0), np.sqrt(2))

        # initialize actor and critic
        hidden_size = self.output_size

        self.actor = nn.Sequential(
            self.init_(nn.Linear(self.config.SRNN.human_node_rnn_size, hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            self.init_(nn.Linear(self.config.SRNN.human_node_rnn_size, hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = self.init_(nn.Linear(hidden_size, 1))

        # initialize robot embedding layers
        robot_size = obs_space_dict['robot_node'].shape[1]

        self.robot_linear = nn.Sequential(self.init_(nn.Linear(robot_size, config.SRNN.robot_embedding_size)),
                                          nn.ReLU())

        # initialize lidar point cloud embedding layers
        # lookup table: given a lidar angular resolution, what is the output size of lidar CNN after flattening
        # 608 if angular resolution is 1, 256 if angular resolution of lidar is 2
        self.lidar_conv_out_size_lookup = {1: 608, 2: 256, 4: 64}

        self.lidar_input_size = int(360. / self.config.lidar.angular_res)
        self.lidar_embed_size = config.SRNN.obs_embedding_size
        if config.env.env_name == 'CrowdSim3DSeg-v0':
            self.lidar_channel_num = 4
            self.lidar_embed_conv_out_size = 608
        else:
            self.lidar_channel_num = 1
            self.lidar_embed_conv_out_size = self.lidar_conv_out_size_lookup[self.config.lidar.angular_res]

        # Linear layers to embed inputs
        # 1d conv
        self.lidar_embed = nn.Sequential(self.init_(nn.Conv1d(self.lidar_channel_num, 16, 10, stride=2)), nn.ReLU(),
                                         # (1, 360) -> (32, 176)
                                         self.init_(nn.Conv1d(16, 32, 5, stride=2)), nn.ReLU(),  # (32, 176) -> (32, 86)
                                         self.init_(nn.Conv1d(32, 32, 5, stride=2)), nn.ReLU(),  # (32, 86) -> (32, 41)
                                         self.init_(nn.Conv1d(32, 32, 5, stride=2)), nn.ReLU(),  # (32, 41) -> (32, 19)
                                         Flatten(),
                                         self.init_(nn.Linear(self.lidar_embed_conv_out_size, self.lidar_embed_size)),
                                         nn.ReLU(),
                                         )

        # initialize human embedding layers
        # todo: did not consider prediction for now
        if config.ob_space.add_human_vel:
            human_size = 4
        else:
            human_size = 2
        self.human_linear = nn.Sequential(self.init_(nn.Linear(human_size, config.SRNN.human_embedding_size)),
                                          nn.ReLU())

        # todo: initialize the self attention network that takes everything

        self.spatial_attn = SpatialEdgeSelfAttn(config)
        self.spatial_linear = nn.Sequential(
            self.init_(nn.Linear(self.config.SRNN.self_attn_size, config.SRNN.human_embedding_size)), nn.ReLU())

        # todo: initialize the GRU
        self.RNN = HumanHumanEdgeRNN(config, type='homo_transformer')


    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_states = reshapeT(inputs['robot_node'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        # [seq len, batch size, 2, pc num] -> [seq_len*batch_size,2, pc num]
        lidar_in = inputs['point_clouds'].reshape(seq_length * nenv, self.lidar_channel_num, self.lidar_input_size)

        hidden_states_node_RNNs = reshapeT(rnn_hxs['rnn'], 1, nenv)

        masks = reshapeT(masks, seq_length, nenv)

        # embed robot states
        robot_states = self.robot_linear(robot_states)

        # embed lidar pc
        lidar_features = self.lidar_embed(lidar_in)
        # reshape it back to dim=4
        lidar_features = lidar_features.view(seq_length, nenv, 1, self.lidar_embed_size)

        # concat [robot states, lidar features, all human states]
        human_features = self.human_linear(spatial_edges)
        all_features = torch.cat([robot_states, lidar_features, human_features], dim=-2)
        # +2 to treat robot and obstacle point cloud as the extra 2 nodes/tokens to the self attention
        spatial_attn_out=self.spatial_attn(all_features, detected_human_num+2).view(seq_length, nenv, self.human_num+2, -1)

        spatial_attn_out = self.spatial_linear(spatial_attn_out)  # (seq len, nenv, human num, 64)
        spatial_attn_out = torch.sum(spatial_attn_out, dim=-2, keepdim=True)

        # Do a forward pass through nodeRNN
        outputs, h_nodes = self.RNN(spatial_attn_out, hidden_states_node_RNNs, masks)

        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['rnn'] = all_hidden_states_node_RNNs

        # x is the output of the robot node and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs