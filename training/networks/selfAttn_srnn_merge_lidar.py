import torch.nn as nn
import torch

from .selfAttn_srnn_merge import selfAttn_merge_SRNN
from training.networks.network_utils import Flatten, EndRNNLidar, reshapeT


class selfAttn_merge_SRNN_lidar(selfAttn_merge_SRNN):
    """
    Class representing the SRNN model
    """
    def __init__(self, obs_space_dict, config):
        """
        Initializer function
        params:
        config : Training arguments
        infer : Training or test time (True at test time)
        """
        super().__init__(obs_space_dict, config)

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNNLidar(config)

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

        # embed human states, add various attention weights to human embeddings
        # human-human self attention
        if self.config.SRNN.use_self_attn:
            # [seq len, nenv, human num, 128]
            spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
        else:
            spatial_attn_out = spatial_edges
        # [seq len, nenv, human num, 64] (64 is human_embedding_size)
        output_spatial = self.spatial_linear(spatial_attn_out)  # (seq len, nenv, human num, 64)

        # robot-human attention
        if self.config.SRNN.use_hr_attn:
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
        else:
            # take sum of all human embeddings (without being weighted by RH attention scores)
            hidden_attn_weighted = torch.sum(output_spatial, dim=2, keepdim=True)


        # Do a forward pass through nodeRNN
        outputs, h_nodes \
            = self.humanNodeRNN(robot_states, hidden_attn_weighted, lidar_features, hidden_states_node_RNNs, masks)

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