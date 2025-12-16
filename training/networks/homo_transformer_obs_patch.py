import torch.nn as nn
import torch

from .selfAttn_srnn_merge import selfAttn_merge_SRNN
from training.networks.network_utils import Flatten, EndRNNLidar, reshapeT


class Homo_Transformer_Obs_Patch(selfAttn_merge_SRNN):
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

        self.om_channel_num = 1
        self.om_embed_size = config.SRNN.obs_embedding_size
        self.om_patch_size = obs_space_dict['om'].shape[-1]
        # initialize lidar point cloud embedding layers
        self.om_encoder = nn.Sequential(self.init_(nn.Conv2d(1, 16, 3, stride=2)), nn.ReLU(),
                                        # (1, 360) -> (32, 176)
                                        self.init_(nn.Conv2d(16, 32, 3, stride=2)), nn.ReLU(),  # (32, 176) -> (32, 86)
                                        self.init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(),
                                        self.init_(nn.Conv2d(32, 32, 5, stride=1)), nn.ReLU(),
                                        Flatten(),
                                        self.init_(nn.Linear(288, self.om_embed_size)),  # todo
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
        om_in = inputs['om'].reshape(seq_length * nenv, 1, self.om_patch_size, self.om_patch_size)

        hidden_states_node_RNNs = reshapeT(rnn_hxs['rnn'], 1, nenv)

        masks = reshapeT(masks, seq_length, nenv)

        # embed robot states
        robot_states = self.robot_linear(robot_states)

        # embed om patches
        om_features = self.om_encoder(om_in)
        # reshape it back to dim=4
        om_features = om_features.view(seq_length, nenv, 1, self.om_embed_size)

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
            = self.humanNodeRNN(robot_states, hidden_attn_weighted, om_features, hidden_states_node_RNNs, masks)

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