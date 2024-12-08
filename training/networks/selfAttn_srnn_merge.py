import numpy as np
import torch.nn as nn
import torch
from training.networks.utils import init
from training.networks.network_utils import RNNBase, reshapeT

class SpatialEdgeSelfAttn(nn.Module):
    def __init__(self, config):
        super(SpatialEdgeSelfAttn, self).__init__()
        self.config = config

        if config.robot.policy == 'selfAttn_merge_srnn_lidar_human_pc':
            self.input_size = config.SRNN.human_embedding_size # 48
        elif config.robot.policy == 'homo_transformer_obs':
            self.input_size = config.SRNN.robot_embedding_size
        else:
            # Store required sizes
            if self.config.env.env_name in ['CrowdSimVarNum-v0']:
                self.input_size = 2
            elif self.config.env.env_name in ['CrowdSim3DTB-v0', 'CrowdSim3DTbObs-v0', 'CrowdSim3DTbObsHie-v0','rosTurtlebot2iEnv-v0']:
                if self.config.ob_space.add_human_vel:
                    self.input_size = 4
                else:
                    self.input_size = 2
            else:
                raise ValueError("Unknown environment name")

        self.num_attn_heads=8
        self.attn_size=self.config.SRNN.self_attn_size


        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, self.attn_size), nn.ReLU()
                                             )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        #self.multihead_attn = MultiHeadAttention(heads=self.num_attn_heads, d_model=self.attn_size)
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)



        # linear layer for concatenated features (spatial encoded + temporal)
        # self.cat_linear = nn.Linear(self.rnn_size*2, self.rnn_size)

        # self.cat_linear =nn.Sequential(
        #                 init_(nn.Linear(self.embedding_size*2, self.embedding_size)), nn.ReLU(),
        #                 init_(nn.Linear(self.embedding_size, 1)))

    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if not self.config.training.cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask


    def forward(self, inp, each_seq_len):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len: the true length of the sequence. Should be the number of detected humans
        '''
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]
        seq_len, nenv, max_human_num, _ = inp.size()
        attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
        attn_mask=attn_mask.squeeze(1) # if we use pytorch builtin function
        input_emb=self.embedding_layer(inp).view(seq_len*nenv, max_human_num, -1) # [seq_len*nenv, max_human_num, self.attn_size]
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # [max_human_num, seq_len*nenv, self.attn_size]
        q=self.q_linear(input_emb)
        k=self.k_linear(input_emb)
        v=self.v_linear(input_emb)

        # [max_human_num, seq_len*nenv, self.attn_size]
        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        z=torch.transpose(z, dim0=0, dim1=1) # [seq_len*nenv, max_human_num, self.attn_size]
        return z


class HumanRobotEdgeRNN(RNNBase):

    def __init__(self, config):
        super(HumanRobotEdgeRNN, self).__init__(config, edge=True)

        self.config = config

        # Store required sizes
        self.rnn_size = config.SRNN.human_human_edge_rnn_size
        self.embedding_size = config.SRNN.human_human_edge_embedding_size
        self.input_size = 512

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.relu = nn.ReLU()


    def forward(self, inp, h, masks):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)

        x, h_new = self._forward_gru(encoded_input, h, masks)

        return x, encoded_input, h_new

class EdgeAttention_M(nn.Module):
    '''
    Class representing the attention module
    attn_type: RH means robot-human attention, RO means robot-obstacle attention
    '''
    def __init__(self, config):
        '''
        Initializer function
        params:
        config : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention_M, self).__init__()

        self.config = config

        # Store required sizes
        self.human_embedding_size = config.SRNN.human_embedding_size

        self.num_attention_head = config.SRNN.hr_attn_head_num
        self.attention_size = config.SRNN.hr_attention_size

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        for _ in range(self.num_attention_head):
            self.temporal_edge_layer.append(nn.Linear(self.human_embedding_size, self.attention_size))
            # Linear layer to embed spatial edgeRNN hidden states
            self.spatial_edge_layer.append(nn.Linear(self.human_embedding_size, self.attention_size))

        if self.num_attention_head > 1:
            self.final_attn_linear = nn.Linear(self.human_embedding_size * self.num_attention_head, self.human_embedding_size)

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if not self.config.training.cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        # [seq len, nenv, human num]
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(np.round(attn[0, 0, 0].cpu().numpy(), 2))

        # reshape h_spatials and attn
        # [seq_len*nenv, human num, attention size] -> [seq_len*nenv, attention size, human num]
        h_spatials = h_spatials.view(seq_len * nenv, self.human_num, h_size).permute(0, 2, 1)

        # add all weighted human embeddings together, size of weight_value is [batch, attention size, 1]
        attn = attn.view(seq_len * nenv, self.human_num).unsqueeze(-1)  # [seq_len*nenv, human num, 1]
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, 1, h_size)  # [seq_len, 12, 6 or 1, 256]

        return weighted_value, attn


    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]

class EndRNN(RNNBase):
    '''
    Class representing human Node RNNs in the st-graph
    '''
    def __init__(self, config):
        '''
        Initializer function
        params:
        config : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(config, edge=False)

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(config.SRNN.robot_embedding_size + config.SRNN.human_embedding_size, config.SRNN.human_human_edge_rnn_size)


    def forward(self, robot_s, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input robot and weighted human embeddings
        concat_encoded = torch.cat((robot_s, h_spatial_other), -1)

        concat_encoded = self.encoder_linear(concat_encoded)
        x, h_new = self._forward_gru(concat_encoded, h, masks)

        return x, h_new

class selfAttn_merge_SRNN(nn.Module):
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
        super(selfAttn_merge_SRNN, self).__init__()
        self.is_recurrent = True
        self.config=config

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = config.ppo.num_steps
        self.nenv = config.training.num_processes
        self.nminibatch = config.ppo.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = config.SRNN.human_node_rnn_size
        self.human_human_edge_rnn_size =  config.SRNN.human_human_edge_rnn_size
        self.output_size = config.SRNN.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN(config)

        # Initialize robot-human attention module
        self.attn = EdgeAttention_M(config)

        self.init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        hidden_size = self.output_size

        self.actor = nn.Sequential(
            self.init_(nn.Linear(self.config.SRNN.human_node_rnn_size, hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            self.init_(nn.Linear(self.config.SRNN.human_node_rnn_size, hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())


        self.critic_linear = self.init_(nn.Linear(hidden_size, 1))
        robot_size = obs_space_dict['robot_node'].shape[1]

        self.robot_linear = nn.Sequential(self.init_(nn.Linear(robot_size, config.SRNN.robot_embedding_size)), nn.ReLU())
        # self.human_node_final_linear=self.init_(nn.Linear(self.output_size,2))

        if self.config.SRNN.use_self_attn:
            self.spatial_attn = SpatialEdgeSelfAttn(config)
            self.spatial_linear = nn.Sequential(self.init_(nn.Linear(self.config.SRNN.self_attn_size, config.SRNN.human_embedding_size)), nn.ReLU())
            # self.spatial_linear = nn.Sequential(self.init_(nn.Linear(self.config.SRNN.self_attn_size, 64)), nn.ReLU())
        else:
            # self.spatial_linear = nn.Sequential(self.init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], 128)), nn.ReLU(),
            #                                     self.init_(nn.Linear(128, 256)), nn.ReLU())
            # todo: 64
            self.spatial_linear = nn.Sequential(self.init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], config.SRNN.human_embedding_size)), nn.ReLU())



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

        # based on old storage.py, compatible with crowdnav1
        # hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        hidden_states_node_RNNs = reshapeT(rnn_hxs['rnn'], 1, nenv)

        masks = reshapeT(masks, seq_length, nenv)

        # based on old storage.py, compatible with crowdnav1
        # if not self.config.training.cuda:
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        # else:
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = self.robot_linear(robot_states)

        # Spatial Edges
        # self attention
        if self.config.SRNN.use_self_attn:
            spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
        else:
            spatial_attn_out = spatial_edges
        output_spatial = self.spatial_linear(spatial_attn_out)

        # robot-human attention
        if self.config.SRNN.use_hr_attn:
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
        else:
            # if we don't add robot-human attention, just take average of all human embeddings
            # output_spatial: [seq_len, nenv, human_num, feature size (256)]
            # hidden_attn_weighted: [seq_len, nenv, 1, feature size (256)]
            hidden_attn_weighted = torch.mean(output_spatial, dim=-2, keepdim=True)

        # Do a forward pass through nodeRNN
        outputs, h_nodes \
            = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)


        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        # based on old storage.py, compatible with crowdnav1
        # rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        # rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs
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
