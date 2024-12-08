import torchvision.models as models
from training.networks.network_utils import Flatten, RNNBase
from .selfAttn_srnn_merge import *



class LIDAR_CNN_GRU_IL(nn.Module):
    '''
    for crowd_sim_pc.py
    an MLP processes the lidar scans, and another MLP processes the robot low-level states,
    then the two features are concatenated together and feed through a GRU
    '''
    def __init__(self, obs_space_dict, config):
        '''
        Initializer function
        params:
        config : Training arguments
        '''
        super(LIDAR_CNN_GRU_IL, self).__init__()

        self.config = config
        self.is_recurrent = True

        if self.config.env.env_name in ['CrowdSimPC-v0', 'CrowdSim3D-v0', 'CrowdSim3DSeg-v0', 'CrowdSim3DTbObs-v0', 'CrowdSim3DTbObsHieTrain-v0']:
            self.lidar_input_size = int(360. / self.config.lidar.angular_res)
        else:
            raise ValueError("Unknown environment name")

        if config.il.train_il:
            self.seq_length = config.il.expert_traj_len
            self.nenv = config.il.batch_size
            self.nminibatch = 1
        else:
            self.seq_length = config.ppo.num_steps
            self.nenv = config.training.num_processes
            self.nminibatch = config.ppo.num_mini_batch

        # workaround to prevent errors
        self.human_num = 1

        self.output_size = config.SRNN.human_node_output_size
        self.lidar_embed_size = 128
        if config.env.env_name == 'CrowdSim3DSeg-v0':
            # old version
            # self.lidar_channel_num = 2 # 4

            # new version
            self.lidar_channel_num = self.config.sim.human_num + self.config.sim.human_num_range + 1

            self.lidar_embed_conv_out_size = 608

        else:
            self.lidar_channel_num = 1
            self.lidar_embed_conv_out_size = 256 # 256 if angular resolution of lidar is 2, 608 if angular resolution is 1
        robot_embed_size = config.SRNN.robot_embedding_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # Linear layers to embed inputs
        # 1d conv
        self.lidar_embed = nn.Sequential(init_(nn.Conv1d(self.lidar_channel_num, 16, 10, stride=2)), nn.ReLU(), # (1, 360) -> (32, 176)
                                         init_(nn.Conv1d(16, 32, 5, stride=2)), nn.ReLU(), # (32, 176) -> (32, 86)
                                         init_(nn.Conv1d(32, 32, 5, stride=2)), nn.ReLU(), # (32, 86) -> (32, 41)
                                         init_(nn.Conv1d(32, 32, 5, stride=2)), nn.ReLU(), # (32, 41) -> (32, 19)
                                         Flatten(),
                                         init_(nn.Linear(self.lidar_embed_conv_out_size, self.lidar_embed_size)), nn.ReLU(),
                                         )
        # print number of trainable parameters
        # model_parameters = filter(lambda p: p.requires_grad, self.lidar_embed.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('total # params:', params)
        self.robot_embed = nn.Sequential(init_(nn.Linear(obs_space_dict['robot_node'].shape[1], robot_embed_size)), nn.ReLU())

        # Output linear layer
        self.concat_layer = init_(nn.Linear(self.lidar_embed_size + robot_embed_size, config.SRNN.human_node_embedding_size*2))

        # gru to add temporal correlation
        self.gru = RNNBase(config, edge=False)

        self.actor = nn.Sequential(
            init_(nn.Linear(self.config.SRNN.human_node_rnn_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())


    def process_inputs(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            seq_length = 1
            nenv = 1
            robot_in = reshapeT(inputs['robot_node'], seq_length, nenv)  # [seq len, batch size, 1, 7]
        else:
            seq_length = self.seq_length
            nenv = self.nenv
            robot_in = inputs['robot_node']  # [seq len, batch size, 1, 7]
        # [seq len, batch size, 2, pc num] -> [seq_len*batch_size,2, pc num]
        lidar_in = inputs['point_clouds'].reshape(seq_length * nenv, self.lidar_channel_num, self.lidar_input_size)
        # masks: [seq len, batch size, 1]

        return robot_in, lidar_in, rnn_hxs, masks, seq_length, nenv


    def forward_actor(self, robot_in, lidar_in, rnn_hxs, masks, seq_length, nenv):

        # use mlps to extract input features
        robot_features = self.robot_embed(robot_in)
        # convert inputs from dim=4 to dim=3 for conv layer
        # lidar_in = lidar_in.view(seq_length*nenv, self.lidar_channel_num, self.lidar_input_size)
        lidar_features = self.lidar_embed(lidar_in)
        # reshape it back to dim=4
        lidar_features = lidar_features.view(seq_length, nenv, 1, self.lidar_embed_size)
        merged_features = torch.cat((robot_features, lidar_features), dim=-1)
        merged_features = self.concat_layer(merged_features)

        # forward gru
        outputs, h = self.gru._forward_gru(merged_features, rnn_hxs['rnn'], masks)

        rnn_hxs['rnn'] = h

        # x is the output of the robot node and will be sent to actor and critic
        x = outputs[:, :, 0, :]

        # feed the new gru hidden states to actor
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        # hidden_actor: [seq_len, nbatch, output_size] -> [seq_len*nbatch, output_size]
        return hidden_actor.view(-1, self.output_size), x, rnn_hxs

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        # reshape inputs
        robot_in, lidar_in, rnn_hxs, masks, seq_length, nenv = self.process_inputs(inputs, rnn_hxs, masks,
                                                                                           infer)
        # forward policy network
        hidden_actor, _, rnn_hxs = self.forward_actor(robot_in, lidar_in, rnn_hxs, masks, seq_length, nenv)
        return hidden_actor, rnn_hxs


class LIDAR_CNN_GRU_RL(LIDAR_CNN_GRU_IL):
    '''
    for crowd_sim_pc.py
    an MLP processes the lidar scans, and another MLP processes the robot low-level states,
    then the two features are concatenated together and feed through a GRU
    '''
    def __init__(self, obs_space_dict, config):
        '''
        Initializer function
        params:
        config : Training arguments
        '''
        super().__init__(obs_space_dict, config)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.critic = nn.Sequential(
            init_(nn.Linear(self.config.SRNN.human_node_rnn_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))


    def process_inputs(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        # [seq_len, nenv, agent_num, feature_size]
        robot_in = reshapeT(inputs['robot_node'], seq_length, nenv)
        # lidar_in = reshapeT(inputs['point_clouds'], seq_length, nenv)
        lidar_in = inputs['point_clouds']

        hidden_states_node_RNNs = reshapeT(rnn_hxs['rnn'], 1, nenv)

        masks = reshapeT(masks, seq_length, nenv)

        return robot_in, lidar_in, hidden_states_node_RNNs, masks, seq_length, nenv

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        # reshape inputs
        robot_in, lidar_in, rnn_hxs, masks, seq_length, nenv = self.process_inputs(inputs, rnn_hxs, masks,
                                                                                           infer)
        rnn_hidden_state = {}
        rnn_hidden_state['rnn'] = rnn_hxs
        # forward actor network
        # hidden_actor: [seq_len*nenv, ?], x: [seq_len, nenv, ?]
        hidden_actor, x, rnn_hxs = self.forward_actor(robot_in, lidar_in, rnn_hidden_state, masks, seq_length, nenv)
        # forward critic network
        # hiden_critic: [seq_len, nenv, ?]
        hidden_critic = self.critic(x)

        if infer:
            # critic output: [1, nenv, ?] -> [nenv, 1]
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor, rnn_hxs
        else: # critic: [seq_len*nenv, ?], actor:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor, rnn_hxs

