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

# Define a Residual Block (Bottleneck)
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

# Define the ResNet-based CNN for occupancy map encoding
class OMEncoder(nn.Module):
    def __init__(self, config, obs_space_dict):
        super(OMEncoder, self).__init__()
        self.in_channels = 64  # First layer output channels
        self.om_embed_size = obs_space_dict['om'].shape[1]

        # Initial Conv Layer for 128x128 input
        self.conv1 = nn.Conv2d(obs_space_dict['om'].shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet-style block (Reduced to 1 bottleneck block)
        self.layer1 = self._make_layer(Bottleneck, 64, blocks=1, stride=1)

        # Feature extraction
        self.conv_final = nn.Conv2d(128, 128, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, self.om_embed_size)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.conv_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class OM_CNN_GRU_IL(nn.Module):
    '''
    Policy network for an environment with 'robot_node' and 'om' as inputs.
    A CNN processes the occupancy map (OM), and an MLP processes the robot state.
    The two features are concatenated and passed through a GRU.
    '''

    def __init__(self, obs_space_dict, config):
        super(OM_CNN_GRU_IL, self).__init__()

        self.config = config
        self.is_recurrent = True

        self.seq_length = config.ppo.num_steps
        self.nenv = config.training.num_processes
        self.nminibatch = config.ppo.num_mini_batch

        self.output_size = config.SRNN.human_node_output_size
        robot_embed_size = config.SRNN.robot_embedding_size
        self.om_embed_size = obs_space_dict['om'].shape[1]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # CNN for processing occupancy map
        # self.om_embed = nn.Sequential(
        #     init_(nn.Conv2d(1, 16, kernel_size=5, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(16, 32, kernel_size=3, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, kernel_size=3, stride=2)), nn.ReLU(),
        #     Flatten(),
        #     init_(nn.Linear(2304, self.om_embed_size)), nn.ReLU()
        # )
        self.om_embed = OMEncoder(config=config, obs_space_dict=obs_space_dict)

        # MLP for robot state
        self.robot_embed = nn.Sequential(
            init_(nn.Linear(obs_space_dict['robot_node'].shape[1] + obs_space_dict['temporal_edges'].shape[1], robot_embed_size)), nn.ReLU()
        )

        # Concatenation layer
        self.concat_layer = init_(
            nn.Linear(self.om_embed_size + robot_embed_size, config.SRNN.human_node_embedding_size * 2)
        )

        # GRU for temporal correlation
        self.gru = RNNBase(config, edge=False)

        # Actor network
        self.actor = nn.Sequential(
            init_(nn.Linear(config.SRNN.human_node_rnn_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh()
        )

    def process_inputs(self, inputs, rnn_hxs, masks, infer=False):
        seq_length = 1 if infer else self.seq_length
        nenv = 1 if infer else self.nenv

        robot_in = reshapeT(inputs['robot_node'], seq_length, nenv)
        om_in = inputs['om'].reshape(seq_length * nenv, 1, 64, 64)  # Grayscale image

        return robot_in, om_in, rnn_hxs, masks, seq_length, nenv

    def forward_actor(self, robot_in, om_in, rnn_hxs, masks, seq_length, nenv):
        robot_features = self.robot_embed(robot_in)
        om_features = self.om_embed(om_in)
        om_features = om_features.view(seq_length, nenv, 1, self.om_embed_size)

        merged_features = torch.cat((robot_features, om_features), dim=-1)
        merged_features = self.concat_layer(merged_features)

        outputs, h = self.gru._forward_gru(merged_features, rnn_hxs['rnn'], masks)
        rnn_hxs['rnn'] = h
        x = outputs[:, :, 0, :]
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        return hidden_actor.view(-1, self.output_size), x, rnn_hxs

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        robot_in, om_in, rnn_hxs, masks, seq_length, nenv = self.process_inputs(inputs, rnn_hxs, masks, infer)
        hidden_actor, _, rnn_hxs = self.forward_actor(robot_in, om_in, rnn_hxs, masks, seq_length, nenv)
        return hidden_actor, rnn_hxs


class OM_CNN_GRU_RL(OM_CNN_GRU_IL):
    '''
    Reinforcement learning version of the policy network with a critic head.
    '''

    def __init__(self, obs_space_dict, config):
        super().__init__(obs_space_dict, config)
        self.in_channel_num = obs_space_dict['om'].shape[0]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.critic = nn.Sequential(
            init_(nn.Linear(self.config.SRNN.human_node_rnn_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh()
        )

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

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        robot_in = torch.cat((robot_node, temporal_edges), dim=-1)
        om_in = inputs['om'].reshape(seq_length * nenv, self.in_channel_num, self.om_embed_size, self.om_embed_size)

        hidden_states_node_RNNs = reshapeT(rnn_hxs['rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        return robot_in, om_in, hidden_states_node_RNNs, masks, seq_length, nenv

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        robot_in, om_in, rnn_hxs, masks, seq_length, nenv = self.process_inputs(inputs, rnn_hxs, masks, infer)
        rnn_hidden_state = {'rnn': rnn_hxs}

        hidden_actor, x, rnn_hxs = self.forward_actor(robot_in, om_in, rnn_hidden_state, masks, seq_length, nenv)
        hidden_critic = self.critic(x)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor, rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor, rnn_hxs