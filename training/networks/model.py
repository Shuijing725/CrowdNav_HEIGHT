
from training.networks.distributions import Bernoulli, Categorical, DiagGaussian
from .raw_sensor_models import *
from .selfAttn_srnn_merge_lidar import selfAttn_merge_SRNN_lidar
from .dsrnn_obs_vertex import DSRNN_obs_vertex
from .homo_transformer_obs import Homo_Transformer_Obs

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == 'dsrnn_obs_vertex':
            base = DSRNN_obs_vertex
        elif base == 'selfAttn_merge_srnn':
            base = selfAttn_merge_SRNN
        elif base == 'selfAttn_merge_srnn_lidar':
            base = selfAttn_merge_SRNN_lidar
        elif base == 'homo_transformer_obs':
            base = Homo_Transformer_Obs
        elif base == 'lidar_gru':
            base = LIDAR_CNN_GRU_RL
        else:
            raise NotImplementedError

        self.base = base(obs_shape, base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs, attn_weights_hh, attn_weights_rh = self.base(inputs, rnn_hxs, masks, infer=True)
        # render = True: actor_features: [256, ], dist: [1, 2, 1, 2]
        # render = False: actor_features: [16, 256], dist: [16, 2]
        dist = self.dist(actor_features)
        # print(f"Check attn weights HH: {attn_weights_hh}")
        # print(f"Check attn weights RH: {attn_weights_rh}")


        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs, attn_weights_hh, attn_weights_rh

    def get_value(self, inputs, rnn_hxs, masks):

        value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



