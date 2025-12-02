import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from train_utils.traffic_transformer import TransformerExtractor

class CustomModelWithTrans(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int ):
        super().__init__(observation_space, features_dim)

        ac_shape = observation_space["ac_attr"].shape[0]
        rv_shape = observation_space["relative_vecs"].shape
        cc_shape = observation_space["cover_counts"].shape[0]
        bs_shape = observation_space["break_spot"].shape[0]

        self.hidden_dim = 32

        self.attr_net = nn.Sequential(
            nn.Linear(ac_shape, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        self.brk_net = nn.Sequential(
            nn.Linear(bs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
            nn.ReLU()
        )
        self.cc_net = nn.Sequential(
            nn.Linear(cc_shape, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
        )

        self.trans_extractor = TransformerExtractor(
            input_dim=rv_shape[1], d_model=self.hidden_dim, dim_feedforward=128, nhead=4, num_layers=2, seq_len=rv_shape[0]
        )

        self.output = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        ac_attr = obs['ac_attr']
        rel_vecs = obs['relative_vecs']
        brk_spot = obs['break_spot']
        cov_cnt = obs['cover_counts']

        z_attr = self.attr_net(ac_attr)
        z_brk = self.brk_net(brk_spot)
        z_rel  = self.trans_extractor(rel_vecs)

        z_cc = self.cc_net(cov_cnt)
        all_feature_output = self.output(torch.cat([z_attr, z_brk, z_rel, z_cc], dim=1))
        return all_feature_output