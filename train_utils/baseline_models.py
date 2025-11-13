'''
@Author: Ricca
@Date: 2024-07-16
@Description: Custom Model
@LastEditTime: 2025-07-15
@LastEditors: Jiachen
'''
import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

class CustomModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int ):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        ac_attr_dim = observation_space["ac_attr"].shape[0]
        veh_pos_dim = observation_space["relative_vecs"].shape[0]
        # bound_dim = observation_space["bound_dist"].shape[0]
        break_spot_dim = observation_space["break_spot"].shape[0]

        self.hidden_dim = 32
        self.linear_encoder_ac = nn.Sequential(
            nn.Linear(ac_attr_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_veh_pos = nn.Sequential(
            nn.Linear(veh_pos_dim, self.hidden_dim),
            nn.ReLU(),
        )
        # self.linear_encoder_bound = nn.Sequential(
        #     nn.Linear(bound_dim, self.hidden_dim),
        #     nn.ReLU(),
        # )
        self.linear_encoder_veh_info = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_spot_info = nn.Sequential(
            nn.Linear(break_spot_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_has_veh = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(32+32+32+32+32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations):
        ac_attr = observations["ac_attr"]
        veh_pos = observations["relative_vecs"]#.reshape(-1)
        veh_covered = observations["cover_counts"]
        #bound = observations["bound_dist"]
        break_spot = observations["break_spot"]
        has_veh = observations["no_vehicles"]

        # for k, v in observations.items():
        #     print(k,"shape",v.shape)

        ac_feat = self.linear_encoder_ac(ac_attr)
        veh_feat = self.linear_encoder_veh_pos(veh_pos)
        #bound_feat = self.linear_encoder_bound(bound)
        covered_feat = self.linear_encoder_veh_info(veh_covered)
        break_feat = self.linear_encoder_spot_info(break_spot)
        vehicle_feat = self.linear_encoder_has_veh(has_veh)

        # action_feat,
        # all_feature_output = self.output(
        # torch.cat([ac_feat, veh_feat,break_feat], dim=1))
        all_feature_output = self.output(torch.cat([ac_feat, veh_feat, covered_feat, break_feat, vehicle_feat], dim=1))
        return all_feature_output