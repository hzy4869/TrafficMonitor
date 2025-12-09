import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch


class CustomModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int):
        super().__init__(observation_space, features_dim)

        # ============= 取出每个 observation 维度 =============
        ac_attr_dim = observation_space["ac_attr"].shape[0]           # 6
        target_rel_dim = observation_space["target_rel"].shape[0] * observation_space["target_rel"].shape[1]  # 5*2
        vis_cover_dim = observation_space["vis_and_cover"].shape[0] * observation_space["vis_and_cover"].shape[1]  # 5*2
        grid_dim = observation_space["grid_counter"].shape[0]          # 100

        hidden = 32

        # ============= 编码 ac_attr =============
        self.encoder_ac = nn.Sequential(
            nn.Linear(ac_attr_dim, hidden),
            nn.ReLU(),
        )

        # ============= 编码 target_rel（展平） =============
        self.encoder_target = nn.Sequential(
            nn.Linear(target_rel_dim, hidden),
            nn.ReLU(),
        )

        # ============= 编码 vis_and_cover（展平） =============
        self.encoder_vis = nn.Sequential(
            nn.Linear(vis_cover_dim, hidden),
            nn.ReLU(),
        )

        # ============= 编码 grid_counts =============
        self.encoder_grid = nn.Sequential(
            nn.Linear(grid_dim, hidden),
            nn.ReLU(),
        )

        # 最终 concat：4个 * 32 维
        concat_dim = hidden * 4

        self.output = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )

    def forward(self, observations):
        # --- 取数据 ---
        ac = observations["ac_attr"]                           # (batch,6)
        target = observations["target_rel"].reshape(ac.shape[0], -1)     # (batch,10)
        vis = observations["vis_and_cover"].reshape(ac.shape[0], -1)     # (batch,10)
        grid = observations["grid_counter"]                     # (batch,100)

        # --- 编码 ---
        ac_f = self.encoder_ac(ac)
        target_f = self.encoder_target(target)
        vis_f = self.encoder_vis(vis)
        grid_f = self.encoder_grid(grid)

        # --- 合并 ---
        cat = torch.cat([ac_f, target_f, vis_f, grid_f], dim=1)

        return self.output(cat)
