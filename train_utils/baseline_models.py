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
        # grid_dim = observation_space["grid_counter"].shape[0]          # 100

        hidden = 32

        # ============= 编码 ac_attr =============
        self.ac_step_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
        )

        # 3 个位置 token + 2 个 motion token = 5 个 token，共 5*hidden 维
        self.ac_traj_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.ac_motion_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
        )

        self.motion_upsample = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 3),
            nn.ReLU(),
        )

        self.ac_fusion_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

        # 使用 LSTM 在 3 个时间步上建模无人机轨迹特征
        self.ac_lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)

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

        self.per_passenger_mlp = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
        )

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=2, batch_first=True)
        self.passenger_ln = nn.LayerNorm(hidden)

        self.passenger_proj = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.ReLU(),
        )

        # 显式朝向编码：基于最近一步的单位速度向量
        self.heading_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
        )

        # # ============= 编码 grid_counts =============
        # self.encoder_grid = nn.Sequential(
        #     nn.Linear(grid_dim, hidden),
        #     nn.ReLU(),
        # )

        # 最终 concat：4个 * 32 维
        concat_dim = hidden * 2

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
        ac_seq = ac.view(ac.shape[0], 3, 2)                     # (batch,3,2)  3 个位置

        target_rel = observations["target_rel"]                 # (batch,5,3)
        vis = observations["vis_and_cover"].float()            # (batch,5,2)
        per_passenger = torch.cat([target_rel, vis], dim=-1)    # (batch,5,5)

        # --- 编码 ---
        # 位置序列
        pos_tokens = ac_seq                                     # (batch,3,2)
        # motion token：相邻位置差分，形成 2 个速度向量（t-1, t）
        vel_tokens = pos_tokens[:, 1:, :] - pos_tokens[:, :-1, :]  # (batch,2,2)

        # # 为 3 个时间步构造速度序列：第一个时间步速度设为 0，后两个用实际速度
        # batch_size = ac.shape[0]
        # zeros = torch.zeros(batch_size, 1, 2, device=ac.device, dtype=ac.dtype)
        # vel_full = torch.cat([zeros, vel_tokens], dim=1)        # (batch,3,2)

        # # 每个时间步的特征向量：[x, y, vx, vy]
        # step_tokens = torch.cat([pos_tokens, vel_full], dim=-1)  # (batch,3,4)

        # per-step 编码（纯 MLP）
        pos_emb = self.ac_step_mlp(pos_tokens)                  # (batch,3,32)
        motion_emb = self.ac_motion_mlp(vel_tokens)             # (batch,2,32)

        # 将 2 个 motion token 通过可学习的上采样映射到 3 个时间步，方便与位置对齐
        B = motion_emb.shape[0]
        motion_flat = motion_emb.view(B, -1)                    # (batch, 2*32)
        motion_up_flat = self.motion_upsample(motion_flat)      # (batch, 3*32)
        motion_up = motion_up_flat.view(B, 3, -1)               # (batch,3,32)

        fused_tokens = torch.cat([pos_emb, motion_up], dim=-1)  # (batch,3,64)
        fused_emb = self.ac_fusion_mlp(fused_tokens)            # (batch,3,32)

        batch_size = fused_emb.shape[0]
        fused_flat = fused_emb.view(batch_size, -1)             # (batch, 3*32)
        ac_f = self.ac_traj_mlp(fused_flat)                     # (batch,32)

        # # 计算最近一步的单位方向向量，并通过 heading_mlp 编码后融入 ac_f
        # last_vel = vel_tokens[:, -1, :]                         # (batch,2)
        # speed = torch.norm(last_vel, dim=-1, keepdim=True).clamp(min=1e-6)
        # heading = last_vel / speed                              # (batch,2)
        # heading_f = self.heading_mlp(heading)                   # (batch,32)
        # ac_f = ac_f + heading_f

        passenger_embed = self.per_passenger_mlp(per_passenger) # (batch,5,32)
        attn_out, _ = self.self_attn(passenger_embed, passenger_embed, passenger_embed)
        passenger_attn = self.passenger_ln(passenger_embed + attn_out)
        passenger_flat = passenger_attn.view(passenger_attn.shape[0], -1)  # (batch,5*32)
        passenger_f = self.passenger_proj(passenger_flat)       # (batch,32)

        # --- 融合（MLP） ---
        cat = torch.cat([ac_f, passenger_f], dim=1)             # (batch,64)

        return self.output(cat)
