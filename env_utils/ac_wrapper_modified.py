'''
@Author: HU Zeyun
description: 三个简单点，ABC，B点后出现，用RL做路径规划
'''
import random
import os

from tensorflow.python.ops.special_math_ops import bessel_y0

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import gymnasium as gym
import math
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict
from typing import List
from collections import defaultdict, deque, Counter

from numpy import floating
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans


class ACEnvWrapper(gym.Wrapper):
    """
    Simplified Aircraft Wrapper
    功能：
    - 仅包含三点 A/B/C
    - B 在 step=30 之后才出现
    - 每步奖励 -1
    - 距离目标点 < 50 给奖励 +5（每个点只奖励一次）
    """

    def __init__(self, env: Env, aircraft_inits, max_states: int = 3):
        super().__init__(env)

        self.initial_pos = np.array(aircraft_inits["drone_1"]["position"])
        self.speed = aircraft_inits["drone_1"]["speed"]

        # 历史轨迹（暂保留你的结构）
        self._pos_set = deque([self.initial_pos] * max_states, maxlen=max_states)

        # -----------------------
        # 定义三个检测点 A/B/C
        # -----------------------
        self.point_A =  np.array([1700, 1400, 0])
        self.point_B =  np.array([1700, 1600, 0])
        self.point_C =  np.array([1900, 1600, 0])

        self.target_points = {
            "A": self.point_A,
            "B": self.point_B,
            "C": self.point_C,
        }

        # 记录每个点是否已被覆盖
        self.covered = {k: False for k in self.target_points}

        # step counter
        self.step_count = 0

        # 保留动作映射
        speed = self.speed
        self.air_actions = {
            0: (speed, 0), 1: (speed, 1), 2: (speed, 2), 3: (speed, 3),
            4: (speed, 4), 5: (speed, 5), 6: (speed, 6), 7: (speed, 7),
            8: (0, 0), 9: (0, 2), 10:(0, 4), 11:(0, 6)
        }

        # 记录哪些 target 已经“可见”
        self.visible_targets = {
            "A": False,
            "B": False,
            "C": False,
        }

        # 记录出现后的坐标
        self.target_locations = {
            "A": (0.0, 0.0),
            "B": (0.0, 0.0),
            "C": (0.0, 0.0),
        }

        # ----- 为 A/B/C 创建独立区块 (x±70, y±70) -----
        self.zone_radius = 70

        self.zones = {}
        for key, point in self.target_points.items():
            px, py = point[:2]
            self.zones[key] = {
                "xmin": px - self.zone_radius,
                "xmax": px + self.zone_radius,
                "ymin": py - self.zone_radius,
                "ymax": py + self.zone_radius,
                "visited": False,       # 是否第一次进入过
                "stay_count": 0         # 在区块中的停留计步
            }

        self.distance_stage = {k: set() for k in ["A", "B", "C"]}


    @property
    def action_space(self):
        return gym.spaces.Discrete(12)
    
    @property
    def observation_space(self):
        spaces = {
            "ac_attr": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,)),
            # "target_rel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,)),
        }
        dict_space = gym.spaces.Dict(spaces)
        return dict_space

    # ------------------------------
    # 简化 state_wrapper
    # ------------------------------
    def state_wrapper(self, state):
        aircraft = state["aircraft"]
        drone = aircraft["drone_1"]
        pos = np.array(drone["position"])

        # 更新无人机历史位置
        self._pos_set.append(pos)
        
        target_rel = []
        for key in ["A", "B", "C"]:
            target_rel.append(self.target_locations[key])
        target_rel = np.array(target_rel).reshape(-1)

        ac_attr = np.array(self._pos_set).reshape(-1)
        ac_attr_norm = (ac_attr - ac_attr.mean()) / (ac_attr.std() + 1e-8)
        target_rel_flat = target_rel.flatten()
        target_rel_norm = (target_rel_flat - target_rel_flat.mean()) / (target_rel_flat.std() + 1e-8)

        # 构造简单观测
        obs = {
            "ac_attr": ac_attr,
            # "target_rel": target_rel_flat,
        }
        return obs, {}

    # ------------------------------
    # 奖励函数
    # ------------------------------
    def reward_wrapper(self, drone_pos):
        reward = -1   # 每步 -1

        # B 在 step < 30 时不算出现
        active_points = ["A", "C"]
        if self.step_count >= 30:
            active_points.append("B")

        for key in ["A", "B", "C"]:
            if key in active_points:
                self.visible_targets[key] = True
                rel_pos = self.target_points[key][:2] - drone_pos[:2]
                self.target_locations[key] = rel_pos
            else:
                self.target_locations[key] = np.array([0.0, 0.0])

        reward_bins = [
            (100, 50, 1.0),
            (50, 40, 2.0),
            (40, 30, 3.0),
            (30, 20, 4.0),
            (20, 10, 8.0),
        ]
        # 遍历检测点
        final_reward = 12.0
        # 遍历所有激活点
        for key in active_points:
            point = self.target_points[key]
            dist = np.linalg.norm(drone_pos[:2] - point[:2])

            # 最终成功奖励（只给一次）
            if dist < 10:
                if "final" not in self.distance_stage[key]:
                    reward += final_reward
                    self.distance_stage[key].add("final")
                    self.covered[key] = True
                continue

            # 分段奖励
            for (high, low, r) in reward_bins:
                stage_name = f"{high}-{low}"
                if dist < high and dist >= low:
                    if stage_name not in self.distance_stage[key]:
                        reward += r
                        self.distance_stage[key].add(stage_name)
                        # print('add reward:', r)

        # # ------------------------------------------
        # # ② 新增：区块奖励 + 停留惩罚 (核心修改部分)
        # # ------------------------------------------
        # x, y = drone_pos[0], drone_pos[1]
        # for key in active_points:
        #     zone = self.zones[key]

        #     inside = (zone["xmin"] <= x <= zone["xmax"]) and \
        #              (zone["ymin"] <= y <= zone["ymax"])

        #     if inside:
        #         # 第一次进入：奖励
        #         if not zone["visited"]:
        #             zone["visited"] = True
        #             reward += 1.0      # ← 区块进入奖励（你可自己调）

        #         # 在区块内停留，计步
        #         zone["stay_count"] += 1

        #         # 停留惩罚
        #         if zone["stay_count"] >= 80:
        #             reward -= 1.0      # ← 停留惩罚
        #             print("give a penalty for staying too long")
        #     else:
        #         # 离开区块，重置计步
        #         zone["stay_count"] = 0

        done = all(self.covered.values())
        return reward, done

    # ------------------------------
    # reset
    # ------------------------------
    def reset(self, seed=1):
        env_state = self.env.reset()
        self.step_count = 0
        self.covered = {k: False for k in self.target_points}
                # 记录哪些 target 已经“可见”
        self.visible_targets = {
            "A": False,
            "B": False,
            "C": False,
        }

        # 记录出现后的坐标
        self.target_locations = {
            "A": (0.0, 0.0),
            "B": (0.0, 0.0),
            "C": (0.0, 0.0),
        }

        for key, point in self.target_points.items():
            px, py = point[:2]
            self.zones[key] = {
                "xmin": px - self.zone_radius,
                "xmax": px + self.zone_radius,
                "ymin": py - self.zone_radius,
                "ymax": py + self.zone_radius,
                "visited": False,       # 是否第一次进入过
                "stay_count": 0         # 在区块中的停留计步
            }
        self.distance_stage = {k: set() for k in ["A", "B", "C"]}
        obs, _ = self.state_wrapper(env_state)
        return obs, {"step_time": 0}

    # ------------------------------
    # step
    # ------------------------------
    def step(self, action):

        # 转换 action 格式（保留你原先的写法）
        if isinstance(action, np.int64):
            new_actions = {"drone_1": self.air_actions[action]}
        else:
            new_actions = {k: self.air_actions[v] for k, v in action.items()}

        states, _, truncated, dones, infos = super().step(new_actions)

        self.step_count += 1

        # 获取无人机位置
        pos = np.array(states["aircraft"]["drone_1"]["position"])

        # state wrapper
        obs, _ = self.state_wrapper(states)
        reward, done_flag = self.reward_wrapper(pos)
        infos["pos_drone"] = pos

        x, y = pos[0], pos[1]
        # print("x",x)
        # print("y", y)
        if not (1500 <= x <= 2000 and 1000 <= y <= 1700):
            # done_flag = True
            # infos["out_of_boundary"] = True
            reward -= 2.0
     

        # print(reward)

        return obs, reward, truncated, done_flag, infos
