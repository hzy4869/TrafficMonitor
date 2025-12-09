'''
@Author: HU Zeyun
description: to be determined.
'''
import os
from tensorflow.python.ops.special_math_ops import bessel_y0

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from collections import deque


class ACEnvWrapper(gym.Wrapper):
    """

    """

    def __init__(self, env: Env, aircraft_inits, max_states: int = 3):
        super().__init__(env)

        self.initial_pos = np.array(aircraft_inits["drone_1"]["position"])
        self.speed = aircraft_inits["drone_1"]["speed"]
        self._pos_set = deque([self.initial_pos[:2]] * max_states, maxlen=max_states)
        speed = self.speed
        self.air_actions = {
            0: (speed, 0), 1: (speed, 1), 2: (speed, 2), 3: (speed, 3),
            4: (speed, 4), 5: (speed, 5), 6: (speed, 6), 7: (speed, 7),
        }
        
        # stable-- not need to update in reset
        # (x, y, step): x, y means the init pos. step means that point occur on that step.
        self.points = {
            "A": np.array([1700, 1400, 0]),
            "B": np.array([1800, 1500, 30]),
            "C": np.array([1900, 1400, 0]),
            "D": np.array([1800, 1300, 30]),
            "E": np.array([2000, 1300, 0])
        }
        self.boundary = {
            "xmin": 1600,
            "xmax": 2100,
            "ymin": 1200,
            "ymax": 1600,
        }
        self.grid_size = 100
        self.grid_stay_counter = {}

        # sequential: A, B, C, D, E
        self.visible_set = [False] * len(self.points)
        self.reward_bins = [
            (100, 50, 1.0),
            (50, 40, 2.0),
            (40, 30, 3.0),
            (30, 20, 4.0),
            (20, 10, 8.0),
            (10, 0, 12.0)
        ]

        self.grid_size = 100
        self.grid_x_num = (self.boundary["xmax"] - self.boundary["xmin"]) // self.grid_size  # 5
        self.grid_y_num = (self.boundary["ymax"] - self.boundary["ymin"]) // self.grid_size  # 4
        self.grid_counter = np.zeros((self.grid_y_num, self.grid_x_num), dtype=np.int32)


        self.covered = {key: [False]*len(self.reward_bins) for key in self.points.keys()}
        self.global_step_count = 0

        self.log_file = "log_record.txt"


    @property
    def action_space(self):
        return gym.spaces.Discrete(8)
    
    @property
    def observation_space(self):
        spaces = {
            "ac_attr": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "target_rel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, 1), dtype=np.float32),
            "vis_and_cover": gym.spaces.Box(low=0, high=1, shape=(5, 2), dtype=np.int32),
            "grid_counter": gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.grid_y_num, self.grid_x_num),
                dtype=np.int32
            ),
        }
        return gym.spaces.Dict(spaces)

    
    def log_if_enter_range(self, key, point, pos, threshold=100):
        """
        如果无人机进入 key 点（例如 B 或 D）100 米范围，写入日志。
        """
        dist = np.linalg.norm(pos[:2] - point[:2])
        if dist <= threshold:
            with open(self.log_file, "a") as f:
                f.write(f"Step {self.global_step_count}: Drone within {threshold}m of point {key}, "
                        f"DronePos=({pos[0]:.2f},{pos[1]:.2f}), "
                        f"Target=({point[0]},{point[1]})\n")
                

    def update_grid_counter(self, pos):
        x, y = pos[:2]

        # 判断是否在边界内
        if not (self.boundary["xmin"] <= x < self.boundary["xmax"] and
                self.boundary["ymin"] <= y < self.boundary["ymax"]):
            return  # 不在区域内，不计数

        # 映射成 grid index
        gx = int((x - self.boundary["xmin"]) // self.grid_size)   # 0~4
        gy = int((y - self.boundary["ymin"]) // self.grid_size)   # 0~3

        # 更新计数
        self.grid_counter[gy, gx] += 1



    def update_visible_set(self):
        """
        根据 self.global_step_count 更新 self.visible_set。
        如果当前步数 >= 对应点的 step，则该点可见。
        """
        self.global_step_count += 1
        self.visible_set = [
            self.global_step_count >= point[2]  # 第三个元素是 step
            for point in self.points.values()
        ]


    # ------------------------------
    # state_wrapper
    # ------------------------------
    def state_wrapper(self, state):
        aircraft = state["aircraft"]
        drone = aircraft["drone_1"]
        pos = np.array(drone["position"])

        # 更新无人机历史位置
        self._pos_set.append(pos[:2])
        ac_attr = np.array(self._pos_set).reshape(-1)

        ## 坐标/距离
        target_rel = np.array([
            # point[:2] - pos[:2]       # (dx, dy)
            [np.linalg.norm(point[:2] - pos[:2])]
            for key, point in self.points.items()
        ])

        vis_and_cover = np.array([
            [
                int(vis),                              # 可见性：1/0
                int(self.covered[key][-1])             # 是否已拿完奖励：1/0
            ]
            for (key, point), vis in zip(self.points.items(), self.visible_set)
        ])

        self.update_grid_counter(pos)

        # 构造简单观测
        obs = {
            "ac_attr": ac_attr,
            "target_rel": target_rel,
            "vis_and_cover": vis_and_cover,
            "grid_counter": self.grid_counter.copy(),
        }
        # print(obs)
        return obs, {}

    # ------------------------------
    # 奖励函数
    # ------------------------------
    def reward_wrapper(self, drone_pos):
        reward = 0
        done = False

        # boundary penalty
        x, y = drone_pos[:2]
        if (x < self.boundary["xmin"] or x > self.boundary["xmax"] or
            y < self.boundary["ymin"] or y > self.boundary["ymax"]):
            reward -= 2

        # 每步默认 -1
        # 每吃掉一个点，step成本减少
        num_covered = sum(self.covered[key][-1] for key in self.points.keys())
        step_cost = -1 + 0.2 * num_covered
        reward += step_cost

        # 遍历每个点
        for key, point in self.points.items():
            dist = np.linalg.norm(drone_pos[:2] - point[:2])

            # 只考虑可见点
            idx = list(self.points.keys()).index(key)
            if not self.visible_set[idx]:
                continue

            # 遍历各阶段奖励
            for i, (upper, lower, r) in enumerate(self.reward_bins):
                if lower < dist <= upper and not self.covered[key][i]:
                    reward += r
                    self.covered[key][i] = True  # 该阶段奖励只给一次
                    break  # 一个阶段匹配后就退出


        # 区块停留/探索奖励
        gx = int((x - self.boundary["xmin"]) // self.grid_size)
        gy = int((y - self.boundary["ymin"]) // self.grid_size)
        if 0 <= gx < self.grid_x_num and 0 <= gy < self.grid_y_num:
            count = self.grid_counter[gy, gx]
            if count == 0:
                reward += 0.5
            elif count > 100:
                reward -= 0.5


        # done 判断：所有点全部覆盖
        if all(all(v) for v in self.covered.values()):
            done = True

        return reward, done


    # ------------------------------
    # reset
    # ------------------------------
    def reset(self, seed=1):
        env_state = self.env.reset()
        obs, _ = self.state_wrapper(env_state)

        self.visible_set = [False] * len(self.points)
        self.covered = {key: [False]*len(self.reward_bins) for key in self.points.keys()}
        self.grid_counter = np.zeros((self.grid_y_num, self.grid_x_num), dtype=np.int32)
        self.grid_stay_counter = {}

        return obs, {"step_time": 0}

    # ------------------------------
    # step
    # ------------------------------
    def step(self, action):
        self.update_visible_set()

        if isinstance(action, np.int64):
            new_actions = {"drone_1": self.air_actions[action]}
        else:
            new_actions = {k: self.air_actions[v] for k, v in action.items()}

        states, _, truncated, dones, infos = super().step(new_actions)


        # 获取无人机位置
        pos = np.array(states["aircraft"]["drone_1"]["position"])

        # for key in ["B", "C"]:
        #     self.log_if_enter_range(key, self.points[key], pos)

        # state wrapper
        obs, _ = self.state_wrapper(states)
        reward, done_flag = self.reward_wrapper(pos)
        infos["pos_drone"] = pos

        x, y = pos[0], pos[1]

        return obs, reward, truncated, done_flag, infos
