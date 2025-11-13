import os
import numpy as np
import math
from typing import List

from env_utils.ac_env import ACEnvironment
from env_utils.ac_wrapper_modified import ACEnvWrapper
import matplotlib.pyplot as plt

# -------------------- 自定义覆盖半径方法 --------------------
def custom_update_cover_radius(position: List[float], communication_range: float) -> float:
    height = position[2]
    cover_radius = height / np.tan(math.radians(75/2))
    return cover_radius

# -------------------- 配置 --------------------
sumo_cfg = "./sumo_envs/TEST_NET_2.0/zy_road2.sumocfg"
base_dir = os.path.dirname(os.path.abspath(__file__))
drone_img_path = os.path.join(base_dir, "asset", "drone.png")

aircraft_inits = {
    'drone_1': {
        "aircraft_type": "drone",
        "action_type": "horizontal_movement",
        "position": (-170, 100, 50),
        "speed": 10,
        "heading": (1, 1, 0),
        "communication_range": 50,
        "if_sumo_visualization": True,
        "img_file": drone_img_path,
        "custom_update_cover_radius": custom_update_cover_radius
    },
    'drone_2': {
        "aircraft_type": "drone",
        "action_type": "horizontal_movement",
        "position": (100, -100, 60),  # 和 drone_1 不同的位置
        "speed": 10,
        "heading": (-1, 1, 0),        # 初始航向不同
        "communication_range": 50,
        "if_sumo_visualization": True,
        "img_file": drone_img_path,
        "custom_update_cover_radius": custom_update_cover_radius
    },
}

# -------------------- 初始化环境 --------------------
env = ACEnvironment(
    sumo_cfg=sumo_cfg,
    num_seconds=1000,
    aircraft_inits=aircraft_inits,
    use_gui=True
)
env = ACEnvWrapper(env, aircraft_inits)

# -------------------- 简单控制循环 --------------------
n_steps = 100
obs, _ = env.reset()
print('obs is given by')
print(obs)
rewards = []

for step in range(n_steps):
    # 每一步动作：drone_1 向右 (0)，drone_2 向上 (2)
    action = {
        'drone_1': np.int64(0),
        'drone_2': np.int64(2)
    }
    obs, reward, truncated, done, info = env.step(action)
    rewards.append(reward)
    print(reward)

    # # 打印两个无人机的位置
    # for drone_id in env.latest_ac_pos.keys():
    #     print(f"Step {step+1}: {drone_id} pos = {env.latest_ac_pos[drone_id]}")
    # print(f"Step {step+1}: Total Reward = {reward}")

    if done or truncated:
        print("Episode ended early due to boundary.")
        break

# -------------------- 绘制奖励 --------------------
plt.figure(figsize=(8,4))
plt.plot(range(1, len(rewards)+1), rewards, marker='o')
plt.title("Two-Drone Simulation")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

# -------------------- 关闭环境 --------------------
env.close()
del env
