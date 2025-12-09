import os
import numpy as np
import math
from typing import List

from env_utils.ac_env import ACEnvironment
from env_utils.ac_wrapper_path_plan import ACEnvWrapper
from stable_baselines3 import PPO

# import libsumo
# print(libsumo.__file__)
# print("success")


def custom_update_cover_radius(position:List[float], communication_range:float) -> float:
    """自定义的更新地面覆盖半径的方法, 在这里实现您的自定义逻辑

    Args:
        position (List[float]): 飞行器的坐标, (x,y,z)
        communication_range (float): 飞行器的通行范围
    """
    height = position[2]
    cover_radius = height / np.tan(math.radians(75/2))
    return cover_radius


# sumo_cfg = "./sumo_envs/LONG_GANG/env/osm.sumocfg"
sumo_cfg = "./sumo_envs/LONG_GANG/env/osm.sumocfg"
base_dir = os.path.dirname(os.path.abspath(__file__))
drone_img_path = os.path.join(base_dir, "asset", "drone.png")

aircraft_inits = {
    'drone_1': {
        "aircraft_type": "drone",
        "action_type": "horizontal_movement",
        "position": (1650, 1550, 50), 
        "speed": 10, 
        "heading": (1, 1, 0), 
        "communication_range": 50,
        "if_sumo_visualization": True,
        "img_file": drone_img_path,  # 传入图片
        "custom_update_cover_radius": custom_update_cover_radius
    },
}

from train_utils.baseline_models import CustomModel
policy_models = CustomModel


env = ACEnvironment(
    sumo_cfg= sumo_cfg,
    num_seconds= 500,
    aircraft_inits= aircraft_inits,
    use_gui= True
)

env = ACEnvWrapper(env, aircraft_inits)


model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo_drone_model")


import matplotlib.pyplot as plt
import numpy as np

# ===================== 多次评估策略 =====================
print("\n========== 策略收敛性测试 ==========")
n_episodes = 100
episode_rewards = []

for ep in range(n_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)  # 测试用确定性动作
        action = np.int64(action)
        obs, reward, truncated, done, info = env.step(action)
        total_reward += reward
    episode_rewards.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward}")

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(f"\n平均奖励: {mean_reward:.2f}, 奖励标准差: {std_reward:.2f}")

# ===================== 绘制评估结果 =====================
plt.figure(figsize=(10,5))

# 奖励曲线
plt.plot(range(1, n_episodes+1), episode_rewards, marker='o', label="Episode Reward")
plt.axhline(mean_reward, color='red', linestyle='--', label=f"Mean Reward = {mean_reward:.2f}")
plt.fill_between(range(1, n_episodes+1),
                 mean_reward - std_reward,
                 mean_reward + std_reward,
                 color='red', alpha=0.2, label="±1 Std Dev")

plt.title("Policy Evaluation after Training")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()


env.close()
del env