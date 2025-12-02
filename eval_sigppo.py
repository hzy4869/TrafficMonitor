'''
@Author: Ricca
@Date: 2024-07-16
@Description: 使用训练好的 RL Agent 进行测试
@LastEditTime:
'''
import argparse
import torch
import os
import numpy as np
import math

import sys
sys.path.append(' ./')
sys.path.append('./TransSimHub')
sys.path.append('./TransSimHub/tshub')

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from env_utils.make_tsc_env import make_env
# from env_utils.vis_snir import render_map
from typing import List
from env_utils.vis_snir import render_map

path_convert = get_abs_path(__file__)
logger.remove()

def custom_update_cover_radius(position:List[float], communication_range:float) -> float:
    """自定义的更新地面覆盖半径的方法, 在这里实现您的自定义逻辑

    Args:
        position (List[float]): 飞行器的坐标, (x,y,z)
        communication_range (float): 飞行器的通行范围
    """
    height = position[2]
    cover_radius = height / np.tan(math.radians(75/2))
    return cover_radius

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('--env_name', type=str, default="LONG_GANG_modified", help='The name of environment')
    # parser.add_argument('--env_name', type=str, default="Nguyen_Dupuis", help='The name of environment')
    parser.add_argument('--speed', type=int, default=160, help="100,160,320") # speed决定了地图的scale
    parser.add_argument('--num_envs', type=int, default=1, help='The number of environments')
    parser.add_argument('--policy_model', type=str, default="fusion", help='policy network: baseline_models or fusion_models_0')
    parser.add_argument('--features_dim', type=int, default=512, help='The dimension of output features 64')
    parser.add_argument('--num_seconds', type=int, default=200, help='exploration steps')
    parser.add_argument('--n_steps', type=int, default=512, help='The number of steps in each environment') #500
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate of PPO') #5e-4
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size of PPO')
    parser.add_argument('--cuda_id', type=int, default=0, help='The id of cuda device')
    args = parser.parse_args()  # Parse the arguments
    device = f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu'
    # #########
    # Init Env
    # #########

    log_path = path_convert('./eval_log/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # args.env_name = "Nguyen_Dupuis"
    sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/env/osm.sumocfg")
    # sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/osm.sumocfg")
    # sumo_cfg = path_convert(f"./sumo_envs/Nguyen_Dupuis/ND_env/resized_rectangle.sumocfg")
    # net_file = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.net.xml")


    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement", # combined_movement
            # "position": (0, 0, 50), "speed": 10, "heading": (1, 1, 0), "communication_range": 50,
            "position": (1750, 1200, 50), "speed": 10, "heading": (1, 1, 0), "communication_range": 50,
            "if_sumo_visualization": True, "img_file": path_convert('./asset/drone.png'),
            "custom_update_cover_radius": custom_update_cover_radius  # 使用自定义覆盖范围的计算
        },
    }

    params = {
        'num_seconds': args.num_seconds,
        'sumo_cfg': sumo_cfg,
        'use_gui': True,
        'log_file': log_path,
        'aircraft_inits': aircraft_inits,
    }
    param_name = f'explore_{args.num_seconds}_n_steps_{args.n_steps}_lr_{str(args.lr)}_batch_size_{args.batch_size}'

    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(args.num_envs)])  # multiprocess
    env = VecNormalize.load(load_path=path_convert(f'Result/{args.env_name}/speed_{args.speed}/{args.policy_model}/{param_name}/models/best_vec_normalize.pkl'), venv=env)

    env.training = False  # 测试的时候不要更新
    env.norm_reward = False

    #model_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/best_model.zip')
    model_path = path_convert(f'Result/{args.env_name}/speed_{args.speed}/{args.policy_model}/{param_name}/models/best_model.zip')
    print(model_path)

    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()
    dones = False  # 默认是 False
    total_reward = 0.0
    total_steps = 0
    efficiency = [0]
    count = 0
    trajectory = []

    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        current_pos = infos[0]["pos_drone"][:2]
        trajectory.append(current_pos.copy())
        total_reward += rewards
        # cover_efficiency = env.get_attr('cover_efficiency')[0]
        # total_steps += 1
        # if cover_efficiency is None:
        #     continue
        # efficiency = [i+j for i,j in zip(efficiency, cover_efficiency)]
        # count += 1
        print(rewards)

    # render_map(
    #     trajectories=env.get_attr('ac_trajectories')[0],
    #     veh_trajectories=env.get_attr('veh_trajectories')[0],
    #     cluster_point=env.get_attr('cluster_point')[0],
    #     img_path=path_convert("./temp_trajectories.jpg")
    # )

    env.close()
    # print(f"total count:{count}.")
    # print(f'cover_efficiency, {np.array(efficiency)/count}.')
    print(f'累积奖励为, {total_reward}.')
    print(f"total steps:{total_steps}.")

    # print(trajectory)
    # import numpy as np
    import matplotlib.pyplot as plt

    # 将轨迹变成 numpy array: shape (T,2)
    drone_traj = np.array(trajectory)

    # 固定点 A/B/C
    point_A = np.array([1700, 1400])
    point_B = np.array([1700, 1600])
    point_C = np.array([1900, 1600])

    plt.figure(figsize=(8,8))

    # 绘制无人机轨迹
    plt.plot(drone_traj[:,0], drone_traj[:,1], '-o', markersize=2, label="Drone Trajectory (10 m/s)")

    # ---- 初始点 Start ----
    start_x, start_y = drone_traj[0]
    plt.scatter(start_x, start_y, s=150, c='blue', marker='s')
    plt.text(start_x + 5, start_y + 5, "Start", fontsize=12, color='blue')

    # 绘制固定点
    plt.scatter(point_A[0], point_A[1], s=120, c='red', marker='*', label='A')
    plt.scatter(point_B[0], point_B[1], s=120, c='green', marker='*', label='B')
    plt.scatter(point_C[0], point_C[1], s=120, c='orange', marker='*', label='C')

    # 标注 A / B / C
    plt.text(point_A[0]+5, point_A[1]+5, "A", fontsize=12, color='red')
    plt.text(point_B[0]+5, point_B[1]+5, "B", fontsize=12, color='green')
    plt.text(point_C[0]+5, point_C[1]+5, "C", fontsize=12, color='orange')

    # ---- 添加说明文字（学术式注释，不复杂） ----
    plt.annotate(
        "Activation rule:\n"
        "Step 0 : A, C\n"
        "Step 30: B",
        xy=(1900, 1400),
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", ec="black")
    )

    plt.title("Drone Trajectory with Points A, B, C")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    plt.savefig('./drone_trajectory_ABC.jpg', dpi=300)
    plt.show()

