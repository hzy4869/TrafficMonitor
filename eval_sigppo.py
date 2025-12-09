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
import matplotlib.pyplot as plt

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
    parser.add_argument('--num_seconds', type=int, default=500, help='exploration steps')
    parser.add_argument('--n_steps', type=int, default=512, help='The number of steps in each environment') #500
    parser.add_argument('--lr', type=float, default=5e-4, help='The learning rate of PPO') #5e-4
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
            "position": (1650, 1550, 50), "speed": 10, "heading": (1, 1, 0), "communication_range": 50,
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
    points_dict = {
        "A": np.array([1700, 1400, 0]),
        "B": np.array([1800, 1500, 30]),
        "C": np.array([1900, 1400, 0]),
        "D": np.array([1800, 1300, 30]),
        "E": np.array([2000, 1300, 0])
    }
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

    ## 绘图
    # 将轨迹变成 numpy array: shape (T,2)
    drone_traj = np.array(trajectory)

    plt.figure(figsize=(8,8))

    # 绘制无人机轨迹
    drone_traj = np.array(trajectory)
    plt.plot(drone_traj[:,0], drone_traj[:,1], '-o', markersize=2, label="Drone Trajectory")

    # 绘制起点
    start_x, start_y = drone_traj[0]
    plt.scatter(start_x, start_y, s=150, c='blue', marker='s')
    plt.text(start_x + 5, start_y + 5, "Start", fontsize=12, color='blue')

    # 绘制目标点
    target_points = np.array([v[:2] for v in points_dict.values()])  # 只取 x,y
    point_names = list(points_dict.keys())  # ['A', 'B', 'C', 'D', 'E']
    point_steps = np.array([v[2] for v in points_dict.values()])  # 可用于标注出现顺序

    colors = plt.cm.get_cmap('tab10', len(target_points))
    for i, (pt, name, step) in enumerate(zip(target_points, point_names, point_steps)):
        plt.scatter(pt[0], pt[1], s=120, c=[colors(i)], marker='*', label=f'{name} (step {step})')
        plt.text(pt[0]+5, pt[1]+5, f'{name}', fontsize=12, color=colors(i))

    plt.title("Drone Trajectory with Points from ACEnvWrapper")
    plt.xlabel("X/m")
    plt.ylabel("Y/m")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.savefig('./drone_trajectory.jpg', dpi=300)
    plt.show()

