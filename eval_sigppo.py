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
    parser.add_argument('--env_name', type=str, default="LONG_GANG", help='The name of environment')
    # parser.add_argument('--env_name', type=str, default="Nguyen_Dupuis", help='The name of environment')
    parser.add_argument('--speed', type=int, default=160, help="100,160,320") # speed决定了地图的scale
    parser.add_argument('--num_envs', type=int, default=1, help='The number of environments')
    parser.add_argument('--policy_model', type=str, default="fusion", help='policy network: baseline_models or fusion_models_0')
    parser.add_argument('--features_dim', type=int, default=512, help='The dimension of output features 64')
    parser.add_argument('--num_seconds', type=int, default=700, help='exploration steps')
    parser.add_argument('--n_steps', type=int, default=512, help='The number of steps in each environment') #500
    parser.add_argument('--lr', type=float, default=5e-4, help='The learning rate of PPO') #5e-5
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
            "position": (1750, 1000, 50), "speed": 10, "heading": (1, 1, 0), "communication_range": 50,
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
    # trajectory = None  # 轨迹

    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
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
