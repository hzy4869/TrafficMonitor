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
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.image as mpimg

path_convert = get_abs_path(__file__)
logger.remove()

from train_sigppo import get_config

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
    args, aircraft_inits, sumo_cfg = get_config()
    args.num_envs = 1
    device = f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu'
    
    log_path = path_convert('./eval_log/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    params = {
        'num_seconds': args.num_seconds,
        'sumo_cfg': sumo_cfg,
        'use_gui': False,
        'log_file': log_path,
        'aircraft_inits': aircraft_inits,
    }
    param_name = f'explore_{args.num_seconds}_n_steps_{args.n_steps}_lr_{str(args.lr)}_batch_size_{args.batch_size}'
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(args.num_envs)])  # multiprocess
    env = VecNormalize.load(load_path=path_convert(f'Result/{args.env_name}/speed_{args.speed}/{args.policy_model}/{param_name}/models/best_vec_normalize.pkl'), venv=env)
    env.training = False  # 测试的时候不要更新
    env.norm_reward = False
    model_path = path_convert(f'Result/{args.env_name}/speed_{args.speed}/{args.policy_model}/{param_name}/models/best_model.zip')
    print(model_path)

    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()

    dones = False
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
        total_reward += rewards[0]
        total_steps += 1
        dones = dones[0]
        print(rewards[0])

    env.close()
    print(f'累积奖励为, {total_reward}.')
    print(f"total steps:{total_steps}.")


    #########
    ## 绘图
    
    # ==================== 接口参数 (调整这里) ====================
    # 1. 缩放与平移接口
    SCALE_FACTOR = 1.5    # 整体缩放系数 (1.0表示原始比例，数值越大地图在坐标系中越大)
    IMAGE_X_OFFSET = 550.0  # 图片在X轴的偏移量
    IMAGE_Y_OFFSET = 450.0  # 图片在Y轴的偏移量

    # 2. 轨迹微调 (如果轨迹相对于地图有偏移)
    TRAJ_X_OFFSET = 0.0
    TRAJ_Y_OFFSET = 0.0
    # ==========================================================

    # 轨迹数据处理
    trajectory_np = np.array(trajectory)
    points_dict = {
        "A": np.array([1408, 955, 0]),
        "B": np.array([1817, 1387, 30]),
        "C": np.array([2063, 1053, 0]),
        "D": np.array([1978, 776, 55]),
        "E": np.array([2553, 861, 0])
    }

    # 绘制初始化
    plt.figure(figsize=(12, 10))

    # 读取图片并计算等比例范围
    img_path = f"./sumo_envs/{args.env_name}/env/osm.png"
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        img_h, img_w = img.shape[:2] # 获取图片像素高度和宽度
        
        # 自动保持长宽比：根据图片像素比例计算坐标系中的 Extent
        # 以宽度为基准进行缩放，高度随比例变动
        base_w = img_w * SCALE_FACTOR
        base_h = img_h * SCALE_FACTOR
        
        # 计算 Extent: [xmin, xmax, ymin, ymax]
        # 结合平移接口
        extent = [
            IMAGE_X_OFFSET, 
            IMAGE_X_OFFSET + base_w, 
            IMAGE_Y_OFFSET, 
            IMAGE_Y_OFFSET + base_h
        ]
        
        # 绘制底图，根据你的要求使用 origin='upper'
        plt.imshow(img, origin='upper', extent=extent, alpha=0.8)
        print(f"底图已加载。像素尺寸: {img_w}x{img_h}，映射坐标范围: {extent}")
    else:
        print(f"警告：未找到底图 {img_path}")

    # 绘制轨迹
    if len(trajectory_np) > 0:
        plot_traj = trajectory_np + np.array([TRAJ_X_OFFSET, TRAJ_Y_OFFSET])
        plt.plot(plot_traj[:, 0], plot_traj[:, 1], color='cyan', linewidth=2, label='UAV Path', zorder=10)
        plt.scatter(plot_traj[0, 0], plot_traj[0, 1], c='lime', marker='s', s=100, label='Start', zorder=11)
        plt.scatter(plot_traj[-1, 0], plot_traj[-1, 1], c='red', marker='X', s=100, label='End', zorder=11)

    # 绘制目标点
    for name, pos in points_dict.items():
        plt.scatter(pos[0], pos[1], c='yellow', marker='*', s=200, edgecolors='black', zorder=12)
        plt.text(pos[0]+15, pos[1]+15, name, color='white', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # 完善图表
    plt.title(f"Evaluation Trajectory: {args.env_name}")
    plt.xlabel("X / meters")
    plt.ylabel("Y / meters")
    plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.3)
    plt.axis('equal') # 极其重要：确保坐标轴比例一致，图片不会被拉伸

    # 保存结果
    save_path = "Evaluate_Result.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存成功：{save_path}")
    plt.show()