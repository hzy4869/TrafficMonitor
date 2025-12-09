'''
@Author: Ricca
@Date: 2024-07-16
@Description: 基于 Stabe Baseline3 控制单飞行汽车接送乘客 ### traffic monitor
@LastEditTime:
'''
import sys
sys.path.append(' ./')
sys.path.append('./TransSimHub')
sys.path.append('./TransSimHub/tshub')
import argparse
import os
import torch
import numpy as np
import math
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from typing import List

from env_utils.make_tsc_env import make_env
from train_utils.sb3_utils import BestVecNormalizeCallback, linear_schedule,cosine_annealing_schedule

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
    EvalCallback
)

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level="ERROR")

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
    parser.add_argument('--num_envs', type=int, default=20, help='The number of environments')
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
    # Init
    # #########
    sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/env/osm.sumocfg")
    # sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/ND_env/resized_rectangle.sumocfg")
    # net_file = path_convert(f"./sumo_envs/{args.env_name}/osm.net.xml")
    print(sumo_cfg)

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

    # #########
    # Save Path
    # #########
    # 不同乘客数量，不同SNIR_min，保存不同的log文件和model
    param_name = f'explore_{args.num_seconds}_n_steps_{args.n_steps}_lr_{str(args.lr)}_batch_size_{args.batch_size}'

    base_dir = os.path.join("Result", args.env_name, f"speed_{args.speed}", args.policy_model, param_name)
    log_path = os.path.join(base_dir, "logs")
    model_path = os.path.join(base_dir, "models")
    tensorboard_path = os.path.join(base_dir, "tensorboard")

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    params = {
        'num_seconds':args.num_seconds,
        'sumo_cfg':sumo_cfg,
        'use_gui':False,
        # "net_file": net_file,
        'log_file':log_path,
        'aircraft_inits':aircraft_inits,
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(args.num_envs)]) # multiprocess
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)
    env = VecNormalize(env, norm_obs=True, norm_obs_keys=[
        "ac_attr", "target_rel", "grid_counter"], norm_reward=True)
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # #########
    # Callback
    # #########
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=model_path,
        save_vecnormalize=True
    )
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=500,
        verbose=True
    )  # 何时停止
    eval_callback = BestVecNormalizeCallback(
        env,
        best_model_save_path=model_path,
        callback_after_eval=stop_callback,  # 每次验证之后调用, 是否已经不变了, 需要停止
        eval_freq=500,  # 每次更新的样本数量为 n_steps*NUM_CPUS, n_steps 太小可能会收敛到局部最优
        verbose=1
    )  # 保存最优模型

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # #########
    # Training
    # #########
    # if args.policy_model.split("_")[0] == "baseline":
    #     from train_utils.baseline_models import CustomModel
    #     policy_models = CustomModel
    # elif args.policy_model.split("_")[0] == "fusion":
    #     # model_version = args.policy_model.split("_")[-1]
    #     # if model_version == "0":
    #     #     from train_utils.fusion_models_v0 import FusionModel # ac_wrapper 最原版的reward

    #     # if model_version == "0":
    #     # from train_utils.new_model import EnhancedTrafficFeatureExtractor
    #     # policy_models = EnhancedTrafficFeatureExtractor

    #     from train_utils.model import CustomModelWithTrans
    #     policy_models = CustomModelWithTrans


    #     # policy_models = FusionModel
    # else:
    #     raise ValueError("Invalid policy network type.")

    # policy_kwargs = dict(\
    #     features_extractor_class=policy_models,
    #     features_extractor_kwargs=dict(features_dim=args.features_dim,), # 27 44 43 64
    # )
    from train_utils.baseline_models import CustomModel
    policy_models = CustomModel

    model = PPO(
                "MultiInputPolicy", # "MultiInputPolicy""MlpPolicy"
                env,
                batch_size=args.batch_size, #256
                n_steps=args.n_steps,
                n_epochs=5, # 每次间隔 n_epoch 去评估一次
                learning_rate= linear_schedule(args.lr), #linear_schedule(args.lr), # args.lr # cosine_annealing_schedule(args.lr, final_lr=1e-5, total_timesteps=5e5)
                verbose=True, 
                # policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_path, 
                device=device,
                # ent_coef=0.04\\\\\\\\\
            )
    model.learn(total_timesteps=3e6, tb_log_name='UAM', callback=callback_list) #3e5 1e6

    # #################
    # 保存 model 和 env
    # #################
    env.save(f'{model_path}/last_vec_normalize.pkl')
    model.save(f'{model_path}/last_rl_model.zip')
    print('训练结束, 达到最大步数.')

    env.close()