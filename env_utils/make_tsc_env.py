'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
@LastEditTime: 2023-09-08 18:25:42
'''
import gymnasium as gym
from env_utils.ac_env import ACEnvironment
from env_utils.ac_wrapper_modified import ACEnvWrapper
from stable_baselines3.common.monitor import Monitor

def make_env(
        num_seconds:int,sumo_cfg:str,use_gui:bool,
        # net_file:str,
        log_file:str, aircraft_inits:dict,env_index:int,
        ):
    def _init() -> gym.Env:
        ac_env = ACEnvironment(
            sumo_cfg=sumo_cfg,
            num_seconds=num_seconds,
            # net_file=net_file,
            aircraft_inits=aircraft_inits,
            use_gui=use_gui
        )
        ac_wrapper = ACEnvWrapper(env=ac_env, aircraft_inits=aircraft_inits)
        ac_env = Monitor(ac_wrapper, filename=f'{log_file}/{env_index}')
        return ac_env

    return _init