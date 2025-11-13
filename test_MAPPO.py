# mappo_multi_drone.py
import os
import math
import copy
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

from env_utils.ac_env import ACEnvironment
from env_utils.ac_MultiAgent import ACEnvWrapper

# ===================== 超参数 =====================
SEED = 42
NUM_UPDATES = 2000
STEPS_PER_UPDATE = 2048    # 采样步数（时间步）
PPO_EPOCHS = 8
MINI_BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===================== 帮助函数 =====================
def flat_concat_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """把多个 agent 的观测按 agent id 顺序拼接成全局观测向量（deterministic order）"""
    keys = sorted(obs_dict.keys())
    arrs = [np.array(obs_dict[k]).ravel() for k in keys]
    return np.concatenate(arrs).astype(np.float32)

def obs_shapes(obs_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple]:
    return {k: np.array(v).shape for k, v in obs_dict.items()}

# ===================== 网络定义 =====================
class Actor(nn.Module):
    """共享参数的 actor（离散动作）—— 每个 agent 使用相同策略网络，但输入为局部观测"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        h = self.net(x)
        logits = self.policy_head(h)
        return logits

class CentralizedCritic(nn.Module):
    """集中式的 critic：输入是全局观测（拼接所有 agent 的观测），输出每个 agent 的 value"""
    def __init__(self, global_obs_dim: int, n_agents: int, hidden_size: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # 对每个 agent 输出一个 value
        self.value_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_agents)])

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        h = self.shared(global_obs)
        values = [vh(h) for vh in self.value_heads]  # list of [B,1]
        values = torch.cat(values, dim=1)  # [B, n_agents]
        return values

# ===================== 经验回放（rollout buffer） =====================
class RolloutBuffer:
    def __init__(self, n_agents: int, obs_dim: int, global_obs_dim: int, capacity: int, device):
        self.n_agents = n_agents
        self.capacity = capacity
        self.device = device
        # per-step lists
        self.local_obs = []       # list of dict(agent->obs) kept as flattened vectors in same order
        self.actions = []
        self.log_probs = []
        self.rewards = []         # list of np arrays shape (n_agents,)
        self.dones = []
        self.global_obs = []      # flattened global obs
        self.values = []          # critic values per agent (np array n_agents)
        self.ptr = 0

    def add(self, local_obs_dict, action_dict, logprob_arr, reward_arr, done_flag, global_obs_vec, value_arr):
        # convert local_obs_dict to ordered concat per agent local obs vector
        # but actor takes per-agent local obs individually, so store as dict->list in fixed order
        keys = sorted(local_obs_dict.keys())
        local_list = [np.array(local_obs_dict[k]).ravel().astype(np.float32) for k in keys]
        # store
        self.local_obs.append(local_list)    # list of list: timestep -> [agent0_obs, agent1_obs, ...]
        self.actions.append(action_dict)     # dict agent->int (keep dict)
        self.log_probs.append(logprob_arr)  # np array shape (n_agents,)
        self.rewards.append(reward_arr)      # np array (n_agents,)
        self.dones.append(done_flag)
        self.global_obs.append(global_obs_vec.astype(np.float32))
        self.values.append(value_arr)        # np array (n_agents,)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma=GAMMA, lam=GAE_LAMBDA):
        """使用 GAE 计算每个 agent 的 advantage 和 returns
           last_values: np array shape (n_agents,) — critic 对最后 next_global_obs 的估计
        """
        T = len(self.rewards)
        n = self.n_agents
        advantages = np.zeros((T, n), dtype=np.float32)
        returns = np.zeros((T, n), dtype=np.float32)
        lastgaelam = np.zeros(n, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - float(self.dones[t])
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - float(self.dones[t+1])
                nextvalues = self.values[t+1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
            returns[t] = advantages[t] + self.values[t]
        # flatten for convenience
        return advantages, returns

    def get_batches(self, advantages, returns, batch_size=MINI_BATCH_SIZE):
        """构建用于 PPO 更新的 minibatches（按时间步展开并把 agent 维度合并）"""
        T = len(self.rewards)
        n = self.n_agents
        # Flatten: total samples = T * n
        samples = []
        keys = None
        for t in range(T):
            # local_obs[t] is list of per-agent obs vectors in same order
            agent_obs_list = self.local_obs[t]
            # actions[t] is dict; we need to convert to ordered list
            if keys is None:
                keys = sorted(self.actions[t].keys())
            action_arr = np.array([self.actions[t][k] for k in keys], dtype=np.int64)
            for i in range(n):
                samples.append({
                    'local_obs': agent_obs_list[i],
                    'action': action_arr[i],
                    'logprob': self.log_probs[t][i],
                    'adv': advantages[t][i],
                    'return': returns[t][i],
                    'global_obs': self.global_obs[t],
                    'agent_idx': i
                })
        random.shuffle(samples)
        # yield minibatches
        for start in range(0, len(samples), batch_size):
            mb = samples[start:start+batch_size]
            # convert to tensors
            local_obs = torch.tensor(np.stack([s['local_obs'] for s in mb]), dtype=torch.float32, device=self.device)
            actions = torch.tensor([s['action'] for s in mb], dtype=torch.long, device=self.device)
            old_logprobs = torch.tensor([s['logprob'] for s in mb], dtype=torch.float32, device=self.device)
            advs = torch.tensor([s['adv'] for s in mb], dtype=torch.float32, device=self.device)
            ret = torch.tensor([s['return'] for s in mb], dtype=torch.float32, device=self.device)
            global_obs = torch.tensor(np.stack([s['global_obs'] for s in mb]), dtype=torch.float32, device=self.device)
            agent_idxs = torch.tensor([s['agent_idx'] for s in mb], dtype=torch.long, device=self.device)
            yield local_obs, actions, old_logprobs, advs, ret, global_obs, agent_idxs

    def clear(self):
        self.local_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.global_obs = []
        self.values = []
        self.ptr = 0

# ===================== 主训练流程 =====================
def train_mappo():
    # ---------------- 环境与初始化 ----------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    drone_img_path = os.path.join(base_dir, "asset", "drone.png")
    sumo_cfg = "./sumo_envs/TEST_NET_twoCluster/zy_road3.sumocfg"
    def custom_update_cover_radius(position:List[float], communication_range:float) -> float:
        """自定义的更新地面覆盖半径的方法, 在这里实现您的自定义逻辑

        Args:
            position (List[float]): 飞行器的坐标, (x,y,z)
            communication_range (float): 飞行器的通行范围
        """
        height = position[2]
        cover_radius = height / np.tan(math.radians(75/2))
        return cover_radius

    # 请把 aircraft_inits 扩展为多个无人机
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
            "position": (-150, 80, 50),
            "speed": 10,
            "heading": (1, 0, 0),
            "communication_range": 50,
            "if_sumo_visualization": True,
            "img_file": drone_img_path,
            "custom_update_cover_radius": custom_update_cover_radius
        },
        # 如需更多无人机，按此添加
    }

    env = ACEnvironment(sumo_cfg=sumo_cfg, num_seconds=500, aircraft_inits=aircraft_inits, use_gui=True)
    env = ACEnvWrapper(env, aircraft_inits)

    # 取一次 reset 来推断 obs shape & agent list
    obs = env.reset()
    agent_ids = sorted(list(obs.keys()))
    n_agents = len(agent_ids)
    print(f"[INFO] Agents: {agent_ids}, n_agents={n_agents}")

    # 局部观测维度（以第一个 agent 为准）
    local_obs0 = np.array(obs[agent_ids[0]]).ravel()
    local_obs_dim = local_obs0.shape[0]

    # 全局观测维度：把所有 agent 的局部 obs concat
    global_obs_dim = flat_concat_obs(obs).shape[0]
    action_dim = 8

    # 模型
    actor = Actor(local_obs_dim, action_dim).to(DEVICE)
    critic = CentralizedCritic(global_obs_dim, n_agents).to(DEVICE)

    actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # rollout buffer
    buffer = RolloutBuffer(n_agents, local_obs_dim, global_obs_dim, STEPS_PER_UPDATE, DEVICE)

    # ---------- 主循环 ----------
    for update in range(NUM_UPDATES):
        # 采样阶段
        obs = env.reset()
        ep_rewards = {aid: 0.0 for aid in agent_ids}
        for step in range(STEPS_PER_UPDATE):
            # 准备数据结构
            # local_obs_dict: agent_id -> obs array
            local_obs_dict = {aid: np.array(obs[aid]).astype(np.float32) for aid in agent_ids}
            global_obs_vec = flat_concat_obs(local_obs_dict)

            print(local_obs_dict)

            # actor: 对每个 agent 生成动作（参数共享）
            actions = {}
            logprob_list = []
            for i, aid in enumerate(agent_ids):
                local = torch.tensor(local_obs_dict[aid].ravel(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits = actor(local)            # [1, action_dim]
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
                actions[aid] = int(act.item())
                logprob_list.append(logp.item())

            # critic value（集中式）
            with torch.no_grad():
                global_t = torch.tensor(global_obs_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                values = critic(global_t).cpu().numpy().squeeze(0)  # shape (n_agents,)

            # 与 env 交互：action dict -> next_obs, reward_dict, done_dict, info
            '''truncated not be used!!!'''
            print("the agent take action:", actions['drone_1'])
            next_obs, reward_dict, truncatted_dict, done_dict, info = env.step(actions)

            # 将 reward_dict 转成 np array 按 agent_ids 顺序
            rewards = np.array([float(reward_dict[aid]) for aid in agent_ids], dtype=np.float32)
            # dones: 当任意 done 为 True 时，我们认为 episode 结束（也可改成 per-agent done）
            done_any = any(bool(done_dict.get(aid, False)) for aid in agent_ids)

            # buffer 存储（values 是上一步的估计）
            buffer.add(local_obs_dict, actions, np.array(logprob_list, dtype=np.float32),
                       rewards, done_any, global_obs_vec, values)

            # 统计
            for aid in agent_ids:
                ep_rewards[aid] += reward_dict.get(aid, 0.0)

            obs = next_obs
            if done_any:
                # 如果环境返回 done（episode 结束），重置 env
                obs = env.reset()

        # 采样结束：计算 next_values（bootstrap）
        # 使用最后一个 obs 的全局 obs 去估计 next_values
        last_global_obs = flat_concat_obs({aid: np.array(obs[aid]).astype(np.float32) for aid in agent_ids})
        with torch.no_grad():
            last_v = critic(torch.tensor(last_global_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)).cpu().numpy().squeeze(0)  # (n_agents,)

        advantages, returns = buffer.compute_returns_and_advantages(last_v, gamma=GAMMA, lam=GAE_LAMBDA)

        # PPO 更新——多次 epoch
        for epoch in range(PPO_EPOCHS):
            for (local_obs_b, actions_b, old_logprobs_b, advs_b, ret_b, global_obs_b, agent_idxs_b) in buffer.get_batches(advantages, returns, batch_size=MINI_BATCH_SIZE):
                # local_obs_b: [B, local_obs_dim]
                # actions_b: [B]
                # agent_idxs_b: which agent index for each sample (for selecting value head)
                # global_obs_b: [B, global_obs_dim]

                # ---------- actor loss ----------
                logits = actor(local_obs_b)  # [B, action_dim]
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs_b)
                surr1 = ratio * advs_b
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advs_b
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # ---------- critic loss ----------
                # critic outputs values for all agents [B, n_agents]; pick corresponding head via agent_idxs_b
                values_all = critic(global_obs_b)  # [B, n_agents]
                # gather the values for each sample's agent index
                values_pred = values_all.gather(1, agent_idxs_b.unsqueeze(1)).squeeze(1)  # [B]
                value_loss = (ret_b - values_pred).pow(2).mean()

                # ---------- 总 loss & 更新 ----------
                actor_optim.zero_grad()
                critic_optim.zero_grad()
                (policy_loss - ENTROPY_COEF * entropy).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                actor_optim.step()

                (VALUE_COEF * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_optim.step()

        # 清空 buffer
        buffer.clear()

        # 打印日志
        avg_reward = sum(ep_rewards.values()) / max(1, len(ep_rewards))
        print(f"Update {update+1}/{NUM_UPDATES} \t AvgEpisodeReward(agent-avg)={avg_reward:.3f}")

        # 保存模型（按需）
        if (update + 1) % 50 == 0:
            torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()},
                       f"mappo_checkpoint_{update+1}.pth")

    # 训练结束
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()}, "mappo_final.pth")
    env.close()
    print("Training finished and models saved.")

if __name__ == "__main__":
    train_mappo()
