from ppo_env import PPOEnv
from stable_baselines3 import PPO
import numpy as np
from ppo_policy import CustomActorCriticPolicy
import torch


env = PPOEnv()
model = PPO(
    CustomActorCriticPolicy,
    env,
    learning_rate=0.00001,
    verbose=0,
    tensorboard_log='./tb_logs',
    n_steps=101,
    batch_size=101
)
# model = PPO(CustomActorCriticPolicy, env, verbose=0, n_steps=505, batch_size=505)
model.learn(total_timesteps=int(1.5 * 1e5), progress_bar=True)
model.save("ppo_model")