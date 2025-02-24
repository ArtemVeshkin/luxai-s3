from ppo_env import PPOEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from ppo_policy import CustomActorCriticPolicy
import torch
from dummy_agent import DummyAgent
from rulebased import Rulebased


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)r
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_last", self.model.ep_info_buffer[-1]["r"])
        return True

env = PPOEnv(opp_agent=DummyAgent)
model = PPO(
    CustomActorCriticPolicy,
    env,
    gamma=1.,
    gae_lambda=1.,
    ent_coef=0.1,
    learning_rate=1e-5,
    clip_range=0.1,
    verbose=0,
    tensorboard_log='./tb_logs/with_global_info_points_gain_reward',
    n_steps=100,
    batch_size=100,
    stats_window_size=10,
    device='cuda'
)

model.learn(total_timesteps=int(5 * 1e5), progress_bar=True, callback=TensorboardCallback())
model.save("ppo_model_vs_dummy_points_gain_force_center")