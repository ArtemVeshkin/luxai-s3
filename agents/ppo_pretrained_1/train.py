from ppo_env import PPOEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from ppo_policy import CustomActorCriticPolicy
import torch
from dummy_agent import DummyAgent
from rulebased import Rulebased
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv



if __name__ == '__main__':
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_last", self.model.ep_info_buffer[-1]["r"])
            return True

    env = make_vec_env(lambda: PPOEnv(opp_agent=DummyAgent), n_envs=32, vec_env_cls=SubprocVecEnv)
    exp_name = 'points_gain'
    model = PPO(
        CustomActorCriticPolicy,
        env,
        gamma=1.,
        gae_lambda=1.,
        ent_coef=0.1,
        learning_rate=1e-5,
        clip_range=0.1,
        verbose=0,
        tensorboard_log=f'./tb_logs/{exp_name}',
        n_steps=100,
        batch_size=800,
        stats_window_size=32,
        device='cuda'
    )

    model.learn(total_timesteps=int(5 * 1e6), progress_bar=True, callback=TensorboardCallback())
    model.save(f"./ppo_models/{exp_name}/ppo_model")
