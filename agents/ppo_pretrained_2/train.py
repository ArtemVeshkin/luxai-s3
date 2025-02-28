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
from stable_baselines3.common.policies import ActorCriticPolicy



if __name__ == '__main__':
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_last", self.model.ep_info_buffer[-1]["r"])
            return True


    n_envs = 48
    exp_name = 'softmax_points_gain'
    env = make_vec_env(lambda: PPOEnv(opp_agent=DummyAgent), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    # env = make_vec_env(lambda: PPOEnv(opp_agent=Rulebased), n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    model = PPO(
        CustomActorCriticPolicy,
        env,
        gamma=1.,
        gae_lambda=1.,
        ent_coef=0.1,
        learning_rate=lambda progress_remaining: float(1e-4) / (2 - progress_remaining),
        clip_range=0.1,
        verbose=0,
        tensorboard_log=f'./tb_logs/{exp_name}',
        n_steps=100,
        batch_size=1200,
        stats_window_size=n_envs,
        device='cuda'
    )
    # model.policy = PPO.load('./ppo_models/points_gain/ppo_model').policy

    model.learn(total_timesteps=int(1.5 * 1e6), progress_bar=True, callback=TensorboardCallback())
    model.save(f"./ppo_models/{exp_name}/ppo_model")
