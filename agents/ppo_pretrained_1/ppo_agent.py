from state.state import State
from stable_baselines3 import PPO
import numpy as np
from sys import stderr


class PPOAgent:
    def __init__(self):
        self.ppo_model = PPO.load("mlp_ppo_model")


    def act(self, state: State):
        obs = state.get_obs()
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        action = np.array([[a, 0, 0] for a in action], dtype=np.int8)
        return action