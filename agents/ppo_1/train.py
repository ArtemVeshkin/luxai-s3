from ppo_env import PPOEnv
from stable_baselines3 import PPO

env = PPOEnv()

model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./tb_logs', n_steps=505, batch_size=505)

model.learn(total_timesteps=int(1e4), progress_bar=True)

model.save("mlp_ppo_model")