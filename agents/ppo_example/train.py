from stable_baselines3 import PPO
from ppo_game_env import PPOGameEnv

# Create an environment instance
env = PPOGameEnv()

# Initialize the PPO model using a multi-layer perceptron strategy
model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./tb_logs')

model.learn(total_timesteps=int(5 * 1e7), progress_bar=True)

model.save("ppo_game_env_model")

# Test: Load the model and perform a simulation
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()