from ppo_env import PPOEnv
import numpy as np


print('Creating env')
env = PPOEnv()
print('Created env')

print('Resetting env')
env.reset()
print('Env is resetted')

print('Making step')
env.step(np.zeros((16, 3), dtype=np.uint8))
print('Made step')