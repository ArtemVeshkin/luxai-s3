from ppo_env import PPOEnv
import numpy as np
from rulebased import Rulebased
from tqdm.auto import tqdm


print('Creating env')
env = PPOEnv(opp_agent=Rulebased)
print('Created env')

print('Resetting env')
env.reset()
print('Env is resetted')

print('Making step')
env.step(np.zeros((16, 3), dtype=np.uint8))
print('Made step')

print('Making 1000 steps')
# for _ in tqdmrange(1000)):
for i in range(1000):
    obs, reward, terminated, info = env.step(np.zeros((16, 3), dtype=np.uint8))
    print(f'step={i} reward={reward}, terminated={terminated}')
    if terminated:
        break
print('Made 1000 steps')