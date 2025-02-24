from ppo_env import PPOEnv
from dummy_agent import DummyAgent
from tqdm.auto import tqdm
import numpy as np


print('Creating env')
env = PPOEnv(opp_agent=DummyAgent)
print('Created env')

for _ in tqdm(range(1000)):
    env.step(np.zeros((16), dtype=np.uint8))