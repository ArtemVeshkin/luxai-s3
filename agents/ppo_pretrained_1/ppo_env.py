import gym.spaces
from luxai_s3.wrappers import LuxAIS3GymEnv
import gym
from state.state import State
from dummy_agent import DummyAgent
import numpy as np
from state.base import SPACE_SIZE, MAX_UNITS
from sys import stderr


class PPOEnv(gym.Env):
    def __init__(self, opp_agent=DummyAgent):
        self.env = LuxAIS3GymEnv()
        
        # env_params = self.env.env_params
        # low = np.zeros((env_params.max_units, 3), dtype=np.int16)
        # low[:, 1:] = -env_params.unit_sap_range
        # high = np.ones((env_params.max_units, 3), dtype=np.int16) * 6
        # high[:, 1:] = env_params.unit_sap_range

        # self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.int16)
        self.action_space = gym.spaces.MultiDiscrete([5] * 16)

        self.observation_space = gym.spaces.Box(low=-20, high=500, shape=(21 + 16, SPACE_SIZE, SPACE_SIZE), dtype=np.int8)

        self.player_0_state = State('player_0')
        self.player_1_state = State('player_1')
        self.player_1_agent = opp_agent()

        self.reset()


    def step(self, actions):
        actions = np.array([[action, 0, 0] for action in actions], dtype=np.int8)
        player_1_actions = self.player_1_agent.act(
            self.player_1_state
        )
        obs, env_reward, terminated, truncated, info = self.env.step({
            'player_0': actions,
            'player_1': player_1_actions
        })
        
        self.player_0_state.update(
            obs['player_0']
        )
        self.player_1_state.update(
            obs['player_1']
        )
        
        # reward = self.player_0_state.points_gain if self.player_0_state.match_step != 0 else 100 * (env_reward['player_0'] - env_reward['player_1'])
        # reward = self.player_0_state.points_gain
        p0_points = self.player_0_state.points
        p1_points = self.player_1_state.points
        points_diff = np.sqrt(np.abs(p0_points - p1_points))
        reward = 0. if self.player_0_state.match_step != 100 else (-1 if p1_points > p0_points else 1) * points_diff
        print(f'match_step={self.player_0_state.match_step} p0={p0_points} p1={p1_points} reward={reward}')

        return self.player_0_state.get_obs(), reward, terminated['player_0'], truncated['player_0'], info['player_0']


    # def render(self):
    #     return self.env.render()
    

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.player_0_state.reset(info['params'])
        self.player_0_state.update(obs['player_0'])
        self.player_1_state.reset(info['params'])
        self.player_1_state.update(obs['player_1'])
        
        return self.player_0_state.get_obs(), info
    

