import gymnasium.spaces
from luxai_s3.wrappers import LuxAIS3GymEnv
import gymnasium as gym
from state.state import State
from dummy_agent import DummyAgent
import numpy as np
from state.base import SPACE_SIZE, MAX_UNITS
from state.action_type import _DIRECTIONS
from sys import stderr
from scipy.special import softmax
from numpy import random


class PPOEnv(gym.Env):
    def __init__(self, opp_agent=DummyAgent):
        self.env = LuxAIS3GymEnv()
        # self.action_space = gym.spaces.MultiDiscrete([5] * 16)
        self.n_rounds = 1
        self.action_space = gym.spaces.Box(
            # low=-1e+2,
            # high=1e+2,
            low=0.,
            high=1.,
            shape=(80,),
            dtype=np.float64
        )

        self.observation_space = gym.spaces.Box(
            low=-1e+5,
            high=1e+5,
            shape=(25 + 16 + 2, SPACE_SIZE, SPACE_SIZE),
            dtype=np.double
        )

        self.player_0_state = State('player_0')
        self.player_1_state = State('player_1')
        self.player_1_agent = opp_agent()

        self.reset()


    @staticmethod
    def get_actions(state: State, actions):
        choosen_actions = np.zeros((16))
        ships = state._get_ships(state.fleet)
        space = state._get_space_nodes()
        for ship_idx in range(MAX_UNITS):
            ship = ships[ship_idx]
            if ship['node'] is None or ship['energy'] == 0:
                continue
            x, y = ship['node'].coordinates
            # ship_predicted_actions = softmax(actions[5 * ship_idx:5 * (ship_idx + 1)])
            prob_sum = actions[5 * ship_idx:5 * (ship_idx + 1)].sum()
            if prob_sum > 0:
                ship_predicted_actions = actions[5 * ship_idx:5 * (ship_idx + 1)] / prob_sum
            else:
                ship_predicted_actions = np.array([0.2] * 5)
            best_actions = np.argsort(-ship_predicted_actions)

            sampled_action_successfully = False
            for _ in range(30):
                try:
                    ship_sampled_action = random.choice(list(range(5)), p=ship_predicted_actions)
                except ValueError as e:
                    print(f'actions={actions[5 * ship_idx:5 * (ship_idx + 1)]}, prob_sum={prob_sum}, normed={ship_predicted_actions}')
                direction = _DIRECTIONS[ship_sampled_action]
                next_x = direction[0] + x
                next_y = direction[1] + y
                if next_x > 23 or next_x < 0 or next_y > 23 or next_y < 0:
                    continue
                if not space[next_x, next_y].is_walkable:
                    continue

                choosen_actions[ship_idx] = ship_sampled_action
                sampled_action_successfully = True
                break
            
            if sampled_action_successfully:
                continue

            for a in best_actions:
                if a == 0 and not ship['node'].reward:
                    continue
                direction = _DIRECTIONS[a]
                next_x = direction[0] + x
                next_y = direction[1] + y
                if next_x > 23 or next_x < 0 or next_y > 23 or next_y < 0:
                    continue
                if not space[next_x, next_y].is_walkable:
                    continue
                choosen_actions[ship_idx] = a
                break

        actions = np.array([[a, 0, 0] for a in choosen_actions], dtype=np.int8)
        return actions


    def step(self, actions):
        actions = self.get_actions(self.player_0_state, actions)
        player_1_actions = self.player_1_agent.act(
            self.player_1_state
        )
        obs, env_reward, terminated, truncated, info = self.env.step({
            'player_0': actions,
            'player_1': player_1_actions
        })
        
        self.player_0_state.update(obs['player_0'])
        self.player_1_state.update(obs['player_1'])

        p0_points = self.player_0_state.points
        p1_points = self.player_1_state.points
        points_diff = np.sqrt(np.abs(p0_points - p1_points))
        # reward = 0. if self.player_0_state.match_step != 100 else (-1 if p1_points > p0_points else 1) * points_diff
        # reward = 0. if self.player_0_state.match_step != 100 else np.sqrt(p0_points)
        reward = self.player_0_state.points_gain
        # print(f'Step={self.player_0_state.step}; reward={reward}; p0_points={p0_points}; p1_points={p1_points}')

        if self.player_0_state.match_step == 100:
            empty_actions = np.zeros((16, 3), dtype=np.int8)
            obs, _, terminated, truncated, _ = self.env.step({
                'player_0': empty_actions,
                'player_1': empty_actions
            })
            self.player_0_state.update(obs['player_0'])
            self.player_1_state.update(obs['player_1'])
        terminated = terminated['player_0']
        truncated = truncated['player_0']
        
        if (self.player_0_state.step // 101) >= self.n_rounds:
            terminated = True

        obs = self.player_0_state.get_obs() if not (terminated or truncated) else self.reset()[0]

        return obs, reward, terminated, truncated, info['player_0']


    # def render(self):
    #     return self.env.render()
    

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.player_0_state.reset(info['params'])
        self.player_0_state.update(obs['player_0'])
        self.player_1_state.reset(info['params'])
        self.player_1_state.update(obs['player_1'])
        
        return self.player_0_state.get_obs(), info
    

