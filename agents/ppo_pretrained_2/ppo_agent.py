from state.state import State, MAX_UNITS
from stable_baselines3 import PPO
import numpy as np
from sys import stderr
import torch
from state.action_type import _DIRECTIONS
from numpy import random


class PPOAgent:
    def __init__(self):
        self.ppo_model = PPO.load("./ppo_model")


    def act(self, state: State):
        obs = torch.Tensor(np.array([state.get_obs()]))
        actions = self.ppo_model.predict(obs)[0][0]
        # print(self.ppo_model.policy.mlp_extractor.actor_net(obs)[0], file=stderr)
        
        return self.get_actions(state, actions)
    

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
            cur_ship_actions = actions[5 * ship_idx:5 * (ship_idx + 1)]
            cur_ship_actions -= cur_ship_actions.min()
            prob_sum = cur_ship_actions.sum()
            if prob_sum > 0:
                ship_predicted_actions = cur_ship_actions / prob_sum
            else:
                ship_predicted_actions = np.array([0.2] * 5)
            best_actions = np.argsort(-ship_predicted_actions)

            sampled_action_successfully = False
            for _ in range(30):
                try:
                    ship_sampled_action = random.choice(list(range(5)), p=ship_predicted_actions)
                except ValueError as e:
                    print(f'ship_predicted_actions={ship_predicted_actions}, prob_sum={prob_sum}, cur_ship_actions={cur_ship_actions}', file=stderr)
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