from state.state import State
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
        dist = self.ppo_model.policy.get_distribution(obs)
        actions_dist = torch.stack([d.probs for d in dist.distribution], dim=1).detach().numpy()[0].flatten()
        # predicted_actions = self.ppo_model.policy.mlp_extractor.forward_actor(obs).detach().numpy()[0]
        # print(f'{actions_dist[0:5]} {predicted_actions[0:5]}', file=stderr)
        choosen_actions = np.zeros((16))
        ships = state._get_ships(state.fleet)
        space = state._get_space_nodes()
        for ship_idx in range(16):
            ship = ships[ship_idx]
            if ship['node'] is None or ship['energy'] == 0:
                continue

            x, y = ship['node'].coordinates
            ship_actions_dist = actions_dist[ship_idx * 5:(ship_idx + 1) * 5]
            # ship_predicted_actions = predicted_actions[ship_idx * 5:(ship_idx + 1) * 5]
            # best_actions = np.argsort(-ship_predicted_actions)
            for a in range(5):
                direction = _DIRECTIONS[a]
                next_x = direction[0] + x
                next_y = direction[1] + y
                if next_x > 23 or next_x < 0 or next_y > 23 or next_y < 0:
                    ship_actions_dist[a] = 0.
                elif not space[next_x, next_y].is_walkable:
                    ship_actions_dist[a] = 0.
            # ship_actions_dist[ship_actions_dist < 1e-9] = 0.
            prob_sum = np.sum(ship_actions_dist)
            if prob_sum < 1e-12:
                continue
            ship_actions_dist /= prob_sum
            ship_action = random.choice(list(range(5)), p=ship_actions_dist)
            # print(f'{ship_actions_dist} {ship_action}', file=stderr)
            choosen_actions[ship_idx] = ship_action

            # for a in best_actions:
            #     if a == 0 and not ship['node'].reward:
            #         continue
            #     direction = _DIRECTIONS[a]
            #     next_x = direction[0] + x
            #     next_y = direction[1] + y
            #     if next_x > 23 or next_x < 0 or next_y > 23 or next_y < 0:
            #         continue
            #     if not space[next_x, next_y].is_walkable:
            #         continue
            #     choosen_actions[ship_idx] = a
            #     break

        action = np.array([[a, 0, 0] for a in choosen_actions], dtype=np.int8)
        return action