from state.state import State
from ppo_policy import ActorNet
import numpy as np
from numpy import random
from sys import stderr
import torch
from state.action_type import _DIRECTIONS
from scipy.special import softmax
from pathlib import Path
import os


class PretrainedAgent:
    def __init__(self):
        self.model = ActorNet({
            'input_channels': 25,
            'n_res_blocks': 8,
            'all_channel': 48,
            'n_actions': 5,
            'num_features_count': 18,
            'cat_features_count': 14,
            'emb_dim': 8,
        })
        model_path_prefix = Path('./' if 'LUXAI_ROOT_PATH' in os.environ else '/kaggle_simulations/agent/')
        self.model.load_state_dict(torch.load(
            model_path_prefix / 'pretrained.pt'
        , weights_only=True))
        self.model.eval()


    def act(self, state: State):
        obs = torch.Tensor(state.get_obs()).unsqueeze(0)
        model_out = self.model(obs).reshape(-1, 16, 5).detach().numpy()
        # model_actions = np.argmax(model_out, axis=2)[0]
        # action = np.array([[a, 0, 0] for a in model_actions], dtype=np.int8)

        choosen_actions = np.zeros((16))
        ships = state._get_ships(state.fleet)
        space = state._get_space_nodes()
        for ship_idx in range(16):
            ship = ships[ship_idx]
            if ship['node'] is None or ship['energy'] == 0:
                continue
            x, y = ship['node'].coordinates
            ship_predicted_actions = softmax(model_out[0, ship_idx])
            best_actions = np.argsort(-ship_predicted_actions)

            sampled_action_successfully = False
            for _ in range(30):
                ship_sampled_action = random.choice(list(range(5)), p=ship_predicted_actions)
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
            
            # print(f'{softmax(ship_predicted_actions)} {choosen_actions[ship_idx]}', file=stderr)

        action = np.array([[a, 0, 0] for a in choosen_actions], dtype=np.int8)
        return action