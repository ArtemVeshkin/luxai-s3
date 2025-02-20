from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from state.base import SPACE_SIZE, MAX_UNITS


class StatesDataset(Dataset):
    def __init__(self, data_path: Path):
        super().__init__()
        self.data_path = data_path
        files = os.listdir(data_path)
        # files = files[:int(len(files) * 0.1)]
        self.files = self.filter_files(files)


    def __len__(self):
        return len(self.files)


    @staticmethod
    def filter_files(files):
        result_files = []
        for file in files:
            game, player, step = map(int, (file.split('.')[0]).split('_'))
            if step % 101 == 0:
                continue
            result_files.append(file)

        return result_files


    @staticmethod
    def _state_to_obs(state: dict):
        is_explored = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_visible = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_empty = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_nebula = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_asteroid = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_explored_for_relic = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_explored_for_reward = np.zeros((SPACE_SIZE, SPACE_SIZE))
        real_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        predicted_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_pos_real_energy_zone = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_pos_predicted_energy_zone = np.zeros((SPACE_SIZE, SPACE_SIZE))

        space_nodes = state['space']['nodes']
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                node = space_nodes[x][y]
                is_explored[x, y] = float(node.type.value != -1)
                is_visible[x, y] = float(node.is_visible)
                is_empty[x, y] = float(node.type.value == 0)
                is_nebula[x, y] = float(node.type.value == 1)
                is_asteroid[x, y] = float(node.type.value == 2)
                is_explored_for_relic[x, y] = float(node.explored_for_relic)
                is_explored_for_reward[x, y] = float(node.explored_for_reward)
                real_energy[x, y] = node.energy if node.energy is not None else 0.
                predicted_energy[x, y] = node.predicted_energy if node.predicted_energy is not None else 0.
                is_pos_real_energy_zone[x, y] = node.energy and (node.energy > 0)
                is_pos_predicted_energy_zone[x, y] = node.predicted_energy and (node.predicted_energy > 0)

        is_relic = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in state['space']['relic_nodes']:
            x, y = node.coordinates
            is_relic[x, y] = 1.

        is_reward = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in state['space']['reward_nodes']:
            x, y = node.coordinates
            is_reward[x, y] = 1.

        ship_masks = np.zeros((SPACE_SIZE, SPACE_SIZE, MAX_UNITS))

        own_ships = np.zeros((SPACE_SIZE, SPACE_SIZE))
        own_ships_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        own_ship_is_harvesting = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for idx, ship in enumerate(state['ships']):
            if ship['node'] is not None:
                x, y = ship['node'].coordinates
                ship_masks[x, y, idx] = 1.
                own_ships[x, y] = 1.
                own_ships_energy[x, y] = ship['energy']
                own_ship_is_harvesting[x, y] = float(ship['node'].reward)

        opp_ships = np.zeros((SPACE_SIZE, SPACE_SIZE))
        opp_ships_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        opp_ship_is_harvesting = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for idx, ship in enumerate(state['opp_ships']):
            if ship['node'] is not None:
                x, y = ship['node'].coordinates
                opp_ships[x, y] = 1.
                opp_ships_energy[x, y] = ship['energy']
                opp_ship_is_harvesting[x, y] = float(ship['node'].reward)

        dist_to_center_x = np.zeros((SPACE_SIZE, SPACE_SIZE))
        dist_to_center_y = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for x in range(SPACE_SIZE // 2):
            dist_to_center_x[SPACE_SIZE // 2 + x, :] = x
            dist_to_center_x[SPACE_SIZE // 2 - x, :] = x
        for y in range(SPACE_SIZE // 2):
            dist_to_center_y[:, SPACE_SIZE // 2 + y] = y
            dist_to_center_y[:, SPACE_SIZE // 2 - y] = y

        obs = np.stack([
            *[ship_masks[:, :, idx] for idx in range(MAX_UNITS)],
            is_explored,
            is_visible,
            is_empty,
            is_nebula,
            is_asteroid,
            is_explored_for_relic,
            is_explored_for_reward,
            real_energy,
            predicted_energy,
            is_pos_real_energy_zone,
            is_pos_predicted_energy_zone,
            is_relic,
            is_reward,
            dist_to_center_x,
            dist_to_center_y,
            own_ships,
            own_ships_energy,
            own_ship_is_harvesting,
            opp_ships,
            opp_ships_energy,
            opp_ship_is_harvesting
        ], axis=0)

        return obs

    def __getitem__(self, index):
        file = self.files[index]
        game, player, step = map(int, (file.split('.')[0]).split('_'))
        with open(self.data_path / file, 'rb') as fh:
            data = pickle.load(fh)

        state = data['state']
        actions = data['actions'][:, 0]

        obs = self._state_to_obs(state)
        alive_ships = ','.join(str(ship_idx) for ship_idx in range(16) if (state['ships'][ship_idx]['node'] is not None) and (state['ships'][ship_idx]['energy'] > 0))

        info = {
            'game': game,
            'player': player,
            'step': step,
            'alive_ships': alive_ships
        }

        return {
            'obs': torch.Tensor(obs),
            'actions': torch.Tensor(actions).long(),
            'info': info
        }
