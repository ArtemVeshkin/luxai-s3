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
        self.files = files
        # self.files = self.filter_files(files)


    def __len__(self):
        return len(self.files)


    @staticmethod
    def filter_files(files):
        result_files = []
        for file in files:
            game, player, step = map(int, (file.split('.')[0]).split('_'))
            if player == 0:
                result_files.append(file)

        return result_files


    @staticmethod
    def _state_to_obs(state: dict):
        space_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        real_energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE))
        predicted_energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE))
        space_nodes = state['space']['nodes']
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                node = space_nodes[x][y]
                space_map[x, y] = node.type.value
                real_energy_map[x, y] = node.energy if node.energy is not None else 0.
                predicted_energy_map[x, y] = node.predicted_energy if node.predicted_energy is not None else 0.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in state['space']['relic_nodes']:
            x, y = node.coordinates
            relic_map[x, y] = 1

        reward_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in state['space']['reward_nodes']:
            x, y = node.coordinates
            reward_map[x, y] = 1

        ship_masks = np.zeros((SPACE_SIZE, SPACE_SIZE, MAX_UNITS))
        ship_energies = np.zeros((SPACE_SIZE, SPACE_SIZE, MAX_UNITS))
        for idx, ship in enumerate(state['ships']):
            if ship['node'] is not None:
                x, y = ship['node'].coordinates
                ship_masks[x, y, idx] = 1
                ship_energies[x, y, idx] = ship['energy']

        obs = np.stack([
            space_map,
            real_energy_map,
            predicted_energy_map,
            relic_map,
            reward_map,
            *[ship_masks[:, :, idx] for idx in range(MAX_UNITS)],
            *[ship_energies[:, :, idx] for idx in range(MAX_UNITS)]
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
        alive_ships = ','.join(str(ship_idx) for ship_idx in range(16) if state['ships'][ship_idx]['node'] is not None)

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
