from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from scipy.ndimage import zoom


class EnergyDataset(Dataset):
    def __init__(self, data_path: Path):
        super().__init__()
        self.data_path = data_path
        self.files = os.listdir(data_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        game, player, step = map(int, (file.split('.')[0]).split('_'))

        with open(self.data_path / file, 'rb') as fh:
            data = pickle.load(fh)

        x = data['x']
        upsampled_x = np.zeros((8, 48, 48))
        gt = data['gt']
        gt = (gt + 20.) / 40.

        for i in range(x.shape[0]):
            upsampled_x[i, :, :] = zoom(x[i, :, :], 2, order=0)
        # gt = zoom(gt, 2, order=0)

        return {
            'original_x': torch.Tensor(x),
            'upsampled_x': torch.Tensor(upsampled_x),
            'gt': torch.unsqueeze(torch.Tensor(gt), 0),
            'meta': {
                'game': game,
                'player': player,
                'step': step
            }
        }