import torch
import os
import numpy as np
from .base import SPACE_SIZE, get_opposite
from sys import stderr
from scipy.ndimage import zoom
from pathlib import Path


class EnergyPredictor:
    def __init__(self):
        model_path_prefix = Path('./state/')
        self.energy_predictor = torch.jit.load(model_path_prefix / 'energy_predictor.pt')
        self.energy_predictor.eval()
        # print("Loaded energy predictor", file=stderr)

        # historical data
        self.prev_energy_fields = [np.zeros((SPACE_SIZE, SPACE_SIZE, 2))] * 3

    
    def update_prev_energy_fields(self, space):
        self.prev_energy_fields.append(space.get_energy_field())

    
    def predict_hidden_energy(self, space):
        energy_field = space.get_energy_field()
        prev_energy_fields = np.array(self.prev_energy_fields[-3:])

        fields = np.stack([
            energy_field[:, :, 0],
            prev_energy_fields[0, :, :, 0],
            prev_energy_fields[1, :, :, 0],
            prev_energy_fields[2, :, :, 0],
        ])
        fields = (fields + 20.) / 40.

        masks = np.stack([
            energy_field[:, :, 1],
            prev_energy_fields[0, :, :, 1],
            prev_energy_fields[1, :, :, 1],
            prev_energy_fields[2, :, :, 1],
        ])

        x = np.concatenate((fields, masks))
        upsampled_x = np.zeros((8, 48, 48))
        for i in range(x.shape[0]):
            upsampled_x[i, :, :] = zoom(x[i, :, :], 2, order=0)

        model_in = torch.unsqueeze(torch.Tensor(upsampled_x), 0)

        predicted_energy = self.energy_predictor(model_in).detach().numpy()[0, 0, :, :]
        predicted_energy = predicted_energy * 40. - 20.

        for node in space:
            x, y = node.coordinates
            opp_x, opp_y = get_opposite(x, y)
            node.predicted_energy = node.energy if node.energy is not None else int((predicted_energy[x, y] + predicted_energy[opp_x, opp_y]) / 2.)
