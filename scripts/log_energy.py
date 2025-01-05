import tyro
from dataclasses import dataclass
from utils import (
    prepare_agent_folder,
    try_get_luxai_root_path,
    clear_and_create_dir,
)
import subprocess
from tqdm.auto import tqdm
import time
import os
import pickle
import numpy as np
import json


@dataclass
class Args:
    agent: str = "energy_logger"
    """Agent 1 name"""
    n_runs: int = 1
    """How much matches to play"""


def process_logs(raw_logs_dir, output_dir, game_idx):
    with open(raw_logs_dir / 'out.json', 'r') as fh:
        json_out = json.load(fh)

    with open(raw_logs_dir / 'player_0_energy_fields.npy', 'rb') as fh:
        player_0_energy_fields: np.ndarray = np.load(fh)
    with open(raw_logs_dir / 'player_0_prev_energy_fields.npy', 'rb') as fh:
        player_0_prev_energy_fields: np.ndarray = np.load(fh)

    with open(raw_logs_dir / 'player_1_energy_fields.npy', 'rb') as fh:
        player_1_energy_fields: np.ndarray = np.load(fh)
    with open(raw_logs_dir / 'player_1_prev_energy_fields.npy', 'rb') as fh:
        player_1_prev_energy_fields: np.ndarray = np.load(fh)

    gt_energy_fields = []
    for i in range(len(json_out['observations'])):
        if i % 101 != 0:
            gt_energy_fields.append(np.array(json_out['observations'][i]['map_features']['energy']))
    gt_energy_fields = np.array(gt_energy_fields)

    for player_idx, (energy_fields, prev_energy_fields) in enumerate(zip(
        (player_0_energy_fields, player_1_energy_fields),
        (player_0_prev_energy_fields, player_1_prev_energy_fields)
    )):
        for step, (gt, cur, prev) in enumerate(zip(gt_energy_fields, energy_fields, prev_energy_fields)):
            fields = np.stack([
                cur[:, :, 0],
                prev[0, :, :, 0],
                prev[1, :, :, 0],
                prev[2, :, :, 0]
            ])
            fields = (fields + 20.) / 40.

            masks = np.stack([
                cur[:, :, 1],
                prev[0, :, :, 1],
                prev[1, :, :, 1],
                prev[2, :, :, 1]
            ])

            x = np.concatenate((fields, masks))

            with open(output_dir / f'{game_idx}_{player_idx}_{step}.pickle', 'wb') as fh:
                pickle.dump({'x': x, 'gt': gt}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = tyro.cli(Args)
    agent_name = args.agent
    n_runs = args.n_runs

    LUXAI_ROOT_PATH = try_get_luxai_root_path()
    if LUXAI_ROOT_PATH is None:
        return

    SCRIPT_DIR = LUXAI_ROOT_PATH / 'energy_logs'
    clear_and_create_dir(SCRIPT_DIR)
    AGENT_DIR = LUXAI_ROOT_PATH / 'agents'/ agent_name
    prepare_agent_folder(SCRIPT_DIR / agent_name, AGENT_DIR)
    os.makedirs(SCRIPT_DIR / 'processed_logs')

    print("Running matches")
    for run_idx in tqdm(range(n_runs)):
        p = subprocess.Popen((
            'luxai-s3',
            SCRIPT_DIR / agent_name / "main.py",
            SCRIPT_DIR / agent_name / "main.py",
            f'--seed={run_idx}',
            f'--output={SCRIPT_DIR / "out.json"}'
        ))
        p.wait()
        time.sleep(0.2)

        process_logs(SCRIPT_DIR, SCRIPT_DIR / 'processed_logs', run_idx)


if __name__ == "__main__":
    main()
