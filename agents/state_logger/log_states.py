import tyro
from dataclasses import dataclass
import subprocess
from tqdm.auto import tqdm
import time
import os
import pickle
import numpy as np
import json
from pathlib import Path


def try_get_luxai_root_path() -> Path:
    try:
        return Path(os.environ['LUXAI_ROOT_PATH'])
    except:
        print("LUXAI_ROOT_PATH is not specified!")
        print("Run: export LUXAI_ROOT_PATH=<path to project>")
        print("For example: export LUXAI_ROOT_PATH=/home/artemveshkin/dev/luxai-s3")
        return None


def clear_and_create_dir(path):
    if path.exists():
        os.system(f'rm -rf {path}')
    os.makedirs(path)


def prepare_agent_folder(target_path: Path, agent_path: Path):
    clear_and_create_dir(target_path)

    LUXAI_ROOT_PATH = try_get_luxai_root_path()
    KIT_PATH = LUXAI_ROOT_PATH / 'Lux-Design-S3/kits/python'

    os.system(f'cp -r {agent_path}/. {target_path}')
    os.system(f'cp {KIT_PATH / "main.py"} {target_path}')
    os.system(f'cp -rp {KIT_PATH / "lux"} {target_path}')


@dataclass
class Args:
    n_runs: int = 250
    """How much matches to play"""
    test_size: float = 0.2
    """Test size"""


def process_logs(logs_dir, run_idx):
    for team_idx in range(2):
        for game_step in range(505):
            saved_name = f'{team_idx}_{game_step}.pickle'
            os.rename(f'{logs_dir}/{saved_name}', f'{logs_dir}/{run_idx}_{saved_name}')


def train_test_split(parsed_logs_dir, train_dir, test_dir, n_test_games):
    for parsed_log in tqdm(os.listdir(parsed_logs_dir)):
        game_idx = int(parsed_log.split('_')[0])
        is_train = True
        if game_idx >= n_test_games:
            is_train = False

        os.system(f'mv {parsed_logs_dir / parsed_log} {(train_dir if is_train else test_dir)}/')


def main():
    args = tyro.cli(Args)
    agent_name = 'state_logger'
    n_runs = args.n_runs

    LUXAI_ROOT_PATH = try_get_luxai_root_path()
    if LUXAI_ROOT_PATH is None:
        return

    SCRIPT_DIR = LUXAI_ROOT_PATH / 'state_logs'
    clear_and_create_dir(SCRIPT_DIR)
    AGENT_DIR = LUXAI_ROOT_PATH / 'agents'/ agent_name
    prepare_agent_folder(SCRIPT_DIR / agent_name, AGENT_DIR)
    os.makedirs(SCRIPT_DIR / 'processed_states')

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
        time.sleep(1.0)

        process_logs(SCRIPT_DIR / 'processed_states', run_idx)

    os.makedirs(SCRIPT_DIR / 'train')
    os.makedirs(SCRIPT_DIR / 'test')

    test_game_idx = int(n_runs * (1 - args.test_size))
    train_test_split(
        SCRIPT_DIR / 'processed_states',
        SCRIPT_DIR / 'train',
        SCRIPT_DIR / 'test',
        test_game_idx
    )

    os.system(f'rm -rf {SCRIPT_DIR}/processed_states')
    os.system(f'rm -rf {SCRIPT_DIR}/state_logger')
    os.system(f'rm -rf {SCRIPT_DIR}/out.json')


if __name__ == "__main__":
    main()
