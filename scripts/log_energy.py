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


@dataclass
class Args:
    agent: str = "energy_logger"
    """Agent 1 name"""
    n_runs: int = 1
    """How much matches to play"""


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
            f'--seed={run_idx + 2}',
            f'--output={SCRIPT_DIR / "out.json"}'
        ))
        p.wait()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
