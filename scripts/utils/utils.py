import os
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
