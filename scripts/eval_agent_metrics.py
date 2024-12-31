import tyro
from dataclasses import dataclass
from utils import (
    prepare_agent_folder,
    try_get_luxai_root_path,
    calc_metrics,
    calc_metrics_aggregates,
    clear_and_create_dir
)
import os
from pathlib import Path
from tqdm.auto import tqdm
import subprocess
import time
import json
import tabulate


@dataclass
class Args:
    agent: str = "baseline"
    """Agent name"""
    n_runs: int = 100
    """How much matches to play"""
    run_matches: bool = True
    """Run matches or only calculate metrics"""

def main():
    args = tyro.cli(Args)
    agent_name = args.agent

    LUXAI_ROOT_PATH = try_get_luxai_root_path()
    if LUXAI_ROOT_PATH is None:
        return

    SCRIPT_DIR = LUXAI_ROOT_PATH / 'runs' / agent_name
    AGENT_DIR = LUXAI_ROOT_PATH / 'agents'/ agent_name
    prepare_agent_folder(SCRIPT_DIR, AGENT_DIR)

    if args.run_matches:
        # run matches
        print("Running matches")
        clear_and_create_dir(SCRIPT_DIR / 'runs')
        for run_idx in tqdm(range(args.n_runs)):
            p = subprocess.Popen((
                'luxai-s3',
                SCRIPT_DIR / "main.py",
                SCRIPT_DIR / "main.py",
                f'--seed={run_idx}',
                f'--output={SCRIPT_DIR / "runs" / (str(run_idx) + ".json")}'
            ))
            p.wait()
            time.sleep(0.2)

    # calc metrics
    print("Calculating metrics")
    point_seqs = []
    for game_result in tqdm(os.listdir(SCRIPT_DIR / 'runs')):
        with open(SCRIPT_DIR / 'runs' / game_result, 'r') as fh:
            game_results_json = json.load(fh)

            point_seqs.append(list(map(lambda x: x['team_points'][0], game_results_json['observations'])))
            point_seqs.append(list(map(lambda x: x['team_points'][1], game_results_json['observations'])))

    metrics = calc_metrics(point_seqs)
    metrics_aggregates = calc_metrics_aggregates(metrics)

    rows = []
    for metric_name, metric_aggs in metrics_aggregates.items():
        for idx, (agg_name, agg_value) in enumerate(metric_aggs.items()):
            rows.append((
                metric_name if idx == 0 else None,
                agg_name,
                agg_value
            ))
        rows.append((None, None, None))

    headers = ('Metric', 'Aggregation', 'Score')
    result_table = tabulate.tabulate(
        rows,
        headers=headers,
        floatfmt=('.1f'),
        tablefmt="presto",
        colalign = ('left','center','right')
    )
    print(result_table)
    EVAL_RESULTS_SAVE_DIR = LUXAI_ROOT_PATH / 'eval_results' / agent_name
    clear_and_create_dir(EVAL_RESULTS_SAVE_DIR)
    with open(EVAL_RESULTS_SAVE_DIR / 'eval_result.txt', 'w') as fh:
        fh.write(result_table)


if __name__ == "__main__":
    main()
