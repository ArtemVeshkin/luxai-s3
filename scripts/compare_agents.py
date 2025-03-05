import tyro
from dataclasses import dataclass
from utils import (
    prepare_agent_folder,
    try_get_luxai_root_path,
    calc_metrics,
    calc_metrics_aggregates,
    clear_and_create_dir,
    calc_winrate_metrics,
    pval_to_confidence_level
)
import os
from pathlib import Path
from tqdm.auto import tqdm
import random
import subprocess
import json
import tabulate
from scipy.stats import mannwhitneyu


@dataclass
class Args:
    agent1: str = "baseline"
    """Agent 1 name"""
    agent2: str = "baseline"
    """Agent 2 name"""
    n_runs: int = 100
    """How much matches to play"""
    run_matches: bool = True
    """Run matches or only calculate metrics"""


from concurrent.futures import ThreadPoolExecutor, as_completed


def run_single_match(run_idx, agent1_name, agent2_name, SCRIPT_DIR):
    """Run one match and return 1 if agent2 wins, 0 otherwise."""
    change_agents_order = random.choice((True, False))
    output_file = SCRIPT_DIR / "runs" / f"{run_idx}.json"

    subprocess.run((
        'luxai-s3',
        str(SCRIPT_DIR / (agent2_name if change_agents_order else agent1_name) / "main.py"),
        str(SCRIPT_DIR / (agent1_name if change_agents_order else agent2_name) / "main.py"),
        f'--seed={run_idx}',
        f'--output={output_file}'
    ))

    with open(output_file, 'r') as fh:
        game_results_json = json.load(fh)
        team_wins = game_results_json['observations'][505]['team_wins']
        # Ensure that team wins order is correct
        player_0 = game_results_json['metadata']['players']['player_0'].split('/')[-2]
        if player_0 != agent1_name:
            team_wins = team_wins[::-1]

    # Return 1 if agent2 wins, otherwise 0.
    return 1 if team_wins[1] > team_wins[0] else 0


def run_matches(n_runs, agent1_name, agent2_name, SCRIPT_DIR, max_workers=4):
    """Run all matches concurrently and return the total wins for agent2."""
    print("Running matches")
    clear_and_create_dir(SCRIPT_DIR / 'runs')

    total_wins = 0
    completed_matches = 0
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all match runs concurrently.
        futures = [executor.submit(run_single_match, run_idx, agent1_name, agent2_name, SCRIPT_DIR)
                   for run_idx in range(n_runs)]

        # Use as_completed to update progress and accumulate wins.
        for future in tqdm(as_completed(futures), total=n_runs):
            total_wins += future.result()
            completed_matches += 1
            print(f'Cur winrate = {100 * total_wins / completed_matches:.1f}%')

    return total_wins


def main():
    args = tyro.cli(Args)
    agent1_name = args.agent1
    agent2_name = args.agent2

    LUXAI_ROOT_PATH = try_get_luxai_root_path()
    if LUXAI_ROOT_PATH is None:
        return

    SCRIPT_DIR = LUXAI_ROOT_PATH / 'runs' / f'{agent1_name}_{agent2_name}'
    AGENT1_DIR = LUXAI_ROOT_PATH / 'agents'/ agent1_name
    AGENT2_DIR = LUXAI_ROOT_PATH / 'agents'/ agent2_name
    prepare_agent_folder(SCRIPT_DIR / agent1_name, AGENT1_DIR)
    prepare_agent_folder(SCRIPT_DIR / agent2_name, AGENT2_DIR)

    if args.run_matches:
        total_wins = run_matches(args.n_runs, agent1_name, agent2_name, SCRIPT_DIR, max_workers=8)
        final_winrate = 100 * total_wins / args.n_runs
        print(f'Final winrate = {final_winrate:.1f}%')

    # calc metrics
    print("Calculating metrics")
    agent1_point_seqs = []
    agent2_point_seqs = []
    match_wins_seqs = []
    agents_order_is_correct_list = []
    for game_result in tqdm(os.listdir(SCRIPT_DIR / 'runs')):
        with open(SCRIPT_DIR / 'runs' / game_result, 'r') as fh:
            game_results_json = json.load(fh)

            player_0 = game_results_json['metadata']['players']['player_0'].split('/')[-2]
            agents_order_is_correct = player_0 == agent1_name
            agents_order_is_correct_list.append(agents_order_is_correct)

            agent1_point_seqs.append(list(map(lambda x: x['team_points'][0 if agents_order_is_correct else 1], game_results_json['observations'])))
            agent2_point_seqs.append(list(map(lambda x: x['team_points'][1 if agents_order_is_correct else 0], game_results_json['observations'])))
            
            cur_match_wins = []
            for match_idx in range(1, 6):
                team_wins = game_results_json['observations'][100 * match_idx + match_idx]['team_wins']
                if not agents_order_is_correct:
                    team_wins = team_wins[::-1]
                cur_match_wins.append(team_wins)
            match_wins_seqs.append(cur_match_wins)

    winrate_metrics = calc_winrate_metrics(match_wins_seqs, agents_order_is_correct_list)
    rows = []
    for metric_name, metric in winrate_metrics.items():
        p_value = metric['p_value']
        confidence_level = pval_to_confidence_level(p_value)
        p_value = f'{confidence_level} {p_value:0.8f}'
        rows.append((
            metric_name,
            metric['value'],
            p_value
        ))
    headers = (
        'Winrate type',
        f'{agent2_name} winrate',
        'p_value'
    )
    winrate_metrics_table = tabulate.tabulate(
        rows,
        headers=headers,
        floatfmt=('.1f'),
        tablefmt="presto",
        colalign = ('left', 'right', 'right')
    )
    
    agent1_metrics = calc_metrics(agent1_point_seqs)
    agent2_metrics = calc_metrics(agent2_point_seqs)

    metric_to_pval = {}
    for metric_name in agent1_metrics.keys():
        p_value = mannwhitneyu(
            agent1_metrics[metric_name],
            agent2_metrics[metric_name]
        ).pvalue
        metric_to_pval[metric_name] = p_value

    agent1_metrics_aggregates = calc_metrics_aggregates(agent1_metrics)
    agent2_metrics_aggregates = calc_metrics_aggregates(agent2_metrics)

    rows = []
    for metric_name in agent1_metrics_aggregates.keys():
        agent1_metric_aggs = agent1_metrics_aggregates[metric_name]
        agent2_metric_aggs = agent2_metrics_aggregates[metric_name]
        
        p_value = metric_to_pval[metric_name]
        confidence_level = pval_to_confidence_level(p_value)
        p_value = f'{confidence_level} {p_value:0.8f}'

        for idx, agg_name in enumerate(agent1_metric_aggs.keys()):
            agent1_agg_value = agent1_metric_aggs[agg_name]
            agent2_agg_value = agent2_metric_aggs[agg_name]

            diff = agent2_agg_value - agent1_agg_value

            p_diff = 100 * (agent2_agg_value - agent1_agg_value) / agent1_agg_value
            p_diff = f'{"+" if p_diff > 0 else ""}{p_diff:.1f}%'

            rows.append((
                metric_name if idx == 0 else None,
                agg_name,
                agent1_agg_value,
                agent2_agg_value,
                diff,
                p_diff,
                p_value if idx == 0 else None
            ))
        rows.append([None] * len(rows[0]))

    headers = (
        'Metric',
        'Aggregation',
        f'{agent1_name} score',
        f'{agent2_name} score',
        'diff',
        'p_diff',
        'p_value'
    )
    point_metrics_table = tabulate.tabulate(
        rows,
        headers=headers,
        floatfmt=('.1f'),
        tablefmt="presto",
        colalign = ('left', 'center', 'right', 'right', 'right', 'right')
    )

    metrics_output = f'Comparing {agent1_name} vs {agent2_name} on {len(match_wins_seqs)} games\n\n' + \
        winrate_metrics_table + '\n\n' + point_metrics_table

    print('\n' + metrics_output)

    EVAL_RESULTS_SAVE_DIR = LUXAI_ROOT_PATH / 'compare_results' / f'{agent1_name}_{agent2_name}'
    clear_and_create_dir(EVAL_RESULTS_SAVE_DIR)
    with open(EVAL_RESULTS_SAVE_DIR / 'compare_result.txt', 'w') as fh:
        fh.write(metrics_output)


if __name__ == "__main__":
    main()
