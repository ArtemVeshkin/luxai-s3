from collections import defaultdict
import numpy as np
from statsmodels.stats.proportion import proportions_ztest


def calc_metrics(point_seqs, max_steps_in_match=100):
    def calc_point_metrics(point_seq):
        metrics = {}
        for match_idx in range(5):
            match_seq = point_seq[(max_steps_in_match + 1) * match_idx:(max_steps_in_match + 1) * (match_idx + 1)]
            metrics[f'match_{match_idx + 1}_points'] = match_seq[-1]

            for step, value in enumerate(match_seq):
                if value > 0:
                    break

            metrics[f'match_{match_idx + 1}_first_point_step'] = step

        return metrics

    metrics = defaultdict(list)
    for point_seq in point_seqs:
        point_metrics = calc_point_metrics(point_seq)
        for metric_name, value in point_metrics.items():
            metrics[metric_name].append(value)

    return metrics


def calc_metrics_aggregates(metrics):

    name_to_agg = {
        # 'MEAN': np.mean,
        'MEDIAN': np.median,
        # 'Q10': lambda x: np.quantile(x, 0.1),
        # 'Q30': lambda x: np.quantile(x, 0.3),
        # 'Q70': lambda x: np.quantile(x, 0.7),
        # 'Q90': lambda x: np.quantile(x, 0.9),
    }

    metrics_aggregates = {}
    for metric_name, values in metrics.items():
        metric_aggs = {
            agg_name: agg(values)
            for agg_name, agg in name_to_agg.items()
        }

        metrics_aggregates[metric_name] = metric_aggs

    return metrics_aggregates


def pval_to_confidence_level(p_value):
    confidence_level = '(Not confident)'
    if p_value <= 0.05:
        confidence_level = '(Confident)'
    if p_value <= 0.01:
        confidence_level = '(Very confident)'
    return confidence_level


def calc_winrate_pval(n_wins, total_matches):
    if n_wins == total_matches or n_wins == 0:
        return 0.
    return proportions_ztest(count=n_wins, nobs=total_matches, value=0.5)[1]


def calc_winrate_metrics(match_wins_seqs, agents_order_is_correct_list):
    team2_wins = 0.
    team2_wins_as_team_0 = 0.
    team2_wins_as_team_1 = 0.
    team2_match_wins = [0.] * 5
    for match_wins, agents_order_is_correct in zip(match_wins_seqs, agents_order_is_correct_list):
        if match_wins[-1][1] > match_wins[-1][0]:
            team2_wins += 1
            if agents_order_is_correct:
                team2_wins_as_team_1 += 1
            else:
                team2_wins_as_team_0 += 1


        prev_team2_match_score = 0
        for match_idx in range(5):
            cur_team2_match_score = match_wins[match_idx][1]
            if cur_team2_match_score > prev_team2_match_score:
                team2_match_wins[match_idx] += 1
            prev_team2_match_score = cur_team2_match_score

    total_matches = len(match_wins_seqs)
    total_matches_as_team_1 = len(list(filter(lambda x: x, agents_order_is_correct_list)))
    total_matches_as_team_0 = total_matches - total_matches_as_team_1
    return {
        'total_winrate': {
            'value': f'{100 * team2_wins / total_matches:.1f}%',
            'p_value': calc_winrate_pval(team2_wins, total_matches)
        },
        'team_0_winrate': {
            'value': f'{100 * team2_wins_as_team_0 / total_matches_as_team_0:.1f}%',
            'p_value': calc_winrate_pval(team2_wins_as_team_0, total_matches_as_team_0)
        },
        'team_1_winrate': {
            'value': f'{100 * team2_wins_as_team_1 / total_matches_as_team_1:.1f}%',
            'p_value': calc_winrate_pval(team2_wins_as_team_1, total_matches_as_team_1)
        }
    } | {
        f'match{match_idx + 1}_winrate': {
            'value': f'{100 * team2_match_wins[match_idx] / total_matches:.1f}%',
            'p_value': calc_winrate_pval(team2_match_wins[match_idx], total_matches)
        } for match_idx in range(5)
    }
