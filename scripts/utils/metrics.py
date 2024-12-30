from collections import defaultdict
import numpy as np


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

    NAME_TO_AGG = {
        'MEAN': np.mean,
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
            for agg_name, agg in NAME_TO_AGG.items()
        }
        metrics_aggregates[metric_name] = metric_aggs

    return metrics_aggregates