"""Scalar metric aggregation."""

from collections.abc import Callable, Mapping, Sequence
from operator import itemgetter

# How a step combines its optimizer windows' values; every other metric takes the mean.
METRIC_REDUCTION: dict[str, Callable[[Sequence[float]], float]] = {
    "log_ratio_abs_max": max,
    "dp_weight_checksum_mismatch": max,
    "log_ratio_diagnostics_failed": max,
    "n_tokens_dp_gt_1pct": sum,
    "n_tokens_dp_gt_10pct": sum,
    "n_tokens_dp_gt_50pct": sum,
    "policy_lr": itemgetter(-1),
    "skipped_steps": sum,
}


def mean_metrics(metrics: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Average each scalar metric sequence, rejecting empty or nonnumeric values."""
    reduced = {}
    for name, values in metrics.items():
        if not values:
            raise ValueError(f"No values for metric {name}")
        if not all(isinstance(value, (int, float)) for value in values):
            raise TypeError(f"Metric {name} contains a non-numeric value")
        reduced[name] = sum(values) / len(values)
    return reduced


def policy_training_metrics(
    metrics: Mapping[str, Sequence[float]],
    policy_update_steps: float,
) -> dict[str, float]:
    """Reduce policy metrics over a step's optimizer windows."""
    status = mean_metrics({name: values for name, values in metrics.items() if name != "response_length"})
    for name in METRIC_REDUCTION.keys() & status.keys():
        status[name] = METRIC_REDUCTION[name](metrics[name])
    status["policy_update_steps"] = policy_update_steps
    return status


def policy_progress_metrics(status: Mapping[str, float]) -> dict[str, float]:
    """Select the compact policy progress fields displayed during training."""
    progress = {
        "pg": status["policy_loss"],
        "glen": status["response_length"],
        "policy_lr": status["policy_lr"],
        "ent": status["policy_entropy"],
    }
    if "raw_grad_norm" in status:
        progress["grad_norm"] = status["raw_grad_norm"]
    return progress
