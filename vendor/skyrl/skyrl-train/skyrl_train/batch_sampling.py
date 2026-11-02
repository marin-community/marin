"""Row selection over trajectory batches."""

from copy import deepcopy
from enum import StrEnum
from typing import cast

from skyrl_train.metric_names import ENVIRONMENT_METRIC_PREFIX
from skyrl_train.trajectory_runners.base import TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_processing import get_rollout_metrics
from skyrl_train.trajectory_runners.trajectory_reward_shaping import refresh_trajectory_reward_shaping_metrics


class RowOwnership(StrEnum):
    ISOLATED = "isolated"
    BORROWED = "borrowed"


def _refresh_filtered_rollout_metrics(filtered: TrajectoryBatch) -> None:
    if "env_metrics" not in filtered and "verification_results" not in filtered:
        return
    metrics = dict(filtered.get("rollout_metrics") or {})
    for name in tuple(metrics):
        if name.startswith(ENVIRONMENT_METRIC_PREFIX):
            del metrics[name]
    metrics.update(
        get_rollout_metrics(
            filtered["response_ids"],
            filtered["rewards"],
            filtered.get("env_metrics"),
            filtered.get("env_classes"),
            verification_results=filtered.get("verification_results"),
        )
    )
    filtered["rollout_metrics"] = metrics


def filter_trajectory_batch(
    output: TrajectoryBatch,
    kept_indices: list[int],
    *,
    row_ownership: RowOwnership = RowOwnership.ISOLATED,
) -> TrajectoryBatch:
    """Select aligned rows; borrowed rows must only be consumed read-only."""
    if not isinstance(row_ownership, RowOwnership):
        raise ValueError(f"unsupported trajectory row ownership: {row_ownership!r}")
    row_count = len(output["response_ids"])
    filtered = {}
    for key, value in output.items():
        if isinstance(value, list) and len(value) == row_count:
            rows = [value[index] for index in kept_indices]
            filtered[key] = [deepcopy(row) for row in rows] if row_ownership is RowOwnership.ISOLATED else rows
        else:
            filtered[key] = value
    result = cast(TrajectoryBatch, filtered)
    _refresh_filtered_rollout_metrics(result)
    refresh_trajectory_reward_shaping_metrics(result)
    return result
