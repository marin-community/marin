# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve lm-eval task metrics while optionally selecting a bounded reward.

The caller supplies already filtered model responses, including likelihood tuples.
Inference, filters, bootstrap errors and group aggregation remain harness-owned.
No harness dependency is imported by this module.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

from verifyit.adapters.harness_native import native_task_metrics
from verifyit.grade import InvalidTask, Reward, invalid_task, scored


class Task(Protocol):
    def process_results(self, doc: Any, results: Sequence[Any]) -> Mapping[str, Any]: ...

    def aggregation(self) -> Mapping[str, Callable[[Sequence[Any]], Any]]: ...


@dataclass(frozen=True)
class TaskResult:
    """Source metrics and an explicitly requested scalar reward, if any."""

    metrics: Mapping[str, Any]
    verdict: Reward | None


def project_metrics(metrics: Mapping[str, Any], reward_metric: str | None = None) -> TaskResult:
    """Select one bounded numeric metric without discarding source metric values."""
    if reward_metric is None:
        return TaskResult(metrics, None)
    if reward_metric not in metrics:
        return TaskResult(metrics, invalid_task(f"reward metric {reward_metric!r} is absent"))
    value = metrics[reward_metric]
    if isinstance(value, bool) or not isinstance(value, Real):
        return TaskResult(metrics, invalid_task(f"reward metric {reward_metric!r} is not numeric"))
    reward = float(value)
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        return TaskResult(metrics, invalid_task(f"reward metric {reward_metric!r} is not finite and bounded in [0, 1]"))
    return TaskResult(metrics, scored(reward, metric=reward_metric))


def score_task(task: Task, doc: Any, filtered_responses: Sequence[Any], reward_metric: str | None = None) -> TaskResult:
    """Use recognized native routes; retain source compatibility for other contracts."""
    try:
        metrics = native_task_metrics(task, doc, filtered_responses)
    except InvalidTask as error:
        return TaskResult({}, invalid_task(str(error)))
    if metrics is None:
        metrics = task.process_results(doc, filtered_responses)
    return project_metrics(metrics, reward_metric)


def aggregate_task(task: Task, samples: Sequence[Mapping[str, Any]], reward_metric: str | None = None) -> TaskResult:
    """Use source aggregators on original structured metric values.

    Optional metrics emitted by only some samples retain harness behavior: each
    aggregator receives the values actually emitted, in original sample order.
    """
    aggregators = task.aggregation()
    values: dict[str, list[Any]] = {}
    for sample in samples:
        for metric, value in sample.items():
            values.setdefault(metric, []).append(value)
    metrics = {metric: aggregators[metric](items) for metric, items in values.items()}
    return project_metrics(metrics, reward_metric)
