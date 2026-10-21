# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the policy on a fixed task set, and the JSON records of graded attempts."""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec

from experiments.post_training.baby_rsi.rollouts import PolicyRunner, TaskRollouts, reward

DIAGNOSTIC_CHARS = 1000


@dataclass(frozen=True)
class Evaluation:
    """The attempts of the policy on each task of an evaluation set, with the task groups for the summary."""

    groups: tuple[TaskRollouts, ...]
    labels: Mapping[str, str]
    """The group label of each task ID, for example its stratum or its capability."""

    def summary(self) -> dict[str, Any]:
        """The mean pass rate over tasks, overall and for each label, with a 95% interval over tasks."""
        by_label: defaultdict[str, list[float]] = defaultdict(list)
        for group in self.groups:
            rate = group.pass_rate
            if rate is not None:
                by_label[self.labels.get(group.task.id, "")].append(rate)
        everything = [rate for rates in by_label.values() for rate in rates]
        return {
            "overall": _mean_interval(everything),
            "by_label": {label: _mean_interval(rates) for label, rates in sorted(by_label.items())},
            "ungraded_tasks": sum(group.pass_rate is None for group in self.groups),
        }


def _mean_interval(rates: Sequence[float]) -> dict[str, float | int]:
    if not rates:
        return {"tasks": 0}
    mean = sum(rates) / len(rates)
    variance = sum((rate - mean) ** 2 for rate in rates) / max(len(rates) - 1, 1)
    return {"tasks": len(rates), "pass_rate": mean, "ci95": 1.96 * math.sqrt(variance / len(rates))}


async def eval_policy(
    runner: PolicyRunner, tasks: Sequence[TaskSpec], labels: Mapping[str, str], samples: int
) -> Evaluation:
    """``samples`` attempts of each task with the policy's current weights."""
    return Evaluation(tuple(await runner.run_all(tasks, samples)), labels)


def group_record(group: TaskRollouts, *, transcripts: bool) -> dict[str, Any]:
    """The JSON record of one task's attempts: rewards and grades, and the conversations if ``transcripts``."""
    record: dict[str, Any] = {
        "task_id": group.task.id,
        "tags": list(group.task.tags),
        "model": group.model,
        "pass_rate": group.pass_rate,
        "rewards": [reward(graded) for graded in group.rollouts],
        "grades": [
            {
                "status": graded.grade.status.value,
                "reward": graded.grade.reward,
                "error": graded.grade.error,
                "diagnostics": json.dumps(graded.grade.diagnostics, default=str)[:DIAGNOSTIC_CHARS],
            }
            for graded in group.rollouts
        ],
        "stop_reasons": [graded.rollout.stop_reason for graded in group.rollouts],
        "response_tokens": [
            sum(sum(conversation.loss_mask) for conversation in graded.rollout.conversations)
            for graded in group.rollouts
        ],
    }
    if transcripts:
        record["conversations"] = [
            [list(conversation.messages) for conversation in graded.rollout.conversations] for graded in group.rollouts
        ]
        record["failures"] = [
            asdict(graded.rollout.failure) if graded.rollout.failure else None for graded in group.rollouts
        ]
    return record


def write_jsonl(path: StoragePath, records: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, default=str) + "\n" for record in records))
