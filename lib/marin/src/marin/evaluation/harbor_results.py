# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Result summaries and deterministic aggregate repair for Harbor jobs."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarborResultSummary:
    """Aggregate result state derived from completed trial files."""

    completed_trials: int
    mean_reward: float | None
    exception_counts: dict[str, int]


def summarize_harbor_trials(job_dir: Path) -> HarborResultSummary:
    """Read result.json and exception.txt from direct trial directories."""

    rewards: list[float] = []
    completed_trials = 0
    exceptions: Counter[str] = Counter()
    for result_path in sorted(job_dir.glob("*/result.json")):
        completed_trials += 1
        data = json.loads(result_path.read_text())
        reward = data.get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        exception = data.get("exception")
        if isinstance(exception, str) and exception:
            exceptions[exception.split(":", 1)[0]] += 1
    for exception_path in sorted(job_dir.glob("*/exception.txt")):
        text = exception_path.read_text().strip()
        if text:
            exceptions[text.split(":", 1)[0]] += 1
    return HarborResultSummary(
        completed_trials=completed_trials,
        mean_reward=sum(rewards) / len(rewards) if rewards else None,
        exception_counts=dict(exceptions),
    )


def write_harbor_result_summary(job_dir: Path, output_path: Path) -> HarborResultSummary:
    """Write a portable result aggregate from existing trial evidence."""

    summary = summarize_harbor_trials(job_dir)
    output_path.write_text(
        json.dumps(
            {
                "completed_trials": summary.completed_trials,
                "mean_reward": summary.mean_reward,
                "exception_counts": summary.exception_counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary
