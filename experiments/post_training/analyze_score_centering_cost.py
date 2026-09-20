# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join saved evaluation times with every Iris GPU task attempt.

Example::

    python -m experiments.post_training.analyze_score_centering_cost \
        --evaluations /tmp/score-evals.csv --attempts /tmp/iris-attempts.csv \
        --task-prefix arm=/romain/my-job --output /tmp/score-cost.csv

The Iris query must include task_id, attempt_id, started_at_ms, and finished_at_ms.
Supply every parent prefix for a run that was continued under a new Iris job.
The cost is reserved GPU time across all attempts, including failures and export.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FIELDS = (
    "run",
    "step",
    "eval_dump_written_utc",
    "elapsed_from_first_gpu_task_hours",
    "reserved_gpu_hours_to_eval",
    "full_run_reserved_gpu_hours",
    "task_attempts",
)


@dataclass(frozen=True)
class TaskAttempt:
    task_id: str
    started_at_ms: int
    finished_at_ms: int | None


def _milliseconds(instant: str) -> int:
    parsed = datetime.fromisoformat(instant)
    if parsed.tzinfo is None:
        raise ValueError(f"evaluation time lacks a timezone: {instant}")
    return round(parsed.timestamp() * 1000)


def _attempts(path: Path) -> list[TaskAttempt]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen: set[tuple[str, str]] = set()
    result = []
    for row in rows:
        if not row["started_at_ms"]:
            continue
        identity = row["task_id"], row["attempt_id"]
        if identity in seen:
            raise ValueError(f"duplicate Iris task attempt: {identity}")
        seen.add(identity)
        start = int(row["started_at_ms"])
        end = int(row["finished_at_ms"]) if row["finished_at_ms"] else None
        if end is not None and end < start:
            raise ValueError(f"Iris task attempt finishes before it starts: {identity}")
        result.append(TaskAttempt(row["task_id"], start, end))
    return result


def summarize_cost(
    evaluations: Path,
    attempts: Path,
    task_prefixes: dict[str, list[str]],
    gpus_per_task: int,
) -> list[dict[str, str | int | float]]:
    if gpus_per_task < 1:
        raise ValueError("gpus_per_task must be positive")
    with evaluations.open(newline="") as stream:
        eval_rows = [row for row in csv.DictReader(stream) if row["dataset"] == "all"]
    if not eval_rows:
        raise ValueError("evaluation CSV has no all-dataset rows")
    all_attempts = _attempts(attempts)
    by_run: dict[str, list[TaskAttempt]] = defaultdict(list)
    for run, prefixes in task_prefixes.items():
        for attempt in all_attempts:
            if "/users-" in attempt.task_id and any(attempt.task_id.startswith(prefix + "/") for prefix in prefixes):
                by_run[run].append(attempt)

    output = []
    seen_evals: set[tuple[str, int]] = set()
    for row in eval_rows:
        run, step = row["run"], int(row["step"])
        if (run, step) in seen_evals:
            raise ValueError(f"duplicate all-dataset evaluation: {run} step {step}")
        seen_evals.add((run, step))
        matched = by_run.get(run, [])
        if not matched:
            raise ValueError(f"{run}: no started Iris GPU task attempts match its prefixes")
        eval_ms = _milliseconds(row["eval_dump_written_utc"])
        first_start = min(attempt.started_at_ms for attempt in matched)
        if eval_ms < first_start:
            raise ValueError(f"{run} step {step}: evaluation predates its GPU tasks")
        reserved_ms = sum(
            max(
                0,
                min(attempt.finished_at_ms if attempt.finished_at_ms is not None else eval_ms, eval_ms)
                - attempt.started_at_ms,
            )
            for attempt in matched
        )
        full_ms = sum(
            attempt.finished_at_ms - attempt.started_at_ms for attempt in matched if attempt.finished_at_ms is not None
        )
        all_finished = all(attempt.finished_at_ms is not None for attempt in matched)
        output.append(
            {
                "run": run,
                "step": step,
                "eval_dump_written_utc": row["eval_dump_written_utc"],
                "elapsed_from_first_gpu_task_hours": (eval_ms - first_start) / 3_600_000,
                "reserved_gpu_hours_to_eval": reserved_ms * gpus_per_task / 3_600_000,
                "full_run_reserved_gpu_hours": full_ms * gpus_per_task / 3_600_000 if all_finished else "",
                "task_attempts": len(matched),
            }
        )
    return sorted(output, key=lambda row: (str(row["run"]), int(row["step"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, type=Path)
    parser.add_argument("--attempts", required=True, type=Path)
    parser.add_argument("--task-prefix", action="append", required=True, metavar="RUN=JOB_PREFIX")
    parser.add_argument("--gpus-per-task", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prefixes: dict[str, list[str]] = defaultdict(list)
    for item in args.task_prefix:
        run, separator, prefix = item.partition("=")
        if not separator or not run or not prefix:
            parser.error(f"invalid --task-prefix {item!r}; use RUN=JOB_PREFIX")
        prefixes[run].append(prefix.rstrip("/"))
    rows = summarize_cost(args.evaluations, args.attempts, prefixes, args.gpus_per_task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cost rows to {args.output}")


if __name__ == "__main__":
    main()
