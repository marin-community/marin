# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv

import pytest

from experiments.post_training.analyze_score_centering_cost import summarize_cost


def test_cost_includes_failed_attempts_and_continuation_at_evaluation_time(tmp_path):
    evaluations = tmp_path / "evals.csv"
    with evaluations.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "step", "dataset", "eval_dump_written_utc"])
        writer.writeheader()
        writer.writerow(
            {
                "run": "arm",
                "step": 10,
                "dataset": "all",
                "eval_dump_written_utc": "1970-01-01T00:00:25+00:00",
            }
        )
    attempts = tmp_path / "attempts.csv"
    with attempts.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "attempt_id", "started_at_ms", "finished_at_ms"])
        writer.writeheader()
        for rank in (0, 1):
            writer.writerow(
                {
                    "task_id": f"/romain/first/users-run/{rank}",
                    "attempt_id": 0,
                    "started_at_ms": 0,
                    "finished_at_ms": 10000,
                }
            )
            writer.writerow(
                {
                    "task_id": f"/romain/continued/users-run/{rank}",
                    "attempt_id": 0,
                    "started_at_ms": 20000,
                    "finished_at_ms": 30000,
                }
            )
        writer.writerow(
            {
                "task_id": "/romain/continued/users-run-export/0",
                "attempt_id": 0,
                "started_at_ms": 30000,
                "finished_at_ms": 35000,
            }
        )

    rows = summarize_cost(
        evaluations,
        attempts,
        {"arm": ["/romain/first", "/romain/continued"]},
        gpus_per_task=8,
    )
    assert len(rows) == 1
    assert rows[0]["elapsed_from_first_gpu_task_hours"] == pytest.approx(25 / 3600)
    assert rows[0]["reserved_gpu_hours_to_eval"] == pytest.approx((2 * 10 + 2 * 5) * 8 / 3600)
    assert rows[0]["full_run_reserved_gpu_hours"] == pytest.approx((2 * 10 + 2 * 10 + 5) * 8 / 3600)
    assert rows[0]["task_attempts"] == 5
