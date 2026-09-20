# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime

import pytest

from experiments.post_training.analyze_score_centering import (
    summarize_iris_log,
    summarize_iris_logs,
    summarize_run,
    verify_membership,
)


def test_saved_eval_uses_completed_correct_answers_and_checks_membership(tmp_path):
    session = tmp_path / "dumped_evals" / "global_step_0_evals"
    session.mkdir(parents=True)
    rows = [
        {"input_prompt": "a", "env_extras": {"reward_spec": {"ground_truth": "1"}}, "score": 1, "stop_reason": "length"},
        {"input_prompt": "b", "env_extras": {"reward_spec": {"ground_truth": "2"}}, "score": -1, "stop_reason": "stop"},
        {
            "input_prompt": "c",
            "env_extras": {"reward_spec": {"ground_truth": "3"}},
            "score": 1,
            "stop_reason": "end_turn",
        },
    ]
    (session / "val-math500.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (session / "aggregated_results.jsonl").write_text(
        json.dumps({"eval/val-math500/response_tokens_mean": 12, "eval/all/response_tokens_mean": 12}) + "\n"
    )

    result = summarize_run("arm", str(tmp_path), "https://unused.example")
    overall = next(row for row in result if row["dataset"] == "all")
    assert overall["completed_correct"] == 1
    assert overall["completed_correct_rate"] == pytest.approx(1 / 3)
    assert overall["correct_any_stop"] == 2
    assert overall["raw_reward_mean"] == pytest.approx(1 / 3)
    assert overall["length_stop_fraction"] == pytest.approx(1 / 3)
    assert datetime.fromisoformat(overall["eval_dump_written_utc"]).tzinfo is not None
    assert verify_membership(result) == overall["membership_sha256"]

    changed = [dict(row) for row in result]
    next(row for row in changed if row["dataset"] == "all")["membership_sha256"] = "different"
    with pytest.raises(ValueError, match="membership differs"):
        verify_membership(result + changed)


def test_iris_mirror_uses_the_resumed_attempt_for_repeated_steps(tmp_path):
    path = tmp_path / "iris.log"

    def line(attempt, step, tokens):
        metrics = {
            "trainer/global_step": step,
            "async/performance/consumed_loss_tokens": tokens,
            "async/performance/configured_policy_gpus": 8,
            "async/performance/configured_inference_gpus": 8,
            "timing/step": 10.0,
            "policy/policy_entropy": 0.4,
        }
        prefix = f"task=/romain/run/0 attempt={attempt} | WANDB_MIRROR kind=train step={step} metrics="
        return prefix + json.dumps(metrics) + "\n"

    path.write_text(line(0, 1, 10) + line(1, 1, 20) + line(1, 2, 30))
    rows = summarize_iris_log("arm", path)
    assert [(row["step"], row["iris_attempt"], row["consumed_tokens"]) for row in rows] == [
        (1, 1, 20),
        (2, 1, 30),
    ]
    assert rows[-1]["cumulative_consumed_tokens"] == 50
    assert rows[-1]["policy_entropy"] == pytest.approx(0.4)
    with path.open("a") as stream:
        stream.write(line(1, 2, 30))
    assert summarize_iris_log("arm", path) == rows
    with path.open("a") as stream:
        stream.write(line(1, 2, 99))
    with pytest.raises(ValueError, match="conflicting Iris mirror"):
        summarize_iris_log("arm", path)


def test_later_iris_job_supersedes_repeated_steps_after_continuation(tmp_path):
    first, second = tmp_path / "first.log", tmp_path / "second.log"

    def line(step, tokens):
        metrics = {
            "trainer/global_step": step,
            "async/performance/consumed_loss_tokens": tokens,
            "async/performance/configured_policy_gpus": 8,
            "async/performance/configured_inference_gpus": 8,
            "timing/step": 10.0,
        }
        return f"task=/romain/run/0 attempt=0 | WANDB_MIRROR kind=train step={step} metrics={json.dumps(metrics)}\n"

    first.write_text(line(1, 10) + line(2, 20))
    second.write_text(line(2, 25) + line(3, 30))
    rows = summarize_iris_logs("arm", [first, second])
    assert [(row["step"], row["iris_job_index"], row["consumed_tokens"]) for row in rows] == [
        (1, 0, 10),
        (2, 1, 25),
        (3, 1, 30),
    ]
    assert rows[-1]["cumulative_consumed_tokens"] == 65
