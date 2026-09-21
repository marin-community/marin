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
from experiments.post_training.analyze_score_centering_format import summarize as summarize_format
from experiments.post_training.analyze_score_centering_pairs import summarize as summarize_pairs


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


def test_core_math_subset_excludes_other_validation_suites(tmp_path):
    session = tmp_path / "dumped_evals" / "global_step_0_evals"
    session.mkdir(parents=True)
    for dataset, count in (("val-gsm8k", 2), ("val-math500", 1), ("val-amc", 1)):
        rows = [
            {
                "input_prompt": f"{dataset}-{index}",
                "env_extras": {"reward_spec": {"ground_truth": "1"}},
                "score": 1 if index == 0 else 0,
                "stop_reason": "stop",
            }
            for index in range(count)
        ]
        (session / f"{dataset}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (session / "aggregated_results.jsonl").write_text(
        json.dumps(
            {
                "eval/val-gsm8k/response_tokens_mean": 10,
                "eval/val-math500/response_tokens_mean": 40,
                "eval/val-amc/response_tokens_mean": 90,
                "eval/all/response_tokens_mean": 37.5,
            }
        )
        + "\n"
    )

    result = summarize_run("arm", str(tmp_path), "https://unused.example", core_math=True)
    core = next(row for row in result if row["dataset"] == "core-math")
    overall = next(row for row in result if row["dataset"] == "all")
    assert (core["questions"], core["completed_correct"], core["response_tokens_mean"]) == (3, 2, 20)
    assert (overall["questions"], overall["completed_correct"]) == (4, 3)
    assert core["membership_sha256"] != overall["membership_sha256"]
    assert verify_membership(result) == overall["membership_sha256"]


def test_pair_comparison_can_use_core_subset_with_broader_validation():
    evaluations = [
        {
            "run": run,
            "step": str(step),
            "dataset": "core-math",
            "questions": "756",
            "membership_sha256": "same-core-questions",
            "completed_correct": str(correct),
        }
        for run, counts in (("control", (100, 130)), ("centered", (100, 140)))
        for step, correct in zip((0, 20), counts, strict=True)
    ]
    repeats = [
        {"run": "control", "scheduled_all": "120", "final_all": "130"},
        {"run": "centered", "scheduled_all": "135", "final_all": "140"},
    ]
    pairs, summary = summarize_pairs(evaluations, repeats, ["pilot:17:control:centered"], 20, dataset="core-math")
    assert pairs[0]["mean_terminal_difference"] == 12.5
    assert summary[0]["mean_terminal_difference"] == 12.5


def test_format_audit_counts_exact_boxes_only_when_completed_and_unrewarded(tmp_path):
    path = tmp_path / "val-gsm8k.jsonl"
    rows = [
        {
            "input_prompt": str(index),
            "env_extras": {"reward_spec": {"ground_truth": "7"}},
            "stop_reason": stop,
            "score": score,
            "output_response": f"<|end_think|>\\boxed{{{answer}}}",
        }
        for index, (stop, score, answer) in enumerate((("stop", 1, 7), ("stop", 0, 7), ("length", 1, 7), ("stop", 0, 8)))
    ]
    rows.append(
        {
            "input_prompt": "earlier correct box",
            "env_extras": {"reward_spec": {"ground_truth": "7"}},
            "stop_reason": "stop",
            "score": 0,
            "output_response": "<|end_think|>\\boxed{7}\nFinal answer: 8",
        }
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = summarize_format(str(path), "https://unused.example")
    assert audit["final_turn_boxed_exact_unrewarded"] == 2
    assert audit["terminal_boxed_exact_unrewarded"] == 1
    assert audit["rewarded_correct_completed"] == 1
    assert audit["completed_rewarded_or_exact_boxed"] == 3
    assert audit["completed_rewarded_or_terminal_boxed"] == 2


def test_format_audit_exact_string_mode_is_conservative_for_math(tmp_path):
    path = tmp_path / "val-math500.jsonl"
    rows = [
        {
            "input_prompt": str(index),
            "env_extras": {"reward_spec": {"ground_truth": truth}},
            "stop_reason": "stop",
            "score": 0,
            "output_response": f"<|end_think|>\\boxed{{{answer}}}<|eot_id|>",
        }
        for index, (truth, answer) in enumerate((("1000", "1,000"), (r"\frac{1}{2}", r"\frac{1}{2}")))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert summarize_format(str(path), "https://unused.example")["terminal_boxed_exact_unrewarded"] == 2
    exact = summarize_format(str(path), "https://unused.example", match_mode="exact_string")
    assert exact["terminal_boxed_exact_unrewarded"] == 1
    assert exact["match_mode"] == "exact_string"


def test_iris_mirror_uses_the_resumed_attempt_for_repeated_steps(tmp_path):
    path = tmp_path / "iris.log"

    def line(attempt, step, tokens):
        metrics = {
            "trainer/global_step": step,
            "async/performance/consumed_loss_tokens": tokens,
            "async/performance/configured_policy_gpus": 8,
            "async/performance/configured_inference_gpus": 8,
            "consumed/sequences": 512,
            "reward/informative_group_fraction": 0.65,
            "async/rejected_count": 3,
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
    assert rows[-1]["consumed_sequences"] == 512
    assert rows[-1]["informative_group_fraction"] == pytest.approx(0.65)
    assert rows[-1]["rejected_count"] == 3
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
