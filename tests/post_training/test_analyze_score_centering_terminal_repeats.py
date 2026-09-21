# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path

from experiments.post_training.analyze_score_centering_terminal_repeats import summarize


def test_retained_qwen_confirmation_history_reproduces_both_terminal_evaluations():
    results = Path(__file__).parents[2] / "experiments/post_training/results"
    with (results / "score_centering_qwen_confirm_evals.csv").open(newline="") as stream:
        evaluations = list(csv.DictReader(stream))
    history = results / "score_centering_qwen_confirm_wandb_evals.jsonl"
    expected = {
        "s20_tis": (293, 297),
        "s20_sc": (321, 326),
        "s21_tis": (279, 262),
        "s21_sc": (256, 238),
        "s22_tis": (264, 268),
        "s22_sc": (259, 262),
    }
    for run, counts in expected.items():
        row = summarize(run, history, evaluations, 40, wandb_history=True)
        assert (row["scheduled_all"], row["final_all"]) == counts


def test_snowball_terminal_repeat_checks_core_math_amid_extra_suites(tmp_path):
    evaluations = [
        {
            "run": "snowball",
            "step": "20",
            "dataset": dataset,
            "questions": str(questions),
            "completed_correct": str(correct),
        }
        for dataset, questions, correct in (
            ("val-gsm8k", 256, 64),
            ("val-math500", 500, 100),
            ("core-math", 756, 164),
            ("val-amc", 128, 20),
            ("all", 884, 184),
        )
    ]
    metrics = {
        "eval/val-gsm8k/completed_stop_score_contribution": 64 / 256,
        "eval/val-math500/completed_stop_fraction": 1.0,
        "eval/val-math500/completed_stop_score_contribution": -300 / 500,
    }
    history = tmp_path / "history.jsonl"
    history.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"run": "snowball", "wandb_run_id": "abc", "_step": 20, "trainer/global_step": 20, **metrics},
                {"run": "snowball", "wandb_run_id": "abc", "_step": 21, **metrics},
            )
        )
    )
    row = summarize("snowball", history, evaluations, 20, wandb_history=True)
    assert (row["scheduled_all"], row["final_all"]) == (164, 164)
