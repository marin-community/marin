# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
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
