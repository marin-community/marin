# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

from experiments.post_training.identity_opd.stop_hook import GAP_METRIC, STEP_METRIC, step_rows, stop_reason


def _rows(advantages):
    return [{STEP_METRIC: step, GAP_METRIC: value, "policy/loss": 0.1} for step, value in enumerate(advantages, 1)]


def test_five_strict_rises_fire_and_four_do_not():
    # gap = -advantage, so a falling advantage is a rising gap.
    assert stop_reason(_rows([0.0, -0.1, -0.2, -0.3, -0.4])) is None
    assert "5 consecutive" in stop_reason(_rows([0.0, -0.1, -0.2, -0.3, -0.4, -0.5]))
    assert stop_reason(_rows([0.0, -0.1, -0.2, -0.2, -0.3, -0.4, -0.5])) is None
    assert stop_reason(_rows([0.5, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.4])) is None


def test_a_nonfinite_metric_fires_at_its_step():
    rows = _rows([0.0, 0.1, 0.2])
    rows[1]["policy/loss"] = math.nan
    assert stop_reason(rows) == "nonfinite policy/loss=nan at step 2"


def test_rows_without_the_gap_are_ignored_and_steps_are_ordered():
    history = [
        {STEP_METRIC: 2, GAP_METRIC: -0.2},
        {"_runtime": 5.0},
        {STEP_METRIC: 1, GAP_METRIC: -0.1},
        {STEP_METRIC: 3, "timing/step": 9.0},
    ]
    assert [row[STEP_METRIC] for row in step_rows(history)] == [1, 2]
