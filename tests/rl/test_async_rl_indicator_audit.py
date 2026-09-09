# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.async_rl_indicator_audit import auroc, indicator_report, preceding_windows


def test_auc_counts_ties_against_hand_ranked_example():
    assert auroc([1, 1, 0, 0], [3, 2, 2, 1]) == 0.875
    assert auroc([1, 0], [0, 1]) == 0
    assert auroc([1, 0], [1, 1]) == 0.5
    assert auroc([0, 0], [1, 2]) is None


def test_feature_window_does_not_see_the_future_drop():
    records = [
        {"step": step, "metric": "policy/policy_entropy", "value": 2 if step <= 20 else 10000} for step in range(1, 41)
    ]
    rows = preceding_windows(records, {0: 0.1, 20: 0.7, 40: 0.3})
    assert len(rows) == 1
    assert rows[0]["features"]["low_entropy"] == -2
    assert rows[0]["feature_end"] == 20
    assert rows[0]["drop"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="Incomplete preceding window"):
        preceding_windows(records[1:], {0: 0.1, 20: 0.7, 40: 0.3})


def test_missing_and_constant_indicators_cannot_become_safe_bound():
    rows = [
        {"run": run, "drop": drop, "features": {"worker_abs_log_ratio": 0.0}} for run, drop in [("a", 0.1), ("b", -0.1)]
    ]
    result = indicator_report(rows)
    assert result["indicators"]["worker_abs_log_ratio"]["status"] == "untestable_constant"
    assert result["indicators"]["gradient_cosine"]["status"] == "untestable_missing"
    assert "untestable" in result["held_out_seed_gate"]


def test_isolated_step_flag_is_not_erased_by_the_window_mean():
    records = [
        {"step": step, "metric": "policy/behavior_drift/abs_log_ratio_mean", "value": 0.06 if step == 10 else 0.01}
        for step in range(1, 21)
    ]
    row = preceding_windows(records, {20: 0.7, 40: 0.6})[0]
    assert row["features"]["legacy_abs_log_ratio"] < 0.03
    assert row["per_step_flags"]["legacy_abs_log_ratio"] is True
