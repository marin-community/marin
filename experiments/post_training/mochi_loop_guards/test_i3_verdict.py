# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect the registered paired comparator, frozen targets and repeat window."""

import pytest

from experiments.post_training.mochi_loop_guards.i3_verdict import paired_verdict


def audit(passes, share, loops):
    return {
        "manifest_sha256": "same-panel",
        "domains": {"tb2": {"N": 100, "passes": passes, "unassessed": 10}},
        "agents": {"tb2": {"repeat_turn_share": share, "loop_trials": loops, "cost": {}}},
        "cohort_hashes": {"tb2": "frozen"},
        "target_status": {"tb2": "target" if loops >= 10 else "non-target"},
    }


def test_i3_keeps_S0_target_and_fails_against_concurrent_control():
    # Treatment halves S0's share, but not concurrent control's. Control's zero
    # confirmed loops must not remove a target frozen before either RL arm.
    result = paired_verdict(audit(80, 0.2, 0), audit(60, 0.15, 12), audit(90, 0.8, 20))
    assert result["target_cohorts"] == ["tb2"]
    assert result["cohorts"][0]["correctness_min_raw"] == pytest.approx(56.25)
    assert result["pooled"]["correctness_min_raw"] == pytest.approx(58.5)
    assert result["validity_pass"] is True
    assert result["correctness_pass"] is True
    assert result["primary_pass"] is False
    assert result["verdict"] == "FAIL"
    assert result["repeat_required"] is False


def test_i3_repeat_uses_overridden_unrounded_cohort_floor():
    result = paired_verdict(audit(80, 0.2, 12), audit(57, 0.09, 4), audit(90, 0.8, 20))
    assert result["cohorts"][0]["cohort_correctness_pass"] is True
    assert result["primary_pass"] is True
    assert result["repeat_triggers"] == ["tb2 correctness"]
    assert result["verdict"] == "PENDING_REPEAT"
