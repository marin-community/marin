# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect result eligibility while restoring the dashboard's read contract."""

import hashlib
import json

import pytest

from infra.marina.apps.rl_progress.server.app import comparison_fields


def test_partial_baseline_is_not_a_trusted_gain_and_names_do_not_change_cohort():
    run = {
        "label": "control",
        "evaluation": {"task_count": 100, "task_set_sha256": "a" * 64, "protocol_sha256": "b" * 64},
        "timing_scope": "coordinator_submission_to_terminal",
        "status": "succeeded",
        "elapsed_seconds": 3600,
        "baseline": {"reward": 0.2, "attempted": 100, "scored": 100},
        "final": {"reward": 0.4, "attempted": 100, "scored": 100},
    }
    complete = comparison_fields(run)
    assert complete["trusted_gain_eligible"] is True
    assert comparison_fields(run | {"label": "Fixed runtime · control"}) == complete
    run["baseline"]["scored"] = 99
    partial = comparison_fields(run)
    assert partial["partial_coverage"] is True
    assert partial["trusted_gain_eligible"] is False
    assert partial["comparison_key"] == complete["comparison_key"]

    run.update(verdict="fail", min_scored_fraction=0.99)
    qualified = comparison_fields(run)
    assert qualified["partial_coverage"] and qualified["trusted_gain_eligible"]
    assert qualified["gain_per_hour"] == pytest.approx(0.2)
    assert qualified["comparison_key"] == complete["comparison_key"]
    run["baseline"]["scored"] = 98
    assert not comparison_fields(run)["trusted_gain_eligible"]
    run.update(baseline=None, final=None, verdict="invalid")
    missing = comparison_fields(run)
    assert missing["gain_per_hour"] is None and missing["partial_coverage"]
    assert not missing["trusted_gain_eligible"] and not missing["pareto_eligible"]
    run.update(
        baseline={"reward": 0.2, "attempted": 100, "scored": 100},
        final={"reward": 0.4, "attempted": 100, "scored": 100},
        elapsed_seconds=0,
    )
    untimed = comparison_fields(run)
    assert untimed["gain_per_hour"] is None and not untimed["pareto_eligible"]


def test_changed_membership_cannot_become_rankable_under_an_old_digest():
    task_ids = ["first", "second"]
    digest = hashlib.sha256(json.dumps(task_ids, separators=(",", ":")).encode()).hexdigest()
    run = {"evaluation": {"task_ids": ["first", "replacement"], "membership_sha256": digest}}
    with pytest.raises(ValueError, match="membership"):
        comparison_fields(run)
