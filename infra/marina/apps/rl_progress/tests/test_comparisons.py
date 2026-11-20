# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the dashboard's evaluation and comparison eligibility."""

import hashlib
import json

import pytest

from infra.marina.apps.rl_progress.server.app import SNOWBALL_INITIALIZER_ALIASES, comparison_fields, run_metadata


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
    assert (
        comparison_fields(run | {"hardware": {"gpu": "H100", "count": 40}})["comparison_key"]
        != comparison_fields(run | {"hardware": {"gpu": "H200", "count": 40}})["comparison_key"]
    )


def test_legacy_preset_comes_from_the_recorded_series():
    first, second = sorted(SNOWBALL_INITIALIZER_ALIASES)
    earlier = {"initializer_identity": first, "evaluation": {"name": "Snowball fixed five-family development panel"}}
    fixed = {"initializer_identity": second, "evaluation": {"name": "snowball-fixed-100-development"}}
    heldout = fixed | {"evaluation": {"name": "test_panel", "task_count": 1500}, "series": "learning-medium"}

    assert run_metadata(earlier) == run_metadata(fixed)
    assert run_metadata(fixed)["preset"] == "unspecified"
    assert run_metadata(heldout)["preset"] == "learning-medium"
    assert run_metadata(heldout | {"preset": "data-medium"})["preset"] == "data-medium"


def test_changed_membership_cannot_become_rankable_under_an_old_digest():
    task_ids = ["first", "second"]
    digest = hashlib.sha256(json.dumps(task_ids, separators=(",", ":")).encode()).hexdigest()
    run = {"evaluation": {"task_ids": ["first", "replacement"], "membership_sha256": digest}}
    with pytest.raises(ValueError, match="membership"):
        comparison_fields(run)
