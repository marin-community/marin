# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality reflects executed QA, Result reflects its findings, and Daytona is never blank."""

import pytest

from infra.marina.apps.rl_data_catalog.server.quality import Quality, Result, daytona, quality, result


def _curation(status: str | None, mode: str = "full") -> dict:
    return {"mode": mode, "verification": {"status": status} if status else None}


@pytest.mark.parametrize(
    ("declaration", "curation", "expected"),
    [
        ({"oversight": "none"}, None, Quality.QUICK),
        ({"oversight": "none", "independent_qc": {"verdict": "good"}}, None, Quality.QUICK),
        ({"oversight": "none"}, _curation("skipped"), Quality.QUICK),
        ({"oversight": "none"}, _curation("passed", mode="quick"), Quality.QUICK),
        ({"oversight": "none"}, _curation("passed"), Quality.SAMPLE),
        ({"oversight": "none"}, _curation("validated"), Quality.FULL),
        ({"oversight": "sample"}, _curation("validated"), Quality.FULL),
        ({"oversight": "full"}, None, Quality.FULL),
    ],
)
def test_quality_is_the_stronger_of_oversight_and_executed_controls(declaration, curation, expected):
    assert quality(declaration, curation) is expected


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"good": 20, "some_issues": 0, "bad": 0, "unknown": 0}, Result.GOOD),
        ({"good": 18, "some_issues": 1, "bad": 1, "unknown": 0}, Result.GOOD),
        ({"good": 17, "some_issues": 3, "bad": 0, "unknown": 0}, Result.SOME_ISSUES),
        ({"good": 11, "some_issues": 4, "bad": 5, "unknown": 0}, Result.SOME_ISSUES),
        ({"good": 10, "some_issues": 6, "bad": 4, "unknown": 0}, Result.BAD),
        ({"good": 0, "some_issues": 0, "bad": 20, "unknown": 0}, Result.BAD),
        ({"good": 17, "some_issues": 1, "bad": 0, "unknown": 2}, Result.SOME_ISSUES),
    ],
)
def test_result_uses_reviewed_sample_clean_and_issue_thresholds(counts, expected):
    assert result({"reviewed": 20, "verdict_counts": counts}) is expected


def test_result_without_review_stays_unassessed():
    assert result(None) is None
    assert result({"reviewed": 0, "verdict_counts": {}}) is None


def test_daytona_is_not_applicable_without_a_sandbox_and_measured_otherwise():
    assert daytona({"tags": ["rlvr"]}, 10)["state"] == "na"
    assert daytona({"tags": ["agentic"], "daytona_snapshot_count": None}, 10)["state"] == "unmeasured"
    measured = {
        "tags": ["agentic"],
        "daytona_snapshot_count": 15,
        "daytona_safety": "unsafe",
        "daytona_snapshot_threshold": 10,
    }
    label = daytona(measured, 10)
    assert label["state"] == "unsafe"
    assert label["snapshots"] == 15
    assert label["reason_code"] == "too_many_snapshots"


def test_grader_only_host_image_execution_failure_is_unsafe_with_zero_snapshots():
    declaration = {
        "tags": ["agentic"],
        "daytona_snapshot_count": 0,
        "daytona_safety": "unsafe",
        "daytona_count_basis": "grader-only HostImage; production validator rejected execution requirements",
    }
    label = daytona(declaration, 10)
    assert label["state"] == "unsafe"
    assert label["snapshots"] == 0


def test_measured_machine_slots_distinguish_no_machine_from_grader_only_failure():
    no_machine = {"tags": ["agentic"], "daytona_snapshot_count": 0, "daytona_machine_slots": 0, "daytona_safety": "safe"}
    grader_only = {**no_machine, "daytona_machine_slots": 12, "daytona_safety": "unsafe"}
    assert daytona(no_machine, 10)["state"] == "na"
    assert daytona(grader_only, 10)["state"] == "unsafe"


@pytest.mark.parametrize("reason", ["UNBUILT_IMAGE", "HOST_IMAGE_REQUIRED", "EXECUTION_REQUIREMENT"])
def test_daytona_exposes_measured_unsafe_reason_with_zero_snapshots(reason):
    label = daytona(
        {
            "tags": ["agentic"],
            "daytona_safety": "unsafe",
            "daytona_snapshot_count": 0,
            "daytona_unsafe_reason": reason,
        },
        10,
    )
    assert label["state"] == "unsafe"
    assert label["reason_code"] == reason


def test_daytona_without_typed_reason_preserves_census_refusal():
    label = daytona(
        {
            "tags": ["agentic"],
            "daytona_safety": "unsafe",
            "daytona_snapshot_count": 0,
            "daytona_roles": {"agent": {"daytona": {"failures": {"registry image required": 20}}}},
        },
        10,
    )
    assert label["reason"] == "registry image required"
