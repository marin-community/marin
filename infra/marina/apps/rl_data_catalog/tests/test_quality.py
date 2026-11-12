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


def test_a_recorded_defect_is_bad_whatever_the_other_evidence():
    assert result({"status": "Excluded"}, _curation("passed"), "good", Quality.SAMPLE) is Result.BAD
    assert result({"status": "Available"}, _curation("rejected"), None, Quality.QUICK) is Result.BAD
    assert result({"status": "Available"}, _curation("passed"), "bad", Quality.SAMPLE) is Result.BAD


def test_sampled_grade_then_executed_qa_then_upstream_qc_decide_the_result():
    available = {"status": "Available"}
    assert result(available, _curation("passed"), "some_issues", Quality.SAMPLE) is Result.SOME_ISSUES
    assert result(available, _curation("passed"), "unknown", Quality.SAMPLE) is Result.SOME_ISSUES
    assert result({**available, "oversight": "sample"}, None, None, Quality.SAMPLE) is Result.GOOD
    assert result({**available, "independent_qc": {"verdict": "some_issues"}}, None, None, Quality.QUICK) is (
        Result.SOME_ISSUES
    )
    assert result(available, None, "unknown", Quality.QUICK) is None


def test_daytona_is_not_applicable_without_a_sandbox_and_measured_otherwise():
    assert daytona({"tags": ["rlvr"]}, 10)["state"] == "na"
    assert daytona({"tags": ["agentic"], "daytona_snapshot_count": None}, 10)["state"] == "unmeasured"
    measured = {"tags": ["agentic"], "daytona_snapshot_count": 15, "daytona_safety": "unsafe", "daytona_snapshot_threshold": 10}
    assert daytona(measured, 10) == {"state": "unsafe", "snapshots": 15, "limit": 10, "basis": None}
