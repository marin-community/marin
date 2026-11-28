# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Row gating records control outcomes and keeps failed controls visible."""

import pytest

from taskcompendium.pipeline.models import CheckResult, CheckStatus, GraderReadiness, SourceVerificationStatus
from taskcompendium.pipeline.row_gate import gate_source_row


def test_defective_task_row_is_rejected_when_its_source_passes():
    row = {"task_id": "0", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
    checks = [CheckResult(check="empty", status=CheckStatus.DEFECT, detail="graded: reward=1.0; expected=0.0")]
    gated = gate_source_row(row, status=SourceVerificationStatus.PASSED, results={"0": checks})
    assert gated["filter_status"] == "reject"
    assert gated["filter_reasons"] == ["check:empty"]
    assert gated["grader_readiness"] == GraderReadiness.READY


def test_passing_source_preserves_individual_failures_and_marks_source_only_evidence():
    failed = {"task_id": "failed", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
    unsampled = {**failed, "task_id": "unsampled"}
    results = {"failed": [CheckResult(check="oracle", status=CheckStatus.FAIL, detail="Wrong captured output")]}
    rejected = gate_source_row(failed, status=SourceVerificationStatus.PASSED, results=results)
    assert rejected["filter_status"] == "reject"
    assert rejected["filter_reasons"] == ["check:oracle"]
    accepted = gate_source_row(unsampled, status=SourceVerificationStatus.PASSED, results=results)
    assert accepted["filter_status"] == "keep"
    assert accepted["grader_readiness"] == "source_sampled"


@pytest.mark.parametrize("source_status", list(SourceVerificationStatus))
def test_source_gate_preserves_unavailable_review_for_retry(source_status):
    row = {
        "task_id": "unreviewed",
        "filter_status": "defer",
        "filter_reasons": ["review:unavailable"],
        "grader_readiness": "unverified",
    }
    gated = gate_source_row(row, status=source_status, results={})
    assert gated == row


def test_inconclusive_source_defers_eligible_rows_but_preserves_failed_controls():
    row = {"task_id": "one", "filter_status": "keep", "filter_reasons": [], "grader_readiness": "unverified"}
    checks = [CheckResult(check="golden", status=CheckStatus.INFRA_ERROR, detail="machine unavailable")]
    deferred = gate_source_row(row, status=SourceVerificationStatus.INCONCLUSIVE, results={"one": checks})
    assert deferred["filter_status"] == "defer"
    assert deferred["filter_reasons"] == ["source_verification:inconclusive"]

    # A definite failure from an earlier trial outlasts a later infrastructure error.
    checks.append(CheckResult(check="golden", status=CheckStatus.FAIL, detail="correct submission rejected"))
    rejected = gate_source_row(row, status=SourceVerificationStatus.INCONCLUSIVE, results={"one": checks})
    assert rejected["filter_status"] == "reject"
    assert rejected["filter_reasons"] == ["check:golden"]
