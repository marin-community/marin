# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record control outcomes on gated task rows and attach their verified routing evidence."""

from collections.abc import Mapping
from typing import Any

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.machine_routes import RouteEvidence, attach_route_evidence
from taskcompendium.pipeline.models import (
    REJECTING_CHECK_STATUSES,
    CheckResult,
    GraderReadiness,
    SourceVerificationStatus,
)
from taskcompendium.pipeline.transforms import is_accepted
from taskcompendium.pipeline.verification import grader_readiness


def record_checks(row: dict[str, Any], checks: list[CheckResult]) -> dict[str, Any]:
    """Attach check outcomes and reject rows with a failed control."""
    row = {
        **row,
        "checks": [check.model_dump(mode="json") for check in checks],
        "grader_readiness": grader_readiness(checks).value,
    }
    failed = {f"check:{check.check}" for check in checks if check.status in REJECTING_CHECK_STATUSES}
    if failed:
        row = {**row, "filter_status": "reject", "filter_reasons": [*row["filter_reasons"], *sorted(failed)]}
    return row


def gate_source_row(
    row: dict[str, Any],
    *,
    status: SourceVerificationStatus,
    results: dict[str, list[CheckResult]],
    evidence_by_task: Mapping[str, RouteEvidence] | None = None,
    source_evidence: RouteEvidence | None = None,
) -> dict[str, Any]:
    if row["task_id"] in results:
        row = record_checks(row, results[row["task_id"]])
    if is_accepted(row):
        evidence = (evidence_by_task or {}).get(row["task_id"])
        # A row the sample did not check inherits the passing sample's verdicts, so a backend
        # whose sampled control failed cannot serve it either.
        if evidence is None and status == SourceVerificationStatus.PASSED and row["task_id"] not in results:
            evidence = source_evidence
        row = attach_route_evidence(row, evidence or {})
    if not is_accepted(row):
        return row
    if status == SourceVerificationStatus.VALIDATED:
        if row["task_id"] not in results:
            raise ValueError(f"Per-task validation omitted a kept row: {row['task_id']}")
        if row["grader_readiness"] == GraderReadiness.READY:
            return row
        return {
            **row,
            "filter_status": "defer",
            "filter_reasons": [*row["filter_reasons"], "task_validation:unavailable"],
        }
    if status == SourceVerificationStatus.PASSED:
        if row["task_id"] not in results:
            return {**row, "grader_readiness": GraderReadiness.SOURCE_SAMPLED.value}
        return row
    if status == SourceVerificationStatus.SKIPPED:
        return {**row, "grader_readiness": GraderReadiness.UNVERIFIED.value}
    return {
        **row,
        "filter_status": "defer" if status == SourceVerificationStatus.INCONCLUSIVE else "reject",
        "filter_reasons": [*row["filter_reasons"], f"source_verification:{status.value}"],
    }


def gate_row_with_previous(
    row: dict[str, Any],
    *,
    status: SourceVerificationStatus,
    results: dict[str, list[CheckResult]],
    previous: dict[str, dict[str, list[CheckResult]]],
    evidence_by_task: Mapping[str, RouteEvidence],
    source_evidence: RouteEvidence | None,
) -> dict[str, Any]:
    """Gate one audit row, reusing definite failure evidence from earlier runs of the same output."""
    if row["task_id"] not in results and row["task_id"] in previous and is_accepted(row):
        digest = canonical_sha256(TaskSpec.model_validate_json(row["task_json"]).model_dump(mode="json"))
        checks = previous[row["task_id"]].get(digest)
        if checks is not None:
            return gate_source_row(
                row,
                status=status,
                results={row["task_id"]: checks},
                evidence_by_task=evidence_by_task,
                source_evidence=source_evidence,
            )
    return gate_source_row(
        row,
        status=status,
        results=results,
        evidence_by_task=evidence_by_task,
        source_evidence=source_evidence,
    )
