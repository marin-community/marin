# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize conclusive review outcomes without discarding uncertain observations."""

from collections.abc import Sequence

from finestore.rl.curation import SourceQualityResult
from taskcompendium.pipeline.models import (
    Confidence,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
    ReviewVerdict,
)

MAX_SUMMARY_ISSUES = 50


def has_review_issues(verdict: ReviewVerdict) -> bool:
    return (
        verdict.quality in {Quality.ISSUES, Quality.BAD}
        or bool(verdict.defects)
        or verdict.reference_status is ReferenceStatus.CONFLICT
    )


def quality_result(reviews: Sequence[ReviewRecord]) -> SourceQualityResult:
    """Compute informational Result without treating missing observations as clean."""
    reviewed = unavailable = uncertain = 0
    issues = []
    issue_count = 0
    for review in reviews:
        verdict = review.verdict
        if review.status is not ReviewStatus.REVIEWED or verdict is None:
            unavailable += 1
            continue
        if verdict.confidence is Confidence.LOW or (
            verdict.quality is Quality.UNKNOWN and not has_review_issues(verdict)
        ):
            uncertain += 1
            continue
        reviewed += 1
        if has_review_issues(verdict):
            issue_count += 1
            if len(issues) < MAX_SUMMARY_ISSUES:
                issues.append(f"{review.task_id}: {verdict.evidence}")
    recommendation = (
        "Inspect all task-level issue evidence and qualify repairs with renewed QA."
        if issue_count
        else "No repairs indicated by conclusive reviews."
    )
    if unavailable or uncertain:
        recommendation += " Resolve unavailable and uncertain observations before source-wide conclusions."
    return SourceQualityResult(
        reviewed=reviewed,
        issue_count=issue_count,
        issues=issues,
        repair_recommendation=recommendation,
        unavailable_count=unavailable,
        uncertain_count=uncertain,
    )
