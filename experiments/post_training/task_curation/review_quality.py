# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize conclusive review outcomes without discarding uncertain observations."""

from collections.abc import Sequence

from finestore.rl.curation import SourceQualityResult
from taskcompendium.pipeline.models import ReviewRecord, ReviewStatus, has_review_issues, is_conclusive

MAX_SUMMARY_ISSUES = 50


def quality_result(reviews: Sequence[ReviewRecord]) -> SourceQualityResult:
    """Compute informational Result without treating missing observations as clean."""
    reviewed = unavailable = uncertain = 0
    issues = []
    issue_count = 0
    for review in reviews:
        if is_conclusive(review):
            reviewed += 1
            verdict = review.verdict
            assert verdict is not None
            if has_review_issues(verdict):
                issue_count += 1
                if len(issues) < MAX_SUMMARY_ISSUES:
                    issues.append(f"{review.task_id}: {verdict.evidence}")
            continue
        if review.status is ReviewStatus.REVIEWED and review.verdict is not None:
            uncertain += 1
        else:
            unavailable += 1
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
