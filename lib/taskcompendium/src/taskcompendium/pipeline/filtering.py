# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply a policy to retained curation evidence without new model calls."""

from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    Confidence,
    Decision,
    Disposition,
    FilterPolicy,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewStatus,
)

CONFIDENCE_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}


def task_decision(task_id: str, checks: list[CheckResult], review: ReviewRecord, policy: FilterPolicy) -> Decision:
    """Separate quality decisions from incomplete review evidence."""
    failed = [f"check:{check.check}" for check in checks if check.status == CheckStatus.FAIL]
    if failed:
        return Decision(task_id=task_id, disposition=Disposition.REJECT, reasons=failed)
    if review.status != ReviewStatus.REVIEWED or review.verdict is None:
        return Decision(task_id=task_id, disposition=Disposition.DEFER, reasons=[f"review:{review.status}"])
    verdict = review.verdict
    if verdict.quality == Quality.BAD or verdict.reference_status == ReferenceStatus.CONFLICT or verdict.defects:
        reasons = [f"defect:{defect}" for defect in verdict.defects] or ["review:bad_or_conflicting_reference"]
        return Decision(task_id=task_id, disposition=Disposition.REJECT, reasons=reasons)
    if verdict.quality != Quality.GOOD:
        return Decision(task_id=task_id, disposition=Disposition.REJECT, reasons=[f"review:{verdict.quality}"])
    if CONFIDENCE_RANK[verdict.confidence] < CONFIDENCE_RANK[policy.minimum_confidence]:
        return Decision(task_id=task_id, disposition=Disposition.REJECT, reasons=["review:below_confidence_threshold"])
    return Decision(task_id=task_id, disposition=Disposition.KEEP, reasons=[])
