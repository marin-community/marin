# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply a policy to retained curation evidence without new model calls."""

from taskcompendium.pipeline.models import (
    REJECTING_CHECK_STATUSES,
    CheckResult,
    Decision,
    Disposition,
)


def task_decision(task_id: str, checks: list[CheckResult]) -> Decision:
    """Apply mechanical checks; sampled QC opinions never reject tasks."""
    failed = [f"check:{check.check}" for check in checks if check.status in REJECTING_CHECK_STATUSES]
    if failed:
        return Decision(task_id=task_id, disposition=Disposition.REJECT, reasons=failed)
    return Decision(task_id=task_id, disposition=Disposition.KEEP, reasons=[])
