# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record control results on audited task rows."""

from typing import Any

from taskcompendium.pipeline.models import REJECTING_CHECK_STATUSES, CheckResult
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
