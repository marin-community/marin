# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality level, result and Daytona label of one catalog source.

The rules are the ones in docs/references/task-curation.md ("Quality and result").
They read only loaded catalog records, so every Atlas view labels a source the same way.
"""

from enum import StrEnum
from typing import Any


class Quality(StrEnum):
    """How much of a source executed QA examined; independent upstream QC counts as QUICK."""

    QUICK = "QUICK"
    SAMPLE = "SAMPLE"
    FULL = "FULL"


class Result(StrEnum):
    """What that QA found."""

    GOOD = "GOOD"
    SOME_ISSUES = "SOME_ISSUES"
    BAD = "BAD"


ORDER = {Quality.QUICK: 0, Quality.SAMPLE: 1, Quality.FULL: 2}
OVERSIGHT_QUALITY = {"sample": Quality.SAMPLE, "full": Quality.FULL}
# Per-task validation covered every eligible task; a passed sample let the rest skip validation.
CONTROL_QUALITY = {"validated": Quality.FULL, "passed": Quality.SAMPLE}
FAILED_CONTROLS = {"rejected"}
GRADE_RESULT = {"good": Result.GOOD, "some_issues": Result.SOME_ISSUES, "bad": Result.BAD}


def controls_status(curation: dict[str, Any] | None) -> str | None:
    verification = (curation or {}).get("verification") or {}
    return verification.get("status")


def quality(declaration: dict[str, Any], curation: dict[str, Any] | None) -> Quality:
    """The stronger of the declared oversight level and the executed grader controls."""
    levels = [Quality.QUICK]
    if declaration.get("oversight") in OVERSIGHT_QUALITY:
        levels.append(OVERSIGHT_QUALITY[declaration["oversight"]])
    status = controls_status(curation)
    if (curation or {}).get("mode") in ("sample", "full") and status is not None and status in CONTROL_QUALITY:
        levels.append(CONTROL_QUALITY[status])
    return max(levels, key=ORDER.__getitem__)


def result(
    declaration: dict[str, Any], curation: dict[str, Any] | None, grade: str | None, level: Quality
) -> Result | None:
    """BAD on any recorded defect; otherwise the sampled grade; otherwise passed QA or upstream QC.

    ``grade`` is the sampled assessment grade of the admitted cohort. An ``unknown``
    grade, a review that could not confirm the sample, counts as SOME_ISSUES once
    executed QA exists. ``None`` means no QA evidence exists.
    """
    independent = (declaration.get("independent_qc") or {}).get("verdict")
    if declaration.get("status") == "Excluded" or controls_status(curation) in FAILED_CONTROLS or grade == "bad":
        return Result.BAD
    if grade in GRADE_RESULT:
        return GRADE_RESULT[grade]
    if level is not Quality.QUICK:
        return Result.SOME_ISSUES if grade == "unknown" else Result.GOOD
    return GRADE_RESULT.get(independent)


def daytona(declaration: dict[str, Any], limit: int) -> dict[str, Any]:
    """``na`` for a source that provisions no agent machine, else its measured verdict."""
    if "agentic" not in declaration.get("tags", []):
        return {"state": "na", "reason": "Non-agentic tasks provision no sandbox"}
    count = declaration.get("daytona_snapshot_count")
    if count == 0:
        return {"state": "na", "reason": "Measured zero machine snapshots; tasks provision no sandbox"}
    if count is None:
        return {"state": "unmeasured", "reason": "No production snapshot-identity measurement is recorded"}
    return {
        "state": declaration["daytona_safety"],
        "snapshots": count,
        "limit": declaration.get("daytona_snapshot_threshold") or limit,
        "basis": declaration.get("daytona_count_basis"),
    }
