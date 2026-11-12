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


def result(review: dict[str, Any] | None) -> Result | None:
    """Label the reviewed sample; QC and admission status do not reject a source."""
    if not review or not review["reviewed"]:
        return None
    counts = review["verdict_counts"]
    reviewed = review["reviewed"]
    if 10 * counts["good"] >= 9 * reviewed:
        return Result.GOOD
    if 2 * (counts["some_issues"] + counts["bad"]) >= reviewed:
        return Result.BAD
    return Result.SOME_ISSUES


def daytona(declaration: dict[str, Any], limit: int) -> dict[str, Any]:
    """``na`` for a source that provisions no agent machine, else its measured verdict."""
    if "agentic" not in declaration.get("tags", []):
        return {"state": "na", "reason": "Non-agentic tasks provision no sandbox"}
    count = declaration.get("daytona_snapshot_count")
    if count is None:
        return {"state": "unmeasured", "reason": "No production snapshot-identity measurement is recorded"}
    return {
        "state": declaration["daytona_safety"],
        "snapshots": count,
        "limit": declaration.get("daytona_snapshot_threshold") or limit,
        "basis": declaration.get("daytona_count_basis"),
    }
