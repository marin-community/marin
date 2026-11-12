# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize actor-safety evidence for the retained Atlas renderer."""

from typing import Any

from finestore.rl.source_policy import GRANDFATHERED_MARINSKYRL_SOURCES


def daytona(declaration: dict[str, Any], limit: int) -> dict[str, Any]:
    """Return the v75 rendering vocabulary while retaining typed measurement evidence."""
    safety = declaration.get("daytona_safety")
    if declaration.get("id") in GRANDFATHERED_MARINSKYRL_SOURCES:
        return {
            "state": "na",
            "exempt": True,
            "reason": "Grandfathered blend; Daytona safety is exempt and no verdict is asserted",
        }
    slots = declaration.get("daytona_machine_slots")
    if (
        slots == 0
        and declaration.get("daytona_snapshot_count") == 0
        and declaration.get("daytona_execution_failures") == 0
        and (declaration.get("daytona_count_basis") or "").strip()
        and safety != "unsafe"
    ):
        return {"state": "na", "reason": "A complete census measured no actor machine"}
    count = declaration.get("daytona_snapshot_count")
    if count is None or safety not in {"safe", "unsafe"}:
        return {"state": "unmeasured", "reason": "No production snapshot-identity measurement is recorded"}
    label = {
        "state": safety,
        "snapshots": count,
        "limit": declaration.get("daytona_snapshot_threshold") or limit,
        "basis": declaration.get("daytona_count_basis"),
    }
    if safety == "unsafe":
        label["unsafe_reasons"] = declaration.get("daytona_unsafe_reasons") or []
        label["unsafe_details"] = declaration.get("daytona_unsafe_details") or {}
        label["unsafe_counts"] = declaration.get("daytona_unsafe_counts") or {}
    return label
