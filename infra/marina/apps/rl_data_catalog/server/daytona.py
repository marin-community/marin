# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize actor-safety evidence for the retained Atlas renderer."""

from typing import Any


def daytona(declaration: dict[str, Any], limit: int) -> dict[str, Any]:
    """Return the v75 rendering vocabulary while retaining typed measurement evidence."""
    tags = declaration.get("tags", [])
    safety = declaration.get("daytona_safety")
    if declaration.get("id", "").startswith("MarinSkyRL:") and "agentic" in tags:
        return {
            "state": "na",
            "exempt": True,
            "reason": "Grandfathered blend; Daytona safety is exempt and no verdict is asserted",
        }
    slots = declaration.get("daytona_machine_slots")
    if safety == "not_applicable" or (slots == 0 and safety != "unsafe"):
        return {"state": "na", "reason": "A complete census measured no actor machine"}
    if "agentic" not in tags and slots is None and safety != "unsafe":
        return {"state": "na", "reason": "Non-agentic tasks provision no sandbox"}
    count = declaration.get("daytona_snapshot_count")
    if count is None:
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
