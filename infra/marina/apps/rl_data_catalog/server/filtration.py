# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project current-pin filtration separately from converted and reviewed counts."""

from typing import Any

from finestore.rl.curation import OracleFiltrationEvidence


def filtration(declaration: dict[str, Any]) -> dict[str, Any]:
    """Expose a count only when the recorded private output matches the current pin."""
    pending = {
        "state": "pending",
        "admitted_count": None,
        "source_status": declaration["status"],
        "reason": "Filtration in progress: no complete current-pin admission receipt is recorded.",
    }
    raw = declaration.get("oracle_filtration")
    if raw is None:
        return pending
    evidence = OracleFiltrationEvidence.model_validate(raw)
    release = evidence.filtered_release
    if (
        evidence.oracle_filter_version not in {3, 4}
        or release.model_dump(mode="json") != declaration["current_release"]
        or (release.repo, release.revision) != (declaration["dataset_id"], declaration["dataset_revision"])
        or evidence.admitted_count != declaration["within_source_dedup"]["unique_count"]
        or evidence.admitted_count != declaration["task_count"]
    ):
        return {**pending, "reason": "The recorded filtration does not establish the current population."}
    if evidence.admitted_count == 0 and declaration["status"] == "Excluded":
        return {
            "state": "excluded",
            "admitted_count": 0,
            "source_status": declaration["status"],
            "reason": "Complete filtration retained no tasks; the source is excluded, with no admission claimed.",
        }
    return {**pending, "reason": "Filtered output is recorded; current production admission remains pending."}
