# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project current-pin filtration separately from converted and reviewed counts."""

from typing import Any

from finestore.rl.curation import OracleFiltrationEvidence

from .source_counts import SourceCountEvidence


def filtration(declaration: dict[str, Any], counts: SourceCountEvidence | None = None) -> dict[str, Any]:
    """Project exact-input private filtration or no-oracle static count evidence."""
    pending = {
        "example_count": None,
        "state": "pending",
        "admitted_count": None,
        "source_status": declaration["status"],
        "reason": "Filtration in progress: no complete current-pin admission receipt is recorded.",
    }
    raw = declaration.get("oracle_filtration")
    if raw is None:
        return counted_population(declaration, counts, pending)
    evidence = OracleFiltrationEvidence.model_validate(raw)
    release = evidence.filtered_release
    if (
        evidence.oracle_filter_version not in {3, 4}
        or release.model_dump(mode="json") != declaration["current_release"]
        or (release.repo, release.revision) != (declaration["dataset_id"], declaration["dataset_revision"])
        or evidence.admitted_count != declaration["within_source_dedup"]["unique_count"]
        or evidence.admitted_count != declaration["task_count"]
    ):
        return counted_population(declaration, counts, pending)
    if evidence.admitted_count == 0 and declaration["status"] == "Excluded":
        return {
            "example_count": 0,
            "state": "excluded",
            "admitted_count": 0,
            "source_status": declaration["status"],
            "reason": "Complete filtration retained no tasks; the source is excluded, with no admission claimed.",
        }
    return counted_population(declaration, counts, pending)


def counted_population(
    declaration: dict[str, Any], counts: SourceCountEvidence | None, pending: dict[str, Any]
) -> dict[str, Any]:
    if counts is None:
        return pending
    if (
        counts.source_id != declaration["id"]
        or counts.input.model_dump(mode="json") != declaration.get("actor_census_input")
        or (counts.input.dataset_id, counts.input.dataset_revision, counts.input.converter_version)
        != (
            declaration.get("dataset_id"),
            declaration.get("dataset_revision"),
            (declaration.get("pipeline") or {}).get("version"),
        )
        or counts.input.measured_tasks != declaration["within_source_dedup"]["unique_count"]
        or counts.intended_use != declaration.get("intended_use")
    ):
        return {**pending, "reason": "Count evidence does not match the current source input and Use."}
    evidence = counts.model_dump(mode="json")
    if counts.oracle_tasks:
        return {**pending, "state": "pending_oracle", "reason": "Pending oracle filtration.", "count_evidence": evidence}
    if counts.static_blocked_tasks is None:
        return {
            **pending,
            "state": "pending_static",
            "reason": "Current static blocker audit is incomplete.",
            "count_evidence": evidence,
        }
    retained = counts.input.measured_tasks - counts.static_blocked_tasks
    training = counts.intended_use == "train"
    return {
        "state": ("static_retained" if retained else "static_blocked") if training else "eval_only",
        "example_count": retained,
        "admitted_count": retained if training else 0,
        "source_status": declaration["status"],
        "reason": (
            "No oracle: retained population minus unique typed static blockers under the pinned policy."
            if training
            else "Measured EVAL population after static exclusions; excluded from admitted TRAIN totals."
        ),
        "count_evidence": evidence,
    }
