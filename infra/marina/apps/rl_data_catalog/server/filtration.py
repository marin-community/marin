# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project current-pin filtration separately from converted and reviewed counts."""

from typing import Any

from finestore.rl.curation import (
    DatasetRelease,
    OracleFiltrationEvidence,
    OraclePendingExclusionReason,
    OracleValidation,
)
from finestore.rl.curation import QualifiedSourceDeclaration as SourceDeclaration

from .admission import (
    require_content_result,
    require_lexical_qualification,
    require_population_admission,
)
from .consumer import consumer_projection
from .licensing import declaration_license_ban
from .source_counts import SourceCountEvidence


def filtration(declaration: dict[str, Any], counts: SourceCountEvidence | None = None) -> dict[str, Any]:
    """Project exact-input private filtration or no-oracle static count evidence."""
    if ban := declaration_license_ban(declaration):
        count = declaration["within_source_dedup"]["unique_count"]
        return {
            "example_count": 0,
            "state": "excluded",
            "admitted_count": 0,
            "population_count": count,
            "excluded_count": count,
            "source_status": "Excluded",
            "reason": f"License ban: {ban.reason} ({ban.issue_url})",
        }
    pending = {
        "example_count": None,
        "state": "pending",
        "admitted_count": None,
        "population_count": declaration["within_source_dedup"]["unique_count"],
        "excluded_count": None,
        "source_status": declaration["status"],
        "reason": "Filtration in progress: no complete current-pin admission receipt is recorded.",
    }
    projected = consumer_projection(declaration)
    if projected.get("consumer_findings") and projected.get("consumer_decision"):
        status = projected["status"]
        if status == "Excluded":
            return {
                "example_count": 0,
                "state": "excluded",
                "admitted_count": 0,
                "population_count": pending["population_count"],
                "excluded_count": pending["population_count"],
                "source_status": status,
                "reason": projected["consumer_decision"]["user_ruling"],
            }
        return {
            **pending,
            "admitted_count": None,
            "source_status": status,
            "reason": projected["consumer_decision"]["user_ruling"],
        }
    raw = declaration.get("oracle_filtration")
    if raw is None:
        return counted_population(declaration, counts, pending)
    # The served row projects the per-task pending-exclusion list past the SQL row
    # budget (the r34-precedent served-row projection). A genuine receipt with
    # infrastructure exclusions always carries the list - the typed model enforces the
    # pair - so a missing list beside a positive infrastructure count identifies the
    # projection. The aggregates and pinned references are authoritative on the served
    # row; the full evidence remains in the catalog snapshot and census receipts.
    infrastructure = raw.get("exclusion_counts", {}).get(OraclePendingExclusionReason.INFRASTRUCTURE.value, 0)
    projected_list = "pending_exclusions" not in raw and infrastructure > 0
    if projected_list:
        filtered_release = DatasetRelease.model_validate(raw["filtered_release"])
        filter_version = raw["oracle_filter_version"]
        admitted = raw["admitted_count"]
        original = raw["original_count"]
    else:
        evidence = OracleFiltrationEvidence.model_validate(raw)
        filtered_release = evidence.filtered_release
        filter_version = evidence.oracle_filter_version
        admitted = evidence.admitted_count
        original = evidence.original_count
    if (
        filter_version not in {3, 4}
        or filtered_release.model_dump(mode="json") != declaration["current_release"]
        or (filtered_release.repo, filtered_release.revision)
        != (declaration["dataset_id"], declaration["dataset_revision"])
        or admitted != declaration["within_source_dedup"]["unique_count"]
        or admitted != declaration["task_count"]
    ):
        return counted_population(declaration, counts, pending)
    if admitted == 0 and declaration["status"] == "Excluded":
        return {
            "example_count": 0,
            "state": "excluded",
            "admitted_count": 0,
            "population_count": original,
            "excluded_count": original,
            "source_status": declaration["status"],
            "reason": "Complete filtration retained no tasks; the source is excluded, with no admission claimed.",
        }
    if (
        filter_version != 4
        or declaration["status"] != "Available"
        or declaration.get("oracle_validation") != OracleValidation.EXECUTED
    ):
        return pending
    try:
        # Served rows also carry display projections, which cannot assert qualification.
        qualified = SourceDeclaration.model_validate(
            {key: value for key, value in declaration.items() if key in SourceDeclaration.model_fields}
        )
        require_population_admission(qualified)
        require_content_result(qualified, None, None)
        if qualified.intended_use is None:
            return pending
        require_lexical_qualification(
            qualified.lexical_contamination, filtered_release.repo, filtered_release.revision, qualified.intended_use
        )
    except ValueError:
        return pending
    # Header totals describe the current retained population; original exclusions stay in the receipt.
    return {
        "example_count": admitted,
        "state": "complete",
        "admitted_count": admitted,
        "population_count": admitted,
        "excluded_count": 0,
        "source_status": declaration["status"],
        "reason": "The current private release is qualified after full-population oracle filtration.",
    }


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
    status = declaration["status"]
    if status != "Available":
        # Headline admitted totals sum admitted_count over sources; only
        # Available sources admit. Excluded and Deprecated populations count
        # zero; Pending (and any other status) keeps its retained count
        # visible as evidence but never enters the admitted total.
        if status in {"Excluded", "Deprecated"}:
            return {
                "state": "excluded",
                "example_count": 0,
                "admitted_count": 0,
                "population_count": counts.input.measured_tasks,
                "excluded_count": counts.input.measured_tasks,
                "source_status": status,
                "reason": (
                    f"Source status {status} removes the population from admission; the current-pin "
                    f"static projection retains {retained} tasks "
                    f"(static blockers {counts.static_blocked_tasks}) as evidence only."
                ),
                "count_evidence": evidence,
            }
        return {
            **pending,
            "count_evidence": evidence,
            "reason": (
                f"Source status {status}: the current-pin static projection retains {retained} tasks "
                "(visible in the count evidence), pending admission; nothing counts as admitted."
            ),
        }
    return {
        "state": "static_retained" if retained else "static_blocked",
        "example_count": retained,
        "admitted_count": retained,
        "population_count": counts.input.measured_tasks,
        "excluded_count": counts.static_blocked_tasks,
        "source_status": status,
        "reason": (
            "No oracle: retained population minus unique typed static blockers under the pinned policy."
            if training
            else "Admitted EVAL tasks after static exclusions; excluded from TRAIN selection."
        ),
        "count_evidence": evidence,
    }
