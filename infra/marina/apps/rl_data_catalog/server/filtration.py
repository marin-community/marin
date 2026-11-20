# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project current-pin filtration separately from converted and reviewed counts."""

from typing import Any

from finestore.rl.curation import OracleFiltrationEvidence, OracleValidation
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
    if projected["consumer_blockers"] and declaration["status"] in {"Available", "Pending"}:
        return {
            **pending,
            "admitted_count": 0,
            "source_status": "Pending",
            "reason": ", ".join(projected["consumer_blockers"]),
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
            "population_count": evidence.original_count,
            "excluded_count": evidence.original_count,
            "source_status": declaration["status"],
            "reason": "Complete filtration retained no tasks; the source is excluded, with no admission claimed.",
        }
    if (
        evidence.oracle_filter_version != 4
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
            qualified.lexical_contamination, release.repo, release.revision, qualified.intended_use
        )
    except ValueError:
        return pending
    # Header totals describe the current retained population; original exclusions stay in the receipt.
    return {
        "example_count": evidence.admitted_count,
        "state": "complete",
        "admitted_count": evidence.admitted_count,
        "population_count": evidence.admitted_count,
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
