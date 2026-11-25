# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The private-data restriction tier: authoritative tier resolution and the served-content projection.

Every served response, published artifact and quoted excerpt flows through this module
when a source's tier matters. ``declaration_restriction`` resolves the effective tier
(the versioned override registry wins, then the stored declaration tag), and the
redaction helpers replace content with a visible private-data marker while counts,
statuses, hashes and receipts stay visible. Fail closed: a malformed stored tag hides
content rather than exposing it.
"""

import json
from functools import cache
from typing import Any

from finestore.rl.status_taxonomy import DataRestriction
from pydantic import BaseModel, ConfigDict

from . import restriction_registry
from .restriction_registry import restriction_registry_version

PRIVATE_TEXT_MARKER = "Private data: content is not displayed."
"""The visible marker shown wherever a PRIVATE_ALWAYS source's content would appear."""

CONTENT_TEXT_FIELDS = frozenset(
    {
        "explanation",
        "note",
        "notes",
        "prompt",
        "answer",
        "instructions",
        "task_preview",
        "message",
        "expected",
        "reason",
        "verification",
        "quote",
        "excerpt",
    }
)
"""Free-text fields that may quote source content; private rows redact every one."""


class RestrictionOverride(BaseModel):
    """One user ruling that forces a source's restriction tier."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    restriction: DataRestriction
    reason: str
    issue_url: str


@cache
def restriction_overrides() -> dict[str, DataRestriction]:
    registry = json.loads(restriction_registry.RESTRICTION_OVERRIDES)
    entries = [RestrictionOverride.model_validate(item) for item in registry["restrictions"]]
    if len({entry.source_id for entry in entries}) != len(entries):
        raise ValueError("Restriction override registry contains duplicate source IDs")
    return {entry.source_id: entry.restriction for entry in entries}


def restriction_registry_key() -> str:
    """The cache-key component for the selected restriction registry version.

    Served responses are cached by SQL row versions; a registry change is invisible
    to those, so the version joins every cache key that redaction touches and a
    policy reload invalidates the cache.
    """
    return f"restriction:{restriction_registry_version()}"


def declaration_restriction(declaration: dict[str, Any]) -> DataRestriction:
    """The effective tier of one stored declaration.

    A stored tag applies when present — ``Open`` only when the declaration
    predates the tier (the key is absent); an explicit ``null`` or malformed
    stored tag fails closed to ``PRIVATE_ALWAYS`` so erased or broken bytes hide
    content. The override registry can only strengthen the tier: it forces a
    source private, but it can never lower an explicit ``PRIVATE_ALWAYS`` — a
    private source stops being private only through a reviewed declaration
    change in code, never through a registry row.
    """
    if "data_restriction" in declaration:
        raw = declaration["data_restriction"]
        try:
            declared = DataRestriction(raw)
        except ValueError:
            declared = DataRestriction.PRIVATE_ALWAYS
    else:
        declared = DataRestriction.OPEN
    override = restriction_overrides().get(declaration.get("id", ""))
    if declared is DataRestriction.PRIVATE_ALWAYS or override is DataRestriction.PRIVATE_ALWAYS:
        return DataRestriction.PRIVATE_ALWAYS
    return declared


def is_private(declaration: dict[str, Any]) -> bool:
    return declaration_restriction(declaration) is DataRestriction.PRIVATE_ALWAYS


def restriction_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Apply the selected override version without altering archived declarations."""
    return {**declaration, "data_restriction": declaration_restriction(declaration).value}


def private_content_marker() -> dict[str, Any]:
    """The visible marker object that replaces content in served rows."""
    return {
        "data_restriction": DataRestriction.PRIVATE_ALWAYS.value,
        "private": True,
        "content": None,
        "marker": PRIVATE_TEXT_MARKER,
    }


def redact_text(text: Any) -> Any:
    """Replace one content-bearing string with the visible marker."""
    return PRIVATE_TEXT_MARKER if text else text


def redact_mapping(payload: dict[str, Any], fields: frozenset[str] = CONTENT_TEXT_FIELDS) -> dict[str, Any]:
    """Redact the named free-text fields of one payload; counts, hashes and statuses stay."""
    return {
        key: redact_text(value) if key in fields and isinstance(value, str) else value for key, value in payload.items()
    }


def redact_advisories(advisories: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Redact each repair advisory's free-text reason; its signal and receipt stay."""
    if not advisories:
        return advisories
    return [{**advisory, "reason": redact_text(advisory.get("reason"))} for advisory in advisories]


def require_public_data(restriction: DataRestriction, source_id: str = "", destination: str = "") -> None:
    """Refuse publishing or exporting a PRIVATE_ALWAYS source outside its private store."""
    if restriction is DataRestriction.PRIVATE_ALWAYS:
        where = f" to {destination}" if destination else ""
        who = f" {source_id}" if source_id else ""
        raise ValueError(f"Publication refused: PRIVATE_ALWAYS source{who} never leaves its private store{where}")


def require_public_declaration(declaration: dict[str, Any], destination: str = "") -> None:
    """Publication guard over one stored or authored declaration."""
    require_public_data(declaration_restriction(declaration), declaration.get("id", ""), destination)


PRIVATE_DECLARATION_FIELDS = frozenset(
    {
        "id",
        "display_name",
        "origin",
        "build_path",
        "upstream_dataset",
        "upstream_name",
        "subset",
        "variant",
        "current_release",
        "family",
        "tags",
        "task_count",
        "dataset_id",
        "dataset_revision",
        "url",
        "task_format",
        "stage",
        "type",
        "turns",
        "is_benchmark",
        "intended_use",
        "data_restriction",
        "kind",
        "status",
        "pipeline",
        "license_receipt",
        "daytona_safety",
        "daytona_snapshot_count",
        "daytona_snapshot_threshold",
        "daytona_count_basis",
        "daytona_machine_slots",
        "daytona_execution_failures",
        "daytona_unsafe_reasons",
        "daytona_unsafe_counts",
        "oracle_validation",
        "oversight",
        "quality_result_finalization",
        "quality_result_waiver",
        "derived_agentic",
        "derived_turns",
        "derived_reply_only",
        "derived_workspace_graded",
        "derived_contract_receipt",
        "grader_type",
        "grader_type_counts",
        "judge_rubric_counts",
        "judge_reference_count",
        "judge_no_reference_count",
        "grading_environment",
        "actor_census_input",
        "consumer_runtime_commit",
        "consumer_receipt",
        "consumer_decision",
        "consumer_requirements",
        "consumer_findings",
        "oracle_filtration",
        "no_oracle_executability_policy",
        "no_oracle_static_retention",
        "lexical_contamination",
        "within_source_dedup",
        "exclusion_claims",
        "updated_at",
    }
)
"""Declaration fields a PRIVATE_ALWAYS source may still serve: identity, counts,
statuses, hashes and receipts. Everything else is dropped, including free-text
evidence (quality_result issues and repair recommendations, independent-QC method
prose, integrity contract examples, converted-example failure reasons) that could
quote its content."""


def private_declaration_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """The served declaration for a PRIVATE_ALWAYS source: marker notes, no content-bearing prose.

    Exclusion-claim evidence keeps its reason, kind and URI (a ruling pointer) and loses
    its quoted message text. ``notes`` and ``recipe_qa_history`` never serve.
    """
    projected = {
        key: declaration[key] for key in PRIVATE_DECLARATION_FIELDS if key in declaration and key != "data_restriction"
    }
    projected["data_restriction"] = declaration_restriction(declaration).value
    projected["notes"] = PRIVATE_TEXT_MARKER
    claims = []
    for claim in declaration.get("exclusion_claims") or []:
        claim = dict(claim)
        evidence = {**claim.get("evidence", {}), "message": PRIVATE_TEXT_MARKER}
        evidence.pop("expected", None)
        evidence.pop("quote", None)
        claim["evidence"] = evidence
        claims.append(claim)
    projected["exclusion_claims"] = claims
    return projected
