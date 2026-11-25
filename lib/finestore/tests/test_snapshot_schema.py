# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior contracts for the versioned snapshot reader (the pre-stage and pre-lineage forms).

The reader lets the current tip re-read every served snapshot generation: the current
union forms pass through, the pre-stage full form (Pending rows without a computed
stage) takes its stage from the evidence-computing callback, and the pre-lineage
forms read as typed records with every legacy field preserved verbatim.
"""

import pydantic
import pytest
from finestore.rl.curation import (
    PreLineageSnapshotDeclaration,
    QualifiedSourceDeclaration,
    SnapshotSource,
    normalize_snapshot_declaration,
    snapshot_declaration_generation,
    source_display_name,
)
from finestore.rl.status_taxonomy import ExclusionReason, PendingStage


def _qualified_document(source_id: str, *, status: str = "Available") -> dict:
    """A current-schema full-form declaration document (the union's first member)."""
    return {
        "id": source_id,
        "display_name": source_display_name(source_id),
        "origin": "Marin",
        "build_path": "fixture",
        "upstream_dataset": "fixture",
        "upstream_name": "fixture",
        "current_release": {"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
        "family": "fixture",
        "tags": [],
        "task_count": 2,
        "within_source_dedup": {
            "raw_count": 2,
            "unique_count": 2,
            "within_source_duplicates": 0,
            "source_dataset_id": "fixture",
            "source_revision": "1",
            "converter_revision": "1",
            "converter_version": "1",
            "drop_manifest_uri": "s3://fixture/drops.json",
            "evidence_url": "s3://fixture/dedup.json",
        },
        "notes": "",
        "dataset_id": "fixture",
        "dataset_revision": "1",
        "url": None,
        "type": "RLVR",
        "turns": "Single-turn",
        "is_benchmark": False,
        "kind": "Dataset",
        "status": status,
        "intended_use": "train",
        "pipeline": {"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
        "daytona_safety": "not_applicable",
        "daytona_count_basis": "Fixture contains no actor environment.",
        "daytona_snapshot_count": 0,
        "daytona_machine_slots": 0,
        "daytona_execution_failures": 0,
        "actor_census_input": {
            "dataset_id": "fixture",
            "dataset_revision": "1",
            "converter_version": "1",
            "converter_revision": "1",
            "measured_tasks": 2,
            "task_specs_sha256": "a" * 64,
            "evidence_url": "s3://fixture/population/.artifact.json",
        },
    }


def _thin_document(source_id: str, origin: str = "Evalchemy") -> dict:
    """A pre-lineage thin declaration (the November 12-era form)."""
    return {
        "id": source_id,
        "name": "fixture-thin",
        "display_name": "Fixture thin source",
        "origin": origin,
        "family": "fixture",
        "tags": [],
        "task_count": 5,
        "notes": "",
        "dataset_id": "fixture-thin",
        "dataset_revision": "1",
        "url": None,
        "type": "RLVR",
        "turns": "Single-turn",
        "is_benchmark": False,
        "kind": "Dataset",
        "status": "Available",
        "pipeline": None,
    }


def _stage_of(source_id: str) -> PendingStage:
    return PendingStage.MEASUREMENTS_PENDING


def test_generation_detection_names_each_form():
    assert snapshot_declaration_generation(_qualified_document("Marin:fixture")) == "qualified-v2"
    current_pending = {**_qualified_document("Marin:fixture", status="Pending"), "stage": "Measurements pending"}
    assert snapshot_declaration_generation(current_pending) == "qualified-v2"
    pre_stage = {**_qualified_document("Marin:fixture", status="Pending"), "name": "fixture"}
    assert snapshot_declaration_generation(pre_stage) == "qualified-v1"
    # Any full-form row still carrying the removed free-text name field is pre-name.
    assert snapshot_declaration_generation({**_qualified_document("Marin:fixture"), "name": "fixture"}) == (
        "qualified-v1"
    )
    assert snapshot_declaration_generation(_thin_document("Evalchemy:fixture")) == "pre-lineage"


def test_current_form_passes_through_without_a_callback():
    document = _qualified_document("Marin:fixture")
    assert normalize_snapshot_declaration(document) == document
    thin = _thin_document("Evalchemy:fixture")
    normalized = normalize_snapshot_declaration(thin)
    assert normalized == thin


def test_pre_stage_form_converts_only_the_named_fields():
    document = {**_qualified_document("Marin:fixture", status="Pending"), "name": "fixture-free-text"}
    with pytest.raises(ValueError, match="supply pending_stage_of"):
        normalize_snapshot_declaration(document)
    normalized = normalize_snapshot_declaration(document, pending_stage_of=_stage_of)
    assert "name" not in normalized
    assert normalized["display_name"] == source_display_name("Marin:fixture")
    assert normalized["stage"] == PendingStage.MEASUREMENTS_PENDING.value
    assert QualifiedSourceDeclaration.model_validate(normalized).stage is PendingStage.MEASUREMENTS_PENDING


@pytest.mark.parametrize("status,stage", [("Available", None), ("Pending", PendingStage.MEASUREMENTS_PENDING.value)])
def test_removed_name_preserves_population_and_existing_stage_without_receipt_callback(status, stage):
    document = {**_qualified_document("Marin:fixture", status=status), "name": "fixture-free-text"}
    if stage is not None:
        document["stage"] = stage
    normalized = normalize_snapshot_declaration(document)
    assert "name" not in normalized
    for key in ("id", "current_release", "pipeline", "actor_census_input", "within_source_dedup"):
        assert normalized.get(key) == document.get(key)
    assert QualifiedSourceDeclaration.model_validate(normalized).stage == stage


def test_pre_ruling_exclusion_reason_becomes_unspecified_legacy_verbatim():
    document = _qualified_document("Marin:fixture", status="Excluded")
    legacy_claim = {
        "reason": "superseded-by-replacement",
        "evidence": {"kind": "note", "message": "Deprecated: replaced by the hardened variant."},
    }
    document["exclusion_claims"] = [legacy_claim]
    normalized = normalize_snapshot_declaration(document, pending_stage_of=_stage_of)
    claim = normalized["exclusion_claims"][0]
    assert claim["reason"] == ExclusionReason.UNSPECIFIED_LEGACY.value
    assert claim["evidence"]["message"].startswith("[legacy reason: superseded-by-replacement] ")
    assert "replaced by the hardened variant" in claim["evidence"]["message"]
    assert QualifiedSourceDeclaration.model_validate(normalized).exclusion_claims[0].reason is (
        ExclusionReason.UNSPECIFIED_LEGACY
    )


def test_snapshot_source_reads_each_generation(tmp_path):
    # The current form validates directly.
    current = SnapshotSource.model_validate(
        {
            "snapshot_id": "s",
            "id": "Marin:fixture",
            "declaration": _qualified_document("Marin:fixture"),
            "assessment_key": None,
            "latest_attempt_key": None,
            "latest_attempt_status": "complete",
            "reason": None,
        }
    )
    assert isinstance(current.declaration, QualifiedSourceDeclaration)
    # The pre-stage form reads with the callback and refuses without it.
    pre_stage = {
        "snapshot_id": "s",
        "id": "Marin:fixture",
        "declaration": {**_qualified_document("Marin:fixture", status="Pending"), "stage": None},
        "assessment_key": None,
        "latest_attempt_key": None,
        "latest_attempt_status": "complete",
        "reason": None,
    }
    pre_stage["declaration"].pop("stage")
    with pytest.raises(pydantic.ValidationError, match="supply pending_stage_of"):
        SnapshotSource.model_validate(pre_stage, context={"pending_stage_of": None})
    staged = SnapshotSource.model_validate(pre_stage, context={"pending_stage_of": _stage_of})
    assert staged.declaration.stage is PendingStage.MEASUREMENTS_PENDING
    # The pre-lineage form reads as a typed record with the legacy fields verbatim.
    thin = {
        "snapshot_id": "s",
        "id": "Evalchemy:fixture",
        "declaration": {**_thin_document("Evalchemy:fixture"), "oversight": None, "oracle_validation": "unmeasured"},
        "assessment_key": None,
        "latest_attempt_key": None,
        "latest_attempt_status": "complete",
        "reason": None,
    }
    legacy = SnapshotSource.model_validate(thin)
    assert isinstance(legacy.declaration, PreLineageSnapshotDeclaration)
    assert legacy.declaration.legacy_fields == {"oversight": None, "oracle_validation": "unmeasured"}


def test_pre_lineage_model_preserves_every_legacy_field_verbatim():
    thin = _thin_document("Snowball Smartie RL:fixture", origin="Snowball Smartie RL")
    legacy = PreLineageSnapshotDeclaration.model_validate({**thin, "grading_environment": {"assessed": 1}})
    assert legacy.origin == "Snowball Smartie RL"
    assert legacy.status == "Available"
    assert legacy.legacy_fields == {"grading_environment": {"assessed": 1}}
    document = legacy.model_dump(mode="json")
    assert document["legacy_fields"]["grading_environment"] == {"assessed": 1}


def test_invalid_current_generation_refuses_without_guessing():
    """A row that claims the current schema but fails it is refused, never converted."""
    document = _qualified_document("Marin:fixture", status="Excluded")
    document.pop("exclusion_claims", None)
    with pytest.raises(ValueError, match="refusing to guess a conversion"):
        normalize_snapshot_declaration(document)
