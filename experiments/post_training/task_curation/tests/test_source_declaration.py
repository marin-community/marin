# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated identities, current release pins and actor-only safety."""

from dataclasses import replace

import pytest
from finestore.rl.curation import (
    DatasetRelease,
    DaytonaSafety,
    DaytonaUnsafeReason,
    LexicalContaminationEvidence,
    OversightEvidence,
    OversightLevel,
    SourceDeclaration,
    SourceOrigin,
    SourceQualityResult,
)

from experiments.post_training.task_curation.source import SourceInfo, SourceReference
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence


def test_generated_identity_and_title_survive_catalog_serialization(tmp_path):
    source = number_source(tmp_path / "input", "math")
    info = SourceInfo(
        origin=SourceOrigin.MARIN,
        upstream_dataset="math",
        upstream_name="Math",
        subset="test",
        variant="repaired",
        build_path="Campaign Release",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture task requires no actor.",
    )
    source = replace(source, info=info)
    declaration = SourceDeclaration.model_validate_json(source.declaration.model_dump_json())
    assert declaration.id == "Marin:math/test@repaired"
    assert declaration.display_name == "Math · test · repaired"
    assert declaration.build_path == "Campaign Release"
    assert declaration.current_release.repo == source.dataset.name
    with pytest.raises(ValueError, match="Source ID"):
        replace(info, id="Marin:another")
    with pytest.raises(ValueError, match="Source title"):
        replace(info, title="Math dataset converted from elsewhere")
    with pytest.raises(ValueError, match="Source title"):
        SourceDeclaration.model_validate({**declaration.model_dump(), "display_name": "hand-set title"})


def test_repaired_release_pin_is_separate_from_upstream_url(tmp_path):
    source = number_source(tmp_path / "input")
    pin = SourceReference(
        "open-athena/repaired",
        "1" * 40,
        "https://huggingface.co/datasets/open-athena/repaired/tree/1111111111111111111111111111111111111111",
    )
    source = replace(
        source,
        config=replace(source.config, dataset=pin),
        info=replace(source.info, upstream_url="https://example.org/upstream", lexical_contamination=None),
    )
    declaration = source.declaration
    assert declaration.url == "https://example.org/upstream"
    assert declaration.current_release == DatasetRelease(repo=pin.name, revision=pin.revision, url=pin.url)


def test_actor_safety_does_not_use_grader_failures(tmp_path):
    source = number_source(tmp_path / "input")
    assert source.declaration.daytona_safety is DaytonaSafety.NOT_APPLICABLE
    actor = replace(
        source,
        info=replace(
            source.info,
            tags=("agentic",),
            daytona_snapshot_count=5,
            daytona_machine_slots=5,
            daytona_count_basis="measured",
            daytona_execution_failures=0,
            daytona_roles={"grader": {"daytona": {"machine_slots": 5, "failures": {"host_image_required": 5}}}},
        ),
    )
    assert actor.declaration.daytona_safety is DaytonaSafety.SAFE
    assert (
        replace(
            actor,
            info=replace(
                actor.info,
                daytona_execution_failures=1,
                daytona_unsafe_reasons=(DaytonaUnsafeReason.EXECUTION_REQUIREMENT,),
                daytona_unsafe_details={"execution_requirement": "one actor refused"},
                daytona_unsafe_counts={"execution_requirement": 1},
            ),
        ).declaration.daytona_safety
        is DaytonaSafety.UNSAFE
    )
    with pytest.raises(ValueError, match="actor safety measurement"):
        _ = replace(actor, info=replace(actor.info, daytona_snapshot_count=None)).declaration


def test_result_cannot_be_hand_set_after_roundtrip():
    result = SourceQualityResult(
        reviewed=100, issue_count=50, issues=["Broken references"], repair_recommendation="Repair references"
    )
    assert SourceQualityResult.model_validate_json(result.model_dump_json()).verdict == "bad"
    with pytest.raises(ValueError, match="computed"):
        SourceQualityResult.model_validate({**result.model_dump(), "verdict": "good"})


def test_current_release_follows_the_configured_remediated_pin(tmp_path):
    source = number_source(tmp_path / "input")
    document = decontamination_evidence(source.dataset.name, source.dataset.revision, 2).model_dump(mode="json")
    document.update(contaminated_tasks=1, contaminated_fraction=0.5)
    document["receipt"].update(status="remediated", retained_tasks=1, dropped_tasks=1)
    document["receipt"]["release"].update(
        dataset_id="open-athena/remediated",
        dataset_revision="b" * 40,
        drop_manifest_uri="s3://test/drop.json",
        original_private_dataset_id=source.dataset.name,
        original_private_dataset_revision=source.dataset.revision,
    )
    evidence = LexicalContaminationEvidence.model_validate(document)
    source = replace(source, info=replace(source.info, lexical_contamination=evidence))
    assert source.declaration.current_release.repo == source.dataset.name
    remediated = SourceReference(
        "open-athena/remediated", "b" * 40, "https://huggingface.co/datasets/open-athena/remediated/tree/" + "b" * 40
    )
    source = replace(source, config=replace(source.config, dataset=remediated))
    assert source.declaration.current_release.repo == "open-athena/remediated"
    assert source.declaration.current_release.revision == "b" * 40
    assert source.declaration.dataset_id == source.dataset.name


@pytest.mark.parametrize("change", ["url", "configured_pin"])
def test_current_release_rejects_mismatched_release_identity(tmp_path, change):
    source = number_source(tmp_path / "input")
    pin = source.current_release.model_dump()
    if change == "url":
        with pytest.raises(ValueError, match="URL must match"):
            DatasetRelease.model_validate({**pin, "url": pin["url"].replace("a" * 40, "b" * 40)})
        return
    declared = DatasetRelease(repo=pin["repo"], revision="b" * 40, url=pin["url"].replace("a" * 40, "b" * 40))
    with pytest.raises(ValueError, match="configured dataset pin"):
        _ = replace(source, info=replace(source.info, current_release=declared)).declaration


@pytest.mark.parametrize("field", ["daytona_machine_slots", "daytona_snapshot_count", "daytona_execution_failures"])
def test_na_cannot_hide_an_actor_requirement(tmp_path, field):
    declaration = number_source(tmp_path / "input").declaration.model_dump(mode="json")
    with pytest.raises(ValueError):
        SourceDeclaration.model_validate({**declaration, field: 1})


def test_quick_pipeline_identity_is_available_before_actor_census(tmp_path):
    source = number_source(tmp_path / "input")
    source = replace(source, info=replace(source.info, daytona_machine_slots=None))
    assert source.pipeline_reference.source == source.dataset.name
    assert source.catalog_declaration is None
    with pytest.raises(ValueError, match="actor safety measurement"):
        _ = source.declaration


def test_reply_only_sources_require_oversight_for_selection(tmp_path):
    source = number_source(tmp_path / "input")
    source = replace(source, info=replace(source.info, oversight=None))
    assert source.declaration.status == "Not selectable"
    qualified = replace(
        source,
        info=replace(
            source.info,
            oversight=OversightEvidence(
                level=OversightLevel.SAMPLE, verification="Fixture ingestion and grader controls executed."
            ),
        ),
    )
    assert qualified.declaration.status == "Available"
