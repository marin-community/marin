# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated identities, current release pins and actor-only safety."""

from dataclasses import replace

import pytest
from finestore.rl.curation import (
    DatasetRelease,
    DaytonaSafety,
    DaytonaUnsafeReason,
    SourceDeclaration,
    SourceOrigin,
    SourceQualityResult,
)

from experiments.post_training.task_curation.source import SourceInfo, SourceReference
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


def test_generated_identity_and_title_survive_catalog_serialization(tmp_path):
    source = number_source(tmp_path / "input", "math")
    info = SourceInfo(
        id="Marin:math/test@repaired",
        origin=SourceOrigin.MARIN,
        upstream_dataset="math",
        upstream_name="Math",
        subset="test",
        variant="repaired",
        build_path="Campaign Release",
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
    assert replace(actor, info=replace(actor.info, daytona_snapshot_count=None)).declaration.daytona_safety is None


def test_result_cannot_be_hand_set_after_roundtrip():
    result = SourceQualityResult(
        reviewed=100, issue_count=50, issues=["Broken references"], repair_recommendation="Repair references"
    )
    assert SourceQualityResult.model_validate_json(result.model_dump_json()).verdict == "bad"
    with pytest.raises(ValueError, match="computed"):
        SourceQualityResult.model_validate({**result.model_dump(), "verdict": "good"})
