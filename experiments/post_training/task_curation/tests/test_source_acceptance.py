# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic sources are admitted only in an accepted task format."""

from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import DatasetRelease, OversightEvidence, OversightLevel
from finestore.rl.status_taxonomy import DataRestriction, SourceStatus

from experiments.post_training.task_curation import sources as source_catalog
from experiments.post_training.task_curation.datasets import native_releases
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    TaskFormat,
)
from experiments.post_training.task_curation.source_policy import GRANDFATHERED_MARINSKYRL_SOURCES
from experiments.post_training.task_curation.sources import all_sources, require_accepted_task_format, runnable_sources
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


@pytest.mark.parametrize(
    "source_id,dataset",
    [
        ("marcusm117:ProofNet-Verified", "marcusm117/ProofNet-Verified"),
        ("project-numina:NuminaMath-LEAN", "project-numina/NuminaMath-LEAN"),
    ],
)
def test_new_lean_sources_wait_for_measurement_before_entering_the_atlas_catalog(monkeypatch, source_id, dataset):
    monkeypatch.setattr(native_releases, "read_registry_json", lambda path: {"entries": []})
    pending = {source.info.id: source for source in source_catalog.PENDING_MEASUREMENT_SOURCES}
    source = pending[source_id]
    assert source.dataset.name == dataset
    assert source.info.actor_census_input is None
    assert source.info.within_source_dedup is None
    assert source.info.lexical_contamination is None
    assert source.info.id not in {source.info.id for source in source_catalog.configured_sources()}


def _agentic(task_format: TaskFormat | None, source_id: str = "Marin:agentic") -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id=source_id,
            upstream_dataset=(source_id).partition(":")[2],
            upstream_name=(source_id).partition(":")[2],
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("agentic",),
            task_format=task_format,
        )
    )


@pytest.mark.parametrize(
    "task_format, source_id, error",
    [
        (TaskFormat.MARINSKYRL, "Marin:a-new-blend", "not grandfathered"),
        (None, "Marin:agentic", "declares no task format"),
    ],
)
def test_unaccepted_agentic_formats_are_rejected(task_format, source_id, error):
    with pytest.raises(ValueError, match=error):
        require_accepted_task_format(_agentic(task_format, source_id))


def test_pipeline_less_pointers_are_rejected():
    with pytest.raises(ValueError, match="no registered conversion pipeline"):
        _ = _agentic(TaskFormat.TASKCOMPENDIUM).declaration


@pytest.mark.parametrize("missing", ["count", "basis", "oversight", "verification"])
def test_unqualified_agentic_pipeline_stays_visible_but_cannot_be_selected(monkeypatch, missing):
    info = replace(
        _agentic(TaskFormat.TASKCOMPENDIUM).info,
        within_source_dedup=number_source(Path("unused")).info.within_source_dedup,
        daytona_snapshot_count=1,
        daytona_machine_slots=1,
        daytona_execution_failures=0,
        daytona_count_basis="production snapshot identities over pinned tasks",
        oversight=OversightEvidence(
            level=OversightLevel.SAMPLE, campaign="fixture", verification="20/20 negative controls"
        ),
    )
    if missing == "count":
        info = replace(info, daytona_snapshot_count=None)
    elif missing == "basis":
        info = replace(info, daytona_count_basis=None)
    elif missing == "oversight":
        info = replace(info, oversight=None)
    else:
        info = replace(
            info, oversight=OversightEvidence(level=OversightLevel.SAMPLE, campaign="fixture", verification="")
        )
    source = replace(number_source(Path("unused")), info=info)
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_sources", lambda: {info.id: source})
    if missing in {"count", "basis"}:
        with pytest.raises(ValueError, match="actor safety measurement"):
            _ = source.declaration
        return
    assert source.name not in runnable_sources()
    assert source.declaration.status == "Excluded"


@pytest.mark.data_integration
def test_existing_internal_blend_selection_does_not_require_new_audit_evidence(monkeypatch):
    source = next(source for source in all_sources().values() if source.is_grandfathered_blend)
    source_id = source.info.id
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_sources", lambda: {source_id: source})
    assert runnable_sources()[source.name] is source
    assert source.declaration.status == "Available"
    assert source.declaration.daytona_exemption
    assert source.declaration.daytona_safety is None


@pytest.mark.data_integration
def test_verified_retained_legacy_subset_inherits_admission_but_not_daytona_exemption(monkeypatch):
    source = next(
        source
        for source in all_sources().values()
        if source.info.id in GRANDFATHERED_MARINSKYRL_SOURCES and source.info.inherited_admission is not None
    )
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_sources", lambda: {source.info.id: source})
    assert runnable_sources()[source.name] is source
    assert source.admission_blockers == ()
    assert source.declaration.daytona_exemption is None
    assert source.declaration.daytona_safety is not None
    assert source.declaration.status == "Available"


@pytest.mark.parametrize("missing", ["verified_pin", "private_receipt", "fresh_actor"])
@pytest.mark.data_integration
def test_retained_legacy_admission_cannot_bypass_pin_privacy_or_fresh_actor_gates(monkeypatch, missing):
    source = next(
        source
        for source in all_sources().values()
        if source.info.id in GRANDFATHERED_MARINSKYRL_SOURCES and source.info.inherited_admission is not None
    )
    if missing == "verified_pin":
        revision = "f" * 40
        release = DatasetRelease(
            repo=source.dataset.name,
            revision=revision,
            url=f"https://huggingface.co/datasets/{source.dataset.name}/tree/{revision}",
        )
        source = replace(
            source, config=replace(source.config, release=release), info=replace(source.info, current_release=release)
        )
    elif missing == "private_receipt":
        source = replace(source, info=replace(source.info, lexical_contamination=None))
    else:
        source = replace(source, info=replace(source.info, actor_census_input=None))
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_sources", lambda: {source.info.id: source})
    assert source.admission_blockers
    assert source.name not in runnable_sources()


def test_grandfathered_execution_exemption_does_not_bypass_grader_census(tmp_path):
    source = number_source(tmp_path / "input")
    source_id = sorted(GRANDFATHERED_MARINSKYRL_SOURCES)[0]
    source = replace(
        source,
        info=replace(
            source.info,
            id=source_id,
            origin=SourceOrigin.NEMOTRON,
            upstream_dataset=source_id.partition(":")[2],
            task_format=TaskFormat.MARINSKYRL,
        ),
    )
    assert source.is_grandfathered_blend
    assert any("census" in reason for reason in source.admission_blockers)


@pytest.mark.data_integration
def test_every_registered_source_declares_catalog_metadata():
    """Missing census evidence remains pending when catalog metadata is declared."""
    for source in all_sources().values():
        declaration = source.declaration
        assert declaration.id == source.info.id


@pytest.mark.data_integration
def test_nemotron_gym_math_v5_is_deprecated_in_favor_of_the_qa_repair():
    """The user ruled math-v5 Deprecated in favor of the repaired math_gym-qa pin.

    Deprecated stays trainable on explicit request: the tag carries no exclusion
    claim, so the row resolves to SourceStatus.DEPRECATED and the default
    training selections skip it.
    """
    source = all_sources()["TaskTrove:laion__nemotron-gym-math-v5"]
    assert source.status is SourceStatus.DEPRECATED
    assert "deprecated" in source.info.tags
    assert source.exclusion_claims == ()
    assert "TaskTrove:tasktrove-math_gym-qa" in source.info.notes
