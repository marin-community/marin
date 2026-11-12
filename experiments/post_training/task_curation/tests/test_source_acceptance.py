# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic sources are admitted only in an accepted task format."""

from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import OversightEvidence, OversightLevel

from experiments.post_training.task_curation.source import (
    GRANDFATHERED_MARINSKYRL_SOURCES,
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    TaskFormat,
)
from experiments.post_training.task_curation.sources import all_sources, require_accepted_task_format, runnable_sources
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


def _agentic(task_format: TaskFormat | None, source_id: str = "Marin:agentic") -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            id=source_id,
            upstream_dataset=(source_id).partition(":")[2],
            upstream_name=(source_id).partition(":")[2],
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("agentic",),
            task_format=task_format,
        )
    )


def test_every_runnable_agentic_source_ingests_a_task_directory_format():
    """No agentic source reaches selection without TaskCompendium or Harbor."""
    agentic = [source for source in all_sources().values() if "agentic" in source.info.tags]
    runnable = [source for source in agentic if source.pipeline is not None]

    missing = sorted(source.info.id for source in runnable if source.info.task_format is None)
    assert missing == []
    accepted = {TaskFormat.TASKCOMPENDIUM, TaskFormat.HARBOR}
    internal = {source.info.id for source in runnable if source.info.task_format is TaskFormat.MARINSKYRL}
    assert internal == set(GRANDFATHERED_MARINSKYRL_SOURCES)
    assert all(
        source.info.task_format in accepted or source.info.id in GRANDFATHERED_MARINSKYRL_SOURCES for source in runnable
    )


def test_the_grandfather_list_holds_only_runnable_agentic_marinskyrl_sources():
    """The frozen list cannot hide a non-agentic or pipeline-less source."""
    sources = all_sources()
    for source_id in GRANDFATHERED_MARINSKYRL_SOURCES:
        source = sources[source_id]
        assert source.info.origin is SourceOrigin.NEMOTRON
        assert "agentic" in source.info.tags
        assert source.pipeline is not None
        assert source.info.task_format is TaskFormat.MARINSKYRL


def test_a_new_internal_format_agentic_source_is_rejected():
    """A MarinSkyRL-format source outside the frozen list cannot be selected."""
    with pytest.raises(ValueError, match="not grandfathered"):
        require_accepted_task_format(_agentic(TaskFormat.MARINSKYRL, "Marin:a-new-blend"))


def test_an_agentic_source_without_a_declared_format_is_rejected():
    with pytest.raises(ValueError, match="declares no task format"):
        require_accepted_task_format(_agentic(None))


@pytest.mark.parametrize("task_format", [TaskFormat.TASKCOMPENDIUM, TaskFormat.HARBOR])
def test_task_directory_formats_pass_and_non_agentic_sources_need_none(task_format):
    require_accepted_task_format(_agentic(task_format))
    require_accepted_task_format(
        RlDataSource(
            info=SourceInfo(
                id="Marin:math",
                upstream_dataset=("Marin:math").partition(":")[2],
                upstream_name=("Marin:math").partition(":")[2],
                build_path="test",
                origin=SourceOrigin.MARIN,
            )
        )
    )


def test_pipeline_less_pointers_are_rejected():
    with pytest.raises(ValueError, match="no registered conversion pipeline"):
        _ = _agentic(TaskFormat.TASKCOMPENDIUM).declaration


@pytest.mark.parametrize("missing", ["count", "basis", "oversight", "verification"])
def test_unqualified_agentic_pipeline_stays_visible_but_cannot_be_selected(monkeypatch, missing):
    info = replace(
        _agentic(TaskFormat.TASKCOMPENDIUM).info,
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
    assert source.name not in runnable_sources()
    assert source.declaration.status == "Not selectable"


def test_existing_internal_blend_selection_does_not_require_new_audit_evidence(monkeypatch):
    source_id = next(iter(GRANDFATHERED_MARINSKYRL_SOURCES))
    source = all_sources()[source_id]
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_sources", lambda: {source_id: source})
    assert runnable_sources()[source.name] is source
    assert source.declaration.status == "Available"
