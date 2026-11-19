# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic sources are admitted only in an accepted task format."""

import pytest
from finestore.rl.curation import DaytonaSafety

from experiments.post_training.task_curation.source import (
    EXCLUDED_TAG,
    GRANDFATHERED_MARINSKYRL_SOURCES,
    RlDataSource,
    SourceInfo,
    TaskFormat,
)
from experiments.post_training.task_curation.sources import all_sources, require_accepted_task_format, runnable_sources


def _agentic(task_format: TaskFormat | None, source_id: str = "fixture:agentic") -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(id=source_id, title=source_id, origin="fixture", tags=("agentic",), task_format=task_format)
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
        assert source.info.origin == "MarinSkyRL"
        assert "agentic" in source.info.tags
        assert source.pipeline is not None
        assert source.info.task_format is TaskFormat.MARINSKYRL


def test_a_new_internal_format_agentic_source_is_rejected():
    """A MarinSkyRL-format source outside the frozen list cannot be selected."""
    with pytest.raises(ValueError, match="not grandfathered"):
        require_accepted_task_format(_agentic(TaskFormat.MARINSKYRL, "MarinSkyRL:a-new-blend"))


def test_an_agentic_source_without_a_declared_format_is_rejected():
    with pytest.raises(ValueError, match="declares no task format"):
        require_accepted_task_format(_agentic(None))


@pytest.mark.parametrize("task_format", [TaskFormat.TASKCOMPENDIUM, TaskFormat.HARBOR])
def test_task_directory_formats_pass_and_non_agentic_sources_need_none(task_format):
    require_accepted_task_format(_agentic(task_format))
    require_accepted_task_format(RlDataSource(info=SourceInfo(id="fixture:math", title="math", origin="fixture")))


def test_pipeline_less_pointers_are_inventory_never_admitted():
    """A pointer stays visible in the Atlas, labelled Inventory and unselectable."""
    sources = all_sources()
    runnable = set(runnable_sources())
    pointers = [source for source in sources.values() if source.pipeline is None]
    assert pointers

    for source in pointers:
        status = source.declaration.status
        if "deprecated" in source.info.tags:
            assert status == "Deprecated"
        elif EXCLUDED_TAG in source.info.tags:
            assert status == "Excluded"
        else:
            assert status == "Inventory"
        assert source.name not in runnable
        if "agentic" in source.info.tags and source.info.daytona_snapshot_count is None:
            assert source.declaration.daytona_safety is DaytonaSafety.UNASSESSED
