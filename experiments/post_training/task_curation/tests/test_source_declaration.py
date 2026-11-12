# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona safety and catalog status."""

from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    AdmissionCounts,
    CurationMode,
    CurationSummary,
    DaytonaSafety,
)

from experiments.post_training.task_curation.source import SourceInfo
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source

MEASURED = {"daytona_snapshot_count": 5, "daytona_count_basis": "shellbox-identity over pinned payloads"}


def test_non_agentic_sources_are_daytona_safe_without_an_authored_value():
    source = replace(
        number_source(Path("unused"), "fixture"),
        info=SourceInfo(id="trove:math", title="Math", origin="fixture", tags=("rlvr", "single-turn")),
    )
    assert source.declaration.daytona_safety is DaytonaSafety.SAFE


def test_an_agentic_verdict_comes_from_the_measured_count():
    """The verdict is derived from the count, so no field on the source can set it."""
    agent = SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",))
    inferred = replace(number_source(Path("unused"), "fixture"), info=replace(agent, **MEASURED))
    assert inferred.declaration.daytona_safety is DaytonaSafety.SAFE
    assert inferred.declaration.daytona_snapshot_count == 5
    assert inferred.declaration.daytona_snapshot_threshold == DAYTONA_SAFE_SNAPSHOT_LIMIT

    over = replace(
        number_source(Path("unused"), "fixture"),
        info=replace(agent, **{**MEASURED, "daytona_snapshot_count": DAYTONA_SAFE_SNAPSHOT_LIMIT + 1}),
    )
    assert over.declaration.daytona_safety is DaytonaSafety.UNSAFE

    refused = replace(
        number_source(Path("unused"), "fixture"), info=replace(agent, **MEASURED, daytona_execution_failures=1)
    )
    assert refused.declaration.daytona_safety is DaytonaSafety.UNSAFE


def test_an_unmeasured_agentic_source_is_unassessed_whatever_it_declares():
    agent = SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",))
    assert (
        replace(number_source(Path("unused"), "fixture"), info=agent).declaration.daytona_safety
        is DaytonaSafety.UNASSESSED
    )
    assert (
        replace(
            number_source(Path("unused"), "fixture"), info=replace(agent, daytona_count_basis="x")
        ).declaration.daytona_safety
        is DaytonaSafety.UNASSESSED
    )


def test_converted_status_preserves_deprecation_and_exclusion():
    deprecated = replace(
        number_source(Path("unused"), "fixture"),
        info=SourceInfo(id="trove:old", title="Old", origin="fixture", tags=("deprecated",)),
    )
    excluded = replace(
        number_source(Path("unused"), "fixture"),
        info=SourceInfo(id="trove:gone", title="Gone", origin="fixture", tags=("deprecated", "excluded")),
    )
    pointer = replace(
        number_source(Path("unused"), "fixture"),
        info=SourceInfo(id="trove:new", title="New", origin="fixture", tags=("rlvr",)),
    )
    assert deprecated.declaration.status == "Deprecated"
    assert excluded.declaration.status == "Deprecated"
    assert pointer.declaration.status == "Available"
    runnable = replace(
        number_source(Path("unused"), "fixture"),
        info=SourceInfo(id="trove:run", title="Run", origin="fixture", tags=("rlvr",)),
        pipeline=lambda source, options: None,
    )
    assert runnable.declaration.status == "Available"


def test_quick_curation_cannot_admit_tasks():
    summary = {
        "status": "complete",
        "source_revision": None,
        "recipe_revision": None,
        "raw_population_count": None,
        "raw_sample_count": None,
        "processed_rows": None,
        "unprocessed_rows": None,
        "verification": None,
    }
    with pytest.raises(ValueError, match="QUICK curation cannot carry admitted tasks"):
        CurationSummary(mode=CurationMode.QUICK, admission_counts=AdmissionCounts(admitted=1), **summary)
