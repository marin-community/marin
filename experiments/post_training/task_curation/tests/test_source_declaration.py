# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona safety and catalog status."""

from dataclasses import replace

import pytest
from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    AdmissionCounts,
    CurationMode,
    CurationSummary,
    DaytonaSafety,
)
from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation.source import EXCLUDED_TAG, RlDataSource, SourceInfo, SourceOrigin

MEASURED = {"daytona_snapshot_count": 5, "daytona_count_basis": "shellbox-identity over pinned payloads"}


def test_non_agentic_sources_are_daytona_safe_without_an_authored_value():
    source = RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="math",
            upstream_name="math",
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("rlvr", "single-turn"),
        )
    )
    assert source.declaration.daytona_safety is DaytonaSafety.SAFE


def test_an_agentic_verdict_comes_from_the_measured_count():
    """The verdict is derived from the count, so no field on the source can set it."""
    agent = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        upstream_dataset="agent",
        upstream_name="agent",
        build_path="test",
        origin=SourceOrigin.MARIN,
        tags=("agentic",),
    )
    inferred = RlDataSource(info=replace(agent, **MEASURED))
    assert inferred.declaration.daytona_safety is DaytonaSafety.SAFE
    assert inferred.declaration.daytona_snapshot_count == 5
    assert inferred.declaration.daytona_snapshot_threshold == DAYTONA_SAFE_SNAPSHOT_LIMIT

    over = RlDataSource(info=replace(agent, **{**MEASURED, "daytona_snapshot_count": DAYTONA_SAFE_SNAPSHOT_LIMIT + 1}))
    assert over.declaration.daytona_safety is DaytonaSafety.UNSAFE

    refused = RlDataSource(info=replace(agent, **MEASURED, daytona_execution_failures=1))
    assert refused.declaration.daytona_safety is DaytonaSafety.UNSAFE


def test_an_unmeasured_agentic_source_is_unassessed_whatever_it_declares():
    agent = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        upstream_dataset="agent",
        upstream_name="agent",
        build_path="test",
        origin=SourceOrigin.MARIN,
        tags=("agentic",),
    )
    assert RlDataSource(info=agent).declaration.daytona_safety is DaytonaSafety.UNASSESSED
    assert (
        RlDataSource(info=replace(agent, daytona_count_basis="x")).declaration.daytona_safety is DaytonaSafety.UNASSESSED
    )


def test_status_prefers_deprecated_then_excluded_then_inventory():
    deprecated = RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="old",
            upstream_name="old",
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("deprecated",),
        )
    )
    excluded = RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="gone",
            upstream_name="gone",
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("deprecated", EXCLUDED_TAG),
        )
    )
    pointer = RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="new",
            upstream_name="new",
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("rlvr",),
        )
    )
    assert deprecated.declaration.status == "Deprecated"
    assert excluded.declaration.status == "Deprecated"
    assert pointer.declaration.status == "Inventory"  # no pipeline: visible, never admitted
    runnable = RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="run",
            upstream_name="run",
            build_path="test",
            origin=SourceOrigin.MARIN,
            tags=("rlvr",),
        ),
        pipeline=lambda source, options: None,
    )
    assert runnable.declaration.status == "Available"


def test_quick_curation_cannot_admit_tasks():
    """A QUICK summary admits nothing regardless of counts; a FULL summary carries them."""
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
    with pytest.raises(ValueError, match="cannot carry admitted tasks"):
        CurationSummary(mode=CurationMode.QUICK, admission_counts=AdmissionCounts(admitted=1), **summary)
    # The same admitted counts are valid on a FULL pass: the mode is the constraint.
    admitted = CurationSummary(mode=CurationMode.FULL, admission_counts=AdmissionCounts(admitted=1), **summary)
    assert admitted.admission_counts.admitted == 1
    # Rejections alone do not trigger the guard; only admissions do.
    rejected = CurationSummary(mode=CurationMode.QUICK, admission_counts=AdmissionCounts(rejected=3), **summary)
    assert rejected.admission_counts.rejected == 3
