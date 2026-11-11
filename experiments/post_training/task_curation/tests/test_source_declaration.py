# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declaration projections: Daytona safety and catalog status."""

from dataclasses import replace

import pytest
from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    AdmissionCounts,
    CurationMode,
    CurationSummary,
    DaytonaSafety,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    mode_admits_training,
)

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.sources import all_sources

MEASURED = {"daytona_snapshot_count": 5, "daytona_count_basis": "shellbox-identity over pinned payloads"}


def test_non_agentic_sources_are_daytona_safe_without_an_authored_value():
    source = RlDataSource(info=SourceInfo(id="trove:math", title="Math", origin="fixture", tags=("rlvr", "single-turn")))
    assert source.declaration.daytona_safety is DaytonaSafety.SAFE


def test_an_agentic_verdict_comes_from_the_measured_count():
    """The verdict is derived from the count, so no field on the source can set it."""
    agent = SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",))
    inferred = RlDataSource(info=replace(agent, **MEASURED))
    assert inferred.declaration.daytona_safety is DaytonaSafety.SAFE
    assert inferred.declaration.daytona_snapshot_count == 5
    assert inferred.declaration.daytona_snapshot_threshold == DAYTONA_SAFE_SNAPSHOT_LIMIT

    over = RlDataSource(info=replace(agent, **{**MEASURED, "daytona_snapshot_count": DAYTONA_SAFE_SNAPSHOT_LIMIT + 1}))
    assert over.declaration.daytona_safety is DaytonaSafety.UNSAFE

    refused = RlDataSource(info=replace(agent, **MEASURED, daytona_execution_failures=1))
    assert refused.declaration.daytona_safety is DaytonaSafety.UNSAFE


def test_an_unmeasured_agentic_source_is_unassessed_whatever_it_declares():
    agent = SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",))
    assert RlDataSource(info=agent).declaration.daytona_safety is DaytonaSafety.UNASSESSED
    assert (
        RlDataSource(info=replace(agent, daytona_count_basis="x")).declaration.daytona_safety is DaytonaSafety.UNASSESSED
    )


def test_status_prefers_deprecated_then_excluded_then_inventory():
    deprecated = RlDataSource(info=SourceInfo(id="trove:old", title="Old", origin="fixture", tags=("deprecated",)))
    excluded = RlDataSource(
        info=SourceInfo(id="trove:gone", title="Gone", origin="fixture", tags=("deprecated", "excluded"))
    )
    pointer = RlDataSource(info=SourceInfo(id="trove:new", title="New", origin="fixture", tags=("rlvr",)))
    assert deprecated.declaration.status == "Deprecated"
    assert excluded.declaration.status == "Deprecated"
    assert pointer.declaration.status == "Inventory"  # no pipeline: visible, never admitted
    runnable = RlDataSource(
        info=SourceInfo(id="trove:run", title="Run", origin="fixture", tags=("rlvr",)),
        pipeline=lambda source, options: None,
    )
    assert runnable.declaration.status == "Available"


def test_oracle_validation_projects_into_the_declaration():
    info = SourceInfo(
        id="trove:no-oracle",
        title="No oracle",
        origin="fixture",
        tags=("agentic",),
        oracle_validation=OracleValidation.VERIFIER_EXECUTED,
    )
    assert RlDataSource(info=info).declaration.oracle_validation is OracleValidation.VERIFIER_EXECUTED


def test_a_source_without_oracles_reports_skipped_until_qualified():
    declaration = all_sources()["Campaign Release:smartie-calibforge-admitted"].declaration
    assert declaration.oracle_validation is OracleValidation.SKIPPED


def test_oversight_level_projects_with_its_evidence():
    info = SourceInfo(
        id="trove:reviewed",
        title="Reviewed",
        origin="fixture",
        tags=("agentic",),
        oversight=OversightEvidence(campaign="campaign", verification="1,357 of 1,357", level=OversightLevel.SAMPLE),
    )
    declaration = RlDataSource(info=info).declaration
    assert declaration.oversight is OversightLevel.SAMPLE
    assert declaration.oversight_evidence.campaign == "campaign"
    assert RlDataSource(info=SourceInfo(id="trove:inv", title="Inv", origin="fixture")).declaration.oversight is (
        OversightLevel.NONE
    )


def test_only_sample_or_full_curation_admits_and_selects():
    assert not mode_admits_training(CurationMode.QUICK)
    assert mode_admits_training(CurationMode.SAMPLE)
    assert mode_admits_training(CurationMode.FULL)
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
    assert CurationSummary(mode=CurationMode.SAMPLE, admission_counts=AdmissionCounts(admitted=1), **summary).mode is (
        CurationMode.SAMPLE
    )
