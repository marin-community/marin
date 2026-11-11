# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declaration projections: Daytona safety and catalog status."""

from dataclasses import replace

from finestore.rl.curation import DAYTONA_SAFE_SNAPSHOT_LIMIT, DaytonaSafety, OracleValidation

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.sources import all_sources

MEASURED = {"daytona_snapshot_count": 5, "daytona_count_basis": "shellbox-identity over pinned payloads"}


def test_non_agentic_sources_are_daytona_safe_without_an_authored_value():
    source = RlDataSource(info=SourceInfo(id="trove:math", title="Math", origin="fixture", tags=("rlvr", "single-turn")))
    assert source.declaration.daytona_safety is DaytonaSafety.SAFE


def test_an_agentic_verdict_requires_a_measured_count_to_survive_the_projection():
    agent = SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",))
    inferred = RlDataSource(info=replace(agent, daytona_safety=DaytonaSafety.UNSAFE))
    assert inferred.declaration.daytona_safety is DaytonaSafety.UNASSESSED
    measured = RlDataSource(info=replace(agent, daytona_safety=DaytonaSafety.UNSAFE, **MEASURED))
    assert measured.declaration.daytona_safety is DaytonaSafety.UNSAFE
    assert measured.declaration.daytona_snapshot_count == 5
    assert measured.declaration.daytona_snapshot_threshold == DAYTONA_SAFE_SNAPSHOT_LIMIT


def test_deprecated_tag_marks_the_status_over_available():
    deprecated = RlDataSource(info=SourceInfo(id="trove:old", title="Old", origin="fixture", tags=("deprecated",)))
    excluded = RlDataSource(
        info=SourceInfo(id="trove:gone", title="Gone", origin="fixture", tags=("deprecated", "excluded"))
    )
    available = RlDataSource(info=SourceInfo(id="trove:new", title="New", origin="fixture", tags=("rlvr",)))
    assert deprecated.declaration.status == "Deprecated"
    assert excluded.declaration.status == "Deprecated"
    assert available.declaration.status == "Available"


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
