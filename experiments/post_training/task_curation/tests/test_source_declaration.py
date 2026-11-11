# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona-safety projection on source declarations."""

from finestore.rl.curation import DaytonaSafety

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def test_non_agentic_sources_are_daytona_safe_without_an_authored_value():
    source = RlDataSource(info=SourceInfo(id="trove:math", title="Math", origin="fixture", tags=("rlvr", "single-turn")))
    assert source.declaration.daytona_safety is DaytonaSafety.SAFE


def test_agentic_sources_carry_their_assessed_snapshot_safety():
    unassessed = RlDataSource(info=SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",)))
    safe = RlDataSource(
        info=SourceInfo(
            id="trove:agent", title="Agent", origin="fixture", tags=("agentic",), daytona_safety=DaytonaSafety.SAFE
        )
    )
    unsafe = RlDataSource(
        info=SourceInfo(
            id="trove:swe", title="SWE", origin="fixture", tags=("agentic",), daytona_safety=DaytonaSafety.UNSAFE
        )
    )
    assert unassessed.declaration.daytona_safety is DaytonaSafety.UNASSESSED
    assert safe.declaration.daytona_safety is DaytonaSafety.SAFE
    assert unsafe.declaration.daytona_safety is DaytonaSafety.UNSAFE
