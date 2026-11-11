# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona-safety projection on source declarations."""

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def test_non_agentic_sources_are_daytona_safe_without_an_authored_value():
    source = RlDataSource(info=SourceInfo(id="trove:math", title="Math", origin="fixture", tags=("rlvr", "single-turn")))
    assert source.declaration.daytona_safe is True


def test_agentic_sources_carry_their_assessed_snapshot_safety():
    unassessed = RlDataSource(info=SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",)))
    safe = RlDataSource(info=SourceInfo(id="trove:agent", title="Agent", origin="fixture", tags=("agentic",), daytona_safe=True))
    unsafe = RlDataSource(info=SourceInfo(id="trove:swe", title="SWE", origin="fixture", tags=("agentic",), daytona_safe=False))
    assert unassessed.declaration.daytona_safe is None
    assert safe.declaration.daytona_safe is True
    assert unsafe.declaration.daytona_safe is False
