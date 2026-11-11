# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declaration projections: Daytona safety and catalog status."""

from finestore.rl.curation import DaytonaSafety

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def test_non_agentic_sources_are_daytona_safe_without_an_authored_value():
    source = RlDataSource(info=SourceInfo(id="trove:math", title="Math", origin="fixture", tags=("rlvr", "single-turn")))
    assert source.declaration.daytona_safety is DaytonaSafety.SAFE


def test_deprecated_tag_marks_the_status_over_available():
    deprecated = RlDataSource(info=SourceInfo(id="trove:old", title="Old", origin="fixture", tags=("deprecated",)))
    excluded = RlDataSource(
        info=SourceInfo(id="trove:gone", title="Gone", origin="fixture", tags=("deprecated", "excluded"))
    )
    available = RlDataSource(info=SourceInfo(id="trove:new", title="New", origin="fixture", tags=("rlvr",)))
    assert deprecated.declaration.status == "Deprecated"
    assert excluded.declaration.status == "Deprecated"
    assert available.declaration.status == "Available"
