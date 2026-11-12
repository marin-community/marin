# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every declaration carries measured coverage, and the catalog refuses one that does not."""

import pytest
from finestore.rl.curation import CurriculumArea, SourceCoverage, SourceUse

from experiments.post_training.task_curation import sources
from experiments.post_training.task_curation.source_coverage import (
    CoverageGap,
    release_coverage,
    require_coverage,
)

MEASURED = SourceCoverage(
    intended_use=SourceUse.TRAIN,
    example_count=10,
    top_area=CurriculumArea(subject_id="D01", name="Mathematics & Statistics", instances=7),
    basis="curriculum release r",
)


def test_the_catalog_rejects_a_declaration_without_coverage(monkeypatch):
    table = sources.load_coverage()
    missing = next(iter(table))
    monkeypatch.setattr(sources, "load_coverage", lambda: {key: value for key, value in table.items() if key != missing})
    with pytest.raises(ValueError, match=f"Source {missing} has no coverage entry"):
        sources.all_sources()


def test_every_declaration_has_coverage_and_runnable_sources_are_measured():
    catalog = sources.all_sources()
    assert all(source.declaration.coverage is not None for source in catalog.values())
    assert all(source.declaration.coverage.example_count is not None for source in sources.runnable_sources().values())


def test_a_runnable_source_cannot_stand_on_a_declared_count():
    declared = MEASURED.model_copy(update={"top_area": None, "unavailable_reason": CoverageGap.NOT_EMBEDDED.value})
    require_coverage("inventory", declared, runnable=False)
    with pytest.raises(ValueError, match="no measured coverage"):
        require_coverage("runnable", declared, runnable=True)
    unplaced = declared.model_copy(update={"unavailable_reason": CoverageGap.NO_PLACEMENT.value})
    require_coverage("runnable", unplaced, runnable=True)
    with pytest.raises(ValueError, match="without a reason"):
        require_coverage("runnable", MEASURED.model_copy(update={"top_area": None}), runnable=True)


def test_release_coverage_counts_conversions_and_picks_the_most_placed_subject():
    release = {
        "sample_size": None,
        "nodes": [
            {"subject_id": "D01", "kind": "subject", "name": "Math"},
            {"subject_id": "D02", "kind": "subject", "name": "Computing"},
        ],
        # A release from before declared rows existed has no basis field.
        "sources": [
            {"source_id": "a", "intended_use": "train", "population": "converted", "task_count": 9, "converted_rows": 9},
            {"source_id": "b", "intended_use": "eval", "population": "converted", "task_count": 4, "converted_rows": 4},
            {"source_id": "c", "basis": "declared", "intended_use": None, "population": "converted", "task_count": 3},
        ],
    }
    statistics = [
        {"source_id": "a", "metric": "placement", "subject_id": "D01", "node_id": "subject:D01", "instances": 3},
        {"source_id": "a", "metric": "placement", "subject_id": "D02", "node_id": "subject:D02", "instances": 3},
        {"source_id": "a", "metric": "placement", "subject_id": "D02", "node_id": "d02.group", "instances": 9},
    ]
    coverage = release_coverage(release, "s3://r/release.json", statistics)
    assert set(coverage) == {"a", "b"}
    assert (coverage["a"].example_count, coverage["a"].top_area.subject_id) == (9, "D01")
    assert coverage["b"].intended_use is SourceUse.EVAL
    assert coverage["b"].top_area is None and coverage["b"].unavailable_reason == CoverageGap.NO_PLACEMENT.value
