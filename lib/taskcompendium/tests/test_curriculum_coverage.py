# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import QualifiedSourceDeclaration, source_display_name
from finestore.rl.status_taxonomy import PendingStage
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from taskcompendium.pipeline.curriculum.coverage import (
    CAPABILITY_DISTANCE,
    CONFIDENCE_LEVELS,
    SUBJECT_DISTANCE,
    build_curriculum_coverage,
)
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumNode
from taskcompendium.pipeline.curriculum.task_alignment import TaskAlignment


def test_curriculum_release_retains_qualified_pins_and_private_tier():
    revision = "a" * 40
    converter_revision = "fixture-converter"
    declaration = QualifiedSourceDeclaration(
        id="Marin:curriculum",
        display_name=source_display_name("Marin:curriculum"),
        origin="Marin",
        upstream_dataset="curriculum",
        upstream_name="Curriculum",
        build_path="Pipeline",
        current_release={"repo": "fixture", "revision": revision, "url": "https://example.org/source"},
        within_source_dedup={
            "raw_count": 3,
            "unique_count": 3,
            "within_source_duplicates": 0,
            "source_dataset_id": "fixture",
            "source_revision": revision,
            "converter_revision": converter_revision,
            "converter_version": "1",
            "drop_manifest_uri": "memory://fixture/drops",
            "evidence_url": "memory://fixture/dedup",
        },
        family="fixture",
        tags=[],
        task_count=3,
        notes="",
        dataset_id="fixture",
        dataset_revision=revision,
        intended_use="train",
        data_restriction="private_always",
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Pending",
        stage=PendingStage.MEASUREMENTS_PENDING,
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": converter_revision, "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
    )
    release = {
        "release_id": "fixture-release",
        "created_at": "2026-11-19T00:00:00Z",
        "curriculum_key": "fixture-curriculum",
        "population": "converted",
        "sample_size": None,
        "task_count": 3,
        "ready_count": 3,
        "insufficient_context_count": 0,
        "sources": [
            {
                "source_id": declaration.id,
                "source_key": "fixture",
                "population": "converted",
                "intended_use": "train",
                "task_count": 3,
                "ready_count": 3,
                "insufficient_context_count": 0,
                "basis": "matched",
                "curation_root": "memory://fixture",
                "input_rows": 3,
                "converted_rows": 3,
            }
        ],
        "inventory": [declaration.model_dump(mode="json")],
        "nodes": [],
        "paths": [],
        "lineage": {},
        "embedding_spec": {
            "model": "fixture",
            "revision": "1",
            "tokenizer_revision": "1",
            "dimensions": 2,
            "max_tokens": 16,
            "compute_dtype": "float32",
        },
    }
    restored = AlignmentRelease.model_validate(release)
    roundtrip = AlignmentRelease.model_validate_json(restored.model_dump_json())
    assert roundtrip.inventory[0].current_release.revision == revision
    assert roundtrip.inventory[0].within_source_dedup.unique_count == roundtrip.task_count == 3
    assert roundtrip.inventory[0].data_restriction.value == "private_always"


def test_atlas_counts_full_instances_and_keeps_bounded_overlapping_samples(tmp_path: Path):
    nodes = [
        CurriculumNode(
            subject_id="math",
            node_id=identifier,
            parent_id=parent,
            kind=kind,
            name=identifier,
            description=identifier,
        )
        for identifier, parent, kind in (
            ("subject:math", None, "subject"),
            ("addition", "subject:math", "capability"),
            ("subtraction", "subject:math", "capability"),
        )
    ]
    nodes.extend(
        CurriculumNode(
            subject_id="math",
            node_id=f"cluster{index}",
            parent_id="subject:math",
            kind="capability",
            name=f"Cluster {index}",
            description="Count objects",
        )
        for index in range(20)
    )
    tasks = []
    for source, count, capability in (("a", 200, "addition"), ("b", 20, "subtraction"), ("c", 200, "cluster")):
        for index in range(count):
            chosen = f"cluster{index//10}" if source == "c" else capability
            tasks.append(
                TaskAlignment(
                    source_id=source,
                    source_key=source,
                    task_id=str(index),
                    task_hash=source + str(index),
                    content_hash="same-public-input",
                    intended_use="train",
                    population="converted",
                    description_status="ready",
                    reason="",
                    description="Count objects",
                    task_preview="Count objects",
                    instruction_characters=13,
                    preview_truncated=False,
                    matches={
                        "subjects": [
                            {
                                "subject_id": "math",
                                "facet": "subject_domain",
                                "membership_similarity": 0.9,
                                "membership_anchor_id": "math",
                                "capability_id": chosen,
                                "margin": 0.1,
                                "candidates": [
                                    {
                                        "subject_id": "math",
                                        "capability_id": chosen,
                                        "similarity": 0.9,
                                        "anchor_id": chosen,
                                    }
                                ],
                                "subject_candidates": [],
                            }
                        ],
                        "capabilities": [],
                    },
                ).model_dump(mode="json")
            )
    tasks.append(
        {**tasks[200], "task_id": "skipped", "description_status": "skipped", "description": "", "matches": None}
    )
    # Full task bodies must survive identity-only sample selection unchanged.
    for index, row in enumerate(tasks):
        row["description"] = f"Task body {index}: " + ("wide task evidence " * 1200)
        row["task_preview"] = f"Preview {index}"
    originals = {(row["source_id"], row["task_id"]): row for row in tasks}
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(tasks), path)
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        data = build_curriculum_coverage(
            context,
            task_paths=[str(path)],
            nodes=nodes,
            output_root=str(tmp_path / "atlas"),
            subject_distance=0.45,
            capability_distance=0.25,
            source_sample_size=100,
            area_sample_size=10,
            sampling_source_ids={source: source for source in ("a", "b", "c")},
        )
    statistics = {
        (row["source_id"], row["metric"], row["node_id"]): row["instances"]
        for part in data.paths
        for row in load_parquet(part)
    }
    assert statistics["a", "denominator", ""] == 200
    assert statistics["b", "denominator", ""] == 21
    assert statistics["a", "placement", "addition"] == 200
    assert statistics["b", "placement", "subtraction"] == 20
    assert statistics["a", "placement", "subject:math"] == 200
    assert statistics["b", "skipped", ""] == 1
    samples = [row for part in data.sample_paths for row in load_parquet(part)]
    assert data.sample_task_count == len(samples) == 321
    assert all(row == originals[row["source_id"], row["task_id"]] for row in samples)
    assert len({(row["source_id"], row["task_id"]) for row in samples}) == 321
    assert sum(row["source_id"] == "a" for row in samples) == 100
    assert {row["source_id"] for row in samples} == {"a", "b", "c"}
    assert sum(row["source_id"] == "c" for row in samples) == 200

    # A renamed source must retain the same deterministic review examples.
    old = "source-before-rename"
    sample_ids = []
    for index, identifier in enumerate((old, "source-after-rename")):
        pq.write_table(pa.Table.from_pylist([{**row, "source_id": identifier} for row in tasks[:200]]), path)
        with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / f"alias-chunks-{index}")) as context:
            migrated = build_curriculum_coverage(
                context,
                task_paths=[str(path)],
                nodes=nodes,
                output_root=str(tmp_path / f"alias-{index}"),
                subject_distance=0.45,
                capability_distance=0.25,
                source_sample_size=7,
                area_sample_size=3,
                sampling_source_ids={identifier: old},
            )
        sample_ids.append(sorted(row["task_id"] for part in migrated.sample_paths for row in load_parquet(part)))
    assert len(sample_ids[0]) == 7
    assert sample_ids[0] == sample_ids[1]


@pytest.mark.parametrize(
    "subject_distance,capability_distance,expected_outcomes,expected_placements",
    [
        (0.45, 0.25, {"assigned": 1, "subject_only": 1, "out_of_scope": 1}, 1),
        (SUBJECT_DISTANCE, CAPABILITY_DISTANCE, {"assigned": 3}, 3),
    ],
    ids=["historical-cutoffs", "approximate-nearest-coverage"],
)
def test_coverage_places_lower_similarity_tasks_and_preserves_unavailable_outcomes(
    tmp_path: Path,
    subject_distance: float,
    capability_distance: float,
    expected_outcomes: dict,
    expected_placements: int,
):
    nodes = [
        CurriculumNode(
            subject_id="earth",
            node_id=identifier,
            parent_id=parent,
            kind=kind,
            name=identifier,
            description="",
        )
        for identifier, parent, kind in (
            ("subject:earth", None, "subject"),
            ("weather", "subject:earth", "capability"),
        )
    ]
    tasks = []
    for identifier, membership, capability in (("high", 0.9, 0.8), ("subject-only", 0.6, 0.65), ("outside", 0.42, 0.44)):
        tasks.append(
            TaskAlignment(
                source_id="science",
                source_key="science-pin",
                task_id=identifier,
                task_hash=identifier,
                content_hash=identifier,
                intended_use="eval",
                population="converted",
                description_status="ready",
                reason="",
                description="Identify the weather instrument",
                task_preview="Identify the weather instrument",
                instruction_characters=31,
                preview_truncated=False,
                matches={
                    "subjects": [
                        {
                            "subject_id": "earth",
                            "facet": "subject_domain",
                            "membership_similarity": membership,
                            "membership_anchor_id": "earth",
                            "capability_id": "weather",
                            "margin": 0.01,
                            "candidates": [
                                {
                                    "subject_id": "earth",
                                    "capability_id": "weather",
                                    "similarity": capability,
                                    "anchor_id": "weather",
                                }
                            ],
                            "subject_candidates": [],
                        }
                    ],
                    "capabilities": [],
                },
            ).model_dump(mode="json")
        )
    for status in ("insufficient_context", "skipped"):
        tasks.append({**tasks[0], "task_id": status, "description_status": status, "description": "", "matches": None})
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(tasks), path)
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        data = build_curriculum_coverage(
            context,
            task_paths=[str(path)],
            nodes=nodes,
            output_root=str(tmp_path / "coverage"),
            subject_distance=subject_distance,
            capability_distance=capability_distance,
            source_sample_size=10,
            area_sample_size=10,
            sampling_source_ids={"science": "science"},
            confidence_levels=CONFIDENCE_LEVELS,
        )
    statistics = [row for part in data.paths for row in load_parquet(part)]
    totals = {row["metric"]: row["instances"] for row in statistics if row["node_id"] == ""}
    assert totals == {"denominator": 5, "insufficient_context": 1, "skipped": 1, **expected_outcomes}
    placements = {row["node_id"]: row["instances"] for row in statistics if row["metric"] == "placement"}
    assert placements["weather"] == expected_placements
    confidence = {
        (row["metric"], row["node_id"]): row["instances"] for part in data.confidence_paths for row in load_parquet(part)
    }
    assert confidence["denominator", ""] == [5, 5, 5, 5, 5]
    assert confidence["assigned", ""] == [3, 2, 2, 1, 1]
    assert confidence["subject_only", ""] == [0, 0, 0, 1, 1]
    assert confidence["out_of_scope", ""] == [0, 1, 1, 1, 1]
    assert confidence["placement", "weather"] == [3, 2, 2, 1, 1]
    assert confidence["placement", "subject:earth"] == [3, 2, 2, 2, 2]
    assert confidence["skipped", ""] == [1, 1, 1, 1, 1]
    assert confidence["insufficient_context", ""] == [1, 1, 1, 1, 1]
    samples = [row for part in data.sample_paths for row in load_parquet(part)]
    assert {row["task_id"] for row in samples} == {row["task_id"] for row in tasks}
    assert {row["task_id"]: row["matches"] for row in samples} == {row["task_id"]: row["matches"] for row in tasks}

    # The nearest-level sample selects "outside"; the strict-level sample must
    # also retain the sole high-scoring task when no source sample is requested.
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "strict-chunks")) as context:
        stratified = build_curriculum_coverage(
            context,
            task_paths=[str(path)],
            nodes=nodes,
            output_root=str(tmp_path / "stratified"),
            subject_distance=SUBJECT_DISTANCE,
            capability_distance=CAPABILITY_DISTANCE,
            source_sample_size=0,
            area_sample_size=1,
            sampling_source_ids={"science": "science"},
            confidence_levels=CONFIDENCE_LEVELS,
        )
    assert {row["task_id"] for part in stratified.sample_paths for row in load_parquet(part)} >= {"outside", "high"}
