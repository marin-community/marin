# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from taskcompendium.pipeline.curriculum_coverage import build_curriculum_coverage
from taskcompendium.pipeline.curriculum_models import CurriculumNode
from taskcompendium.pipeline.task_alignment import TaskAlignment


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
    assert len({(row["source_id"], row["task_id"]) for row in samples}) == 321
    assert sum(row["source_id"] == "a" for row in samples) == 100
    assert {row["source_id"] for row in samples} == {"a", "b", "c"}
    assert sum(row["source_id"] == "c" for row in samples) == 200
