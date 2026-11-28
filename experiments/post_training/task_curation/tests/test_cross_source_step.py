# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The campaign cross-source dedup step records per-source drops and evidence."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import RawRow

from experiments.post_training.task_curation.cross_source_step import (
    CrossSourceRequest,
    cross_source_dedup_step,
    run_cross_source_dedup,
)


def arithmetic_task(task_id: str, prompt: str, answer: str) -> TaskSpec:
    row = RawRow(task_id, Source(dataset="fixture", revision="1", row=task_id, importer_revision="1"), {})
    task = numeric_answer_task(row, prompt=prompt, answer=answer, tolerance_abs=0, tolerance_rel=0)
    assert isinstance(task, TaskSpec)
    return task.model_copy(update={"id": task_id})


def write_final(path: Path, tasks: list[TaskSpec]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"task_id": task.id, "task_json": task.model_dump_json()} for task in tasks]
    pq.write_table(pa.Table.from_pylist(rows), path / "part-00000.parquet")


def test_step_returns_none_for_a_single_source():
    assert (
        cross_source_dedup_step(
            {"alpha": _fake_step("alpha")},
            max_workers=1,
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
        )
        is None
    )


def _fake_step(source_id: str):
    return ArtifactStep(
        name=f"source/{source_id}",
        version="2026.11.27.1",
        artifact_type=pa.Table,
        deps=[],
        run=lambda: None,
        build_config=lambda ctx: None,
    )


def test_run_records_the_later_source_duplicates_and_the_campaign_evidence(tmp_path):
    shared = arithmetic_task("shared-task", "What is 7 + 5?", "12")
    later_copy = shared.model_copy(update={"id": "shared-task-copy"})
    write_final(tmp_path / "alpha" / "final", [shared])
    write_final(tmp_path / "beta" / "final", [later_copy, arithmetic_task("second-only", "What is 10 - 2?", "8")])
    output = tmp_path / "cross-source"
    request = CrossSourceRequest(
        sources=(
            ("alpha", str(tmp_path / "alpha" / "final"), 0),
            ("beta", str(tmp_path / "beta" / "final"), 1),
        ),
        output_path=str(output),
    )

    artifact = run_cross_source_dedup(request, max_workers=1, worker_resources=ResourceConfig(cpu=1, ram="1g"))

    assert artifact.manifest["cross_source_duplicates"] == 1
    assert artifact.manifest["sources"]["beta"]["cross_source_duplicates"] == 1
    assert artifact.manifest["sources"]["alpha"]["cross_source_duplicates"] == 0
    drops = pq.read_table(Path(artifact.manifest["sources"]["beta"]["drop_manifest_uri"])).to_pylist()
    assert drops == [
        {
            "task_id": "shared-task-copy",
            "kept_id": "shared-task",
            "kept_source_id": "alpha",
            "reason": "DUPLICATE_CROSS_SOURCE",
        }
    ]
    assert (Path(request.output_path) / "manifest.json").exists()
