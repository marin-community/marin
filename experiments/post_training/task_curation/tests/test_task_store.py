# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from marin.execution.artifact import FingerprintMismatchError
from marin.execution.lazy import ArtifactStep, resolve, run
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.task_store import TaskRow, TaskStore, task_store_step


@pytest.fixture(autouse=True)
def local_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)


def task(identifier: str) -> TaskSpec:
    return TaskSpec(
        id=identifier,
        source=Source(dataset="fixture", revision="1", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=f"Solve {identifier}."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="store fixture"),
    )


def curated_source(root: Path, *shards: list[str]) -> ArtifactStep:
    """A curation artifact whose final/ shards hold the given task IDs; normalize/ holds an unadmitted task."""
    (root / "final").mkdir(parents=True)
    (root / "normalize").mkdir()
    manifest = {
        "intended_use": "train",
        "mode": "full",
        "datasets": {"final": str(root / "final")},
        "admission_counts": {"admitted": sum(map(len, shards))},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for index, identifiers in enumerate(shards):
        rows = [{"task_id": identifier, "task_json": task(identifier).model_dump_json()} for identifier in identifiers]
        pq.write_table(pa.Table.from_pylist(rows), root / f"final/part-{index:05d}.parquet")
    unadmitted = [{"task_id": "unadmitted", "task_json": task("unadmitted").model_dump_json()}]
    pq.write_table(pa.Table.from_pylist(unadmitted), root / "normalize/part-00000.parquet")
    return ArtifactStep.adopt(f"curated/{root.name}", "2026.11.26.1", str(root))


def read(store: TaskStore, tmp_path: Path) -> list[TaskRow]:
    with ZephyrContext(client=LocalClient(), max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        return list(context.execute(store.to_dataset()).results)


def test_opened_store_reads_every_admitted_task_as_one_dataset(tmp_path):
    math = curated_source(tmp_path / "math", ["m1", "m2"], ["m3"])
    code = curated_source(tmp_path / "code", ["c1"])
    run(task_store_step("fixture", "2026.11.26.1", sources={"fixture:math": math, "fixture:code": code}))

    store = TaskStore.open("fixture", "2026.11.26.1")
    rows = read(store, tmp_path)

    assert [(row.source_id, row.task_id) for row in rows] == [
        ("fixture:code", "c1"),
        ("fixture:math", "m1"),
        ("fixture:math", "m2"),
        ("fixture:math", "m3"),
    ]
    assert all(row.task.id == row.task_id for row in rows)
    selected = read(store.select_sources("fixture:math", "fixture:code"), tmp_path)
    assert [row.task_id for row in selected] == ["m1", "m2", "m3", "c1"]
    with pytest.raises(FileNotFoundError, match=r"No task store fixture@2026\.11\.26\.2"):
        TaskStore.open("fixture", "2026.11.26.2")


def test_store_refuses_a_source_whose_manifest_changed(tmp_path):
    source = curated_source(tmp_path / "math", ["m1"])
    store = resolve(task_store_step("fixture", "2026.11.26.1", sources={"fixture:math": source}))

    (tmp_path / "math/manifest.json").write_text(json.dumps({"intended_use": "train", "mode": "sample"}))
    with pytest.raises(ValueError, match="Curated source manifest changed after the task store pinned it"):
        store.to_dataset()


def test_store_reads_the_admitted_files_its_manifest_records_and_refuses_missing_ones(tmp_path):
    source = curated_source(tmp_path / "math", ["m1", "m2"])
    moved = tmp_path / "expiring/final"
    moved.parent.mkdir()
    (tmp_path / "math/final").rename(moved)
    manifest = json.loads((tmp_path / "math/manifest.json").read_text())
    (tmp_path / "math/manifest.json").write_text(json.dumps({**manifest, "datasets": {"final": str(moved)}}))
    store = resolve(task_store_step("fixture", "2026.11.26.1", sources={"fixture:math": source}))

    assert [row.task_id for row in read(store, tmp_path)] == ["m1", "m2"]
    for path in moved.iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="admitted 2 tasks"):
        store.to_dataset()


def test_a_store_version_keeps_the_sources_it_was_first_built_with(tmp_path):
    math = curated_source(tmp_path / "math", ["m1"])
    code = curated_source(tmp_path / "code", ["c1"])
    run(task_store_step("fixture", "2026.11.26.1", sources={"fixture:math": math}))
    run(task_store_step("fixture", "2026.11.26.1", sources={"fixture:math": math}))

    with pytest.raises(FingerprintMismatchError):
        run(task_store_step("fixture", "2026.11.26.1", sources={"fixture:math": math, "fixture:code": code}))
    assert list(TaskStore.open("fixture", "2026.11.26.1").sources) == ["fixture:math"]
