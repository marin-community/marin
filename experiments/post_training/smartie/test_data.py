# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.test_support import arithmetic_task, lowered, machine_runtime

from experiments.post_training.smartie.data import (
    AdmittedInput,
    build_real_dataset,
    convert_admitted_source,
    split_real_tasks,
)


def test_admitted_tasks_roundtrip_and_split_without_order_leakage():
    tasks = [arithmetic_task(id=f"task-{index}") for index in range(10)]
    rows = [{"task_id": task.id, "task_json": task.model_dump_json()} for task in tasks]
    manifest = {
        "mode": "full",
        "status": "completed",
        "intended_use": "train",
        "unprocessed_rows": 0,
        "source_dataset": "fixture",
        "source_revision": "1",
        "admission_counts": {"admitted": 10},
    }
    converted = convert_admitted_source(
        manifest, rows, machine=machine_runtime(), session=lowered(tasks[0]).session, factories={}
    )
    assert not converted.failures
    restored = tuple(LoweredTaskSpec.model_validate_json(task.model_dump_json()) for task in converted.tasks)
    assert [task.task for task in restored] == tasks
    train, validation = split_real_tasks(restored, validation_count=2, seed=42)
    assert len(train) == 8 and len(validation) == 2
    assert {task.task.id for task in train}.isdisjoint(task.task.id for task in validation)
    assert split_real_tasks(reversed(restored), validation_count=2, seed=42) == (train, validation)
    with pytest.raises(ValueError, match="Duplicate task identities"):
        split_real_tasks((*restored, restored[0]), validation_count=2, seed=42)


def test_conversion_reports_all_invalid_rows_and_population_mismatch():
    task = arithmetic_task()
    manifest = {
        "mode": "full",
        "status": "completed",
        "intended_use": "train",
        "unprocessed_rows": 0,
        "source_dataset": "fixture",
        "source_revision": "1",
        "admission_counts": {"admitted": 4},
    }
    rows = [
        {"task_id": "bad-json", "task_json": "{}"},
        {"task_id": "wrong-id", "task_json": task.model_dump_json()},
        {"task_id": task.id, "task_json": task.model_dump_json()},
    ]
    result = convert_admitted_source(
        manifest, rows, machine=machine_runtime(), session=lowered(task).session, factories={}
    )
    assert [failure.task_id for failure in result.failures] == ["bad-json", "wrong-id", "<source>"]
    assert len(result.tasks) == 1
    with pytest.raises(ValueError, match="completed FULL"):
        convert_admitted_source(
            {**manifest, "mode": "quick"},
            rows,
            machine=machine_runtime(),
            session=lowered(task).session,
            factories={},
        )


@pytest.mark.parametrize("invalid", [False, True])
def test_source_files_publish_native_splits_only_after_all_sources_pass(tmp_path, invalid):
    inputs = []
    expected = {}
    for source_index in range(2):
        root = tmp_path / f"source-{source_index}"
        (root / "final").mkdir(parents=True)
        tasks = [arithmetic_task(id=f"source-{source_index}-task-{index}") for index in range(4)]
        expected.update({task.id: task for task in tasks})
        rows = [{"task_id": task.id, "task_json": task.model_dump_json()} for task in tasks]
        if invalid:
            rows[0]["task_json"] = "{}"
        pq.write_table(pa.Table.from_pylist(rows), root / "final/part-00000.parquet")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "mode": "full",
                    "status": "completed",
                    "intended_use": "train",
                    "unprocessed_rows": 0,
                    "source_dataset": "fixture",
                    "source_revision": "1",
                    "admission_counts": {"admitted": 4},
                    "datasets": {"final": str(root / "final")},
                }
            )
        )
        inputs.append(AdmittedInput(str(root), 4, 1))
    output = tmp_path / "dataset"
    kwargs = dict(
        sources=tuple(inputs),
        output=StoragePath(str(output)),
        seed=42,
        machine=machine_runtime(),
        session=lowered(arithmetic_task()).session,
        factories={},
    )
    if invalid:
        with pytest.raises(ValueError, match="publication rejected"):
            build_real_dataset(**kwargs)
        report = json.loads((output / "conversion-report.json").read_text())
        assert len(report["failures"]) == 2
        assert {failure["source"] for failure in report["failures"]} == {source.root for source in inputs}
        assert not (output / "train.parquet").exists()
        assert not (output / "manifest.json").exists()
        return
    report = build_real_dataset(**kwargs)
    assert len(report["train_ids"]) == 6 and len(report["validation_ids"]) == 2
    assert set(report["train_ids"]).isdisjoint(report["validation_ids"])
    recovered = {}
    for name in ("train", "validation"):
        for row in pq.read_table(output / f"{name}.parquet").to_pylist():
            task = LoweredTaskSpec.model_validate_json(row["lowered_task_spec"]).task
            recovered[task.id] = task
    assert recovered == expected
    with pytest.raises(FileExistsError):
        build_real_dataset(**kwargs)
