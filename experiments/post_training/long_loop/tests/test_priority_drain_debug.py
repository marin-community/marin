# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow.parquet as pq
from taskcompendium.models import TaskSpec

from experiments.post_training.long_loop.priority_drain_debug import DebugData, build_data


def test_debug_cohort_round_trip_is_repeatable_and_train_val_disjoint(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    build_data(DebugData(str(first)))
    build_data(DebugData(str(second)))

    manifest = json.loads((first / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        payload = (first / name).read_bytes()
        assert payload == (second / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert pq.read_table(first / name).num_rows == record["rows"]
    train = [
        TaskSpec.model_validate_json(row["task_spec"]) for row in pq.read_table(first / "train.parquet").to_pylist()
    ]
    validation = [
        TaskSpec.model_validate_json(row["task_spec"]) for row in pq.read_table(first / "validation.parquet").to_pylist()
    ]
    assert train and validation
    assert {task.id for task in train}.isdisjoint(task.id for task in validation)
    assert {task.context.model_dump_json() for task in train}.isdisjoint(
        task.context.model_dump_json() for task in validation
    )
    assert {task.source.dataset for task in train}.isdisjoint(task.source.dataset for task in validation)
