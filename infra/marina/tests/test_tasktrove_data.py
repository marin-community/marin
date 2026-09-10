# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove's Marina Parquet covers each verifier-mode and environment pair."""

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

MODULE_PATH = Path(__file__).parents[1] / "apps" / "tasktrove" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("tasktrove_prepare_data", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prepare_data_samples_every_mode_and_environment(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    tasks = clean / "tasks"
    tasks.mkdir(parents=True)
    rows = [
        {
            "source": "source-a",
            "path": "task-a",
            "family": "answers",
            "converter": "answer",
            "mode": "exact",
            "dockerfile_id": "python",
            "language": "python",
            "tags": ["qa"],
            "has_solution": True,
            "task_binary": b"task-a",
        },
        {
            "source": "source-a",
            "path": "task-b",
            "family": "answers",
            "converter": "answer",
            "mode": "exact",
            "dockerfile_id": "python",
            "language": "python",
            "tags": ["qa"],
            "has_solution": True,
            "task_binary": b"task-b",
        },
        {
            "source": "source-b",
            "path": "task-c",
            "family": "tests",
            "converter": "tests",
            "mode": "script",
            "dockerfile_id": "ubuntu",
            "language": "go",
            "tags": ["code"],
            "has_solution": False,
            "task_binary": b"task-c",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tasks / "part-00000.parquet")
    manifest = {
        "dockerfiles": {
            "python": {"base_image": "python:3.12", "tasks": 2, "converters": {}, "sources": {}},
            "ubuntu": {"base_image": "ubuntu:24.04", "tasks": 1, "converters": {}, "sources": {}},
        }
    }
    (clean / "manifest.json").write_text(json.dumps(manifest))

    output = tmp_path / "output"
    result = MODULE.prepare_data(StoragePath(str(clean)), output, samples_per_group=1)
    viewer = pq.read_table(output / "tasks.parquet")
    sampled = viewer.select(["mode", "dockerfile_id", "task_binary"]).to_pylist()

    assert result == {"tasks": 2, "groups": 2}
    assert {(row["mode"], row["dockerfile_id"]) for row in sampled} == {
        ("exact", "python"),
        ("script", "ubuntu"),
    }
    assert all(row["task_binary"].startswith(b"task-") for row in sampled)
    assert json.loads((output / "manifest.json").read_text()) == manifest
