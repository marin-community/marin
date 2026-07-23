# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import os
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.harbor.task_archive import extract_task_archive, write_task_archive


def _write_task(root: Path, name: str, contents: str) -> Path:
    task = root / name
    task.mkdir(parents=True)
    (task / "instruction.md").write_text(contents)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM busybox\n")
    return task


def test_task_archive_round_trips_nested_task_content(tmp_path):
    tasks = tmp_path / "tasks"
    _write_task(tasks, "first", "first task")
    _write_task(tasks, "nested/second", "second task")
    archive = write_task_archive(tasks, tmp_path / "tasks.parquet")

    extracted = extract_task_archive(archive.parquet_path, tmp_path / "extracted")

    assert archive.task_count == 2
    assert [path.relative_to(tmp_path / "extracted").as_posix() for path in extracted] == ["first", "nested/second"]
    assert (tmp_path / "extracted/nested/second/instruction.md").read_text() == "second task"


def test_task_archive_bytes_ignore_host_file_timestamps(tmp_path):
    tasks = tmp_path / "tasks"
    task = _write_task(tasks, "first", "first task")
    first = write_task_archive(tasks, tmp_path / "first.parquet")

    instruction = task / "instruction.md"
    os.utime(instruction, (1_700_000_000, 1_700_000_000))
    second = write_task_archive(tasks, tmp_path / "second.parquet")

    assert first.parquet_path.read_bytes() == second.parquet_path.read_bytes()


def test_task_archive_rejects_path_traversal_members(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        member = tarfile.TarInfo("../outside")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    table = pa.Table.from_pylist([{"path": "task", "task_binary": payload.getvalue()}])
    archive_path = tmp_path / "unsafe.parquet"
    pq.write_table(table, archive_path)

    with pytest.raises(ValueError, match="Unsafe task archive member"):
        extract_task_archive(archive_path, tmp_path / "output")
