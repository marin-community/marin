# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskforge.atomic_file import write_atomic


def test_write_atomic_replaces_content_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"old")
    write_atomic(path, b"new")
    assert path.read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_failed_write_atomic_keeps_the_old_file_and_removes_the_temporary_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"old")
    with pytest.raises(TypeError):
        write_atomic(path, "not bytes")  # type: ignore[arg-type]
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]
