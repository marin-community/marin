# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the guarded fsspec factory: the url_to_fs/open_url/filesystem entry
points, unique_temp_path, atomic_rename, and fetch_file_atomic."""

from pathlib import Path

import pytest
from rigging.filesystem.cross_region import CrossRegionGuardedFS
from rigging.filesystem.factory import (
    atomic_rename,
    fetch_file_atomic,
    filesystem,
    open_url,
    unique_temp_path,
    url_to_fs,
)


def test_unique_temp_path_produces_distinct_paths():
    """Each call to unique_temp_path returns a different path."""
    paths = {unique_temp_path("/some/output.txt") for _ in range(10)}
    assert len(paths) == 10
    for p in paths:
        assert p.startswith("/some/output.txt.tmp.")


def test_atomic_rename_uses_unique_temp_paths(tmp_path):
    """Concurrent atomic_rename calls use distinct temp paths (UUID collision avoidance)."""
    output = str(tmp_path / "out.txt")
    observed_temps = []

    for _ in range(5):
        with atomic_rename(output) as temp_path:
            observed_temps.append(temp_path)
            Path(temp_path).write_text("data")

    assert len(set(observed_temps)) == 5, "Each call should produce a unique temp path"
    for tp in observed_temps:
        assert ".tmp." in tp


def test_atomic_rename_cleans_up_on_error(tmp_path):
    """Temp file is removed when the context raises an exception."""
    output = str(tmp_path / "out.txt")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_rename(output) as temp_path:
            Path(temp_path).write_text("bad")
            raise RuntimeError("boom")

    assert not Path(temp_path).exists()
    assert not Path(output).exists()


# ---------------------------------------------------------------------------
# fetch_file_atomic
# ---------------------------------------------------------------------------


def test_fetch_file_atomic_copies_source(tmp_path):
    src = tmp_path / "remote" / "tokenizer.json"
    src.parent.mkdir(parents=True)
    src.write_bytes(b'{"version": 1}')
    dest = tmp_path / "cache" / "tokenizer.json"
    dest.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(src), str(dest)) is True
    assert dest.read_bytes() == b'{"version": 1}'


def test_fetch_file_atomic_missing_source_returns_false(tmp_path):
    dest = tmp_path / "cache" / "tokenizer.json"
    dest.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(tmp_path / "remote" / "absent.json"), str(dest)) is False
    assert not dest.exists()


def test_fetch_file_atomic_failure_preserves_dest_and_cleans_temp(tmp_path, monkeypatch):
    # Regression for marin#7167: a fetch that dies mid-finalize must not leave a
    # partial file at dest (poisoning a shared cache) and must not orphan its temp.
    src = tmp_path / "remote" / "tokenizer.json"
    src.parent.mkdir(parents=True)
    src.write_bytes(b'{"version": 2}')
    dest = tmp_path / "cache" / "tokenizer.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'{"version": 1}')  # a previous complete file

    def boom(*args, **kwargs):
        raise OSError("simulated failure finalizing the fetch")

    monkeypatch.setattr("os.replace", boom)

    with pytest.raises(OSError, match="simulated failure"):
        fetch_file_atomic(str(src), str(dest))

    assert dest.read_bytes() == b'{"version": 1}'
    assert [p.name for p in dest.parent.iterdir()] == ["tokenizer.json"]


# ---------------------------------------------------------------------------
# Guarded entry point tests
# ---------------------------------------------------------------------------


def test_url_to_fs_does_not_wrap_local(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    fs, _path = url_to_fs(str(test_file))
    assert not isinstance(fs, CrossRegionGuardedFS)


def test_open_url_local_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    result = open_url(str(test_file), "r")
    with result as f:
        assert f.read() == "hello"


def test_filesystem_local():
    fs = filesystem("file")
    assert not isinstance(fs, CrossRegionGuardedFS)
