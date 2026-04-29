# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flat → per-namespace layout migration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from finelog.store.layout_migration import (
    LOG_NAMESPACE_DIR,
    SENTINEL_FILENAME,
    SENTINEL_VERSION,
    migrate_to_namespaced_layout,
)


def _write_flat_segment(data_dir: Path, name: str, payload: bytes = b"placeholder") -> Path:
    """Drop a stub Parquet file directly under ``data_dir`` (no real parquet
    content; the migration only renames bytes)."""
    p = data_dir / name
    p.write_bytes(payload)
    return p


def _read_sentinel(data_dir: Path) -> dict:
    return json.loads((data_dir / SENTINEL_FILENAME).read_text())


def test_fresh_install_creates_log_dir_and_sentinel(tmp_path: Path):
    migrate_to_namespaced_layout(tmp_path)
    assert (tmp_path / LOG_NAMESPACE_DIR).is_dir()
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["version"] == SENTINEL_VERSION
    assert sentinel["state"] == "done"


def test_cold_start_with_flat_files_migrates_into_log_dir(tmp_path: Path):
    flat_files = {
        "tmp_0000000000000000001.parquet": b"tmp1",
        "tmp_0000000000000000002.parquet": b"tmp2",
        "logs_0000000000000000003.parquet": b"logs3",
    }
    for name, body in flat_files.items():
        _write_flat_segment(tmp_path, name, body)

    migrate_to_namespaced_layout(tmp_path)

    log_dir = tmp_path / LOG_NAMESPACE_DIR
    assert log_dir.is_dir()
    for name, body in flat_files.items():
        # Files moved to subdir.
        assert (log_dir / name).read_bytes() == body
        # Originals are gone.
        assert not (tmp_path / name).exists()

    sentinel = _read_sentinel(tmp_path)
    assert sentinel["state"] == "done"


def test_idempotent_when_done_state(tmp_path: Path):
    migrate_to_namespaced_layout(tmp_path)
    # A subsequent run on the fast path must not scan or modify the dir.
    # Plant a flat file that *would* be migrated if the fast path didn't fire,
    # and assert it's still there afterward (proving no scan happened).
    sentinel_first = _read_sentinel(tmp_path)
    _write_flat_segment(tmp_path, "tmp_0000000000000000099.parquet", b"should-stay")

    migrate_to_namespaced_layout(tmp_path)

    sentinel_second = _read_sentinel(tmp_path)
    assert sentinel_first == sentinel_second
    # Flat file untouched: fast path skipped the directory scan entirely.
    assert (tmp_path / "tmp_0000000000000000099.parquet").exists()
    assert not (tmp_path / LOG_NAMESPACE_DIR / "tmp_0000000000000000099.parquet").exists()


def test_existing_log_dir_no_flat_files_is_idempotent(tmp_path: Path):
    """A directory already in the per-namespace shape (with the sentinel
    missing — e.g. created manually by an admin) must be stamped done
    without moving anything."""
    log_dir = tmp_path / LOG_NAMESPACE_DIR
    log_dir.mkdir()
    (log_dir / "logs_0000000000000000001.parquet").write_bytes(b"already-here")

    migrate_to_namespaced_layout(tmp_path)

    sentinel = _read_sentinel(tmp_path)
    assert sentinel["state"] == "done"
    assert (log_dir / "logs_0000000000000000001.parquet").read_bytes() == b"already-here"


def test_resume_after_crash_with_in_progress_sentinel(tmp_path: Path):
    """A crash mid-migration leaves the sentinel in ``in-progress`` and some
    files moved, others not. Re-running converges."""
    log_dir = tmp_path / LOG_NAMESPACE_DIR
    log_dir.mkdir()

    # Simulate: file A already moved to log/, file B still flat.
    (log_dir / "tmp_0000000000000000001.parquet").write_bytes(b"A")
    _write_flat_segment(tmp_path, "tmp_0000000000000000002.parquet", b"B")

    # Plant an in-progress sentinel.
    (tmp_path / SENTINEL_FILENAME).write_text(
        json.dumps({"version": 1, "state": "in-progress", "started_at": 0, "finished_at": None}) + "\n"
    )

    migrate_to_namespaced_layout(tmp_path)

    assert (log_dir / "tmp_0000000000000000001.parquet").read_bytes() == b"A"
    assert (log_dir / "tmp_0000000000000000002.parquet").read_bytes() == b"B"
    assert not (tmp_path / "tmp_0000000000000000002.parquet").exists()
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["state"] == "done"


def test_resume_with_size_mismatch_aborts_loudly(tmp_path: Path):
    """If a flat file and its dst-side counterpart exist but have different
    sizes, refuse to delete data — abort with a clear error."""
    log_dir = tmp_path / LOG_NAMESPACE_DIR
    log_dir.mkdir()
    (log_dir / "tmp_0000000000000000001.parquet").write_bytes(b"AAAA")
    _write_flat_segment(tmp_path, "tmp_0000000000000000001.parquet", b"BB")

    (tmp_path / SENTINEL_FILENAME).write_text(
        json.dumps({"version": 1, "state": "in-progress", "started_at": 0, "finished_at": None}) + "\n"
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        migrate_to_namespaced_layout(tmp_path)


def test_log_dir_exists_as_file_is_refused(tmp_path: Path):
    (tmp_path / LOG_NAMESPACE_DIR).write_bytes(b"not-a-dir")
    _write_flat_segment(tmp_path, "tmp_0000000000000000001.parquet")

    with pytest.raises(RuntimeError, match="not a directory"):
        migrate_to_namespaced_layout(tmp_path)


def test_cross_filesystem_data_dir_is_refused(tmp_path: Path):
    """If ``{data_dir}`` and ``{data_dir}/log`` resolve to different
    ``st_dev`` values, the migration must refuse — cross-mount rename is not
    atomic on POSIX. We can't easily create a real cross-mount in a unit
    test, so we patch ``os.stat`` to return divergent ``st_dev`` values."""
    _write_flat_segment(tmp_path, "tmp_0000000000000000001.parquet")

    real_stat = os.stat
    log_dir = tmp_path / LOG_NAMESPACE_DIR

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        # Force the log subdir onto a different "device" than the parent.
        if Path(path) == log_dir:
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev + 1,
                    result.st_nlink,
                    result.st_uid,
                    result.st_gid,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )
        return result

    with patch("finelog.store.layout_migration.os.stat", side_effect=fake_stat):
        with pytest.raises(RuntimeError, match="different filesystems"):
            migrate_to_namespaced_layout(tmp_path)


def test_sentinel_records_start_and_end_times(tmp_path: Path):
    _write_flat_segment(tmp_path, "tmp_0000000000000000001.parquet")
    migrate_to_namespaced_layout(tmp_path)

    s = _read_sentinel(tmp_path)
    assert s["state"] == "done"
    assert isinstance(s["started_at"], int) and s["started_at"] > 0
    assert isinstance(s["finished_at"], int) and s["finished_at"] >= s["started_at"]


def test_only_known_globs_are_migrated(tmp_path: Path):
    """Files outside ``tmp_*.parquet`` / ``logs_*.parquet`` are left in
    place; only the two known globs are walked."""
    _write_flat_segment(tmp_path, "tmp_0000000000000000001.parquet", b"keep-moves")
    other = tmp_path / "metadata.duckdb"
    other.write_bytes(b"db")
    hidden = tmp_path / ".something"
    hidden.write_bytes(b"hidden")

    migrate_to_namespaced_layout(tmp_path)

    log_dir = tmp_path / LOG_NAMESPACE_DIR
    assert (log_dir / "tmp_0000000000000000001.parquet").exists()
    assert other.read_bytes() == b"db"
    assert hidden.read_bytes() == b"hidden"


def test_full_store_round_trip_after_migration(tmp_path: Path):
    """End-to-end: pre-existing flat files are migrated, then a fresh store
    over the same dir reads them back via the per-namespace layout."""
    # Use the real store to produce a flat-layout dir, then "downgrade" it
    # by moving its contents back to the flat shape and removing the
    # sentinel — simulating an upgrade from an old flat-layout deployment.
    from finelog.rpc import logging_pb2
    from finelog.store.duckdb_store import DuckDBLogStore

    data_dir = tmp_path / "store"

    # 1. Bootstrap: write some logs through the new code path.
    s1 = DuckDBLogStore(log_dir=data_dir)
    entry = logging_pb2.LogEntry(source="stdout", data="hello")
    entry.timestamp.epoch_ms = 1
    s1.append("/k", [entry])
    s1._force_flush()
    s1.close()

    # 2. Manually demote to flat layout: move every file out of log/ to the
    # parent. The store doesn't write the sentinel today (only the migration
    # does), so there's nothing to remove there — but we still drop the dir.
    log_dir = data_dir / LOG_NAMESPACE_DIR
    for f in list(log_dir.iterdir()):
        f.rename(data_dir / f.name)
    log_dir.rmdir()
    sentinel = data_dir / SENTINEL_FILENAME
    if sentinel.exists():
        sentinel.unlink()

    # Confirm we're in the flat shape.
    assert any(p.name.startswith("tmp_") or p.name.startswith("logs_") for p in data_dir.iterdir())

    # 3. Migrate, then re-open the store. The data is reachable.
    migrate_to_namespaced_layout(data_dir)
    s2 = DuckDBLogStore(log_dir=data_dir)
    try:
        result = s2.get_logs("/k")
        assert [e.data for e in result.entries] == ["hello"]
    finally:
        s2.close()
