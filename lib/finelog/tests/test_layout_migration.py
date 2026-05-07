# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from finelog.rpc import logging_pb2
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.layout_migration import (
    LOG_NAMESPACE_DIR,
    SENTINEL_FILENAME,
    SENTINEL_VERSION,
    migrate_to_namespaced_layout,
)


def _write_flat_segment(data_dir: Path, name: str, payload: bytes = b"placeholder") -> Path:
    # Stub bytes; the migration only renames files.
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
        assert (log_dir / name).read_bytes() == body
        assert not (tmp_path / name).exists()

    sentinel = _read_sentinel(tmp_path)
    assert sentinel["state"] == "done"


def test_idempotent_when_done_state(tmp_path: Path):
    migrate_to_namespaced_layout(tmp_path)
    # Plant a flat file that would be migrated if the fast path didn't
    # fire; assert it stays put.
    sentinel_first = _read_sentinel(tmp_path)
    _write_flat_segment(tmp_path, "tmp_0000000000000000099.parquet", b"should-stay")

    migrate_to_namespaced_layout(tmp_path)

    sentinel_second = _read_sentinel(tmp_path)
    assert sentinel_first == sentinel_second
    assert (tmp_path / "tmp_0000000000000000099.parquet").exists()
    assert not (tmp_path / LOG_NAMESPACE_DIR / "tmp_0000000000000000099.parquet").exists()


def test_existing_log_dir_no_flat_files_is_idempotent(tmp_path: Path):
    log_dir = tmp_path / LOG_NAMESPACE_DIR
    log_dir.mkdir()
    (log_dir / "logs_0000000000000000001.parquet").write_bytes(b"already-here")

    migrate_to_namespaced_layout(tmp_path)

    sentinel = _read_sentinel(tmp_path)
    assert sentinel["state"] == "done"
    assert (log_dir / "logs_0000000000000000001.parquet").read_bytes() == b"already-here"


def test_resume_after_crash_with_in_progress_sentinel(tmp_path: Path):
    log_dir = tmp_path / LOG_NAMESPACE_DIR
    log_dir.mkdir()

    # File A already moved to log/, file B still flat.
    (log_dir / "tmp_0000000000000000001.parquet").write_bytes(b"A")
    _write_flat_segment(tmp_path, "tmp_0000000000000000002.parquet", b"B")

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
    """Cross-mount rename is not atomic on POSIX; refuse if st_dev differs."""
    _write_flat_segment(tmp_path, "tmp_0000000000000000001.parquet")

    real_stat = os.stat
    log_dir = tmp_path / LOG_NAMESPACE_DIR

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
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
    """End-to-end: pre-existing flat files migrate cleanly, then a fresh
    store over the same dir reads them back.

    The flat layout pre-dates the per-namespace dir convention by enough
    that segments were written with the legacy ``tmp_*`` / ``logs_*``
    prefixes, not today's ``seg_L<n>_*``. The layout migration only
    knows the legacy prefixes; migration 0003 rewrites them to the
    leveled scheme on the next store boot. We synthesize a legacy file
    by hand here rather than asking the new code path to produce it.
    """
    data_dir = tmp_path / "store"
    data_dir.mkdir()

    # Build one legacy logs_*.parquet that contains a single appendable
    # log row in today's schema. Using the new code path then renaming
    # would defeat the point of the test.
    s_temp = DuckDBLogStore(log_dir=tmp_path / "_seed")
    entry = logging_pb2.LogEntry(source="stdout", data="hello")
    entry.timestamp.epoch_ms = 1
    s_temp.append("/k", [entry])
    s_temp._force_flush()
    seed_dir = tmp_path / "_seed" / LOG_NAMESPACE_DIR
    seg = next(seed_dir.glob("seg_L0_*.parquet"))
    legacy_name = "logs_0000000000000000001.parquet"
    (data_dir / legacy_name).write_bytes(seg.read_bytes())
    s_temp.close()

    migrate_to_namespaced_layout(data_dir)
    s2 = DuckDBLogStore(log_dir=data_dir)
    try:
        # Migration 0003 renamed logs_*.parquet -> seg_L1_*.parquet.
        log_dir = data_dir / LOG_NAMESPACE_DIR
        assert sorted(p.name for p in log_dir.glob("seg_L1_*.parquet"))
        result = s2.get_logs("/k")
        assert [e.data for e in result.entries] == ["hello"]
    finally:
        s2.close()
