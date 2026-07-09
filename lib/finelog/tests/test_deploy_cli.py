# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for namespace discovery + view registration in ``finelog gcs-query``.

The CLI command itself opens a real connection (gcsfs in production); these
tests cover the testable seams against a local parquet directory.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import duckdb
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finelog.deploy.cli import (
    _list_namespace_dirs,
    _list_namespace_segments,
    _register_namespace_views,
)


def _write_segment(path: Path, level: int, seq: int, rows: list[dict[str, object]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"seg_L{level}_{seq:019d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out)
    return out


@contextmanager
def _stub_file_timestamps(timestamps: dict[str, int]):
    """Stub _info_time_created_ms to return fixed epoch_ms values keyed by filename.

    os.utime controls mtime but not ctime on Linux (the kernel always resets ctime
    to now on any metadata change), so tests that rely on time-window filtering must
    inject timestamps at the abstraction boundary instead of via the filesystem.
    """

    def _stubbed(info: dict) -> int | None:
        return timestamps.get(Path(str(info.get("name", ""))).name)

    with patch("finelog.deploy.cli._info_time_created_ms", side_effect=_stubbed):
        yield


def test_list_namespace_dirs_keeps_only_dirs_with_segments(tmp_path: Path) -> None:
    """Discovery returns directories that actually hold ``seg_L*.parquet`` —
    not stray top-level files (REGISTRY.json) and not empty subdirs."""
    _write_segment(tmp_path / "log", level=1, seq=1, rows=[{"key": "/u/0", "data": "x"}])
    _write_segment(tmp_path / "iris.worker", level=1, seq=1, rows=[{"worker_id": "w-1"}])
    (tmp_path / "REGISTRY.json").write_text("{}")
    (tmp_path / "empty_ns").mkdir()

    fs, _ = fsspec.url_to_fs(str(tmp_path))
    assert _list_namespace_dirs(str(tmp_path), fs) == ["iris.worker", "log"]


def test_register_namespace_views_query_roundtrip(tmp_path: Path) -> None:
    """A registered view round-trips through SELECT, including dotted names
    that require the double-quoted-identifier escape."""
    _write_segment(tmp_path / "log", level=1, seq=1, rows=[{"key": "/u/0:0", "data": "hello"}])
    _write_segment(tmp_path / "iris.worker", level=1, seq=1, rows=[{"worker_id": "w-1", "rss_bytes": 42}])

    conn = duckdb.connect()
    _register_namespace_views(conn, str(tmp_path), ["log", "iris.worker"])
    assert conn.execute("SELECT key, data FROM log").fetchall() == [("/u/0:0", "hello")]
    assert conn.execute('SELECT worker_id, rss_bytes FROM "iris.worker"').fetchall() == [("w-1", 42)]


def test_list_namespace_segments_filters_by_time_created(tmp_path: Path) -> None:
    """``created_since_ms`` drops files older than the threshold."""
    ns_dir = tmp_path / "log"
    old = _write_segment(ns_dir, level=1, seq=1, rows=[{"x": 1}])
    new = _write_segment(ns_dir, level=1, seq=2, rows=[{"x": 2}])

    fs, _ = fsspec.url_to_fs(str(tmp_path))
    with _stub_file_timestamps({old.name: 1_700_000_000_000, new.name: 1_800_000_000_000}):
        after = _list_namespace_segments(
            str(tmp_path), "log", fs, created_since_ms=1_750_000_000 * 1000, created_until_ms=None
        )
    assert [p.split("/")[-1] for p in after] == [new.name]


def test_register_namespace_views_with_time_window_only_reads_matching_files(tmp_path: Path) -> None:
    """View body contains only files passing the time-window filter — the
    SELECT must not return rows from the dropped file."""
    ns_dir = tmp_path / "log"
    old = _write_segment(ns_dir, level=1, seq=1, rows=[{"key": "/old", "data": "old"}])
    new = _write_segment(ns_dir, level=1, seq=2, rows=[{"key": "/new", "data": "new"}])

    fs, _ = fsspec.url_to_fs(str(tmp_path))
    conn = duckdb.connect()
    with _stub_file_timestamps({old.name: 1_700_000_000_000, new.name: 1_800_000_000_000}):
        _register_namespace_views(
            conn,
            str(tmp_path),
            ["log"],
            fs=fs,
            created_since_ms=1_750_000_000 * 1000,
            created_until_ms=None,
        )
    assert conn.execute("SELECT key FROM log").fetchall() == [("/new",)]


def test_register_namespace_views_empty_filter_skips_view(tmp_path: Path) -> None:
    """When the filter drops every file, no view is created and a SELECT
    fails — better than silently scanning all files."""
    f = _write_segment(tmp_path / "log", level=1, seq=1, rows=[{"key": "/k", "data": "v"}])

    fs, _ = fsspec.url_to_fs(str(tmp_path))
    conn = duckdb.connect()
    with _stub_file_timestamps({f.name: 1_700_000_000_000}):
        _register_namespace_views(
            conn,
            str(tmp_path),
            ["log"],
            fs=fs,
            created_since_ms=1_800_000_000 * 1000,
            created_until_ms=None,
        )
    with pytest.raises(duckdb.CatalogException):
        conn.execute("SELECT * FROM log").fetchall()
