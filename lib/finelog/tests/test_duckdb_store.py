# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the DuckDB-backed LogStore."""

from __future__ import annotations

from pathlib import Path

import pytest
from finelog.rpc import logging_pb2
from finelog.store.duckdb_store import DuckDBLogStore

KEY = "/job/test/0:0"


def _entry(data: str, epoch_ms: int = 0) -> logging_pb2.LogEntry:
    e = logging_pb2.LogEntry(source="stdout", data=data)
    e.timestamp.epoch_ms = epoch_ms
    return e


def _make_store(**kwargs) -> DuckDBLogStore:
    """Wrap get_logs to force a sync pending->chunks step before reads."""
    store = DuckDBLogStore(**kwargs)
    original = store.get_logs

    def get_logs(*args, **kw):
        store._compact_step()
        return original(*args, **kw)

    store.get_logs = get_logs  # type: ignore[method-assign]
    return store


@pytest.fixture()
def store():
    s = _make_store()
    yield s
    s.close()


def test_append_and_read_roundtrip(store: DuckDBLogStore):
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(10)])
    result = store.get_logs(KEY)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(10)]


def test_tail_returns_last_n(store: DuckDBLogStore):
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(50)])
    result = store.get_logs(KEY, max_lines=5, tail=True)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(45, 50)]


def test_cursor_round_trip(store: DuckDBLogStore):
    store.append(KEY, [_entry(f"a-{i}", epoch_ms=i) for i in range(5)])
    first = store.get_logs(KEY)
    assert len(first.entries) == 5
    store.append(KEY, [_entry(f"b-{i}", epoch_ms=10 + i) for i in range(3)])
    second = store.get_logs(KEY, cursor=first.cursor)
    assert [e.data for e in second.entries] == ["b-0", "b-1", "b-2"]


def test_regex_pattern_query(store: DuckDBLogStore):
    store.append("/job/test/0:0", [_entry("a", epoch_ms=1)])
    store.append("/job/test/1:0", [_entry("b", epoch_ms=2)])
    store.append("/job/other/0:0", [_entry("c", epoch_ms=3)])
    result = store.get_logs("/job/test/.*")
    assert sorted(e.data for e in result.entries) == ["a", "b"]
    # attempt_id parsed from key suffix
    assert all(e.attempt_id == 0 for e in result.entries)


def test_flush_and_compaction(tmp_path: Path):
    data_dir = tmp_path / "logs"
    # The log namespace's parquet files live under {data_dir}/log/, not
    # directly under {data_dir}.
    namespace_dir = data_dir / "log"
    store = _make_store(log_dir=data_dir)
    try:
        for batch in range(3):
            store.append(KEY, [_entry(f"b{batch}-{i}", epoch_ms=batch * 10 + i) for i in range(5)])
            store._force_flush()
        assert len(sorted(namespace_dir.glob("tmp_*.parquet"))) == 3

        store._force_compaction()

        assert len(sorted(namespace_dir.glob("tmp_*.parquet"))) == 0
        assert len(sorted(namespace_dir.glob("logs_*.parquet"))) == 1

        result = store.get_logs(KEY)
        assert len(result.entries) == 15
    finally:
        store.close()


def test_persistent_log_dir(tmp_path: Path):
    log_dir = tmp_path / "logs"
    s1 = _make_store(log_dir=log_dir)
    s1.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(5)])
    s1.close()

    s2 = _make_store(log_dir=log_dir)
    try:
        result = s2.get_logs(KEY)
        assert len(result.entries) == 5
    finally:
        s2.close()


def test_substring_filter_escapes_wildcards(store: DuckDBLogStore):
    store.append(
        KEY,
        [
            _entry("100% done", epoch_ms=1),
            _entry("a_b_c", epoch_ms=2),
            _entry("plain", epoch_ms=3),
        ],
    )
    result = store.get_logs(KEY, substring_filter="100%")
    assert [e.data for e in result.entries] == ["100% done"]
    result_us = store.get_logs(KEY, substring_filter="a_b")
    assert [e.data for e in result_us.entries] == ["a_b_c"]
