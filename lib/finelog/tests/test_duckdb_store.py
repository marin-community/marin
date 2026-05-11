# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from finelog.rpc import logging_pb2
from finelog.store.duckdb_store import DuckDBLogStore

KEY = "/job/test/0:0"

# Each char is a regex metacharacter that, under the old auto-detect
# dispatch, silently turned literal-key reads into regex queries and
# dropped the row. Reads must return the literal entry under both EXACT
# (the in-process Python default) and PREFIX. Backslash is escaped twice
# for Python and once for the literal in the key.
_REGEX_METACHARS = list(".*+?[](){}^$|\\")


def _entry(data: str, epoch_ms: int = 0) -> logging_pb2.LogEntry:
    e = logging_pb2.LogEntry(source="stdout", data=data)
    e.timestamp.epoch_ms = epoch_ms
    return e


@pytest.fixture()
def store():
    s = DuckDBLogStore()
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


def test_regex_scope_query(store: DuckDBLogStore):
    store.append("/job/test/0:0", [_entry("a", epoch_ms=1)])
    store.append("/job/test/1:0", [_entry("b", epoch_ms=2)])
    store.append("/job/other/0:0", [_entry("c", epoch_ms=3)])
    result = store.get_logs("/job/test/.*", match_scope=logging_pb2.MATCH_SCOPE_REGEX)
    assert sorted(e.data for e in result.entries) == ["a", "b"]
    assert all(e.attempt_id == 0 for e in result.entries)


def test_prefix_scope_query(store: DuckDBLogStore):
    """PREFIX matches literal-prefix keys without regex semantics."""
    store.append("/job/test/0:0", [_entry("a", epoch_ms=1)])
    store.append("/job/test/0:1", [_entry("b", epoch_ms=2)])
    store.append("/job/other/0:0", [_entry("c", epoch_ms=3)])
    # PREFIX with delimiter only matches the task's attempts, not the sibling job.
    result = store.get_logs("/job/test/0:", match_scope=logging_pb2.MATCH_SCOPE_PREFIX)
    assert sorted(e.data for e in result.entries) == ["a", "b"]


def test_python_get_logs_defaults_to_exact(store: DuckDBLogStore):
    """The Python API default is EXACT: ``get_logs(key)`` reads only the
    literal row, not every key with ``key`` as a prefix.

    Regression for the case where the in-process default leaked PREFIX
    semantics into single-key helpers like ``has_logs`` and ``LogCursor``.
    """
    store.append("/job/a", [_entry("hit", epoch_ms=1)])
    store.append("/job/ab", [_entry("decoy", epoch_ms=2)])
    # No match_scope passed → in-process default EXACT. The decoy starts
    # with the queried key but is a different row.
    result = store.get_logs("/job/a")
    assert [e.data for e in result.entries] == ["hit"]


def test_literal_key_with_regex_metachar_under_default(store: DuckDBLogStore):
    """Regression for #5392: literal keys containing scientific-notation
    metacharacters must read correctly under the default EXACT scope."""
    store.append("/job/curation-9e+20", [_entry("hit", epoch_ms=1)])
    store.append("/job/curation-other", [_entry("miss", epoch_ms=2)])
    # No match_scope passed → EXACT; `+` matches itself, not "one or more e".
    result = store.get_logs("/job/curation-9e+20")
    assert [e.data for e in result.entries] == ["hit"]


@pytest.mark.parametrize("meta", _REGEX_METACHARS)
def test_literal_key_read_with_metachar(store: DuckDBLogStore, meta: str):
    """EXACT (the in-process default) and PREFIX must both return rows for
    keys containing every regex metacharacter — none of them should be
    reinterpreted as regex syntax on the read path."""
    key = f"/job/literal{meta}value:0"
    store.append(key, [_entry(f"line-with-{meta}", epoch_ms=1)])
    # Sentinel under a different key must not leak in.
    store.append("/job/literal-other:0", [_entry("decoy", epoch_ms=2)])

    # Default (EXACT) reads only the literal row.
    default = store.get_logs(key)
    assert [e.data for e in default.entries] == [f"line-with-{meta}"]

    # Explicit EXACT behaves identically.
    explicit = store.get_logs(key, match_scope=logging_pb2.MATCH_SCOPE_EXACT)
    assert [e.data for e in explicit.entries] == [f"line-with-{meta}"]


@pytest.mark.parametrize("meta", _REGEX_METACHARS)
def test_prefix_scope_returns_literal_key_with_metachar(store: DuckDBLogStore, meta: str):
    """PREFIX must treat the source as a literal prefix, not as regex."""
    key = f"/job/literal{meta}value:0"
    store.append(key, [_entry(f"prefix-hit-{meta}", epoch_ms=1)])
    store.append("/job/different-prefix:0", [_entry("decoy", epoch_ms=2)])

    prefix = f"/job/literal{meta}"
    result = store.get_logs(prefix, match_scope=logging_pb2.MATCH_SCOPE_PREFIX)
    assert [e.data for e in result.entries] == [f"prefix-hit-{meta}"]


def test_has_logs_is_exact(store: DuckDBLogStore):
    """``has_logs`` reads a single literal key. ``has_logs("/job/a")`` must
    not return True just because ``/job/ab`` exists.
    """
    store.append("/job/ab", [_entry("decoy", epoch_ms=1)])
    assert not store.has_logs("/job/a")
    store.append("/job/a", [_entry("hit", epoch_ms=2)])
    assert store.has_logs("/job/a")


def test_log_cursor_reads_single_key(store: DuckDBLogStore):
    """``LogStore.cursor(key)`` returns a stateful reader for a single key.
    Entries written to sibling keys (e.g. ``/k`` vs ``/ka``) must not leak
    into the cursor's stream.
    """
    store.append("/k", [_entry("k-1", epoch_ms=1)])
    store.append("/ka", [_entry("ka-1", epoch_ms=2)])
    cur = store.cursor("/k")
    assert [e.data for e in cur.read()] == ["k-1"]
    store.append("/k", [_entry("k-2", epoch_ms=3)])
    store.append("/ka", [_entry("ka-2", epoch_ms=4)])
    assert [e.data for e in cur.read()] == ["k-2"]


def test_empty_prefix_source_is_rejected(store: DuckDBLogStore):
    """An empty source under PREFIX would match every key in the store; the
    query layer must refuse it instead of returning the first page of every
    stream.
    """
    store.append("/job/a", [_entry("a", epoch_ms=1)])
    with pytest.raises(ValueError, match="source is required"):
        store.get_logs("", match_scope=logging_pb2.MATCH_SCOPE_PREFIX)


def test_flush_and_compaction(tmp_path: Path):
    data_dir = tmp_path / "logs"
    namespace_dir = data_dir / "log"
    store = DuckDBLogStore(log_dir=data_dir)
    try:
        for batch in range(3):
            store.append(KEY, [_entry(f"b{batch}-{i}", epoch_ms=batch * 10 + i) for i in range(5)])
            store._force_flush()
        assert len(sorted(namespace_dir.glob("seg_L0_*.parquet"))) == 3

        store._force_compaction()

        assert len(sorted(namespace_dir.glob("seg_L0_*.parquet"))) == 0
        assert len(sorted(namespace_dir.glob("seg_L1_*.parquet"))) == 1

        result = store.get_logs(KEY)
        assert len(result.entries) == 15
    finally:
        store.close()


def test_persistent_log_dir(tmp_path: Path):
    log_dir = tmp_path / "logs"
    s1 = DuckDBLogStore(log_dir=log_dir)
    s1.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(5)])
    s1.close()

    s2 = DuckDBLogStore(log_dir=log_dir)
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
