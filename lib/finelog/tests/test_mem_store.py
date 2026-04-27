# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory LogStore (MemStore)."""

from __future__ import annotations

import pytest

from finelog.rpc import logging_pb2
from finelog.store.mem_store import MemStore

KEY = "/job/test/0:0"


def _entry(data: str, epoch_ms: int = 0) -> logging_pb2.LogEntry:
    e = logging_pb2.LogEntry(source="stdout", data=data)
    e.timestamp.epoch_ms = epoch_ms
    return e


@pytest.fixture()
def store():
    s = MemStore()
    yield s
    s.close()


def test_append_and_read_roundtrip(store: MemStore):
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(10)])
    result = store.get_logs(KEY)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(10)]
    assert result.cursor > 0


def test_tail_returns_last_n(store: MemStore):
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(50)])
    result = store.get_logs(KEY, max_lines=5, tail=True)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(45, 50)]


def test_cursor_skips_already_seen(store: MemStore):
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(10)])
    first = store.get_logs(KEY, max_lines=5)
    assert len(first.entries) == 5
    second = store.get_logs(KEY, cursor=first.cursor)
    assert [e.data for e in second.entries] == [f"line-{i}" for i in range(5, 10)]


def test_substring_filter(store: MemStore):
    store.append(KEY, [_entry(f"{'ERROR' if i % 3 == 0 else 'INFO'}: msg-{i}", epoch_ms=i) for i in range(9)])
    result = store.get_logs(KEY, substring_filter="ERROR")
    assert all("ERROR" in e.data for e in result.entries)
    assert len(result.entries) == 3


def test_since_ms(store: MemStore):
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i * 10) for i in range(10)])
    result = store.get_logs(KEY, since_ms=50)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(6, 10)]


def test_regex_pattern_returns_all_matching(store: MemStore):
    store.append("/job/test/0:0", [_entry("a", epoch_ms=1)])
    store.append("/job/test/1:0", [_entry("b", epoch_ms=2)])
    store.append("/job/other/0:0", [_entry("c", epoch_ms=3)])
    result = store.get_logs("/job/test/.*")
    assert sorted(e.data for e in result.entries) == ["a", "b"]


def test_regex_pattern_includes_key(store: MemStore):
    store.append("/job/test/0:7", [_entry("a", epoch_ms=1)])
    result = store.get_logs("/job/test/.*")
    assert len(result.entries) == 1
    assert result.entries[0].key == "/job/test/0:7"
    assert result.entries[0].attempt_id == 7


def test_exact_key_sets_attempt_id(store: MemStore):
    store.append("/job/test/0:3", [_entry("a")])
    result = store.get_logs("/job/test/0:3")
    assert result.entries[0].attempt_id == 3


def test_has_logs(store: MemStore):
    assert not store.has_logs(KEY)
    store.append(KEY, [_entry("hello")])
    assert store.has_logs(KEY)
    assert not store.has_logs("/job/test/0:99")


def test_cursor_object_advances(store: MemStore):
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(5)])
    cursor = store.cursor(KEY)
    first = cursor.read()
    assert len(first) == 5
    store.append(KEY, [_entry(f"new-{i}", epoch_ms=10 + i) for i in range(3)])
    second = cursor.read()
    assert [e.data for e in second] == ["new-0", "new-1", "new-2"]


def test_min_level_filter(store: MemStore):
    e_info = _entry("info", epoch_ms=1)
    e_info.level = logging_pb2.LOG_LEVEL_INFO
    e_err = _entry("err", epoch_ms=2)
    e_err.level = logging_pb2.LOG_LEVEL_ERROR
    store.append(KEY, [e_info, e_err])

    result = store.get_logs(KEY, min_level="ERROR")
    assert [e.data for e in result.entries] == ["err"]


def test_append_batch(store: MemStore):
    store.append_batch(
        [
            ("/job/a/0:0", [_entry("a1", epoch_ms=1)]),
            ("/job/b/0:0", [_entry("b1", epoch_ms=2)]),
        ]
    )
    assert store.has_logs("/job/a/0:0")
    assert store.has_logs("/job/b/0:0")


def test_max_rows_evicts_oldest_via_append():
    store = MemStore(max_rows=10)
    for i in range(25):
        store.append(KEY, [_entry(f"line-{i}", epoch_ms=i)])
    result = store.get_logs(KEY)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(15, 25)]
    store.close()


def test_max_rows_evicts_oldest_via_append_batch():
    store = MemStore(max_rows=10)
    store.append_batch([(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(25)])])
    result = store.get_logs(KEY)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(15, 25)]
    store.close()


def test_max_rows_none_is_unbounded():
    store = MemStore(max_rows=None)
    store.append(KEY, [_entry(f"line-{i}", epoch_ms=i) for i in range(5000)])
    result = store.get_logs(KEY, max_lines=10000)
    assert len(result.entries) == 5000
    store.close()
