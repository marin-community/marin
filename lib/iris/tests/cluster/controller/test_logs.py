# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for log backends: standalone LogStore (worker) and ControllerDB (controller).

Both implement the same log interface (append, get_logs, get_logs_by_prefix, has_logs).
Shared tests are parameterized over both backends; backend-specific tests (persistence,
eviction internals) are tested separately.
"""

import threading
from pathlib import Path
from typing import Protocol

import pytest

from iris.cluster.controller.db import ControllerDB
from iris.cluster.log_store import LogStore, LogReadResult, task_log_key
from iris.cluster.types import JobName, TaskAttempt
from iris.rpc import logging_pb2


def _make_entry(data: str, epoch_ms: int = 0) -> logging_pb2.LogEntry:
    entry = logging_pb2.LogEntry(source="stdout", data=data)
    entry.timestamp.epoch_ms = epoch_ms
    return entry


class LogBackend(Protocol):
    """Shared interface tested across LogStore and ControllerDB."""

    def append(self, key: str, entries: list) -> None: ...
    def get_logs(self, key: str, **kwargs) -> LogReadResult: ...
    def get_logs_by_prefix(self, prefix: str, **kwargs) -> LogReadResult: ...
    def has_logs(self, key: str) -> bool: ...


TASK_ID = JobName.from_wire("/job/test/task/0")
KEY = task_log_key(TaskAttempt(task_id=TASK_ID, attempt_id=0))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def log_store():
    """Standalone LogStore (worker-side)."""
    store = LogStore()
    yield store
    store.close()


@pytest.fixture()
def controller_db(tmp_path):
    """ControllerDB with log methods (controller-side)."""
    db = ControllerDB(db_path=tmp_path / "controller.sqlite3")
    yield db
    db.close()


@pytest.fixture(params=["log_store", "controller_db"])
def backend(request) -> LogBackend:
    """Parameterized fixture yielding both log backends."""
    return request.getfixturevalue(request.param)


# =============================================================================
# Shared interface tests (run against both backends)
# =============================================================================


def test_get_logs_tail_returns_last_n(backend: LogBackend):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    backend.append(KEY, entries)

    result = backend.get_logs(KEY, max_lines=10, tail=True)
    assert len(result.entries) == 10
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(90, 100)]


def test_get_logs_tail_with_substring_filter(backend: LogBackend):
    entries = [_make_entry(f"{'ERROR' if i % 10 == 0 else 'INFO'}: msg-{i}", epoch_ms=i) for i in range(100)]
    backend.append(KEY, entries)

    result = backend.get_logs(KEY, max_lines=3, tail=True, substring_filter="ERROR")
    assert len(result.entries) == 3
    assert all("ERROR" in e.data for e in result.entries)
    assert [e.data for e in result.entries] == ["ERROR: msg-70", "ERROR: msg-80", "ERROR: msg-90"]


def test_get_logs_tail_with_since_ms(backend: LogBackend):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    backend.append(KEY, entries)

    result = backend.get_logs(KEY, max_lines=5, tail=True, since_ms=94)
    assert [e.data for e in result.entries] == ["line-95", "line-96", "line-97", "line-98", "line-99"]


def test_get_logs_tail_empty(backend: LogBackend):
    result = backend.get_logs(KEY, max_lines=10, tail=True)
    assert result.entries == []
    assert result.cursor == 0


def test_get_logs_forward_unchanged(backend: LogBackend):
    """Forward mode (tail=False) still returns first N lines."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    backend.append(KEY, entries)

    result = backend.get_logs(KEY, max_lines=10, tail=False)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(10)]


def test_cursor_skips_already_seen(backend: LogBackend):
    """Append 1000 lines, use cursor from first 990 to get last 10."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(1000)]
    backend.append(KEY, entries)

    result_head = backend.get_logs(KEY, max_lines=990)
    assert len(result_head.entries) == 990

    result = backend.get_logs(KEY, cursor=result_head.cursor)
    assert len(result.entries) == 10
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(990, 1000)]


def test_cursor_round_trip(backend: LogBackend):
    """Append -> read -> append more -> read(cursor=prev) returns only new."""
    entries1 = [_make_entry(f"batch1-{i}", epoch_ms=i) for i in range(10)]
    backend.append(KEY, entries1)

    result1 = backend.get_logs(KEY)
    assert len(result1.entries) == 10
    cursor = result1.cursor

    entries2 = [_make_entry(f"batch2-{i}", epoch_ms=100 + i) for i in range(5)]
    backend.append(KEY, entries2)

    result2 = backend.get_logs(KEY, cursor=cursor)
    assert len(result2.entries) == 5
    assert [e.data for e in result2.entries] == [f"batch2-{i}" for i in range(5)]


def test_has_logs(backend: LogBackend):
    assert not backend.has_logs(KEY)
    backend.append(KEY, [_make_entry("hello")])
    assert backend.has_logs(KEY)


def test_get_logs_by_prefix_returns_all_matching(backend: LogBackend):
    """Prefix query returns entries from all matching keys in id order."""
    t0 = JobName.from_wire("/job/test/task/0")
    t1 = JobName.from_wire("/job/test/task/1")

    backend.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry("t0-a0-line0", epoch_ms=1)])
    backend.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=1)), [_make_entry("t0-a1-line0", epoch_ms=2)])
    backend.append(task_log_key(TaskAttempt(task_id=t1, attempt_id=0)), [_make_entry("t1-a0-line0", epoch_ms=3)])

    result = backend.get_logs_by_prefix("/job/test/")
    assert len(result.entries) == 3
    assert [e.data for e in result.entries] == ["t0-a0-line0", "t0-a1-line0", "t1-a0-line0"]
    assert [e.attempt_id for e in result.entries] == [0, 1, 0]


def test_get_logs_by_prefix_cursor_continuation(backend: LogBackend):
    """Cursor-based continuation with prefix returns no duplicates."""
    t0 = JobName.from_wire("/job/test/task/0")
    backend.append(
        task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry(f"line-{i}", epoch_ms=i) for i in range(5)]
    )

    result1 = backend.get_logs_by_prefix("/job/test/", max_lines=3)
    assert len(result1.entries) == 3

    result2 = backend.get_logs_by_prefix("/job/test/", cursor=result1.cursor)
    assert len(result2.entries) == 2
    assert [e.data for e in result2.entries] == ["line-3", "line-4"]


def test_get_logs_by_prefix_isolation(backend: LogBackend):
    """Prefix /job/test/ does not match /job/testing/."""
    t_test = JobName.from_wire("/job/test/task/0")
    t_testing = JobName.from_wire("/job/testing/task/0")

    backend.append(task_log_key(TaskAttempt(task_id=t_test, attempt_id=0)), [_make_entry("test-line")])
    backend.append(task_log_key(TaskAttempt(task_id=t_testing, attempt_id=0)), [_make_entry("testing-line")])

    result = backend.get_logs_by_prefix("/job/test/")
    assert len(result.entries) == 1
    assert result.entries[0].data == "test-line"


def test_get_logs_by_prefix_shallow_excludes_children(backend: LogBackend):
    """shallow=True excludes logs from nested/child jobs."""
    t0 = JobName.from_wire("/job/parent/0")
    child_t0 = JobName.from_wire("/job/parent/child/0")

    backend.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry("parent-line", epoch_ms=1)])
    backend.append(task_log_key(TaskAttempt(task_id=child_t0, attempt_id=0)), [_make_entry("child-line", epoch_ms=2)])

    result_all = backend.get_logs_by_prefix("/job/parent/")
    assert len(result_all.entries) == 2

    result_shallow = backend.get_logs_by_prefix("/job/parent/", shallow=True)
    assert len(result_shallow.entries) == 1
    assert result_shallow.entries[0].data == "parent-line"


def test_get_logs_by_prefix_tail_returns_last_n(backend: LogBackend):
    """Tail mode with prefix returns the last N entries across all matching keys."""
    t0 = JobName.from_wire("/job/test/task/0")
    t1 = JobName.from_wire("/job/test/task/1")

    backend.append(
        task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry(f"t0-line-{i}", epoch_ms=i) for i in range(5)]
    )
    backend.append(
        task_log_key(TaskAttempt(task_id=t1, attempt_id=0)),
        [_make_entry(f"t1-line-{i}", epoch_ms=10 + i) for i in range(5)],
    )

    result = backend.get_logs_by_prefix("/job/test/", max_lines=3, tail=True)
    assert len(result.entries) == 3
    assert [e.data for e in result.entries] == ["t1-line-2", "t1-line-3", "t1-line-4"]


def test_get_logs_by_prefix_tail_with_substring_filter(backend: LogBackend):
    """Tail mode with prefix + substring filter returns last N matching entries."""
    t0 = JobName.from_wire("/job/test/task/0")

    entries = [_make_entry(f"{'ERROR' if i % 3 == 0 else 'INFO'}: msg-{i}", epoch_ms=i) for i in range(12)]
    backend.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), entries)

    result = backend.get_logs_by_prefix("/job/test/", max_lines=2, tail=True, substring_filter="ERROR")
    assert len(result.entries) == 2
    assert [e.data for e in result.entries] == ["ERROR: msg-6", "ERROR: msg-9"]


def test_get_logs_by_prefix_tail_with_shallow(backend: LogBackend):
    """Tail mode with shallow=True excludes child jobs and returns last N."""
    t0 = JobName.from_wire("/job/parent/0")
    child_t0 = JobName.from_wire("/job/parent/child/0")

    backend.append(
        task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry(f"parent-{i}", epoch_ms=i) for i in range(5)]
    )
    backend.append(
        task_log_key(TaskAttempt(task_id=child_t0, attempt_id=0)),
        [_make_entry(f"child-{i}", epoch_ms=10 + i) for i in range(5)],
    )

    result = backend.get_logs_by_prefix("/job/parent/", max_lines=2, tail=True, shallow=True)
    assert len(result.entries) == 2
    assert [e.data for e in result.entries] == ["parent-3", "parent-4"]


def test_substring_filter_escapes_like_wildcards(backend: LogBackend):
    """Percent and underscore in substring_filter match literally, not as SQL wildcards."""
    backend.append(
        KEY,
        [
            _make_entry("100% done", epoch_ms=1),
            _make_entry("a_b_c", epoch_ms=2),
            _make_entry("no match", epoch_ms=3),
        ],
    )

    result_pct = backend.get_logs(KEY, substring_filter="100%")
    assert [e.data for e in result_pct.entries] == ["100% done"]

    result_us = backend.get_logs(KEY, substring_filter="a_b")
    assert [e.data for e in result_us.entries] == ["a_b_c"]

    result_prefix = backend.get_logs_by_prefix("/job/test/", substring_filter="100%")
    assert [e.data for e in result_prefix.entries] == ["100% done"]


def test_cursor_with_since_ms(backend: LogBackend):
    """Combined cursor + since_ms filters correctly."""
    backend.append(KEY, [_make_entry(f"line-{i}", epoch_ms=i * 10) for i in range(10)])

    result1 = backend.get_logs(KEY, max_lines=5)
    cursor = result1.cursor

    result2 = backend.get_logs(KEY, cursor=cursor, since_ms=70)
    assert [e.data for e in result2.entries] == ["line-8", "line-9"]


# =============================================================================
# LogStore-specific tests
# =============================================================================


def test_logstore_clear_removes_logs(log_store: LogStore):
    entries = [_make_entry(f"line-{i}") for i in range(5)]
    log_store.append(KEY, entries)
    assert log_store.has_logs(KEY)

    log_store.clear(KEY)
    assert not log_store.has_logs(KEY)
    result = log_store.get_logs(KEY)
    assert result.entries == []
    assert result.cursor == 0


def test_logstore_persistent_log_dir(tmp_path: Path):
    """Append, close, reopen with same dir, read logs."""
    log_dir = tmp_path / "logs"
    store1 = LogStore(log_dir=log_dir)
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(5)]
    store1.append(KEY, entries)
    store1.close()

    store2 = LogStore(log_dir=log_dir)
    result = store2.get_logs(KEY)
    assert len(result.entries) == 5
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(5)]
    store2.close()


def test_logstore_has_logs_across_instances(tmp_path: Path):
    """has_logs works across store instances."""
    log_dir = tmp_path / "logs"
    store1 = LogStore(log_dir=log_dir)
    store1.append(KEY, [_make_entry("hello")])
    store1.close()

    store2 = LogStore(log_dir=log_dir)
    assert store2.has_logs(KEY)
    assert not store2.has_logs(task_log_key(TaskAttempt(task_id=TASK_ID, attempt_id=99)))
    store2.close()


def test_logstore_concurrent_tail(log_store: LogStore):
    """Tail reads survive concurrent appends."""
    errors: list[Exception] = []

    def writer():
        for i in range(500):
            log_store.append(KEY, [_make_entry(f"line-{i}", epoch_ms=i)])

    def reader():
        for _ in range(500):
            try:
                log_store.get_logs(KEY, max_lines=10, tail=True)
            except RuntimeError as e:
                errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == [], f"Concurrent access raised: {errors}"


def test_logstore_eviction_caps_total_rows():
    """Appending beyond max_records triggers eviction down to max_records // 2."""
    max_records = 50
    store = LogStore(max_records=max_records)
    try:
        entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(max_records)]
        store.append(KEY, entries)

        total = store._write_conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        assert total == max_records

        entries2 = [_make_entry(f"line2-{i}", epoch_ms=100 + i) for i in range(max_records)]
        store.append(KEY, entries2)

        total = store._write_conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        assert total <= max_records // 2 + 1
    finally:
        store.close()


# =============================================================================
# ControllerDB-specific log tests
# =============================================================================


def test_controller_db_clear_logs(controller_db: ControllerDB):
    entries = [_make_entry(f"line-{i}") for i in range(5)]
    controller_db.append(KEY, entries)
    assert controller_db.has_logs(KEY)

    controller_db.clear_logs(KEY)
    assert not controller_db.has_logs(KEY)
    result = controller_db.get_logs(KEY)
    assert result.entries == []
    assert result.cursor == 0


def test_controller_db_append_batch(controller_db: ControllerDB):
    """append_batch writes multiple keys atomically."""
    t0 = JobName.from_wire("/job/test/task/0")
    t1 = JobName.from_wire("/job/test/task/1")
    key0 = task_log_key(TaskAttempt(task_id=t0, attempt_id=0))
    key1 = task_log_key(TaskAttempt(task_id=t1, attempt_id=0))

    controller_db.append_batch(
        [
            (key0, [_make_entry("k0-line0", epoch_ms=1)]),
            (key1, [_make_entry("k1-line0", epoch_ms=2)]),
        ]
    )

    assert controller_db.has_logs(key0)
    assert controller_db.has_logs(key1)
    result = controller_db.get_logs_by_prefix("/job/test/")
    assert len(result.entries) == 2


def test_controller_db_eviction_caps_total_rows(tmp_path):
    """Appending beyond max_records triggers eviction down to max_records // 2."""
    max_records = 50
    db = ControllerDB(db_path=tmp_path / "evict_controller.sqlite3")
    # Override the cap and interval so we can trigger eviction with a small dataset.
    db._MAX_LOG_RECORDS = max_records
    db._eviction_check_interval = max_records // 10
    db._rows_since_eviction_check = 0
    try:
        entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(max_records)]
        db.append(KEY, entries)

        total = db._log_read_conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        assert total == max_records

        entries2 = [_make_entry(f"line2-{i}", epoch_ms=100 + i) for i in range(max_records)]
        db.append(KEY, entries2)

        total = db._log_read_conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        assert total <= max_records // 2 + 1
    finally:
        db.close()


def test_controller_db_log_cursor(controller_db: ControllerDB):
    """log_cursor provides incremental reads."""
    controller_db.append(KEY, [_make_entry(f"line-{i}", epoch_ms=i) for i in range(10)])

    cursor = controller_db.log_cursor(KEY)
    batch1 = cursor.read(max_entries=5)
    assert len(batch1) == 5
    assert [e.data for e in batch1] == [f"line-{i}" for i in range(5)]

    batch2 = cursor.read(max_entries=10)
    assert len(batch2) == 5
    assert [e.data for e in batch2] == [f"line-{i}" for i in range(5, 10)]

    batch3 = cursor.read()
    assert len(batch3) == 0
