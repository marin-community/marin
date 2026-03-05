# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerLogStore with line-offset streaming."""

import re
import threading
from pathlib import Path

import pytest

from iris.cluster.controller.logs import ControllerLogStore
from iris.cluster.types import JobName
from iris.rpc import logging_pb2


def _make_entry(data: str, epoch_ms: int = 0) -> logging_pb2.LogEntry:
    entry = logging_pb2.LogEntry(source="stdout", data=data)
    entry.timestamp.epoch_ms = epoch_ms
    return entry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def log_store():
    store = ControllerLogStore()
    yield store
    store.close()


TASK_ID = JobName.from_wire("/job/test/task/0")


# =============================================================================
# Basic tail tests
# =============================================================================


def test_get_logs_tail_returns_last_n(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=10, tail=True)
    assert len(result.entries) == 10
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(90, 100)]


def test_get_logs_tail_chronological_order(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(50)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=5, tail=True)
    timestamps = [e.timestamp.epoch_ms for e in result.entries]
    assert timestamps == sorted(timestamps)


def test_get_logs_tail_with_regex(log_store: ControllerLogStore):
    entries = [_make_entry(f"{'ERROR' if i % 10 == 0 else 'INFO'}: msg-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=3, tail=True, regex_filter=re.compile("ERROR"))
    assert len(result.entries) == 3
    assert all("ERROR" in e.data for e in result.entries)
    assert [e.data for e in result.entries] == ["ERROR: msg-70", "ERROR: msg-80", "ERROR: msg-90"]


def test_get_logs_tail_with_since_ms(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=5, tail=True, since_ms=94)
    assert [e.data for e in result.entries] == ["line-95", "line-96", "line-97", "line-98", "line-99"]


def test_get_logs_tail_fewer_than_max(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(3)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=100, tail=True)
    assert len(result.entries) == 3


def test_get_logs_tail_empty_file(log_store: ControllerLogStore):
    result = log_store.get_logs(TASK_ID, 0, max_lines=10, tail=True)
    assert result.entries == []
    assert result.lines_read == 0


def test_get_logs_forward_unchanged(log_store: ControllerLogStore):
    """Forward mode (tail=False) still returns first N lines."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=10, tail=False)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(10)]


def test_get_logs_tail_concurrent(log_store: ControllerLogStore):
    """Tail reads survive concurrent appends."""
    errors: list[Exception] = []

    def writer():
        for i in range(500):
            log_store.append(TASK_ID, 0, [_make_entry(f"line-{i}", epoch_ms=i)])

    def reader():
        for _ in range(500):
            try:
                log_store.get_logs(TASK_ID, 0, max_lines=10, tail=True)
            except RuntimeError as e:
                errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == [], f"Concurrent access raised: {errors}"


# =============================================================================
# Persistent log dir tests
# =============================================================================


def test_persistent_log_dir(tmp_path: Path):
    """Append, close, reopen with same dir, read logs."""
    log_dir = tmp_path / "logs"
    store1 = ControllerLogStore(log_dir=log_dir)
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(5)]
    store1.append(TASK_ID, 0, entries)
    store1.close()

    store2 = ControllerLogStore(log_dir=log_dir)
    result = store2.get_logs(TASK_ID, 0)
    assert len(result.entries) == 5
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(5)]
    store2.close()


# =============================================================================
# Skip-lines / offset tests
# =============================================================================


def test_skip_lines_seeks_efficiently(log_store: ControllerLogStore):
    """Append 1000 lines, skip_lines=990 returns last 10."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(1000)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, skip_lines=990)
    assert len(result.entries) == 10
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(990, 1000)]
    assert result.lines_read == 1000


def test_skip_lines_round_trip(log_store: ControllerLogStore):
    """Append → read → append more → read(skip=prev) returns only new."""
    entries1 = [_make_entry(f"batch1-{i}", epoch_ms=i) for i in range(10)]
    log_store.append(TASK_ID, 0, entries1)

    result1 = log_store.get_logs(TASK_ID, 0)
    assert len(result1.entries) == 10
    cursor = result1.lines_read

    entries2 = [_make_entry(f"batch2-{i}", epoch_ms=100 + i) for i in range(5)]
    log_store.append(TASK_ID, 0, entries2)

    result2 = log_store.get_logs(TASK_ID, 0, skip_lines=cursor)
    assert len(result2.entries) == 5
    assert [e.data for e in result2.entries] == [f"batch2-{i}" for i in range(5)]
    assert result2.lines_read == 15


def test_line_offsets_rebuilt_on_restart(tmp_path: Path):
    """Append, close, new store with same dir, skip works."""
    log_dir = tmp_path / "logs"
    store1 = ControllerLogStore(log_dir=log_dir)
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(20)]
    store1.append(TASK_ID, 0, entries)
    store1.close()

    store2 = ControllerLogStore(log_dir=log_dir)
    result = store2.get_logs(TASK_ID, 0, skip_lines=15)
    assert len(result.entries) == 5
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(15, 20)]
    assert result.lines_read == 20
    store2.close()


def test_tail_uses_offset_array(log_store: ControllerLogStore):
    """tail=True returns last N via offset skip."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(50)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=5, tail=True)
    assert len(result.entries) == 5
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(45, 50)]
    assert result.lines_read == 50


def test_has_logs_no_known_attempts(tmp_path: Path):
    """has_logs uses path.exists() without _known_attempts cache."""
    log_dir = tmp_path / "logs"
    store1 = ControllerLogStore(log_dir=log_dir)
    store1.append(TASK_ID, 0, [_make_entry("hello")])
    store1.close()

    store2 = ControllerLogStore(log_dir=log_dir)
    assert store2.has_logs(TASK_ID, 0)
    assert not store2.has_logs(TASK_ID, 99)
    store2.close()


def test_clear_attempt_removes_offsets(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}") for i in range(5)]
    log_store.append(TASK_ID, 0, entries)
    assert log_store.has_logs(TASK_ID, 0)

    log_store.clear_attempt(TASK_ID, 0)
    assert not log_store.has_logs(TASK_ID, 0)
    result = log_store.get_logs(TASK_ID, 0)
    assert result.entries == []
    assert result.lines_read == 0
