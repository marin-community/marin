# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerLogStore and reverse-read helpers."""

import re
import threading
from pathlib import Path

import pytest

from iris.cluster.controller.logs import ControllerLogStore, _reverse_read_lines
from iris.cluster.types import JobName
from iris.rpc import logging_pb2


def _make_entry(data: str, epoch_ms: int = 0) -> logging_pb2.LogEntry:
    entry = logging_pb2.LogEntry(source="stdout", data=data)
    entry.timestamp.epoch_ms = epoch_ms
    return entry


# =============================================================================
# _reverse_read_lines tests
# =============================================================================


def test_reverse_read_lines_empty_file(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert list(_reverse_read_lines(f)) == []


def test_reverse_read_lines_single_line(tmp_path: Path):
    f = tmp_path / "single.txt"
    f.write_text("hello\n")
    assert list(_reverse_read_lines(f)) == ["hello"]


def test_reverse_read_lines_single_line_no_trailing_newline(tmp_path: Path):
    f = tmp_path / "single.txt"
    f.write_text("hello")
    assert list(_reverse_read_lines(f)) == ["hello"]


def test_reverse_read_lines_multiple_lines(tmp_path: Path):
    f = tmp_path / "multi.txt"
    lines = [f"line-{i}" for i in range(10)]
    f.write_text("\n".join(lines) + "\n")
    result = list(_reverse_read_lines(f))
    assert result == list(reversed(lines))


def test_reverse_read_lines_skips_blank_lines(tmp_path: Path):
    f = tmp_path / "blanks.txt"
    f.write_text("a\n\nb\n\nc\n")
    assert list(_reverse_read_lines(f)) == ["c", "b", "a"]


def test_reverse_read_lines_spanning_multiple_blocks(tmp_path: Path):
    """Lines that span multiple 25KB blocks are read correctly."""
    f = tmp_path / "large.txt"
    # Each line is ~100 bytes, 500 lines = ~50KB = spans 2+ blocks of 25KB
    lines = [f"line-{i:04d}-{'x' * 90}" for i in range(500)]
    f.write_text("\n".join(lines) + "\n")
    result = list(_reverse_read_lines(f))
    assert result == list(reversed(lines))


def test_reverse_read_lines_small_block_size(tmp_path: Path):
    """Verify correctness with a very small block size to stress boundary handling."""
    f = tmp_path / "small_block.txt"
    lines = [f"line-{i}" for i in range(20)]
    f.write_text("\n".join(lines) + "\n")
    result = list(_reverse_read_lines(f, block_size=16))
    assert result == list(reversed(lines))


# =============================================================================
# ControllerLogStore tail tests
# =============================================================================


@pytest.fixture()
def log_store():
    store = ControllerLogStore()
    yield store
    store.close()


TASK_ID = JobName.from_wire("/job/test/task/0")


def test_get_logs_tail_returns_last_n(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=10, tail=True)
    assert len(result) == 10
    assert [e.data for e in result] == [f"line-{i}" for i in range(90, 100)]


def test_get_logs_tail_chronological_order(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(50)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=5, tail=True)
    timestamps = [e.timestamp.epoch_ms for e in result]
    assert timestamps == sorted(timestamps)


def test_get_logs_tail_with_regex(log_store: ControllerLogStore):
    entries = [_make_entry(f"{'ERROR' if i % 10 == 0 else 'INFO'}: msg-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=3, tail=True, regex_filter=re.compile("ERROR"))
    assert len(result) == 3
    assert all("ERROR" in e.data for e in result)
    # Last 3 ERROR lines are at i=70, 80, 90
    assert [e.data for e in result] == ["ERROR: msg-70", "ERROR: msg-80", "ERROR: msg-90"]


def test_get_logs_tail_with_since_ms(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=5, tail=True, since_ms=94)
    assert [e.data for e in result] == ["line-95", "line-96", "line-97", "line-98", "line-99"]


def test_get_logs_tail_fewer_than_max(log_store: ControllerLogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(3)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=100, tail=True)
    assert len(result) == 3


def test_get_logs_tail_empty_file(log_store: ControllerLogStore):
    result = log_store.get_logs(TASK_ID, 0, max_lines=10, tail=True)
    assert result == []


def test_get_logs_forward_unchanged(log_store: ControllerLogStore):
    """Forward mode (tail=False) still returns first N lines."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(TASK_ID, 0, entries)

    result = log_store.get_logs(TASK_ID, 0, max_lines=10, tail=False)
    assert [e.data for e in result] == [f"line-{i}" for i in range(10)]


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
