# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LogStore (rotating RAM buffers + Parquet + DuckDB)."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from iris.cluster.log_store import LogStore, _EST_BYTES_PER_ROW, task_log_key
from iris.cluster.log_store.duckdb_store import DuckDBLogStore
from iris.cluster.types import JobName, TaskAttempt
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
    store = LogStore()
    yield store
    store.close()


TASK_ID = JobName.from_wire("/job/test/task/0")
KEY = task_log_key(TaskAttempt(task_id=TASK_ID, attempt_id=0))


# =============================================================================
# Basic tail tests
# =============================================================================


def test_get_logs_tail_returns_last_n(log_store: LogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(KEY, entries)

    result = log_store.get_logs(KEY, max_lines=10, tail=True)
    assert len(result.entries) == 10
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(90, 100)]


def test_get_logs_tail_with_substring_filter(log_store: LogStore):
    entries = [_make_entry(f"{'ERROR' if i % 10 == 0 else 'INFO'}: msg-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(KEY, entries)

    result = log_store.get_logs(KEY, max_lines=3, tail=True, substring_filter="ERROR")
    assert len(result.entries) == 3
    assert all("ERROR" in e.data for e in result.entries)
    assert [e.data for e in result.entries] == ["ERROR: msg-70", "ERROR: msg-80", "ERROR: msg-90"]


def test_get_logs_tail_with_since_ms(log_store: LogStore):
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(KEY, entries)

    result = log_store.get_logs(KEY, max_lines=5, tail=True, since_ms=94)
    assert [e.data for e in result.entries] == ["line-95", "line-96", "line-97", "line-98", "line-99"]


def test_get_logs_tail_empty_file(log_store: LogStore):
    result = log_store.get_logs(KEY, max_lines=10, tail=True)
    assert result.entries == []
    assert result.cursor == 0


def test_get_logs_forward_unchanged(log_store: LogStore):
    """Forward mode (tail=False) still returns first N lines."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
    log_store.append(KEY, entries)

    result = log_store.get_logs(KEY, max_lines=10, tail=False)
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(10)]


def test_get_logs_tail_concurrent(log_store: LogStore):
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


# =============================================================================
# Persistent log dir tests
# =============================================================================


def test_persistent_log_dir(tmp_path: Path):
    """Append, close, reopen with same dir, read logs."""
    log_dir = tmp_path / "logs"
    store1 = DuckDBLogStore(log_dir=log_dir)
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(5)]
    store1.append(KEY, entries)
    store1.close()

    store2 = DuckDBLogStore(log_dir=log_dir)
    result = store2.get_logs(KEY)
    assert len(result.entries) == 5
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(5)]
    store2.close()


# =============================================================================
# Cursor-based pagination tests
# =============================================================================


def test_cursor_skips_already_seen(log_store: LogStore):
    """Append 1000 lines, use cursor from first 990 to get last 10."""
    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(1000)]
    log_store.append(KEY, entries)

    result_head = log_store.get_logs(KEY, max_lines=990)
    assert len(result_head.entries) == 990

    result = log_store.get_logs(KEY, cursor=result_head.cursor)
    assert len(result.entries) == 10
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(990, 1000)]


def test_cursor_round_trip(log_store: LogStore):
    """Append -> read -> append more -> read(cursor=prev) returns only new."""
    entries1 = [_make_entry(f"batch1-{i}", epoch_ms=i) for i in range(10)]
    log_store.append(KEY, entries1)

    result1 = log_store.get_logs(KEY)
    assert len(result1.entries) == 10
    cursor = result1.cursor

    entries2 = [_make_entry(f"batch2-{i}", epoch_ms=100 + i) for i in range(5)]
    log_store.append(KEY, entries2)

    result2 = log_store.get_logs(KEY, cursor=cursor)
    assert len(result2.entries) == 5
    assert [e.data for e in result2.entries] == [f"batch2-{i}" for i in range(5)]


def test_has_logs_no_known_attempts(tmp_path: Path):
    """has_logs works across store instances."""
    log_dir = tmp_path / "logs"
    store1 = DuckDBLogStore(log_dir=log_dir)
    store1.append(KEY, [_make_entry("hello")])
    store1.close()

    store2 = DuckDBLogStore(log_dir=log_dir)
    assert store2.has_logs(KEY)
    assert not store2.has_logs(task_log_key(TaskAttempt(task_id=TASK_ID, attempt_id=99)))
    store2.close()


# =============================================================================
# Regex pattern query tests
# =============================================================================


def test_get_logs_regex_pattern_returns_all_matching(log_store: LogStore):
    """Regex pattern query returns entries from all matching keys in id order."""
    t0 = JobName.from_wire("/job/test/task/0")
    t1 = JobName.from_wire("/job/test/task/1")

    log_store.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry("t0-a0-line0", epoch_ms=1)])
    log_store.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=1)), [_make_entry("t0-a1-line0", epoch_ms=2)])
    log_store.append(task_log_key(TaskAttempt(task_id=t1, attempt_id=0)), [_make_entry("t1-a0-line0", epoch_ms=3)])

    result = log_store.get_logs("/job/test/.*")
    assert len(result.entries) == 3
    assert [e.data for e in result.entries] == ["t0-a0-line0", "t0-a1-line0", "t1-a0-line0"]


def test_get_logs_regex_pattern_cursor_continuation(log_store: LogStore):
    """Cursor-based continuation with regex pattern returns no duplicates."""
    t0 = JobName.from_wire("/job/test/task/0")
    log_store.append(
        task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry(f"line-{i}", epoch_ms=i) for i in range(5)]
    )

    result1 = log_store.get_logs("/job/test/.*", max_lines=3)
    assert len(result1.entries) == 3

    result2 = log_store.get_logs("/job/test/.*", cursor=result1.cursor)
    assert len(result2.entries) == 2
    assert [e.data for e in result2.entries] == ["line-3", "line-4"]


def test_get_logs_regex_pattern_isolation(log_store: LogStore):
    """Regex pattern /job/test/.* does not match /job/testing/."""
    t_test = JobName.from_wire("/job/test/task/0")
    t_testing = JobName.from_wire("/job/testing/task/0")

    log_store.append(task_log_key(TaskAttempt(task_id=t_test, attempt_id=0)), [_make_entry("test-line")])
    log_store.append(task_log_key(TaskAttempt(task_id=t_testing, attempt_id=0)), [_make_entry("testing-line")])

    result = log_store.get_logs("/job/test/.*")
    assert len(result.entries) == 1
    assert result.entries[0].data == "test-line"


def test_get_logs_regex_pattern_scoping(log_store: LogStore):
    """Regex pattern can scope to direct children vs all descendants."""
    # Direct tasks of /job/parent
    t0 = JobName.from_wire("/job/parent/0")
    # Child job task: /job/parent/child/0
    child_t0 = JobName.from_wire("/job/parent/child/0")

    log_store.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry("parent-line", epoch_ms=1)])
    log_store.append(task_log_key(TaskAttempt(task_id=child_t0, attempt_id=0)), [_make_entry("child-line", epoch_ms=2)])

    # Broad pattern: both are returned
    result_all = log_store.get_logs("/job/parent/.*")
    assert len(result_all.entries) == 2

    # Direct-tasks-only pattern: excludes child job entries
    result_direct = log_store.get_logs(r"/job/parent/\d+:.*")
    assert len(result_direct.entries) == 1
    assert result_direct.entries[0].data == "parent-line"


def test_get_logs_regex_pattern_tail_returns_last_n(log_store: LogStore):
    """Tail mode with regex pattern returns the last N entries across all matching keys."""
    t0 = JobName.from_wire("/job/test/task/0")
    t1 = JobName.from_wire("/job/test/task/1")

    # Insert 10 entries total: 5 per task
    log_store.append(
        task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry(f"t0-line-{i}", epoch_ms=i) for i in range(5)]
    )
    log_store.append(
        task_log_key(TaskAttempt(task_id=t1, attempt_id=0)),
        [_make_entry(f"t1-line-{i}", epoch_ms=10 + i) for i in range(5)],
    )

    result = log_store.get_logs("/job/test/.*", max_lines=3, tail=True)
    assert len(result.entries) == 3
    # Should return the last 3 entries by insertion order (autoincrement id)
    assert [e.data for e in result.entries] == ["t1-line-2", "t1-line-3", "t1-line-4"]


def test_get_logs_regex_pattern_tail_with_substring_filter(log_store: LogStore):
    """Tail mode with regex pattern + substring filter returns last N matching entries."""
    t0 = JobName.from_wire("/job/test/task/0")

    entries = [_make_entry(f"{'ERROR' if i % 3 == 0 else 'INFO'}: msg-{i}", epoch_ms=i) for i in range(12)]
    log_store.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), entries)

    result = log_store.get_logs("/job/test/.*", max_lines=2, tail=True, substring_filter="ERROR")
    assert len(result.entries) == 2
    assert [e.data for e in result.entries] == ["ERROR: msg-6", "ERROR: msg-9"]


def test_substring_filter_escapes_like_wildcards(log_store: LogStore):
    """Percent and underscore in substring_filter match literally, not as SQL wildcards."""
    log_store.append(
        KEY,
        [
            _make_entry("100% done", epoch_ms=1),
            _make_entry("a_b_c", epoch_ms=2),
            _make_entry("no match", epoch_ms=3),
        ],
    )

    result_pct = log_store.get_logs(KEY, substring_filter="100%")
    assert [e.data for e in result_pct.entries] == ["100% done"]

    result_us = log_store.get_logs(KEY, substring_filter="a_b")
    assert [e.data for e in result_us.entries] == ["a_b_c"]

    # Regex pattern key with substring filter containing wildcards
    result_pattern = log_store.get_logs("/job/test/.*", substring_filter="100%")
    assert [e.data for e in result_pattern.entries] == ["100% done"]


def test_job_level_regex_uses_slash_not_colon(log_store: LogStore):
    """Job-level regex query must use /job/.* (slash), not /job:.* (colon).

    Reproduces the E2E smoke failure where the dashboard job-level log viewer
    showed no entries because it used jobId:.* instead of jobId/.*.
    """
    t0 = JobName.from_wire("/alice/train/0")
    log_store.append(task_log_key(TaskAttempt(task_id=t0, attempt_id=0)), [_make_entry("hello", epoch_ms=1)])

    # Correct pattern: slash-dotstar matches task keys under the job
    result_slash = log_store.get_logs("/alice/train/.*")
    assert len(result_slash.entries) == 1
    assert result_slash.entries[0].data == "hello"

    # Wrong pattern: colon-dotstar matches nothing (task keys use /task_idx:attempt)
    result_colon = log_store.get_logs("/alice/train:.*")
    assert len(result_colon.entries) == 0, "colon-dotstar should NOT match task keys stored under /alice/train/0:0"


def test_cursor_with_since_ms(log_store: LogStore):
    """Combined cursor + since_ms filters correctly."""
    log_store.append(KEY, [_make_entry(f"line-{i}", epoch_ms=i * 10) for i in range(10)])

    # Get cursor past first 5 entries
    result1 = log_store.get_logs(KEY, max_lines=5)
    cursor = result1.cursor

    # Only entries with epoch_ms > 70 and id > cursor
    result2 = log_store.get_logs(KEY, cursor=cursor, since_ms=70)
    assert [e.data for e in result2.entries] == ["line-8", "line-9"]


# =============================================================================
# Segment GC tests
# =============================================================================


def test_gc_drops_oldest_segments_by_count(tmp_path: Path):
    """When local segments exceed max_local_segments, oldest are removed."""
    log_dir = tmp_path / "logs"
    # segment_target_bytes=1 means every parquet file is bigger than the target,
    # so consolidation is skipped and each seal creates a new file.
    store = DuckDBLogStore(
        log_dir=log_dir,
        max_local_segments=2,
        max_local_bytes=10 * 1024**3,  # effectively unlimited
        segment_target_bytes=1,  # seal after every append
    )
    try:
        for batch in range(4):
            entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
            store.append(KEY, entries)

        # Wait for background flushes to complete.
        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        remaining_files = sorted(store._log_dir.glob("logs_*.parquet"))
        assert len(remaining_files) <= 2

        # The most recent data should still be readable.
        result = store.get_logs(KEY, max_lines=10, tail=True)
        assert len(result.entries) > 0
        assert all("batch3" in e.data for e in result.entries)
    finally:
        store.close()


def test_gc_drops_oldest_segments_by_bytes(tmp_path: Path):
    """When local parquet bytes exceed max_local_bytes, oldest are removed."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(
        log_dir=log_dir,
        max_local_segments=100,  # effectively unlimited
        segment_target_bytes=1,  # seal after every append
    )
    try:
        for batch in range(4):
            entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
            store.append(KEY, entries)

        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        parquet_files = sorted(store._log_dir.glob("logs_*.parquet"))
        assert len(parquet_files) == 4
        one_file_size = parquet_files[0].stat().st_size

        # Set budget to allow only 2 files, then trigger GC.
        store._max_local_bytes = one_file_size * 2
        store._gc_local_segments()

        remaining = sorted(store._log_dir.glob("logs_*.parquet"))
        assert len(remaining) <= 2

        # The most recent data should still be readable.
        result = store.get_logs(KEY, max_lines=10, tail=True)
        assert len(result.entries) > 0
        assert all("batch3" in e.data for e in result.entries)
    finally:
        store.close()


def test_reads_recover_when_local_segment_vanishes(tmp_path: Path):
    """Out-of-band deletion of a Parquet file must not permanently break reads.

    The store's in-memory ``_local_segments`` is populated at startup by
    globbing the log dir, but it can drift from disk if files are removed
    by anything other than ``_gc_local_segments`` (manual ``rm``, eviction,
    leftover entries from an old filename format). Without self-healing,
    DuckDB raises ``No files found that match the pattern …`` on every
    subsequent read until the process restarts.
    """
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(
        log_dir=log_dir,
        max_local_segments=100,  # don't GC during the test
        segment_target_bytes=1,  # seal on every append
    )
    try:
        for batch in range(3):
            store.append(KEY, [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)])
        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        parquet_files = sorted(log_dir.glob("logs_*.parquet"))
        assert len(parquet_files) == 3
        # Simulate out-of-band deletion of the oldest segment.
        parquet_files[0].unlink()

        # Read still succeeds and returns the rows from the surviving segments.
        result = store.get_logs(KEY)
        data = [e.data for e in result.entries]
        assert any("batch1" in d for d in data)
        assert any("batch2" in d for d in data)
        assert not any("batch0" in d for d in data)

        # In-memory index was pruned, so a follow-up read does not re-trigger the failure.
        assert len(store._local_segments) == 2
        result2 = store.get_logs(KEY)
        assert [e.data for e in result2.entries] == data
    finally:
        store.close()


# =============================================================================
# Parquet-specific tests
# =============================================================================


def test_flush_creates_parquet_segment(tmp_path: Path):
    """Verify that sealing the buffer writes a Parquet file."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    try:
        entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(10)]
        store.append(KEY, entries)
        # Wait for background flush.
        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")
        parquet_files = list(log_dir.glob("logs_*.parquet"))
        assert len(parquet_files) >= 1
    finally:
        store.close()


def test_cursor_continuity_across_flush(tmp_path: Path):
    """Cursor from pre-flush read correctly filters post-flush entries."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    try:
        entries1 = [_make_entry(f"batch1-{i}", epoch_ms=i) for i in range(10)]
        store.append(KEY, entries1)
        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        result1 = store.get_logs(KEY)
        assert len(result1.entries) == 10
        cursor = result1.cursor

        entries2 = [_make_entry(f"batch2-{i}", epoch_ms=100 + i) for i in range(5)]
        store.append(KEY, entries2)

        result2 = store.get_logs(KEY, cursor=cursor)
        assert len(result2.entries) == 5
        assert all("batch2" in e.data for e in result2.entries)
    finally:
        store.close()


def test_seq_recovery_on_restart(tmp_path: Path):
    """After close and reopen, sequence numbers don't collide."""
    log_dir = tmp_path / "logs"
    store1 = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    entries1 = [_make_entry(f"s1-{i}", epoch_ms=i) for i in range(10)]
    store1.append(KEY, entries1)
    result1 = store1.get_logs(KEY)
    cursor1 = result1.cursor
    store1.close()

    store2 = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    entries2 = [_make_entry(f"s2-{i}", epoch_ms=100 + i) for i in range(5)]
    store2.append(KEY, entries2)

    # All entries from both sessions should be readable
    result_all = store2.get_logs(KEY)
    assert len(result_all.entries) == 15

    # Cursor from session 1 should still work to get only session 2 entries
    result_new = store2.get_logs(KEY, cursor=cursor1)
    assert len(result_new.entries) == 5
    assert all("s2" in e.data for e in result_new.entries)
    store2.close()


def test_concurrent_read_write(tmp_path: Path):
    """Concurrent appends and reads don't crash or corrupt data."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=50 * _EST_BYTES_PER_ROW)
    errors: list[Exception] = []

    def writer():
        for i in range(200):
            store.append(KEY, [_make_entry(f"line-{i}", epoch_ms=i)])

    def reader():
        for _ in range(200):
            try:
                store.get_logs(KEY, max_lines=10)
            except Exception as e:
                errors.append(e)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent access raised: {errors}"

    # All 200 entries should be present
    result = store.get_logs(KEY)
    assert len(result.entries) == 200
    store.close()


def test_small_segments_are_consolidated(tmp_path: Path):
    """Multiple small flushes consolidate into a single parquet file."""
    log_dir = tmp_path / "logs"
    # Large segment target means all small files are below the threshold
    # and will be consolidated into one file.
    store = DuckDBLogStore(
        log_dir=log_dir,
        segment_target_bytes=100 * 1024 * 1024,  # 100MB — way bigger than our test data
        flush_interval_sec=0,  # seal on every append (time threshold always satisfied)
    )
    try:
        for batch in range(5):
            entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
            store.append(KEY, entries)
            # Wait for flush to complete before next append so consolidation runs.
            store._executor.shutdown(wait=True)
            store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        parquet_files = list(log_dir.glob("logs_*.parquet"))
        # All 5 batches should be consolidated into a single parquet file.
        assert len(parquet_files) == 1, f"Expected 1 consolidated file, got {len(parquet_files)}"

        # All data should still be readable.
        result = store.get_logs(KEY)
        assert len(result.entries) == 50
        assert result.entries[0].data == "batch0-0"
        assert result.entries[-1].data == "batch4-9"
    finally:
        store.close()


def test_consolidation_preserves_cursor_continuity(tmp_path: Path):
    """Cursors from before consolidation still work after consolidation."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(
        log_dir=log_dir,
        segment_target_bytes=100 * 1024 * 1024,
        flush_interval_sec=0,
    )
    try:
        entries1 = [_make_entry(f"batch1-{i}", epoch_ms=i) for i in range(10)]
        store.append(KEY, entries1)
        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        result1 = store.get_logs(KEY)
        cursor = result1.cursor

        # Second batch will consolidate with the first.
        entries2 = [_make_entry(f"batch2-{i}", epoch_ms=100 + i) for i in range(5)]
        store.append(KEY, entries2)
        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        # Cursor from before consolidation should still return only new entries.
        result2 = store.get_logs(KEY, cursor=cursor)
        assert len(result2.entries) == 5
        assert all("batch2" in e.data for e in result2.entries)
    finally:
        store.close()


def test_connection_pool_memory_limit(tmp_path: Path):
    """DuckDB connections respect the configured memory limit."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, duckdb_memory_limit="64MB")
    try:
        entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(100)]
        store.append(KEY, entries)
        result = store.get_logs(KEY)
        assert len(result.entries) == 100
    finally:
        store.close()


def test_concurrent_reads_no_concat_copy(tmp_path: Path):
    """Multiple concurrent reads work correctly without pa.concat_tables."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, duckdb_memory_limit="64MB")
    errors: list[Exception] = []

    # Create data across multiple RAM buffers: pending + chunks + sealed
    for batch in range(5):
        entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
        store.append(KEY, entries)

    # Seal to create sealed buffers, then add more to pending
    store._seal_head()
    store.append(KEY, [_make_entry("after-seal", epoch_ms=999)])

    def reader():
        for _ in range(50):
            try:
                result = store.get_logs(KEY, max_lines=100, tail=True)
                assert len(result.entries) > 0
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent reads raised: {errors}"

    result = store.get_logs(KEY)
    assert len(result.entries) == 51  # 5*10 + 1
    store.close()


def test_sealed_buffers_readable_before_flush(tmp_path: Path):
    """Data in sealed buffers is still readable even before Parquet flush completes."""
    log_dir = tmp_path / "logs"
    # Use a large flush interval so time-based sealing won't trigger.
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=10 * 1024 * 1024, flush_interval_sec=9999)

    entries = [_make_entry(f"line-{i}", epoch_ms=i) for i in range(10)]
    # Append to head buffer only (segment_target_bytes is large, won't seal).
    store.append(KEY, entries)

    # Data should be readable from the head buffer.
    result = store.get_logs(KEY)
    assert len(result.entries) == 10
    assert [e.data for e in result.entries] == [f"line-{i}" for i in range(10)]

    # Now seal manually and verify sealed buffer is still readable.
    store._seal_head()
    # Data is in the sealed deque (flush is async, might not have completed).
    result2 = store.get_logs(KEY)
    assert len(result2.entries) == 10

    store.close()


# =============================================================================
# Tail fallback, working-set cap, and concurrency limiter tests
# =============================================================================


def _make_store_with_segments(log_dir: Path, num_segments: int, rows_per_segment: int) -> DuckDBLogStore:
    """Build a store with `num_segments` parquet segments, each holding
    `rows_per_segment` rows under a regex-matchable key prefix.

    Uses segment_target_bytes=1 so every seal creates a new file.
    """
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    for batch in range(num_segments):
        entries = [
            _make_entry(
                f"{'MATCH' if i == 0 and batch == 0 else 'nomatch'}-batch{batch}-{i}",
                epoch_ms=batch * 1000 + i,
            )
            for i in range(rows_per_segment)
        ]
        store.append(KEY, entries)
        store._executor.shutdown(wait=True)
        store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")
    return store


def test_tail_fallback_finds_match_in_oldest_of_capped_window(tmp_path: Path):
    """When the seq-bounded fast path misses, the fallback query still finds
    a match that lives in the oldest segment within the per-read cap."""
    log_dir = tmp_path / "logs"
    # 5 segments (fits within _MAX_PARQUETS_PER_READ); match is in segment 0.
    store = _make_store_with_segments(log_dir, num_segments=5, rows_per_segment=20)
    try:
        # Regex pattern forces include_key_in_select=True, which engages the
        # fast-path + fallback logic.
        result = store.get_logs("/job/test/.*", max_lines=10, tail=True, substring_filter="MATCH")
        assert len(result.entries) == 1
        assert "MATCH" in result.entries[0].data
    finally:
        store.close()


def test_read_caps_to_newest_segments(tmp_path: Path):
    """A single read never scans more than _MAX_PARQUETS_PER_READ segments;
    matches outside that window are not visible to this call."""
    from iris.cluster.log_store.duckdb_store import _MAX_PARQUETS_PER_READ

    log_dir = tmp_path / "logs"
    num_segments = _MAX_PARQUETS_PER_READ + 3
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    try:
        for batch in range(num_segments):
            # Match marker lives only in the oldest segment (batch 0).
            entries = [
                _make_entry(f"{'MATCH' if batch == 0 else 'nomatch'}-b{batch}-{i}", epoch_ms=batch * 1000 + i)
                for i in range(10)
            ]
            store.append(KEY, entries)
            store._executor.shutdown(wait=True)
            store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        result = store.get_logs("/job/test/.*", max_lines=20, tail=True, substring_filter="MATCH")
        # Oldest segment is outside the newest-N window; match is invisible.
        assert len(result.entries) == 0
    finally:
        store.close()


def test_tail_fallback_accumulates_across_segments(tmp_path: Path):
    """Filter matches spread across older segments are returned when newer
    segments have fewer than max_lines matches."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    try:
        # Write 4 segments. Each has 2 MATCH rows.
        for batch in range(4):
            entries = []
            for i in range(10):
                tag = "MATCH" if i < 2 else "nomatch"
                entries.append(_make_entry(f"{tag}-batch{batch}-{i}", epoch_ms=batch * 1000 + i))
            store.append(KEY, entries)
            store._executor.shutdown(wait=True)
            store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        # Ask for 6 MATCH rows; must pull from at least 3 segments.
        result = store.get_logs("/job/test/.*", max_lines=6, tail=True, substring_filter="MATCH")
        assert len(result.entries) == 6
        assert all("MATCH" in e.data for e in result.entries)
        # Results are sorted ascending in LogReadResult (DESC then reversed).
        seqs = [e.timestamp.epoch_ms for e in result.entries]
        assert seqs == sorted(seqs)
    finally:
        store.close()


def test_tail_fast_path_short_circuits(tmp_path: Path, monkeypatch):
    """When the tail window has enough matches, only one SQL query runs."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    try:
        # Two segments, all rows match the filter, so the fast path is
        # sufficient for max_lines=5.
        for batch in range(2):
            entries = [_make_entry(f"MATCH-{batch}-{i}", epoch_ms=batch * 1000 + i) for i in range(20)]
            store.append(KEY, entries)
            store._executor.shutdown(wait=True)
            store._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_flush")

        call_count = 0
        from iris.cluster.log_store import duckdb_store as mod

        original = mod._build_union_source

        def counting(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(mod, "_build_union_source", counting)

        result = store.get_logs("/job/test/.*", max_lines=5, tail=True, substring_filter="MATCH")
        assert len(result.entries) == 5
        # Exactly one source build: the fast path. No fallback query.
        assert call_count == 1
    finally:
        store.close()
