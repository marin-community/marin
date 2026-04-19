# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LogStore (rotating RAM buffers + Parquet + DuckDB)."""

import threading
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
    store = DuckDBLogStore(
        log_dir=log_dir,
        max_local_segments=2,
        max_local_bytes=10 * 1024**3,  # effectively unlimited
    )
    try:
        for batch in range(4):
            entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
            store.append(KEY, entries)
            store._force_flush()  # one file per batch

        remaining_files = sorted(store._log_dir.glob("tmp_*.parquet"))
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
            store._force_flush()  # one file per batch

        parquet_files = sorted(store._log_dir.glob("tmp_*.parquet"))
        assert len(parquet_files) == 4
        one_file_size = parquet_files[0].stat().st_size

        # Set budget to allow only 2 files, then trigger GC.
        store._max_local_bytes = one_file_size * 2
        store._gc_local_segments()

        remaining = sorted(store._log_dir.glob("tmp_*.parquet"))
        assert len(remaining) <= 2

        # The most recent data should still be readable.
        result = store.get_logs(KEY, max_lines=10, tail=True)
        assert len(result.entries) > 0
        assert all("batch3" in e.data for e in result.entries)
    finally:
        store.close()


# =============================================================================
# Compaction tests
# =============================================================================


def test_compaction_merges_tmps_into_log(tmp_path: Path):
    """Compaction merges all tmp_*.parquet files into a single logs_*.parquet,
    preserves data, and unlinks the sources."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir)
    try:
        for batch in range(5):
            entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
            store.append(KEY, entries)
            store._force_flush()

        assert len(sorted(log_dir.glob("tmp_*.parquet"))) == 5
        assert len(sorted(log_dir.glob("logs_*.parquet"))) == 0

        store._force_compaction()

        assert len(sorted(log_dir.glob("tmp_*.parquet"))) == 0
        assert len(sorted(log_dir.glob("logs_*.parquet"))) == 1

        result = store.get_logs(KEY)
        assert len(result.entries) == 50
        assert result.entries[0].data == "batch0-0"
        assert result.entries[-1].data == "batch4-9"
    finally:
        store.close()


def test_compaction_skips_single_tmp(tmp_path: Path):
    """One tmp isn't worth rewriting — compaction leaves it alone."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir)
    try:
        store.append(KEY, [_make_entry("only", epoch_ms=0)])
        store._force_flush()
        tmps_before = sorted(log_dir.glob("tmp_*.parquet"))
        assert len(tmps_before) == 1

        store._force_compaction()

        tmps_after = sorted(log_dir.glob("tmp_*.parquet"))
        assert tmps_after == tmps_before
        assert len(sorted(log_dir.glob("logs_*.parquet"))) == 0
    finally:
        store.close()


def test_close_compacts_and_offloads_single_tmp(tmp_path: Path):
    """close() with exactly one tmp must still produce a logs_ segment and
    offload it to remote storage. Regression: the < 2 tmp skip in the
    steady-state compaction path was inherited by close(), so a low-volume
    shutdown left data only in local tmp_*.parquet and lost it on a fresh
    restart with empty local storage."""
    log_dir = tmp_path / "logs"
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    store = DuckDBLogStore(log_dir=log_dir, remote_log_dir=str(remote_dir))
    store.append(KEY, [_make_entry("only", epoch_ms=0)])
    store._force_flush()
    assert len(sorted(log_dir.glob("tmp_*.parquet"))) == 1
    assert len(sorted(log_dir.glob("logs_*.parquet"))) == 0

    store.close()

    assert sorted(log_dir.glob("tmp_*.parquet")) == []
    local_logs = sorted(log_dir.glob("logs_*.parquet"))
    assert len(local_logs) == 1
    remote_logs = sorted(remote_dir.glob("logs_*.parquet"))
    assert len(remote_logs) == 1
    assert remote_logs[0].name == local_logs[0].name


def test_startup_drops_tmps_covered_by_log(tmp_path: Path):
    """A compaction crash between ``rename(logs_)`` and ``unlink(tmp_)``
    leaves both on disk with overlapping ranges. Restart must drop the
    now-redundant tmps or reads double-count those rows."""
    log_dir = tmp_path / "logs"
    store1 = DuckDBLogStore(log_dir=log_dir)
    for batch in range(3):
        store1.append(KEY, [_make_entry(f"b{batch}-{i}", epoch_ms=batch * 10 + i) for i in range(3)])
        store1._force_flush()

    # Snapshot the tmp bytes before compaction destroys them.
    tmps_before_compact = sorted(log_dir.glob("tmp_*.parquet"))
    assert len(tmps_before_compact) == 3
    saved_tmps = {p.name: p.read_bytes() for p in tmps_before_compact}

    store1._force_compaction()
    store1.close()
    assert len(sorted(log_dir.glob("logs_*.parquet"))) == 1
    assert sorted(log_dir.glob("tmp_*.parquet")) == []

    # Simulate a mid-compaction crash: rename landed, unlink didn't.
    for name, data in saved_tmps.items():
        (log_dir / name).write_bytes(data)
    assert len(sorted(log_dir.glob("tmp_*.parquet"))) == 3
    assert len(sorted(log_dir.glob("logs_*.parquet"))) == 1

    store2 = DuckDBLogStore(log_dir=log_dir)
    try:
        assert sorted(log_dir.glob("tmp_*.parquet")) == []
        assert len(sorted(log_dir.glob("logs_*.parquet"))) == 1
        result = store2.get_logs(KEY)
        assert len(result.entries) == 9
        assert [e.data for e in result.entries] == [
            "b0-0",
            "b0-1",
            "b0-2",
            "b1-0",
            "b1-1",
            "b1-2",
            "b2-0",
            "b2-1",
            "b2-2",
        ]
    finally:
        store2.close()


def test_recovery_reads_both_tmp_and_log(tmp_path: Path):
    """After a non-graceful exit, a dir with mixed tmp_ and logs_ files is
    fully readable. Emulates a crash by halting the bg thread without going
    through close() (which would compact the trailing tmp into a logs_)."""
    log_dir = tmp_path / "logs"
    store1 = DuckDBLogStore(log_dir=log_dir)
    try:
        # Two flushes, then compact -> produces one logs_ file.
        for batch in range(2):
            store1.append(KEY, [_make_entry(f"old-{batch}-{i}", epoch_ms=batch * 10 + i) for i in range(3)])
            store1._force_flush()
        store1._force_compaction()
        # Third flush after compaction produces a tmp_ file.
        store1.append(KEY, [_make_entry(f"new-{i}", epoch_ms=100 + i) for i in range(3)])
        store1._force_flush()
    finally:
        # Simulate crash: stop bg thread and release DuckDB without the
        # shutdown-compaction path close() runs.
        store1._stop.set()
        store1._wake.set()
        store1._bg_thread.join()
        store1._pool.close()

    assert len(sorted(log_dir.glob("logs_*.parquet"))) == 1
    assert len(sorted(log_dir.glob("tmp_*.parquet"))) == 1

    store2 = DuckDBLogStore(log_dir=log_dir)
    try:
        result = store2.get_logs(KEY)
        assert len(result.entries) == 9
        assert [e.data for e in result.entries] == [
            "old-0-0",
            "old-0-1",
            "old-0-2",
            "old-1-0",
            "old-1-1",
            "old-1-2",
            "new-0",
            "new-1",
            "new-2",
        ]
    finally:
        store2.close()


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
        store._force_flush()
        parquet_files = list(log_dir.glob("tmp_*.parquet"))
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
        store._force_flush()

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


def test_each_flush_writes_a_new_segment(tmp_path: Path):
    """Every flush produces its own parquet — the write path is append-only."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir)
    try:
        for batch in range(5):
            entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
            store.append(KEY, entries)
            store._force_flush()

        parquet_files = list(log_dir.glob("tmp_*.parquet"))
        assert len(parquet_files) == 5, f"Expected 5 segment files, got {len(parquet_files)}"

        result = store.get_logs(KEY)
        assert len(result.entries) == 50
        assert result.entries[0].data == "batch0-0"
        assert result.entries[-1].data == "batch4-9"
    finally:
        store.close()


def test_cursor_continuity_across_segments(tmp_path: Path):
    """Cursors remain valid across the N-segment boundary the new write path produces."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir)
    try:
        entries1 = [_make_entry(f"batch1-{i}", epoch_ms=i) for i in range(10)]
        store.append(KEY, entries1)
        store._force_flush()

        result1 = store.get_logs(KEY)
        cursor = result1.cursor

        entries2 = [_make_entry(f"batch2-{i}", epoch_ms=100 + i) for i in range(5)]
        store.append(KEY, entries2)
        store._force_flush()

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


def test_concurrent_reads_span_all_sources(tmp_path: Path):
    """Concurrent reads crossing parquet + chunks + pending don't collide on
    the per-cursor RAM table registrations."""
    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, duckdb_memory_limit="64MB")
    errors: list[Exception] = []

    for batch in range(5):
        entries = [_make_entry(f"batch{batch}-{i}", epoch_ms=batch * 100 + i) for i in range(10)]
        store.append(KEY, entries)

    # Flush to segment, then add more to pending so reads cross all three sources.
    store._force_flush()
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

    # Now flush manually and verify the data is still readable once the
    # snapshot has moved from _pending into _flushing / a new segment.
    store._force_flush()
    result2 = store.get_logs(KEY)
    assert len(result2.entries) == 10

    store.close()


# =============================================================================
# Per-read working-set cap tests
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
        store._force_flush()
    return store


def test_tail_finds_match_in_oldest_of_capped_window(tmp_path: Path):
    """A match in the oldest segment within the per-read cap is returned."""
    log_dir = tmp_path / "logs"
    # 5 segments all fit within the default byte cap; match is in segment 0.
    store = _make_store_with_segments(log_dir, num_segments=5, rows_per_segment=20)
    try:
        result = store.get_logs("/job/test/.*", max_lines=10, tail=True, substring_filter="MATCH")
        assert len(result.entries) == 1
        assert "MATCH" in result.entries[0].data
    finally:
        store.close()


def test_read_caps_to_newest_segments(tmp_path: Path, monkeypatch):
    """A single read never scans past _MAX_PARQUET_BYTES_PER_READ of segments;
    matches in older (outside-budget) segments are not visible to this call."""
    from iris.cluster.log_store import duckdb_store as mod

    log_dir = tmp_path / "logs"
    store = DuckDBLogStore(log_dir=log_dir, segment_target_bytes=1)
    try:
        num_segments = 6
        for batch in range(num_segments):
            # Match marker lives only in the oldest segment (batch 0). With a
            # tiny byte cap, only the newest few segments are included.
            entries = [
                _make_entry(f"{'MATCH' if batch == 0 else 'nomatch'}-b{batch}-{i}", epoch_ms=batch * 1000 + i)
                for i in range(10)
            ]
            store.append(KEY, entries)
            store._force_flush()

        # Measure a single segment's bytes, then cap the read to ~2 segments.
        first = next(log_dir.glob("tmp_*.parquet"))
        per_file = first.stat().st_size
        monkeypatch.setattr(mod, "_MAX_PARQUET_BYTES_PER_READ", per_file * 2)

        result = store.get_logs("/job/test/.*", max_lines=20, tail=True, substring_filter="MATCH")
        # Oldest segment is outside the newest-bytes window; match is invisible.
        assert len(result.entries) == 0
    finally:
        store.close()


def test_tail_accumulates_matches_across_segments(tmp_path: Path):
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
            store._force_flush()

        # Ask for 6 MATCH rows; must pull from at least 3 segments.
        result = store.get_logs("/job/test/.*", max_lines=6, tail=True, substring_filter="MATCH")
        assert len(result.entries) == 6
        assert all("MATCH" in e.data for e in result.entries)
        # Results are sorted ascending in LogReadResult (DESC then reversed).
        seqs = [e.timestamp.epoch_ms for e in result.entries]
        assert seqs == sorted(seqs)
    finally:
        store.close()
