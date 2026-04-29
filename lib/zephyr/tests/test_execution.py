# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the actor-based execution engine (ZephyrContext)."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fray import ResourceConfig
from fray.local_backend import LocalClient
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import (
    MAX_SHARD_FAILURES,
    CounterSnapshot,
    ListShard,
    PickleDiskChunk,
    ShardTask,
    TaskResult,
    WorkerState,
    ZephyrContext,
    ZephyrCoordinator,
    ZephyrWorkerError,
    zephyr_worker_ctx,
)
from zephyr.plan import compute_plan


def test_counter_flusher(tmp_path):
    """Counter file flushed during shard execution reflects actual counter increments."""
    import cloudpickle
    import zephyr.subprocess_worker as sw

    from zephyr import counters
    from zephyr.execution import ListShard, ShardTask
    from zephyr.plan import Map
    from zephyr.shuffle import MemChunk

    original_interval = sw.SUBPROCESS_COUNTER_FLUSH_INTERVAL
    sw.SUBPROCESS_COUNTER_FLUSH_INTERVAL = 0.01  # flush aggressively during the test

    try:

        def counting_map(stream):
            for item in stream:
                counters.increment("items", 1)
                time.sleep(0.05)  # longer than flush interval — guarantees ≥1 flush
                yield item

        chunk_prefix = str(tmp_path / "chunks")
        execution_id = "test-exec"
        task = ShardTask(
            shard_idx=0,
            total_shards=1,
            shard=ListShard(refs=[MemChunk([1, 2, 3])]),
            operations=[Map(fn=counting_map)],
            stage_name="test",
        )

        task_file = str(tmp_path / "task.pkl")
        result_file = str(tmp_path / "result.pkl")
        counter_file = f"{result_file}.counters"

        with open(task_file, "wb") as f:
            cloudpickle.dump((task, chunk_prefix, execution_id), f)

        sw.execute_shard(task_file, result_file)

        assert Path(counter_file).exists(), "counter file was never written — flusher did not run"
        with open(counter_file, "rb") as f:
            flushed = cloudpickle.load(f)
        assert flushed.get("items", 0) > 0, (
            f"counter file is empty ({flushed!r}); flusher likely held a dummy context "
            "instead of the real one created from the task file"
        )
    finally:
        sw.SUBPROCESS_COUNTER_FLUSH_INTERVAL = original_interval


def test_simple_map(zephyr_ctx):
    """Map pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = zephyr_ctx.execute(ds).results
    assert sorted(results) == [2, 4, 6]


def test_filter(zephyr_ctx):
    """Filter pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).filter(lambda x: x > 3)
    results = zephyr_ctx.execute(ds).results
    assert sorted(results) == [4, 5]


def test_subprocess_propagates_user_counters(zephyr_ctx):
    """User counters incremented inside the shard subprocess flow back to the coordinator.

    Each task runs in a fresh Python subprocess, so ``counters.increment`` writes
    into a ``_SubprocessWorkerContext`` that lives only in the child. This test
    verifies the result file ships those increments back to the parent worker,
    which then forwards them to the coordinator via ``report_result``. Without
    that round-trip, the coordinator's ``get_counters`` would silently report 0.

    Uses a direct logging handler attachment (rather than ``caplog``) so the
    test works whether or not pytest's logging plugin is enabled.
    """

    def increment_per_item(x: int) -> int:
        counters.increment("docs", 1)
        counters.increment("doubled_sum", x * 2)
        return x

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Capture(level=logging.INFO)
    target_logger = logging.getLogger("zephyr.execution")
    prior_level = target_logger.level
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.INFO)
    try:
        ds = Dataset.from_list([1, 2, 3, 4, 5]).map(increment_per_item)
        results = zephyr_ctx.execute(ds).results
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(prior_level)

    assert sorted(results) == [1, 2, 3, 4, 5]

    # Coordinator logs the aggregated counters on shutdown. Look for the most
    # recent "Final counters:" line and check the dict it printed.
    final_lines = [m for m in captured if "Final counters:" in m]
    assert final_lines, "coordinator did not log Final counters — counter plumbing is broken"
    last = final_lines[-1]
    assert "'docs': 5" in last, f"expected 'docs': 5 in {last!r}"
    assert "'doubled_sum': 30" in last, f"expected 'doubled_sum': 30 in {last!r}"


def test_subprocess_exception_includes_subprocess_traceback(zephyr_ctx):
    """Exceptions raised inside the shard subprocess surface with the original frame info.

    Cloudpickling an exception drops ``__traceback__`` so a naive re-raise in
    the parent shows only the parent's stack at the re-raise site, not where
    the exception actually happened in the user lambda. The subprocess
    attaches the formatted traceback as a ``__notes__`` entry, which Python
    prints inline when the exception finally propagates. Verify both the
    original exception type AND a snippet from the user-code frame survive
    the round-trip.
    """

    def buggy_index_lookup(x: int) -> int:
        # Force a non-trivial subprocess-side exception with a recognizable
        # source line. The parent should surface the function name and the
        # offending statement, not just the bare `IndexError`.
        empty: tuple = ()
        return empty[x]

    ds = Dataset.from_list([0]).map(buggy_index_lookup)

    with pytest.raises(ZephyrWorkerError) as exc_info:
        zephyr_ctx.execute(ds)

    rendered = str(exc_info.value) + "".join(getattr(exc_info.value, "__notes__", []))
    # The wrapping ZephyrWorkerError should chain through the parent's
    # report_error path. Either the chained cause or the notes payload
    # must contain the user-frame breadcrumb.
    chained = rendered
    cur: BaseException | None = exc_info.value
    while cur is not None:
        chained += str(cur) + "".join(getattr(cur, "__notes__", []))
        cur = cur.__cause__ or cur.__context__

    assert (
        "buggy_index_lookup" in chained or "tuple index out of range" in chained
    ), f"subprocess traceback was not preserved through report_error; got: {chained!r}"


def test_shared_data(integration_client, tmp_path):
    """Workers can access shared data via zephyr_worker_ctx().

    Shared data is serialized to disk by put() and loaded lazily by workers.
    """

    def use_shared(x):
        multiplier = zephyr_worker_ctx().get_shared("multiplier")
        return x * multiplier

    zctx = ZephyrContext(
        client=integration_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    zctx.put("multiplier", 10)
    ds = Dataset.from_list([1, 2, 3]).map(use_shared)
    results = zctx.execute(ds).results
    assert sorted(results) == [10, 20, 30]
    zctx.shutdown()


def test_multi_stage(zephyr_ctx):
    """Multi-stage pipeline (map + filter) works."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(lambda x: x > 5)
    results = zephyr_ctx.execute(ds).results
    assert sorted(results) == [6, 8, 10]


def test_context_manager(local_client):
    """ZephyrContext works without context manager."""
    zctx = ZephyrContext(
        client=local_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x + 1)
    results = zctx.execute(ds).results
    assert sorted(results) == [2, 3, 4]


def test_write_jsonl(tmp_path, zephyr_ctx):
    """Pipeline writing to jsonl file."""
    output = str(tmp_path / "out-{shard}.jsonl")
    ds = Dataset.from_list([{"a": 1}, {"a": 2}, {"a": 3}]).write_jsonl(output)
    results = zephyr_ctx.execute(ds).results
    assert len(results) == 3
    # Verify all files were written and contain correct data
    all_records = []
    for path_str in results:
        written = Path(path_str)
        assert written.exists()
        lines = written.read_text().strip().split("\n")
        all_records.extend(json.loads(line) for line in lines)
    assert sorted(all_records, key=lambda r: r["a"]) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_dry_run(zephyr_ctx):
    """Dry run shows plan without executing."""
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = zephyr_ctx.execute(ds, dry_run=True).results
    assert results == []


def test_flat_map(zephyr_ctx):
    """FlatMap pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3]).flat_map(lambda x: [x, x * 10])
    results = zephyr_ctx.execute(ds).results
    assert sorted(results) == [1, 2, 3, 10, 20, 30]


def test_empty_dataset(zephyr_ctx):
    """Empty dataset produces empty results."""
    ds = Dataset.from_list([])
    results = zephyr_ctx.execute(ds).results
    assert results == []


def test_chunk_cleanup(local_client, tmp_path):
    """Verify chunks are cleaned up after execution."""
    chunk_prefix = str(tmp_path / "chunks")
    ctx = ZephyrContext(
        client=local_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )

    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = ctx.execute(ds).results

    assert sorted(results) == [2, 4, 6]

    # Verify chunks directory is cleaned up
    if os.path.exists(chunk_prefix):
        # Should be empty or not exist
        files = list(Path(chunk_prefix).rglob("*"))
        assert len(files) == 0, f"Expected cleanup but found: {files}"


def test_status_reports_alive_workers_not_total(actor_context, tmp_path):
    """After heartbeat timeout, get_status workers dict reflects FAILED state,
    and the status log distinguishes alive from total workers.

    Also verifies that re-registering a worker that had an in-flight task
    requeues that task so it is not silently lost.
    """
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])

    # Register 3 workers
    for i in range(3):
        coord.register_worker(f"worker-{i}", MagicMock())

    status = coord.get_status()
    assert len(status.workers) == 3
    assert all(w["state"] == "ready" for w in status.workers.values())

    # worker-0 pulls the task so it becomes in-flight
    pulled = coord.pull_task("worker-0")
    assert pulled is not None and pulled != "SHUTDOWN"

    # Simulate 2 workers dying via heartbeat timeout
    coord._last_seen["worker-0"] = 0.0
    coord._last_seen["worker-1"] = 0.0
    coord.check_heartbeats(timeout=30.0)

    status = coord.get_status()
    assert status.workers["worker-0"]["state"] == "failed"
    assert status.workers["worker-1"]["state"] == "failed"
    assert status.workers["worker-2"]["state"] == "ready"

    # Total workers in dict is still 3, but only 1 is alive
    alive = sum(1 for w in status.workers.values() if w["state"] in ("ready", "busy"))
    assert alive == 1
    assert len(status.workers) == 3

    # worker-2 picks up the requeued task
    pulled2 = coord.pull_task("worker-2")
    assert pulled2 is not None and pulled2 != "SHUTDOWN"

    # Simulate worker-0 re-registering while worker-2 holds the task in-flight
    # (race between heartbeat requeue and re-registration).
    # Since worker-0 has no in-flight task anymore, this is a no-op for requeueing.
    coord.register_worker("worker-0", MagicMock())
    status = coord.get_status()
    assert status.workers["worker-0"]["state"] == "ready"
    alive = sum(1 for w in status.workers.values() if w["state"] in ("ready", "busy"))
    assert alive == 2  # worker-0 ready, worker-1 still failed, worker-2 busy

    # Now test the direct re-registration requeue path:
    # worker-2 dies while holding the task, and before heartbeat fires,
    # it re-registers — the in-flight task should be requeued.
    assert "worker-2" in coord._in_flight  # worker-2 has the task
    coord.register_worker("worker-2", MagicMock())
    assert "worker-2" not in coord._in_flight  # in-flight cleared
    assert len(coord._task_queue) == 1  # task was requeued


def test_log_status_omits_throughput_when_counters_missing(actor_context, tmp_path, caplog):
    """Map-only stages don't populate item/byte counters, so the coordinator's
    status log should drop the ``items=... bytes_processed=...`` segment rather
    than print misleading zeros. Once either counter is recorded, the segment
    reappears."""
    from zephyr.execution import ZEPHYR_STAGE_BYTES_PROCESSED_KEY, ZEPHYR_STAGE_ITEM_COUNT_KEY

    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="map_only",
    )
    coord._start_stage("map_only", [task])

    # No counters recorded → throughput segment is suppressed.
    with caplog.at_level(logging.INFO, logger="zephyr.execution"):
        caplog.clear()
        coord._log_status()
    msgs = [r.getMessage() for r in caplog.records if "complete" in r.getMessage()]
    assert msgs, "expected a status line"
    assert all("items=" not in m and "bytes_processed=" not in m for m in msgs), msgs

    # Once a counter snapshot exists, the throughput segment reappears.
    coord._worker_counters["worker-A"] = CounterSnapshot(
        counters={ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name="map_only"): 7}, generation=1
    )
    with caplog.at_level(logging.INFO, logger="zephyr.execution"):
        caplog.clear()
        coord._log_status()
    msgs = [r.getMessage() for r in caplog.records if "complete" in r.getMessage()]
    assert msgs and "items=7" in msgs[-1] and "bytes_processed=0.0MiB" in msgs[-1], msgs

    # Same when only the byte counter is present.
    coord._worker_counters["worker-A"] = CounterSnapshot(
        counters={ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name="map_only"): 1024}, generation=2
    )
    with caplog.at_level(logging.INFO, logger="zephyr.execution"):
        caplog.clear()
        coord._log_status()
    msgs = [r.getMessage() for r in caplog.records if "complete" in r.getMessage()]
    assert msgs and "items=0" in msgs[-1] and "bytes_processed=" in msgs[-1], msgs


def test_no_duplicate_results_on_heartbeat_timeout(actor_context, tmp_path):
    """When a task is requeued after heartbeat timeout, the original worker's
    stale result (from a previous attempt) is rejected by the coordinator."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])

    # Worker A pulls task (attempt 0)
    pulled = coord.pull_task("worker-A")
    assert pulled is not None
    assert pulled != "SHUTDOWN"
    _task_a, attempt_a, _config = pulled

    # Simulate heartbeat timeout: mark worker-A stale and requeue
    coord._last_seen["worker-A"] = 0.0
    coord.check_heartbeats(timeout=0.0)

    # Task should be requeued with incremented attempt
    assert coord._task_attempts[0] == 1

    # Worker B picks up the requeued task (attempt 1)
    pulled_b = coord.pull_task("worker-B")
    assert pulled_b is not None
    assert pulled_b != "SHUTDOWN"
    _task_b, attempt_b, _config = pulled_b
    assert attempt_b == 1

    # Worker B reports success
    coord.report_result("worker-B", 0, attempt_b, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    # Worker A's stale result (attempt 0) should be ignored
    coord.report_result("worker-A", 0, attempt_a, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    # Only one completion should be counted
    assert coord._completed_shards == 1


def test_disk_chunk_write_uses_unique_paths(tmp_path):
    """Each PickleDiskChunk.write() writes to a unique location, avoiding collisions."""
    base_path = str(tmp_path / "chunk.pkl")
    refs = [PickleDiskChunk.write(base_path, [i]) for i in range(3)]

    # Each written to a distinct UUID path (no rename needed)
    paths = [r.path for r in refs]
    assert len(set(paths)) == 3
    for p in paths:
        assert ".tmp." in p
        assert Path(p).exists()

    # Each chunk is directly readable
    for i, ref in enumerate(refs):
        assert ref.read() == [i]


def test_coordinator_accepts_winner_ignores_stale(actor_context, tmp_path):
    """Coordinator accepts the winning result and ignores stale ones.

    Stale chunk files are left for context-dir cleanup (no per-chunk deletion).
    """
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])

    # Worker A pulls task (attempt 0)
    pulled_a = coord.pull_task("worker-A")
    _task_a, attempt_a, _config = pulled_a

    # Worker A writes a chunk (simulating slow completion)
    stale_ref = PickleDiskChunk.write(str(tmp_path / "stale-chunk.pkl"), [1, 2, 3])
    assert Path(stale_ref.path).exists()

    # Heartbeat timeout re-queues the task
    coord._last_seen["worker-A"] = 0.0
    coord.check_heartbeats(timeout=0.0)

    # Worker B pulls and completes the re-queued task (attempt 1)
    pulled_b = coord.pull_task("worker-B")
    _task_b, attempt_b, _config = pulled_b

    winner_ref = PickleDiskChunk.write(str(tmp_path / "winner-chunk.pkl"), [4, 5, 6])

    coord.report_result(
        "worker-B",
        0,
        attempt_b,
        TaskResult(shard=ListShard(refs=[winner_ref])),
        CounterSnapshot.empty(),
    )

    # Worker A's stale result is rejected
    coord.report_result(
        "worker-A",
        0,
        attempt_a,
        TaskResult(shard=ListShard(refs=[stale_ref])),
        CounterSnapshot.empty(),
    )

    # Winner's data is directly readable (no rename needed)
    assert Path(winner_ref.path).exists()
    assert winner_ref.read() == [4, 5, 6]

    # Stale file still exists (cleaned up by context-dir cleanup, not coordinator)
    assert Path(stale_ref.path).exists()
    assert coord._completed_shards == 1


def test_shard_streaming_low_memory(tmp_path):
    """ListShard loads refs one at a time from disk via get_iterators.

    Verifies get_iterators yields data lazily and flat iteration works.
    """
    # Write 3 refs to disk (directly readable, no finalize needed)
    refs = []
    for i in range(3):
        path = str(tmp_path / f"chunk-{i}.pkl")
        chunk = PickleDiskChunk.write(path, [i * 10 + j for j in range(5)])
        refs.append(chunk)

    shard = ListShard(refs=refs)

    # get_iterators yields one iterator per ref
    chunks = [list(it) for it in shard.get_iterators()]
    assert len(chunks) == 3
    assert chunks[0] == [0, 1, 2, 3, 4]
    assert chunks[2] == [20, 21, 22, 23, 24]

    # flat iteration yields all items in order
    assert list(shard) == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]

    # Re-iteration works (reads from disk again, not cached)
    assert list(shard) == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]


def test_report_error_requeues_until_max_shard_failures(actor_context, tmp_path):
    """report_error re-queues a task until MAX_SHARD_FAILURES, then aborts."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])
    coord.register_worker("worker-0", MagicMock())

    # Each failure should re-queue until the limit
    for i in range(MAX_SHARD_FAILURES - 1):
        pulled = coord.pull_task("worker-0")
        assert pulled is not None and pulled != "SHUTDOWN"
        _task, _attempt, _config = pulled
        coord.report_error("worker-0", 0, f"error-{i}")
        assert coord._fatal_error is None, f"Should not abort on failure {i + 1}"
        assert coord._worker_states["worker-0"] == WorkerState.READY

    # The final failure should set _fatal_error
    pulled = coord.pull_task("worker-0")
    assert pulled is not None and pulled != "SHUTDOWN"
    coord.report_error("worker-0", 0, "final-error")
    assert coord._fatal_error is not None
    assert "Shard 0" in coord._fatal_error
    assert "final-error" in coord._fatal_error


def test_heartbeat_timeouts_do_not_count_toward_shard_failures(actor_context, tmp_path):
    """Heartbeat-timeout requeues (preemption) must not consume MAX_SHARD_FAILURES."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])
    coord.register_worker("worker-0", MagicMock())

    # Far more heartbeat timeouts than MAX_SHARD_FAILURES — must not abort.
    for _ in range(MAX_SHARD_FAILURES * 5):
        pulled = coord.pull_task("worker-0")
        assert pulled is not None and pulled != "SHUTDOWN"
        coord._last_seen["worker-0"] = 0.0
        coord.check_heartbeats(timeout=0.0)
        assert coord._fatal_error is None

    # Task-error budget is untouched; a successful completion closes the shard.
    assert coord._task_error_attempts[0] == 0
    pulled = coord.pull_task("worker-0")
    assert pulled is not None and pulled != "SHUTDOWN"
    _task, attempt, _config = pulled
    coord.report_result("worker-0", 0, attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())
    assert coord._completed_shards == 1
    assert coord._fatal_error is None


def test_worker_reregistration_does_not_count_toward_shard_failures(actor_context, tmp_path):
    """Preemption-driven worker re-registration requeues without burning MAX_SHARD_FAILURES."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])
    coord.register_worker("worker-0", MagicMock())

    for _ in range(MAX_SHARD_FAILURES * 5):
        pulled = coord.pull_task("worker-0")
        assert pulled is not None and pulled != "SHUTDOWN"
        # Simulate preemption + Iris reconstruction: worker re-registers while
        # a task is still recorded as in-flight on the old handle.
        coord.register_worker("worker-0", MagicMock())
        assert "worker-0" not in coord._in_flight
        assert coord._fatal_error is None

    assert coord._task_error_attempts[0] == 0


def test_report_error_still_aborts_at_max_shard_failures_after_preemptions(actor_context, tmp_path):
    """Task errors still abort at MAX_SHARD_FAILURES even after many survived preemptions."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])
    coord.register_worker("worker-0", MagicMock())

    # Several preemption cycles first — these must not count.
    for _ in range(5):
        pulled = coord.pull_task("worker-0")
        assert pulled is not None and pulled != "SHUTDOWN"
        coord._last_seen["worker-0"] = 0.0
        coord.check_heartbeats(timeout=0.0)

    assert coord._fatal_error is None

    # Now MAX_SHARD_FAILURES explicit task errors should abort.
    for i in range(MAX_SHARD_FAILURES):
        pulled = coord.pull_task("worker-0")
        assert pulled is not None and pulled != "SHUTDOWN"
        coord.report_error("worker-0", 0, f"boom-{i}")

    assert coord._fatal_error is not None
    assert "Shard 0" in coord._fatal_error


def test_wait_for_stage_fails_when_all_workers_die(actor_context, tmp_path):
    """When all registered workers become dead/failed, _wait_for_stage raises
    after the no_workers_timeout instead of waiting forever."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord._no_workers_timeout = 0.5  # short timeout for test

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])

    # Register 2 workers
    coord.register_worker("worker-0", MagicMock())
    coord.register_worker("worker-1", MagicMock())

    # Kill all workers via heartbeat timeout
    coord._last_seen["worker-0"] = 0.0
    coord._last_seen["worker-1"] = 0.0
    coord.check_heartbeats(timeout=0.0)

    assert coord._worker_states["worker-0"] == WorkerState.FAILED
    assert coord._worker_states["worker-1"] == WorkerState.FAILED

    # _wait_for_stage should raise after the dead timer expires
    with pytest.raises(ZephyrWorkerError, match="No alive workers"):
        coord._wait_for_stage()


def test_wait_for_stage_resets_dead_timer_on_recovery(actor_context, tmp_path):
    """When a worker recovers (re-registers) after all workers died,
    the dead timer resets and execution can continue."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord._no_workers_timeout = 2.0

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )
    coord._start_stage("test", [task])

    # Register and kill a worker
    coord.register_worker("worker-0", MagicMock())
    coord._last_seen["worker-0"] = 0.0
    coord.check_heartbeats(timeout=0.0)
    assert coord._worker_states["worker-0"] == WorkerState.FAILED

    # In a background thread, re-register the worker and complete the task
    # after a short delay (simulating recovery before timeout expires)
    def recover_and_complete():
        time.sleep(0.1)
        coord.register_worker("worker-0", MagicMock())
        pulled = coord.pull_task("worker-0")
        assert pulled is not None and pulled != "SHUTDOWN"
        _task, attempt, _config = pulled
        coord.report_result("worker-0", 0, attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    t = threading.Thread(target=recover_and_complete)
    t.start()

    # _wait_for_stage should succeed (worker recovers before timeout)
    coord._wait_for_stage()
    t.join(timeout=5.0)

    assert coord._completed_shards == 1


def test_fresh_actors_per_execute(integration_client, tmp_path):
    """Each execute() creates and tears down its own coordinator and workers."""
    chunk_prefix = str(tmp_path / "chunks")

    zctx = ZephyrContext(
        client=integration_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x + 1)
    results = zctx.execute(ds).results
    assert sorted(results) == [2, 3, 4]

    # After execute(): coordinator job is torn down
    assert zctx._coordinator_job is None
    assert zctx._pipeline_id == 0

    # Can execute again (creates fresh coordinator job)
    ds2 = Dataset.from_list([10, 20]).map(lambda x: x * 2)
    results2 = zctx.execute(ds2).results
    assert sorted(results2) == [20, 40]

    assert zctx._coordinator_job is None
    assert zctx._pipeline_id == 1


def test_fatal_errors_fail_fast(local_client, tmp_path):
    """Application errors (e.g. ValueError) cause immediate failure, no retries."""
    chunk_prefix = str(tmp_path / "chunks")

    def exploding_map(x):
        raise ValueError(f"bad value: {x}")

    zctx = ZephyrContext(
        client=local_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(exploding_map)

    start = time.monotonic()
    with pytest.raises(ZephyrWorkerError, match="ValueError"):
        zctx.execute(ds)
    elapsed = time.monotonic() - start

    # Should fail fast — well under the 30s heartbeat timeout
    assert elapsed < 15.0, f"Took {elapsed:.1f}s, expected fast failure"


def test_chunk_storage_with_join(integration_client, tmp_path):
    """Verify chunk storage works with join operations."""
    chunk_prefix = str(tmp_path / "chunks")
    ctx = ZephyrContext(
        client=integration_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )

    # Create datasets for join
    left = Dataset.from_list([{"id": 1, "a": "x"}, {"id": 2, "a": "y"}])
    right = Dataset.from_list([{"id": 1, "b": "p"}, {"id": 2, "b": "q"}])

    joined = left.sorted_merge_join(
        right,
        left_key=lambda x: x["id"],
        right_key=lambda x: x["id"],
        combiner=lambda left, right: {**left, **right},
    )

    results = sorted(ctx.execute(joined).results, key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0] == {"id": 1, "a": "x", "b": "p"}
    assert results[1] == {"id": 2, "a": "y", "b": "q"}


def test_workers_capped_to_shard_count(local_client, tmp_path):
    """When max_workers > num_shards, only num_shards workers are created."""
    ds = Dataset.from_list([1, 2, 3])  # 3 shards
    ctx = ZephyrContext(
        client=local_client,
        max_workers=10,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    results = ctx.execute(ds.map(lambda x: x * 2)).results
    assert sorted(results) == [2, 4, 6]
    # Everything torn down after execute; correct results prove workers
    # were created and sized properly (min(10, 3) = 3)
    assert ctx._pipeline_id == 0


def test_pipeline_id_increments(local_client, tmp_path):
    """Pipeline ID increments after each execute(), ensuring unique actor names."""
    ctx = ZephyrContext(
        client=local_client,
        max_workers=10,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ctx.execute(Dataset.from_list([1, 2]).map(lambda x: x))
    assert ctx._pipeline_id == 0

    ctx.execute(Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x))
    assert ctx._pipeline_id == 1


def test_pull_task_returns_shutdown_on_last_stage_empty_queue(actor_context, tmp_path):
    """When the last stage's tasks are all in-flight or done, pull_task returns SHUTDOWN."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
    )

    # Non-last stage: empty queue returns None
    coord._start_stage("stage-0", [task], is_last_stage=False)
    pulled = coord.pull_task("worker-A")
    assert pulled is not None and pulled != "SHUTDOWN"
    _task, attempt, _config = pulled
    coord.report_result("worker-A", 0, attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    # Queue empty, but not last stage -> None
    result = coord.pull_task("worker-A")
    assert result is None

    # Last stage: single task, worker completes it, queue empty -> SHUTDOWN
    task2 = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test-last",
    )
    coord._start_stage("stage-1", [task2], is_last_stage=True)
    pulled = coord.pull_task("worker-A")
    assert pulled is not None and pulled != "SHUTDOWN"
    _task, attempt, _config = pulled
    coord.report_result("worker-A", 0, attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    # Queue empty on last stage, nothing in-flight -> SHUTDOWN
    result = coord.pull_task("worker-A")
    assert result == "SHUTDOWN"

    # Last stage with in-flight tasks: idle workers still get SHUTDOWN
    tasks_2 = [
        ShardTask(shard_idx=i, total_shards=2, shard=ListShard(refs=[]), operations=[], stage_name="test-last2")
        for i in range(2)
    ]
    coord._start_stage("stage-2", tasks_2, is_last_stage=True)
    coord.pull_task("worker-A")  # task 0 in-flight
    # Queue has one task left; worker-B takes it
    coord.pull_task("worker-B")  # task 1 in-flight
    # Queue empty, tasks in-flight -> SHUTDOWN (workers exit immediately)
    result = coord.pull_task("worker-C")
    assert result == "SHUTDOWN"


def test_last_stage_deadlock_detected_when_worker_job_dies(actor_context, tmp_path):
    """Coordinator aborts if the worker job dies while last-stage work is outstanding."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    tasks = [
        ShardTask(shard_idx=i, total_shards=2, shard=ListShard(refs=[]), operations=[], stage_name="test")
        for i in range(2)
    ]
    coord._start_stage("last-stage", tasks, is_last_stage=True)

    # Set up a mock worker group so _check_worker_group can query it.
    mock_group = MagicMock()
    mock_group.is_done.return_value = False
    coord.set_worker_group(mock_group)

    # Two workers pull both tasks.
    coord.heartbeat("worker-A")
    coord.heartbeat("worker-B")
    pulled_a = coord.pull_task("worker-A")
    coord.pull_task("worker-B")

    # Worker A finishes → gets SHUTDOWN (queue empty, last stage).
    _task_a, attempt_a, _ = pulled_a
    coord.report_result(
        "worker-A", _task_a.shard_idx, attempt_a, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )
    assert coord.pull_task("worker-A") == "SHUTDOWN"

    # Worker B crashes → heartbeat timeout → shard 1 requeued.
    coord._last_seen["worker-B"] = coord._last_seen["worker-B"] - 200
    coord.check_heartbeats(timeout=10)
    assert len(coord._task_queue) == 1

    # Worker job is still running — no abort yet.
    coord._check_worker_group()
    assert coord._fatal_error is None

    # Worker job dies permanently (Iris exhausted retries).
    mock_group.is_done.return_value = True
    coord._check_worker_group()

    # Coordinator should detect the deadlock and abort.
    assert coord._fatal_error is not None
    assert "terminated permanently" in coord._fatal_error


def test_coordinator_loop_crash_aborts_pipeline(actor_context, tmp_path):
    """Coordinator loop crash sets _fatal_error instead of dying silently. #3996."""
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")

    crashed = threading.Event()
    original = coord.check_heartbeats

    def crashing_heartbeats(*a, **kw):
        if not crashed.is_set():
            crashed.set()
            raise RuntimeError("dictionary changed size during iteration")
        return original(*a, **kw)

    coord.check_heartbeats = crashing_heartbeats

    t = threading.Thread(target=coord._coordinator_loop, daemon=True, name="zephyr-coordinator-loop")
    t.start()
    assert crashed.wait(timeout=5.0)
    t.join(timeout=2.0)
    assert coord._fatal_error is not None


def test_run_pipeline_rejects_concurrent_calls(actor_context, tmp_path):
    """Calling run_pipeline while another is already running raises RuntimeError."""
    coord = ZephyrCoordinator()
    coord.initialize(str(tmp_path / "chunks"), MagicMock())

    gate = threading.Event()
    ds = Dataset.from_list([42]).map(lambda x: gate.wait(timeout=5) or x)
    plan = compute_plan(ds)
    # First call blocks because the map waits on `gate` (no workers to run it
    # anyway). We patch _wait_for_stage to signal when it's entered.
    first_entered = threading.Event()
    original_wait = coord._wait_for_stage

    def blocking_wait():
        first_entered.set()
        time.sleep(0.1)
        coord._fatal_error = "test: forced exit"
        try:
            original_wait()
        except Exception:
            pass

    coord._wait_for_stage = blocking_wait

    t = threading.Thread(target=lambda: coord.run_pipeline(plan, "exec-1"), daemon=True)
    t.start()
    first_entered.wait(timeout=5.0)

    # Second call should fail immediately
    with pytest.raises(RuntimeError, match="already running"):
        coord.run_pipeline(plan, "exec-2")

    t.join(timeout=10.0)
    coord.shutdown()


def test_execute_stops_coordinator_thread(local_client, tmp_path):
    """execute() tears down the hosted coordinator loop before returning."""
    chunk_prefix = str(tmp_path / "chunks")
    baseline = sum(t.is_alive() and t.name == "zephyr-coordinator-loop" for t in threading.enumerate())

    ctx = ZephyrContext(
        client=local_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )

    results = ctx.execute(Dataset.from_list([1, 2, 3]).map(lambda x: x + 1)).results
    assert sorted(results) == [2, 3, 4]

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        alive = sum(t.is_alive() and t.name == "zephyr-coordinator-loop" for t in threading.enumerate())
        if alive == baseline:
            break
        time.sleep(0.05)
    else:
        pytest.fail("zephyr-coordinator-loop thread remained alive after execute() returned")

    ctx.shutdown()


def test_execute_retries_on_coordinator_death(tmp_path):
    """When the coordinator job fails, execute() retries with a fresh job
    and eventually succeeds.

    Patches client.submit so the first coordinator job submission raises,
    then the retry submits a real job that succeeds.
    """
    client = LocalClient()
    chunk_prefix = str(tmp_path / "chunks")

    ctx = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        max_execution_retries=2,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )

    # First execute() succeeds normally
    results = ctx.execute(Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)).results
    assert sorted(results) == [2, 4, 6]

    # Patch submit to fail on the first coordinator job, then succeed on retry.
    original_submit = client.submit
    submit_count = [0]

    def flaky_submit(request, adopt_existing=True):
        if "zephyr-" in request.name:
            submit_count[0] += 1
            if submit_count[0] == 1:
                raise RuntimeError("Simulated coordinator job submission failure")
        return original_submit(request, adopt_existing)

    client.submit = flaky_submit

    # Next execute() should: fail on attempt 0 (submit raises),
    # then succeed on attempt 1 with a fresh coordinator job.
    results = ctx.execute(Dataset.from_list([10, 20]).map(lambda x: x + 1)).results
    assert sorted(results) == [11, 21]
    assert submit_count[0] >= 2, "Expected at least 2 submit attempts (1 failed + 1 succeeded)"

    ctx.shutdown()
    client.shutdown(wait=True)


def test_execute_does_not_retry_worker_errors(local_client, tmp_path):
    """ZephyrWorkerError (application errors) are never retried."""
    chunk_prefix = str(tmp_path / "chunks")

    def exploding_map(x):
        raise ValueError(f"bad value: {x}")

    ctx = ZephyrContext(
        client=local_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        max_execution_retries=3,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(exploding_map)

    start = time.monotonic()
    with pytest.raises(ZephyrWorkerError, match="ValueError"):
        ctx.execute(ds)
    elapsed = time.monotonic() - start

    # Should fail fast — no retries for application errors
    assert elapsed < 15.0, f"Took {elapsed:.1f}s, expected fast failure (no retries)"


# --- Integration tests (all backends) ---


def test_simple_map_integration(integration_ctx):
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    assert sorted(integration_ctx.execute(ds).results) == [2, 4, 6]


def test_multi_stage_integration(integration_ctx):
    ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(lambda x: x > 5)
    assert sorted(integration_ctx.execute(ds).results) == [6, 8, 10]
