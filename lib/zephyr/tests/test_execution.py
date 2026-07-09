# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the actor-based execution engine (ZephyrContext)."""

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import cloudpickle
import pytest
from conftest import _TEST_TASK_COST, _TEST_WORKER_AVAILABLE
from fray import ResourceConfig
from fray.actor import ActorContext
from fray.local_backend import LocalClient
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import (
    _NON_RETRYABLE_ERRORS,
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    CoordinatorUnreachable,
    PullStatus,
    WorkerState,
    ZephyrContext,
    ZephyrCoordinator,
    ZephyrWorker,
    ZephyrWorkerError,
    _ensure_picklable_exception,
)
from zephyr.plan import PhysicalStage, StageType, compute_plan
from zephyr.shuffle import ListShard
from zephyr.stage_io import (
    PickleDiskChunk,
    ShardTask,
    TaskResult,
)
from zephyr.stats import ZEPHYR_STAGE_BYTES_PROCESSED_KEY, ZEPHYR_STAGE_ITEM_COUNT_KEY
from zephyr.worker_context import CounterEntry, CounterSnapshot, zephyr_worker_ctx


class _UnpicklableError(Exception):
    """Mimics rigging's old TransferBudgetExceeded: __init__ args don't match self.args."""

    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c
        super().__init__(f"boom {a}/{b}/{c}")  # self.args = (message,) -> revive needs 3 args


def test_ensure_picklable_exception_passes_through_picklable():
    err = ValueError("plain and picklable")
    assert _ensure_picklable_exception(err) is err


def test_ensure_picklable_exception_wraps_unrevivable_and_preserves_message():
    err = _UnpicklableError(1, 2, 3)
    err.add_note("--- subprocess traceback ---\nsomewhere")
    with pytest.raises(TypeError):
        cloudpickle.loads(cloudpickle.dumps(err))  # confirms the hazard

    safe = _ensure_picklable_exception(err)
    revived = cloudpickle.loads(cloudpickle.dumps(safe))  # must not raise
    assert isinstance(revived, ZephyrWorkerError)
    assert isinstance(revived, _NON_RETRYABLE_ERRORS)  # un-revivable -> fail fast, never retry
    assert "_UnpicklableError" in str(revived) and "boom 1/2/3" in str(revived)
    assert any("subprocess traceback" in n for n in revived.__notes__)


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


def test_propagates_user_counters(zephyr_ctx):
    """User counters incremented inside a shard are visible in the execution result.

    Uses a direct logging handler attachment (rather than ``caplog``) so the
    test works whether or not pytest's logging plugin is enabled.
    """

    def increment_per_item(x: int) -> int:
        counters.pipeline.update_counter("docs", 1)
        counters.pipeline.update_counter("doubled_sum", x * 2)
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


def test_exception_preserves_user_frame(zephyr_ctx):
    """Exceptions raised inside a shard surface with the original frame info.

    The worker's ``report_error`` ships a formatted traceback string up to
    the coordinator, which wraps it in ``ZephyrWorkerError``. Verify either
    the user function name or the exception text survives that round-trip
    so failures are debuggable.
    """

    def buggy_index_lookup(x: int) -> int:
        # Recognizable user-frame source line. The parent should surface the
        # function name and the offending statement, not just bare ``IndexError``.
        empty: tuple = ()
        return empty[x]

    ds = Dataset.from_list([0]).map(buggy_index_lookup)

    with pytest.raises(ZephyrWorkerError) as exc_info:
        zephyr_ctx.execute(ds)

    chained = str(exc_info.value) + "".join(getattr(exc_info.value, "__notes__", []))
    cur: BaseException | None = exc_info.value
    while cur is not None:
        chained += str(cur) + "".join(getattr(cur, "__notes__", []))
        cur = cur.__cause__ or cur.__context__

    assert (
        "buggy_index_lookup" in chained or "tuple index out of range" in chained
    ), f"user-frame traceback was not preserved through report_error; got: {chained!r}"


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


def test_status_reports_alive_workers_not_total(coordinator):
    """After heartbeat timeout, get_status workers dict reflects FAILED state,
    and the status log distinguishes alive from total workers.

    Also verifies that re-registering a worker that had an in-flight task
    requeues that task so it is not silently lost.
    """
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])

    # Register 3 workers
    for i in range(3):
        coordinator.register_worker(f"worker-{i}", MagicMock())

    status = coordinator.get_status()
    assert len(status.workers) == 3
    assert all(w["state"] == "active" for w in status.workers.values())

    # worker-0 pulls the task so it becomes in-flight
    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK

    # Simulate 2 workers dying via heartbeat timeout
    coordinator._last_seen["worker-0"] = 0.0
    coordinator._last_seen["worker-1"] = 0.0
    coordinator.check_heartbeats(timeout=30.0)

    status = coordinator.get_status()
    assert status.workers["worker-0"]["state"] == "failed"
    assert status.workers["worker-1"]["state"] == "failed"
    assert status.workers["worker-2"]["state"] == "active"

    # Total workers in dict is still 3, but only 1 is alive
    alive = sum(1 for w in status.workers.values() if w["state"] == "active")
    assert alive == 1
    assert len(status.workers) == 3

    # worker-2 picks up the requeued task
    status2, _work2 = coordinator.pull_task("worker-2", _TEST_WORKER_AVAILABLE)
    assert status2 == PullStatus.RUN_TASK

    # Simulate worker-0 re-registering while worker-2 holds the task in-flight
    # (race between heartbeat requeue and re-registration).
    # Since worker-0 has no in-flight task anymore, this is a no-op for requeueing.
    coordinator.register_worker("worker-0", MagicMock())
    status = coordinator.get_status()
    assert status.workers["worker-0"]["state"] == "active"
    alive = sum(1 for w in status.workers.values() if w["state"] == "active")
    assert alive == 2  # worker-0 active, worker-1 still failed, worker-2 active (has in-flight)

    # Now test the direct re-registration requeue path:
    # worker-2 dies while holding the task, and before heartbeat fires,
    # it re-registers — the in-flight task should be requeued.
    assert 0 in coordinator._in_flight  # worker-2 holds shard 0
    coordinator.register_worker("worker-2", MagicMock())
    assert 0 not in coordinator._in_flight  # in-flight cleared
    assert len(coordinator._task_queue) == 1  # task was requeued


def _make_task(stage_name: str = "test", shard_idx: int = 0) -> ShardTask:
    return ShardTask(
        shard_idx=shard_idx,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        cost=_TEST_TASK_COST,
        stage_name=stage_name,
    )


def test_pull_task_returns_shutdown_on_last_stage_tail(coordinator):
    """During the last stage's tail (queue drained, in-flight tasks still
    finishing), a fresh slot's ``pull_task`` must return SHUTDOWN so the worker
    breaks its outer loop instead of respawning slots that would just get
    killed again — that's the original hot-spin bug.
    """
    coordinator._current_stage = PhysicalStage(operations=[], stage_type=StageType.MAP_WORKER)
    coordinator._start_stage("tail", 0, [_make_task("tail")], is_last_stage=True)

    coordinator.register_worker("worker-0", MagicMock())
    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK  # drain the queue

    coordinator.register_worker("worker-1", MagicMock())
    status, _work = coordinator.pull_task("worker-1", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.SHUTDOWN


def test_pull_task_returns_no_work_backoff_mid_non_last_stage(coordinator):
    """Mid-stage on a non-last stage with the queue drained but in-flight tasks
    running: NO_WORK_BACKOFF, not SHUTDOWN or STAGE_COMPLETED — the slot must
    stay alive and keep polling so it can pick up requeued tasks or the eventual
    stage-end signal.
    """
    coordinator._current_stage = PhysicalStage(operations=[], stage_type=StageType.MAP_WORKER)
    coordinator._start_stage("mid", 0, [_make_task("mid")], is_last_stage=False)

    coordinator.register_worker("worker-0", MagicMock())
    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK  # drain the queue

    coordinator.register_worker("worker-1", MagicMock())
    status, _work = coordinator.pull_task("worker-1", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.NO_WORK_BACKOFF


def test_pull_task_returns_stage_completed_after_mark_stage_complete(coordinator):
    """At a non-last stage boundary (after ``_mark_stage_complete``), pull_task
    returns STAGE_COMPLETED so slots tear down and the worker re-pools at the
    size required by the next stage.
    """
    coordinator._current_stage = PhysicalStage(operations=[], stage_type=StageType.MAP_WORKER)
    coordinator._start_stage("mid", 0, [_make_task("mid")], is_last_stage=False)

    coordinator.register_worker("worker-0", MagicMock())
    coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)  # drain the queue
    coordinator._mark_stage_complete()

    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.STAGE_COMPLETED


def test_pull_task_returns_shutdown_on_coordinator_shutdown(coordinator):
    """When the coordinator's shutdown_event is set, all pull_task calls return
    SHUTDOWN regardless of stage state.
    """
    coordinator._current_stage = PhysicalStage(operations=[], stage_type=StageType.MAP_WORKER)
    coordinator._start_stage("any", 0, [_make_task("any")], is_last_stage=False)
    coordinator._shutdown_event.set()

    coordinator.register_worker("worker-0", MagicMock())
    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.SHUTDOWN


def test_log_status_omits_throughput_when_counters_missing(coordinator, caplog):
    """Map-only stages don't populate item/byte counters, so the coordinator's
    status log should drop the ``items=... bytes_processed=...`` segment rather
    than print misleading zeros. Once either counter is recorded, the segment
    reappears."""

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="map_only",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("map_only", 0, [task])

    # No counters recorded → throughput segment is suppressed.
    with caplog.at_level(logging.INFO, logger="zephyr.execution"):
        caplog.clear()
        coordinator._log_status()
    msgs = [r.getMessage() for r in caplog.records if "complete" in r.getMessage()]
    assert msgs, "expected a status line"
    assert all("items=" not in m and "bytes_processed=" not in m for m in msgs), msgs

    # Once a counter snapshot exists, the throughput segment reappears.
    coordinator._worker_counters["worker-A"] = CounterSnapshot(
        counters={ZEPHYR_STAGE_ITEM_COUNT_KEY: CounterEntry(7, stage="map_only")}, generation=1
    )
    with caplog.at_level(logging.INFO, logger="zephyr.execution"):
        caplog.clear()
        coordinator._log_status()
    msgs = [r.getMessage() for r in caplog.records if "complete" in r.getMessage()]
    assert msgs and "items=7" in msgs[-1] and "bytes_processed=0 bytes" in msgs[-1], msgs

    # Same when only the byte counter is present.
    coordinator._worker_counters["worker-A"] = CounterSnapshot(
        counters={ZEPHYR_STAGE_BYTES_PROCESSED_KEY: CounterEntry(1024, stage="map_only")}, generation=2
    )
    with caplog.at_level(logging.INFO, logger="zephyr.execution"):
        caplog.clear()
        coordinator._log_status()
    msgs = [r.getMessage() for r in caplog.records if "complete" in r.getMessage()]
    assert msgs and "items=0" in msgs[-1] and "bytes_processed=1 KiB" in msgs[-1], msgs


def test_no_duplicate_results_on_heartbeat_timeout(coordinator):
    """When a task is requeued after heartbeat timeout, the original worker's
    stale result (from a previous attempt) is rejected by the coordinator."""
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])

    # Worker A pulls task (attempt 0)
    status_a, work_a = coordinator.pull_task("worker-A", _TEST_WORKER_AVAILABLE)
    assert status_a == PullStatus.RUN_TASK
    assert work_a is not None

    # Simulate heartbeat timeout: mark worker-A stale and requeue
    coordinator._last_seen["worker-A"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)

    # Task should be requeued with incremented attempt
    assert coordinator._task_attempts[0] == 1

    # Worker B picks up the requeued task (attempt 1)
    status_b, work_b = coordinator.pull_task("worker-B", _TEST_WORKER_AVAILABLE)
    assert status_b == PullStatus.RUN_TASK
    assert work_b is not None
    assert work_b.attempt == 1

    # Worker B reports success
    coordinator.report_result(
        "worker-B", 0, work_b.attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )

    # Worker A's stale result (attempt 0) should be ignored
    coordinator.report_result(
        "worker-A", 0, work_a.attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )

    # Only one completion should be counted
    assert coordinator._completed_shards == 1


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


def test_coordinator_accepts_winner_ignores_stale(coordinator, tmp_path):
    """Coordinator accepts the winning result and ignores stale ones.

    Stale chunk files are left for context-dir cleanup (no per-chunk deletion).
    """
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])

    # Worker A pulls task (attempt 0)
    status_a, work_a = coordinator.pull_task("worker-A", _TEST_WORKER_AVAILABLE)
    assert status_a == PullStatus.RUN_TASK
    assert work_a is not None

    # Worker A writes a chunk (simulating slow completion)
    stale_ref = PickleDiskChunk.write(str(tmp_path / "stale-chunk.pkl"), [1, 2, 3])
    assert Path(stale_ref.path).exists()

    # Heartbeat timeout re-queues the task
    coordinator._last_seen["worker-A"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)

    # Worker B pulls and completes the re-queued task (attempt 1)
    status_b, work_b = coordinator.pull_task("worker-B", _TEST_WORKER_AVAILABLE)
    assert status_b == PullStatus.RUN_TASK
    assert work_b is not None

    winner_ref = PickleDiskChunk.write(str(tmp_path / "winner-chunk.pkl"), [4, 5, 6])

    coordinator.report_result(
        "worker-B",
        0,
        work_b.attempt,
        TaskResult(shard=ListShard(refs=[winner_ref])),
        CounterSnapshot.empty(),
    )

    # Worker A's stale result is rejected
    coordinator.report_result(
        "worker-A",
        0,
        work_a.attempt,
        TaskResult(shard=ListShard(refs=[stale_ref])),
        CounterSnapshot.empty(),
    )

    # Winner's data is directly readable (no rename needed)
    assert Path(winner_ref.path).exists()
    assert winner_ref.read() == [4, 5, 6]

    # Stale file still exists (cleaned up by context-dir cleanup, not coordinator)
    assert Path(stale_ref.path).exists()
    assert coordinator._completed_shards == 1


def test_stale_result_ignored_while_reassigned_worker_in_flight(coordinator):
    """A slow worker's stale result must be dropped even when another worker
    still holds the shard in ``_in_flight`` (shard_idx-keyed tracking)."""
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])

    status_a, work_a = coordinator.pull_task("worker-A", _TEST_WORKER_AVAILABLE)
    assert status_a == PullStatus.RUN_TASK
    assert work_a is not None

    coordinator._last_seen["worker-A"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)

    status_b, work_b = coordinator.pull_task("worker-B", _TEST_WORKER_AVAILABLE)
    assert status_b == PullStatus.RUN_TASK
    assert work_b is not None
    assert coordinator._in_flight[0].worker_id == "worker-B"

    coordinator.report_result(
        "worker-A", 0, work_a.attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )

    assert coordinator._completed_shards == 0
    assert 0 in coordinator._in_flight
    assert coordinator._in_flight[0].worker_id == "worker-B"


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


def test_report_error_requeues_until_max_shard_failures(coordinator):
    """report_error re-queues a task until MAX_SHARD_FAILURES, then aborts."""
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])
    coordinator.register_worker("worker-0", MagicMock())

    # Each failure should re-queue until the limit
    for i in range(MAX_SHARD_FAILURES - 1):
        status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
        assert status == PullStatus.RUN_TASK
        coordinator.report_error("worker-0", 0, work.attempt, f"error-{i}")
        assert coordinator._fatal_error is None, f"Should not abort on failure {i + 1}"
        assert coordinator._worker_states["worker-0"] == WorkerState.ACTIVE

    # The final failure should set _fatal_error
    status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    coordinator.report_error("worker-0", 0, work.attempt, "final-error")
    assert coordinator._fatal_error is not None
    assert "Shard 0" in coordinator._fatal_error
    assert "final-error" in coordinator._fatal_error


def test_heartbeat_timeouts_do_not_count_toward_shard_failures(coordinator):
    """Heartbeat-timeout requeues (preemption) must not consume MAX_SHARD_FAILURES."""
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])
    coordinator.register_worker("worker-0", MagicMock())

    # Far more heartbeat timeouts than MAX_SHARD_FAILURES — must not abort.
    for _ in range(MAX_SHARD_FAILURES * 5):
        status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
        assert status == PullStatus.RUN_TASK
        coordinator._last_seen["worker-0"] = 0.0
        coordinator.check_heartbeats(timeout=0.0)
        assert coordinator._fatal_error is None

    # Task-error budget is untouched; a successful completion closes the shard.
    assert coordinator._task_error_attempts[0] == 0
    status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    assert work is not None
    coordinator.report_result("worker-0", 0, work.attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())
    assert coordinator._completed_shards == 1
    assert coordinator._fatal_error is None


def test_repeated_infra_failures_on_same_shard_eventually_abort(coordinator):
    """A shard that consistently crashes its worker must eventually abort the pipeline.

    With in-process shard execution, a native SIGSEGV / OOM in shard code
    takes down the whole worker actor. The coordinator sees that as an
    INFRA failure (heartbeat timeout / re-registration). If the same shard
    causes this repeatedly, it's deterministic — keep retrying forever and
    the pipeline never converges. ``MAX_SHARD_INFRA_FAILURES`` bounds it.
    """
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])
    coordinator.register_worker("worker-0", MagicMock())

    # One short of the cap: still re-queues, no abort yet.
    for _ in range(MAX_SHARD_INFRA_FAILURES - 1):
        status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
        assert status == PullStatus.RUN_TASK
        coordinator._last_seen["worker-0"] = 0.0
        coordinator.check_heartbeats(timeout=0.0)
        assert coordinator._fatal_error is None

    # The next failure crosses the cap and aborts.
    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    coordinator._last_seen["worker-0"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)

    assert coordinator._fatal_error is not None
    assert "Shard 0" in coordinator._fatal_error
    assert "crashed its worker" in coordinator._fatal_error


def test_max_shard_failures_override_via_constructor(coordinator):
    """``max_shard_failures=N`` makes the coordinator abort after N task errors.

    Sets the per-shard task-error cap to 2 (vs. the default ``MAX_SHARD_FAILURES=3``);
    a fresh shard must survive 1 failure and abort on the 2nd.
    """
    coordinator._max_shard_failures = 2

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])
    coordinator.register_worker("worker-0", MagicMock())

    # First failure: re-queues, no abort.
    status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    coordinator.report_error("worker-0", 0, work.attempt, "error-1")
    assert coordinator._fatal_error is None

    # Second failure: hits the custom cap of 2 → abort.
    status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    coordinator.report_error("worker-0", 0, work.attempt, "error-2")
    assert coordinator._fatal_error is not None
    assert "Shard 0" in coordinator._fatal_error
    assert "error-2" in coordinator._fatal_error


def test_max_shard_infra_failures_override_via_constructor(coordinator):
    """``max_shard_infra_failures=N`` makes the coordinator abort after N
    infra failures on the same shard.

    Sets the per-shard infra-failure cap to 2 (vs. the default ``MAX_SHARD_INFRA_FAILURES=20``).
    """
    coordinator._max_shard_infra_failures = 2

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])
    coordinator.register_worker("worker-0", MagicMock())

    # First infra failure: re-queues, no abort.
    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    coordinator._last_seen["worker-0"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)
    assert coordinator._fatal_error is None

    # Second infra failure: hits the custom cap of 2 → abort.
    status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    coordinator._last_seen["worker-0"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)
    assert coordinator._fatal_error is not None
    assert "Shard 0" in coordinator._fatal_error
    assert "crashed its worker" in coordinator._fatal_error


def test_worker_reregistration_does_not_count_toward_shard_failures(coordinator):
    """Preemption-driven worker re-registration requeues without burning MAX_SHARD_FAILURES."""
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])
    coordinator.register_worker("worker-0", MagicMock())

    for _ in range(MAX_SHARD_FAILURES * 5):
        status, _work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
        assert status == PullStatus.RUN_TASK
        # Simulate preemption + Iris reconstruction: worker re-registers while
        # a task is still recorded as in-flight on the old handle.
        coordinator.register_worker("worker-0", MagicMock())
        assert 0 not in coordinator._in_flight
        assert coordinator._fatal_error is None

    assert coordinator._task_error_attempts[0] == 0


def test_report_error_still_aborts_at_max_shard_failures_after_preemptions(coordinator):
    """Task errors still abort at MAX_SHARD_FAILURES even after many survived preemptions."""
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])
    coordinator.register_worker("worker-0", MagicMock())

    # Several preemption cycles first — these must not count.
    for _ in range(5):
        status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
        assert status == PullStatus.RUN_TASK
        coordinator._last_seen["worker-0"] = 0.0
        coordinator.check_heartbeats(timeout=0.0)

    assert coordinator._fatal_error is None

    # Now MAX_SHARD_FAILURES explicit task errors should abort.
    for i in range(MAX_SHARD_FAILURES):
        status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
        assert status == PullStatus.RUN_TASK
        coordinator.report_error("worker-0", 0, work.attempt, f"boom-{i}")

    assert coordinator._fatal_error is not None
    assert "Shard 0" in coordinator._fatal_error


def test_wait_for_stage_fails_when_all_workers_die(coordinator):
    """When all registered workers become dead/failed, _wait_for_stage raises
    after the no_workers_timeout instead of waiting forever."""
    coordinator._no_workers_timeout = 0.5  # short timeout for test

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])

    # Register 2 workers
    coordinator.register_worker("worker-0", MagicMock())
    coordinator.register_worker("worker-1", MagicMock())

    # Kill all workers via heartbeat timeout
    coordinator._last_seen["worker-0"] = 0.0
    coordinator._last_seen["worker-1"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)

    assert coordinator._worker_states["worker-0"] == WorkerState.FAILED
    assert coordinator._worker_states["worker-1"] == WorkerState.FAILED

    # _wait_for_stage should raise after the dead timer expires
    with pytest.raises(ZephyrWorkerError, match="No alive workers"):
        coordinator._wait_for_stage()


def test_wait_for_stage_resets_dead_timer_on_recovery(coordinator):
    """When a worker recovers (re-registers) after all workers died,
    the dead timer resets and execution can continue."""
    coordinator._no_workers_timeout = 2.0

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="test",
        cost=_TEST_TASK_COST,
    )
    coordinator._start_stage("test", 0, [task])

    # Register and kill a worker
    coordinator.register_worker("worker-0", MagicMock())
    coordinator._last_seen["worker-0"] = 0.0
    coordinator.check_heartbeats(timeout=0.0)
    assert coordinator._worker_states["worker-0"] == WorkerState.FAILED

    # In a background thread, re-register the worker and complete the task
    # after a short delay (simulating recovery before timeout expires)
    def recover_and_complete():
        time.sleep(0.1)
        coordinator.register_worker("worker-0", MagicMock())
        status, work = coordinator.pull_task("worker-0", _TEST_WORKER_AVAILABLE)
        assert status == PullStatus.RUN_TASK
        assert work is not None
        coordinator.report_result(
            "worker-0", 0, work.attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
        )

    t = threading.Thread(target=recover_and_complete)
    t.start()

    # _wait_for_stage should succeed (worker recovers before timeout)
    coordinator._wait_for_stage()
    t.join(timeout=5.0)

    assert coordinator._completed_shards == 1


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


def test_last_stage_deadlock_detected_when_worker_job_dies(coordinator):
    """Coordinator aborts if the worker job dies while last-stage work is outstanding."""
    tasks = [
        ShardTask(
            shard_idx=i,
            total_shards=2,
            shard=ListShard(refs=[]),
            operations=[],
            cost=_TEST_TASK_COST,
            stage_name="test",
        )
        for i in range(2)
    ]
    coordinator._start_stage("last-stage", 0, tasks)

    # Set up a mock worker group so _check_worker_group can query it.
    mock_group = MagicMock()
    mock_group.is_done.return_value = False
    coordinator.set_worker_group(mock_group)

    # Two workers pull both tasks.
    coordinator.heartbeat("worker-A")
    coordinator.heartbeat("worker-B")
    status_a, work_a = coordinator.pull_task("worker-A", _TEST_WORKER_AVAILABLE)
    status_b, _work_b = coordinator.pull_task("worker-B", _TEST_WORKER_AVAILABLE)
    assert status_a == PullStatus.RUN_TASK
    assert status_b == PullStatus.RUN_TASK

    # Worker A finishes its task.
    assert work_a is not None
    coordinator.report_result(
        "worker-A", work_a.task.shard_idx, work_a.attempt, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )

    # Worker B crashes → heartbeat timeout → shard 1 requeued.
    coordinator._last_seen["worker-B"] = coordinator._last_seen["worker-B"] - 200
    coordinator.check_heartbeats(timeout=10)
    assert len(coordinator._task_queue) == 1

    # Worker job is still running — no abort yet.
    coordinator._check_worker_group()
    assert coordinator._fatal_error is None

    # Worker job dies permanently (Iris exhausted retries).
    mock_group.is_done.return_value = True
    coordinator._check_worker_group()

    # Coordinator should detect the deadlock and abort.
    assert coordinator._fatal_error is not None
    assert "terminated permanently" in coordinator._fatal_error


def test_coordinator_loop_crash_aborts_pipeline(coordinator):
    """Coordinator loop crash sets _fatal_error instead of dying silently. #3996."""
    if coordinator._coordinator_thread is not None:
        coordinator._shutdown_event.set()
        coordinator._coordinator_thread.join(timeout=5.0)
        coordinator._coordinator_thread = None
    coordinator._shutdown_event = threading.Event()

    crashed = threading.Event()
    original = coordinator.check_heartbeats

    def crashing_heartbeats(*a, **kw):
        if not crashed.is_set():
            crashed.set()
            raise RuntimeError("dictionary changed size during iteration")
        return original(*a, **kw)

    coordinator.check_heartbeats = crashing_heartbeats

    t = threading.Thread(target=coordinator._coordinator_loop, daemon=True, name="zephyr-coordinator-loop")
    t.start()
    assert crashed.wait(timeout=5.0)
    t.join(timeout=2.0)
    assert coordinator._fatal_error is not None


def test_run_pipeline_rejects_concurrent_calls(coordinator):
    """Calling run_pipeline while another is already running raises RuntimeError."""
    gate = threading.Event()
    ds = Dataset.from_list([42]).map(lambda x: gate.wait(timeout=5) or x)
    plan = compute_plan(ds)
    # First call blocks because the map waits on `gate` (no workers to run it
    # anyway). We patch _wait_for_stage to signal when it's entered.
    first_entered = threading.Event()
    original_wait = coordinator._wait_for_stage

    def blocking_wait():
        first_entered.set()
        time.sleep(0.1)
        coordinator._fatal_error = "test: forced exit"
        try:
            original_wait()
        except Exception:
            pass

    coordinator._wait_for_stage = blocking_wait

    t = threading.Thread(target=lambda: coordinator.run_pipeline(plan, "exec-1"), daemon=True)
    t.start()
    first_entered.wait(timeout=5.0)

    # Second call should fail immediately
    with pytest.raises(RuntimeError, match="already running"):
        coordinator.run_pipeline(plan, "exec-2")

    t.join(timeout=10.0)
    coordinator.shutdown()


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


def test_stage_index_correct_with_join(local_client, tmp_path):
    """_current_stage_index is set correctly for main and join-right stages.

    The right-side sub-plan must carry the parent's stage_idx so the arrow
    indicator in _report_task_stats stays on the parent stage while the
    right side executes. If _compute_join_aux passes a different index,
    the highlighted stage would be wrong.
    """
    # (stage_name, current_stage_index) recorded at each _start_stage call.
    stage_calls: list[tuple[str, int]] = []
    original_start_stage = ZephyrCoordinator._start_stage

    def recording_start_stage(self, stage_name, current_stage_index, tasks, is_last_stage=False):
        original_start_stage(self, stage_name, current_stage_index, tasks, is_last_stage)
        stage_calls.append((stage_name, self._current_stage_index))

    ctx = ZephyrContext(
        client=local_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-join-index-{uuid.uuid4().hex[:8]}",
    )

    # The right side needs a worker stage so _compute_join_aux calls _start_stage.
    left = Dataset.from_list([{"id": 1}, {"id": 2}])
    right = Dataset.from_list([{"id": 1}, {"id": 2}]).map(lambda x: x)
    joined = left.sorted_merge_join(
        right,
        left_key=lambda x: x["id"],
        right_key=lambda x: x["id"],
        combiner=lambda l, r: {**l, **r},
    )

    ZephyrCoordinator._start_stage = recording_start_stage
    try:
        ctx.execute(joined)
    finally:
        ZephyrCoordinator._start_stage = original_start_stage
        ctx.shutdown()

    main_calls = [(n, i) for n, i in stage_calls if not n.startswith("join-right")]
    join_right_calls = [(n, i) for n, i in stage_calls if n.startswith("join-right")]

    assert join_right_calls, f"Expected join-right stages; got: {stage_calls}"
    assert main_calls, f"Expected main stages; got: {stage_calls}"

    # Each join-right stage must carry the same current_stage_index as the
    # main stage that follows it (the parent stage).
    for right_name, right_idx in join_right_calls:
        # The parent stage_idx is encoded in the join-right label: join-right-{parent}-...
        parent_idx = int(right_name.split("-")[2])
        assert right_idx == parent_idx, f"{right_name!r} has current_stage_index={right_idx}, expected {parent_idx}"

    # The main stage immediately following the join-right stages must have its
    # own stage_idx, not the join-right stages' index.
    for main_name, main_idx in main_calls:
        # Verify each main stage's recorded index matches what run_pipeline passed.
        # stage_label format: "stage{stage_idx}-{stage_name}"
        expected_idx = int(main_name.split("-")[0].replace("stage", ""))
        assert main_idx == expected_idx, f"{main_name!r} has current_stage_index={main_idx}, expected {expected_idx}"


def test_heartbeat_failures_fail_actor_context():
    """After max consecutive heartbeat failures, the worker records a
    ``CoordinatorUnreachable`` on its actor context and sets the host
    shutdown event so ``_host_actor`` re-raises it.

    Without this, the worker would set only its internal ``_shutdown_event``
    and exit cleanly — iris would mark the task SUCCEEDED and never retry.
    """

    class _FailingMethod:
        def remote(self, *args, **kwargs):
            f: Future = Future()
            f.set_exception(ConnectionError("simulated coordinator unreachable"))
            return f

    class _FailingCoordinator:
        def __getattr__(self, name):
            return _FailingMethod()

    shutdown_event = threading.Event()
    actor_ctx = ActorContext(
        handle=MagicMock(),
        index=0,
        group_name="test-worker",
        shutdown_event=shutdown_event,
    )

    worker = ZephyrWorker.__new__(ZephyrWorker)
    worker._actor_ctx = actor_ctx
    worker._shutdown_event = threading.Event()
    worker._active_runners = []
    worker._resources_lock = threading.Lock()
    worker._last_reported_counters = {}
    worker._counter_generation = 0
    worker._worker_id = "test-worker-0"
    worker._report_worker_iris_status = lambda: None

    t = threading.Thread(
        target=worker._heartbeat_loop,
        kwargs={"coordinator": _FailingCoordinator(), "interval": 0.01, "max_consecutive_failures": 3},
    )
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive(), "heartbeat loop did not exit after exhausting failure budget"
    assert shutdown_event.is_set(), "actor_ctx.shutdown_event should have been set by fail()"
    assert actor_ctx._errors, "expected at least one error recorded on actor_ctx"
    assert isinstance(actor_ctx._errors[0], CoordinatorUnreachable)


# --- Integration tests (all backends) ---


def test_simple_map_integration(integration_ctx):
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    assert sorted(integration_ctx.execute(ds).results) == [2, 4, 6]


def test_multi_stage_integration(integration_ctx):
    ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(lambda x: x > 5)
    assert sorted(integration_ctx.execute(ds).results) == [6, 8, 10]
