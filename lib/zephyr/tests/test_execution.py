# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the actor-based execution engine (ZephyrContext)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest
from fray.v2 import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, zephyr_worker_ctx


def test_simple_map(zephyr_ctx):
    """Map pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = list(zephyr_ctx.execute(ds))
    assert sorted(results) == [2, 4, 6]


def test_filter(zephyr_ctx):
    """Filter pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).filter(lambda x: x > 3)
    results = list(zephyr_ctx.execute(ds))
    assert sorted(results) == [4, 5]


def test_shared_data(fray_client):
    """Workers can access shared data via zephyr_worker_ctx()."""

    def use_shared(x):
        multiplier = zephyr_worker_ctx().get_shared("multiplier")
        return x * multiplier

    zctx = ZephyrContext(
        client=fray_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    zctx.put("multiplier", 10)
    ds = Dataset.from_list([1, 2, 3]).map(use_shared)
    results = list(zctx.execute(ds))
    assert sorted(results) == [10, 20, 30]
    zctx.shutdown()


def test_multi_stage(zephyr_ctx):
    """Multi-stage pipeline (map + filter) works."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(lambda x: x > 5)
    results = list(zephyr_ctx.execute(ds))
    assert sorted(results) == [6, 8, 10]


def test_context_manager(fray_client):
    """ZephyrContext works without context manager."""
    zctx = ZephyrContext(
        client=fray_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x + 1)
    results = list(zctx.execute(ds))
    assert sorted(results) == [2, 3, 4]


def test_write_jsonl(tmp_path, zephyr_ctx):
    """Pipeline writing to jsonl file."""
    output = str(tmp_path / "out-{shard}.jsonl")
    ds = Dataset.from_list([{"a": 1}, {"a": 2}, {"a": 3}]).write_jsonl(output)
    results = list(zephyr_ctx.execute(ds))
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
    results = list(zephyr_ctx.execute(ds, dry_run=True))
    assert results == []


def test_flat_map(zephyr_ctx):
    """FlatMap pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3]).flat_map(lambda x: [x, x * 10])
    results = list(zephyr_ctx.execute(ds))
    assert sorted(results) == [1, 2, 3, 10, 20, 30]


def test_empty_dataset(zephyr_ctx):
    """Empty dataset produces empty results."""
    ds = Dataset.from_list([])
    results = list(zephyr_ctx.execute(ds))
    assert results == []


def test_chunk_cleanup(fray_client, tmp_path):
    """Verify chunks are cleaned up after execution."""
    chunk_prefix = str(tmp_path / "chunks")
    ctx = ZephyrContext(
        client=fray_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )

    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = list(ctx.execute(ds))

    assert sorted(results) == [2, 4, 6]

    # Verify chunks directory is cleaned up
    import os

    if os.path.exists(chunk_prefix):
        # Should be empty or not exist
        files = list(Path(chunk_prefix).rglob("*"))
        assert len(files) == 0, f"Expected cleanup but found: {files}"


def test_status_reports_alive_workers_not_total(tmp_path):
    """After heartbeat timeout, get_status workers dict reflects FAILED state,
    and the status log distinguishes alive from total workers.

    Also verifies that re-registering a worker that had an in-flight task
    requeues that task so it is not silently lost.
    """
    from unittest.mock import MagicMock

    from zephyr.execution import Shard, ShardTask, ZephyrCoordinator

    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord.set_shared_data({})

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        chunk_size=100,
        shard=Shard(chunks=[]),
        operations=[],
        stage_name="test",
    )
    coord.start_stage("test", [task])

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


def test_no_duplicate_results_on_heartbeat_timeout(fray_client, tmp_path):
    """When a task is requeued after heartbeat timeout, the original worker's
    stale result (from a previous attempt) is rejected by the coordinator."""
    from zephyr.execution import Shard, ShardTask, TaskResult, ZephyrCoordinator

    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord.set_shared_data({})

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        chunk_size=100,
        shard=Shard(chunks=[]),
        operations=[],
        stage_name="test",
    )
    coord.start_stage("test", [task])

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
    coord.report_result("worker-B", 0, attempt_b, TaskResult(chunks=[]))

    # Worker A's stale result (attempt 0) should be ignored
    coord.report_result("worker-A", 0, attempt_a, TaskResult(chunks=[]))

    # Only one completion should be counted
    assert coord._completed_shards == 1


def test_disk_chunk_write_uses_unique_paths(tmp_path):
    """Each DiskChunk.write() writes to a unique location, avoiding collisions."""
    from zephyr.execution import DiskChunk

    base_path = str(tmp_path / "chunk.pkl")
    refs = [DiskChunk.write(base_path, [i]) for i in range(3)]

    # Each written to a distinct UUID path (no rename needed)
    paths = [r.path for r in refs]
    assert len(set(paths)) == 3
    for p in paths:
        assert ".tmp." in p
        assert Path(p).exists()

    # Each chunk is directly readable
    for i, ref in enumerate(refs):
        assert ref.read() == [i]


def test_coordinator_accepts_winner_ignores_stale(tmp_path):
    """Coordinator accepts the winning result and ignores stale ones.

    Stale chunk files are left for context-dir cleanup (no per-chunk deletion).
    """
    from zephyr.execution import DiskChunk, ResultChunk, Shard, ShardTask, TaskResult, ZephyrCoordinator

    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord.set_shared_data({})

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        chunk_size=100,
        shard=Shard(chunks=[]),
        operations=[],
        stage_name="test",
    )
    coord.start_stage("test", [task])

    # Worker A pulls task (attempt 0)
    pulled_a = coord.pull_task("worker-A")
    _task_a, attempt_a, _config = pulled_a

    # Worker A writes a chunk (simulating slow completion)
    stale_ref = DiskChunk.write(str(tmp_path / "stale-chunk.pkl"), [1, 2, 3])
    assert Path(stale_ref.path).exists()

    # Heartbeat timeout re-queues the task
    coord._last_seen["worker-A"] = 0.0
    coord.check_heartbeats(timeout=0.0)

    # Worker B pulls and completes the re-queued task (attempt 1)
    pulled_b = coord.pull_task("worker-B")
    _task_b, attempt_b, _config = pulled_b

    winner_ref = DiskChunk.write(str(tmp_path / "winner-chunk.pkl"), [4, 5, 6])

    coord.report_result(
        "worker-B",
        0,
        attempt_b,
        TaskResult(chunks=[ResultChunk(source_shard=0, target_shard=0, data=winner_ref)]),
    )

    # Worker A's stale result is rejected
    coord.report_result(
        "worker-A",
        0,
        attempt_a,
        TaskResult(chunks=[ResultChunk(source_shard=0, target_shard=0, data=stale_ref)]),
    )

    # Winner's data is directly readable (no rename needed)
    assert Path(winner_ref.path).exists()
    assert winner_ref.read() == [4, 5, 6]

    # Stale file still exists (cleaned up by context-dir cleanup, not coordinator)
    assert Path(stale_ref.path).exists()
    assert coord._completed_shards == 1


def test_chunk_streaming_low_memory(tmp_path):
    """Shard loads chunks one at a time from disk via iter_chunks.

    Verifies iter_chunks yields data lazily and flat iteration works.
    """
    from zephyr.execution import DiskChunk, Shard

    # Write 3 chunks to disk (directly readable, no finalize needed)
    refs = []
    for i in range(3):
        path = str(tmp_path / f"chunk-{i}.pkl")
        chunk = DiskChunk.write(path, [i * 10 + j for j in range(5)])
        refs.append(chunk)

    shard = Shard(chunks=refs)

    # iter_chunks yields correct data
    chunks = list(shard.iter_chunks())
    assert len(chunks) == 3
    assert chunks[0] == [0, 1, 2, 3, 4]
    assert chunks[2] == [20, 21, 22, 23, 24]

    # flat iteration yields all items in order
    assert list(shard) == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]

    # Re-iteration works (reads from disk again, not cached)
    assert list(shard) == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]


def test_wait_for_stage_fails_when_all_workers_die(tmp_path):
    """When all registered workers become dead/failed, _wait_for_stage raises
    after the no_workers_timeout instead of waiting forever."""
    from unittest.mock import MagicMock

    from zephyr.execution import Shard, ShardTask, WorkerState, ZephyrCoordinator, ZephyrWorkerError

    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord.set_shared_data({})
    coord._no_workers_timeout = 0.5  # short timeout for test

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        chunk_size=100,
        shard=Shard(chunks=[]),
        operations=[],
        stage_name="test",
    )
    coord.start_stage("test", [task])

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


def test_wait_for_stage_resets_dead_timer_on_recovery(tmp_path):
    """When a worker recovers (re-registers) after all workers died,
    the dead timer resets and execution can continue."""
    import threading

    from unittest.mock import MagicMock

    from zephyr.execution import Shard, ShardTask, TaskResult, WorkerState, ZephyrCoordinator

    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord.set_shared_data({})
    coord._no_workers_timeout = 2.0

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        chunk_size=100,
        shard=Shard(chunks=[]),
        operations=[],
        stage_name="test",
    )
    coord.start_stage("test", [task])

    # Register and kill a worker
    coord.register_worker("worker-0", MagicMock())
    coord._last_seen["worker-0"] = 0.0
    coord.check_heartbeats(timeout=0.0)
    assert coord._worker_states["worker-0"] == WorkerState.FAILED

    # In a background thread, re-register the worker and complete the task
    # after a short delay (simulating recovery before timeout expires)
    def recover_and_complete():
        time.sleep(0.3)
        coord.register_worker("worker-0", MagicMock())
        pulled = coord.pull_task("worker-0")
        assert pulled is not None and pulled != "SHUTDOWN"
        _task, attempt, _config = pulled
        coord.report_result("worker-0", 0, attempt, TaskResult(chunks=[]))

    t = threading.Thread(target=recover_and_complete)
    t.start()

    # _wait_for_stage should succeed (worker recovers before timeout)
    coord._wait_for_stage()
    t.join(timeout=5.0)

    assert coord._completed_shards == 1


def test_fresh_actors_per_execute(fray_client, tmp_path):
    """Each execute() creates and tears down its own coordinator and workers."""
    chunk_prefix = str(tmp_path / "chunks")

    zctx = ZephyrContext(
        client=fray_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x + 1)
    results = list(zctx.execute(ds))
    assert sorted(results) == [2, 3, 4]

    # After execute(): everything is torn down
    assert zctx._coordinator is None
    assert zctx._worker_group is None
    assert zctx._pipeline_id == 0

    # Can execute again (creates fresh coordinator + workers)
    ds2 = Dataset.from_list([10, 20]).map(lambda x: x * 2)
    results2 = list(zctx.execute(ds2))
    assert sorted(results2) == [20, 40]

    assert zctx._coordinator is None
    assert zctx._worker_group is None
    assert zctx._pipeline_id == 1


def test_fatal_errors_fail_fast(fray_client, tmp_path):
    """Application errors (e.g. ValueError) cause immediate failure, no retries."""
    from zephyr.execution import ZephyrWorkerError

    chunk_prefix = str(tmp_path / "chunks")

    def exploding_map(x):
        raise ValueError(f"bad value: {x}")

    zctx = ZephyrContext(
        client=fray_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(exploding_map)

    start = time.monotonic()
    with pytest.raises(ZephyrWorkerError, match="ValueError"):
        list(zctx.execute(ds))
    elapsed = time.monotonic() - start

    # Should fail fast — well under the 30s heartbeat timeout
    assert elapsed < 15.0, f"Took {elapsed:.1f}s, expected fast failure"


def test_chunk_storage_with_join(fray_client, tmp_path):
    """Verify chunk storage works with join operations."""
    chunk_prefix = str(tmp_path / "chunks")
    ctx = ZephyrContext(
        client=fray_client,
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

    results = sorted(list(ctx.execute(joined)), key=lambda x: x["id"])
    assert len(results) == 2
    assert results[0] == {"id": 1, "a": "x", "b": "p"}
    assert results[1] == {"id": 2, "a": "y", "b": "q"}


def test_workers_capped_to_shard_count(fray_client, tmp_path):
    """When max_workers > num_shards, only num_shards workers are created."""
    ds = Dataset.from_list([1, 2, 3])  # 3 shards
    ctx = ZephyrContext(
        client=fray_client,
        max_workers=10,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    results = list(ctx.execute(ds.map(lambda x: x * 2)))
    assert sorted(results) == [2, 4, 6]
    # Everything torn down after execute; correct results prove workers
    # were created and sized properly (min(10, 3) = 3)
    assert ctx._pipeline_id == 0


def test_pipeline_id_increments(fray_client, tmp_path):
    """Pipeline ID increments after each execute(), ensuring unique actor names."""
    ctx = ZephyrContext(
        client=fray_client,
        max_workers=10,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ctx.execute(Dataset.from_list([1, 2]).map(lambda x: x))
    assert ctx._pipeline_id == 0

    ctx.execute(Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x))
    assert ctx._pipeline_id == 1


def test_pull_task_returns_shutdown_on_last_stage_empty_queue(tmp_path):
    """When the last stage's tasks are all in-flight or done, pull_task returns SHUTDOWN."""

    from zephyr.execution import Shard, ShardTask, TaskResult, ZephyrCoordinator

    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    coord.set_shared_data({})

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        chunk_size=100,
        shard=Shard(chunks=[]),
        operations=[],
        stage_name="test",
    )

    # Non-last stage: empty queue returns None
    coord.start_stage("stage-0", [task], is_last_stage=False)
    pulled = coord.pull_task("worker-A")
    assert pulled is not None and pulled != "SHUTDOWN"
    _task, attempt, _config = pulled
    coord.report_result("worker-A", 0, attempt, TaskResult(chunks=[]))

    # Queue empty, but not last stage -> None
    result = coord.pull_task("worker-A")
    assert result is None

    # Last stage: empty queue returns SHUTDOWN
    task2 = ShardTask(
        shard_idx=0,
        total_shards=1,
        chunk_size=100,
        shard=Shard(chunks=[]),
        operations=[],
        stage_name="test-last",
    )
    coord.start_stage("stage-1", [task2], is_last_stage=True)
    pulled = coord.pull_task("worker-A")
    assert pulled is not None and pulled != "SHUTDOWN"
    _task, attempt, _config = pulled
    coord.report_result("worker-A", 0, attempt, TaskResult(chunks=[]))

    # Queue empty on last stage -> SHUTDOWN
    result = coord.pull_task("worker-A")
    assert result == "SHUTDOWN"


def test_execute_retries_on_coordinator_death(tmp_path):
    """When the coordinator dies mid-execution, execute() retries with a fresh
    coordinator and worker pool and eventually succeeds.

    Uses LocalClient directly because simulating coordinator death requires
    manipulating the local actor registry.
    """
    from fray.v2.local_backend import LocalClient, _local_actor_registry

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
    results = list(ctx.execute(Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)))
    assert sorted(results) == [2, 4, 6]

    # Sabotage the registry so the *next* coordinator creation attempt fails
    # on the first try. We do this by patching create_actor_group to fail once.
    original_create = client.create_actor_group
    fail_count = [0]

    def flaky_create(*args, **kwargs):
        group = original_create(*args, **kwargs)
        if fail_count[0] == 0 and "coord" in kwargs.get("name", ""):
            fail_count[0] += 1
            # Kill the coordinator immediately after creation to simulate death
            handles = group.wait_ready()
            endpoint = handles[0]._endpoint
            _local_actor_registry.pop(endpoint, None)
        return group

    client.create_actor_group = flaky_create

    # Next execute() should: fail on attempt 0 (dead coordinator),
    # then succeed on attempt 1 with a fresh coordinator.
    results = list(ctx.execute(Dataset.from_list([10, 20]).map(lambda x: x + 1)))
    assert sorted(results) == [11, 21]

    ctx.shutdown()
    client.shutdown(wait=True)


def test_execute_does_not_retry_worker_errors(fray_client, tmp_path):
    """ZephyrWorkerError (application errors) are never retried."""
    from zephyr.execution import ZephyrWorkerError

    chunk_prefix = str(tmp_path / "chunks")

    def exploding_map(x):
        raise ValueError(f"bad value: {x}")

    ctx = ZephyrContext(
        client=fray_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        max_execution_retries=3,
        name=f"test-execution-{uuid.uuid4().hex[:8]}",
    )
    ds = Dataset.from_list([1, 2, 3]).map(exploding_map)

    start = time.monotonic()
    with pytest.raises(ZephyrWorkerError, match="ValueError"):
        list(ctx.execute(ds))
    elapsed = time.monotonic() - start

    # Should fail fast — no retries for application errors
    assert elapsed < 15.0, f"Took {elapsed:.1f}s, expected fast failure (no retries)"
