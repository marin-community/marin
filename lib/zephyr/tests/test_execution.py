# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the actor-based execution engine (ZephyrContext)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fray.v2 import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, shard_ctx


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
    """Workers can access shared data via shard_ctx()."""

    def use_shared(x):
        multiplier = shard_ctx().get_shared("multiplier")
        return x * multiplier

    zctx = ZephyrContext(
        client=fray_client, num_workers=1, resources=ResourceConfig(cpu=1, ram="512m"), name="test-execution"
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
    """ZephyrContext works as context manager."""
    with ZephyrContext(
        client=fray_client, num_workers=1, resources=ResourceConfig(cpu=1, ram="512m"), name="test-execution"
    ) as zctx:
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
        num_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name="test-execution",
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


def test_chunk_streaming_low_memory(tmp_path):
    """Shard loads chunks one at a time from disk via iter_chunks.

    Verifies iter_chunks yields data lazily and flat iteration works.
    """
    from zephyr.execution import DiskChunk, Shard

    # Write 3 chunks to disk
    refs = []
    for i in range(3):
        path = str(tmp_path / f"chunk-{i}.pkl")
        refs.append(DiskChunk.write(path, [i * 10 + j for j in range(5)]))

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


def test_workers_persist_across_executes(fray_client, tmp_path):
    """Workers persist across multiple execute() calls within a context."""
    chunk_prefix = str(tmp_path / "chunks")

    with ZephyrContext(
        client=fray_client,
        num_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name="test-execution",
    ) as zctx:
        ds = Dataset.from_list([1, 2, 3]).map(lambda x: x + 1)
        results = list(zctx.execute(ds))
        assert sorted(results) == [2, 3, 4]

        # After execute(): coordinator still exists (workers persist)
        assert zctx._coordinator is not None
        assert zctx._coordinator_group is not None

        # Can execute again (reuses same workers)
        ds2 = Dataset.from_list([10, 20]).map(lambda x: x * 2)
        results2 = list(zctx.execute(ds2))
        assert sorted(results2) == [20, 40]

    # After context exit: all resources are cleaned up
    assert zctx._coordinator is None
    assert zctx._coordinator_group is None


def test_fatal_errors_fail_fast(fray_client, tmp_path):
    """Application errors (e.g. ValueError) cause immediate failure, no retries."""
    from zephyr.execution import ZephyrWorkerError

    chunk_prefix = str(tmp_path / "chunks")

    def exploding_map(x):
        raise ValueError(f"bad value: {x}")

    with ZephyrContext(
        client=fray_client,
        num_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name="test-execution",
    ) as zctx:
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
        num_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name="test-execution",
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
