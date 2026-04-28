# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephyr.subprocess_worker."""

import logging
import threading

import cloudpickle

import zephyr.subprocess_worker as sw
from zephyr.execution import (
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    ListShard,
    ShardTask,
)
from zephyr.shuffle import MemChunk


def test_execute_shard_sets_stage_scoped_output_counters(tmp_path):
    """execute_shard emits output counters under the task's stage name."""
    chunk_prefix = str(tmp_path / "chunks")
    execution_id = "test-exec"
    stage_name = "test"
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[MemChunk(list(range(10)))]),
        operations=[],
        stage_name=stage_name,
    )

    task_file = str(tmp_path / "task.pkl")
    result_file = str(tmp_path / "result.pkl")
    with open(task_file, "wb") as f:
        cloudpickle.dump((task, chunk_prefix, execution_id), f)

    sw.execute_shard(task_file, result_file)

    with open(result_file, "rb") as f:
        _result_or_error, counters_out = cloudpickle.load(f)

    assert counters_out[ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)] == 10
    assert counters_out[ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)] > 0


def _run_status_logger_once(ctx: sw._SubprocessWorkerContext, stage_name: str) -> threading.Thread:
    """Spin up the status-logger thread, let one tick fire, then stop it."""
    stop_event = threading.Event()

    def _target():
        sw._periodic_status_logger(
            stop_event=stop_event,
            ctx=ctx,
            stage_name=stage_name,
            execution_id="exec-id",
            shard_idx=0,
            total_shards=1,
            monotonic_start=0.0,
            interval=0.01,
        )

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    # Give the loop time to wake at least once.
    thread.join(timeout=0.1)
    stop_event.set()
    thread.join(timeout=1.0)
    return thread


def test_periodic_status_logger_skips_when_no_counters(caplog):
    """Map-only stages never populate item/byte counters; we shouldn't log zeros."""
    stage_name = "map-only"
    ctx = sw._SubprocessWorkerContext(chunk_prefix="x", execution_id="exec-id")

    with caplog.at_level(logging.INFO, logger=sw.__name__):
        _run_status_logger_once(ctx, stage_name)

    status_records = [r for r in caplog.records if "shard 0/1" in r.getMessage()]
    assert status_records == []


def test_periodic_status_logger_logs_when_counters_present(caplog):
    """Once a counter has been recorded, the status line is emitted."""
    stage_name = "with-stats"
    ctx = sw._SubprocessWorkerContext(chunk_prefix="x", execution_id="exec-id")
    ctx._counters[ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)] = 5

    with caplog.at_level(logging.INFO, logger=sw.__name__):
        _run_status_logger_once(ctx, stage_name)

    status_records = [r for r in caplog.records if "shard 0/1" in r.getMessage()]
    assert status_records, "expected at least one status log line"
    assert "items=5" in status_records[0].getMessage()
