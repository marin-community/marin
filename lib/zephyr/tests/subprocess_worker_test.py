# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephyr.subprocess_worker."""

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
