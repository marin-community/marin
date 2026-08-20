# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator drivers for Zephyr tests."""

from zephyr.coordinator import ZephyrCoordinator, _PipelineExecution
from zephyr.stage_io import ShardTask, ZephyrTaskResources

TEST_WORKER_RAM = 1 << 30
TEST_TASK_COST = ZephyrTaskResources(cpu=1.0, memory=TEST_WORKER_RAM)
TEST_WORKER_AVAILABLE = ZephyrTaskResources(cpu=1.0, memory=TEST_WORKER_RAM)
TEST_EXECUTION_ID = "test-exec"


def make_test_coordinator(tmp_path, **kwargs) -> ZephyrCoordinator:
    return ZephyrCoordinator(str(tmp_path / "chunks"), TEST_WORKER_AVAILABLE, **kwargs)


def start_test_stage(
    coordinator: ZephyrCoordinator,
    tasks: list[ShardTask],
    *,
    stage_name: str = "test",
    is_last_stage: bool = False,
    execution_id: str = TEST_EXECUTION_ID,
) -> _PipelineExecution:
    """Register a coordinator execution and load its first stage."""
    run = _PipelineExecution(execution_id=execution_id, map_cost=TEST_TASK_COST, reduce_cost=TEST_TASK_COST)
    coordinator._executions[execution_id] = run
    coordinator._start_stage(run, stage_name, 0, tasks, is_last_stage=is_last_stage)
    return run
