# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job-based execution engine for Zephyr pipelines.

The coordinator runs as a fray *job* that internally creates coordinator and
worker *actors* as child jobs. Workers pull tasks from the coordinator actor,
execute shard operations, and report results back. Because actors are children
of the coordinator job, Iris cascading termination automatically cleans them
up when the coordinator exits or is killed — preventing stale-coordinator
bugs where orphaned coordinators and workers consume resources indefinitely.

This package's submodules host the implementation:

- :mod:`zephyr.execution.internals` — shared types (incl. ``ListShard`` /
  ``MemChunk`` / ``PickleDiskChunk``), the ``WorkerContext`` /
  ``StageRunner`` protocols, ``_worker_ctx_var``, and formatting/IO helpers.
- :mod:`zephyr.execution.coordinator` — ``ZephyrCoordinator`` and shard
  plumbing (``_regroup_result_refs``, ``_build_source_shards``,
  ``_reshard_refs``, ``_compute_tasks_from_shards``).
- :mod:`zephyr.execution.worker` — ``ZephyrWorker``.
- :mod:`zephyr.execution.context` — ``ZephyrContext``,
  ``ZephyrExecutionResult``, and the coordinator-as-job entrypoint.

Callers outside this package import everything they need from
``zephyr.execution`` directly — never from a submodule. The re-export order
below also matters: ``internals`` must be re-exported before ``context`` so
that ``zephyr.runners`` (loaded transitively through ``context`` → ``worker``)
can resolve its ``zephyr.execution`` imports against this partially-loaded
package.

``zephyr.shuffle`` depends on ``zephyr.execution`` (it imports ``ListShard``,
``MemChunk``, and ``_worker_ctx_var`` for ``_scatter_write_buffer_bytes``),
and nothing in ``zephyr.execution`` imports from ``zephyr.shuffle`` — the
stage-output writers (``_write_stage_output`` / ``_write_pickle_chunks``)
that call ``_write_scatter`` live in ``zephyr.runners`` (their only caller).
That one-way edge is what lets ``shuffle`` use a normal top-level import.
"""

# isort: off
# Import order matters: ``internals`` must be re-exported before ``context`` so that
# ``zephyr.runners`` — loaded transitively via ``context`` → ``worker`` — can resolve
# its ``from zephyr.execution import ...`` against this partially-loaded package.
from zephyr.execution.internals import (
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    CoordinatorUnreachable,
    CounterSnapshot,
    ListShard,
    MemChunk,
    PickleDiskChunk,
    PullStatus,
    ShardTask,
    StageRunner,
    TaskResult,
    WorkerContext,
    WorkerState,
    ZephyrWorkerError,
    _format_bytes,
    _format_count,
    _shared_data_path,
    _stage_throughput,
    _worker_ctx_var,
    zephyr_worker_ctx,
)
from zephyr.execution.coordinator import ZephyrCoordinator
from zephyr.execution.worker import ZephyrWorker
from zephyr.execution.context import ZephyrContext, ZephyrExecutionResult

# isort: on

__all__ = [
    "CoordinatorUnreachable",
    "CounterSnapshot",
    "ListShard",
    "MemChunk",
    "PickleDiskChunk",
    "PullStatus",
    "ShardTask",
    "StageRunner",
    "TaskResult",
    "WorkerContext",
    "WorkerState",
    "ZephyrContext",
    "ZephyrCoordinator",
    "ZephyrExecutionResult",
    "ZephyrWorker",
    "ZephyrWorkerError",
    "zephyr_worker_ctx",
]
