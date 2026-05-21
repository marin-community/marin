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

- :mod:`zephyr.execution.internals` — shared types, the ``WorkerContext`` /
  ``StageRunner`` protocols, ``_worker_ctx_var``, and formatting/IO helpers
  (imported by ``zephyr.counters`` and ``zephyr.runners``).
- :mod:`zephyr.execution.coordinator` — ``ZephyrCoordinator`` and shard
  plumbing (``_regroup_result_refs``, ``_build_source_shards``,
  ``_reshard_refs``, ``_compute_tasks_from_shards``).
- :mod:`zephyr.execution.worker` — ``ZephyrWorker``.
- :mod:`zephyr.execution.context` — ``ZephyrContext``,
  ``ZephyrExecutionResult``, and the coordinator-as-job entrypoint.

Only the genuine public surface is re-exported here; tests and internal
callers should import private symbols (``_worker_ctx_var``, ``ShardTask``, …)
from the submodule that owns them.
"""

from zephyr.execution.context import ZephyrContext, ZephyrExecutionResult
from zephyr.execution.internals import CounterSnapshot, WorkerContext, ZephyrWorkerError, zephyr_worker_ctx

__all__ = [
    "CounterSnapshot",
    "WorkerContext",
    "ZephyrContext",
    "ZephyrExecutionResult",
    "ZephyrWorkerError",
    "zephyr_worker_ctx",
]
