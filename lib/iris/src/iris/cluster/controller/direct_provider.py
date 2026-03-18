# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DirectTaskProvider protocol: execution backends with no synthetic workers in DB."""

from typing import Protocol

from iris.cluster.controller.transitions import DirectProviderBatch, DirectProviderSyncResult
from iris.rpc import logging_pb2

# Re-export data classes so callers can import from this module.
from iris.cluster.controller.transitions import (  # noqa: F401
    ClusterCapacity,
    SchedulingEvent,
)


class DirectTaskProvider(Protocol):
    """Abstraction for execution backends that manage tasks directly without synthetic workers.

    Unlike TaskProvider (which operates on DispatchBatch per worker in DB), DirectTaskProvider
    receives the full cluster view and manages all tasks directly. The controller skips
    the scheduler and autoscaler when a DirectTaskProvider is configured.
    """

    @property
    def is_direct_provider(self) -> bool:
        """Whether this provider manages tasks directly (no worker DB rows)."""
        return True

    def sync(self, batch: DirectProviderBatch) -> DirectProviderSyncResult:
        """Sync all pending, running, and kill-queued tasks with the execution backend."""
        ...

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch live logs for a running task. Returns (entries, next_cursor)."""
        ...

    def close(self) -> None:
        """Release provider resources."""
        ...
