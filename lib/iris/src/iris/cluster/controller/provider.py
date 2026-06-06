# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskProvider protocol: abstraction over a task execution backend.

Superseded by :mod:`iris.cluster.controller.backend` (``TaskBackend``); this
module survives only for the legacy ``TaskProvider`` protocol used as a type
hint until the controller is fully migrated. The exceptions are re-exported
from ``backend`` so there is a single canonical class for each.
"""

from typing import Protocol

from iris.cluster.controller.backend import ProviderError, ProviderUnsupportedError
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2

__all__ = ["ProviderError", "ProviderUnsupportedError", "TaskProvider"]


class TaskProvider(Protocol):
    """Abstraction over a task execution backend.

    Implementations dispatch task lifecycle RPCs (Ping, Reconcile, etc.) to
    remote workers. Logs are pushed directly to the LogService by
    workers/tasks, not fetched via this protocol.
    """

    def get_process_status(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        """Fetch full process status (pid, memory, CPU) from an execution unit.

        Returns GetProcessStatusResponse. Raises ProviderUnsupportedError if not applicable.
        """
        ...

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        """Called when a worker is definitively failed. Evict cached connection state."""
        ...

    def profile_task(
        self,
        address: str,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        """Profile a task via RPC. Raises ProviderUnsupportedError if not applicable."""
        ...

    def close(self) -> None:
        """Release provider resources (connections, thread pools, etc.)."""
        ...
