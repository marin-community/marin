# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskProvider protocol: abstraction over a task execution backend."""

from typing import Protocol

from iris.cluster.types import WorkerId
from iris.rpc import job_pb2


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this provider implementation."""


class TaskProvider(Protocol):
    """Abstraction over a task execution backend.

    Implementations dispatch task lifecycle RPCs (Ping, StartTasks, StopTasks,
    PollTasks, etc.) to remote workers. Logs are pushed directly to the
    LogService by workers/tasks, not fetched via this protocol.
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
