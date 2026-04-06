# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskProvider protocol: abstraction over a task execution backend."""

from typing import Protocol

from iris.cluster.controller.transitions import DispatchBatch, HeartbeatApplyRequest
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this provider implementation."""


class TaskProvider(Protocol):
    """Abstraction over a task execution backend.

    The controller calls sync() in a loop. The provider is responsible for
    submitting/cancelling tasks and collecting their state. It returns
    HeartbeatApplyRequest batches which the controller applies via
    ControllerTransitions.apply_heartbeat().

    Logs are pushed directly to the LogService by workers/tasks, not carried
    via heartbeats or fetched from the provider.
    """

    def sync(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        """Sync task state with the execution backend.

        Args:
            batches: One DispatchBatch per active execution unit, drained from the DB.

        Returns:
            For each batch: (batch, apply_request | None, error_str | None).
            apply_request is None on communication failure (caller uses fail_heartbeat).
        """
        ...

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
