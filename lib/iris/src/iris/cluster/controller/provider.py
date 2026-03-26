# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskProvider protocol: abstraction over a task execution backend."""

from typing import Protocol

from iris.cluster.controller.transitions import DispatchBatch, HeartbeatApplyRequest
from iris.cluster.types import WorkerId
from iris.rpc import cluster_pb2, logging_pb2


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

    Log fetching for live tasks is provider-specific. Completed task logs are
    always available from the controller's local LogStore (written via
    HeartbeatApplyRequest log_entries), so the protocol only covers live
    log streaming.
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

    def fetch_live_logs(
        self,
        worker_id: WorkerId,
        address: str | None,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch live logs for a running task from the execution backend.

        Returns (entries, next_cursor). Raises ProviderError on backend failure.
        """
        ...

    def fetch_process_logs(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.FetchLogsRequest,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch execution-unit process logs (daemon logs).

        Returns (entries, next_cursor). Raises ProviderUnsupportedError if not applicable.
        """
        ...

    def get_process_status(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.GetProcessStatusRequest,
    ) -> cluster_pb2.GetProcessStatusResponse:
        """Fetch full process status (pid, memory, CPU, logs) from an execution unit.

        Returns GetProcessStatusResponse. Raises ProviderUnsupportedError if not applicable.
        """
        ...

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        """Called when a worker is definitively failed. Evict cached connection state."""
        ...

    def profile_task(
        self,
        address: str,
        request: cluster_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> cluster_pb2.ProfileTaskResponse:
        """Profile a task via RPC. Raises ProviderUnsupportedError if not applicable."""
        ...

    def close(self) -> None:
        """Release provider resources (connections, thread pools, etc.)."""
        ...
