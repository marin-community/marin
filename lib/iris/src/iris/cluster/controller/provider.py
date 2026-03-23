# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskProvider protocol: unified abstraction over task execution backends.

Two implementations:
- WorkerProviderAdapter: wraps WorkerProvider (worker-daemon heartbeat RPC model)
- DirectProviderAdapter: wraps KubernetesProvider (direct-pod model, no workers)
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from iris.cluster.controller.transitions import (
    ClusterCapacity,
    ControllerTransitions,
    SchedulingEvent,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2, logging_pb2

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this provider implementation."""


@dataclass(frozen=True)
class FailedWorker:
    """A worker that failed during a sync cycle."""

    worker_id: WorkerId
    address: str | None
    is_permanent: bool  # True = WORKER_FAILED, False = TRANSIENT_FAILURE


@dataclass(frozen=True)
class ProviderSyncOutcome:
    """Unified result from a provider sync cycle.

    Returned by TaskProvider.sync(). The controller uses this to handle
    side effects (autoscaler notification, sibling failure, kill dispatch).
    """

    tasks_to_kill: set[JobName] = field(default_factory=set)
    scheduling_events: list[SchedulingEvent] = field(default_factory=list)
    capacity: ClusterCapacity | None = None
    failed_workers: list[FailedWorker] = field(default_factory=list)
    batch_count: int = 0
    error_count: int = 0
    error_worker_ids: list[str] = field(default_factory=list)


class TaskProvider(Protocol):
    """Unified abstraction over a task execution backend.

    The controller calls sync() in a loop. The provider drains work from
    the controller's transitions, syncs with the execution backend, applies
    results, and returns a ProviderSyncOutcome.

    Two execution models share this protocol:
    - Worker-daemon (WorkerProviderAdapter): per-worker heartbeat RPC fanout
    - Direct-pod (DirectProviderAdapter): cluster-wide pod management
    """

    @property
    def has_workers(self) -> bool:
        """Whether this provider uses persistent worker daemons.

        When True, the controller spawns scheduling and profiling threads.
        When False, the provider handles scheduling internally (e.g. k8s API).
        """
        ...

    def sync(
        self,
        transitions: ControllerTransitions,
    ) -> ProviderSyncOutcome:
        """Execute a complete drain -> sync -> apply cycle.

        Drains pending work from transitions, syncs with the execution backend,
        applies results back to transitions, and returns the outcome.
        """
        ...

    def kill_unmapped_tasks(
        self,
        task_ids: set[JobName],
        transitions: ControllerTransitions,
    ) -> bool:
        """Kill tasks that have no worker mapping.

        For worker-backed providers, this is a no-op (unmapped tasks are not yet
        assigned). For direct providers, buffers kills in the dispatch queue.

        Returns True if any kills were buffered.
        """
        ...

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
        worker_id: WorkerId | None = None,
        address: str | None = None,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch live logs for a running task from the execution backend.

        Returns (entries, next_cursor). Raises ProviderError on backend failure.
        Worker-backed providers use worker_id/address to route the request.
        Direct providers ignore worker_id/address and route via task_id.
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
        """Fetch full process status from an execution unit.

        Returns GetProcessStatusResponse. Raises ProviderUnsupportedError if not applicable.
        """
        ...

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        """Called when a worker is definitively failed. Evict cached connection state."""
        ...

    def profile_task(
        self,
        task_id: str,
        attempt_id: int,
        address: str | None,
        request: cluster_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> cluster_pb2.ProfileTaskResponse:
        """Profile a task.

        Worker-backed providers route via address.
        Direct providers route via task_id/attempt_id.
        """
        ...

    def exec_in_container(
        self,
        address: str,
        request: cluster_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int,
    ) -> cluster_pb2.Worker.ExecInContainerResponse:
        """Execute a command in a running task's container via worker RPC.

        Raises ProviderUnsupportedError for direct providers.
        """
        ...

    def get_cluster_status(self) -> cluster_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Get Kubernetes cluster status. Returns empty response for non-k8s providers."""
        ...

    def close(self) -> None:
        """Release provider resources (connections, thread pools, etc.)."""
        ...
