# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskBackend: the contract every Iris execution backend implements.

The controller owns the database. Each tick it builds a placement-appropriate
:class:`BackendReconcileInput` from the DB, calls :meth:`TaskBackend.reconcile`,
and applies the returned :class:`BackendReconcileResult`. Backends perform
backend-specific I/O (worker-daemon RPC fan-out, ``kubectl apply``) but never
read or write the controller database — the interface is plain data in, plain
data out.

Two execution models exist, distinguished by :attr:`TaskBackend.placement`:

* ``Placement.IRIS`` — the Iris :class:`~iris.cluster.controller.scheduler.Scheduler`
  assigns task→worker, then the backend fans the per-worker reconcile RPC out
  to worker daemons. The controller passes pre-built ``plans`` and applies the
  raw ``worker_results`` through ``ops.worker.apply_reconcile`` (which emits
  worker heartbeats and runs the ``WORKER_RECONCILE`` transition source).
* ``Placement.BACKEND`` — the backend (Kueue, and later slurmctld) owns
  placement. The controller passes the desired ``tasks_to_run`` plus the
  ``running_tasks`` snapshot; the backend converges its own resources and
  returns pre-computed ``updates`` applied through
  ``ops.task.apply_direct_provider_updates`` (``DIRECT_PROVIDER`` source).

The two apply paths are NOT interchangeable — they emit different effects — so
the controller selects between them on the declared ``placement`` capability
rather than on the concrete backend type. A new backend slots in by declaring
its capabilities, with no ``isinstance`` branches in the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol

from finelog.client.log_client import Table
from finelog.types import LogWriterProtocol
from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.worker import ReconcileResult, WorkerReconcilePlan
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.runtime.profile import IrisProfile
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2, worker_pb2


class ProviderError(Exception):
    """Communication failure with the execution backend."""


class ProviderUnsupportedError(ProviderError):
    """Operation not supported by this backend implementation."""


class Placement(StrEnum):
    """Who decides which node a task runs on."""

    IRIS = "iris"
    """Iris schedules task→worker; the backend fans reconcile RPCs to daemons."""

    BACKEND = "backend"
    """The backend places tasks itself (Kueue, slurmctld); Iris does not schedule."""


@dataclass(frozen=True)
class SchedulingEvent:
    """A scheduling event surfaced by the execution backend (e.g. k8s events)."""

    task_id: str
    attempt_id: int
    event_type: str
    reason: str
    message: str
    timestamp: Timestamp


@dataclass(frozen=True)
class ClusterCapacity:
    """Aggregate capacity reported by the execution backend."""

    schedulable_nodes: int
    total_cpu_millicores: int
    available_cpu_millicores: int
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class BackendReconcileInput:
    """Desired + observed task state handed to :meth:`TaskBackend.reconcile`.

    Each backend reads the subset matching its :attr:`TaskBackend.placement`;
    the controller leaves the other fields empty. No ``worker_id`` is attached
    to ``tasks_to_run`` (BACKEND placement chooses the node itself); ``plans``
    are already worker-bound (IRIS placement scheduled them).
    """

    # BACKEND placement: tasks to ensure running + the snapshot of running ones.
    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)
    # IRIS placement: pre-built per-worker reconcile plans + their addresses.
    plans: list[WorkerReconcilePlan] = field(default_factory=list)
    worker_addresses: dict[WorkerId, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendReconcileResult:
    """Observed outcome returned by :meth:`TaskBackend.reconcile`.

    BACKEND placement returns pre-computed ``updates`` (the backend mapped its
    own observations to task states). IRIS placement returns raw
    ``worker_results``; the controller converts them against its DB snapshot at
    apply time (uid resolution + worker-loss interpretation). ``scheduling_events``
    and ``capacity`` are shared and may be empty/None where a backend has none.
    """

    updates: list[TaskUpdate] = field(default_factory=list)
    worker_results: list[ReconcileResult] = field(default_factory=list)
    scheduling_events: list[SchedulingEvent] = field(default_factory=list)
    capacity: ClusterCapacity | None = None


@dataclass(frozen=True)
class PingResult:
    """Result of a liveness Ping to a single worker (IRIS placement)."""

    worker_id: WorkerId
    worker_address: str | None
    healthy: bool = True
    health_error: str = ""
    error: str | None = None


@dataclass(frozen=True)
class TaskTarget:
    """Addresses one task attempt for on-demand RPCs (status / profile / exec).

    Worker-daemon backends route by :attr:`address`; direct backends route by
    :attr:`task_id` / :attr:`attempt_id`. Each backend reads the fields it needs;
    the controller fills them from the DB once at the RPC boundary.
    """

    task_id: str
    attempt_id: int
    worker_id: WorkerId | None
    address: str | None


class TaskBackend(Protocol):
    """Drives task execution + capacity reporting for a single cluster backend.

    Implementations dispatch backend-specific I/O and return plain data; they
    never touch the controller database.
    """

    name: str
    """Stable identifier, e.g. ``"gcp"``, ``"coreweave"``, later ``"slurm-stanford"``."""

    placement: ClassVar[Placement]
    """Who schedules task→node (selects the controller's input-build + apply path)."""

    manages_capacity: ClassVar[bool]
    """True when the backend provisions its own nodes (k8s cluster autoscaler);
    False when the Iris :class:`Autoscaler` provisions capacity for it."""

    def reconcile(self, batch: BackendReconcileInput) -> BackendReconcileResult:
        """Converge the backend toward ``batch`` and report observed state."""
        ...

    def capacity(self) -> ClusterCapacity | None:
        """Latest aggregate capacity, or None if the backend does not report it."""
        ...

    def ping_workers(self, workers: list[tuple[WorkerId, str | None]]) -> list[PingResult]:
        """Liveness-probe worker daemons (IRIS placement). May be a no-op."""
        ...

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        """Fetch full process status. Raises ProviderUnsupportedError if N/A."""
        ...

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        """Profile a task attempt. Raises ProviderUnsupportedError if N/A."""
        ...

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Exec a command in a task's container. Raises ProviderUnsupportedError if N/A."""
        ...

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        """Evict cached connection state for a definitively-failed worker."""
        ...

    def set_log_sink(
        self,
        log_client: LogWriterProtocol,
        task_stats_table: Table,
        profile_table: Table[IrisProfile],
    ) -> None:
        """Inject the finelog handles the controller resolves after connecting.

        Backends without a worker daemon (BACKEND placement) collect logs and
        write resource/profile samples directly to finelog. Daemon-backed
        backends ignore these — the worker writes its own rows.
        """
        ...
