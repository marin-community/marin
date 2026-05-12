# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed row dataclasses shared between reads/* and transitions.py.

These were previously embedded in stores.py. They carry no SQL or mutable
state — just typed projections over query results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from iris.cluster.controller.codec import resource_spec_from_scalars
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2


@dataclass(frozen=True, slots=True)
class JobRecomputeBasis:
    state: int
    started_at_ms: int | None
    max_task_failures: int


@dataclass(frozen=True, slots=True)
class WorkerAttributeParams:
    key: str
    value_type: str
    str_value: str | None
    int_value: int | None
    float_value: float | None


@dataclass(frozen=True, slots=True)
class TaskScope:
    """Scope predicate for active-task queries.

    Exactly one field must be set. The store validates at the call boundary.
    ``null_worker=True`` matches rows where ``current_worker_id IS NULL``
    (direct-provider-promoted tasks).
    """

    job_id: JobName | None = None
    job_subtree: Sequence[JobName] | None = None
    worker_id: WorkerId | None = None
    worker_ids: Sequence[WorkerId] | None = None
    task_ids: Sequence[JobName] | None = None
    null_worker: bool = False


@dataclass(frozen=True, slots=True)
class ActiveTaskRow:
    """Task projection joined with ``jobs`` + ``job_config``.

    Shared by every cascade/scheduling query (``_kill_non_terminal_tasks``,
    ``_find_coscheduled_siblings``, ``cancel_job``, ``preempt_task``,
    ``cancel_tasks_for_timeout``, ``_remove_failed_worker``, poll paths). The
    resource columns are decoded into a single ``ResourceSpecProto`` so
    callers stop re-running ``resource_spec_from_scalars(...)`` at every
    site. Reservation-holder rows carry a populated ``resources`` that
    callers are expected to ignore (they never commit resources).
    """

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    current_worker_id: WorkerId | None
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    is_reservation_holder: bool
    has_coscheduling: bool
    resources: job_pb2.ResourceSpecProto


_ACTIVE_TASK_PROJECTION = (
    "t.task_id, t.job_id, t.state, t.current_attempt_id, t.current_worker_id, "
    "t.failure_count, t.preemption_count, t.max_retries_failure, t.max_retries_preemption, "
    "j.is_reservation_holder, "
    "jc.has_coscheduling, "
    "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json"
)


def _decode_active_task_row(row) -> ActiveTaskRow:
    worker_id = row["current_worker_id"]
    return ActiveTaskRow(
        task_id=JobName.from_wire(str(row["task_id"])),
        job_id=JobName.from_wire(str(row["job_id"])),
        state=int(row["state"]),
        current_attempt_id=int(row["current_attempt_id"]),
        current_worker_id=WorkerId(str(worker_id)) if worker_id is not None else None,
        failure_count=int(row["failure_count"]),
        preemption_count=int(row["preemption_count"]),
        max_retries_failure=int(row["max_retries_failure"]),
        max_retries_preemption=int(row["max_retries_preemption"]),
        is_reservation_holder=bool(int(row["is_reservation_holder"])),
        has_coscheduling=bool(int(row["has_coscheduling"])),
        resources=resource_spec_from_scalars(
            int(row["res_cpu_millicores"]),
            int(row["res_memory_bytes"]),
            int(row["res_disk_bytes"]),
            row["res_device_json"],
        ),
    )


@dataclass(frozen=True, slots=True)
class PendingDispatchRow:
    """Scheduling payload for a task being dispatched to a direct provider.

    Unlike :class:`ActiveTaskRow`, this row carries the full serialized
    runtime configuration (entrypoint / environment / ports / constraints
    / task_image / timeout) so the caller can assemble a
    ``RunTaskRequest``. Kept separate so other active-task queries don't
    pay for loading these JSON blobs. Used for both PENDING-promotion and
    ASSIGNED-redrive paths (see ``TaskStore.list_*_for_direct_provider``).
    """

    task_id: JobName
    job_id: JobName
    current_attempt_id: int
    num_tasks: int
    resources: job_pb2.ResourceSpecProto
    entrypoint_json: str
    environment_json: str
    bundle_id: str
    ports_json: str
    constraints_json: str | None
    task_image: str
    timeout_ms: int | None


_DISPATCH_PROJECTION = (
    "t.task_id, t.job_id, t.current_attempt_id, j.num_tasks, "
    "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json, "
    "jc.entrypoint_json, jc.environment_json, jc.bundle_id, jc.ports_json, "
    "jc.constraints_json, jc.task_image, jc.timeout_ms"
)


def _decode_dispatch_row(row) -> PendingDispatchRow:
    timeout_ms = row["timeout_ms"]
    return PendingDispatchRow(
        task_id=JobName.from_wire(str(row["task_id"])),
        job_id=JobName.from_wire(str(row["job_id"])),
        current_attempt_id=int(row["current_attempt_id"]),
        num_tasks=int(row["num_tasks"]),
        resources=resource_spec_from_scalars(
            int(row["res_cpu_millicores"]),
            int(row["res_memory_bytes"]),
            int(row["res_disk_bytes"]),
            row["res_device_json"],
        ),
        entrypoint_json=str(row["entrypoint_json"]),
        environment_json=str(row["environment_json"]),
        bundle_id=str(row["bundle_id"]),
        ports_json=str(row["ports_json"]),
        constraints_json=row["constraints_json"],
        task_image=str(row["task_image"]),
        timeout_ms=int(timeout_ms) if timeout_ms is not None else None,
    )


@dataclass(frozen=True, slots=True)
class WorkerResourceUsage:
    """Aggregate resources currently held by unfinished worker-bound attempts.

    Computed by ``reads.scheduler.resource_usage_by_worker``; the scheduler
    subtracts these from a worker's totals to derive available capacity.
    """

    cpu_millicores: int
    memory_bytes: int
    gpu_count: int
    tpu_count: int


@dataclass(frozen=True, slots=True)
class ReconcileRow:
    """One (task, attempt, worker) tuple driving per-worker reconcile.

    Returned by ``reads.scheduler.reconcile_rows_for_workers``; rows whose
    task is in ASSIGNED produce start payloads, rows in BUILDING/RUNNING
    populate the worker's expected-task set.
    """

    worker_id: WorkerId
    task_id: JobName
    attempt_id: int
    task_state: int
    attempt_state: int
    job_id: JobName
