# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Row dataclasses for the controller database.

These are the canonical typed result shapes returned by SA Core read helpers
in ``reads/``. The legacy ``Table``, ``Column``, ``Projection``, and decoder
machinery has been deleted; use ``schema_v2`` for DDL and SA Core for queries.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.types import JobName, WorkerId

# ---------------------------------------------------------------------------
# Hand-written row dataclasses
#
# Detail types are strict supersets of their corresponding Row types: they
# contain all the same fields (in a compatible order) plus extras. This
# structural relationship lets callers accept a Detail wherever a Row is
# expected without inheritance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobRow:
    """Lightweight job row for listings (jobs JOIN job_config, excludes constraints)."""

    job_id: JobName
    state: int
    submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    error: str | None
    exit_code: int | None
    name: str
    depth: int
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None


@dataclass(frozen=True, slots=True)
class JobSchedulingRow:
    """Full job row for scheduling — adds constraints over JobRow."""

    job_id: JobName
    state: int
    submitted_at: Timestamp
    root_submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    scheduling_deadline_epoch_ms: int | None
    error: str | None
    exit_code: int | None
    num_tasks: int
    is_reservation_holder: bool
    has_reservation: bool
    name: str
    depth: int
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None
    constraints_json: str | None
    has_coscheduling: bool
    coscheduling_group_by: str
    scheduling_timeout_ms: int | None
    max_task_failures: int


@dataclass(frozen=True, slots=True)
class JobDetailRow:
    """Full job detail — superset of JobSchedulingRow, adds dispatch config from job_config."""

    job_id: JobName
    state: int
    submitted_at: Timestamp
    root_submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    scheduling_deadline_epoch_ms: int | None
    error: str | None
    exit_code: int | None
    num_tasks: int
    is_reservation_holder: bool
    has_reservation: bool
    name: str
    depth: int
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None
    constraints_json: str | None
    has_coscheduling: bool
    coscheduling_group_by: str
    scheduling_timeout_ms: int | None
    max_task_failures: int
    entrypoint_json: str
    environment_json: str
    bundle_id: str
    ports_json: str
    max_retries_failure: int
    max_retries_preemption: int
    timeout_ms: int | None
    preemption_policy: int
    existing_job_policy: int
    priority_band: int
    task_image: str
    submit_argv_json: str
    reservation_json: str | None
    fail_if_exists: bool


@dataclass(frozen=True, slots=True)
class JobReservationRow:
    """Slim row for the per-tick reservation-claim recomputation.

    Decodes only the two columns the reservation loop touches — keeping the
    JSON proto blobs and other 35 ``JobDetailRow`` columns out of the hot
    path. See ``_jobs_with_reservations`` in ``controller.py``.
    """

    job_id: JobName
    reservation_json: str | None


@dataclass(frozen=True, slots=True)
class TaskRow:
    """Lightweight task row for scheduling."""

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at: Timestamp
    priority_band: int = 2
    priority_neg_depth: int = 0
    priority_root_submitted_ms: int = 0
    priority_insertion: int = 0


@dataclass(frozen=True, slots=True)
class TaskDetailRow:
    """Full task detail — superset of TaskRow, adds diagnostics and attempts."""

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at: Timestamp
    priority_band: int
    error: str | None
    exit_code: int | None
    started_at: Timestamp | None
    finished_at: Timestamp | None
    current_worker_id: WorkerId | None
    current_worker_address: str | None
    container_id: str | None = None
    attempts: tuple = dataclasses.field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class WorkerRow:
    """Durable worker columns: identity and capability."""

    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    attributes: dict = dataclasses.field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerDetailRow:
    """Full worker detail — superset of WorkerRow, adds metadata scalar columns."""

    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    md_hostname: str
    md_ip_address: str
    md_cpu_count: int
    md_memory_bytes: int
    md_disk_bytes: int
    md_tpu_name: str
    md_tpu_worker_hostnames: str
    md_tpu_worker_id: str
    md_tpu_chips_per_host_bounds: str
    md_gpu_count: int
    md_gpu_name: str
    md_gpu_memory_mb: int
    md_gce_instance_name: str
    md_gce_zone: str
    md_git_hash: str
    md_device_json: str
    attributes: dict = dataclasses.field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttemptRow:
    """Task attempt row."""

    task_id: JobName
    attempt_id: int
    worker_id: WorkerId | None
    state: int
    created_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    exit_code: int | None
    error: str | None


@dataclass(frozen=True, slots=True)
class EndpointRow:
    """Registered service endpoint."""

    endpoint_id: str
    name: str
    address: str
    task_id: JobName
    metadata: dict
    registered_at: Timestamp


@dataclass(frozen=True, slots=True)
class ApiKeyRow:
    """API key row."""

    key_id: str
    key_hash: str
    key_prefix: str
    user_id: str
    name: str
    created_at: Timestamp
    last_used_at: Timestamp | None


@dataclass(frozen=True, slots=True)
class UserBudgetRow:
    """User budget row."""

    user_id: str
    budget_limit: int
    max_band: int
    updated_at: Timestamp


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def tasks_with_attempts(tasks: Sequence[TaskDetailRow], attempts: Sequence[AttemptRow]) -> list[TaskDetailRow]:
    """Attach attempt rows to their parent task detail rows.

    Groups attempts by task_id and returns copies of each task with its
    ``attempts`` field populated as a tuple.
    """
    attempts_by_task: dict[JobName, list[AttemptRow]] = {}
    for attempt in attempts:
        attempts_by_task.setdefault(attempt.task_id, []).append(attempt)
    return [dataclasses.replace(task, attempts=tuple(attempts_by_task.get(task.task_id, ()))) for task in tasks]
