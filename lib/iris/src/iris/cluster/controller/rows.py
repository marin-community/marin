# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed row dataclasses shared between reads.py, writes.py, and transitions.py.

These carry no SQL or mutable state — just typed projections over query results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rigging.timing import Timestamp

from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2


@dataclass(frozen=True, slots=True)
class WorkerAttributeParams:
    key: str
    value_type: str
    str_value: str | None
    int_value: int | None
    float_value: float | None


@dataclass(frozen=True, slots=True)
class ActiveTaskRow:
    """Task projection joined with ``jobs`` + ``job_config``.

    Shared by every cascade/scheduling query (``_kill_non_terminal_tasks``,
    ``_find_coscheduled_siblings``, ``cancel_job``, ``preempt_task``,
    ``cancel_tasks_for_timeout``, ``_remove_failed_worker``, poll paths).
    Callers that need resource info for RPC payloads use ``PendingDispatchRow``
    instead; ``ActiveTaskRow`` carries only the fields needed for state-machine
    and cascade logic.
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


@dataclass(frozen=True, slots=True)
class PendingDispatchRow:
    """Scheduling payload for a task being dispatched to a direct provider.

    Unlike :class:`ActiveTaskRow`, this row carries the full serialized
    runtime configuration (entrypoint / environment / ports / constraints
    / task_image / timeout) so the caller can assemble a
    ``RunTaskRequest``. Kept separate so other active-task queries don't
    pay for loading these JSON blobs. Used for both PENDING-promotion and
    ASSIGNED-redrive paths (see `reads.tasks.list_*_for_direct_provider`).
    """

    task_id: JobName
    job_id: JobName
    current_attempt_id: int
    num_tasks: int
    resources: job_pb2.ResourceSpecProto
    entrypoint_json: str
    environment_json: str
    bundle_id: str
    ports_json: list
    constraints_json: str | None
    task_image: str
    timeout_ms: int | None


@dataclass(frozen=True, slots=True)
class WorkerResourceUsage:
    """Aggregate resources currently held by unfinished worker-bound attempts.

    Computed by ``reads.resource_usage_by_worker``; the scheduler
    subtracts these from a worker's totals to derive available capacity.
    """

    cpu_millicores: int
    memory_bytes: int
    gpu_count: int
    tpu_count: int


# ---------------------------------------------------------------------------
# User / budget result dataclasses (moved from db.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserStats:
    user: str
    task_state_counts: dict[int, int] = field(default_factory=dict)
    job_state_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskJobSummary:
    job_id: JobName
    task_count: int = 0
    completed_count: int = 0
    failure_count: int = 0
    preemption_count: int = 0
    task_state_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class UserBudget:
    user_id: str
    budget_limit: int
    max_band: int
    updated_at: Timestamp


@dataclass(frozen=True)
class EndpointQuery:
    endpoint_ids: tuple[str, ...] = ()
    name_prefix: str | None = None
    exact_name: str | None = None
    task_ids: tuple[JobName, ...] = ()
    limit: int | None = None


# ---------------------------------------------------------------------------
# Reservation dataclass (moved from transitions.py to break the
# reads.py → transitions.py → reads.py import cycle)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReservationClaim:
    """A claim binding a worker to a specific reservation entry.

    The controller assigns unclaimed workers to unsatisfied reservation entries
    each scheduling cycle. Once every entry for a job is claimed, the
    reservation gate opens and the job's tasks can be scheduled.
    """

    job_id: str
    entry_idx: int
