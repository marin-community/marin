# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller RPC service implementation handling job, task, and worker operations.

The controller expands jobs into tasks at submission time (a job with replicas=N
creates N tasks). Tasks are the unit of scheduling and execution. Job state is
aggregated from task states.
"""

import json
import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from finelog.client import LogClient
from rigging.server_auth import get_verified_identity, require_identity
from rigging.timing import Duration, ExponentialBackoff, Timer, Timestamp
from sqlalchemy import bindparam, case, func, select, text, tuple_

from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import (
    Constraint,
    backend_directive,
    cluster_directive,
    constraints_from_resources,
    merge_constraints,
    validate_tpu_request,
)
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.auth import (
    DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS,
    DEFAULT_JWT_TTL_SECONDS,
    MAX_ENDPOINT_TOKEN_TTL_SECONDS,
    ControllerAuth,
    create_api_key,
    list_api_keys,
    lookup_api_key_by_id,
    revoke_api_key,
    revoke_login_keys_for_user,
)
from iris.cluster.controller.autoscaler.status import PendingHint
from iris.cluster.controller.backend import BackendCapability, ProviderError, TaskBackend, TaskTarget
from iris.cluster.controller.budget import (
    compute_effective_band,
    compute_user_spend,
)
from iris.cluster.controller.codec import (
    decode_attribute_value,
    reconstruct_launch_job_request,
    resource_spec_from_job_row,
    resource_spec_from_scalars,
    worker_metadata_to_proto,
)
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.reads import TaskJobSummary
from iris.cluster.controller.reconcile.policy import MAX_ACTIVE_TASKS_PER_USER
from iris.cluster.controller.reconcile.task import TerminalKind
from iris.cluster.controller.run_template import RunTemplateCache
from iris.cluster.controller.scheduling.scheduler import SchedulingContext
from iris.cluster.controller.schema import (
    job_config_table,
    jobs_table,
    local_tasks,
    task_attempts_table,
    tasks_table,
    worker_attributes_table,
    workers_table,
)
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, task_row_can_be_scheduled
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.router import RoutingRequest, SubmitRouting
from iris.cluster.federation.store import HandoffState
from iris.cluster.process_status import get_process_status
from iris.cluster.redaction import redact_request_env_vars
from iris.cluster.runtime.profile import (
    PROFILE_NAMESPACE,
    IrisProfile,
    build_profile_row,
    profile_local_process,
)
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    JobName,
    TaskAttempt,
    UserBudgetDefaults,
    WorkerId,
    is_federated,
    is_job_finished,
)
from iris.rpc import controller_pb2, job_pb2, query_pb2, vm_pb2, worker_pb2
from iris.rpc.auth import AuthzAction, authorize, authorize_resource_owner
from iris.rpc.proto_display import (
    job_state_friendly,
    priority_band_name,
    resolve_container_profile,
    task_state_friendly,
)
from iris.time_proto import duration_from_proto, timestamp_to_proto

logger = logging.getLogger(__name__)


def attempt_is_worker_failure(state: int) -> bool:
    """Whether a terminal state (worker-failed or preempted) is a worker-side failure, not an application failure."""
    return state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED)


@dataclass(frozen=True)
class UserStats:
    user: str
    task_state_counts: dict[int, int] = field(default_factory=dict)
    job_state_counts: dict[int, int] = field(default_factory=dict)


# Maximum bundle size in bytes (25 MB) - matches client-side limit
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024
WORKDIR_FILE_OFFLOAD_THRESHOLD = 10 * 1024  # 10KB — externalize large workdir files to blob store

# Soft cap on how long launch_job waits for a replaced job's worker-bound
# attempts to finalize before force-reaping them. Sized to exceed the worst-
# case worker-death detection window so a vanished worker's attempts can be
# self-finalized by the reconcile-driven health path: the default
# worker_unreachable_grace (~50s) plus the per-RPC reconcile deadline and
# teardown, with slack. Past this point we log a warning, CASCADE-delete the
# rows, and proceed with the replacement — a vanished worker must not block the
# new submission indefinitely.
_JOB_REPLACEMENT_DRAIN_WAIT = Duration.from_seconds(120)

# Cap on the merged autoscaler action log returned by GetAutoscalerStatus; matches
# the per-autoscaler action_log deque cap so a single-backend view is unchanged.
_MERGED_AUTOSCALER_ACTIONS = 100

# Max unroutable job sample entries returned by ListBackends.
_UNROUTABLE_SAMPLE_SIZE = 10


def _accumulate_routing_decision(merged: vm_pb2.RoutingDecision, sub: vm_pb2.RoutingDecision) -> None:
    """Fold one backend's routing decision into the merged decision.

    Scale groups partition disjointly across backends (the single
    scale-group->backend key space), so the group-keyed maps never collide and the
    per-group lists concatenate. With a single backend this reproduces that
    backend's decision exactly.
    """
    for group, launch in sub.group_to_launch.items():
        merged.group_to_launch[group] = launch
    for group, reason in sub.group_reasons.items():
        merged.group_reasons[group] = reason
    for group, entries in sub.routed_entries.items():
        merged.routed_entries[group].CopyFrom(entries)
    merged.unmet_entries.extend(sub.unmet_entries)
    merged.group_statuses.extend(sub.group_statuses)


# A root LaunchJob submission is rejected if its client_revision_date is more
# than FRESHNESS_WINDOW older than today. Clients get exactly this long to
# upgrade after a new marin-iris release is cut.
FRESHNESS_WINDOW = timedelta(days=14)

# Date this freshness check shipped. An empty client_revision_date is
# interpreted as this date — already-deployed clients that don't set the field
# start being rejected FRESHNESS_WINDOW after rollout.
FEATURE_INTRODUCTION_DATE = date(2026, 4, 22)


def _check_client_freshness(client_date_str: str, now: date) -> None:
    """Reject root LaunchJob submissions whose client is older than FRESHNESS_WINDOW.

    Empty string is treated as FEATURE_INTRODUCTION_DATE so old clients (which
    don't set the field at all) behave as if they shipped the day this check
    rolled out.
    """
    if not client_date_str:
        client_date = FEATURE_INTRODUCTION_DATE
    else:
        try:
            client_date = date.fromisoformat(client_date_str)
        except ValueError as err:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"client_revision_date must be ISO YYYY-MM-DD, got {client_date_str!r}",
            ) from err
    floor = now - FRESHNESS_WINDOW
    if client_date < floor:
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"marin-iris client is too old (build {client_date.isoformat()}; "
            f"minimum {floor.isoformat()}). Run `uv sync` or upgrade "
            f"marin-iris and retry.",
        )


def _encode_query_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<blob:{len(value)} bytes>"
    return value


USER_TASK_STATES = (
    job_pb2.TASK_STATE_PENDING,
    job_pb2.TASK_STATE_ASSIGNED,
    job_pb2.TASK_STATE_BUILDING,
    job_pb2.TASK_STATE_RUNNING,
    job_pb2.TASK_STATE_SUCCEEDED,
    job_pb2.TASK_STATE_FAILED,
    job_pb2.TASK_STATE_KILLED,
    job_pb2.TASK_STATE_UNSCHEDULABLE,
    job_pb2.TASK_STATE_WORKER_FAILED,
    job_pb2.TASK_STATE_PREEMPTED,
    job_pb2.TASK_STATE_COSCHED_FAILED,
)
USER_JOB_STATES = (
    job_pb2.JOB_STATE_PENDING,
    job_pb2.JOB_STATE_BUILDING,
    job_pb2.JOB_STATE_RUNNING,
    job_pb2.JOB_STATE_SUCCEEDED,
    job_pb2.JOB_STATE_FAILED,
    job_pb2.JOB_STATE_KILLED,
    job_pb2.JOB_STATE_WORKER_FAILED,
    job_pb2.JOB_STATE_UNSCHEDULABLE,
)

# Terminal states KickTasks can force, mapped to the reconcile kernel's
# terminal-transition kind. PREEMPTED retries if budget remains; FAILED does not.
_KICK_KIND_BY_STATE: dict[int, TerminalKind] = {
    job_pb2.TASK_STATE_PREEMPTED: TerminalKind.PREEMPT,
    job_pb2.TASK_STATE_FAILED: TerminalKind.TIMEOUT,
}


@dataclass(frozen=True, slots=True)
class TaskWithAttempts:
    """Task detail columns with attempt rows attached."""

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at_ms: Timestamp
    priority_band: int
    error: str | None
    exit_code: int | None
    started_at_ms: Timestamp | None
    finished_at_ms: Timestamp | None
    current_worker_id: WorkerId | None
    current_worker_address: str | None
    container_id: str | None
    backend_id: str
    cluster: str
    # Display worker identity for a federated task (the peer-side worker name from
    # the federated_tasks sidecar); "" for a local task, which uses its worker FK.
    peer_worker_label: str
    attempts: tuple[Any, ...]

    @classmethod
    def from_row(cls, row, attempts: tuple[Any, ...]) -> "TaskWithAttempts":
        """Build from an SA Row (matching TASK_DETAIL_COLS + peer_worker_label) plus attempt rows."""
        return cls(
            task_id=row.task_id,
            job_id=row.job_id,
            state=row.state,
            current_attempt_id=row.current_attempt_id,
            failure_count=row.failure_count,
            preemption_count=row.preemption_count,
            max_retries_failure=row.max_retries_failure,
            max_retries_preemption=row.max_retries_preemption,
            submitted_at_ms=row.submitted_at_ms,
            priority_band=row.priority_band,
            error=row.error,
            exit_code=row.exit_code,
            started_at_ms=row.started_at_ms,
            finished_at_ms=row.finished_at_ms,
            current_worker_id=row.current_worker_id,
            current_worker_address=row.current_worker_address,
            container_id=row.container_id,
            backend_id=str(row.backend_id or ""),
            cluster=str(row.cluster),
            peer_worker_label=str(row.peer_worker_label or ""),
            attempts=attempts,
        )


def _current_attempt(task: TaskWithAttempts):
    """Get the latest attempt for a task, or None."""
    if not task.attempts:
        return None
    return task.attempts[-1]


def _task_worker_id(task: TaskWithAttempts) -> WorkerId | None:
    """Get the effective worker_id for a task."""
    current = _current_attempt(task)
    if current is None:
        return task.current_worker_id
    return current.worker_id


def _active_worker_id(task: TaskWithAttempts) -> WorkerId | None:
    """Get the active worker_id (None for pending tasks)."""
    if task.state == job_pb2.TASK_STATE_PENDING:
        return None
    return _task_worker_id(task)


def task_to_proto(task: TaskWithAttempts, worker_address: str = "") -> job_pb2.TaskStatus:
    """Convert a task row to a TaskStatus proto.

    Handles attempt conversion and timestamps. Per-attempt resource samples
    live in the ``iris.task`` stats namespace and are not populated here. The
    caller is responsible for resolving worker_address from worker_id if
    needed.
    """
    current_attempt = _current_attempt(task)

    attempts = []
    for attempt in task.attempts:
        proto_attempt = job_pb2.TaskAttempt(
            attempt_id=attempt.attempt_id,
            worker_id=str(attempt.worker_id) if attempt.worker_id else "",
            state=attempt.state,
            exit_code=attempt.exit_code or 0,
            error=attempt.error or "",
            is_worker_failure=attempt_is_worker_failure(attempt.state),
            attempt_uid=attempt.attempt_uid,
        )
        if attempt.started_at_ms is not None:
            proto_attempt.started_at.CopyFrom(timestamp_to_proto(attempt.started_at_ms))
        if attempt.finished_at_ms is not None:
            proto_attempt.finished_at.CopyFrom(timestamp_to_proto(attempt.finished_at_ms))
        attempts.append(proto_attempt)

    active_wid = _active_worker_id(task)
    # A federated task has no local worker FK; its worker identity is the peer-side
    # label mirrored into the federated_tasks sidecar. Surface it as worker_id so
    # the task views render the worker natively (as non-link text, since there is
    # no local worker detail page for a remote worker).
    worker_id = str(active_wid) if active_wid else (task.peer_worker_label if is_federated(task.cluster) else "")
    proto = job_pb2.TaskStatus(
        task_id=task.task_id.to_wire(),
        state=task.state,
        worker_id=worker_id,
        worker_address=worker_address or task.current_worker_address or "",
        exit_code=task.exit_code or 0,
        error=task.error or "",
        current_attempt_id=task.current_attempt_id,
        attempts=attempts,
        failure_count=task.failure_count,
        preemption_count=task.preemption_count,
        backend_id=task.backend_id,
        cluster=task.cluster,
    )
    # ``submitted_at_ms`` is a non-optional Timestamp (never None), so guard on the
    # value: a mirrored federated task not yet started carries the peer's real
    # submit time (> 0), while a 0 would render as epoch 1970.
    if task.submitted_at_ms.epoch_ms() > 0:
        proto.submitted_at.CopyFrom(timestamp_to_proto(task.submitted_at_ms))
    if current_attempt and current_attempt.started_at_ms:
        proto.started_at.CopyFrom(timestamp_to_proto(current_attempt.started_at_ms))
    if current_attempt and current_attempt.finished_at_ms:
        proto.finished_at.CopyFrom(timestamp_to_proto(current_attempt.finished_at_ms))
    if task.container_id:
        proto.container_id = task.container_id
    # For pending tasks with prior terminal attempts, surface retry context.
    if task.state == job_pb2.TASK_STATE_PENDING and task.attempts and task.attempts[-1].state in TERMINAL_TASK_STATES:
        last = task.attempts[-1]
        # current_attempt_id is the authoritative attempt index; len(attempts) now
        # counts only the current + failed attempts attached for the list view.
        proto.pending_reason = (
            f"Retrying (attempt {task.current_attempt_id + 1}, last: {job_pb2.TaskState.Name(last.state).lower()})"
        )
        proto.can_be_scheduled = True
    return proto


def worker_status_message(liveness: WorkerLiveness) -> str:
    """Build a human-readable status message for unhealthy workers."""
    if liveness.healthy:
        return ""
    age_ms = max(0, Timestamp.now().epoch_ms() - liveness.last_heartbeat_ms)
    return f"Unhealthy (last seen {age_ms // 1000}s ago)"


_WORKER_TARGET_PREFIX = "/system/worker/"


def _parse_worker_target(target: str) -> str | None:
    """Extract worker_id from a /system/worker/<worker_id> target.

    Returns the worker_id string, or None if the target does not match.
    """
    if target.startswith(_WORKER_TARGET_PREFIX):
        worker_id = target[len(_WORKER_TARGET_PREFIX) :]
        if worker_id:
            return worker_id
    return None


def _active_job_count(job_state_counts: dict[int, int]) -> int:
    """Return the count of non-terminal jobs in a user aggregate."""
    return sum(count for state, count in job_state_counts.items() if state not in TERMINAL_JOB_STATES)


def _task_state_counts_for_summary(task_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed task counts to the string-keyed RPC shape."""
    counts = {task_state_friendly(state): 0 for state in USER_TASK_STATES}
    for state, count in task_state_counts.items():
        counts[task_state_friendly(state)] = count
    return counts


def _job_state_counts_for_summary(job_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed job counts to the string-keyed RPC shape."""
    counts = {job_state_friendly(state): 0 for state in USER_JOB_STATES}
    for state, count in job_state_counts.items():
        counts[job_state_friendly(state)] = count
    return counts


# =============================================================================
# DB query helpers — thin wrappers over snapshot() for common read patterns
# =============================================================================


def _read_task_with_attempts(db: ControllerDB, task_id: JobName) -> TaskWithAttempts | None:
    """Return a TaskWithAttempts for ``task_id``, or None if absent."""
    with db.read_snapshot() as tx:
        task_row = reads.get_task_detail(tx, task_id)
        if task_row is None:
            return None
        attempt_rows = tx.execute(
            select(*reads.ATTEMPT_COLS)
            .where(task_attempts_table.c.task_id == task_id)
            .order_by(task_attempts_table.c.attempt_id.asc())
        ).all()
    return TaskWithAttempts.from_row(task_row, tuple(attempt_rows))


def _job_state(db: ControllerDB, job_id: JobName) -> int | None:
    """Fetch only the state column for a job, avoiding proto decode."""
    with db.read_snapshot() as tx:
        row = tx.execute(select(jobs_table.c.state).where(jobs_table.c.job_id == job_id)).first()
        return int(row.state) if row else None


def _worker_address(db: ControllerDB, worker_id: WorkerId) -> str | None:
    """Fetch only the address column for a worker, avoiding proto decode."""
    with db.read_snapshot() as tx:
        row = tx.execute(select(workers_table.c.address).where(workers_table.c.worker_id == worker_id)).first()
        return str(row.address) if row else None


def _read_worker(db: ControllerDB, worker_id: WorkerId):
    """Return a slim (worker_id, address, scale_group) row for ``worker_id``, or None."""
    with db.read_snapshot() as tx:
        return tx.execute(
            select(
                workers_table.c.worker_id,
                workers_table.c.address,
                workers_table.c.scale_group,
            ).where(workers_table.c.worker_id == worker_id)
        ).first()


@dataclass(frozen=True)
class _WorkerDetail:
    worker: Any  # SA Row from reads.get_worker_detail
    attributes: dict[str, str | int | float]
    running_tasks: frozenset[JobName]


def _read_worker_detail(db: ControllerDB, worker_id: WorkerId) -> _WorkerDetail | None:
    with db.read_snapshot() as tx:
        worker = reads.get_worker_detail(tx, worker_id)
        if worker is None:
            return None
        attr_rows = tx.execute(
            select(
                worker_attributes_table.c.key,
                worker_attributes_table.c.value_type,
                worker_attributes_table.c.str_value,
                worker_attributes_table.c.int_value,
                worker_attributes_table.c.float_value,
            ).where(worker_attributes_table.c.worker_id == worker_id)
        ).all()
        attrs = dict(decode_attribute_value(row) for row in attr_rows)
        running_rows = tx.execute(
            select(tasks_table.c.task_id)
            .select_from(
                tasks_table.join(
                    task_attempts_table,
                    (tasks_table.c.task_id == task_attempts_table.c.task_id)
                    & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
                )
            )
            .where(
                task_attempts_table.c.worker_id == worker_id,
                tasks_table.c.state.in_(bindparam("active_states", expanding=True)),
            ),
            {"active_states": list(ACTIVE_TASK_STATES)},
        ).all()
    return _WorkerDetail(
        worker=worker,
        attributes=attrs,
        running_tasks=frozenset(r.task_id for r in running_rows),
    )


# Terminal failure states the list view attaches in addition to the current
# attempt. PREEMPTED and COSCHED_FAILED are deliberately excluded: those are the
# high-volume churn — capacity preemptions and gang-cancellation collateral —
# that buries the genuine failures the dashboard wants to surface.
_LISTING_FAILURE_STATES = (job_pb2.TASK_STATE_FAILED, job_pb2.TASK_STATE_WORKER_FAILED)


def _tasks_for_listing(tx: Tx, *, job_id: JobName) -> list[TaskWithAttempts]:
    """Load tasks for the list view with their current and latest-failed attempts.

    Each task carries its current attempt plus the most recent attempt in each
    genuine-failure state (``_LISTING_FAILURE_STATES``), so a failure stays
    visible after the task is retried back into a running/pending state. Only the
    latest per state is attached, keeping the payload bounded for tasks with long
    retry histories. Attempts are ascending by ``attempt_id`` so the current
    attempt (the highest id) stays last. Reads within the caller's snapshot.
    """
    job_task_ids = select(tasks_table.c.task_id).where(tasks_table.c.job_id == job_id)
    task_rows = tx.execute(
        reads.task_detail_query()
        .where(tasks_table.c.job_id == job_id)
        .order_by(tasks_table.c.job_id.asc(), tasks_table.c.task_index.asc())
    ).all()
    # Current attempt per task (composite-PK lookup, at most one row each).
    current_attempt_rows = tx.execute(
        select(*reads.ATTEMPT_COLS).where(
            tuple_(task_attempts_table.c.task_id, task_attempts_table.c.attempt_id).in_(
                select(tasks_table.c.task_id, tasks_table.c.current_attempt_id).where(
                    tasks_table.c.job_id == job_id, tasks_table.c.current_attempt_id >= 0
                )
            )
        )
    ).all()
    # Highest failed attempt_id per (task, failure-state). One aggregate scan
    # of this job's failed attempts, then a PK join back to the full rows —
    # bounded to <= 2 rows per task no matter how deep the retry history runs.
    latest_failed = (
        select(
            task_attempts_table.c.task_id.label("task_id"),
            func.max(task_attempts_table.c.attempt_id).label("attempt_id"),
        )
        .where(
            task_attempts_table.c.task_id.in_(job_task_ids),
            task_attempts_table.c.state.in_(_LISTING_FAILURE_STATES),
        )
        .group_by(task_attempts_table.c.task_id, task_attempts_table.c.state)
        .subquery()
    )
    failed_attempt_rows = tx.execute(
        select(*reads.ATTEMPT_COLS).join(
            latest_failed,
            (task_attempts_table.c.task_id == latest_failed.c.task_id)
            & (task_attempts_table.c.attempt_id == latest_failed.c.attempt_id),
        )
    ).all()
    # Merge, deduping the current attempt when it is itself a failure.
    attempts_by_task: dict[JobName, dict[int, Any]] = {}
    for a in (*current_attempt_rows, *failed_attempt_rows):
        attempts_by_task.setdefault(a.task_id, {})[a.attempt_id] = a
    return [
        TaskWithAttempts.from_row(r, tuple(a for _, a in sorted(attempts_by_task.get(r.task_id, {}).items())))
        for r in task_rows
    ]


MAX_LIST_JOBS_LIMIT = 500
# Hard cap on how deep ListJobs callers may page. A correctly-filtered query
# should narrow the result set; anything reaching offsets this deep is a sign
# of a caller scanning the entire jobs table page-by-page, which is what the
# snapshot is supposed to prevent. Force callers to filter instead.
MAX_LIST_JOBS_OFFSET = 5000
MAX_LIST_WORKERS_LIMIT = 1000

# JobStatus carries int32 counter fields (failure_count, preemption_count,
# task_count, completed_count, task_state_counts values). SQLite SUM aggregates
# are 64-bit, so a runaway retry counter on a single task row would otherwise
# blow up proto serialization for the whole page. Clamp at the boundary and
# log so the underlying corruption surfaces in controller logs.
_PROTO_INT32_MAX = (1 << 31) - 1

# Substituted for a missing TaskJobSummary so the proto-construction paths can
# read counter fields uniformly without an ``if summary else 0`` per field.
# The ``job_id`` is a placeholder — callers feed the *enclosing* JobRow's
# job_id into ``_clamp_int32`` for diagnostics, never this one.
_EMPTY_TASK_SUMMARY = TaskJobSummary(job_id=JobName.from_wire("/_/_empty"))


def _clamp_int32(value: int, *, job_id: JobName, field: str) -> int:
    if value > _PROTO_INT32_MAX:
        logger.warning(
            "JobStatus.%s for %s overflowed int32 (%d > %d); clamping. "
            "Investigate the upstream counter — this usually means a task row "
            "has a corrupted failure_count/preemption_count.",
            field,
            job_id.to_wire(),
            value,
            _PROTO_INT32_MAX,
        )
        return _PROTO_INT32_MAX
    return value


def _job_status_counts(
    summary: TaskJobSummary | None, job_id: JobName, *, pre_sync_task_count: int = 0
) -> dict[str, Any]:
    """Return the clamped int32 counter fields for a ``JobStatus``.

    Spread into ``JobStatus(...)`` as ``**_job_status_counts(summary, job_id)``.
    A ``None`` summary collapses to all-zero counters (no log noise); a real
    summary runs each field through ``_clamp_int32`` so 64-bit aggregates
    never trip the proto encoder.

    ``pre_sync_task_count`` is the requested replica count of a federated job
    (``jobs.num_tasks``). A handed-off job has no local task rows until the first
    FederationSync mirrors the peer's set, so with a ``None`` summary it is
    surfaced as ``task_count`` — the job reads "N tasks, awaiting the peer"
    instead of an unexplained empty table.
    """
    s = summary or _EMPTY_TASK_SUMMARY
    task_count = s.task_count if summary is not None else pre_sync_task_count
    return {
        "failure_count": _clamp_int32(s.failure_count, job_id=job_id, field="failure_count"),
        "preemption_count": _clamp_int32(s.preemption_count, job_id=job_id, field="preemption_count"),
        "task_count": _clamp_int32(task_count, job_id=job_id, field="task_count"),
        "completed_count": _clamp_int32(s.completed_count, job_id=job_id, field="completed_count"),
        "task_state_counts": {
            task_state_friendly(state): _clamp_int32(count, job_id=job_id, field=f"task_state_counts[{state}]")
            for state, count in s.task_state_counts.items()
        },
    }


def _peer_status(cluster: str, handoff_state: int | None, has_reported_tasks: bool) -> int:
    """The ``PeerStatus`` for a job, derived from its cluster coordinate, its
    ``federated_jobs.handoff_state``, and whether the peer has mirrored any task
    rows back yet.

    Branch order matters: a peer that has reported tasks is ``SYNCED`` even if
    the local handle still reads ``PENDING_HANDOFF``. The sync loop can mirror a
    job's state before a transient RPC failure lets ``mark_handed_off`` run, so
    the presence of mirrored task rows is the more current signal — checking it
    before the handle avoids labelling a running, populated job "awaiting the
    peer's acceptance". A rejected handoff never reports tasks, so ``REJECTED``
    is checked first.

    For a terminal job this is the last posture observed (e.g. a handoff
    cancelled before delivery stays ``PENDING_SCHEDULING``). ``handoff_state`` is
    ``None`` only if the SENT handle is gone; the handle and jobs row are created
    and CASCADE-deleted together, so for a live federated job that is a
    can't-happen fallback, treated as handed off.
    """
    if not is_federated(cluster):
        return job_pb2.PEER_STATUS_NONE
    if handoff_state == int(HandoffState.HANDOFF_REJECTED):
        return job_pb2.PEER_STATUS_REJECTED
    if has_reported_tasks:
        return job_pb2.PEER_STATUS_SYNCED
    if handoff_state == int(HandoffState.PENDING_HANDOFF):
        return job_pb2.PEER_STATUS_PENDING_SCHEDULING
    return job_pb2.PEER_STATUS_ASSIGNED


def _federated_pending_reason(cluster: str, peer_status: int) -> str:
    """Pending reason for a federated job, which the local scheduler never sees.

    A handed-off job's tasks live on the peer and are excluded from the local
    fold, so the local scheduling diagnostic is meaningless for it. Derive the
    message from the handoff posture (single source of truth): awaiting the
    peer's acceptance, awaiting its first status report, or pending on the peer
    once it has reported tasks.
    """
    if peer_status == job_pb2.PEER_STATUS_PENDING_SCHEDULING:
        return f"Awaiting acceptance by peer {cluster}"
    if peer_status == job_pb2.PEER_STATUS_ASSIGNED:
        return f"Handed off to peer {cluster}; awaiting first status report"
    return f"Pending on peer {cluster}"


def _filter_and_sort_workers(
    workers: list[tuple[Any, dict]],
    liveness_by_id: dict[WorkerId, WorkerLiveness],
    query: controller_pb2.Controller.WorkerQuery,
) -> list[tuple[Any, dict]]:
    """Apply the ``WorkerQuery`` contains filter and sort the cached roster.

    Filtering and sorting happen in Python against the cached worker roster
    rather than in SQL: the roster is bounded by cluster size (low thousands)
    and already cached on the controller, so the marginal cost of a re-scan
    per request is much smaller than reissuing the SELECT + worker_attributes
    fan-out.
    """
    needle = query.contains.lower() if query.contains else ""
    if needle:
        workers = [
            (w, attrs)
            for w, attrs in workers
            if needle in str(w.worker_id).lower() or (w.address and needle in w.address.lower())
        ]

    sort_field = query.sort_field or controller_pb2.Controller.WORKER_SORT_FIELD_WORKER_ID
    descending = query.sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC
    if sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_LAST_HEARTBEAT:
        workers = sorted(workers, key=lambda wa: liveness_by_id[wa[0].worker_id].last_heartbeat_ms, reverse=descending)
    elif sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_DEVICE_TYPE:
        # CPU workers persist with ``device_type == ""``; under ascending sort
        # they group first (treating CPU as the no-accelerator baseline).
        workers = sorted(workers, key=lambda wa: (wa[0].device_type, str(wa[0].worker_id)), reverse=descending)
    else:
        workers = sorted(workers, key=lambda wa: str(wa[0].worker_id), reverse=descending)
    return workers


def _resolve_state_filter(state_filter: str) -> tuple[int, ...] | None:
    """Resolve a ``JobQuery.state_filter`` string into concrete state ids.

    Returns ``USER_JOB_STATES`` when no filter is set, a single-element tuple
    when it matches a known user-visible state, or ``None`` when the filter
    does not match any known state (caller should return an empty page).
    """
    if not state_filter:
        return USER_JOB_STATES
    normalized = state_filter.lower()
    for st in USER_JOB_STATES:
        if job_state_friendly(st) == normalized:
            return (st,)
    return None


def _query_jobs(
    tx,
    query: controller_pb2.Controller.JobQuery,
    state_ids: tuple[int, ...],
) -> tuple[list, int]:
    """Execute a ``JobQuery`` and return ``(rows, total_count)``.

    ``state_ids`` is the pre-resolved state filter (always non-empty); the
    caller owns "unknown state -> empty page" handling so that a bad filter
    never reaches SQL. The caller also owns the read snapshot — list_jobs
    chains the SELECT, COUNT, and downstream summary/parent queries on a
    single snapshot to keep the per-connection page cache hot.
    """
    if query.scope == controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN and not query.parent_job_id:
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            "query.parent_job_id is required for JOB_QUERY_SCOPE_CHILDREN",
        )
    return reads.list_jobs(tx, query, state_ids)


def _query_from_list_jobs_request(
    request: controller_pb2.Controller.ListJobsRequest,
) -> controller_pb2.Controller.JobQuery:
    """Return the request's ``JobQuery`` with paging clamped to safe bounds."""
    query = controller_pb2.Controller.JobQuery()
    if request.HasField("query"):
        query.CopyFrom(request.query)

    # Clamp paging: 0 (unset) or out-of-range values default to MAX. Unbounded
    # listing is not supported because downstream per-page work
    # (task_summaries_for_jobs, parent_ids_with_children) grows an IN-clause
    # with one placeholder per returned row.
    if query.limit <= 0 or query.limit > MAX_LIST_JOBS_LIMIT:
        query.limit = MAX_LIST_JOBS_LIMIT
    if query.offset < 0:
        query.offset = 0
    if query.offset > MAX_LIST_JOBS_OFFSET:
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            f"query.offset={query.offset} exceeds MAX_LIST_JOBS_OFFSET={MAX_LIST_JOBS_OFFSET}; "
            "narrow the result set with state_filter/name_filter/parent_job_id instead of paging deeper.",
        )
    return query


def _worker_roster(db: ControllerDB) -> list[tuple[Any, dict]]:
    """Return ``(worker_row, attrs_dict)`` pairs for all registered workers.

    Two queries total: one SELECT over ``workers`` for the full roster and
    one over ``worker_attributes`` filtered to those worker_ids. The old
    shape (per-worker ``get_worker_detail``) issued N+1 queries.
    """
    with db.read_snapshot() as tx:
        decoded = tx.execute(select(*reads.WORKER_DETAIL_COLS)).all()
        if not decoded:
            return []
        worker_ids = [w.worker_id for w in decoded]
        attr_rows = tx.execute(
            select(
                worker_attributes_table.c.worker_id,
                worker_attributes_table.c.key,
                worker_attributes_table.c.value_type,
                worker_attributes_table.c.str_value,
                worker_attributes_table.c.int_value,
                worker_attributes_table.c.float_value,
            ).where(worker_attributes_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))),
            {"worker_ids": list(worker_ids)},
        ).all()
        attrs_by_worker: dict[str, dict[str, str | int | float]] = {}
        for row in attr_rows:
            wid = str(row.worker_id)
            key, value = decode_attribute_value(row)
            attrs_by_worker.setdefault(wid, {})[key] = value
    return [(w, attrs_by_worker.get(str(w.worker_id), {})) for w in decoded]


_ACTIVE_JOB_STATES = (
    job_pb2.JOB_STATE_PENDING,
    job_pb2.JOB_STATE_BUILDING,
    job_pb2.JOB_STATE_RUNNING,
)


def _live_user_stats(db: ControllerDB) -> list[UserStats]:
    """Aggregate job/task counts per user.

    The user set is every owner who has ever submitted a job (any state), so the
    landing page lists people even when none of their jobs are currently active.
    The per-state counts only cover active (non-terminal) jobs/tasks, so the
    Running/Pending/Active columns reflect current load and an idle user shows
    all zeros rather than disappearing.
    """
    active_states = list(_ACTIVE_JOB_STATES)
    with db.read_snapshot() as tx:
        user_rows = tx.execute(select(jobs_table.c.user_id).distinct()).all()
        job_rows = tx.execute(
            select(
                jobs_table.c.user_id,
                jobs_table.c.state,
                func.count().label("cnt"),
            )
            .where(jobs_table.c.state.in_(bindparam("active_states", expanding=True)))
            .group_by(jobs_table.c.user_id, jobs_table.c.state),
            {"active_states": active_states},
        ).all()
        task_rows = tx.execute(
            select(
                jobs_table.c.user_id,
                tasks_table.c.state,
                func.count().label("cnt"),
            )
            .select_from(tasks_table.join(jobs_table, tasks_table.c.job_id == jobs_table.c.job_id))
            .where(jobs_table.c.state.in_(bindparam("active_states", expanding=True)))
            .group_by(jobs_table.c.user_id, tasks_table.c.state),
            {"active_states": active_states},
        ).all()
    by_user: dict[str, UserStats] = {str(row.user_id): UserStats(user=str(row.user_id)) for row in user_rows}
    for row in job_rows:
        stats = by_user.setdefault(str(row.user_id), UserStats(user=str(row.user_id)))
        stats.job_state_counts[int(row.state)] = int(row.cnt)
    for row in task_rows:
        stats = by_user.setdefault(str(row.user_id), UserStats(user=str(row.user_id)))
        stats.task_state_counts[int(row.state)] = int(row.cnt)
    return list(by_user.values())


def _attempts_for_worker(
    db: ControllerDB, worker_id: WorkerId, limit: int = 50
) -> list[controller_pb2.Controller.WorkerTaskAttempt]:
    """Return per-attempt history for ``worker_id``, newest first.

    Indexed scan of ``task_attempts`` via ``idx_task_attempts_worker_task``;
    each retry of the same task is its own row so the dashboard can render
    independent state/duration per attempt rather than inheriting from the
    parent task (which produced bogus duplicate-RUNNING rows).
    """
    with db.read_snapshot() as tx:
        raw_rows = tx.execute(
            select(*reads.ATTEMPT_COLS)
            .where(task_attempts_table.c.worker_id == worker_id)
            .order_by(
                case(
                    (task_attempts_table.c.started_at_ms.is_not(None), task_attempts_table.c.started_at_ms),
                    else_=task_attempts_table.c.created_at_ms,
                ).desc()
            )
            .limit(limit)
        ).all()
        # Each task inherits its resource allocation from its parent job. Batch
        # the job_config lookup so a worker that ran many attempts across a few
        # jobs costs one extra query rather than one per attempt.
        job_ids = {row.task_id.parent for row in raw_rows if row.task_id.parent is not None}
        resources_by_job: dict[JobName, job_pb2.ResourceSpecProto] = {}
        if job_ids:
            jc_rows = tx.execute(
                select(
                    job_config_table.c.job_id,
                    job_config_table.c.res_cpu_millicores,
                    job_config_table.c.res_memory_bytes,
                    job_config_table.c.res_disk_bytes,
                    job_config_table.c.res_device_json,
                ).where(job_config_table.c.job_id.in_(list(job_ids)))
            ).all()
            for jc in jc_rows:
                if jc.res_cpu_millicores or jc.res_memory_bytes or jc.res_disk_bytes or jc.res_device_json:
                    resources_by_job[jc.job_id] = resource_spec_from_scalars(
                        jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json
                    )
    out: list[controller_pb2.Controller.WorkerTaskAttempt] = []
    for row in raw_rows:
        proto_attempt = job_pb2.TaskAttempt(
            attempt_id=row.attempt_id,
            worker_id=str(row.worker_id) if row.worker_id else "",
            state=row.state,
            exit_code=row.exit_code or 0,
            error=row.error or "",
            is_worker_failure=attempt_is_worker_failure(row.state),
            attempt_uid=row.attempt_uid,
        )
        if row.started_at_ms is not None:
            proto_attempt.started_at.CopyFrom(timestamp_to_proto(row.started_at_ms))
        if row.finished_at_ms is not None:
            proto_attempt.finished_at.CopyFrom(timestamp_to_proto(row.finished_at_ms))
        out.append(
            controller_pb2.Controller.WorkerTaskAttempt(
                task_id=row.task_id.to_wire(),
                attempt=proto_attempt,
                resources=resources_by_job.get(row.task_id.parent),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class PendingKick:
    """A queued administrative task kick.

    ``attempt_id`` is the targeted attempt, or ``None`` to take whatever attempt
    is current when the kick is applied.
    """

    task_id: JobName
    attempt_id: int | None
    kind: TerminalKind
    reason: str


class ControllerProtocol(Protocol):
    """Protocol for controller operations used by ControllerServiceImpl."""

    def wake(self) -> None: ...

    def request_worker_eviction(self, worker_ids: Sequence[WorkerId]) -> None: ...

    def request_task_kicks(self, kicks: Sequence[PendingKick]) -> None: ...

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None: ...

    def begin_checkpoint(self) -> tuple[str, Any]: ...

    @property
    def last_scheduling_context(self) -> SchedulingContext | None: ...

    @property
    def provider(self) -> Any: ...

    @property
    def backends(self) -> dict[str, TaskBackend]: ...

    @property
    def federation(self) -> FederationManager: ...

    @property
    def capabilities(self) -> frozenset[BackendCapability]: ...

    @property
    def run_template_cache(self) -> RunTemplateCache: ...

    def backend_id_for_scale_group(self, scale_group: str) -> str: ...

    def all_liveness(self) -> dict[WorkerId, WorkerLiveness]: ...

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness: ...

    @property
    def last_unroutable_jobs(self) -> dict[str, str]: ...

    @property
    def scale_group_to_backend(self) -> dict[str, str]: ...


def _profile_is_elevated(profile: int) -> bool:
    """Whether a container profile is host-root-equivalent and so requires admin."""
    return resolve_container_profile(profile) in (
        job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS,
        job_pb2.CONTAINER_PROFILE_PRIVILEGED,
    )


def _inject_resource_constraints(
    request: controller_pb2.Controller.LaunchJobRequest,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Merge auto-generated device constraints into a job submission request.

    Constraints derived from ResourceSpecProto.device (device-type, device-variant)
    are merged with any explicit user constraints on the request.  For canonical
    keys the user's explicit constraints replace auto-generated ones, so e.g.
    a user-provided multi-variant IN constraint overrides the single-variant
    EQ constraint from the resource spec.
    """
    auto = constraints_from_resources(request.resources)
    if not auto:
        return request

    user = [Constraint.from_proto(c) for c in request.constraints]
    merged = merge_constraints(auto, user)

    new_request = controller_pb2.Controller.LaunchJobRequest()
    new_request.CopyFrom(request)
    del new_request.constraints[:]
    for c in merged:
        new_request.constraints.append(c.to_proto())
    return new_request


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Args:
        controller: Controller runtime for scheduling and worker management
        bundle_store: Bundle store for zip storage.
        log_client: LogClient for reading task logs through LogService.FetchLogs.
        db: Underlying database connection.
        endpoints: Endpoint projection (in-memory cache over the endpoints table).
    """

    def __init__(
        self,
        controller: ControllerProtocol,
        bundle_store: BundleStore,
        log_client: LogClient,
        *,
        db: ControllerDB,
        endpoints: EndpointsProjection,
        endpoint_service: EndpointServiceImpl,
        auth: ControllerAuth | None = None,
        user_budget_defaults: UserBudgetDefaults | None = None,
    ):
        self._db = db
        self._endpoints = endpoints
        # The leased registry owns endpoint logic; the legacy
        # ControllerService.{Register,Unregister,List}Endpoint RPCs delegate here.
        self._endpoint_service = endpoint_service
        self._controller = controller
        self._bundle_store = bundle_store
        self._log_client = log_client
        self._timer = Timer()
        self._auth = auth or ControllerAuth()
        self._user_budget_defaults = user_budget_defaults or UserBudgetDefaults()
        self._profile_table = self._log_client.get_table(PROFILE_NAMESPACE, IrisProfile)

    def bundle_zip(self, bundle_id: str) -> bytes:
        return self._bundle_store.get(bundle_id)

    def blob_data(self, blob_id: str) -> bytes:
        return self._bundle_store.get(blob_id)

    def _get_autoscaler_pending_hints(self) -> dict[str, PendingHint]:
        """Build autoscaler-based pending hints keyed by job id, merged across
        every backend's autoscaler.

        Each backend owns a disjoint set of scale groups (and thus jobs), so the
        per-backend hint dicts never collide on a job id. Each autoscaler caches
        its hint dict per evaluate() cycle, so this stays a cheap merge rather
        than a full AutoscalerStatus rebuild on every GetJobStatus RPC.
        """
        hints: dict[str, PendingHint] = {}
        for backend in self._controller.backends.values():
            autoscaler = backend.autoscaler
            if autoscaler is not None:
                hints.update(autoscaler.get_pending_hints())
        return hints

    def _authorize_job_owner(self, job_id: JobName) -> None:
        """Raise PERMISSION_DENIED if the authenticated user doesn't own this job.

        Skipped when no auth provider is configured (null-auth mode).
        """
        if not self._auth.provider:
            return
        authorize_resource_owner(job_id.user)

    def _wait_until_job_drained(self, job_id: JobName, wait: Duration) -> bool:
        """Wait up to ``wait`` for ``job_id`` to have no unfinished worker-bound
        attempts. Returns ``True`` if drained, ``False`` if the wait elapsed.

        Polls the snapshot DB; the reconcile-observation path landing terminal
        updates is what flips the predicate. Caller decides whether to reap the
        predecessor when the wait elapses — a stuck worker must not block
        the new submission forever.
        """

        def drained() -> bool:
            with self._db.read_snapshot() as tx:
                return not reads.has_unfinished_worker_attempts(tx, job_id)

        return ExponentialBackoff(initial=1.0, maximum=10.0, factor=2).wait_until(drained, timeout=wait)

    def _replace_finished_job(self, cur, job_id: JobName) -> bool:
        """Attempt to replace a terminal job; signal whether a drain is needed.

        CASCADE-deleting a job's tasks while its attempts are still worker-
        bound destroys the rows the reconcile-observation path needs to find when it
        stamps ``finished_at_ms``. Returns ``True`` when the caller must wait
        for worker-bound attempts to finalize before retrying (the job rows
        are left in place), ``False`` when removal completed in this
        transaction. Every replacement path in ``launch_job`` funnels through
        here so the contract is uniform.
        """
        if reads.has_unfinished_worker_attempts(cur, job_id):
            return True
        ops.job.remove_finished(cur, job_id)
        return False

    def _admit_federated_resubmit(
        self,
        cur: Tx,
        job_id: JobName,
        request: controller_pb2.Controller.LaunchJobRequest,
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Federation-aware admission for a handoff whose job id already exists.

        Runs before the generic ``existing_job_policy`` switch, so it governs a
        handoff regardless of that policy. A re-drive from the *same* requester
        (recorded as a RECEIVED handle) is an idempotent replay — return the existing
        job. Any other existing row — a local job, a job received from a different
        requester, or a SENT handle — is a genuine collision the parent must see, so
        reject with ``ALREADY_EXISTS``.
        """
        if reads.received_requester(cur, job_id) == request.federation.requester_id:
            return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
        raise ConnectError(
            Code.ALREADY_EXISTS,
            f"Job {job_id} already exists and was not handed off by {request.federation.requester_id!r}",
        )

    def _hand_off_job(
        self,
        job_id: JobName,
        request: controller_pb2.Controller.LaunchJobRequest,
        peer_id: str,
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Hand a job off to a federation peer and return the parent's job id.

        Only a root job is ever handed off — the peer runs it under the same,
        cluster-invariant job id, so handing off a non-root job would clash with the
        job's own tree on the peer. The manager persists the federated handle in one
        local transaction, then synchronously delivers it to the peer. A failed
        delivery is not fatal — the sync loop re-drives the handle — so an admitted or
        already-present handle still returns its id.
        """
        if not job_id.is_root:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Job {job_id} is not a root job; only whole root jobs may be federated to a peer.",
            )
        self._controller.federation.submit_federated_handle(
            local_job_id=job_id,
            request=request,
            peer_id=peer_id,
            owner_principal=job_id.user,
        )
        return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())

    def launch_job(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Submit a new job to the controller.

        The job is expanded into tasks based on the replicas field
        (defaulting to 1). Each task has ID "/job/.../index".
        """
        if not request.name:
            raise ConnectError(Code.INVALID_ARGUMENT, "Job name is required")

        # Coscheduling requires a non-empty group_by: it names the topology level
        # the gang is scheduled onto. An empty group_by is permanently
        # unschedulable on the worker path and silently admits without gang on the
        # K8s direct path, so reject it here rather than let it fail differently
        # (and silently) downstream.
        if request.HasField("coscheduling") and not request.coscheduling.group_by:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                "coscheduling requires a non-empty group_by (the topology level to gang on)",
            )

        job_id = JobName.from_wire(request.name)

        # Reject root RPC submissions from stale clients. Direct in-process
        # calls have no wire client; tests and harnesses use ctx=None.
        # Nested submissions are exempt because they come from an already
        # running workload, which would otherwise crash mid-flight as the
        # freshness window slides forward.
        if job_id.is_root and ctx is not None:
            _check_client_freshness(request.client_revision_date, date.today())

        # Reconcile the requested job owner with the authenticated principal.
        #
        # The job name's user segment names the *acting* owner the job is
        # attributed to; the verified identity is the authenticated *principal*.
        # These are distinct: a non-admin may only act as themselves, so we pin
        # the owner to the principal to prevent impersonation. An admin — which
        # includes a trusted-loopback caller (see docs/auth-loopback-transition)
        # — may submit on behalf of any user, so the requested owner stands.
        # This makes loopback-trust attribute jobs exactly as null-auth always
        # has (the name is authoritative), while token users stay pinned.
        # Only root submissions carry an owner segment; child jobs inherit it.
        #
        # A received federation handoff is exempt from re-pinning: the acting owner
        # it asserts rides in ``federation.owner_principal`` (already encoded in the
        # handoff name's user segment), so re-pinning to the peer's own principal
        # would corrupt the attribution.
        identity = get_verified_identity()

        # ``federation`` is a field on the public LaunchJobRequest, so guard who may
        # set it. With auth on it marks a trusted peer-to-peer handoff — honor it
        # only from an admin principal (a peer authenticates with admin credentials).
        # A non-admin, or an unauthenticated caller under configured auth, that sets
        # it is forging a handoff to run a job as another user, so reject it. In
        # no-auth mode the cluster already trusts the name, so there is nothing to
        # forge. A received handoff is always a root job.
        if request.HasField("federation") and self._auth.provider:
            if identity is None or identity.role != "admin":
                raise ConnectError(
                    Code.PERMISSION_DENIED, "The federation handoff field may only be set by a trusted peer."
                )
            if not job_id.is_root:
                raise ConnectError(Code.INVALID_ARGUMENT, "A federation handoff must be a root job.")

        if self._auth.provider and identity is not None and job_id.is_root and identity.role != "admin":
            job_id = JobName.root(identity.user_id, job_id.name)

        # For non-root jobs, verify the caller owns the parent hierarchy
        if self._auth.provider and identity is not None and not job_id.is_root:
            self._authorize_job_owner(job_id)

        # Priority band validation.
        #
        # - PRODUCTION additionally requires MANAGE_BUDGETS when auth is on;
        #   admins pass here and skip the max_band cap below.
        # - The max_band cap fires regardless of auth mode, keyed on the
        #   claimed job_id.user. In anonymous mode this doesn't guarantee the
        #   user is who they claim to be, but it ensures the cluster's
        #   configured tiers and UserBudgetDefaults still bite — an unlisted
        #   submitter hits the INTERACTIVE default cap and can't punch up to
        #   PRODUCTION just by skipping auth.
        # UNSPECIFIED (0) defaults to INTERACTIVE.
        band = request.priority_band or job_pb2.PRIORITY_BAND_INTERACTIVE
        if band == job_pb2.PRIORITY_BAND_PRODUCTION and self._auth.provider:
            authorize(AuthzAction.MANAGE_BUDGETS)
        else:
            with self._db.read_snapshot() as _snap:
                user_budget = reads.get_user_budget(_snap, job_id.user)
            max_band = user_budget.max_band if user_budget is not None else self._user_budget_defaults.max_band
            if band < max_band:
                raise ConnectError(
                    Code.PERMISSION_DENIED,
                    f"User {job_id.user} cannot submit {priority_band_name(band)} jobs "
                    f"(max band: {priority_band_name(max_band)}). "
                    f"Resubmit with `--priority {priority_band_name(max_band).lower()}` "
                    f"(e.g. `--priority batch`) to launch opportunistically, or ping @Helw150 "
                    f"if you believe your username ({job_id.user}) should have a higher band — "
                    f"either to be added to the researcher list or to confirm your username is "
                    f"registered correctly.",
                )

        # Elevated profiles (DOCKER_ACCESS, PRIVILEGED) are host-root-equivalent
        # and require the admin role. The check only runs when an auth provider is
        # configured; a trusted-loopback caller resolves to admin.
        if _profile_is_elevated(request.container_profile):
            if self._auth.provider:
                authorize(AuthzAction.SET_CONTAINER_PROFILE)
            logger.info(
                "Job %s using elevated container profile %s",
                job_id.to_wire(),
                job_pb2.ContainerProfile.Name(request.container_profile),
            )

        # DOCKER_ACCESS mounts the host docker socket, which only the docker
        # worker backend has. Reject it at submit on a cluster backend (k8s nodes
        # run containerd, no host socket) so the job never lands — the k8s
        # manifest builder would otherwise raise mid-reconcile and stall dispatch.
        if (
            resolve_container_profile(request.container_profile) == job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS
            and BackendCapability.WORKER_DAEMON not in self._controller.capabilities
        ):
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                "Container profile docker_access requires the docker worker backend (it mounts the "
                "host docker socket); this cluster's backend does not support it. Use a privileged "
                "profile with an in-pod runtime, or submit to a docker-worker cluster.",
            )

        # Cap the number of non-terminal tasks a single user may hold at once.
        # A burst of eval submissions once materialized enough tasks to OOM the
        # controller (#6411); reject up front any submission that would push the
        # user past the cap. Keyed on job_id.user, so a launcher that admits
        # tasks gradually stays under the cap as earlier tasks finish.
        incoming_tasks = int(request.replicas)
        if incoming_tasks > 0:
            with self._db.read_snapshot() as _snap:
                active_tasks = reads.count_active_tasks_for_user(_snap, job_id.user)
            if active_tasks + incoming_tasks > MAX_ACTIVE_TASKS_PER_USER:
                raise ConnectError(
                    Code.RESOURCE_EXHAUSTED,
                    f"User {job_id.user} has {active_tasks} active task(s); submitting "
                    f"{incoming_tasks} more would exceed the per-user cap of "
                    f"{MAX_ACTIVE_TASKS_PER_USER}. Wait for running tasks to finish, or "
                    f"structure the work as a launcher job that admits tasks gradually.",
                )

        # Reject submissions whose parent is absent or already terminated.
        # Absent parents can appear after a controller restart restores from a
        # checkpoint that did not capture the parent row; accepting the child
        # anyway would insert an orphan with `parent_job_id = NULL` and a
        # `depth` computed from the name path, which the dashboard `WHERE
        # depth = 1` query never surfaces.
        if job_id.parent:
            parent_state = _job_state(self._db, job_id.parent)
            if parent_state is None:
                raise ConnectError(
                    Code.FAILED_PRECONDITION,
                    f"Cannot submit job: parent job {job_id.parent} is absent from the database",
                )
            if parent_state in TERMINAL_JOB_STATES:
                raise ConnectError(
                    Code.FAILED_PRECONDITION,
                    f"Cannot submit job: parent job {job_id.parent} has terminated "
                    f"(state={job_pb2.JobState.Name(parent_state)})",
                )

        # Existence check + conditional cleanup run in one transaction so a
        # concurrent submitter cannot land a row between the read and the
        # cleanup write. The new job's ``submit_job`` still opens its own
        # transaction further down (we can't hold a write tx across the
        # drain wait below) — between the two txs another submitter can
        # race, so the INSERT tx re-checks existence before INSERTing to
        # avoid tripping the jobs.job_id PK. See the inner re-check at
        # the second ``with self._db.transaction()`` below.
        needs_drain = False
        with self._db.transaction() as cur:
            existing_state = reads.get_job_state(cur, job_id)
            if existing_state is not None:
                # A received handoff whose id already exists takes the federation
                # admission path (idempotent re-drive vs. genuine collision) before
                # any generic replace/keep policy applies.
                if request.HasField("federation"):
                    return self._admit_federated_resubmit(cur, job_id, request)
                policy = request.existing_job_policy
                if policy == job_pb2.EXISTING_JOB_POLICY_ERROR:
                    raise ConnectError(
                        Code.ALREADY_EXISTS,
                        f"Job {job_id} already exists (state={job_pb2.JobState.Name(existing_state)})",
                    )
                elif policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
                    if not is_job_finished(existing_state):
                        return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
                    # Job finished, replace it (KEEP only preserves running jobs).
                    # If worker-bound attempts haven't finalized yet (e.g. the
                    # task is terminal at the job level but its attempt is still
                    # pending a heartbeat), defer to the drain wait below.
                    needs_drain = needs_drain or self._replace_finished_job(cur, job_id)
                elif policy == job_pb2.EXISTING_JOB_POLICY_RECREATE:
                    if not is_job_finished(existing_state):
                        ops.job.cancel(
                            cur,
                            job_id=job_id,
                            reason="Replaced by new submission",
                            endpoints=self._endpoints,
                        )
                        # Cancel is a producer transition: attempts stay
                        # unfinished until the worker confirms termination.
                        # Defer remove_finished_job to a second tx after the
                        # drain wait so we don't destroy task_attempts rows
                        # whose finished_at_ms write the reconcile-observation
                        # path is still racing to land.
                        needs_drain = True
                    else:
                        needs_drain = needs_drain or self._replace_finished_job(cur, job_id)
                elif is_job_finished(existing_state):
                    # Default/UNSPECIFIED: replace finished jobs
                    logger.info(
                        "Replacing finished job %s (state=%s) with new submission",
                        job_id,
                        job_pb2.JobState.Name(existing_state),
                    )
                    needs_drain = needs_drain or self._replace_finished_job(cur, job_id)
                else:
                    raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists and is still running")

        if needs_drain:
            # Nudge the polling loop so workers see the cancelled tasks excluded
            # from their desired set on the next reconcile and auto-kill the
            # containers; the reconcile-observation path then stamps finished_at_ms. If
            # the terminal observation never lands, force-reap so a stuck worker can't
            # block the resubmit forever.
            self._controller.wake()
            if not self._wait_until_job_drained(job_id, _JOB_REPLACEMENT_DRAIN_WAIT):
                logger.warning(
                    "Job %s did not drain within %ss; force-reaping predecessor and proceeding",
                    job_id,
                    _JOB_REPLACEMENT_DRAIN_WAIT.to_seconds(),
                )
            with self._db.transaction() as cur:
                ops.job.remove_finished(cur, job_id)

        # Handle bundle_blob: upload to bundle store, then replace blob
        # with the resulting GCS path (preserving all other fields).
        if request.bundle_blob:
            # Validate bundle size
            bundle_size = len(request.bundle_blob)
            if bundle_size > MAX_BUNDLE_SIZE_BYTES:
                bundle_size_mb = bundle_size / (1024 * 1024)
                max_size_mb = MAX_BUNDLE_SIZE_BYTES / (1024 * 1024)
                raise ConnectError(
                    Code.INVALID_ARGUMENT,
                    f"Bundle size {bundle_size_mb:.1f}MB exceeds maximum {max_size_mb:.0f}MB",
                )

            bundle_id = self._bundle_store.write(request.bundle_blob)

            new_request = controller_pb2.Controller.LaunchJobRequest()
            new_request.CopyFrom(request)
            new_request.ClearField("bundle_blob")
            new_request.bundle_id = bundle_id
            request = new_request

        # Externalize large workdir files to the blob store so request_proto
        # (and every RunTaskRequest dispatch) stays small.
        large_files = {
            name: data
            for name, data in request.entrypoint.workdir_files.items()
            if len(data) > WORKDIR_FILE_OFFLOAD_THRESHOLD
        }
        if large_files:
            new_request = controller_pb2.Controller.LaunchJobRequest()
            new_request.CopyFrom(request)
            for name, data in large_files.items():
                blob_id = self._bundle_store.write(data)
                del new_request.entrypoint.workdir_files[name]
                new_request.entrypoint.workdir_file_refs[name] = blob_id
                logger.info("Externalized workdir file %s (%d bytes) as blob %s", name, len(data), blob_id[:12])
            request = new_request

        # Auto-inject device constraints from the resource spec.
        # Explicit user constraints for canonical keys (device-type,
        # device-variant, etc.) replace auto-generated ones.
        request = _inject_resource_constraints(request)

        # Reject TPU requests whose chip count doesn't match a single VM, or
        # whose device-variant alternatives mix incompatible VM shapes (e.g.
        # v6e-4 + v6e-8). Co-scheduling jobs onto a single-VM slice like v6e-8
        # would put two tenants on one indivisible VM.
        tpu_error = validate_tpu_request(request.resources, [Constraint.from_proto(c) for c in request.constraints])
        if tpu_error:
            raise ConnectError(Code.INVALID_ARGUMENT, tpu_error)

        # Resolve the routing directives (mutually exclusive): --backend pins a
        # local backend, --cluster hands the job to a federation peer. A job may
        # not pin both, and a --cluster pin must name a configured peer.
        replicas = request.replicas if request.HasField("coscheduling") else None
        constraints = [Constraint.from_proto(c) for c in request.constraints]
        directive = backend_directive(constraints)
        cluster_pin = cluster_directive(constraints)
        if directive is not None and cluster_pin is not None:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Job {job_id} pins both a local backend ({directive!r}) and a peer cluster "
                f"({cluster_pin!r}); these are mutually exclusive.",
            )
        if cluster_pin is not None and not self._controller.federation.has_peer(cluster_pin):
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Job {job_id} pins cluster {cluster_pin!r}, which is not a configured federation peer.",
            )

        # Compute local feasibility WITHOUT failing yet: a job no local backend
        # can host is not an error if a peer can take it. A job pinned to one
        # backend is checked only against that backend; an unpinned job is
        # feasible if any backend can host its shape. A backend without an
        # autoscaler (e.g. a cluster-view backend) can't prove infeasibility, so
        # its presence means we treat the job as feasible. For coscheduled jobs
        # this also verifies the replica count fits some group's num_vms.
        if directive is not None:
            pinned = self._controller.backends.get(directive)
            candidate_backends = [pinned] if pinned is not None else []
        else:
            candidate_backends = list(self._controller.backends.values())
        feasibility_errors: list[str] = []
        feasible = False
        for backend in candidate_backends:
            autoscaler = backend.autoscaler
            if autoscaler is None:
                feasible = True
                break
            error = autoscaler.job_feasibility(
                constraints=constraints,
                replicas=replicas,
                resources=request.resources,
            )
            if error is None:
                feasible = True
                break
            feasibility_errors.append(error)

        # Decide at submit whether this job runs locally or hands off to a peer.
        # A job this cluster received via handoff (federation field set) always
        # runs here — it is never re-federated — so it skips peer routing.
        is_received_handoff = request.HasField("federation")
        if is_received_handoff:
            routing = SubmitRouting()
        else:
            routing = self._controller.federation.route_submit(
                RoutingRequest(
                    constraints=constraints,
                    local_feasible=feasible,
                    cluster_pin=cluster_pin or "",
                )
            )

        if not routing.is_local:
            return self._hand_off_job(job_id, request, routing.peer_id)

        # Local (including a received handoff): only now is infeasibility fatal —
        # no peer could take it either.
        if not feasible and feasibility_errors:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Job {job_id} is unschedulable: {feasibility_errors[0]} (constraints: {constraints})",
            )

        # A received handoff runs as an ordinary local job; ``ops.job.submit``
        # records it as a RECEIVED federated_jobs row so FederationSync reports it
        # back to the requester and this controller writes its changelog.
        with self._db.transaction() as cur:
            # Re-check inside the same tx as the INSERT. Two LaunchJob
            # handlers can race past the earlier existence check (separate
            # transaction above) — almost always the same logical request
            # from a client that retried after blowing past its deadline.
            # SQLite serializes write transactions, so whichever handler
            # gets the lock first INSERTs and the loser sees the row here
            # and short-circuits, instead of tripping the jobs.job_id PK
            # (which would surface as INTERNAL — retryable, compounding the
            # storm). KEEP mirrors line 1126: return the existing handle as
            # success. Other policies surface ALREADY_EXISTS; the outer
            # block's RECREATE / replace-finished path already ran for any
            # row that was visible at that time, so any row showing up here
            # belongs to the racing submitter and is too late for us to
            # replace without re-running the whole flow.
            if reads.get_job_state(cur, job_id) is not None:
                if request.HasField("federation"):
                    return self._admit_federated_resubmit(cur, job_id, request)
                if request.existing_job_policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
                    return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
                raise ConnectError(
                    Code.ALREADY_EXISTS,
                    f"Job {job_id} already exists (concurrent submission)",
                )
            ops.job.submit(
                cur,
                job_id=job_id,
                request=request,
                ts=Timestamp.now(),
                run_template_cache=self._controller.run_template_cache,
            )
        self._controller.wake()

        with self._db.read_snapshot() as tx:
            num_tasks = tx.execute(select(func.count()).where(tasks_table.c.job_id == job_id)).scalar() or 0
        logger.info(f"Job {job_id} submitted with {num_tasks} task(s)")
        return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())

    def get_job_status(
        self,
        request: controller_pb2.Controller.GetJobStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetJobStatusResponse:
        """Get job-level status with aggregated task counts.

        Per-task detail (attempts, worker addresses) is NOT included — callers
        that need it should use ListTasks instead.  This keeps GetJobStatus
        cheap: one job row read + one GROUP BY query vs loading every task,
        attempt, and worker address.
        """
        with self._db.read_snapshot() as q:
            job = reads.get_job_detail(q, JobName.from_wire(request.job_id))
            if not job:
                raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")
            # Aggregate task counts via a single GROUP BY query.
            summaries = reads.task_summaries_for_jobs(q, {job.job_id})
            has_children = bool(reads.parent_ids_with_children(q, [job.job_id]))
            # A federated job's subtree lives on the peer; load the handle to surface
            # its handoff posture in the pending reason.
            handle = reads.federated_handle(q, job.job_id) if is_federated(job.cluster) else None
        summary = summaries.get(job.job_id)

        # Get scheduling diagnostics for pending jobs from cache
        # (populated each scheduling cycle by the controller). The autoscaler
        # hint dict is cached per evaluate() cycle (#4848), so the lookup here
        # is a single dict get — we only attach this job's hint, never the
        # full routing decision.
        peer_status = _peer_status(job.cluster, handle.handoff_state if handle else None, summary is not None)
        pending_reason = ""
        if job.state == job_pb2.JOB_STATE_PENDING and is_federated(job.cluster):
            pending_reason = _federated_pending_reason(job.cluster, peer_status)
        elif job.state == job_pb2.JOB_STATE_PENDING:
            sched_reason = self._controller.get_job_scheduling_diagnostics(job.job_id.to_wire())
            pending_reason = sched_reason or "Pending scheduler feedback"
            hint = self._get_autoscaler_pending_hints().get(job.job_id.to_wire())
            if hint is not None:
                scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
                pending_reason = f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

        resources = resource_spec_from_job_row(job)

        proto_job_status = job_pb2.JobStatus(
            job_id=job.job_id.to_wire(),
            state=job.state,
            error=job.error or "",
            exit_code=job.exit_code or 0,
            name=job.name,
            pending_reason=pending_reason,
            resources=resources,
            has_children=has_children,
            parent_job_id=job.parent_job_id.to_wire() if job.parent_job_id else "",
            backend_id=job.backend_id or "",
            cluster=job.cluster,
            peer_status=peer_status,
            **_job_status_counts(
                summary, job.job_id, pre_sync_task_count=job.num_tasks if is_federated(job.cluster) else 0
            ),
        )
        if job.started_at_ms:
            proto_job_status.started_at.CopyFrom(timestamp_to_proto(job.started_at_ms))
        if job.finished_at_ms:
            proto_job_status.finished_at.CopyFrom(timestamp_to_proto(job.finished_at_ms))
        if job.submitted_at_ms:
            proto_job_status.submitted_at.CopyFrom(timestamp_to_proto(job.submitted_at_ms))

        reconstructed_request = reconstruct_launch_job_request(job)
        return controller_pb2.Controller.GetJobStatusResponse(
            job=proto_job_status,
            request=redact_request_env_vars(reconstructed_request),
        )

    def get_job_state(
        self,
        request: controller_pb2.Controller.GetJobStateRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetJobStateResponse:
        """Lightweight batch job state query.

        Returns only the state enum for each requested job, avoiding the cost
        of loading tasks, attempts, and worker addresses.
        """
        wire_ids = list(request.job_ids)
        if not wire_ids:
            return controller_pb2.Controller.GetJobStateResponse()

        with self._db.read_snapshot() as tx:
            rows = tx.execute(
                select(jobs_table.c.job_id, jobs_table.c.state).where(
                    jobs_table.c.job_id.in_(bindparam("job_ids", expanding=True))
                ),
                {"job_ids": wire_ids},
            ).all()

        states = {row.job_id.to_wire(): int(row.state) for row in rows}
        return controller_pb2.Controller.GetJobStateResponse(states=states)

    def terminate_job(
        self,
        request: controller_pb2.Controller.TerminateJobRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        """Terminate a running job and all its children.

        Cascade termination is performed depth-first: all children are
        terminated before the parent. All tasks within each job are killed.
        """
        job_id = JobName.from_wire(request.job_id)
        state = _job_state(self._db, job_id)
        if state is None:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        self._authorize_job_owner(job_id)

        # A federated handle owns no local tasks — its subtree lives on the peer.
        # Route a versioned, idempotent cancel there; the next sync mirrors the
        # peer's terminal state (and eventually its tombstone) back.
        with self._db.read_snapshot() as snap:
            has_federated_handle = reads.federated_handle(snap, job_id) is not None
        if has_federated_handle:
            self._controller.federation.cancel_federated(job_id)
            return job_pb2.Empty()

        # cancel_job uses a recursive CTE to walk the full subtree in a single
        # transaction, so there is no need to recurse manually.
        with self._db.transaction() as cur:
            ops.job.cancel(
                cur,
                job_id=job_id,
                reason="Terminated by user",
                endpoints=self._endpoints,
            )
        # The next polling tick reconciles each affected worker; the
        # cancellation appears in the desired-set diff so the worker stops
        # the attempt within one tick rather than waiting on the next backoff.
        self._controller.wake()
        return job_pb2.Empty()

    def _job_to_proto(
        self,
        j: Any,
        task_summary: TaskJobSummary | None,
        autoscaler_pending_hints: dict[str, PendingHint],
        *,
        has_children: bool = False,
        handoff_state: int | None = None,
    ) -> job_pb2.JobStatus:
        """Convert a job row and its task summary into a JobStatus proto."""
        peer_status = _peer_status(j.cluster, handoff_state, task_summary is not None)
        pending_reason = j.error or ""
        if j.state == job_pb2.JOB_STATE_PENDING and is_federated(j.cluster):
            pending_reason = _federated_pending_reason(j.cluster, peer_status)
        elif j.state == job_pb2.JOB_STATE_PENDING:
            sched_reason = self._controller.get_job_scheduling_diagnostics(j.job_id.to_wire())
            pending_reason = sched_reason or "Pending scheduler feedback"
            hint = autoscaler_pending_hints.get(j.job_id.to_wire())
            if hint is not None:
                scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
                pending_reason = f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

        proto_job = job_pb2.JobStatus(
            job_id=j.job_id.to_wire(),
            state=j.state,
            error=j.error or "",
            exit_code=j.exit_code or 0,
            name=j.name,
            pending_reason=pending_reason,
            has_children=has_children,
            backend_id=j.backend_id or "",
            cluster=j.cluster,
            peer_status=peer_status,
            **_job_status_counts(
                task_summary, j.job_id, pre_sync_task_count=j.num_tasks if is_federated(j.cluster) else 0
            ),
        )
        if j.started_at_ms:
            proto_job.started_at.CopyFrom(timestamp_to_proto(j.started_at_ms))
        if j.finished_at_ms:
            proto_job.finished_at.CopyFrom(timestamp_to_proto(j.finished_at_ms))
        if j.submitted_at_ms:
            proto_job.submitted_at.CopyFrom(timestamp_to_proto(j.submitted_at_ms))
        return proto_job

    def _jobs_to_protos(
        self,
        jobs: list,
        task_summaries: dict[JobName, TaskJobSummary],
        autoscaler_pending_hints: dict[str, PendingHint],
        has_children: set[JobName] | None = None,
        handoff_states: dict[JobName, int] | None = None,
    ) -> list[job_pb2.JobStatus]:
        child_parent_ids = has_children or set()
        handoffs = handoff_states or {}
        return [
            self._job_to_proto(
                j,
                task_summaries.get(j.job_id),
                autoscaler_pending_hints,
                has_children=j.job_id in child_parent_ids,
                handoff_state=handoffs.get(j.job_id),
            )
            for j in jobs
        ]

    def list_jobs(
        self,
        request: controller_pb2.Controller.ListJobsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListJobsResponse:
        """List jobs with filtering, sorting, and pagination.

        Served directly from indexed SQL via ``_query_jobs``. Per-page task
        summaries and parent->child flags are looked up against the same read
        snapshot so the whole RPC observes a single transactionally-consistent
        view.
        """
        query = _query_from_list_jobs_request(request)

        state_ids = _resolve_state_filter(query.state_filter)
        if state_ids is None:
            return controller_pb2.Controller.ListJobsResponse(jobs=[], total_count=0, has_more=False)

        with self._db.read_snapshot() as q:
            page, total_count = _query_jobs(q, query, state_ids)
            page_ids = [j.job_id for j in page]
            summaries = reads.task_summaries_for_jobs(q, set(page_ids)) if page_ids else {}
            children = reads.parent_ids_with_children(q, page_ids) if page_ids else set()
            # Batch-load the SENT handoff state for the federated jobs on this page
            # so each JobStatus carries its handoff posture without a per-job read.
            federated_ids = [j.job_id for j in page if is_federated(j.cluster)]
            handoffs = reads.handoff_states(q, federated_ids)

        has_pending = any(j.state == job_pb2.JOB_STATE_PENDING for j in page)
        autoscaler_pending_hints = self._get_autoscaler_pending_hints() if has_pending else {}
        all_jobs = self._jobs_to_protos(
            page, summaries, autoscaler_pending_hints, has_children=children, handoff_states=handoffs
        )
        limit = query.limit
        offset = query.offset
        has_more = limit > 0 and offset + limit < total_count
        return controller_pb2.Controller.ListJobsResponse(
            jobs=all_jobs,
            total_count=total_count,
            has_more=has_more,
        )

    # --- Task Management ---

    def get_task_status(
        self,
        request: controller_pb2.Controller.GetTaskStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetTaskStatusResponse:
        """Get status of a specific task."""
        try:
            task_id = JobName.from_wire(request.task_id)
            task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        task = _read_task_with_attempts(self._db, task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {task_id} not found")
        worker_address = ""
        twid = _task_worker_id(task)
        if twid:
            worker_address = _worker_address(self._db, twid) or ""

        proto = task_to_proto(task, worker_address=worker_address)

        # Resource history / latest usage now comes from the ``iris.task``
        # stats namespace; the controller only attaches the static job
        # resource limits here.
        job_resources = None
        with self._db.read_snapshot() as tx:
            jc_row = tx.execute(
                select(
                    job_config_table.c.res_cpu_millicores,
                    job_config_table.c.res_memory_bytes,
                    job_config_table.c.res_disk_bytes,
                    job_config_table.c.res_device_json,
                ).where(job_config_table.c.job_id == task.job_id)
            ).first()
        if jc_row is not None:
            if jc_row.res_cpu_millicores or jc_row.res_memory_bytes or jc_row.res_disk_bytes or jc_row.res_device_json:
                job_resources = resource_spec_from_scalars(
                    jc_row.res_cpu_millicores, jc_row.res_memory_bytes, jc_row.res_disk_bytes, jc_row.res_device_json
                )

        return controller_pb2.Controller.GetTaskStatusResponse(task=proto, job_resources=job_resources)

    def list_tasks(
        self,
        request: controller_pb2.Controller.ListTasksRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListTasksResponse:
        """List tasks for a job."""
        if not request.job_id:
            raise ConnectError(Code.INVALID_ARGUMENT, "job_id is required")
        job_id = JobName.from_wire(request.job_id)
        with self._db.read_snapshot() as tx:
            tasks = _tasks_for_listing(tx, job_id=job_id)

        task_statuses = []
        for task in tasks:
            # task.current_worker_address is denormalized into the tasks row at
            # assignment time and is identical to the workers.address for active tasks.
            # task_to_proto uses it as a fallback when worker_address="" so no
            # extra workers-table lookup is needed here.
            proto_task_status = task_to_proto(task)

            # Don't add scheduling diagnostics in list view - too expensive
            # Users should check job detail page for scheduling diagnostics
            if task.state == job_pb2.TASK_STATE_PENDING:
                proto_task_status.can_be_scheduled = task_row_can_be_scheduled(task)

            task_statuses.append(proto_task_status)

        return controller_pb2.Controller.ListTasksResponse(tasks=task_statuses)

    def kick_tasks(
        self,
        request: controller_pb2.Controller.KickTasksRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.KickTasksResponse:
        """Force task attempts into a terminal state out-of-band (emergency override).

        Validates each target against the current snapshot and queues the accepted
        ones on the controller for the next control tick to apply. Returns one
        ``KickResult`` per resolved task reporting whether it was queued.
        """
        kind = _KICK_KIND_BY_STATE.get(request.desired_state)
        if kind is None:
            allowed = ", ".join(task_state_friendly(state) for state in _KICK_KIND_BY_STATE)
            raise ConnectError(Code.INVALID_ARGUMENT, f"desired_state must be one of: {allowed}")
        if not request.targets:
            raise ConnectError(Code.INVALID_ARGUMENT, "at least one target is required")

        reason = request.reason or f"Kicked to {task_state_friendly(request.desired_state)} by operator"

        results: list[controller_pb2.Controller.KickResult] = []
        kicks: list[PendingKick] = []
        with self._db.read_snapshot() as tx:
            for target in request.targets:
                self._resolve_kick_target(tx, target, kind, reason, kicks, results)

        self._controller.request_task_kicks(kicks)
        return controller_pb2.Controller.KickTasksResponse(results=results)

    def _resolve_kick_target(
        self,
        tx: Tx,
        target: str,
        kind: TerminalKind,
        reason: str,
        kicks: list[PendingKick],
        results: list[controller_pb2.Controller.KickResult],
    ) -> None:
        """Validate one kick target, appending its queued kicks and result rows.

        A task or task-attempt id targets a single task; a job id expands to the
        job's active tasks. Only tasks running on a worker (ASSIGNED / BUILDING /
        RUNNING) can be kicked; anything else is rejected with a reason.
        """

        def reject(detail: str, *, task_id: str = "") -> None:
            results.append(
                controller_pb2.Controller.KickResult(target=target, task_id=task_id, queued=False, detail=detail)
            )

        try:
            task_attempt = TaskAttempt.from_wire(target)
        except ValueError as exc:
            reject(str(exc))
            return

        name = task_attempt.task_id
        self._authorize_job_owner(name)

        if name.is_task:
            detail = reads.get_task_detail(tx, name)
            if detail is None:
                reject("task not found")
                return
            if task_attempt.attempt_id is not None and task_attempt.attempt_id != detail.current_attempt_id:
                reject(
                    f"attempt {task_attempt.attempt_id} is not current (current is {detail.current_attempt_id})",
                    task_id=name.to_wire(),
                )
                return
            if detail.state not in ACTIVE_TASK_STATES:
                reject(f"task is {task_state_friendly(detail.state)}, not running on a worker", task_id=name.to_wire())
                return
            kicks.append(PendingKick(task_id=name, attempt_id=task_attempt.attempt_id, kind=kind, reason=reason))
            results.append(controller_pb2.Controller.KickResult(target=target, task_id=name.to_wire(), queued=True))
            return

        # Job target: expand to its active tasks.
        if task_attempt.attempt_id is not None:
            reject("a job target cannot carry an ':attempt' suffix")
            return
        if reads.get_job_state(tx, name) is None:
            reject("job not found")
            return
        active = reads.list_active_tasks(tx, reads.TaskScope(job_id=name), states=ACTIVE_TASK_STATES)
        if not active:
            reject("job has no tasks running on a worker", task_id=name.to_wire())
            return
        for row in active:
            kicks.append(PendingKick(task_id=row.task_id, attempt_id=None, kind=kind, reason=reason))
            results.append(
                controller_pb2.Controller.KickResult(target=target, task_id=row.task_id.to_wire(), queued=True)
            )

    # --- Worker Management ---

    def register(
        self,
        request: controller_pb2.Controller.RegisterRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RegisterResponse:
        """One-shot worker registration. Returns worker_id.

        Worker registers once, then waits for heartbeats from the controller.
        """
        if self._auth.provider is not None:
            authorize(AuthzAction.ACT_AS_WORKER)

        if not request.worker_id:
            logger.error("Worker at %s registered without worker_id", request.address)
            return controller_pb2.Controller.RegisterResponse(
                worker_id="",
                accepted=False,
            )
        worker_id = WorkerId(request.worker_id)

        # Route the worker into the liveness tracker and attributes projection owned
        # by the backend that owns its scale group; a worker never registers into a
        # k8s scale group.
        backend = self._backend_for_id(self._controller.backend_id_for_scale_group(request.scale_group))
        health = backend.health
        worker_attrs = backend.worker_attrs
        assert health is not None, f"worker {worker_id} registered into a scale group with no liveness tracker"
        assert worker_attrs is not None, f"worker {worker_id} registered into a scale group with no attrs projection"
        with self._db.transaction() as cur:
            ops.worker.register(
                cur,
                worker_id=worker_id,
                address=request.address,
                metadata=request.metadata,
                ts=Timestamp.now(),
                health=health,
                worker_attrs=worker_attrs,
                slice_id=request.slice_id,
                scale_group=request.scale_group,
            )
        self._request_recycled_address_eviction(worker_id, request.address)
        logger.info("Worker registered: %s at %s", worker_id, request.address)
        return controller_pb2.Controller.RegisterResponse(
            worker_id=str(worker_id),
            accepted=True,
        )

    def _request_recycled_address_eviction(self, worker_id: WorkerId, address: str) -> None:
        """Hand any stale prior owner of ``address`` to the controller for teardown.

        Detects a recycled internal IP (see :func:`reads.worker_ids_at_address`)
        and defers the reap to :meth:`Controller.request_worker_eviction`.
        """
        with self._db.read_snapshot() as snap:
            stale = reads.worker_ids_at_address(snap, address, exclude=worker_id)
        if not stale:
            return
        logger.warning(
            "Worker %s registered at %s held by %d stale row(s) (recycled IP); evicting: %s",
            worker_id,
            address,
            len(stale),
            [str(wid) for wid in stale],
        )
        self._controller.request_worker_eviction(stale)

    def list_workers(
        self,
        request: controller_pb2.Controller.ListWorkersRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListWorkersResponse:
        """List workers with their running task counts.

        Served directly from the workers table (cluster size is in the low
        thousands at most), with liveness queried from the backends' trackers
        (unioned by the controller) and a single per-page running-task lookup.
        ``query.limit == 0`` disables paging (preserves CLI callers that fetch the
        whole roster); ``limit > 0`` is clamped to ``MAX_LIST_WORKERS_LIMIT``.
        """
        if BackendCapability.WORKER_DAEMON not in self._controller.capabilities:
            return controller_pb2.Controller.ListWorkersResponse()

        query = controller_pb2.Controller.WorkerQuery()
        if request.HasField("query"):
            query.CopyFrom(request.query)

        workers_all = _worker_roster(self._db)
        all_liveness = self._controller.all_liveness()
        liveness_by_id = {w.worker_id: all_liveness.get(w.worker_id, WorkerLiveness()) for w, _attrs in workers_all}
        if query.backend_id:
            backend_filter = query.backend_id
            workers_all = [
                (w, attrs)
                for w, attrs in workers_all
                if self._controller.backend_id_for_scale_group(str(w.scale_group or "")) == backend_filter
            ]
        filtered = _filter_and_sort_workers(workers_all, liveness_by_id, query)
        total_count = len(filtered)

        offset = max(query.offset, 0)
        limit = max(query.limit, 0)
        if limit > MAX_LIST_WORKERS_LIMIT:
            limit = MAX_LIST_WORKERS_LIMIT
        if limit > 0:
            page_rows = filtered[offset : offset + limit]
            has_more = offset + limit < total_count
        else:
            page_rows = filtered[offset:] if offset else filtered
            has_more = False

        if page_rows:
            with self._db.read_snapshot() as tx:
                running = reads.running_tasks_by_worker(tx, {w.worker_id for w, _attrs in page_rows})
        else:
            running = {}
        workers = []
        for worker, attrs in page_rows:
            liveness = liveness_by_id[worker.worker_id]
            workers.append(
                controller_pb2.Controller.WorkerHealthStatus(
                    worker_id=worker.worker_id,
                    healthy=liveness.healthy,
                    consecutive_failures=liveness.consecutive_failures,
                    last_heartbeat=timestamp_to_proto(Timestamp.from_ms(liveness.last_heartbeat_ms)),
                    running_job_ids=[task_id.to_wire() for task_id in running.get(worker.worker_id, set())],
                    address=worker.address,
                    metadata=worker_metadata_to_proto(worker, attrs),
                    status_message=worker_status_message(liveness),
                    backend_id=self._controller.backend_id_for_scale_group(str(worker.scale_group or "")),
                    scale_group=str(worker.scale_group or ""),
                )
            )
        return controller_pb2.Controller.ListWorkersResponse(
            workers=workers,
            total_count=total_count,
            has_more=has_more,
        )

    # --- Endpoint Management (compatibility surface) ---
    #
    # These RPCs forward to the leased EndpointService backend so clients that
    # call the old surface keep working; clients that want to renew call
    # EndpointService directly to learn their lease.

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        """Register a service endpoint (forwards to EndpointService).

        The lease is dropped from the response so this surface stays
        wire-identical to its lease-less callers, which do not renew.
        """
        response = self._endpoint_service.register_endpoint(request, ctx)
        return controller_pb2.Controller.RegisterEndpointResponse(endpoint_id=response.endpoint_id)

    def unregister_endpoint(
        self,
        request: controller_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        """Unregister a service endpoint (forwards to EndpointService). Idempotent."""
        return self._endpoint_service.unregister_endpoint(request, ctx)

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        """List endpoints by name prefix (forwards to EndpointService)."""
        return self._endpoint_service.list_endpoints(request, ctx)

    @property
    def provider(self) -> TaskBackend:
        """The live execution backend (read-only handle for dashboard descriptors)."""
        return self._controller.provider

    @property
    def backends(self) -> dict[str, TaskBackend]:
        """The controller's full backend collection (for the union capabilities descriptor)."""
        return self._controller.backends

    def _backend_for_id(self, backend_id: str) -> TaskBackend:
        """Resolve a backend by id for per-task/-worker dispatch (profile, exec,
        process status), falling back to the representative backend when the id is
        empty or unknown — the single-backend case and any pre-routing rows."""
        return self._controller.backends.get(backend_id) or self._controller.provider

    def _federated_handle_for_task(self, task_id: JobName) -> reads.FederatedHandle | None:
        """The SENT federated handle owning ``task_id``'s root job, or ``None`` if
        that root job runs locally."""
        with self._db.read_snapshot() as snap:
            return reads.federated_handle(snap, task_id.root_job)

    @property
    def endpoint_service(self) -> EndpointServiceImpl:
        """The leased endpoint registry these RPCs delegate to (shared with the dashboard)."""
        return self._endpoint_service

    # --- Autoscaler ---

    def get_autoscaler_status(
        self,
        request: controller_pb2.Controller.GetAutoscalerStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetAutoscalerStatusResponse:
        """Get autoscaler status, merged across every backend's autoscaler.

        When ``request.backend_id`` is set, restricts the view to that one
        backend's autoscaler; empty merges all.
        """
        if BackendCapability.IRIS_AUTOSCALER not in self._controller.capabilities:
            return controller_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())

        status = self._merge_autoscaler_status(only_backend_id=request.backend_id)
        return controller_pb2.Controller.GetAutoscalerStatusResponse(status=status)

    def _merge_autoscaler_status(self, only_backend_id: str = "") -> vm_pb2.AutoscalerStatus:
        """Merge each backend's authored autoscaler status into one.

        Merges every backend by default; ``only_backend_id`` restricts the view to
        a single backend. Each backend authors its own status — groups already
        tagged with its backend_id and every VM overlaid with usability/running-task
        counts — so this only concatenates; a backend with no autoscaler authors an
        empty status that contributes nothing. Each backend owns a disjoint set of
        scale groups (the single scale-group->backend key space), so group-keyed
        fields (``current_demand``, ``recent_actions``) need no further
        disambiguation. ``recent_actions`` are re-sorted newest-first and capped;
        each backend's ``last_routing_decision`` folds into one merged decision
        (disjoint groups, so the per-group fields concatenate).
        """
        merged = vm_pb2.AutoscalerStatus()
        last_evaluation = 0
        for backend_id, backend in self._controller.backends.items():
            if only_backend_id and backend_id != only_backend_id:
                continue
            sub = backend.autoscaler_status()
            merged.groups.extend(sub.groups)
            for key, value in sub.current_demand.items():
                merged.current_demand[key] = value
            merged.recent_actions.extend(sub.recent_actions)
            last_evaluation = max(last_evaluation, sub.last_evaluation.epoch_ms)
            if sub.HasField("last_routing_decision"):
                _accumulate_routing_decision(merged.last_routing_decision, sub.last_routing_decision)
        merged.recent_actions.sort(key=lambda action: action.timestamp.epoch_ms, reverse=True)
        del merged.recent_actions[_MERGED_AUTOSCALER_ACTIONS:]
        if last_evaluation:
            merged.last_evaluation.epoch_ms = last_evaluation
        return merged

    # --- Kubernetes Cluster Status ---

    def get_kubernetes_cluster_status(
        self,
        request: controller_pb2.Controller.GetKubernetesClusterStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Get Kubernetes cluster status: node counts, capacity, and recent pod statuses.

        Routes to the ``CLUSTER_VIEW`` backend named by ``request.backend_id``, or
        the sole such backend if there is exactly one; raises ``INVALID_ARGUMENT``
        when the choice is ambiguous.
        """
        cluster_view_backends = [
            (bid, backend)
            for bid, backend in sorted(self._controller.backends.items())
            if BackendCapability.CLUSTER_VIEW in backend.capabilities
        ]

        if request.backend_id:
            for bid, backend in cluster_view_backends:
                if bid == request.backend_id:
                    return backend.status().kubernetes
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Backend {request.backend_id!r} does not exist or has no cluster view",
            )

        if len(cluster_view_backends) > 1:
            ids = ", ".join(bid for bid, _ in cluster_view_backends)
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Multiple cluster-view backends ({ids}); specify backend_id in the request",
            )

        if cluster_view_backends:
            return cluster_view_backends[0][1].status().kubernetes

        return controller_pb2.Controller.GetKubernetesClusterStatusResponse()

    # --- VM Logs ---

    # --- Profiling ---

    def profile_task(
        self,
        request: job_pb2.ProfileTaskRequest,
        ctx: RequestContext,
    ) -> job_pb2.ProfileTaskResponse:
        """Dashboard-facing on-demand profile dispatch.

        Behaviour by target:
          /system/controller (also /system/process — CLI default)
            - Capture this controller process via profile_local_process,
              write one IrisProfile row (source='/system/controller',
              vm_id='controller-self', attempt_id=None, trigger='on_demand'),
              return bytes inline.
          /system/worker/<id>
            - Forward as /system/process to the named worker via
              WorkerService.ProfileTask. Worker writes the row with
              source='/system/worker/<id>'.
          /job/.../task/N[:attempt_id]
            - Resolve task and worker; delegate to provider.profile_task.
              Worker-based: forwards to worker; worker writes IrisProfile
              (all types), returns bytes. K8s: K8sTaskProvider captures via
              kubectl exec, writes IrisProfile (all types), returns bytes.
          Anything else
            - INVALID_ARGUMENT.
        """
        if not request.HasField("profile_type"):
            raise ConnectError(Code.INVALID_ARGUMENT, "profile_type is required")

        # /system/controller (or its alias /system/process from the CLI): capture
        # this controller process itself.
        if request.target in ("/system/controller", "/system/process"):
            try:
                duration = request.duration_seconds or 10
                data = profile_local_process(duration, request.profile_type)
                if self._profile_table is not None:
                    self._profile_table.write(
                        [
                            build_profile_row(
                                source="/system/controller",
                                attempt_id=None,
                                vm_id="controller-self",
                                duration_seconds=duration,
                                profile_type=request.profile_type,
                                profile_data=data,
                            )
                        ]
                    )
                return job_pb2.ProfileTaskResponse(profile_data=data)
            except Exception as e:
                return job_pb2.ProfileTaskResponse(error=str(e))

        # /system/worker/<worker_id>: proxy profile to the worker's own process
        worker_id_str = _parse_worker_target(request.target)
        if worker_id_str is not None:
            worker = _read_worker(self._db, WorkerId(worker_id_str))
            if not worker:
                raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id_str} not found")
            if not self._controller.liveness_for_worker(worker.worker_id).healthy:
                raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id_str} is unavailable")
            forwarded = job_pb2.ProfileTaskRequest(
                target="/system/process",
                duration_seconds=request.duration_seconds,
                profile_type=request.profile_type,
            )
            timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
            worker_target = TaskTarget(
                task_id="",
                attempt_id=0,
                worker_id=worker.worker_id,
                address=worker.address,
            )
            worker_backend = self._backend_for_id(
                self._controller.backend_id_for_scale_group(str(worker.scale_group or ""))
            )
            resp = worker_backend.profile_task(worker_target, forwarded, timeout_ms)
            return job_pb2.ProfileTaskResponse(
                profile_data=resp.profile_data,
                error=resp.error,
            )

        # Task target: parse optional :attempt_id, validate, proxy to worker
        try:
            target = TaskAttempt.from_wire(request.target)
            target.task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        task = _read_task_with_attempts(self._db, target.task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.target} not found")

        # A federated task's subtree runs on a peer — it has no local worker or
        # attempt rows. Proxy the profile through the peer controller (which does
        # its own task->worker resolution) before the local resolution below, so
        # it is never dispatched to _backend_for_id's local fallback.
        handle = self._federated_handle_for_task(target.task_id)
        if handle is not None:
            return self._controller.federation.proxy_profile(
                peer_id=handle.peer_id,
                request=request,
            )

        attempt_id = target.attempt_id if target.attempt_id is not None else task.current_attempt_id
        task_worker_id = _task_worker_id(task)
        if not task_worker_id:
            # A cluster backend (K8s) chooses the node itself: route by task, no worker.
            if BackendCapability.CLUSTER_VIEW not in self._controller.capabilities:
                raise ConnectError(Code.FAILED_PRECONDITION, f"Task {request.target} not yet assigned to a worker")
            task_target = TaskTarget(
                task_id=task.task_id.to_wire(),
                attempt_id=attempt_id,
                worker_id=None,
                address=None,
            )
        else:
            worker = _read_worker(self._db, task_worker_id)
            if not worker or not self._controller.liveness_for_worker(task_worker_id).healthy:
                raise ConnectError(Code.UNAVAILABLE, f"Worker {task_worker_id} is unavailable")
            task_target = TaskTarget(
                task_id=task.task_id.to_wire(),
                attempt_id=attempt_id,
                worker_id=task_worker_id,
                address=worker.address,
            )

        timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
        resp = self._backend_for_id(str(task.backend_id or "")).profile_task(task_target, request, timeout_ms)
        return job_pb2.ProfileTaskResponse(
            profile_data=resp.profile_data,
            error=resp.error,
        )

    def list_users(
        self,
        request: controller_pb2.Controller.ListUsersRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListUsersResponse:
        """Return live per-user aggregate counts for the dashboard."""
        del request, ctx
        users = sorted(
            _live_user_stats(self._db),
            key=lambda entry: (
                -_active_job_count(entry.job_state_counts),
                -(entry.task_state_counts.get(job_pb2.TASK_STATE_RUNNING, 0)),
                entry.user,
            ),
        )
        return controller_pb2.Controller.ListUsersResponse(
            users=[
                controller_pb2.Controller.UserSummary(
                    user=entry.user,
                    task_state_counts=_task_state_counts_for_summary(entry.task_state_counts),
                    job_state_counts=_job_state_counts_for_summary(entry.job_state_counts),
                )
                for entry in users
            ]
        )

    # --- Worker Detail ---

    def get_worker_status(
        self,
        request: controller_pb2.Controller.GetWorkerStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetWorkerStatusResponse:
        """Return detail for a single worker, keyed by worker ID.

        Workers and VMs are independent: the worker detail page shows only
        worker state (health, tasks, logs). VM status lives on the Autoscaler
        tab.
        """
        if BackendCapability.WORKER_DAEMON not in self._controller.capabilities:
            raise ConnectError(Code.UNIMPLEMENTED, "Direct provider mode: no workers")
        if not request.id:
            raise ConnectError(Code.INVALID_ARGUMENT, "id is required")

        detail = _read_worker_detail(self._db, WorkerId(str(request.id)))
        if not detail:
            raise ConnectError(Code.NOT_FOUND, f"No worker found for '{request.id}'")

        worker = detail.worker
        liveness = self._controller.liveness_for_worker(worker.worker_id)
        scale_group = str(worker.scale_group or "")
        worker_health = controller_pb2.Controller.WorkerHealthStatus(
            worker_id=worker.worker_id,
            healthy=liveness.healthy,
            consecutive_failures=liveness.consecutive_failures,
            last_heartbeat=timestamp_to_proto(Timestamp.from_ms(liveness.last_heartbeat_ms)),
            running_job_ids=[tid.to_wire() for tid in detail.running_tasks],
            address=worker.address,
            metadata=worker_metadata_to_proto(worker, detail.attributes),
            status_message=worker_status_message(liveness),
            scale_group=scale_group,
            backend_id=self._controller.backend_id_for_scale_group(scale_group),
        )

        # Worker daemon logs are NOT inlined here — when the worker is
        # unreachable the LogService proxy blocks for its full timeout
        # (~10s) and stalls the worker page render. The dashboard fetches
        # them in parallel via LogService.FetchLogs with
        # source=/system/worker/<worker_id>.
        recent_attempts = _attempts_for_worker(self._db, worker.worker_id, limit=50)

        resp = controller_pb2.Controller.GetWorkerStatusResponse(
            recent_attempts=recent_attempts,
        )
        resp.worker.CopyFrom(worker_health)
        return resp

    def begin_checkpoint(
        self,
        request: controller_pb2.Controller.BeginCheckpointRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.BeginCheckpointResponse:
        path, result = self._controller.begin_checkpoint()
        resp = controller_pb2.Controller.BeginCheckpointResponse(
            checkpoint_path=path,
            job_count=result.job_count,
            task_count=result.task_count,
            worker_count=result.worker_count,
        )
        resp.created_at.CopyFrom(timestamp_to_proto(result.created_at))
        return resp

    def get_process_status(
        self,
        request: job_pb2.GetProcessStatusRequest,
        ctx: Any,
    ) -> job_pb2.GetProcessStatusResponse:
        """Return process info (no logs — use FetchLogs instead).

        Target routing (same convention as ProfileTask):
        - empty or /system/process: the controller process itself
        - /system/worker/<worker_id>: proxy to a specific worker
        """
        target = request.target
        if not target or target == "/system/process":
            return get_process_status(self._timer)

        # Parse /system/worker/<worker_id>
        worker_id = _parse_worker_target(target)
        if worker_id is None:
            raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid target: {target}")

        worker = _read_worker(self._db, WorkerId(worker_id))
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
        if not self._controller.liveness_for_worker(worker.worker_id).healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")

        process_target = TaskTarget(
            task_id="",
            attempt_id=0,
            worker_id=WorkerId(worker_id),
            address=worker.address,
        )
        try:
            worker_backend = self._backend_for_id(
                self._controller.backend_id_for_scale_group(str(worker.scale_group or ""))
            )
            return worker_backend.get_process_status(process_target, request)
        except ProviderError as exc:
            raise ConnectError(Code.UNAVAILABLE, str(exc)) from exc

    # ── Auth RPCs ────────────────────────────────────────────────────────

    def get_auth_info(
        self,
        request: job_pb2.GetAuthInfoRequest,
        ctx: Any,
    ) -> job_pb2.GetAuthInfoResponse:
        return job_pb2.GetAuthInfoResponse(
            provider=self._auth.provider or "",
            gcp_project_id=self._auth.gcp_project_id or "",
        )

    def login(
        self,
        request: job_pb2.LoginRequest,
        ctx: Any,
    ) -> job_pb2.LoginResponse:
        if not self._auth.login_verifier:
            raise ConnectError(Code.UNIMPLEMENTED, "Login not available (no identity provider configured)")
        if not self._auth.jwt_manager:
            raise ConnectError(Code.INTERNAL, "JWT manager not configured")

        try:
            login_identity = self._auth.login_verifier.verify(request.identity_token)
        except ValueError as exc:
            logger.info("Login verification failed: %s", exc)
            raise ConnectError(Code.UNAUTHENTICATED, "Identity verification failed") from exc

        username = login_identity.user_id
        if username.startswith("system:"):
            raise ConnectError(Code.PERMISSION_DENIED, "Reserved username prefix")

        now = Timestamp.now()
        with self._db.transaction() as _tx:
            writes.ensure_user(_tx, username, now)
            role = reads.get_user_role(_tx, username)

        # Revoke old login keys and propagate to in-memory revocation set
        revoked_ids = revoke_login_keys_for_user(self._db, username, now)
        for jti in revoked_ids:
            self._auth.jwt_manager.revoke(jti)

        key_id = f"iris_k_{secrets.token_urlsafe(8)}"
        expires_at = Timestamp.from_ms(now.epoch_ms() + DEFAULT_JWT_TTL_SECONDS * 1000)
        create_api_key(
            self._db,
            key_id=key_id,
            key_prefix="jwt",
            user_id=username,
            name=f"login-{now.epoch_ms()}",
            now=now,
            expires_at=expires_at,
        )

        jwt_token = self._auth.jwt_manager.create_token(username, role, key_id)
        logger.info(
            "Login: user=%s, role=%s, new_key=%s, revoked=%d old login keys", username, role, key_id, len(revoked_ids)
        )
        return job_pb2.LoginResponse(token=jwt_token, key_id=key_id, user_id=username)

    def mint_endpoint_token(
        self,
        request: controller_pb2.Controller.MintEndpointTokenRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.MintEndpointTokenResponse:
        """Mint a scoped bearer token for one endpoint's /proxy path.

        Authorized to the endpoint's owning user (the registering task's owner)
        or an admin. The token carries no RPC authority (see
        ``authorize_method``); it is bound to the endpoint's canonical wire name
        with an ``exp`` deadline, and is revocable via its ``jti`` like any key.
        """
        if not self._auth.jwt_manager:
            raise ConnectError(Code.INTERNAL, "JWT manager not configured")

        row = self._endpoint_service.resolve_task_endpoint(request.endpoint_name)
        if row is None:
            raise ConnectError(Code.NOT_FOUND, f"No endpoint '{request.endpoint_name}'")
        # Owner (or admin) only; skipped in null-auth mode like _authorize_job_owner.
        if self._auth.provider:
            authorize_resource_owner(row.task_id.user)

        if request.HasField("ttl"):
            ttl = int(duration_from_proto(request.ttl).to_seconds())
            ttl = max(1, min(ttl, MAX_ENDPOINT_TOKEN_TTL_SECONDS))
        else:
            ttl = DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS

        now = Timestamp.now()
        expires_at = Timestamp.from_ms(now.epoch_ms() + ttl * 1000)
        key_id = f"iris_ket_{secrets.token_urlsafe(8)}"
        # Audit row under the owning user so the token surfaces in `iris key list`
        # and revoking its jti kills it.
        create_api_key(
            self._db,
            key_id=key_id,
            key_prefix="ep",
            user_id=row.task_id.user,
            name=f"endpoint-token-{row.name}",
            now=now,
            expires_at=expires_at,
        )
        token = self._auth.jwt_manager.create_endpoint_token(row.name, key_id, ttl_seconds=ttl)
        return controller_pb2.Controller.MintEndpointTokenResponse(
            token=token,
            expires_at=timestamp_to_proto(expires_at),
        )

    def create_api_key(
        self,
        request: job_pb2.CreateApiKeyRequest,
        ctx: Any,
    ) -> job_pb2.CreateApiKeyResponse:
        if not self._auth.jwt_manager:
            raise ConnectError(Code.INTERNAL, "JWT manager not configured")

        identity = require_identity()
        target_user = request.user_id or identity.user_id
        if target_user != identity.user_id:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)

        now = Timestamp.now()
        with self._db.transaction() as _tx:
            writes.ensure_user(_tx, target_user, now)
            role = reads.get_user_role(_tx, target_user)

        key_id = f"iris_k_{secrets.token_urlsafe(8)}"
        ttl = request.ttl_ms // 1000 if request.ttl_ms > 0 else DEFAULT_JWT_TTL_SECONDS
        # Always persist the actual JWT expiry so the DB and token agree.
        expires_at = Timestamp.from_ms(now.epoch_ms() + ttl * 1000)

        create_api_key(
            self._db,
            key_id=key_id,
            key_prefix="jwt",
            user_id=target_user,
            name=request.name or f"key-{now.epoch_ms()}",
            now=now,
            expires_at=expires_at,
        )

        jwt_token = self._auth.jwt_manager.create_token(target_user, role, key_id, ttl_seconds=ttl)
        # Use key_id prefix (not JWT prefix — all HS256 JWTs share the same header)
        return job_pb2.CreateApiKeyResponse(key_id=key_id, token=jwt_token, key_prefix=key_id[:8])

    def revoke_api_key(
        self,
        request: job_pb2.RevokeApiKeyRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        identity = require_identity()
        key = lookup_api_key_by_id(self._db, request.key_id)
        if key is None:
            raise ConnectError(Code.NOT_FOUND, f"API key not found: {request.key_id}")
        if key.user_id != identity.user_id:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)
        revoke_api_key(self._db, request.key_id, Timestamp.now())
        if self._auth.jwt_manager:
            self._auth.jwt_manager.revoke(request.key_id)
        return job_pb2.Empty()

    def list_api_keys(
        self,
        request: job_pb2.ListApiKeysRequest,
        ctx: Any,
    ) -> job_pb2.ListApiKeysResponse:
        identity = require_identity()
        target_user = request.user_id or identity.user_id
        if target_user != identity.user_id:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)

        keys = list_api_keys(self._db, user_id=target_user if target_user else None)
        key_infos = []
        for k in keys:
            key_infos.append(
                job_pb2.ApiKeyInfo(
                    key_id=k.key_id,
                    key_prefix=k.key_prefix,
                    user_id=k.user_id,
                    name=k.name,
                    created_at_ms=k.created_at_ms.epoch_ms(),
                    last_used_at_ms=k.last_used_at_ms.epoch_ms() if k.last_used_at_ms else 0,
                    expires_at_ms=k.expires_at_ms.epoch_ms() if k.expires_at_ms else 0,
                    revoked=k.revoked_at_ms is not None,
                )
            )
        return job_pb2.ListApiKeysResponse(keys=key_infos)

    def get_current_user(
        self,
        request: job_pb2.GetCurrentUserRequest,
        ctx: Any,
    ) -> job_pb2.GetCurrentUserResponse:
        identity = get_verified_identity()
        if identity is None:
            return job_pb2.GetCurrentUserResponse(user_id="anonymous", role="")
        return job_pb2.GetCurrentUserResponse(
            user_id=identity.user_id,
            role=identity.role,
        )

    def exec_in_container(
        self,
        request: controller_pb2.Controller.ExecInContainerRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        """Execute a command in a running task's container.

        Proxies to the worker that owns the task. On K8s, delegates to the provider.
        """
        try:
            task_id = JobName.from_wire(request.task_id)
            task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc

        task = _read_task_with_attempts(self._db, task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

        # A federated task's subtree runs on a peer — it has no local worker or
        # attempt rows. Proxy the exec through the peer controller (which does its
        # own task->worker resolution) before the local resolution below, so it is
        # never dispatched to _backend_for_id's local fallback.
        handle = self._federated_handle_for_task(task_id)
        if handle is not None:
            return self._controller.federation.proxy_exec(
                peer_id=handle.peer_id,
                request=request,
            )

        worker_request = worker_pb2.Worker.ExecInContainerRequest(
            task_id=request.task_id,
            command=request.command,
            timeout_seconds=request.timeout_seconds,
        )

        task_worker_id = _task_worker_id(task)
        if not task_worker_id:
            # A cluster backend (K8s) execs directly into the pod; no worker daemon.
            if BackendCapability.CLUSTER_VIEW not in self._controller.capabilities:
                raise ConnectError(Code.FAILED_PRECONDITION, f"Task {request.task_id} not assigned to a worker")
            exec_target = TaskTarget(
                task_id=task.task_id.to_wire(),
                attempt_id=task.current_attempt_id,
                worker_id=None,
                address=None,
            )
            timeout = request.timeout_seconds if request.timeout_seconds else 60
        else:
            worker = _read_worker(self._db, task_worker_id)
            if not worker or not self._controller.liveness_for_worker(task_worker_id).healthy:
                raise ConnectError(Code.UNAVAILABLE, f"Worker {task_worker_id} is unavailable")
            exec_target = TaskTarget(
                task_id=task.task_id.to_wire(),
                attempt_id=task.current_attempt_id,
                worker_id=task_worker_id,
                address=worker.address,
            )
            timeout = request.timeout_seconds

        resp = self._backend_for_id(str(task.backend_id or "")).exec_in_container(exec_target, worker_request, timeout)
        return controller_pb2.Controller.ExecInContainerResponse(
            exit_code=resp.exit_code,
            stdout=resp.stdout,
            stderr=resp.stderr,
            error=resp.error,
        )

    def execute_raw_query(
        self,
        request: query_pb2.RawQueryRequest,
        ctx: Any,
    ) -> query_pb2.RawQueryResponse:
        identity = require_identity()
        if identity.role != "admin":
            raise ConnectError(Code.PERMISSION_DENIED, "admin role required for raw queries")

        # The read snapshot connection sets ``PRAGMA query_only = ON``, but a
        # query of the form ``PRAGMA query_only = OFF; UPDATE ...`` flips it
        # back before the snapshot rejects anything. Reject up front: only
        # statements whose first token is ``SELECT`` are permitted.
        if request.sql.lstrip()[:6].upper() != "SELECT":
            raise ConnectError(Code.INVALID_ARGUMENT, "only SELECT statements are allowed")

        with self._db.read_snapshot() as tx:
            result = tx.execute(text(request.sql))
            columns = [query_pb2.ColumnMeta(name=name, type="unknown") for name in result.keys()]
            rows = [json.dumps([_encode_query_cell(value) for value in row]) for row in result.all()]

        return query_pb2.RawQueryResponse(
            columns=columns,
            rows=rows,
        )

    def set_user_budget(
        self,
        request: controller_pb2.Controller.SetUserBudgetRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.SetUserBudgetResponse:
        """Set budget limit and max band for a user. Admin-only."""
        authorize(AuthzAction.MANAGE_BUDGETS)
        if not request.user_id:
            raise ConnectError(Code.INVALID_ARGUMENT, "user_id is required")
        max_band = request.max_band or job_pb2.PRIORITY_BAND_INTERACTIVE
        if max_band not in (
            job_pb2.PRIORITY_BAND_PRODUCTION,
            job_pb2.PRIORITY_BAND_INTERACTIVE,
            job_pb2.PRIORITY_BAND_BATCH,
        ):
            raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid max_band: {request.max_band}")
        now = Timestamp.now()
        with self._db.transaction() as _tx:
            writes.ensure_user(_tx, request.user_id, now)
            writes.set_user_budget(_tx, request.user_id, request.budget_limit, max_band, now)
        return controller_pb2.Controller.SetUserBudgetResponse()

    def get_user_budget(
        self,
        request: controller_pb2.Controller.GetUserBudgetRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetUserBudgetResponse:
        """Get budget config and current spend for a user."""
        require_identity()
        if not request.user_id:
            raise ConnectError(Code.INVALID_ARGUMENT, "user_id is required")
        with self._db.read_snapshot() as _snap:
            budget = reads.get_user_budget(_snap, request.user_id)
        if budget is None:
            raise ConnectError(Code.NOT_FOUND, f"No budget found for user {request.user_id}")
        with self._db.read_snapshot() as snap:
            spend = compute_user_spend(snap)
        return controller_pb2.Controller.GetUserBudgetResponse(
            user_id=budget.user_id,
            budget_limit=budget.budget_limit,
            budget_spent=spend.get(request.user_id, 0),
            max_band=budget.max_band,
        )

    def list_user_budgets(
        self,
        request: controller_pb2.Controller.ListUserBudgetsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListUserBudgetsResponse:
        """List all user budgets with current spend."""
        require_identity()
        with self._db.read_snapshot() as snap:
            budgets = reads.list_user_budgets(snap)
            spend = compute_user_spend(snap)
        users = []
        for b in budgets:
            users.append(
                controller_pb2.Controller.GetUserBudgetResponse(
                    user_id=b.user_id,
                    budget_limit=b.budget_limit,
                    budget_spent=spend.get(b.user_id, 0),
                    max_band=b.max_band,
                )
            )
        return controller_pb2.Controller.ListUserBudgetsResponse(users=users)

    def get_scheduler_state(
        self,
        request: controller_pb2.Controller.GetSchedulerStateRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetSchedulerStateResponse:
        """Return aggregated scheduler state for the dashboard.

        The dashboard SchedulerTab + AutoscalerTab consume rolled-up counts:
        per-(band, user, job) for pending and per-(band, user, worker, job)
        for running. Aggregation runs server-side and emits one proto entry
        per bucket rather than per task.
        """
        require_identity()

        with self._db.read_snapshot() as snap:
            budgets = reads.list_user_budgets(snap)
            budget_limits: dict[str, int] = {b.user_id: b.budget_limit for b in budgets}
            user_spend = compute_user_spend(snap)

            # Pending tasks: the scheduler's pending-task projection, reused here for
            # task_row_can_be_scheduled + band aggregation. No ORDER BY — we aggregate, not display.
            pending_raw = snap.execute(
                select(*reads.PENDING_TASK_COLS).where(local_tasks.c.state == job_pb2.TASK_STATE_PENDING)
            ).all()
            pending_rows = pending_raw
            pending_requested_bands = reads.get_priority_bands(snap, {row.job_id for row in pending_rows})

            # Running tasks: only task_id, priority_band, worker, and backend_id — no
            # job_config join is needed for the rolled-up counts below.
            running_raw = snap.execute(
                select(
                    tasks_table.c.task_id,
                    tasks_table.c.priority_band,
                    tasks_table.c.current_worker_id.label("worker_id"),
                    tasks_table.c.backend_id,
                ).where(
                    tasks_table.c.state == job_pb2.TASK_STATE_RUNNING,
                    tasks_table.c.current_worker_id.is_not(None),
                )
            ).all()

            running_rows = running_raw

        # Aggregate pending into (band, user, job, backend_id) → count buckets.
        pending_counts: dict[tuple[int, str, str, str], int] = {}
        total_pending = 0
        for row in pending_rows:
            if not task_row_can_be_scheduled(row):
                continue
            user_id = row.task_id.user
            eff_band = compute_effective_band(
                pending_requested_bands.get(row.job_id, row.priority_band),
                user_id,
                user_spend,
                budget_limits,
                self._user_budget_defaults,
            )
            job_id = (row.task_id.parent or row.task_id).to_wire()
            backend_id = str(row.backend_id or "")
            key = (eff_band, user_id, job_id, backend_id)
            pending_counts[key] = pending_counts.get(key, 0) + 1
            total_pending += 1

        # Aggregate running into (band, user, worker, job, backend_id) → count buckets.
        # Use the stamped ``tasks.priority_band`` directly: the scheduler stamps the
        # effective band at assign time (see ``_commit_assignments``), so re-running
        # ``compute_effective_band`` here against current spend would double-demote.
        running_counts: dict[tuple[int, str, str, str, str], int] = {}
        total_running = 0
        for row in running_rows:
            user_id = row.task_id.user
            job_id = (row.task_id.parent or row.task_id).to_wire()
            backend_id = str(row.backend_id or "")
            key = (row.priority_band, user_id, str(row.worker_id), job_id, backend_id)
            running_counts[key] = running_counts.get(key, 0) + 1
            total_running += 1

        # Synthesize budget rows for users with active spend but no explicit
        # user_budgets entry; the dashboard renders their utilization from
        # UserBudgetDefaults instead of '-'.
        budget_protos: list[controller_pb2.Controller.SchedulerUserBudget] = []
        defaults = self._user_budget_defaults
        seen_users = {b.user_id for b in budgets}
        budget_rows: list[tuple[str, int, int]] = [(b.user_id, b.budget_limit, b.max_band) for b in budgets]
        for uid in user_spend:
            if uid not in seen_users:
                budget_rows.append((uid, defaults.budget_limit, defaults.max_band))
        for user_id, budget_limit, max_band in budget_rows:
            spent = user_spend.get(user_id, 0)
            utilization = (spent / budget_limit * 100.0) if budget_limit > 0 else 0.0
            # Probe with INTERACTIVE so the dashboard sees whether this user is
            # currently downgraded.
            eff = compute_effective_band(
                job_pb2.PRIORITY_BAND_INTERACTIVE,
                user_id,
                user_spend,
                budget_limits,
                self._user_budget_defaults,
            )
            budget_protos.append(
                controller_pb2.Controller.SchedulerUserBudget(
                    user_id=user_id,
                    budget_limit=budget_limit,
                    budget_spent=spent,
                    max_band=max_band,
                    effective_band=eff,
                    utilization_percent=utilization,
                )
            )

        pending_buckets = [
            controller_pb2.Controller.PendingTaskBucket(
                band=band,
                user_id=user_id,
                job_id=job_id,
                backend_id=backend_id,
                count=count,
            )
            for (band, user_id, job_id, backend_id), count in pending_counts.items()
        ]
        running_buckets = [
            controller_pb2.Controller.RunningTaskBucket(
                band=band,
                user_id=user_id,
                worker_id=worker_id,
                job_id=job_id,
                backend_id=backend_id,
                count=count,
            )
            for (band, user_id, worker_id, job_id, backend_id), count in running_counts.items()
        ]

        return controller_pb2.Controller.GetSchedulerStateResponse(
            user_budgets=budget_protos,
            total_pending=total_pending,
            total_running=total_running,
            pending_buckets=pending_buckets,
            running_buckets=running_buckets,
        )

    def list_backends(
        self,
        request: controller_pb2.Controller.ListBackendsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListBackendsResponse:
        """List all backends with aggregate task/worker statistics.

        Counts come from grouped SQL queries joined in Python; capacity health is
        read from the in-memory autoscaler snapshot, not the DB.
        """
        require_identity()

        backends = self._controller.backends
        sg_to_backend = self._controller.scale_group_to_backend

        # Invert sg_to_backend: backend_id → list[scale_group]
        backend_to_sgs: dict[str, list[str]] = {bid: [] for bid in backends}
        for sg, bid in sg_to_backend.items():
            if bid in backend_to_sgs:
                backend_to_sgs[bid].append(sg)

        with self._db.read_snapshot() as snap:
            pending_by_backend: dict[str, int] = {}
            for row in snap.execute(
                select(tasks_table.c.backend_id, func.count().label("cnt"))
                .where(tasks_table.c.state == job_pb2.TASK_STATE_PENDING)
                .group_by(tasks_table.c.backend_id)
            ).all():
                pending_by_backend[str(row.backend_id or "")] = int(row.cnt)

            running_by_backend: dict[str, int] = {}
            for row in snap.execute(
                select(tasks_table.c.backend_id, func.count().label("cnt"))
                .where(tasks_table.c.state == job_pb2.TASK_STATE_RUNNING)
                .group_by(tasks_table.c.backend_id)
            ).all():
                running_by_backend[str(row.backend_id or "")] = int(row.cnt)

            worker_sg_rows = snap.execute(
                select(workers_table.c.scale_group, func.count().label("cnt")).group_by(workers_table.c.scale_group)
            ).all()

        worker_count_by_backend: dict[str, int] = {bid: 0 for bid in backends}
        for row in worker_sg_rows:
            bid = self._controller.backend_id_for_scale_group(str(row.scale_group or ""))
            worker_count_by_backend[bid] = worker_count_by_backend.get(bid, 0) + int(row.cnt)

        summaries: list[controller_pb2.Controller.BackendSummary] = []
        for backend_id, backend in sorted(backends.items()):
            allowed_users = backend.allowed_users
            restricted = "*" not in allowed_users

            caps = backend.capabilities
            if BackendCapability.CLUSTER_VIEW in caps:
                kind = "kubernetes"
            elif BackendCapability.WORKER_DAEMON in caps:
                kind = "worker-daemon"
            else:
                kind = "unknown"

            adv: dict[str, set[str]] = backend.advertised_attributes()

            # Each backend authors its own expanded status variant in full: a
            # worker-daemon backend reads its own liveness tracker and running-task
            # rows to stamp health counts, per-VM usability, and the backend_id on its
            # autoscaler groups. The controller renders the result verbatim — the
            # Backends tab's detail panel shows whichever variant the backend selected,
            # and the per-group capacity-health tally is read off the same authored view.
            backend_status = backend.status()
            variant = backend_status.WhichOneof("detail")

            cap_health: dict[str, int] = {}
            if variant == "worker":
                for group in backend_status.worker.autoscaler.groups:
                    st = group.availability_status or "unknown"
                    cap_health[st] = cap_health.get(st, 0) + 1

            summary = controller_pb2.Controller.BackendSummary(
                backend_id=backend_id,
                name=backend.name,
                kind=kind,
                capabilities=sorted(c.value for c in caps),
                restricted=restricted,
                allowed_user_count=len(allowed_users),
                scale_groups=sorted(backend_to_sgs.get(backend_id, [])),
                worker_count=worker_count_by_backend.get(backend_id, 0),
                pending_task_count=pending_by_backend.get(backend_id, 0),
                running_task_count=running_by_backend.get(backend_id, 0),
                has_autoscaler=backend.autoscaler is not None,
                capacity_health=cap_health,
            )
            # advertised_attributes is a proto map<string, StringList> (message
            # values), which doesn't support dict-style assignment/update; populate
            # each entry's repeated field in place.
            for key, values in adv.items():
                summary.advertised_attributes[key].values.extend(sorted(values))

            if variant == "kubernetes":
                summary.detail.kubernetes.CopyFrom(backend_status.kubernetes)
            elif variant == "worker":
                summary.detail.worker.CopyFrom(backend_status.worker)

            summaries.append(summary)

        unroutable = self._controller.last_unroutable_jobs
        sample = [
            controller_pb2.Controller.UnroutableJob(job_id=jid, reason=reason)
            for jid, reason in list(unroutable.items())[:_UNROUTABLE_SAMPLE_SIZE]
        ]

        return controller_pb2.Controller.ListBackendsResponse(
            backends=summaries,
            unroutable_job_count=len(unroutable),
            unroutable_sample=sample,
        )

    def list_peers(
        self,
        request: controller_pb2.Controller.ListPeersRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListPeersResponse:
        """List federation peers this controller may delegate whole jobs to.

        Each summary carries the peer's identity, controller/dashboard addresses,
        and its last capability-heartbeat result: reachability plus the peer's
        forwarded backends.
        """
        require_identity()
        return controller_pb2.Controller.ListPeersResponse(peers=self._controller.federation.peer_summaries())

    def _federated_job_summary(self, q, job) -> job_pb2.JobStatus:
        """A ``JobStatus`` for a handed-off job as this peer holds it (sync summary)."""
        summaries = reads.task_summaries_for_jobs(q, {job.job_id})
        status = job_pb2.JobStatus(
            job_id=job.job_id.to_wire(),
            state=job.state,
            error=job.error or "",
            exit_code=job.exit_code or 0,
            name=job.name,
            backend_id=job.backend_id or "",
            cluster=job.cluster,
            **_job_status_counts(summaries.get(job.job_id), job.job_id),
        )
        if job.started_at_ms:
            status.started_at.CopyFrom(timestamp_to_proto(job.started_at_ms))
        if job.finished_at_ms:
            status.finished_at.CopyFrom(timestamp_to_proto(job.finished_at_ms))
        if job.submitted_at_ms:
            status.submitted_at.CopyFrom(timestamp_to_proto(job.submitted_at_ms))
        return status

    def _federated_job_delta(
        self, q, job_id: JobName, *, all_tasks: bool, task_indexes: set[int]
    ) -> controller_pb2.Controller.FederationJobDelta | None:
        """One non-tombstone delta: the job's current summary plus its changed tasks.

        ``all_tasks`` includes every current task (a job-level change or full
        resync); otherwise only ``task_indexes``. ``None`` if the job raced with its
        own prune — the tombstone event follows on a later sync.
        """
        job = reads.get_job_detail(q, job_id)
        if job is None:
            return None
        # Full attempt history per changed task (not the list view's reduced set),
        # so the parent mirrors the complete attempt list for the handed-off job.
        task_rows = [
            row
            for row in q.execute(reads.task_detail_query().where(tasks_table.c.job_id == job_id)).all()
            if all_tasks or row.task_id.task_index in task_indexes
        ]
        attempts_by_task = reads.all_attempts_for_tasks(q, [row.task_id for row in task_rows])
        changed_tasks = [
            task_to_proto(TaskWithAttempts.from_row(row, attempts_by_task.get(row.task_id, ()))) for row in task_rows
        ]
        return controller_pb2.Controller.FederationJobDelta(
            job_id=job_id.to_wire(),
            summary=self._federated_job_summary(q, job),
            changed_tasks=changed_tasks,
        )

    def federation_sync(
        self,
        request: controller_pb2.Controller.FederationSyncRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.FederationSyncResponse:
        """Peer side: report jobs ``requester_id`` handed off, changed since its cursor.

        On first contact (empty cursor) or a cursor below the retained changelog
        window, returns the requester's full active set with ``cursor_stale`` so the
        parent set-replaces. Otherwise returns the incremental set: one delta per job
        whose changelog rows advanced past the cursor — a tombstone for a pruned job,
        else the job's summary plus its changed tasks. Assembled in one snapshot.
        """
        require_identity()
        requester_id = request.requester_id
        cursor = request.cursor
        cursor_seq = int(cursor) if cursor else 0
        deltas: list[controller_pb2.Controller.FederationJobDelta] = []
        with self._db.read_snapshot() as q:
            min_seq = reads.changelog_min_seq(q)
            next_cursor = str(reads.changelog_max_seq(q))
            # Stale when the requester has no cursor, or its cursor sits below the
            # oldest retained event (an unconsumed event, including a tombstone, may
            # be gone). A full resync subsumes everything, so it advances to the max.
            stale = (not cursor) or (min_seq > 0 and cursor_seq < min_seq - 1)

            if stale:
                for job_id in reads.received_jobs_for_requester(q, requester_id):
                    delta = self._federated_job_delta(q, job_id, all_tasks=True, task_indexes=set())
                    if delta is not None:
                        deltas.append(delta)
                return controller_pb2.Controller.FederationSyncResponse(
                    deltas=deltas, next_cursor=next_cursor, cursor_stale=True
                )

            tombstoned: dict[JobName, bool] = {}
            all_tasks: dict[JobName, bool] = {}
            indexes: dict[JobName, set[int]] = {}
            order: list[JobName] = []
            for row in reads.changelog_rows_since(q, requester_id, cursor_seq):
                if row.job_id not in tombstoned:
                    tombstoned[row.job_id] = False
                    all_tasks[row.job_id] = False
                    indexes[row.job_id] = set()
                    order.append(row.job_id)
                if row.tombstone:
                    tombstoned[row.job_id] = True
                elif row.task_index is None:
                    all_tasks[row.job_id] = True
                else:
                    indexes[row.job_id].add(row.task_index)

            for job_id in order:
                if tombstoned[job_id]:
                    deltas.append(controller_pb2.Controller.FederationJobDelta(job_id=job_id.to_wire(), tombstone=True))
                    continue
                delta = self._federated_job_delta(q, job_id, all_tasks=all_tasks[job_id], task_indexes=indexes[job_id])
                if delta is not None:
                    deltas.append(delta)

        return controller_pb2.Controller.FederationSyncResponse(
            deltas=deltas, next_cursor=next_cursor, cursor_stale=False
        )
