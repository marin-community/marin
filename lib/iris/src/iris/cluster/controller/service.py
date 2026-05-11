# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller RPC service implementation handling job, task, and worker operations.

The controller expands jobs into tasks at submission time (a job with replicas=N
creates N tasks). Tasks are the unit of scheduling and execution. Job state is
aggregated from task states.
"""

import dataclasses
import json
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from finelog.client import LogClient
from rigging.timing import Duration, ExponentialBackoff, Timer, Timestamp

from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import Constraint, constraints_from_resources, merge_constraints, validate_tpu_request
from iris.cluster.controller.auth import (
    DEFAULT_JWT_TTL_SECONDS,
    ControllerAuth,
    create_api_key,
    list_api_keys,
    revoke_api_key,
    revoke_login_keys_for_user,
)
from iris.cluster.controller.autoscaler.status import PendingHint
from iris.cluster.controller.budget import (
    UserBudgetDefaults,
    compute_effective_band,
    compute_user_spend,
)
from iris.cluster.controller.codec import (
    constraints_from_json,
    proto_from_json,
    reservation_entries_from_json,
    resource_spec_from_scalars,
)
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    ControllerDB,
    EndpointQuery,
    QuerySnapshot,
    SchedulableWorker,
    TaskJobSummary,
    UserStats,
    attempt_is_worker_failure,
    running_tasks_by_worker,
    task_row_can_be_scheduled,
)
from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.query import execute_raw_query
from iris.cluster.controller.scheduler import SchedulingContext
from iris.cluster.controller.schema import (
    API_KEY_PROJECTION,
    ATTEMPT_PROJECTION,
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    JOB_ROW_PROJECTION,
    TASK_DETAIL_PROJECTION,
    TASK_ROW_PROJECTION,
    WORKER_DETAIL_PROJECTION,
    AttemptRow,
    EndpointRow,
    JobDetailRow,
    JobRow,
    TaskDetailRow,
    WorkerDetailRow,
    tasks_with_attempts,
)
from iris.cluster.controller.stores import AddEndpointOutcome, ControllerStore
from iris.cluster.controller.transitions import (
    ControllerTransitions,
    HeartbeatApplyRequest,
    task_updates_from_proto,
)
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.process_status import get_process_status
from iris.cluster.redaction import redact_request_env_vars
from iris.cluster.runtime.profile import (
    PROFILE_NAMESPACE,
    IrisProfile,
    build_profile_row,
    parse_profile_target,
    profile_local_process,
)
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    JobName,
    WorkerId,
    is_job_finished,
)
from iris.rpc import controller_pb2, job_pb2, query_pb2, vm_pb2, worker_pb2
from iris.rpc.async_adapter import on_loop
from iris.rpc.auth import (
    AuthzAction,
    authorize,
    authorize_resource_owner,
    get_verified_identity,
    get_verified_user,
    require_identity,
)
from iris.rpc.proto_utils import job_state_friendly, priority_band_name, task_state_friendly
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)


# Maximum bundle size in bytes (25 MB) - matches client-side limit
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024

# Time the launch RPC blocks waiting for a replaced job's worker-bound attempts
# to finalize before deleting them. Normal worker shutdown is well under one
# minute; the bound is finite so a stuck finalization surfaces as
# DEADLINE_EXCEEDED rather than hanging launches forever.
_JOB_REPLACEMENT_DRAIN_TIMEOUT = Duration.from_minutes(10)

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


def _current_attempt(task: TaskDetailRow) -> AttemptRow | None:
    """Get the latest attempt for a task detail row."""
    if not task.attempts:
        return None
    return task.attempts[-1]


def _task_worker_id(task: TaskDetailRow) -> WorkerId | None:
    """Get the effective worker_id for a task detail row."""
    current = _current_attempt(task)
    if current is None:
        return task.current_worker_id
    return current.worker_id


def _active_worker_id(task: TaskDetailRow) -> WorkerId | None:
    """Get the active worker_id (None for pending tasks)."""
    if task.state == job_pb2.TASK_STATE_PENDING:
        return None
    return _task_worker_id(task)


def task_to_proto(task: TaskDetailRow, worker_address: str = "") -> job_pb2.TaskStatus:
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
        )
        if attempt.started_at is not None:
            proto_attempt.started_at.CopyFrom(timestamp_to_proto(attempt.started_at))
        if attempt.finished_at is not None:
            proto_attempt.finished_at.CopyFrom(timestamp_to_proto(attempt.finished_at))
        attempts.append(proto_attempt)

    active_wid = _active_worker_id(task)
    proto = job_pb2.TaskStatus(
        task_id=task.task_id.to_wire(),
        state=task.state,
        worker_id=str(active_wid) if active_wid else "",
        worker_address=worker_address or task.current_worker_address or "",
        exit_code=task.exit_code or 0,
        error=task.error or "",
        current_attempt_id=task.current_attempt_id,
        attempts=attempts,
    )
    if current_attempt and current_attempt.started_at:
        proto.started_at.CopyFrom(timestamp_to_proto(current_attempt.started_at))
    if current_attempt and current_attempt.finished_at:
        proto.finished_at.CopyFrom(timestamp_to_proto(current_attempt.finished_at))
    if task.container_id:
        proto.container_id = task.container_id
    # For pending tasks with prior terminal attempts, surface retry context.
    if task.state == job_pb2.TASK_STATE_PENDING and task.attempts and task.attempts[-1].state in TERMINAL_TASK_STATES:
        last = task.attempts[-1]
        proto.pending_reason = (
            f"Retrying (attempt {len(task.attempts)}, " f"last: {job_pb2.TaskState.Name(last.state).lower()})"
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


def _read_job(q: QuerySnapshot, job_id: JobName) -> JobDetailRow | None:
    return JOB_DETAIL_PROJECTION.decode_one(
        q.fetchall(
            f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} " f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id.to_wire(),),
        )
    )


def _read_task_with_attempts(db: ControllerDB, task_id: JobName) -> TaskDetailRow | None:
    task_wire = task_id.to_wire()
    with db.read_snapshot() as q:
        task = TASK_DETAIL_PROJECTION.decode_one(
            q.fetchall(
                f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} FROM tasks t WHERE t.task_id = ?",
                (task_wire,),
            )
        )
        if task is None:
            return None
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
                "WHERE ta.task_id = ? ORDER BY ta.attempt_id ASC",
                (task_wire,),
            ),
        )
    return tasks_with_attempts([task], attempts)[0]


def _read_worker(store: ControllerStore, worker_id: WorkerId) -> WorkerDetailRow | None:
    with store.read_snapshot() as q:
        return store.workers.get_detail(q, worker_id)


def _job_state(db: ControllerDB, job_id: JobName) -> int | None:
    """Fetch only the state column for a job, avoiding proto decode."""
    with db.read_snapshot() as q:
        row = q.fetchone("SELECT state FROM jobs WHERE job_id = ?", (job_id.to_wire(),))
        return int(row[0]) if row else None


def _worker_address(db: ControllerDB, worker_id: WorkerId) -> str | None:
    """Fetch only the address column for a worker, avoiding proto decode."""
    with db.read_snapshot() as q:
        row = q.fetchone("SELECT address FROM workers WHERE worker_id = ?", (str(worker_id),))
        return str(row[0]) if row else None


def _resource_spec_from_job_row(job: Any) -> job_pb2.ResourceSpecProto:
    """Reconstruct a ResourceSpecProto from native job columns."""
    return resource_spec_from_scalars(
        job.res_cpu_millicores, job.res_memory_bytes, job.res_disk_bytes, job.res_device_json
    )


def _reconstruct_launch_job_request(job: JobDetailRow) -> controller_pb2.Controller.LaunchJobRequest:
    """Reconstruct a LaunchJobRequest proto from native JobDetailRow columns."""
    req = controller_pb2.Controller.LaunchJobRequest(
        name=job.name,
        bundle_id=job.bundle_id,
        max_task_failures=job.max_task_failures,
        max_retries_failure=job.max_retries_failure,
        max_retries_preemption=job.max_retries_preemption,
        replicas=job.num_tasks,
        preemption_policy=job.preemption_policy,
        existing_job_policy=job.existing_job_policy,
        priority_band=job.priority_band,
        task_image=job.task_image,
        fail_if_exists=job.fail_if_exists,
    )
    req.entrypoint.CopyFrom(proto_from_json(job.entrypoint_json, job_pb2.RuntimeEntrypoint))
    req.environment.CopyFrom(proto_from_json(job.environment_json, job_pb2.EnvironmentConfig))
    req.resources.CopyFrom(
        resource_spec_from_scalars(job.res_cpu_millicores, job.res_memory_bytes, job.res_disk_bytes, job.res_device_json)
    )

    for c in constraints_from_json(job.constraints_json):
        req.constraints.append(c.to_proto())
    for port in json.loads(job.ports_json):
        req.ports.append(port)
    for arg in json.loads(job.submit_argv_json):
        req.submit_argv.append(arg)

    if job.has_coscheduling:
        req.coscheduling.CopyFrom(job_pb2.CoschedulingConfig(group_by=job.coscheduling_group_by))

    if job.scheduling_timeout_ms is not None and job.scheduling_timeout_ms > 0:
        req.scheduling_timeout.milliseconds = job.scheduling_timeout_ms

    if job.timeout_ms is not None and job.timeout_ms > 0:
        req.timeout.milliseconds = job.timeout_ms

    if job.reservation_json:
        for entry in reservation_entries_from_json(job.reservation_json):
            req.reservation.entries.append(entry)

    return req


def _worker_metadata_to_proto(worker: WorkerDetailRow) -> job_pb2.WorkerMetadata:
    """Reconstruct a WorkerMetadata proto from scalar columns."""
    md = job_pb2.WorkerMetadata(
        hostname=worker.md_hostname,
        ip_address=worker.md_ip_address,
        cpu_count=worker.md_cpu_count,
        memory_bytes=worker.md_memory_bytes,
        disk_bytes=worker.md_disk_bytes,
        tpu_name=worker.md_tpu_name,
        tpu_worker_hostnames=worker.md_tpu_worker_hostnames,
        tpu_worker_id=worker.md_tpu_worker_id,
        tpu_chips_per_host_bounds=worker.md_tpu_chips_per_host_bounds,
        gpu_count=worker.md_gpu_count,
        gpu_name=worker.md_gpu_name,
        gpu_memory_mb=worker.md_gpu_memory_mb,
        gce_instance_name=worker.md_gce_instance_name,
        gce_zone=worker.md_gce_zone,
        git_hash=worker.md_git_hash,
    )
    if worker.md_device_json and worker.md_device_json != "{}":
        md.device.CopyFrom(proto_from_json(worker.md_device_json, job_pb2.DeviceConfig))
    # Populate attributes from the worker_attributes table data stored on the row.
    for key, value in worker.attributes.items():
        av = job_pb2.AttributeValue()
        if isinstance(value, str):
            av.string_value = value
        elif isinstance(value, int):
            av.int_value = value
        elif isinstance(value, float):
            av.float_value = value
        md.attributes[key].CopyFrom(av)
    return md


def _decode_attribute_value(row: Any) -> tuple[str, str | int | float]:
    """Decode a worker_attributes row into a (key, value) pair."""
    vtype = str(row["value_type"])
    key = str(row["key"])
    if vtype == "str":
        return key, str(row["str_value"])
    elif vtype == "int":
        return key, int(row["int_value"])
    elif vtype == "float":
        return key, float(row["float_value"])
    raise ValueError(f"Unknown attribute value_type: {vtype!r}")


@dataclass(frozen=True)
class _WorkerDetail:
    worker: WorkerDetailRow
    running_tasks: frozenset[JobName]


def _read_worker_detail(store: ControllerStore, worker_id: WorkerId) -> _WorkerDetail | None:
    with store.read_snapshot() as q:
        worker = store.workers.get_detail(q, worker_id)
        if worker is None:
            return None
        attr_rows = q.fetchall(
            "SELECT key, value_type, str_value, int_value, float_value " "FROM worker_attributes WHERE worker_id = ?",
            (str(worker_id),),
        )
        attrs = dict(_decode_attribute_value(row) for row in attr_rows)
        if attrs:
            worker = dataclasses.replace(worker, attributes=attrs)
        running_rows = q.raw(
            "SELECT t.task_id FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "WHERE a.worker_id = ? AND t.state IN (?, ?, ?)",
            (str(worker_id), *ACTIVE_TASK_STATES),
            decoders={"task_id": JobName.from_wire},
        )
    return _WorkerDetail(
        worker=worker,
        running_tasks=frozenset(r.task_id for r in running_rows),
    )


def _tasks_for_listing(db: ControllerDB, *, job_id: JobName) -> list[TaskDetailRow]:
    """Load tasks for the list view, attaching only the current attempt.

    The list UI only needs the current attempt's ``started_at`` /
    ``finished_at`` and a single ``proto.attempts`` entry. Full history is
    fetched separately by ``get_task_status``.
    """
    job_wire = job_id.to_wire()
    with db.read_snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall(
                f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} "
                "FROM tasks t WHERE t.job_id = ? ORDER BY t.job_id ASC, t.task_index ASC",
                (job_wire,),
            ),
        )
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
                "WHERE (ta.task_id, ta.attempt_id) IN ("
                "    SELECT t.task_id, t.current_attempt_id FROM tasks t "
                "    WHERE t.job_id = ? AND t.current_attempt_id >= 0"
                ")",
                (job_wire,),
            ),
        )
    return tasks_with_attempts(tasks, attempts)


def _worker_addresses_for_tasks(db: ControllerDB, tasks: list[TaskDetailRow]) -> dict[WorkerId, str]:
    """Fetch addresses only for workers referenced by the given tasks."""
    worker_ids = {_task_worker_id(t) for t in tasks}
    worker_ids.discard(None)
    if not worker_ids:
        return {}
    placeholders = ",".join("?" for _ in worker_ids)
    with db.read_snapshot() as q:
        rows = q.raw(
            f"SELECT worker_id, address FROM workers WHERE worker_id IN ({placeholders})",
            tuple(str(wid) for wid in worker_ids),
        )
    return {WorkerId(str(row.worker_id)): row.address for row in rows}


# State display order for sorting (active states first). Rendered into
# ``_STATE_SORT_EXPR`` as a CASE expression for the JOB_SORT_FIELD_STATE path.
_STATE_SORT_ORDER: dict[int, int] = {
    job_pb2.JOB_STATE_RUNNING: 0,
    job_pb2.JOB_STATE_BUILDING: 1,
    job_pb2.JOB_STATE_PENDING: 2,
    job_pb2.JOB_STATE_SUCCEEDED: 3,
    job_pb2.JOB_STATE_FAILED: 4,
    job_pb2.JOB_STATE_KILLED: 5,
    job_pb2.JOB_STATE_WORKER_FAILED: 6,
    job_pb2.JOB_STATE_UNSCHEDULABLE: 7,
}
_STATE_SORT_EXPR = (
    "CASE j.state"
    + "".join(f" WHEN {state} THEN {order}" for state, order in _STATE_SORT_ORDER.items())
    + " ELSE 99 END"
)

_SORT_FIELD_TO_SQL: dict[int, str] = {
    controller_pb2.Controller.JOB_SORT_FIELD_DATE: "j.submitted_at_ms",
    controller_pb2.Controller.JOB_SORT_FIELD_NAME: "j.name",
    controller_pb2.Controller.JOB_SORT_FIELD_STATE: _STATE_SORT_EXPR,
    controller_pb2.Controller.JOB_SORT_FIELD_FAILURES: "agg_failures",
    controller_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS: "agg_preemptions",
}


MAX_LIST_JOBS_LIMIT = 500
# Hard cap on how deep ListJobs callers may page. A correctly-filtered query
# should narrow the result set; anything reaching offsets this deep is a sign
# of a caller scanning the entire jobs table page-by-page, which is what the
# snapshot is supposed to prevent. Force callers to filter instead.
MAX_LIST_JOBS_OFFSET = 5000
MAX_LIST_WORKERS_LIMIT = 1000


def _filter_and_sort_workers(
    workers: list[WorkerDetailRow],
    liveness_by_id: dict[WorkerId, WorkerLiveness],
    query: controller_pb2.Controller.WorkerQuery,
) -> list[WorkerDetailRow]:
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
            w for w in workers if needle in str(w.worker_id).lower() or (w.address and needle in w.address.lower())
        ]

    sort_field = query.sort_field or controller_pb2.Controller.WORKER_SORT_FIELD_WORKER_ID
    descending = query.sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC
    if sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_LAST_HEARTBEAT:
        workers = sorted(workers, key=lambda w: liveness_by_id[w.worker_id].last_heartbeat_ms, reverse=descending)
    elif sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_DEVICE_TYPE:
        # CPU workers persist with ``device_type == ""``; under ascending sort
        # they group first (treating CPU as the no-accelerator baseline).
        workers = sorted(workers, key=lambda w: (w.device_type, str(w.worker_id)), reverse=descending)
    else:
        workers = sorted(workers, key=lambda w: str(w.worker_id), reverse=descending)
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
    q: QuerySnapshot,
    query: controller_pb2.Controller.JobQuery,
    state_ids: tuple[int, ...],
) -> tuple[list[JobRow], int]:
    """Execute a ``JobQuery`` and return ``(rows, total_count)``.

    ``state_ids`` is the pre-resolved state filter (always non-empty); the
    caller owns "unknown state -> empty page" handling so that a bad filter
    never reaches SQL. The caller also owns the read snapshot — list_jobs
    chains the SELECT, COUNT, and downstream summary/parent queries on a
    single snapshot to keep the per-connection page cache hot.
    """
    assert state_ids, "_query_jobs requires at least one state id"

    conditions: list[str] = []
    params: list[object] = []

    scope = query.scope or controller_pb2.Controller.JOB_QUERY_SCOPE_ALL
    if scope == controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS:
        conditions.append("j.depth = 1")
    elif scope == controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN:
        if not query.parent_job_id:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                "query.parent_job_id is required for JOB_QUERY_SCOPE_CHILDREN",
            )
        conditions.append("j.parent_job_id = ?")
        params.append(query.parent_job_id)
    # JOB_QUERY_SCOPE_ALL: no ancestry constraint.

    state_placeholders = ",".join("?" for _ in state_ids)
    conditions.append(f"j.state IN ({state_placeholders})")
    params.extend(state_ids)

    if query.name_filter:
        conditions.append("j.name LIKE ?")
        params.append(f"%{query.name_filter.lower()}%")

    where_clause = " AND ".join(conditions)

    sort_field = query.sort_field or controller_pb2.Controller.JOB_SORT_FIELD_DATE
    sort_direction = query.sort_direction
    if sort_direction == controller_pb2.Controller.SORT_DIRECTION_UNSPECIFIED:
        sort_direction = (
            controller_pb2.Controller.SORT_DIRECTION_DESC
            if sort_field == controller_pb2.Controller.JOB_SORT_FIELD_DATE
            else controller_pb2.Controller.SORT_DIRECTION_ASC
        )
    direction = "DESC" if sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC else "ASC"
    order_expr = _SORT_FIELD_TO_SQL.get(sort_field, "j.submitted_at_ms")

    # COUNT only filters on j.* columns; the projection-side join to job_config
    # adds a B-tree probe per candidate row for nothing. The FK guarantees the
    # row exists, so the COUNT cannot diverge from the unjoined form.
    count_sql = f"SELECT COUNT(*) FROM jobs j WHERE {where_clause}"

    # Only join tasks when sorting by failure/preemption aggregates.
    # The common case (sort by date, name, state) skips the expensive LEFT JOIN + GROUP BY.
    needs_task_agg = sort_field in (
        controller_pb2.Controller.JOB_SORT_FIELD_FAILURES,
        controller_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS,
    )

    if needs_task_agg:
        select_sql = f"""
            SELECT {JOB_ROW_PROJECTION.select_clause()},
                   COALESCE(SUM(t.failure_count), 0) AS agg_failures,
                   COALESCE(SUM(t.preemption_count), 0) AS agg_preemptions
            FROM jobs j {JOB_CONFIG_JOIN}
            LEFT JOIN tasks t ON j.job_id = t.job_id
            WHERE {where_clause}
            GROUP BY j.job_id
            ORDER BY {order_expr} {direction}
        """
    else:
        select_sql = f"""
            SELECT {JOB_ROW_PROJECTION.select_clause()}
            FROM jobs j {JOB_CONFIG_JOIN}
            WHERE {where_clause}
            ORDER BY {order_expr} {direction}
        """

    offset = max(query.offset, 0)
    limit = max(query.limit, 0)
    select_params = list(params)
    if limit > 0:
        select_sql += " LIMIT ? OFFSET ?"
        select_params.extend([limit, offset])

    rows = q.execute_sql(select_sql, tuple(select_params)).fetchall()
    total = q.execute_sql(count_sql, tuple(params)).fetchone()[0]
    return JOB_ROW_PROJECTION.decode(rows), total


def _query_from_list_jobs_request(
    request: controller_pb2.Controller.ListJobsRequest,
) -> controller_pb2.Controller.JobQuery:
    """Return the request's ``JobQuery`` with paging clamped to safe bounds."""
    query = controller_pb2.Controller.JobQuery()
    if request.HasField("query"):
        query.CopyFrom(request.query)

    # Clamp paging: 0 (unset) or out-of-range values default to MAX. Unbounded
    # listing is not supported because downstream per-page work
    # (_task_summaries_for_jobs, _parent_ids_with_children) grows an IN-clause
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


def _parent_ids_with_children(q: QuerySnapshot, job_ids: list[JobName]) -> set[JobName]:
    """Return the subset of *job_ids* that currently have direct children."""
    if not job_ids:
        return set()
    placeholders = ",".join("?" for _ in job_ids)
    sql = f"""
        SELECT DISTINCT j.parent_job_id
        FROM jobs j
        WHERE j.parent_job_id IN ({placeholders})
    """
    rows = q.raw(sql, tuple(job_id.to_wire() for job_id in job_ids))
    return {JobName.from_wire(row.parent_job_id) for row in rows if row.parent_job_id}


def _task_summaries_for_jobs(q: QuerySnapshot, job_ids: set[JobName]) -> dict[JobName, TaskJobSummary]:
    """Aggregate task counts per job via a SQL GROUP BY."""
    placeholders = ",".join("?" for _ in job_ids)
    params: tuple[object, ...] = tuple(j.to_wire() for j in job_ids)

    sql = f"""
        SELECT t.job_id,
               t.state,
               COUNT(*) as cnt,
               SUM(t.failure_count) as total_failures,
               SUM(t.preemption_count) as total_preemptions
        FROM tasks t
        WHERE t.job_id IN ({placeholders})
        GROUP BY t.job_id, t.state
    """
    completed_states = (job_pb2.TASK_STATE_SUCCEEDED, job_pb2.TASK_STATE_KILLED)
    rows = q.raw(sql, params, decoders={"job_id": JobName.from_wire})

    summaries: dict[JobName, TaskJobSummary] = {}
    for row in rows:
        prev = summaries.get(row.job_id, TaskJobSummary(job_id=row.job_id))
        summaries[row.job_id] = TaskJobSummary(
            job_id=row.job_id,
            task_count=prev.task_count + row.cnt,
            completed_count=prev.completed_count + (row.cnt if row.state in completed_states else 0),
            failure_count=prev.failure_count + row.total_failures,
            preemption_count=prev.preemption_count + row.total_preemptions,
            task_state_counts={**prev.task_state_counts, row.state: row.cnt},
        )
    return summaries


def _worker_roster(store: ControllerStore) -> list[WorkerDetailRow]:
    with store.read_snapshot() as q:
        decoded = WORKER_DETAIL_PROJECTION.decode(
            q.fetchall(f"SELECT {WORKER_DETAIL_PROJECTION.select_clause()} FROM workers w")
        )
        if not decoded:
            return []
        worker_ids = tuple(str(w.worker_id) for w in decoded)
        placeholders = ",".join("?" for _ in worker_ids)
        attr_rows = q.fetchall(
            f"SELECT worker_id, key, value_type, str_value, int_value, float_value "
            f"FROM worker_attributes WHERE worker_id IN ({placeholders})",
            worker_ids,
        )
        attrs_by_worker: dict[str, dict[str, str | int | float]] = {}
        for row in attr_rows:
            wid = str(row["worker_id"])
            key, value = _decode_attribute_value(row)
            attrs_by_worker.setdefault(wid, {})[key] = value
    return [dataclasses.replace(w, attributes=attrs_by_worker.get(str(w.worker_id), {})) for w in decoded]


def _live_user_stats(db: ControllerDB) -> list[UserStats]:
    """Aggregate job/task counts per user for active (non-terminal) jobs."""
    active_states = ",".join(
        str(s)
        for s in (
            job_pb2.JOB_STATE_PENDING,
            job_pb2.JOB_STATE_BUILDING,
            job_pb2.JOB_STATE_RUNNING,
        )
    )
    with db.read_snapshot() as q:
        job_rows = q.raw(
            f"SELECT j.user_id, j.state, COUNT(*) as cnt FROM jobs j "
            f"WHERE j.state IN ({active_states}) GROUP BY j.user_id, j.state"
        )
        task_rows = q.raw(
            f"SELECT j.user_id, t.state, COUNT(*) as cnt "
            f"FROM tasks t JOIN jobs j ON t.job_id = j.job_id "
            f"WHERE j.state IN ({active_states}) "
            f"GROUP BY j.user_id, t.state"
        )
    by_user: dict[str, UserStats] = {}
    for row in job_rows:
        stats = by_user.setdefault(row.user_id, UserStats(user=row.user_id))
        stats.job_state_counts[row.state] = row.cnt
    for row in task_rows:
        stats = by_user.setdefault(row.user_id, UserStats(user=row.user_id))
        stats.task_state_counts[row.state] = row.cnt
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
    with db.read_snapshot() as q:
        rows = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
                "WHERE ta.worker_id = ? "
                "ORDER BY COALESCE(ta.started_at_ms, ta.created_at_ms) DESC "
                "LIMIT ?",
                (str(worker_id), limit),
            ),
        )
    out: list[controller_pb2.Controller.WorkerTaskAttempt] = []
    for row in rows:
        proto_attempt = job_pb2.TaskAttempt(
            attempt_id=row.attempt_id,
            worker_id=str(row.worker_id) if row.worker_id else "",
            state=row.state,
            exit_code=row.exit_code or 0,
            error=row.error or "",
            is_worker_failure=attempt_is_worker_failure(row.state),
        )
        if row.started_at is not None:
            proto_attempt.started_at.CopyFrom(timestamp_to_proto(row.started_at))
        if row.finished_at is not None:
            proto_attempt.finished_at.CopyFrom(timestamp_to_proto(row.finished_at))
        out.append(controller_pb2.Controller.WorkerTaskAttempt(task_id=row.task_id.to_wire(), attempt=proto_attempt))
    return out


class AutoscalerProtocol(Protocol):
    """Protocol for autoscaler operations used by ControllerServiceImpl."""

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Get autoscaler status."""
        ...

    def get_pending_hints(self) -> dict[str, PendingHint]:
        """Get cached pending-hint dict keyed by job id."""
        ...

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get info for a specific VM."""
        ...

    def job_feasibility(
        self,
        constraints: list[Constraint],
        *,
        replicas: int | None = None,
    ) -> str | None:
        """Check if a job can ever be scheduled. Returns error message or None."""
        ...

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get initialization log for a VM."""
        ...


class ControllerProtocol(Protocol):
    """Protocol for controller operations used by ControllerServiceImpl."""

    def wake(self) -> None: ...

    def create_scheduling_context(self, workers: list[SchedulableWorker]) -> SchedulingContext: ...

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None: ...

    def begin_checkpoint(self) -> tuple[str, Any]: ...

    @property
    def autoscaler(self) -> AutoscalerProtocol | None: ...

    @property
    def provider(self) -> Any: ...

    @property
    def has_direct_provider(self) -> bool: ...

    @property
    def provider_scheduling_events(self) -> list: ...

    @property
    def provider_capacity(self) -> Any: ...


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
        transitions: State machine for DB mutations (submit, cancel, register, etc.)
        store: Controller store bundle (per-entity stores + transaction / read_snapshot).
        controller: Controller runtime for scheduling and worker management
        bundle_store: Bundle store for zip storage.
        log_client: LogClient for reading task logs through LogService.FetchLogs.
    """

    def __init__(
        self,
        transitions: ControllerTransitions,
        store: ControllerStore,
        controller: ControllerProtocol,
        bundle_store: BundleStore,
        log_client: LogClient,
        auth: ControllerAuth | None = None,
        system_endpoints: dict[str, str] | None = None,
        user_budget_defaults: UserBudgetDefaults | None = None,
    ):
        self._transitions = transitions
        self._store = store
        self._db = store._db
        self._controller = controller
        self._bundle_store = bundle_store
        self._log_client = log_client
        self._timer = Timer()
        self._auth = auth or ControllerAuth()
        self._system_endpoints: dict[str, str] = system_endpoints or {}
        self._user_budget_defaults = user_budget_defaults or UserBudgetDefaults()
        self._profile_table = self._log_client.get_table(PROFILE_NAMESPACE, IrisProfile)

    def bundle_zip(self, bundle_id: str) -> bytes:
        return self._bundle_store.get_zip(bundle_id)

    def blob_data(self, blob_id: str) -> bytes:
        return self._bundle_store.get_zip(blob_id)

    def _get_autoscaler_pending_hints(self) -> dict[str, PendingHint]:
        """Build autoscaler-based pending hints keyed by job id."""
        autoscaler = self._controller.autoscaler
        if autoscaler is None:
            return {}
        # Autoscaler caches the hint dict per evaluate() cycle; this avoids
        # rebuilding the full AutoscalerStatus proto on every GetJobStatus
        # RPC (#4844).
        return autoscaler.get_pending_hints()

    def _authorize_job_owner(self, job_id: JobName) -> None:
        """Raise PERMISSION_DENIED if the authenticated user doesn't own this job.

        Skipped when no auth provider is configured (null-auth mode).
        """
        if not self._auth.provider:
            return
        authorize_resource_owner(job_id.user)

    def _wait_until_job_drained(self, job_id: JobName, timeout: Duration) -> None:
        """Block until ``job_id`` has no unfinished worker-bound attempts.

        Polls the snapshot DB; the heartbeat path landing terminal updates is
        what finally clears the predicate. Used to gate ``remove_finished_job``
        on a replacement so we never CASCADE-delete tasks whose attempts the
        worker is still racing to finalize. Raises ``ConnectError`` with
        ``DEADLINE_EXCEEDED`` if the job hasn't drained within ``timeout``.
        """

        def drained() -> bool:
            with self._db.read_snapshot() as snap:
                return not self._store.jobs.has_unfinished_worker_attempts(snap, job_id)

        try:
            ExponentialBackoff(initial=0.05, maximum=1.0, factor=1.5).wait_until_or_raise(
                drained,
                timeout=timeout,
                error_message=f"Timed out waiting for job {job_id} to drain before replacement",
            )
        except TimeoutError as exc:
            raise ConnectError(Code.DEADLINE_EXCEEDED, str(exc)) from exc

    def _remove_finished_job_safely(self, cur, job_id: JobName) -> None:
        """``remove_finished_job`` gated on no unfinished worker-bound attempts.

        CASCADE-deleting a job's tasks while its attempts are still worker-
        bound destroys the rows the heartbeat path needs to find when it
        stamps ``finished_at_ms``. Every replacement / prune / admin-delete
        path goes through here so that contract is uniform.
        """
        if self._store.jobs.has_unfinished_worker_attempts(cur, job_id):
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Refusing to remove job {job_id}: worker-bound attempts have not finalized",
            )
        self._transitions.remove_finished_job(cur, job_id)

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

        job_id = JobName.from_wire(request.name)

        # Reject root RPC submissions from stale clients. Direct in-process
        # calls have no wire client; tests and harnesses use ctx=None.
        # Nested submissions are exempt because they come from an already
        # running workload, which would otherwise crash mid-flight as the
        # freshness window slides forward.
        if job_id.is_root and ctx is not None:
            _check_client_freshness(request.client_revision_date, date.today())

        # When an auth provider is configured, override the user segment with
        # the verified identity to prevent impersonation. Only override for
        # root-level submissions; child jobs inherit the parent's user.
        verified_user = get_verified_user()
        if self._auth.provider and verified_user is not None and job_id.is_root:
            job_id = JobName.root(verified_user, job_id.name)

        # For non-root jobs, verify the caller owns the parent hierarchy
        if self._auth.provider and verified_user is not None and not job_id.is_root:
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
            user_budget = self._db.get_user_budget(job_id.user)
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
        # transaction further down — between the two txs another submitter
        # can race, but ``INSERT INTO jobs`` then PK-conflicts, which is a
        # legitimate error rather than a correctness bug.
        needs_drain = False
        with self._store.transaction() as cur:
            existing_state = self._store.jobs.get_state(cur, job_id)
            if existing_state is not None:
                policy = request.existing_job_policy
                if policy == job_pb2.EXISTING_JOB_POLICY_ERROR:
                    raise ConnectError(
                        Code.ALREADY_EXISTS,
                        f"Job {job_id} already exists (state={job_pb2.JobState.Name(existing_state)})",
                    )
                elif policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
                    if not is_job_finished(existing_state):
                        return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
                    # Job finished, replace it (KEEP only preserves running jobs)
                    self._remove_finished_job_safely(cur, job_id)
                elif policy == job_pb2.EXISTING_JOB_POLICY_RECREATE:
                    if not is_job_finished(existing_state):
                        self._transitions.cancel_job(cur, job_id, "Replaced by new submission")
                        # Cancel is a producer transition: attempts stay
                        # unfinished until the worker confirms termination.
                        # Defer remove_finished_job to a second tx after the
                        # drain wait so we don't destroy task_attempts rows
                        # whose finished_at_ms write the heartbeat path is
                        # still racing to land.
                        needs_drain = True
                    else:
                        self._remove_finished_job_safely(cur, job_id)
                elif is_job_finished(existing_state):
                    # Default/UNSPECIFIED: replace finished jobs
                    logger.info(
                        "Replacing finished job %s (state=%s) with new submission",
                        job_id,
                        job_pb2.JobState.Name(existing_state),
                    )
                    self._remove_finished_job_safely(cur, job_id)
                else:
                    raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists and is still running")

        if needs_drain:
            # Nudge the polling loop so workers see the cancelled tasks excluded
            # from their expected set on the next reconcile and auto-kill the
            # containers; the heartbeat path then stamps finished_at_ms.
            self._controller.wake()
            self._wait_until_job_drained(job_id, _JOB_REPLACEMENT_DRAIN_TIMEOUT)
            with self._store.transaction() as cur:
                self._remove_finished_job_safely(cur, job_id)

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

            bundle_id = self._bundle_store.write_zip(request.bundle_blob)

            new_request = controller_pb2.Controller.LaunchJobRequest()
            new_request.CopyFrom(request)
            new_request.ClearField("bundle_blob")
            new_request.bundle_id = bundle_id
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

        # Reject jobs that can never be scheduled so they fail fast instead
        # of sitting in the pending queue. For coscheduled jobs this also
        # verifies the replica count is compatible with some group's num_vms.
        autoscaler = self._controller.autoscaler
        if autoscaler is not None:
            replicas = request.replicas if request.HasField("coscheduling") else None
            constraints = [Constraint.from_proto(c) for c in request.constraints]
            error = autoscaler.job_feasibility(
                constraints=constraints,
                replicas=replicas,
            )
            if error:
                raise ConnectError(
                    Code.FAILED_PRECONDITION,
                    f"Job {job_id} is unschedulable: {error} (constraints: {constraints})",
                )

        with self._store.transaction() as cur:
            self._transitions.submit_job(cur, job_id, request, Timestamp.now())
        self._controller.wake()

        with self._db.read_snapshot() as q:
            num_tasks = q.execute_sql("SELECT COUNT(*) FROM tasks WHERE job_id = ?", (job_id.to_wire(),)).fetchone()[0]
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
            job = _read_job(q, JobName.from_wire(request.job_id))
            if not job:
                raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")
            # Aggregate task counts via a single GROUP BY query.
            summaries = _task_summaries_for_jobs(q, {job.job_id})
            has_children = bool(_parent_ids_with_children(q, [job.job_id]))
        summary = summaries.get(job.job_id)

        task_state_counts = (
            {task_state_friendly(state): count for state, count in summary.task_state_counts.items()} if summary else {}
        )

        # Get scheduling diagnostics for pending jobs from cache
        # (populated each scheduling cycle by the controller). The autoscaler
        # hint dict is cached per evaluate() cycle (#4848), so the lookup here
        # is a single dict get — we only attach this job's hint, never the
        # full routing decision.
        pending_reason = ""
        if job.state == job_pb2.JOB_STATE_PENDING:
            sched_reason = self._controller.get_job_scheduling_diagnostics(job.job_id.to_wire())
            pending_reason = sched_reason or "Pending scheduler feedback"
            hint = self._get_autoscaler_pending_hints().get(job.job_id.to_wire())
            if hint is not None:
                scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
                pending_reason = f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

        resources = _resource_spec_from_job_row(job)

        proto_job_status = job_pb2.JobStatus(
            job_id=job.job_id.to_wire(),
            state=job.state,
            error=job.error or "",
            exit_code=job.exit_code or 0,
            failure_count=summary.failure_count if summary else 0,
            preemption_count=summary.preemption_count if summary else 0,
            name=job.name,
            pending_reason=pending_reason,
            task_state_counts=task_state_counts,
            task_count=summary.task_count if summary else 0,
            completed_count=summary.completed_count if summary else 0,
            resources=resources,
            has_children=has_children,
        )
        if job.started_at:
            proto_job_status.started_at.CopyFrom(timestamp_to_proto(job.started_at))
        if job.finished_at:
            proto_job_status.finished_at.CopyFrom(timestamp_to_proto(job.finished_at))
        if job.submitted_at:
            proto_job_status.submitted_at.CopyFrom(timestamp_to_proto(job.submitted_at))

        reconstructed_request = _reconstruct_launch_job_request(job)
        return controller_pb2.Controller.GetJobStatusResponse(
            job=proto_job_status,
            request=redact_request_env_vars(reconstructed_request),
        )

    @on_loop
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

        with self._db.read_snapshot() as q:
            placeholders = ",".join("?" for _ in wire_ids)
            rows = q.raw(
                f"SELECT job_id, state FROM jobs WHERE job_id IN ({placeholders})",
                tuple(wire_ids),
            )

        states = {row.job_id: row.state for row in rows}
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
        # cancel_job uses a recursive CTE to walk the full subtree in a single
        # transaction, so there is no need to recurse manually.
        with self._store.transaction() as cur:
            self._transitions.cancel_job(cur, job_id, reason="Terminated by user")
        # The next polling tick reconciles each affected worker and sends
        # StopTasks via the expected_tasks diff; wake the loops so it lands
        # within one tick rather than waiting on the next backoff.
        self._controller.wake()
        return job_pb2.Empty()

    def _job_to_proto(
        self,
        j: JobRow,
        task_summary: TaskJobSummary | None,
        autoscaler_pending_hints: dict[str, PendingHint],
        *,
        has_children: bool = False,
    ) -> job_pb2.JobStatus:
        """Convert a JobRow + its task summary into a JobStatus proto."""
        job_name = j.name
        task_state_counts = (
            {task_state_friendly(state): count for state, count in task_summary.task_state_counts.items()}
            if task_summary
            else {}
        )

        pending_reason = j.error or ""
        if j.state == job_pb2.JOB_STATE_PENDING:
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
            failure_count=task_summary.failure_count if task_summary else 0,
            preemption_count=task_summary.preemption_count if task_summary else 0,
            name=job_name,
            task_state_counts=task_state_counts,
            task_count=task_summary.task_count if task_summary else 0,
            completed_count=task_summary.completed_count if task_summary else 0,
            pending_reason=pending_reason,
            has_children=has_children,
        )
        if j.started_at:
            proto_job.started_at.CopyFrom(timestamp_to_proto(j.started_at))
        if j.finished_at:
            proto_job.finished_at.CopyFrom(timestamp_to_proto(j.finished_at))
        if j.submitted_at:
            proto_job.submitted_at.CopyFrom(timestamp_to_proto(j.submitted_at))
        return proto_job

    def _jobs_to_protos(
        self,
        jobs: list[JobRow],
        task_summaries: dict[JobName, TaskJobSummary],
        autoscaler_pending_hints: dict[str, PendingHint],
        has_children: set[JobName] | None = None,
    ) -> list[job_pb2.JobStatus]:
        child_parent_ids = has_children or set()
        return [
            self._job_to_proto(
                j,
                task_summaries.get(j.job_id),
                autoscaler_pending_hints,
                has_children=j.job_id in child_parent_ids,
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
        if query.scope == controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN and not query.parent_job_id:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                "query.parent_job_id is required for JOB_QUERY_SCOPE_CHILDREN",
            )

        with self._db.read_snapshot() as q:
            page, total_count = _query_jobs(q, query, state_ids)
            page_ids = [j.job_id for j in page]
            summaries = _task_summaries_for_jobs(q, set(page_ids)) if page_ids else {}
            children = _parent_ids_with_children(q, page_ids) if page_ids else set()

        has_pending = any(j.state == job_pb2.JOB_STATE_PENDING for j in page)
        autoscaler_pending_hints = self._get_autoscaler_pending_hints() if has_pending else {}
        all_jobs = self._jobs_to_protos(page, summaries, autoscaler_pending_hints, has_children=children)
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
        with self._db.read_snapshot() as q:
            jc_row = q.raw(
                "SELECT jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json "
                "FROM job_config jc WHERE jc.job_id = ?",
                (task.job_id.to_wire(),),
            )
        if jc_row:
            row = jc_row[0]
            if row.res_cpu_millicores or row.res_memory_bytes or row.res_disk_bytes or row.res_device_json:
                job_resources = resource_spec_from_scalars(
                    row.res_cpu_millicores, row.res_memory_bytes, row.res_disk_bytes, row.res_device_json
                )

        proto.status_text_detail_md = self._store.tasks.get_status_text_detail(task_id.to_wire())
        proto.status_text_summary_md = self._store.tasks.get_status_text_summary(task_id.to_wire())

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
        tasks = _tasks_for_listing(self._db, job_id=job_id)
        worker_addr_by_id = _worker_addresses_for_tasks(self._db, tasks)

        task_statuses = []
        for task in tasks:
            twid = _task_worker_id(task)
            proto_task_status = task_to_proto(task, worker_address=worker_addr_by_id.get(twid, "") if twid else "")

            # Don't add scheduling diagnostics in list view - too expensive
            # Users should check job detail page for scheduling diagnostics
            if task.state == job_pb2.TASK_STATE_PENDING:
                proto_task_status.can_be_scheduled = task_row_can_be_scheduled(task)

            proto_task_status.status_text_summary_md = self._store.tasks.get_status_text_summary(task.task_id.to_wire())

            task_statuses.append(proto_task_status)

        return controller_pb2.Controller.ListTasksResponse(tasks=task_statuses)

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
            authorize(AuthzAction.REGISTER_WORKER)

        if not request.worker_id:
            logger.error("Worker at %s registered without worker_id", request.address)
            return controller_pb2.Controller.RegisterResponse(
                worker_id="",
                accepted=False,
            )
        worker_id = WorkerId(request.worker_id)

        with self._store.transaction() as cur:
            self._transitions.register_or_refresh_worker(
                cur,
                worker_id=worker_id,
                address=request.address,
                metadata=request.metadata,
                ts=Timestamp.now(),
                slice_id=request.slice_id,
                scale_group=request.scale_group,
            )
        logger.info("Worker registered: %s at %s", worker_id, request.address)
        return controller_pb2.Controller.RegisterResponse(
            worker_id=str(worker_id),
            accepted=True,
        )

    def list_workers(
        self,
        request: controller_pb2.Controller.ListWorkersRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListWorkersResponse:
        """List workers with their running task counts.

        Served directly from the workers table (cluster size is in the low
        thousands at most), with liveness queried from
        :class:`~iris.cluster.controller.worker_health.WorkerHealthTracker` and
        a single per-page running-task lookup. ``query.limit == 0`` disables
        paging (preserves CLI callers that fetch the whole roster); ``limit > 0``
        is clamped to ``MAX_LIST_WORKERS_LIMIT``.
        """
        if self._controller.has_direct_provider:
            return controller_pb2.Controller.ListWorkersResponse()

        query = controller_pb2.Controller.WorkerQuery()
        if request.HasField("query"):
            query.CopyFrom(request.query)

        workers_all = _worker_roster(self._store)
        liveness_by_id = self._store.health.liveness_many(w.worker_id for w in workers_all)
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

        running = running_tasks_by_worker(self._db, {w.worker_id for w in page_rows}) if page_rows else {}
        workers = []
        for worker in page_rows:
            liveness = liveness_by_id[worker.worker_id]
            workers.append(
                controller_pb2.Controller.WorkerHealthStatus(
                    worker_id=worker.worker_id,
                    healthy=liveness.healthy,
                    consecutive_failures=liveness.consecutive_failures,
                    last_heartbeat=timestamp_to_proto(Timestamp.from_ms(liveness.last_heartbeat_ms)),
                    running_job_ids=[task_id.to_wire() for task_id in running.get(worker.worker_id, set())],
                    address=worker.address,
                    metadata=_worker_metadata_to_proto(worker),
                    status_message=worker_status_message(liveness),
                )
            )
        return controller_pb2.Controller.ListWorkersResponse(
            workers=workers,
            total_count=total_count,
            has_more=has_more,
        )

    # --- Endpoint Management ---

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        """Register a service endpoint.

        The ``task_id`` field carries the calling task's wire-format task ID
        (e.g. ``/user/job/0``).  The endpoint is associated with the owning
        task so that retry cleanup removes stale endpoints from earlier
        attempts.

        Endpoints are registered regardless of job state, but only become
        visible to clients (via lookup/list) when the job is executing (not
        in a terminal state).
        """
        endpoint_id = request.endpoint_id or str(uuid.uuid4())

        task_id = JobName.from_wire(request.task_id)
        task_id.require_task()

        endpoint = EndpointRow(
            endpoint_id=endpoint_id,
            name=request.name,
            address=request.address,
            task_id=task_id,
            metadata=dict(request.metadata),
            registered_at=Timestamp.now(),
        )

        # Validation runs inside the writer transaction in
        # :meth:`EndpointStore.add`: NOT_FOUND if the task row is missing,
        # FAILED_PRECONDITION if the task is terminal or the attempt is stale.
        with self._store.transaction() as cur:
            outcome = self._transitions.add_endpoint(cur, endpoint, expected_attempt_id=request.attempt_id)
        if outcome is AddEndpointOutcome.NOT_FOUND:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")
        if outcome is AddEndpointOutcome.STALE_ATTEMPT:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Stale attempt for task {request.task_id} (attempt {request.attempt_id})",
            )
        if outcome is AddEndpointOutcome.TERMINAL:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Task {request.task_id} is already terminal; endpoint not registered",
            )

        return controller_pb2.Controller.RegisterEndpointResponse(endpoint_id=endpoint_id)

    def unregister_endpoint(
        self,
        request: controller_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        """Unregister a service endpoint. Idempotent."""
        with self._store.transaction() as cur:
            self._transitions.remove_endpoint(cur, request.endpoint_id)
        return job_pb2.Empty()

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        """List endpoints by name prefix (or exact name when request.exact is set).

        System endpoints (names starting with ``/system/``) are resolved from
        an in-memory map rather than the DB.  This allows system services like
        the LogService to be discovered via the same API as job-scoped actors.
        """
        prefix = request.prefix
        if prefix.startswith("/system/"):
            return self._list_system_endpoints(prefix, exact=request.exact)

        endpoints = self._store.endpoints.query(
            EndpointQuery(
                exact_name=prefix if request.exact else None,
                name_prefix=None if request.exact else prefix,
            ),
        )
        return controller_pb2.Controller.ListEndpointsResponse(
            endpoints=[
                controller_pb2.Controller.Endpoint(
                    endpoint_id=e.endpoint_id,
                    name=e.name,
                    address=e.address,
                    task_id=e.task_id.to_wire(),
                    metadata=e.metadata,
                )
                for e in endpoints
            ]
        )

    def _list_system_endpoints(self, prefix: str, *, exact: bool) -> controller_pb2.Controller.ListEndpointsResponse:
        """Resolve system endpoints from the in-memory map."""
        results: list[controller_pb2.Controller.Endpoint] = []
        for name, address in self._system_endpoints.items():
            if exact and name == prefix:
                results.append(
                    controller_pb2.Controller.Endpoint(
                        endpoint_id=name,
                        name=name,
                        address=address,
                    )
                )
            elif not exact and name.startswith(prefix):
                results.append(
                    controller_pb2.Controller.Endpoint(
                        endpoint_id=name,
                        name=name,
                        address=address,
                    )
                )
        return controller_pb2.Controller.ListEndpointsResponse(endpoints=results)

    # --- Autoscaler ---

    def get_autoscaler_status(
        self,
        request: controller_pb2.Controller.GetAutoscalerStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetAutoscalerStatusResponse:
        """Get current autoscaler status with worker info populated."""
        if self._controller.has_direct_provider:
            return controller_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())
        autoscaler = self._controller.autoscaler
        if not autoscaler:
            return controller_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())

        status = autoscaler.get_status()

        workers = _worker_roster(self._store)
        liveness_by_id = self._store.health.liveness_many(w.worker_id for w in workers)
        worker_id_to_health: dict[str, bool] = {str(w.worker_id): liveness_by_id[w.worker_id].healthy for w in workers}

        # The vm_ids appearing in the autoscaler status are the only candidates
        # for the running-task lookup; restrict to those known to be in the
        # roster to keep the IN-clause bounded by visible VMs, not roster size.
        vm_ids = {vm.vm_id for group in status.groups for slice_info in group.slices for vm in slice_info.vms}
        candidate_ids = {WorkerId(vid) for vid in vm_ids if vid in worker_id_to_health}
        running = running_tasks_by_worker(self._db, candidate_ids) if candidate_ids else {}

        for group in status.groups:
            for slice_info in group.slices:
                for vm in slice_info.vms:
                    healthy = worker_id_to_health.get(vm.vm_id)
                    if healthy is None:
                        continue
                    vm.worker_id = vm.vm_id
                    vm.worker_healthy = healthy
                    vm.running_task_count = len(running.get(WorkerId(vm.vm_id), set()))

        return controller_pb2.Controller.GetAutoscalerStatusResponse(status=status)

    # --- Provider Status ---

    def get_provider_status(
        self,
        request: controller_pb2.Controller.GetProviderStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetProviderStatusResponse:
        """Get provider status for direct-dispatch providers."""
        if not self._controller.has_direct_provider:
            return controller_pb2.Controller.GetProviderStatusResponse(has_direct_provider=False)
        events = [
            controller_pb2.Controller.SchedulingEvent(
                task_id=e.task_id,
                attempt_id=e.attempt_id,
                event_type=e.event_type,
                reason=e.reason,
                message=e.message,
                timestamp=timestamp_to_proto(e.timestamp),
            )
            for e in self._controller.provider_scheduling_events
        ]
        resp = controller_pb2.Controller.GetProviderStatusResponse(
            has_direct_provider=True,
            scheduling_events=events,
        )
        cap = self._controller.provider_capacity
        if cap is not None:
            resp.capacity.CopyFrom(
                controller_pb2.Controller.ClusterCapacity(
                    schedulable_nodes=cap.schedulable_nodes,
                    total_cpu_millicores=cap.total_cpu_millicores,
                    available_cpu_millicores=cap.available_cpu_millicores,
                    total_memory_bytes=cap.total_memory_bytes,
                    available_memory_bytes=cap.available_memory_bytes,
                )
            )
        return resp

    # --- Kubernetes Cluster Status ---

    def get_kubernetes_cluster_status(
        self,
        request: controller_pb2.Controller.GetKubernetesClusterStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Get Kubernetes cluster status: node counts, capacity, and recent pod statuses."""
        if not self._controller.has_direct_provider:
            return controller_pb2.Controller.GetKubernetesClusterStatusResponse()

        # KubernetesProvider exposes get_cluster_status().
        # Access via the provider after the guard.
        provider = self._controller.provider
        return provider.get_cluster_status()  # type: ignore[union-attr]

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
            worker = _read_worker(self._store, WorkerId(worker_id_str))
            if not worker:
                raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id_str} not found")
            if not self._store.health.liveness(worker.worker_id).healthy:
                raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id_str} is unavailable")
            forwarded = job_pb2.ProfileTaskRequest(
                target="/system/process",
                duration_seconds=request.duration_seconds,
                profile_type=request.profile_type,
            )
            timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
            resp = self._controller.provider.profile_task(worker.address, forwarded, timeout_ms)
            return job_pb2.ProfileTaskResponse(
                profile_data=resp.profile_data,
                error=resp.error,
            )

        # Task target: parse optional :attempt_id, validate, proxy to worker
        try:
            target = parse_profile_target(request.target)
            target.task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        task = _read_task_with_attempts(self._db, target.task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.target} not found")

        task_worker_id = _task_worker_id(task)
        if not task_worker_id:
            if self._controller.has_direct_provider:
                provider = self._controller.provider
                attempt_id = target.attempt_id if target.attempt_id is not None else task.current_attempt_id
                resp = provider.profile_task(task.task_id.to_wire(), attempt_id, request)
                return job_pb2.ProfileTaskResponse(
                    profile_data=resp.profile_data,
                    error=resp.error,
                )
            raise ConnectError(Code.FAILED_PRECONDITION, f"Task {request.target} not yet assigned to a worker")

        worker = _read_worker(self._store, task_worker_id)
        if not worker or not self._store.health.liveness(task_worker_id).healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {task_worker_id} is unavailable")

        timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
        resp = self._controller.provider.profile_task(worker.address, request, timeout_ms)
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
        if self._controller.has_direct_provider:
            raise ConnectError(Code.UNIMPLEMENTED, "Direct provider mode: no workers")
        if not request.id:
            raise ConnectError(Code.INVALID_ARGUMENT, "id is required")

        detail = _read_worker_detail(self._store, WorkerId(str(request.id)))
        if not detail:
            raise ConnectError(Code.NOT_FOUND, f"No worker found for '{request.id}'")

        worker = detail.worker
        liveness = self._store.health.liveness(worker.worker_id)
        worker_health = controller_pb2.Controller.WorkerHealthStatus(
            worker_id=worker.worker_id,
            healthy=liveness.healthy,
            consecutive_failures=liveness.consecutive_failures,
            last_heartbeat=timestamp_to_proto(Timestamp.from_ms(liveness.last_heartbeat_ms)),
            running_job_ids=[tid.to_wire() for tid in detail.running_tasks],
            address=worker.address,
            metadata=_worker_metadata_to_proto(worker),
            status_message=worker_status_message(liveness),
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

        worker = _read_worker(self._store, WorkerId(worker_id))
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
        if not self._store.health.liveness(worker.worker_id).healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")

        try:
            return self._controller.provider.get_process_status(WorkerId(worker_id), worker.address, request)
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
        self._db.ensure_user(username, now)
        role = self._db.get_user_role(username)

        # Revoke old login keys and propagate to in-memory revocation set
        revoked_ids = revoke_login_keys_for_user(self._db, username, now)
        for jti in revoked_ids:
            self._auth.jwt_manager.revoke(jti)

        key_id = f"iris_k_{secrets.token_urlsafe(8)}"
        expires_at = Timestamp.from_ms(now.epoch_ms() + DEFAULT_JWT_TTL_SECONDS * 1000)
        create_api_key(
            self._db,
            key_id=key_id,
            key_hash=f"jwt:{key_id}",
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
        self._db.ensure_user(target_user, now)
        role = self._db.get_user_role(target_user)

        key_id = f"iris_k_{secrets.token_urlsafe(8)}"
        ttl = request.ttl_ms // 1000 if request.ttl_ms > 0 else DEFAULT_JWT_TTL_SECONDS
        # Always persist the actual JWT expiry so the DB and token agree.
        expires_at = Timestamp.from_ms(now.epoch_ms() + ttl * 1000)

        create_api_key(
            self._db,
            key_id=key_id,
            key_hash=f"jwt:{key_id}",
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
        with self._db.snapshot() as q:
            key = API_KEY_PROJECTION.decode_one(
                q.fetchall("SELECT * FROM api_keys ak WHERE ak.key_id = ?", (request.key_id,))
            )
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
                    created_at_ms=k.created_at.epoch_ms(),
                    last_used_at_ms=k.last_used_at.epoch_ms() if k.last_used_at else 0,
                    expires_at_ms=k.expires_at.epoch_ms() if k.expires_at else 0,
                    revoked=k.revoked_at is not None,
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

        task_worker_id = _task_worker_id(task)
        if not task_worker_id:
            if self._controller.has_direct_provider:
                provider = self._controller.provider
                timeout = request.timeout_seconds if request.timeout_seconds else 60
                resp = provider.exec_in_container(
                    task.task_id.to_wire(), task.current_attempt_id, list(request.command), timeout
                )
                return controller_pb2.Controller.ExecInContainerResponse(
                    exit_code=resp.exit_code,
                    stdout=resp.stdout,
                    stderr=resp.stderr,
                    error=resp.error,
                )
            raise ConnectError(Code.FAILED_PRECONDITION, f"Task {request.task_id} not assigned to a worker")

        worker = _read_worker(self._store, task_worker_id)
        if not worker or not self._store.health.liveness(task_worker_id).healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {task_worker_id} is unavailable")

        # Proxy to worker
        worker_request = worker_pb2.Worker.ExecInContainerRequest(
            task_id=request.task_id,
            command=request.command,
            timeout_seconds=request.timeout_seconds,
        )
        resp = self._controller.provider.exec_in_container(worker.address, worker_request, request.timeout_seconds)
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
        result = execute_raw_query(self._db, request.sql)
        return query_pb2.RawQueryResponse(
            columns=result.columns,
            rows=result.rows,
        )

    def restart_worker(
        self,
        request: controller_pb2.Controller.RestartWorkerRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RestartWorkerResponse:
        """Restart a worker while preserving its running containers.

        Delegates to the worker's platform handle which knows how to restart
        the worker process (e.g., `docker restart` on GCE). The new worker
        discovers and adopts existing task containers via Docker labels.
        """
        require_identity()
        worker_id = request.worker_id
        if not worker_id:
            return controller_pb2.Controller.RestartWorkerResponse(accepted=False, error="worker_id is required")

        autoscaler = self._controller.autoscaler
        if autoscaler is None:
            return controller_pb2.Controller.RestartWorkerResponse(accepted=False, error="autoscaler not configured")

        try:
            autoscaler.restart_worker(worker_id)
            logger.info("Initiated restart for worker %s", worker_id)
            return controller_pb2.Controller.RestartWorkerResponse(accepted=True)
        except Exception as e:
            logger.warning("Failed to restart worker %s: %s", worker_id, e)
            return controller_pb2.Controller.RestartWorkerResponse(accepted=False, error=str(e))

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
        # Ensure the user row exists (FK on user_budgets → users)
        self._db.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at_ms) VALUES (?, ?)",
            (request.user_id, now.epoch_ms()),
        )
        self._db.set_user_budget(
            user_id=request.user_id,
            budget_limit=request.budget_limit,
            max_band=max_band,
            now=now,
        )
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
        budget = self._db.get_user_budget(request.user_id)
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
        budgets = self._db.list_user_budgets()
        with self._db.read_snapshot() as snap:
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

        budgets = self._db.list_user_budgets()
        budget_limits: dict[str, int] = {b.user_id: b.budget_limit for b in budgets}

        with self._db.read_snapshot() as snap:
            user_spend = compute_user_spend(snap)

            # Pending tasks: full TASK_ROW_PROJECTION so ``task_row_can_be_scheduled``
            # can apply the schedulable filter (excludes retry-exhausted
            # PENDING tasks). No ORDER BY — we aggregate, not display.
            pending_rows = TASK_ROW_PROJECTION.decode(
                snap.fetchall(
                    f"SELECT {TASK_ROW_PROJECTION.select_clause()} FROM tasks t WHERE t.state = ?",
                    (job_pb2.TASK_STATE_PENDING,),
                ),
            )
            pending_requested_bands = self._store.jobs.get_priority_bands(snap, {row.job_id for row in pending_rows})

            # Running tasks: only task_id, priority_band, and worker — no
            # job_config join is needed for the rolled-up counts below.
            running_rows = snap.raw(
                "SELECT t.task_id, t.priority_band, t.current_worker_id AS worker_id "
                "FROM tasks t "
                "WHERE t.state = ? AND t.current_worker_id IS NOT NULL",
                (job_pb2.TASK_STATE_RUNNING,),
                decoders={"task_id": JobName.from_wire, "priority_band": int, "worker_id": WorkerId},
            )

        # Aggregate pending into (band, user, job) → count buckets.
        pending_counts: dict[tuple[int, str, str], int] = {}
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
            key = (eff_band, user_id, job_id)
            pending_counts[key] = pending_counts.get(key, 0) + 1
            total_pending += 1

        # Aggregate running into (band, user, worker, job) → count buckets.
        # Use the stamped ``tasks.priority_band`` directly: the scheduler stamps the
        # effective band at assign time (see ``_commit_assignments``), so re-running
        # ``compute_effective_band`` here against current spend would double-demote.
        running_counts: dict[tuple[int, str, str, str], int] = {}
        total_running = 0
        for row in running_rows:
            user_id = row.task_id.user
            job_id = (row.task_id.parent or row.task_id).to_wire()
            key = (row.priority_band, user_id, str(row.worker_id), job_id)
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
                count=count,
            )
            for (band, user_id, job_id), count in pending_counts.items()
        ]
        running_buckets = [
            controller_pb2.Controller.RunningTaskBucket(
                band=band,
                user_id=user_id,
                worker_id=worker_id,
                job_id=job_id,
                count=count,
            )
            for (band, user_id, worker_id, job_id), count in running_counts.items()
        ]

        return controller_pb2.Controller.GetSchedulerStateResponse(
            user_budgets=budget_protos,
            total_pending=total_pending,
            total_running=total_running,
            pending_buckets=pending_buckets,
            running_buckets=running_buckets,
        )

    # --- Worker Push ---

    def update_task_status(
        self,
        request: controller_pb2.Controller.UpdateTaskStatusRequest,
        _ctx: Any,
    ) -> controller_pb2.Controller.UpdateTaskStatusResponse:
        """Worker pushes task state transitions to controller.

        Converts the proto updates into TaskUpdate dataclasses and applies
        them via ``ControllerTransitions.apply_task_updates``. Stop decisions
        are delivered via the StopTasks RPC, not piggy-backed on the response.

        The kill decisions produced here are ignored: the poll loop reruns the
        same transition logic and routes kills through ``_stop_tasks_direct``,
        so push-path kills are recovered with ≤60s latency.
        """
        updates = task_updates_from_proto(request.updates)
        if updates:
            with self._store.transaction() as cur:
                self._transitions.apply_task_updates(
                    cur,
                    HeartbeatApplyRequest(
                        worker_id=WorkerId(request.worker_id),
                        updates=updates,
                    ),
                )
            self._controller.wake()
        return controller_pb2.Controller.UpdateTaskStatusResponse()

    # --- Task Status Text Push ---

    @on_loop
    def set_task_status_text(
        self,
        request: job_pb2.SetTaskStatusTextRequest,
        _ctx: Any,
    ) -> job_pb2.SetTaskStatusTextResponse:
        """Task pushes a markdown status string to the coordinator.

        Status text lives entirely in the in-memory ``TaskStore`` dict; the
        write is idempotent and stale task IDs are evicted by
        ``remove_status_text_by_job_ids`` during pruning.
        """
        task_id = JobName.from_wire(request.task_id)
        self._transitions.record_task_status_text(task_id, request.status_text_detail_md, request.status_text_summary_md)
        return job_pb2.SetTaskStatusTextResponse()
