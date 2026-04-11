# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller RPC service implementation handling job, task, and worker operations.

The controller expands jobs into tasks at submission time (a job with replicas=N
creates N tasks). Tasks are the unit of scheduling and execution. Job state is
aggregated from task states.
"""

import json
import logging
import re
import secrets
import uuid
import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.constraints import Constraint, constraints_from_resources, merge_constraints
from iris.cluster.controller.codec import (
    constraints_from_json,
    proto_from_json,
    reservation_entries_from_json,
    resource_spec_from_scalars,
)
from iris.cluster.controller.budget import (
    UserTask,
    compute_effective_band,
    compute_user_spend,
    interleave_by_user,
    resource_value,
)
from iris.rpc.proto_utils import priority_band_name
from iris.cluster.controller.auth import (
    DEFAULT_JWT_TTL_SECONDS,
    ControllerAuth,
    create_api_key,
    list_api_keys,
    revoke_api_key,
    revoke_login_keys_for_user,
)
from iris.rpc.auth import (
    AuthzAction,
    authorize,
    authorize_resource_owner,
    get_verified_identity,
    get_verified_user,
    require_identity,
)
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    ControllerDB,
    EndpointQuery,
    TaskJobSummary,
    UserStats,
    attempt_is_worker_failure,
    endpoint_query_sql,
    job_is_finished,
    running_tasks_by_worker,
    task_row_can_be_scheduled,
)
from iris.cluster.controller.schema import (
    API_KEY_PROJECTION,
    ATTEMPT_PROJECTION,
    ENDPOINT_PROJECTION,
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    JOB_ROW_PROJECTION,
    TASK_DETAIL_PROJECTION,
    TASK_ROW_PROJECTION,
    TXN_ACTION_PROJECTION,
    WORKER_DETAIL_PROJECTION,
    AttemptRow,
    EndpointRow,
    JobDetailRow,
    JobRow,
    TaskDetailRow,
    TransactionActionRow,
    WorkerDetailRow,
    WorkerRow,
    tasks_with_attempts,
)
from iris.cluster.controller.autoscaler.status import PendingHint, build_job_pending_hints
from iris.cluster.controller.query import execute_raw_query
from iris.rpc import query_pb2
from iris.cluster.controller.scheduler import SchedulingContext
from iris.cluster.controller.transitions import ControllerTransitions, TASK_RESOURCE_HISTORY_RETENTION
from iris.cluster.controller.provider import ProviderError
from iris.cluster.log_store import build_log_source, worker_log_key
from iris.cluster.process_status import get_process_status
from iris.cluster.runtime.profile import is_system_target, parse_profile_target, profile_local_process
from iris.cluster.types import JobName, WorkerId, get_gpu_count, get_tpu_count
from iris.log_server.server import LogServiceImpl
from iris.rpc import logging_pb2, vm_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc import worker_pb2
from iris.rpc.proto_utils import job_state_friendly, job_state_name, task_state_name
from iris.time_proto import timestamp_to_proto
from rigging.timing import Timestamp, Timer

logger = logging.getLogger(__name__)

DEFAULT_TRANSACTION_LIMIT = 50

# Pattern for env var keys that contain secrets and should be redacted in API responses.
_SENSITIVE_ENV_KEY_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
REDACTED_VALUE = "**REDACTED**"


def redact_request_env_vars(
    request: controller_pb2.Controller.LaunchJobRequest,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Return a copy of *request* with sensitive env var values replaced."""
    if not request.environment.env_vars:
        return request

    redacted = controller_pb2.Controller.LaunchJobRequest()
    redacted.CopyFrom(request)
    env_vars = redacted.environment.env_vars
    for key in list(env_vars):
        if _SENSITIVE_ENV_KEY_RE.search(key):
            env_vars[key] = REDACTED_VALUE
    return redacted


DEFAULT_MAX_TOTAL_LINES = 100000

# Maximum bundle size in bytes (25 MB) - matches client-side limit
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024

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

    Handles attempt conversion, timestamps, and resource_usage.
    The caller is responsible for resolving worker_address from worker_id if needed.
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
    has_resource_usage = (
        task.resource_usage_memory_mb is not None
        or task.resource_usage_cpu_millicores is not None
        or task.resource_usage_disk_mb is not None
        or task.resource_usage_memory_peak_mb is not None
        or task.resource_usage_process_count is not None
    )
    if has_resource_usage:
        proto.resource_usage.CopyFrom(
            job_pb2.ResourceUsage(
                memory_mb=task.resource_usage_memory_mb or 0,
                disk_mb=task.resource_usage_disk_mb or 0,
                cpu_millicores=task.resource_usage_cpu_millicores or 0,
                memory_peak_mb=task.resource_usage_memory_peak_mb or 0,
                process_count=task.resource_usage_process_count or 0,
            )
        )
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


def worker_status_message(w: WorkerDetailRow) -> str:
    """Build a human-readable status message for unhealthy workers."""
    if w.healthy:
        return ""
    if w.consecutive_failures > 0:
        age = w.last_heartbeat.age_ms()
        return f"Heartbeat timeout ({w.consecutive_failures} failures, last seen {age // 1000}s ago)"
    return "Unhealthy (no failures recorded)"


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


def _task_state_key(state: int) -> str:
    """Return the lowercase RPC key for a task state enum."""
    return task_state_name(state).removeprefix("TASK_STATE_").lower()


def _job_state_key(state: int) -> str:
    """Return the lowercase RPC key for a job state enum."""
    return job_state_name(state).removeprefix("JOB_STATE_").lower()


def _active_job_count(job_state_counts: dict[int, int]) -> int:
    """Return the count of non-terminal jobs in a user aggregate."""
    return sum(count for state, count in job_state_counts.items() if state not in TERMINAL_JOB_STATES)


def _task_state_counts_for_summary(task_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed task counts to the string-keyed RPC shape."""
    counts = {_task_state_key(state): 0 for state in USER_TASK_STATES}
    for state, count in task_state_counts.items():
        counts[_task_state_key(state)] = count
    return counts


def _job_state_counts_for_summary(job_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed job counts to the string-keyed RPC shape."""
    counts = {_job_state_key(state): 0 for state in USER_JOB_STATES}
    for state, count in job_state_counts.items():
        counts[_job_state_key(state)] = count
    return counts


# =============================================================================
# DB query helpers — thin wrappers over snapshot() for common read patterns
# =============================================================================


def _read_job(db: ControllerDB, job_id: JobName) -> JobDetailRow | None:
    with db.read_snapshot() as q:
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


def _read_worker(db: ControllerDB, worker_id: WorkerId) -> WorkerDetailRow | None:
    with db.read_snapshot() as q:
        return WORKER_DETAIL_PROJECTION.decode_one(
            q.fetchall(
                f"SELECT {WORKER_DETAIL_PROJECTION.select_clause()} FROM workers w WHERE w.worker_id = ?",
                (str(worker_id),),
            )
        )


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


def _snapshot_row_to_proto(row: Any) -> job_pb2.WorkerResourceSnapshot:
    """Reconstruct a WorkerResourceSnapshot proto from scalar columns."""
    snap = job_pb2.WorkerResourceSnapshot()
    if row.snapshot_host_cpu_percent is not None:
        snap.host_cpu_percent = row.snapshot_host_cpu_percent
    if row.snapshot_memory_used_bytes is not None:
        snap.memory_used_bytes = row.snapshot_memory_used_bytes
    if row.snapshot_memory_total_bytes is not None:
        snap.memory_total_bytes = row.snapshot_memory_total_bytes
    if row.snapshot_disk_used_bytes is not None:
        snap.disk_used_bytes = row.snapshot_disk_used_bytes
    if row.snapshot_disk_total_bytes is not None:
        snap.disk_total_bytes = row.snapshot_disk_total_bytes
    if row.snapshot_running_task_count is not None:
        snap.running_task_count = row.snapshot_running_task_count
    if row.snapshot_total_process_count is not None:
        snap.total_process_count = row.snapshot_total_process_count
    if row.snapshot_net_recv_bps is not None:
        snap.net_recv_bps = row.snapshot_net_recv_bps
    if row.snapshot_net_sent_bps is not None:
        snap.net_sent_bps = row.snapshot_net_sent_bps
    # timestamp_ms is present when reading from worker_resource_history.
    ts = getattr(row, "timestamp_ms", None)
    if ts is not None:
        snap.timestamp.epoch_ms = ts
    return snap


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
        req.constraints.append(c)
    for port in json.loads(job.ports_json):
        req.ports.append(port)

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


@dataclass(frozen=True)
class _WorkerDetail:
    worker: WorkerDetailRow
    running_tasks: frozenset[JobName]
    resource_history: tuple[job_pb2.WorkerResourceSnapshot, ...]


def _read_worker_detail(
    db: ControllerDB, worker_id: WorkerId, *, resource_history_limit: int = 200
) -> _WorkerDetail | None:
    with db.read_snapshot() as q:
        worker = WORKER_DETAIL_PROJECTION.decode_one(
            q.fetchall(
                f"SELECT {WORKER_DETAIL_PROJECTION.select_clause()} FROM workers w WHERE w.worker_id = ?",
                (str(worker_id),),
            ),
        )
        if worker is None:
            return None
        attr_rows = q.fetchall(
            "SELECT key, value_type, str_value, int_value, float_value " "FROM worker_attributes WHERE worker_id = ?",
            (str(worker_id),),
        )
        attrs: dict[str, str | int | float] = {}
        for row in attr_rows:
            vtype = str(row["value_type"])
            if vtype == "str":
                attrs[str(row["key"])] = str(row["str_value"])
            elif vtype == "int":
                attrs[str(row["key"])] = int(row["int_value"])
            elif vtype == "float":
                attrs[str(row["key"])] = float(row["float_value"])
        if attrs:
            worker = dataclasses.replace(worker, attributes=attrs)
        running_rows = q.raw(
            "SELECT t.task_id FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "WHERE a.worker_id = ? AND t.state IN (?, ?, ?)",
            (str(worker_id), *ACTIVE_TASK_STATES),
            decoders={"task_id": JobName.from_wire},
        )
        resource_rows = q.raw(
            "SELECT wrh.snapshot_host_cpu_percent, wrh.snapshot_memory_used_bytes, "
            "wrh.snapshot_memory_total_bytes, wrh.snapshot_disk_used_bytes, "
            "wrh.snapshot_disk_total_bytes, wrh.snapshot_running_task_count, "
            "wrh.snapshot_total_process_count, wrh.snapshot_net_recv_bps, "
            "wrh.snapshot_net_sent_bps, wrh.timestamp_ms "
            "FROM worker_resource_history wrh "
            "WHERE wrh.worker_id = ? ORDER BY wrh.id DESC LIMIT ?",
            (str(worker_id), max(resource_history_limit, 0)),
        )
    resource_history = tuple(reversed([_snapshot_row_to_proto(r) for r in resource_rows]))
    return _WorkerDetail(
        worker=worker,
        running_tasks=frozenset(r.task_id for r in running_rows),
        resource_history=resource_history,
    )


def _tasks_for_listing(db: ControllerDB, *, job_id: JobName) -> list[TaskDetailRow]:
    with db.read_snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall(
                f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} "
                "FROM tasks t WHERE t.job_id = ? ORDER BY t.job_id ASC, t.task_index ASC",
                (job_id.to_wire(),),
            ),
        )
        if not tasks:
            return []
        task_wires = [t.task_id.to_wire() for t in tasks]
        placeholders = ",".join("?" for _ in task_wires)
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
                f"WHERE ta.task_id IN ({placeholders}) "
                "ORDER BY ta.task_id ASC, ta.attempt_id ASC",
                tuple(task_wires),
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


# State display order for sorting (active states first)
_STATE_SORT_EXPR = (
    "CASE j.state"
    " WHEN 3 THEN 0"  # RUNNING
    " WHEN 2 THEN 1"  # BUILDING
    " WHEN 1 THEN 2"  # PENDING
    " WHEN 4 THEN 3"  # SUCCEEDED
    " WHEN 5 THEN 4"  # FAILED
    " WHEN 6 THEN 5"  # KILLED
    " WHEN 7 THEN 6"  # WORKER_FAILED
    " WHEN 8 THEN 7"  # UNSCHEDULABLE
    " ELSE 99 END"
)

_SORT_FIELD_TO_SQL: dict[int, str] = {
    controller_pb2.Controller.JOB_SORT_FIELD_DATE: "j.submitted_at_ms",
    controller_pb2.Controller.JOB_SORT_FIELD_NAME: "j.name",
    controller_pb2.Controller.JOB_SORT_FIELD_STATE: _STATE_SORT_EXPR,
    controller_pb2.Controller.JOB_SORT_FIELD_FAILURES: "agg_failures",
    controller_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS: "agg_preemptions",
}


MAX_LIST_JOBS_LIMIT = 500


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
    db: ControllerDB,
    query: controller_pb2.Controller.JobQuery,
    state_ids: tuple[int, ...],
) -> tuple[list[JobRow], int]:
    """Execute a ``JobQuery`` and return ``(rows, total_count)``.

    ``state_ids`` is the pre-resolved state filter (always non-empty); the
    caller owns "unknown state -> empty page" handling so that a bad filter
    never reaches SQL.
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

    count_sql = f"SELECT COUNT(*) FROM jobs j {JOB_CONFIG_JOIN} WHERE {where_clause}"

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

    with db.read_snapshot() as q:
        rows = q.execute_sql(select_sql, tuple(select_params)).fetchall()
        # Skip the COUNT query when we can infer the total from the result set:
        # first page + short result means we already have everything.
        if offset == 0 and limit > 0 and len(rows) < limit:
            total = len(rows)
        else:
            total = q.execute_sql(count_sql, tuple(params)).fetchone()[0]

    return JOB_ROW_PROJECTION.decode(rows), total


def _query_from_list_jobs_request(
    request: controller_pb2.Controller.ListJobsRequest,
) -> controller_pb2.Controller.JobQuery:
    """Normalize a ``ListJobsRequest`` into a single ``JobQuery``.

    If ``request.query`` is set, it is used verbatim. Otherwise the legacy
    flat fields on ``ListJobsRequest`` are mapped into an equivalent
    ``JobQuery`` so the rest of the server only ever sees one shape.

    The legacy-field path exists only to keep older clients working during
    the JobQuery rollout; delete it together with the legacy fields when
    https://github.com/marin-community/marin/issues/4573 is resolved.
    """
    if request.HasField("query"):
        query = controller_pb2.Controller.JobQuery()
        query.CopyFrom(request.query)
    else:
        query = controller_pb2.Controller.JobQuery(
            scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ALL,
            parent_job_id=request.parent_job_id,
            name_filter=request.name_filter,
            state_filter=request.state_filter,
            sort_field=request.sort_field,
            sort_direction=request.sort_direction,
            offset=request.offset,
            limit=request.limit,
        )

    # Clamp paging: 0 means "unbounded" (used by the Python client); otherwise cap at MAX.
    if query.limit > MAX_LIST_JOBS_LIMIT:
        query.limit = MAX_LIST_JOBS_LIMIT
    if query.limit < 0:
        query.limit = 0
    if query.offset < 0:
        query.offset = 0
    return query


def _parent_ids_with_children(db: ControllerDB, job_ids: list[JobName]) -> set[JobName]:
    """Return the subset of *job_ids* that currently have direct children."""
    if not job_ids:
        return set()
    placeholders = ",".join("?" for _ in job_ids)
    sql = f"""
        SELECT DISTINCT j.parent_job_id
        FROM jobs j
        WHERE j.parent_job_id IN ({placeholders})
    """
    with db.read_snapshot() as q:
        rows = q.raw(sql, tuple(job_id.to_wire() for job_id in job_ids))
    return {JobName.from_wire(row.parent_job_id) for row in rows if row.parent_job_id}


def _task_summaries_for_jobs(db: ControllerDB, job_ids: set[JobName] | None = None) -> dict[JobName, TaskJobSummary]:
    """Aggregate task counts per job using SQL GROUP BY instead of Python-side iteration."""
    if job_ids is not None:
        placeholders = ",".join("?" for _ in job_ids)
        where = f"WHERE t.job_id IN ({placeholders})"
        params: tuple[object, ...] = tuple(j.to_wire() for j in job_ids)
    else:
        where = ""
        params = ()

    sql = f"""
        SELECT t.job_id,
               t.state,
               COUNT(*) as cnt,
               SUM(t.failure_count) as total_failures,
               SUM(t.preemption_count) as total_preemptions
        FROM tasks t
        {where}
        GROUP BY t.job_id, t.state
    """
    completed_states = (job_pb2.TASK_STATE_SUCCEEDED, job_pb2.TASK_STATE_KILLED)
    with db.read_snapshot() as q:
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


def _worker_roster(db: ControllerDB) -> list[WorkerDetailRow]:
    with db.read_snapshot() as q:
        workers = WORKER_DETAIL_PROJECTION.decode(
            q.fetchall(f"SELECT {WORKER_DETAIL_PROJECTION.select_clause()} FROM workers w")
        )
        # Populate attributes from worker_attributes table.
        if workers:
            worker_ids = tuple(str(w.worker_id) for w in workers)
            placeholders = ",".join("?" for _ in worker_ids)
            attr_rows = q.fetchall(
                f"SELECT worker_id, key, value_type, str_value, int_value, float_value "
                f"FROM worker_attributes WHERE worker_id IN ({placeholders})",
                worker_ids,
            )
            attrs_by_worker: dict[str, dict[str, str | int | float]] = {}
            for row in attr_rows:
                wid = str(row["worker_id"])
                attrs = attrs_by_worker.setdefault(wid, {})
                vtype = str(row["value_type"])
                if vtype == "str":
                    attrs[str(row["key"])] = str(row["str_value"])
                elif vtype == "int":
                    attrs[str(row["key"])] = int(row["int_value"])
                elif vtype == "float":
                    attrs[str(row["key"])] = float(row["float_value"])
            workers = [dataclasses.replace(w, attributes=attrs_by_worker.get(str(w.worker_id), {})) for w in workers]
        return workers


def _query_endpoints(db: ControllerDB, query: EndpointQuery = EndpointQuery()) -> list[EndpointRow]:
    sql, params = endpoint_query_sql(query)
    with db.read_snapshot() as q:
        return ENDPOINT_PROJECTION.decode(q.fetchall(sql, tuple(params)))


def _descendant_jobs(db: ControllerDB, job_id: JobName) -> list[JobDetailRow]:
    # PK range scan: '0' (ASCII 48) is the next char after '/' (ASCII 47),
    # so this matches all job_ids starting with "<job_id>/" without LIKE.
    prefix = job_id.to_wire() + "/"
    upper = job_id.to_wire() + chr(ord("/") + 1)
    with db.read_snapshot() as q:
        return JOB_DETAIL_PROJECTION.decode(
            q.fetchall(
                f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} FROM jobs j {JOB_CONFIG_JOIN} "
                f"WHERE j.job_id >= ? AND j.job_id < ?",
                (prefix, upper),
            ),
        )


def _transaction_actions(db: ControllerDB, limit: int = 100) -> list[TransactionActionRow]:
    with db.read_snapshot() as q:
        actions = TXN_ACTION_PROJECTION.decode(
            q.fetchall(
                f"SELECT {TXN_ACTION_PROJECTION.select_clause()} " "FROM txn_actions ta2 ORDER BY ta2.id DESC LIMIT ?",
                (limit,),
            ),
        )
    return list(reversed(actions))


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


def _tasks_for_worker(db: ControllerDB, worker_id: WorkerId, limit: int = 50) -> list[TaskDetailRow]:
    with db.read_snapshot() as q:
        history_rows = q.raw(
            "SELECT wth.task_id FROM worker_task_history wth "
            "WHERE wth.worker_id = ? ORDER BY wth.assigned_at_ms DESC LIMIT ?",
            (str(worker_id), limit),
            decoders={"task_id": JobName.from_wire},
        )
    task_ids = [r.task_id for r in history_rows]
    if not task_ids:
        return []
    task_wires = [tid.to_wire() for tid in task_ids]
    placeholders = ",".join("?" for _ in task_wires)
    with db.read_snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall(
                f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} "
                f"FROM tasks t WHERE t.task_id IN ({placeholders}) ORDER BY t.task_id ASC",
                tuple(task_wires),
            ),
        )
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
                f"WHERE ta.task_id IN ({placeholders}) "
                "ORDER BY ta.task_id ASC, ta.attempt_id ASC",
                tuple(task_wires),
            ),
        )
    task_map = {t.task_id: t for t in tasks_with_attempts(tasks, attempts)}
    return [task for tid in task_ids if (task := task_map.get(tid)) is not None]


class AutoscalerProtocol(Protocol):
    """Protocol for autoscaler operations used by ControllerServiceImpl."""

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Get autoscaler status."""
        ...

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get info for a specific VM."""
        ...

    def check_coscheduling_feasibility(
        self,
        replicas: int,
        constraints: list[job_pb2.Constraint],
    ) -> str | None:
        """Check if a coscheduled job can be scheduled. Returns error message or None."""
        ...

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get initialization log for a VM."""
        ...


class ControllerProtocol(Protocol):
    """Protocol for controller operations used by ControllerServiceImpl."""

    def wake(self) -> None: ...

    def kill_tasks_on_workers(
        self,
        task_ids: set[JobName],
        task_kill_workers: dict[JobName, WorkerId] | None = None,
    ) -> None: ...

    def create_scheduling_context(self, workers: list[WorkerRow]) -> SchedulingContext: ...

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
        db: Query interface for direct DB reads
        controller: Controller runtime for scheduling and worker management
        bundle_store: Bundle store for zip storage.
        log_service: LogService that owns the log store.
    """

    def __init__(
        self,
        transitions: ControllerTransitions,
        db: ControllerDB,
        controller: ControllerProtocol,
        bundle_store: BundleStore,
        log_service: LogServiceImpl,
        auth: ControllerAuth | None = None,
        system_endpoints: dict[str, str] | None = None,
    ):
        self._transitions = transitions
        self._db = db
        self._controller = controller
        self._bundle_store = bundle_store
        self._log_service = log_service
        self._timer = Timer()
        self._auth = auth or ControllerAuth()
        self._system_endpoints: dict[str, str] = system_endpoints or {}

    def bundle_zip(self, bundle_id: str) -> bytes:
        return self._bundle_store.get_zip(bundle_id)

    def blob_data(self, blob_id: str) -> bytes:
        return self._bundle_store.get_zip(blob_id)

    def _get_autoscaler_pending_hints(self) -> dict[str, PendingHint]:
        """Build autoscaler-based pending hints keyed by job id."""
        autoscaler = self._controller.autoscaler
        if autoscaler is None:
            return {}
        status = autoscaler.get_status()
        if not status.HasField("last_routing_decision"):
            return {}
        return build_job_pending_hints(status.last_routing_decision)

    def _authorize_job_owner(self, job_id: JobName) -> None:
        """Raise PERMISSION_DENIED if the authenticated user doesn't own this job.

        Skipped when no auth provider is configured (null-auth mode).
        """
        if not self._auth.provider:
            return
        authorize_resource_owner(job_id.user)

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

        # When an auth provider is configured, override the user segment with
        # the verified identity to prevent impersonation. Only override for
        # root-level submissions; child jobs inherit the parent's user.
        verified_user = get_verified_user()
        if self._auth.provider and verified_user is not None and job_id.is_root:
            job_id = JobName.root(verified_user, job_id.name)

        # For non-root jobs, verify the caller owns the parent hierarchy
        if self._auth.provider and verified_user is not None and not job_id.is_root:
            self._authorize_job_owner(job_id)

        # Priority band validation: PRODUCTION requires MANAGE_BUDGETS permission,
        # and user's max_band from budget table must allow the requested band.
        # UNSPECIFIED (0) defaults to INTERACTIVE.
        band = request.priority_band or job_pb2.PRIORITY_BAND_INTERACTIVE
        if band == job_pb2.PRIORITY_BAND_PRODUCTION and self._auth.provider:
            authorize(AuthzAction.MANAGE_BUDGETS)
        if self._auth.provider and verified_user is not None:
            user_budget = self._db.get_user_budget(job_id.user)
            if user_budget is not None and band < user_budget.max_band:
                raise ConnectError(
                    Code.PERMISSION_DENIED,
                    f"User {job_id.user} cannot submit {priority_band_name(band)} jobs (max band: "
                    f"{priority_band_name(user_budget.max_band)})",
                )

        # Reject submissions if the parent job has already terminated
        if job_id.parent:
            parent_state = _job_state(self._db, job_id.parent)
            if parent_state is not None and parent_state in TERMINAL_JOB_STATES:
                raise ConnectError(
                    Code.FAILED_PRECONDITION,
                    f"Cannot submit job: parent job {job_id.parent} has terminated "
                    f"(state={job_pb2.JobState.Name(parent_state)})",
                )

        existing_job = _read_job(self._db, job_id)
        if existing_job:
            policy = request.existing_job_policy
            if policy == job_pb2.EXISTING_JOB_POLICY_ERROR:
                raise ConnectError(
                    Code.ALREADY_EXISTS,
                    f"Job {job_id} already exists (state={job_pb2.JobState.Name(existing_job.state)})",
                )
            elif policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
                if not job_is_finished(existing_job.state):
                    return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
                # Job finished, replace it (KEEP only preserves running jobs)
                self._transitions.remove_finished_job(job_id)
            elif policy == job_pb2.EXISTING_JOB_POLICY_RECREATE:
                if not job_is_finished(existing_job.state):
                    self._transitions.cancel_job(job_id, "Replaced by new submission")
                self._transitions.remove_finished_job(job_id)
            elif job_is_finished(existing_job.state):
                # Default/UNSPECIFIED: replace finished jobs
                logger.info(
                    "Replacing finished job %s (state=%s) with new submission",
                    job_id,
                    job_pb2.JobState.Name(existing_job.state),
                )
                self._transitions.remove_finished_job(job_id)
            else:
                raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists and is still running")

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

        # Reject coscheduled jobs that can never be scheduled: if no scaling
        # group has num_vms matching the replica count, the job would sit in
        # the queue forever.
        if request.HasField("coscheduling"):
            autoscaler = self._controller.autoscaler
            if autoscaler is not None:
                error = autoscaler.check_coscheduling_feasibility(
                    replicas=request.replicas,
                    constraints=list(request.constraints),
                )
                if error:
                    raise ConnectError(
                        Code.FAILED_PRECONDITION,
                        f"Job is unschedulable: {error}",
                    )

        self._transitions.submit_job(job_id, request, Timestamp.now())
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
        job = _read_job(self._db, JobName.from_wire(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        # Aggregate task counts via a single GROUP BY query.
        summaries = _task_summaries_for_jobs(self._db, {job.job_id})
        summary = summaries.get(job.job_id)

        task_state_counts = (
            {_task_state_key(state): count for state, count in summary.task_state_counts.items()} if summary else {}
        )

        # Get scheduling diagnostics for pending jobs from cache
        # (populated each scheduling cycle by the controller).
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
        )
        if job.started_at:
            proto_job_status.started_at.CopyFrom(timestamp_to_proto(job.started_at))
        if job.finished_at:
            proto_job_status.finished_at.CopyFrom(timestamp_to_proto(job.finished_at))
        if job.submitted_at:
            proto_job_status.submitted_at.CopyFrom(timestamp_to_proto(job.submitted_at))

        # Compute min/max resource usage across running tasks via pure SQL.
        resource_min = None
        resource_max = None
        with self._db.read_snapshot() as q:
            agg = q.raw(
                "SELECT "
                "  MIN(trh.cpu_millicores) as min_cpu, MAX(trh.cpu_millicores) as max_cpu, "
                "  MIN(trh.memory_mb) as min_mem, MAX(trh.memory_mb) as max_mem, "
                "  MIN(trh.disk_mb) as min_disk, MAX(trh.disk_mb) as max_disk, "
                "  MAX(trh.memory_peak_mb) as max_peak_mem, "
                "  COUNT(*) as cnt "
                "FROM task_resource_history trh "
                "JOIN tasks t ON trh.task_id = t.task_id AND trh.attempt_id = t.current_attempt_id "
                "WHERE t.job_id = ? AND t.state IN (?, ?)",
                (job.job_id.to_wire(), job_pb2.TASK_STATE_RUNNING, job_pb2.TASK_STATE_BUILDING),
            )
        if agg and agg[0].cnt > 0:
            row = agg[0]
            resource_min = job_pb2.ResourceUsage(
                memory_mb=row.min_mem,
                cpu_millicores=row.min_cpu,
                disk_mb=row.min_disk,
            )
            resource_max = job_pb2.ResourceUsage(
                memory_mb=row.max_mem,
                cpu_millicores=row.max_cpu,
                disk_mb=row.max_disk,
                memory_peak_mb=row.max_peak_mem,
            )

        reconstructed_request = _reconstruct_launch_job_request(job)
        return controller_pb2.Controller.GetJobStatusResponse(
            job=proto_job_status,
            request=redact_request_env_vars(reconstructed_request),
            resource_min=resource_min,
            resource_max=resource_max,
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
        result = self._transitions.cancel_job(job_id, reason="Terminated by user")
        if result.tasks_to_kill:
            self._controller.kill_tasks_on_workers(result.tasks_to_kill, result.task_kill_workers)
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
            {_task_state_key(state): count for state, count in task_summary.task_state_counts.items()}
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

        resources = _resource_spec_from_job_row(j)

        proto_job = job_pb2.JobStatus(
            job_id=j.job_id.to_wire(),
            state=j.state,
            error=j.error or "",
            exit_code=j.exit_code or 0,
            failure_count=task_summary.failure_count if task_summary else 0,
            preemption_count=task_summary.preemption_count if task_summary else 0,
            name=job_name,
            resources=resources,
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
        """List jobs with SQL-level filtering, sorting, and pagination."""
        query = _query_from_list_jobs_request(request)

        state_ids = _resolve_state_filter(query.state_filter)
        if state_ids is None:
            return controller_pb2.Controller.ListJobsResponse(jobs=[], total_count=0, has_more=False)

        jobs, total_count = _query_jobs(self._db, query, state_ids)

        task_summaries = _task_summaries_for_jobs(self._db, {j.job_id for j in jobs})
        has_pending = any(j.state == job_pb2.JOB_STATE_PENDING for j in jobs)
        autoscaler_pending_hints = self._get_autoscaler_pending_hints() if has_pending else {}
        has_children = _parent_ids_with_children(self._db, [j.job_id for j in jobs])
        all_jobs = self._jobs_to_protos(jobs, task_summaries, autoscaler_pending_hints, has_children=has_children)
        has_more = query.limit > 0 and query.offset + query.limit < total_count
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

        # Attach resource history and job resource limits (detail view only).
        job_resources = None
        with self._db.read_snapshot() as q:
            # Fetch newest rows first so active tasks with >N rows get current data.
            history_rows = q.raw(
                "SELECT trh.cpu_millicores, trh.memory_mb, trh.disk_mb, trh.memory_peak_mb "
                "FROM task_resource_history trh "
                "WHERE trh.task_id = ? AND trh.attempt_id = ? ORDER BY trh.id DESC LIMIT ?",
                (task_id.to_wire(), task.current_attempt_id, TASK_RESOURCE_HISTORY_RETENTION),
            )
            job_row = q.raw(
                "SELECT j.request_proto FROM jobs j WHERE j.job_id = ?",
                (task.job_id.to_wire(),),
                decoders={"request_proto": lambda b: controller_pb2.Controller.LaunchJobRequest.FromString(b)},
            )
        # Reverse to oldest-first for the API contract.
        for r in reversed(history_rows):
            proto.resource_history.append(
                job_pb2.ResourceUsage(
                    cpu_millicores=r.cpu_millicores,
                    memory_mb=r.memory_mb,
                    disk_mb=r.disk_mb,
                    memory_peak_mb=r.memory_peak_mb,
                )
            )
        if job_row and job_row[0].request_proto.HasField("resources"):
            job_resources = job_row[0].request_proto.resources

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

        self._transitions.register_or_refresh_worker(
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
        """List all workers with their running task counts."""
        if self._controller.has_direct_provider:
            return controller_pb2.Controller.ListWorkersResponse()
        workers = []
        worker_rows = _worker_roster(self._db)
        running_by_worker = running_tasks_by_worker(self._db, {worker.worker_id for worker in worker_rows})
        for worker in worker_rows:
            workers.append(
                controller_pb2.Controller.WorkerHealthStatus(
                    worker_id=worker.worker_id,
                    healthy=worker.healthy,
                    consecutive_failures=worker.consecutive_failures,
                    last_heartbeat=timestamp_to_proto(worker.last_heartbeat),
                    running_job_ids=[task_id.to_wire() for task_id in running_by_worker.get(worker.worker_id, [])],
                    address=worker.address,
                    metadata=_worker_metadata_to_proto(worker),
                    status_message=worker_status_message(worker),
                )
            )
        return controller_pb2.Controller.ListWorkersResponse(workers=workers)

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
        job_id, _task_index = task_id.require_task()

        if _job_state(self._db, job_id) is None:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.task_id} not found")

        task = _read_task_with_attempts(self._db, task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")
        if request.attempt_id != task.current_attempt_id:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Stale attempt: task {request.task_id} attempt {request.attempt_id} "
                f"!= current {task.current_attempt_id}",
            )

        endpoint = EndpointRow(
            endpoint_id=endpoint_id,
            name=request.name,
            address=request.address,
            job_id=job_id,
            metadata=dict(request.metadata),
            registered_at=Timestamp.now(),
        )

        if not self._transitions.add_endpoint(endpoint, task_id=task_id):
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
        self._transitions.remove_endpoint(request.endpoint_id)
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

        endpoints = _query_endpoints(
            self._db,
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
                    task_id=e.job_id.to_wire(),
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

        # Build a map of worker_id -> (worker_id, healthy) for enriching VmInfo
        workers = _worker_roster(self._db)
        worker_id_to_info: dict[str, tuple[str, bool]] = {}
        for w in workers:
            worker_id_to_info[w.worker_id] = (w.worker_id, w.healthy)

        # Fetch running task counts per worker for dashboard display
        all_worker_ids = {WorkerId(w.worker_id) for w in workers}
        running_by_worker = running_tasks_by_worker(self._db, all_worker_ids) if all_worker_ids else {}

        # Enrich VmInfo objects with worker information by matching vm_id to worker_id
        for group in status.groups:
            for slice_info in group.slices:
                for vm in slice_info.vms:
                    worker_info = worker_id_to_info.get(vm.vm_id)
                    if worker_info:
                        vm.worker_id = worker_info[0]
                        vm.worker_healthy = worker_info[1]
                        wid = WorkerId(vm.worker_id)
                        vm.running_task_count = len(running_by_worker.get(wid, set()))

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

    # --- Task/Job Logs (batch fetching) ---

    def get_task_logs(
        self,
        request: controller_pb2.Controller.GetTaskLogsRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetTaskLogsResponse:
        """DEPRECATED: use FetchLogs with regex patterns instead. Scheduled for removal 2026-05-01.

        Forwards to fetch_logs internally, wrapping the response in the legacy format.
        """
        job_name = JobName.from_wire(request.id)

        # Build the regex source pattern from the legacy request fields
        if job_name.is_task:
            source = build_log_source(job_name, request.attempt_id)
        elif request.include_children:
            source = build_log_source(job_name)
        else:
            # Direct tasks only: match keys like /user/job/0:attempt but not
            # /user/job/child-job/0:attempt. Use \d+ to restrict to numeric
            # task indices, pushing the filter into DuckDB.
            escaped_wire = re.escape(job_name.to_wire())
            source = f"{escaped_wire}/\\d+:.*"

        max_lines = request.max_total_lines if request.max_total_lines > 0 else DEFAULT_MAX_TOTAL_LINES

        fetch_request = logging_pb2.FetchLogsRequest(
            source=source,
            since_ms=request.since_ms,
            cursor=request.cursor,
            substring=request.substring,
            max_lines=max_lines,
            tail=request.tail,
            min_level=request.min_level,
        )

        fetch_response = self._log_service.fetch_logs(fetch_request, ctx)
        entries = fetch_response.entries

        batch = controller_pb2.Controller.TaskLogBatch(
            task_id=request.id,
            logs=entries,
        )

        truncated = max_lines > 0 and len(fetch_response.entries) >= max_lines

        return controller_pb2.Controller.GetTaskLogsResponse(
            task_logs=[batch],
            truncated=truncated,
            cursor=fetch_response.cursor,
        )

    # --- Profiling ---

    def profile_task(
        self,
        request: job_pb2.ProfileTaskRequest,
        ctx: RequestContext,
    ) -> job_pb2.ProfileTaskResponse:
        """Profile a running task or system process.

        Target routing:
        - /system/process: the controller process itself
        - /system/worker/<worker_id>: proxy to a specific worker (profiles the worker process)
        - /job/.../task/N: proxied to the task's worker
        """
        # Handle controller-local targets: profile the controller process itself
        if is_system_target(request.target):
            if not request.HasField("profile_type"):
                raise ConnectError(Code.INVALID_ARGUMENT, "profile_type is required")
            try:
                duration = request.duration_seconds or 10
                data = profile_local_process(duration, request.profile_type)
                return job_pb2.ProfileTaskResponse(profile_data=data)
            except Exception as e:
                return job_pb2.ProfileTaskResponse(error=str(e))

        # /system/worker/<worker_id>: proxy profile to the worker's own process
        worker_id = _parse_worker_target(request.target)
        if worker_id is not None:
            worker = self._transitions.get_worker(WorkerId(worker_id))
            if not worker:
                raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
            if not worker.healthy:
                raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")
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

        worker = _read_worker(self._db, task_worker_id)
        if not worker or not worker.healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {task_worker_id} is unavailable")

        timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
        resp = self._controller.provider.profile_task(worker.address, request, timeout_ms)
        return job_pb2.ProfileTaskResponse(
            profile_data=resp.profile_data,
            error=resp.error,
        )

    # --- Transactions ---

    def get_transactions(
        self,
        request: controller_pb2.Controller.GetTransactionsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetTransactionsResponse:
        """Get recent controller actions for the dashboard action log."""
        limit = request.limit if request.limit > 0 else DEFAULT_TRANSACTION_LIMIT
        actions = []
        for action in _transaction_actions(self._db, limit=limit):
            details_str = json.dumps(action.details) if action.details else ""
            proto_action = controller_pb2.Controller.TransactionAction(
                action=action.action,
                entity_id=action.entity_id,
                details=details_str,
            )
            proto_action.timestamp.CopyFrom(timestamp_to_proto(action.timestamp))
            actions.append(proto_action)
        return controller_pb2.Controller.GetTransactionsResponse(actions=actions)

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

        detail = _read_worker_detail(self._db, WorkerId(str(request.id)))
        if not detail:
            raise ConnectError(Code.NOT_FOUND, f"No worker found for '{request.id}'")

        worker = detail.worker
        worker_health = controller_pb2.Controller.WorkerHealthStatus(
            worker_id=worker.worker_id,
            healthy=worker.healthy,
            consecutive_failures=worker.consecutive_failures,
            last_heartbeat=timestamp_to_proto(worker.last_heartbeat),
            running_job_ids=[tid.to_wire() for tid in detail.running_tasks],
            address=worker.address,
            metadata=_worker_metadata_to_proto(worker),
            status_message=worker_status_message(worker),
        )

        # Fetch worker daemon logs via LogService
        worker_log_entries: list[logging_pb2.LogEntry] = []
        try:
            fetch_resp = self._log_service.fetch_logs(
                logging_pb2.FetchLogsRequest(
                    source=worker_log_key(worker.worker_id),
                    max_lines=200,
                    tail=True,
                ),
                ctx,
            )
            worker_log_entries = list(fetch_resp.entries)
        except Exception:
            logger.debug("Failed to fetch worker logs for %s", request.id, exc_info=True)

        # Collect recent task history for this worker
        tasks = _tasks_for_worker(self._db, worker.worker_id, limit=50)
        recent_tasks = [task_to_proto(task) for task in tasks]

        resp = controller_pb2.Controller.GetWorkerStatusResponse(
            worker_log_entries=worker_log_entries,
            recent_tasks=recent_tasks,
        )
        resp.worker.CopyFrom(worker_health)
        resource_history = detail.resource_history
        if resource_history:
            resp.current_resources.CopyFrom(resource_history[-1])
        for snapshot in resource_history:
            resp.resource_history.append(snapshot)
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

        worker = self._transitions.get_worker(WorkerId(worker_id))
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
        if not worker.healthy:
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

        worker = _read_worker(self._db, task_worker_id)
        if not worker or not worker.healthy:
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
        """Return scheduler state for the dashboard: pending queue, budgets, running tasks."""
        require_identity()

        # --- User budgets and spend ---
        budgets = self._db.list_user_budgets()
        budget_limits: dict[str, int] = {b.user_id: b.budget_limit for b in budgets}
        with self._db.read_snapshot() as snap:
            user_spend = compute_user_spend(snap)

        # --- Pending queue ---
        # Inline _schedulable_tasks logic to avoid circular import with controller.py
        with self._db.read_snapshot() as snap:
            task_rows = TASK_ROW_PROJECTION.decode(
                snap.fetchall(
                    f"SELECT {TASK_ROW_PROJECTION.select_clause()} FROM tasks t WHERE t.state = ? "
                    "ORDER BY t.priority_band ASC, t.priority_neg_depth ASC, "
                    "t.priority_root_submitted_ms ASC, t.submitted_at_ms ASC, t.priority_insertion ASC",
                    (job_pb2.TASK_STATE_PENDING,),
                ),
            )
        pending_tasks = [t for t in task_rows if task_row_can_be_scheduled(t)]
        # Batch-fetch resource spec columns for resource values
        job_ids = {t.job_id for t in pending_tasks}
        job_resources: dict[JobName, job_pb2.ResourceSpecProto] = {}
        if job_ids:
            wires = [jid.to_wire() for jid in job_ids]
            placeholders = ",".join("?" for _ in wires)
            with self._db.read_snapshot() as snap:
                rows = snap.raw(
                    f"SELECT jc.job_id, jc.res_cpu_millicores, jc.res_memory_bytes, "
                    f"jc.res_disk_bytes, jc.res_device_json "
                    f"FROM job_config jc WHERE jc.job_id IN ({placeholders})",
                    tuple(wires),
                    decoders={"job_id": JobName.from_wire},
                )
            for row in rows:
                job_resources[row.job_id] = _resource_spec_from_job_row(row)

        # Group by effective band, interleaving by user within each band
        BAND_ORDER = [
            job_pb2.PRIORITY_BAND_PRODUCTION,
            job_pb2.PRIORITY_BAND_INTERACTIVE,
            job_pb2.PRIORITY_BAND_BATCH,
        ]
        MAX_TASKS_PER_BAND = 100

        band_groups: list[controller_pb2.Controller.SchedulerBandGroup] = []
        total_pending = len(pending_tasks)
        # Partition tasks by effective band
        tasks_by_band: dict[int, list[UserTask]] = {b: [] for b in BAND_ORDER}
        for task in pending_tasks:
            eff_band = compute_effective_band(task.priority_band, task.task_id.user, user_spend, budget_limits)
            ut: UserTask = UserTask(user_id=task.task_id.user, task=(task, eff_band))
            target_band = eff_band if eff_band in tasks_by_band else job_pb2.PRIORITY_BAND_BATCH
            tasks_by_band[target_band].append(ut)

        global_position = 0
        for band in BAND_ORDER:
            band_tasks = tasks_by_band.get(band, [])
            if not band_tasks:
                continue
            interleaved = interleave_by_user(band_tasks, user_spend)
            total_in_band = len(interleaved)
            entries: list[controller_pb2.Controller.SchedulerTaskEntry] = []
            for task_and_band in interleaved[:MAX_TASKS_PER_BAND]:
                task, eff_band = task_and_band
                res = job_resources.get(task.job_id)
                rv = 0
                if res is not None:
                    accel = get_gpu_count(res.device) + get_tpu_count(res.device)
                    rv = resource_value(res.cpu_millicores, res.memory_bytes, accel)
                entries.append(
                    controller_pb2.Controller.SchedulerTaskEntry(
                        task_id=task.task_id.to_wire(),
                        job_id=task.job_id.to_wire(),
                        user_id=task.task_id.user,
                        original_band=task.priority_band,
                        effective_band=eff_band,
                        queue_position=global_position,
                        resource_value=rv,
                    )
                )
                global_position += 1
            # Advance position for truncated tasks
            global_position += max(0, total_in_band - MAX_TASKS_PER_BAND)
            band_groups.append(
                controller_pb2.Controller.SchedulerBandGroup(
                    band=band,
                    tasks=entries,
                    total_in_band=total_in_band,
                )
            )

        # --- User budgets for response ---
        budget_protos: list[controller_pb2.Controller.SchedulerUserBudget] = []
        for b in budgets:
            spent = user_spend.get(b.user_id, 0)
            utilization = (spent / b.budget_limit * 100.0) if b.budget_limit > 0 else 0.0
            # Show effective band: use INTERACTIVE as the test band to see if user is downgraded
            eff = compute_effective_band(job_pb2.PRIORITY_BAND_INTERACTIVE, b.user_id, user_spend, budget_limits)
            budget_protos.append(
                controller_pb2.Controller.SchedulerUserBudget(
                    user_id=b.user_id,
                    budget_limit=b.budget_limit,
                    budget_spent=spent,
                    max_band=b.max_band,
                    effective_band=eff,
                    utilization_percent=utilization,
                )
            )

        # --- Running tasks ---
        # Inline _get_running_tasks_with_band_and_value logic to avoid circular import
        with self._db.read_snapshot() as snap:
            running_rows = snap.raw(
                "SELECT t.task_id, t.priority_band, t.current_worker_id AS worker_id, "
                "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, "
                "jc.res_device_json, jc.has_coscheduling "
                "FROM tasks t "
                "JOIN job_config jc ON jc.job_id = t.job_id "
                "WHERE t.state = ? AND t.current_worker_id IS NOT NULL",
                (job_pb2.TASK_STATE_RUNNING,),
                decoders={
                    "task_id": JobName.from_wire,
                    "priority_band": int,
                    "worker_id": WorkerId,
                },
            )
        running_protos: list[controller_pb2.Controller.SchedulerRunningTask] = []
        for row in running_rows:
            res = _resource_spec_from_job_row(row)
            eff_band = compute_effective_band(row.priority_band, row.task_id.user, user_spend, budget_limits)
            accel = get_gpu_count(res.device) + get_tpu_count(res.device)
            rv = resource_value(res.cpu_millicores, res.memory_bytes, accel)
            is_cosched = bool(row.has_coscheduling)
            preemptible_by: list[int] = []
            preemptible = False
            if not is_cosched:
                if eff_band == job_pb2.PRIORITY_BAND_BATCH:
                    preemptible = True
                    preemptible_by = [
                        job_pb2.PRIORITY_BAND_PRODUCTION,
                        job_pb2.PRIORITY_BAND_INTERACTIVE,
                    ]
                elif eff_band == job_pb2.PRIORITY_BAND_INTERACTIVE:
                    preemptible = True
                    preemptible_by = [job_pb2.PRIORITY_BAND_PRODUCTION]
            running_protos.append(
                controller_pb2.Controller.SchedulerRunningTask(
                    task_id=row.task_id.to_wire(),
                    job_id=(row.task_id.parent or row.task_id).to_wire(),
                    user_id=row.task_id.user,
                    worker_id=str(row.worker_id),
                    effective_band=eff_band,
                    resource_value=rv,
                    preemptible=preemptible,
                    preemptible_by=preemptible_by,
                    is_coscheduled=is_cosched,
                )
            )

        return controller_pb2.Controller.GetSchedulerStateResponse(
            pending_queue=band_groups,
            user_budgets=budget_protos,
            running_tasks=running_protos,
            total_pending=total_pending,
            total_running=len(running_protos),
        )
