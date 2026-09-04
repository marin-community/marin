# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations and projections for tasks and their attempts."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from finelog.client import LogClient
from finelog.rpc import logging_pb2
from rigging.timing import Timestamp
from sqlalchemy import func, select, tuple_

from iris.cluster.controller import reads
from iris.cluster.controller.auth import ControllerAuth, authorize_owner_if_configured
from iris.cluster.controller.backend import TaskBackend
from iris.cluster.controller.codec import proto_from_json, resource_spec_from_scalars
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.reconcile.task import TerminalKind
from iris.cluster.controller.schema import job_config_table, task_attempts_table, tasks_table, workers_table
from iris.cluster.controller.task_state import (
    ACTIVE_TASK_STATES,
    AttemptDetailRow,
    TaskDetailRow,
    attempt_is_worker_failure,
    task_row_can_be_scheduled,
)
from iris.cluster.log_highlights import extract_failure_highlights
from iris.cluster.log_keys import build_log_source
from iris.cluster.types import TERMINAL_TASK_STATES, JobName, TaskAttempt, WorkerId, is_federated
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.proto_display import task_state_friendly
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)

_LISTING_FAILURE_STATES = (job_pb2.TASK_STATE_FAILED, job_pb2.TASK_STATE_WORKER_FAILED)
_ROOT_CAUSE_LOG_TAIL = 200
_KICK_KIND_BY_STATE = {
    job_pb2.TASK_STATE_PREEMPTED: TerminalKind.PREEMPT,
    job_pb2.TASK_STATE_FAILED: TerminalKind.TIMEOUT,
}


@dataclass(frozen=True, slots=True)
class PendingKick:
    """Administrative terminal transition queued for the next control tick."""

    task_id: JobName
    attempt_id: int | None
    kind: TerminalKind
    reason: str


class TaskRuntime(Protocol):
    @property
    def backend(self) -> TaskBackend: ...

    def request_task_kicks(self, kicks: Sequence[PendingKick]) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskDependencies:
    db: ControllerDB
    logs: LogClient
    runtime: TaskRuntime
    auth: ControllerAuth


@dataclass(frozen=True, slots=True)
class TaskWithAttempts:
    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
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
    status_message: str | None
    backend_id: str
    cluster: str
    peer_worker_label: str
    attempts: tuple[AttemptDetailRow, ...]

    @classmethod
    def from_row(cls, row: TaskDetailRow, attempts: tuple[AttemptDetailRow, ...]) -> "TaskWithAttempts":
        return cls(
            task_id=row.task_id,
            job_id=row.job_id,
            state=row.state,
            current_attempt_id=row.current_attempt_id,
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
            status_message=row.status_message,
            backend_id=str(row.backend_id or ""),
            cluster=str(row.cluster),
            peer_worker_label=str(row.peer_worker_label or ""),
            attempts=attempts,
        )


def get_task_status(
    dependencies: TaskDependencies,
    request: controller_pb2.Controller.GetTaskStatusRequest,
    context: RequestContext,
) -> controller_pb2.Controller.GetTaskStatusResponse:
    """Return one task, its Attempt history, and static resource limits."""
    del context
    try:
        task_id = JobName.from_wire(request.task_id)
        task_id.require_task()
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error
    task = read_task_with_attempts(dependencies.db, task_id)
    if task is None:
        raise ConnectError(Code.NOT_FOUND, f"Task {task_id} not found")

    worker_id = task_worker_id(task)
    task_proto = task_to_proto(task, worker_address=worker_address(dependencies.db, worker_id) if worker_id else "")
    job_resources = None
    with dependencies.db.read_snapshot() as tx:
        job_config = tx.execute(
            select(
                job_config_table.c.res_cpu_millicores,
                job_config_table.c.res_memory_bytes,
                job_config_table.c.res_disk_bytes,
                job_config_table.c.res_device_json,
                job_config_table.c.task_image,
            ).where(job_config_table.c.job_id == task.job_id)
        ).first()
    if job_config is not None:
        if (
            job_config.res_cpu_millicores
            or job_config.res_memory_bytes
            or job_config.res_disk_bytes
            or job_config.res_device_json
        ):
            job_resources = resource_spec_from_scalars(
                job_config.res_cpu_millicores,
                job_config.res_memory_bytes,
                job_config.res_disk_bytes,
                job_config.res_device_json,
            )
        runtime_image = dependencies.runtime.backend.runtime_image(job_config.task_image)
        if runtime_image:
            task_proto.build_metrics.image_tag = runtime_image

    return controller_pb2.Controller.GetTaskStatusResponse(
        task=task_proto,
        job_resources=job_resources,
        root_cause_highlights=_task_root_cause_highlights(dependencies.logs, task_id, task_proto.state),
    )


def list_tasks(
    dependencies: TaskDependencies,
    request: controller_pb2.Controller.ListTasksRequest,
    context: RequestContext,
) -> controller_pb2.Controller.ListTasksResponse:
    """Return all tasks belonging to one job."""
    del context
    if not request.job_id:
        raise ConnectError(Code.INVALID_ARGUMENT, "job_id is required")
    job_id = JobName.from_wire(request.job_id)
    with dependencies.db.read_snapshot() as tx:
        task_rows = tasks_for_listing(tx, job_id=job_id)
    task_statuses = []
    for task in task_rows:
        status = task_to_proto(task)
        if task.state == job_pb2.TASK_STATE_PENDING:
            status.can_be_scheduled = task_row_can_be_scheduled(task)
        task_statuses.append(status)
    return controller_pb2.Controller.ListTasksResponse(tasks=task_statuses)


def kick_tasks(
    dependencies: TaskDependencies,
    request: controller_pb2.Controller.KickTasksRequest,
    context: RequestContext,
) -> controller_pb2.Controller.KickTasksResponse:
    """Queue administrative terminal transitions for active tasks."""
    del context
    kind = _KICK_KIND_BY_STATE.get(request.desired_state)
    if kind is None:
        allowed = ", ".join(task_state_friendly(state) for state in _KICK_KIND_BY_STATE)
        raise ConnectError(Code.INVALID_ARGUMENT, f"desired_state must be one of: {allowed}")
    if not request.targets:
        raise ConnectError(Code.INVALID_ARGUMENT, "at least one target is required")

    reason = request.reason or f"Kicked to {task_state_friendly(request.desired_state)} by operator"
    results: list[controller_pb2.Controller.KickResult] = []
    kicks: list[PendingKick] = []
    with dependencies.db.read_snapshot() as tx:
        for target in request.targets:
            _resolve_kick_target(dependencies, tx, target, kind, reason, kicks, results)
    dependencies.runtime.request_task_kicks(kicks)
    return controller_pb2.Controller.KickTasksResponse(results=results)


def read_task_with_attempts(db: ControllerDB, task_id: JobName) -> TaskWithAttempts | None:
    with db.read_snapshot() as tx:
        task_row = reads.get_task_detail(tx, task_id)
        if task_row is None:
            return None
        attempt_rows = tx.execute(
            reads.attempt_select()
            .where(task_attempts_table.c.task_id == task_id)
            .order_by(task_attempts_table.c.attempt_id.asc())
        ).all()
    return TaskWithAttempts.from_row(task_row, tuple(AttemptDetailRow.from_row(row) for row in attempt_rows))


def tasks_for_listing(tx: Tx, *, job_id: JobName) -> list[TaskWithAttempts]:
    job_task_ids = select(tasks_table.c.task_id).where(tasks_table.c.job_id == job_id)
    task_rows = tx.execute(
        reads.task_detail_query()
        .where(tasks_table.c.job_id == job_id)
        .order_by(tasks_table.c.job_id.asc(), tasks_table.c.task_index.asc())
    ).all()
    current_attempt_rows = tx.execute(
        reads.attempt_select().where(
            tuple_(task_attempts_table.c.task_id, task_attempts_table.c.attempt_id).in_(
                select(tasks_table.c.task_id, tasks_table.c.current_attempt_id).where(
                    tasks_table.c.job_id == job_id,
                    tasks_table.c.current_attempt_id >= 0,
                )
            )
        )
    ).all()
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
        reads.attempt_select(
            reads.ATTEMPTS_WITH_OUTPUT.join(
                latest_failed,
                (task_attempts_table.c.task_id == latest_failed.c.task_id)
                & (task_attempts_table.c.attempt_id == latest_failed.c.attempt_id),
            )
        )
    ).all()
    attempts_by_task: dict[JobName, dict[int, AttemptDetailRow]] = {}
    for row in (*current_attempt_rows, *failed_attempt_rows):
        attempt = AttemptDetailRow.from_row(row)
        attempts_by_task.setdefault(attempt.task_id, {})[attempt.attempt_id] = attempt
    return [
        TaskWithAttempts.from_row(
            row,
            tuple(attempt for _, attempt in sorted(attempts_by_task.get(row.task_id, {}).items())),
        )
        for row in task_rows
    ]


def attempt_to_proto(attempt: AttemptDetailRow) -> job_pb2.TaskAttempt:
    result = job_pb2.TaskAttempt(
        attempt_id=attempt.attempt_id,
        worker_id=str(attempt.worker_id) if attempt.worker_id else "",
        state=attempt.state,
        exit_code=attempt.exit_code or 0,
        error=attempt.error or "",
        is_worker_failure=attempt_is_worker_failure(attempt.state),
        attempt_uid=attempt.attempt_uid,
        pod_name=attempt.pod_name or "",
        pod_uid=attempt.pod_uid or "",
        node_name=attempt.node_name or "",
        terminal_reason=attempt.terminal_reason or "",
    )
    if attempt.output_archive_json:
        result.output_archive.CopyFrom(proto_from_json(attempt.output_archive_json, job_pb2.TaskOutputArchive))
    if attempt.started_at_ms is not None:
        result.started_at.CopyFrom(timestamp_to_proto(attempt.started_at_ms))
    if attempt.finished_at_ms is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(attempt.finished_at_ms))
    return result


def task_to_proto(task: TaskWithAttempts, worker_address: str = "") -> job_pb2.TaskStatus:
    current_attempt = task.attempts[-1] if task.attempts else None
    attempts = [attempt_to_proto(attempt) for attempt in task.attempts]

    active_worker_id = None if task.state == job_pb2.TASK_STATE_PENDING else task_worker_id(task)
    display_worker_id = (
        str(active_worker_id) if active_worker_id else (task.peer_worker_label if is_federated(task.cluster) else "")
    )
    result = job_pb2.TaskStatus(
        task_id=task.task_id.to_wire(),
        state=task.state,
        worker_id=display_worker_id,
        worker_address=worker_address or task.current_worker_address or "",
        exit_code=task.exit_code or 0,
        error=task.error or "",
        current_attempt_id=task.current_attempt_id,
        attempts=attempts,
        backend_id=task.backend_id,
        cluster=task.cluster,
    )
    if task.submitted_at_ms.epoch_ms() > 0:
        result.submitted_at.CopyFrom(timestamp_to_proto(task.submitted_at_ms))
    if current_attempt and current_attempt.started_at_ms:
        result.started_at.CopyFrom(timestamp_to_proto(current_attempt.started_at_ms))
    if current_attempt and current_attempt.finished_at_ms:
        result.finished_at.CopyFrom(timestamp_to_proto(current_attempt.finished_at_ms))
    if task.container_id:
        result.container_id = task.container_id
    if task.status_message:
        result.status_message = task.status_message
    if task.state == job_pb2.TASK_STATE_PENDING and task.attempts and task.attempts[-1].state in TERMINAL_TASK_STATES:
        last_attempt = task.attempts[-1]
        result.pending_reason = (
            f"Retrying (attempt {task.current_attempt_id + 1}, "
            f"last: {job_pb2.TaskState.Name(last_attempt.state).lower()})"
        )
        result.can_be_scheduled = True
    return result


def task_worker_id(task: TaskWithAttempts) -> WorkerId | None:
    if task.attempts:
        return task.attempts[-1].worker_id
    return task.current_worker_id


def worker_address(db: ControllerDB, worker_id: WorkerId) -> str:
    with db.read_snapshot() as tx:
        row = tx.execute(select(workers_table.c.address).where(workers_table.c.worker_id == worker_id)).first()
    return str(row.address) if row else ""


def _task_root_cause_highlights(logs: LogClient, task_id: JobName, state: int) -> list[str]:
    if state not in _LISTING_FAILURE_STATES:
        return []
    source, match_scope = build_log_source(task_id)
    try:
        response = logs.fetch_logs(
            logging_pb2.FetchLogsRequest(
                source=source,
                match_scope=match_scope,
                max_lines=_ROOT_CAUSE_LOG_TAIL,
                tail=True,
            )
        )
    except Exception:
        logger.warning("Failed to fetch logs for root-cause highlights of %s", task_id, exc_info=True)
        return []
    return extract_failure_highlights([entry.data for entry in response.entries])


def _resolve_kick_target(
    dependencies: TaskDependencies,
    tx: Tx,
    target: str,
    kind: TerminalKind,
    reason: str,
    kicks: list[PendingKick],
    results: list[controller_pb2.Controller.KickResult],
) -> None:
    def reject(detail: str, *, task_id: str = "") -> None:
        results.append(
            controller_pb2.Controller.KickResult(
                target=target,
                task_id=task_id,
                queued=False,
                detail=detail,
            )
        )

    try:
        task_attempt = TaskAttempt.from_wire(target)
    except ValueError as error:
        reject(str(error))
        return

    name = task_attempt.task_id
    authorize_owner_if_configured(dependencies.auth, name.user)
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
            reject(
                f"task is {task_state_friendly(detail.state)}, not running on a worker",
                task_id=name.to_wire(),
            )
            return
        kicks.append(PendingKick(task_id=name, attempt_id=task_attempt.attempt_id, kind=kind, reason=reason))
        results.append(controller_pb2.Controller.KickResult(target=target, task_id=name.to_wire(), queued=True))
        return

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
        results.append(controller_pb2.Controller.KickResult(target=target, task_id=row.task_id.to_wire(), queued=True))
