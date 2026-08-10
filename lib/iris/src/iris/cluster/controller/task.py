# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from collections.abc import Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.backends.protocol import BackendCapability
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.pagination import (
    _decode_page_token,
    _encode_page_token,
    _escaped_prefix,
    _page_size,
    _query_fingerprint,
    _require_kind,
    _stored_cluster,
)
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.persistence.database import Tx
from iris.cluster.controller.resource_identity import (
    _execution_cluster,
    _job_uid,
    _task_uid,
)
from iris.cluster.controller.source_status import (
    _available_source,
    _unavailable_finelog_source,
    resource_source_statuses,
)
from iris.cluster.controller.task_state import TaskDetailRow
from iris.cluster.federation.protocol import FederationDirection
from iris.cluster.log_highlights import extract_failure_highlights
from iris.cluster.log_keys import build_log_source
from iris.resources.attempt import AttemptCounts, AttemptSummary
from iris.resources.errors import (
    BackendIdentityUnknown,
    ResourceNotFound,
    ResourceReplaced,
)
from iris.resources.identity import (
    AttemptIdentity,
    JobIdentity,
    NodeIdentity,
    ResourceKey,
    ResourceKind,
    TaskIdentity,
)
from iris.resources.log import LogQuery, LogReadError
from iris.resources.names import JobName
from iris.resources.source import (
    Page,
    ResourceSourceStatus,
)
from iris.resources.state import TaskState
from iris.resources.task import TaskDetail, TaskQuery, TaskSummary

_MAX_TASK_PAGE = 500
_FINELOG_NOT_CONFIGURED = "finelog is not configured"


@dataclass(frozen=True, slots=True)
class _FailureHighlights:
    entries: tuple[str, ...]
    source_status: ResourceSourceStatus | None


class TaskResources:
    """Task resource operations."""

    def __init__(self, dependencies: ResourceDependencies) -> None:
        self._dependencies = dependencies

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]:
        page_size = _page_size(query.page_size, _MAX_TASK_PAGE)
        fingerprint = _query_fingerprint(
            "tasks",
            {
                "job": query.job.resource_id if query.job else None,
                "job_id_prefix": query.job_id_prefix,
                "states": sorted(int(state) for state in query.states),
                "backend_id": query.backend_id,
                "authority_cluster_id": query.authority_cluster_id,
                "execution_cluster_id": query.execution_cluster_id,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        job_id = None
        if query.job is not None:
            _require_kind(query.job, ResourceKind.JOB)
            job_id = JobName.from_wire(query.job.resource_id)
        position_submitted_at = None
        position_task_id = None
        if position is not None:
            position_submitted_at = Timestamp.from_ms(int(position["submitted_at_ms"]))
            position_task_id = JobName.from_wire(str(position["task_id"]))
        with self._dependencies.db.read_snapshot() as tx:
            rows = reads.list_resource_tasks(
                tx,
                job_id=job_id,
                job_id_prefix=_escaped_prefix(query.job_id_prefix) if query.job_id_prefix else None,
                states=[int(state) for state in query.states],
                backend_id=query.backend_id,
                execution_cluster=(
                    _stored_cluster(self._dependencies.cluster_id, query.execution_cluster_id)
                    if query.execution_cluster_id is not None
                    else None
                ),
                position_submitted_at=position_submitted_at,
                position_task_id=position_task_id,
                limit=page_size + 1,
            )
            page_rows = rows[:page_size]
            attempt_keys = [
                (row.task_id, int(row.current_attempt_id)) for row in page_rows if int(row.current_attempt_id) >= 0
            ]
            current_attempts = reads.bulk_get_attempts(tx, attempt_keys)
            jobs = self._job_rows(tx, {row.job_id for row in page_rows})
        items = tuple(
            self._task_summary(
                row,
                current_attempts.get((row.task_id, int(row.current_attempt_id))),
                AttemptCounts(row.failure_count, row.preemption_count),
                jobs[row.job_id],
            )
            for row in page_rows
            if query.authority_cluster_id is None
            or self._authority_cluster(jobs[row.job_id]) == query.authority_cluster_id
        )
        if query.job is not None:
            items = tuple(item for item in items if item.job.key == query.job)
        next_token = None
        if len(rows) > page_size and page_rows:
            last = page_rows[-1]
            next_token = _encode_page_token(
                fingerprint,
                {"submitted_at_ms": last.submitted_at_ms.epoch_ms(), "task_id": last.task_id.to_wire()},
            )
        return Page(
            items=items, next_page_token=next_token, source_statuses=resource_source_statuses(self._dependencies)
        )

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        return self._describe_tasks((key,), include_failure_highlights=True)[0]

    def describe_tasks(self, keys: Sequence[ResourceKey]) -> tuple[TaskDetail, ...]:
        """Describe a bounded Task collection without per-Task finelog enrichment."""
        return self._describe_tasks(keys, include_failure_highlights=False)

    def _describe_tasks(
        self,
        keys: Sequence[ResourceKey],
        *,
        include_failure_highlights: bool,
    ) -> tuple[TaskDetail, ...]:
        if not keys:
            return ()
        if len(keys) > _MAX_TASK_PAGE:
            raise ValueError(f"Task detail batch cannot exceed {_MAX_TASK_PAGE} items")
        for key in keys:
            _require_kind(key, ResourceKind.TASK)
        task_ids = [JobName.from_wire(key.resource_id) for key in keys]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task detail keys must be unique")
        with self._dependencies.db.read_snapshot() as tx:
            rows_by_id = reads.bulk_get_task_detail(tx, task_ids)
            rows = list(rows_by_id.values())
            attempts_by_task = reads.all_attempts_for_tasks(tx, task_ids)
            jobs = self._job_rows(tx, {row.job_id for row in rows})
        source_statuses = resource_source_statuses(self._dependencies)
        details = []
        for key, task_id in zip(keys, task_ids, strict=True):
            row = rows_by_id.get(task_id)
            if row is None:
                raise ResourceNotFound(key.resource_id)
            attempts = attempts_by_task.get(task_id, ())
            current = next((candidate for candidate in attempts if candidate.attempt_id == row.current_attempt_id), None)
            job = jobs[row.job_id]
            summary = self._task_summary(
                row,
                current,
                AttemptCounts(row.failure_count, row.preemption_count),
                job,
            )
            if summary.identity.key.cluster_id != key.cluster_id:
                raise ResourceNotFound(key.resource_id)
            failure_highlights = (
                self._failure_highlights(task_id, summary.state)
                if include_failure_highlights
                else _FailureHighlights((), None)
            )
            detail_sources = source_statuses
            if failure_highlights.source_status is not None:
                detail_sources += (failure_highlights.source_status,)
            details.append(
                TaskDetail(
                    summary=summary,
                    attempts=tuple(self._attempt_summary(row, candidate, job) for candidate in attempts),
                    source_statuses=detail_sources,
                    root_cause_highlights=failure_highlights.entries,
                )
            )
        return tuple(details)

    def _failure_highlights(self, task_id: JobName, state: TaskState) -> _FailureHighlights:
        if state not in (TaskState.FAILED, TaskState.WORKER_FAILED):
            return _FailureHighlights((), None)
        if self._dependencies.log_reader is None:
            status = _unavailable_finelog_source(self._dependencies.cluster_id, RuntimeError(_FINELOG_NOT_CONFIGURED))
            return _FailureHighlights((), status)
        source, match_scope = build_log_source(task_id)
        try:
            entries, _ = self._dependencies.log_reader.fetch_logs(
                source=source,
                match_scope=match_scope,
                query=LogQuery(max_lines=200, tail=True),
            )
        except LogReadError as exc:
            return _FailureHighlights((), _unavailable_finelog_source(self._dependencies.cluster_id, exc))
        return _FailureHighlights(
            tuple(extract_failure_highlights([entry.data for entry in entries])),
            _available_source(f"finelog:{self._dependencies.cluster_id}"),
        )

    def require_task(self, identity: TaskIdentity) -> TaskDetail:
        detail = self.describe_task(identity.key)
        if detail.summary.identity.task_uid != identity.task_uid:
            raise ResourceReplaced(identity.key.resource_id)
        return detail

    def _task_summary(
        self,
        row: TaskDetailRow,
        current_attempt: reads.AttemptRecord | None,
        counts: AttemptCounts,
        job: reads.JobCoordinates,
    ) -> TaskSummary:
        authority = self._authority_cluster(job)
        execution = _execution_cluster(self._dependencies.cluster_id, str(row.cluster))
        task_key = ResourceKey(authority, ResourceKind.TASK, row.task_id.to_wire())
        job_identity = self._job_identity(job)
        task_identity = TaskIdentity(task_key, _task_uid(job_identity.job_uid, row.task_id))
        attempt_identity = None
        node_identity = None
        stored_backend_id = str(row.backend_id)
        backend_id = (
            self._execution_backend_id(stored_backend_id, execution)
            if stored_backend_id or current_attempt is not None
            else ""
        )
        if current_attempt is not None:
            attempt_identity = AttemptIdentity(
                task_key, int(current_attempt.attempt_id), str(current_attempt.attempt_uid)
            )
            node_id = str(current_attempt.node_name or current_attempt.worker_id or row.peer_worker_label or "")
            if node_id and backend_id:
                node_identity = self._current_node_identity(execution, backend_id, node_id)
        return TaskSummary(
            identity=task_identity,
            job=job_identity,
            task_index=int(row.task_index),
            state=TaskState(row.state),
            execution_cluster_id=execution,
            backend_id=backend_id,
            current_attempt=attempt_identity,
            current_node=node_identity,
            failure_count=counts.failure_count,
            preemption_count=counts.preemption_count,
            submitted_at=row.submitted_at_ms,
            started_at=current_attempt.started_at_ms if current_attempt is not None else None,
            finished_at=current_attempt.finished_at_ms if current_attempt is not None else None,
            status_message=str(row.status_message or ""),
            error_message=str(row.error or ""),
        )

    def _attempt_summary(
        self,
        task: TaskDetailRow,
        attempt: reads.AttemptRecord,
        job: reads.JobCoordinates,
    ) -> AttemptSummary:
        authority = self._authority_cluster(job)
        task_key = ResourceKey(authority, ResourceKind.TASK, task.task_id.to_wire())
        backend_id = str(attempt.backend_id or task.backend_id or "")
        execution = _execution_cluster(self._dependencies.cluster_id, str(task.cluster)) if backend_id else ""
        node_id = str(attempt.node_name or attempt.worker_id or "")
        node = None
        if node_id and backend_id:
            node = self._current_node_identity(execution, backend_id, node_id)
        return AttemptSummary(
            identity=AttemptIdentity(task_key, int(attempt.attempt_id), str(attempt.attempt_uid)),
            state=TaskState(attempt.state),
            execution_cluster_id=execution,
            backend_id=backend_id,
            node=node,
            created_at=attempt.created_at_ms,
            started_at=attempt.started_at_ms,
            finished_at=attempt.finished_at_ms,
            exit_code=attempt.exit_code,
            error_message=str(attempt.error or ""),
            terminal_reason=str(attempt.terminal_reason or ""),
        )

    def _job_identity(self, row: reads.JobCoordinates) -> JobIdentity:
        authority = self._authority_cluster(row)
        return JobIdentity(
            ResourceKey(authority, ResourceKind.JOB, row.job_id.to_wire()),
            _job_uid(
                authority,
                row.job_id,
                row.submitted_at_ms,
                handoff_nonce=str(row.handoff_nonce or ""),
            ),
        )

    def _current_node_identity(self, execution: str, backend_id: str, node_id: str) -> NodeIdentity | None:
        if execution != self._dependencies.cluster_id or not backend_id or not node_id:
            return None
        backend = self._dependencies.backends.get(backend_id)
        if backend is None or BackendCapability.WORKER_DAEMON not in backend.capabilities:
            return None
        return NodeIdentity(ResourceKey(execution, ResourceKind.NODE, node_id), backend_id, node_id)

    def _authority_cluster(self, row: reads.JobCoordinates) -> str:
        if row.direction == int(FederationDirection.RECEIVED):
            return str(row.peer_id)
        return self._dependencies.cluster_id

    def _backend_id(self, stored: str) -> str:
        if stored:
            if stored not in self._dependencies.backends:
                raise BackendIdentityUnknown(stored)
            return stored
        if len(self._dependencies.backends) == 1:
            return next(iter(self._dependencies.backends))
        raise BackendIdentityUnknown("Task has no retained backend coordinate")

    def _execution_backend_id(self, stored: str, execution_cluster_id: str) -> str:
        if execution_cluster_id != self._dependencies.cluster_id:
            return stored
        return self._backend_id(stored)

    def _job_rows(self, tx: Tx, job_ids: set[JobName]) -> dict[JobName, reads.JobCoordinates]:
        return reads.job_coordinates(tx, job_ids)
