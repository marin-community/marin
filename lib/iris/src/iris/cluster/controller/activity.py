# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

import hashlib
import json
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.attempt import AttemptResources
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.job import JobService
from iris.cluster.controller.pagination import (
    _decode_page_token,
    _encode_page_token,
    _page_size,
    _query_fingerprint,
)
from iris.cluster.controller.persistence import action as action_persistence
from iris.cluster.controller.source_status import (
    _available_source,
    _unavailable_finelog_source,
    _unsupported_source,
)
from iris.cluster.controller.task import TaskResources
from iris.cluster.log_keys import build_log_source
from iris.cluster.types import (
    JobName,
)
from iris.resources.action import ActionReceipt, ActionState
from iris.resources.activity import ActivityEntry, ActivityQuery
from iris.resources.errors import (
    InvalidPageToken,
    InvalidResourceKey,
    ResourceReplaced,
)
from iris.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    ResourceKey,
    ResourceKind,
    TaskIdentity,
)
from iris.resources.log import (
    LogPage,
    LogQuery,
    LogReadError,
    TaskEvent,
    TaskEventCursor,
    TaskEventKey,
    TaskEventQuery,
)
from iris.resources.source import (
    Page,
    ResourceSourceStatus,
)

_MAX_ACTIVITY_PAGE = 500
_FINELOG_NOT_CONFIGURED = "finelog is not configured"


@dataclass(frozen=True, slots=True)
class _ActivityItem:
    entry: ActivityEntry
    source_rank: int
    source_key: tuple[int | str, ...]

    @property
    def order_key(self) -> tuple[int, int, tuple[int | str, ...]]:
        return (self.entry.occurred_at.epoch_ms(), self.source_rank, self.source_key)


class ActivityResources:
    """Activity resource operations."""

    def __init__(
        self,
        dependencies: ResourceDependencies,
        jobs: "JobService",
        tasks: "TaskResources",
        attempts: "AttemptResources",
    ) -> None:
        self._dependencies = dependencies
        self._jobs = jobs
        self._tasks = tasks
        self._attempts = attempts

    def fetch_logs(
        self,
        target: JobIdentity | TaskIdentity | AttemptIdentity,
        query: LogQuery = LogQuery(),
    ) -> LogPage:
        job_name, attempt_number = self._validated_log_target(target)
        if self._dependencies.log_reader is None:
            return LogPage(
                entries=(),
                next_cursor=query.cursor,
                source_statuses=(
                    _unavailable_finelog_source(
                        self._dependencies.cluster_id,
                        RuntimeError(_FINELOG_NOT_CONFIGURED),
                    ),
                ),
            )
        source, match_scope = build_log_source(job_name, attempt_number)
        try:
            entries, next_cursor = self._dependencies.log_reader.fetch_logs(
                source=source,
                match_scope=match_scope,
                query=query,
            )
        except LogReadError as exc:
            return LogPage(
                entries=(),
                next_cursor=query.cursor,
                source_statuses=(_unavailable_finelog_source(self._dependencies.cluster_id, exc),),
            )
        return LogPage(
            entries=entries,
            next_cursor=next_cursor,
            source_statuses=(_available_source(f"finelog:{self._dependencies.cluster_id}"),),
        )

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        page_size = _page_size(query.page_size, _MAX_ACTIVITY_PAGE)
        fingerprint = _query_fingerprint(
            "activity",
            {
                "cluster_id": query.target.cluster_id,
                "kind": query.target.kind.value,
                "resource_id": query.target.resource_id,
                "attempt_uid": query.attempt_uid,
                "after_ms": query.after.epoch_ms() if query.after is not None else None,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        before_time = None
        before_source_rank = None
        before_source_key: tuple[int | str, ...] = ()
        if position is not None:
            try:
                before_time = Timestamp.from_ms(int(position["occurred_at_ms"]))
                before_source_rank = int(position["source_rank"])
                source_key = position["source_key"]
                if before_source_rank not in (0, 1) or not isinstance(source_key, list):
                    raise ValueError
                before_source_key = tuple(source_key)
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidPageToken("malformed activity page position") from exc
        attempt_uids = self._activity_attempt_uids(query)
        action_before = None
        if before_time is not None:
            if before_source_rank == 0:
                if len(before_source_key) != 1 or not isinstance(before_source_key[0], str):
                    raise InvalidPageToken("malformed action activity position")
                action_before = (before_time, f"action:{before_source_key[0]}")
            else:
                action_before = (before_time, "\U0010ffff")
        with self._dependencies.db.read_snapshot() as tx:
            receipts = action_persistence.actions_for_target(
                tx,
                query.target,
                after=query.after,
                before=action_before,
                limit=page_size + 1,
            )
        entries = [_ActivityItem(self._action_activity(receipt), 0, (receipt.action_id,)) for receipt in receipts]
        source_statuses = [_available_source(f"controller:{self._dependencies.cluster_id}")]
        if attempt_uids:
            event_entries, event_status = self._task_event_activity(
                query.target,
                attempt_uids,
                query.after,
                before_time,
                before_source_rank,
                before_source_key,
                page_size + 1,
            )
            entries.extend(event_entries)
            source_statuses.append(event_status)
        else:
            source_statuses.append(_unsupported_source(f"finelog:{self._dependencies.cluster_id}"))
        entries.sort(key=lambda item: item.order_key, reverse=True)
        items = tuple(item.entry for item in entries[:page_size])
        next_token = None
        if len(entries) > page_size:
            last = entries[page_size - 1]
            next_token = _encode_page_token(
                fingerprint,
                {
                    "occurred_at_ms": last.entry.occurred_at.epoch_ms(),
                    "source_rank": last.source_rank,
                    "source_key": last.source_key,
                },
            )
        return Page(items=items, next_page_token=next_token, source_statuses=tuple(source_statuses))

    def _activity_attempt_uids(self, query: ActivityQuery) -> tuple[str, ...]:
        target = query.target
        if target.kind is ResourceKind.JOB:
            self._jobs.describe_job(target)
            return ()
        if target.kind is ResourceKind.TASK:
            detail = self._tasks.describe_task(target)
            available = tuple(item.identity.attempt_uid for item in detail.attempts)
        elif target.kind is ResourceKind.ATTEMPT:
            task_id, _, number_text = target.resource_id.rpartition(":")
            task_key = ResourceKey(target.cluster_id, ResourceKind.TASK, task_id)
            detail = self._attempts.describe_attempt(AttemptLocator(task_key, int(number_text)))
            available = (detail.summary.identity.attempt_uid,)
        else:
            raise InvalidResourceKey(f"activity is unsupported for {target.kind.value}")
        if query.attempt_uid is None:
            return available
        if query.attempt_uid not in available:
            raise ResourceReplaced(target.resource_id)
        return (query.attempt_uid,)

    def _task_event_activity(
        self,
        target: ResourceKey,
        attempt_uids: tuple[str, ...],
        after: Timestamp | None,
        before_time: Timestamp | None,
        before_source_rank: int | None,
        before_source_key: tuple[int | str, ...],
        limit: int,
    ) -> tuple[tuple[_ActivityItem, ...], ResourceSourceStatus]:
        if self._dependencies.log_reader is None:
            error = RuntimeError(_FINELOG_NOT_CONFIGURED)
            return (), _unavailable_finelog_source(self._dependencies.cluster_id, error)
        task_id = target.resource_id.rpartition(":")[0] if target.kind is ResourceKind.ATTEMPT else target.resource_id
        before = None
        if before_time is not None:
            key = self._task_event_key(before_source_key) if before_source_rank == 1 else None
            before = TaskEventCursor(before_time, key)
        try:
            events = self._dependencies.log_reader.task_events(
                TaskEventQuery(
                    task_id=task_id,
                    attempt_uids=attempt_uids,
                    after=after,
                    before=before,
                    limit=limit,
                )
            )
        except LogReadError as exc:
            return (), _unavailable_finelog_source(self._dependencies.cluster_id, exc)
        entries = tuple(
            _ActivityItem(self._task_event_entry(task_id, event), 1, self._task_event_source_key(event))
            for event in events
        )
        return entries, _available_source(f"finelog:{self._dependencies.cluster_id}")

    @staticmethod
    def _task_event_key(source_key: tuple[int | str, ...]) -> TaskEventKey:
        if len(source_key) != 7:
            raise InvalidPageToken("malformed task event activity position")
        attempt_id, attempt_uid, event_type, reason, message, source, count = source_key
        if type(attempt_id) is not int or type(count) is not int:
            raise InvalidPageToken("malformed task event activity position")
        if not all(isinstance(value, str) for value in (attempt_uid, event_type, reason, message, source)):
            raise InvalidPageToken("malformed task event activity position")
        return TaskEventKey(attempt_id, attempt_uid, event_type, reason, message, source, count)

    @staticmethod
    def _task_event_source_key(event: TaskEvent) -> tuple[int | str, ...]:
        key = event.key
        return (
            key.attempt_id,
            key.attempt_uid,
            key.event_type,
            key.reason,
            key.message,
            key.source,
            key.count,
        )

    def _task_event_entry(self, task_id: str, event: TaskEvent) -> ActivityEntry:
        sequence_source = json.dumps(
            [
                task_id,
                event.attempt_uid,
                event.occurred_at.epoch_ms(),
                event.event_type,
                event.reason,
                event.message,
                event.source,
                event.count,
            ],
            separators=(",", ":"),
        ).encode()
        sequence = int.from_bytes(hashlib.sha256(sequence_source).digest()[:8], "big")
        return ActivityEntry(
            entry_id=f"finelog:task-events:{sequence}",
            occurred_at=event.occurred_at,
            source=event.source,
            severity=event.event_type,
            kind=event.reason,
            message=event.message,
            target=ResourceKey(
                self._dependencies.cluster_id,
                ResourceKind.ATTEMPT,
                f"{task_id}:{event.attempt_id}",
            ),
            attempt_uid=event.attempt_uid,
            correlation_id=None,
            attributes={"count": str(event.count)},
        )

    @staticmethod
    def _action_activity(receipt: ActionReceipt) -> ActivityEntry:
        return ActivityEntry(
            entry_id=f"action:{receipt.action_id}",
            occurred_at=receipt.updated_at,
            source="controller",
            severity="error" if receipt.state is ActionState.FAILED else "info",
            kind=receipt.kind.value,
            message=receipt.result_message or receipt.result_code.value,
            target=receipt.target,
            attempt_uid=receipt.expected_attempt_uid,
            correlation_id=receipt.action_id,
            attributes={"state": receipt.state.value, "result": receipt.result_code.value},
        )

    def _validated_log_target(
        self,
        target: JobIdentity | TaskIdentity | AttemptIdentity,
    ) -> tuple[JobName, int]:
        if isinstance(target, JobIdentity):
            detail = self._jobs.describe_job(target.key)
            if detail.summary.identity.job_uid != target.job_uid:
                raise ResourceReplaced(target.key.resource_id)
            return JobName.from_wire(target.key.resource_id), -1
        if isinstance(target, TaskIdentity):
            self._tasks.require_task(target)
            return JobName.from_wire(target.key.resource_id), -1
        self._attempts.require_attempt(target)
        return JobName.from_wire(target.task.resource_id), target.attempt_number
