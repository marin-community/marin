# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact resource log queries and pages."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from rigging.timing import Timestamp

from iris.resources.source import ResourceSourceStatus


class LogLevel(IntEnum):
    UNKNOWN = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


class LogMatchScope(IntEnum):
    """How the log store matches a native source key."""

    UNSPECIFIED = 0
    EXACT = 1
    PREFIX = 2
    REGEX = 3


@dataclass(frozen=True, slots=True)
class LogEntry:
    timestamp: Timestamp | None
    source: str
    data: str
    attempt_id: int
    level: LogLevel
    key: str
    sequence: int


@dataclass(frozen=True, slots=True)
class LogQuery:
    after: Timestamp | None = None
    cursor: int = 0
    max_lines: int = 1_000
    substring: str = ""
    minimum_level: LogLevel = LogLevel.UNKNOWN
    tail: bool = False


@dataclass(frozen=True, slots=True)
class LogPage:
    entries: tuple[LogEntry, ...]
    next_cursor: int
    source_statuses: tuple[ResourceSourceStatus, ...]


class LogReadError(RuntimeError):
    """The external log store could not serve a read."""


@dataclass(frozen=True, slots=True)
class TaskEventKey:
    """Stable ordering key for task-event activity at one timestamp."""

    attempt_id: int
    attempt_uid: str
    event_type: str
    reason: str
    message: str
    source: str
    count: int


@dataclass(frozen=True, slots=True)
class TaskEvent:
    """One native task-event row returned by the activity store."""

    attempt_id: int
    attempt_uid: str
    occurred_at: Timestamp
    event_type: str
    reason: str
    message: str
    source: str
    count: int

    @property
    def key(self) -> TaskEventKey:
        return TaskEventKey(
            attempt_id=self.attempt_id,
            attempt_uid=self.attempt_uid,
            event_type=self.event_type,
            reason=self.reason,
            message=self.message,
            source=self.source,
            count=self.count,
        )


@dataclass(frozen=True, slots=True)
class TaskEventCursor:
    """Exclusive upper bound for reverse-chronological task-event reads."""

    occurred_at: Timestamp
    key: TaskEventKey | None = None


@dataclass(frozen=True, slots=True)
class TaskEventQuery:
    """Bounded native query for one Task's retained Attempt events."""

    task_id: str
    attempt_uids: tuple[str, ...]
    after: Timestamp | None = None
    before: TaskEventCursor | None = None
    limit: int = 200

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.attempt_uids or any(not attempt_uid for attempt_uid in self.attempt_uids):
            raise ValueError("attempt_uids must be non-empty")
        if self.limit <= 0:
            raise ValueError("limit must be positive")


class LogReader(Protocol):
    """Native log and activity reads consumed by Iris behavior."""

    def fetch_logs(
        self,
        *,
        source: str,
        match_scope: LogMatchScope,
        query: LogQuery,
    ) -> tuple[tuple[LogEntry, ...], int]: ...

    def task_events(self, query: TaskEventQuery) -> tuple[TaskEvent, ...]: ...
