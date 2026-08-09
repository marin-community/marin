# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact resource log queries and pages."""

from dataclasses import dataclass
from enum import IntEnum

from rigging.timing import Timestamp

from iris.resources.source import ResourceSourceStatus


class LogLevel(IntEnum):
    UNKNOWN = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


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
