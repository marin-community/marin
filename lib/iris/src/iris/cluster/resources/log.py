# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact resource log queries and pages."""

from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.resources.source import ResourceSourceStatus
from iris.rpc import iris_logging_pb2


@dataclass(frozen=True, slots=True)
class LogQuery:
    after: Timestamp | None = None
    cursor: int = 0
    max_lines: int = 1_000
    substring: str = ""
    minimum_level: iris_logging_pb2.LogLevel = iris_logging_pb2.LOG_LEVEL_UNKNOWN
    tail: bool = False


@dataclass(frozen=True, slots=True)
class LogPage:
    entries: tuple[iris_logging_pb2.LogEntry, ...]
    next_cursor: int
    source_statuses: tuple[ResourceSourceStatus, ...]
