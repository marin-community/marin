# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types and constants for the log store implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from iris.cluster.types import JobName, TaskAttempt
from iris.rpc import logging_pb2

CONTROLLER_LOG_KEY = "/system/controller"
_WORKER_LOG_PREFIX = "/system/worker/"


def worker_log_key(worker_id: str) -> str:
    """Build the log store key for a worker's process logs."""
    return f"{_WORKER_LOG_PREFIX}{worker_id}"


# Characters that indicate a regex pattern (vs. a literal key).
REGEX_META_RE = re.compile(r"[.*+?\[\](){}^$|\\]")

_EST_BYTES_PER_ROW = 256


def task_log_key(task_attempt: TaskAttempt) -> str:
    """Build a hierarchical key for task attempt logs."""
    task_attempt.require_attempt()
    return task_attempt.to_wire()


def build_log_source(target: JobName, attempt_id: int = -1) -> str:
    """Build a FetchLogs source regex pattern from a JobName.

    Escapes regex metacharacters in the job name so they match literally,
    then appends the appropriate wildcard suffix.

    - Task + specific attempt: /user/job/0:<attempt_id>  (exact match)
    - Task + all attempts:     /user/job/0:.*
    - Job (all tasks):         /user/job/.*
    """
    wire = re.escape(target.to_wire())
    if target.is_task:
        if attempt_id >= 0:
            return f"{wire}:{attempt_id}"
        return f"{wire}:.*"
    return f"{wire}/.*"


class LogStoreProtocol(Protocol):
    """Minimal interface for log storage used by background collectors."""

    def append_batch(self, items: list[tuple[str, list]]) -> None: ...


class LogClientProtocol(Protocol):
    """Minimal interface for pushing log entries to the LogService."""

    def push(self, key: str, entries: list[logging_pb2.LogEntry]) -> None: ...


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry] = field(default_factory=list)
    cursor: int = 0  # max seq seen
