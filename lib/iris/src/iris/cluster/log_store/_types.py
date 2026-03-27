# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types and constants for the log store implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from iris.cluster.types import TaskAttempt
from iris.rpc import logging_pb2

PROCESS_LOG_KEY = "/system/process"

# Characters that indicate a regex pattern (vs. a literal key).
REGEX_META_RE = re.compile(r"[.*+?\[\](){}^$|\\]")

_EST_BYTES_PER_ROW = 256


def task_log_key(task_attempt: TaskAttempt) -> str:
    """Build a hierarchical key for task attempt logs."""
    task_attempt.require_attempt()
    return task_attempt.to_wire()


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry] = field(default_factory=list)
    cursor: int = 0  # max seq seen
