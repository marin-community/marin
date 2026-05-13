# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-domain helpers that produce opaque string keys for the finelog store.

Keys are plain strings on the finelog side; this module owns the mapping from
iris-domain values (`JobName`, `TaskAttempt`) to those strings.
"""

from __future__ import annotations

from finelog.rpc import logging_pb2

from iris.cluster.types import JobName, TaskAttempt

CONTROLLER_LOG_KEY = "/system/controller"
_WORKER_LOG_PREFIX = "/system/worker/"


def worker_log_key(worker_id: str) -> str:
    """Build the log store key for a worker's process logs."""
    return f"{_WORKER_LOG_PREFIX}{worker_id}"


def task_log_key(task_attempt: TaskAttempt) -> str:
    """Build a hierarchical key for task attempt logs."""
    task_attempt.require_attempt()
    return task_attempt.to_wire()


def build_log_source(target: JobName, attempt_id: int = -1) -> tuple[str, logging_pb2.MatchScope]:
    """Build a (literal source, match scope) tuple for FetchLogs.

    The source is always a literal string — finelog matches `+`, `.`, `[` etc.
    byte-for-byte. ``match_scope`` tells the server how to interpret it.

    - Task + specific attempt: ``(/user/job/0:<attempt_id>, EXACT)``
    - Task + all attempts:     ``(/user/job/0:, PREFIX)``
    - Job (all tasks):         ``(/user/job/, PREFIX)``
    """
    wire = target.to_wire()
    if target.is_task:
        if attempt_id >= 0:
            return f"{wire}:{attempt_id}", logging_pb2.MATCH_SCOPE_EXACT
        return f"{wire}:", logging_pb2.MATCH_SCOPE_PREFIX
    return f"{wire}/", logging_pb2.MATCH_SCOPE_PREFIX
