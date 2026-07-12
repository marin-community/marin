# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-domain helpers that map iris values onto the finelog store.

This module owns the mappings between iris-domain values (`JobName`,
`TaskAttempt`, capture streams) and the finelog wire types: the opaque string
keys that identify log streams, and the `LogLevel` assigned to each line.
"""

from finelog.rpc import logging_pb2
from finelog.types import str_to_log_level
from rigging.log_setup import parse_log_level

from iris.cluster.log_highlights import is_progress_bar_line
from iris.cluster.types import JobName, TaskAttempt

CONTROLLER_LOG_KEY = "/system/controller"
_WORKER_LOG_PREFIX = "/system/worker/"

STDOUT_SOURCE = "stdout"
STDERR_SOURCE = "stderr"
# The synthetic source iris stamps on failure lines it injects itself (OOM kills,
# infrastructure errors), as opposed to anything the task wrote.
INJECTED_ERROR_SOURCE = "error"

# Default level per capture stream when a line carries no parseable prefix.
# Streams not listed here (e.g. "build") fall back to UNKNOWN, which stays
# visible under every min_level filter.
_STREAM_DEFAULT_LEVEL = {
    STDOUT_SOURCE: logging_pb2.LOG_LEVEL_INFO,
    STDERR_SOURCE: logging_pb2.LOG_LEVEL_ERROR,
}


def classify_log_level(source: str, data: str) -> int:
    """Assign a finelog ``LogLevel`` to a captured task log line.

    Lines from ``INJECTED_ERROR_SOURCE`` are errors whatever they say. Otherwise
    a glog-style level prefix in ``data`` wins, so a prefixed ``INFO`` line on
    ``stderr`` classifies as ``INFO``. An unprefixed tqdm progress bar on
    ``stderr`` classifies as ``INFO``. Any other unprefixed line takes its
    stream's default: ``stdout`` informational, ``stderr`` error, and an
    unrecognized stream ``UNKNOWN``, which passes every ``min_level`` filter.
    """
    if source == INJECTED_ERROR_SOURCE:
        return logging_pb2.LOG_LEVEL_ERROR
    parsed = str_to_log_level(parse_log_level(data))
    if parsed != logging_pb2.LOG_LEVEL_UNKNOWN:
        return parsed
    if source == STDERR_SOURCE and is_progress_bar_line(data):
        return logging_pb2.LOG_LEVEL_INFO
    return _STREAM_DEFAULT_LEVEL.get(source, logging_pb2.LOG_LEVEL_UNKNOWN)


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
