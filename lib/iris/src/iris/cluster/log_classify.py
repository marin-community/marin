# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assign a finelog ``LogLevel`` to a captured task log line.

A line carrying a recognizable level prefix (the glog-style ``I20260102
12:34:56 ...`` format) keeps that level. A line without one — raw user
``print`` output, tracebacks, third-party tool chatter — takes a default from
the stream it was captured on: ``stdout`` is informational, ``stderr`` and
iris's own injected failure lines are errors.

This keeps mundane stdout (boot diagnostics, ``sys.path`` dumps) out of the
``min_level``-filtered error view: an ``INFO`` line is excluded by an ``ERROR``
filter, whereas an ``UNKNOWN`` line passes through every filter.
"""

from finelog.rpc import logging_pb2
from finelog.types import str_to_log_level
from rigging.log_setup import parse_log_level

# Default level per capture stream when the line carries no parseable prefix.
# "error" is the synthetic source iris uses for injected failure lines (OOM
# kills, infrastructure errors). Streams not listed here (e.g. "build") fall
# back to UNKNOWN, which stays visible under every min_level filter.
_STREAM_DEFAULT_LEVEL = {
    "stdout": logging_pb2.LOG_LEVEL_INFO,
    "stderr": logging_pb2.LOG_LEVEL_ERROR,
    "error": logging_pb2.LOG_LEVEL_ERROR,
}


def classify_log_level(source: str, data: str) -> int:
    """Return the ``LogLevel`` enum value for a captured log line.

    A parseable level prefix in ``data`` always wins, so a glog-prefixed
    ``INFO`` line on ``stderr`` is classified ``INFO``, not ``ERROR``. Otherwise
    the level defaults from ``source`` (the capture stream).
    """
    parsed = str_to_log_level(parse_log_level(data))
    if parsed != logging_pb2.LOG_LEVEL_UNKNOWN:
        return parsed
    return _STREAM_DEFAULT_LEVEL.get(source, logging_pb2.LOG_LEVEL_UNKNOWN)
