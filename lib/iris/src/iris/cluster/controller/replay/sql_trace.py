# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite trace plumbing for replay scenarios.

Hooks into ``ControllerDB._trace_callback`` (a class-level slot) so every
SQL statement executed on either the writer connection or the read pool
is captured. Callers must enter ``sql_tracing`` *before* constructing
``ControllerDB`` — the writer connection runs ``_configure`` once at
``__init__`` and binds whatever callback is set at that moment.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager

from iris.cluster.controller.db import ControllerDB

# Collapse any run of internal whitespace (including newlines and tabs that
# multiline SQL strings inevitably produce) to a single space, but never
# touch parameter placeholders.
_INTERNAL_WS = re.compile(r"\s+")
# Strip random temp-dir paths from ATTACH DATABASE and similar statements
# so the trace stays stable across runs.
_TMP_PATH = re.compile(r"'/tmp/[^']*'")
# Migration timestamps are recorded as the wall-clock time when migrations
# ran (we can't shift _now_ms before the writer connection exists), so
# erase them from the trace.
_MIGRATION_INSERT = re.compile(r"(INSERT INTO schema_migrations\(name, applied_at_ms\) VALUES \('[^']+', )\d+(\))")


def normalize(statement: str) -> str:
    """Render an executed SQL statement into a stable, comparable form.

    Strips trailing whitespace, collapses internal whitespace runs, and
    rewrites a small set of unstable values (temp paths, migration
    timestamps) so the captured trace is byte-stable across runs.
    """
    text = _INTERNAL_WS.sub(" ", statement).strip()
    text = _TMP_PATH.sub("'<tmp>'", text)
    text = _MIGRATION_INSERT.sub(r"\1<ms>\2", text)
    return text


@contextmanager
def sql_tracing(log: list[str]) -> Iterator[list[str]]:
    """Capture every executed SQL statement into ``log`` for the duration.

    Sets ``ControllerDB._trace_callback`` on entry and restores it on
    exit. Connections opened *while* the manager is active will register
    the callback in ``_configure``; connections opened beforehand will
    not, so build the ``ControllerDB`` inside the ``with`` block.
    """

    def callback(sql: str) -> None:
        log.append(normalize(sql))

    prior = ControllerDB._trace_callback
    ControllerDB._trace_callback = callback
    try:
        yield log
    finally:
        ControllerDB._trace_callback = prior
