# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""On-demand check that the controller's observability rows reached Finelog.

The controller writes ``iris.controller_db__jobs``, ``__job_config``, ``__tasks``
and ``__task_attempts`` without blocking on Finelog, so a failed snapshot, enqueue
or flush drops rows (see ``iris.cluster.controller.observability_rows``). This
check compares, for one cluster and one UTC day, the keys in the controller DB
with the keys in Finelog:

- jobs and job configs submitted that day, by job key;
- tasks of the jobs submitted that day, by ``task_id``;
- attempts created that day, by ``attempt_uid``;
- of those jobs and attempts, the ones that finished before the settle cutoff,
  against Finelog keys that have a terminal-state row. A dropped finish batch
  leaves the key in Finelog (from its submit or start row), so only this count
  catches it.

The controller keeps terminal jobs (and, through the cascade, their attempts)
for ``JOB_RETENTION`` after they finish, so a day inside that window is
complete in the DB. Finelog keys the DB lacks are reported but are not a
failure: a finished job is also removed early when its id is resubmitted.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

from iris.cluster.controller.pruner import JOB_RETENTION
from iris.cluster.stats.tables import (
    CONTROLLER_JOB_CONFIG_NAMESPACE,
    CONTROLLER_JOB_NAMESPACE,
    CONTROLLER_TASK_ATTEMPT_NAMESPACE,
    CONTROLLER_TASK_NAMESPACE,
)
from iris.cluster.types import LOCAL_CLUSTER, TERMINAL_JOB_STATES, TERMINAL_TASK_STATES
from iris.rpc import job_pb2

# Runs a SELECT and returns its rows as sequences, in column order.
RowQuery = Callable[[str], Sequence[Sequence[object]]]

# A job key (root_job_id, root_submitted_at_ms, job_id, submitted_at_ms), a task
# key (task_id,), or an attempt key (attempt_uid,).
Key = tuple[object, ...]

_CLUSTER_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# Full days inside the retention window. No cluster config overrides
# ControllerConfig.job_retention, so its default is the retention in force.
# ``days_to_check`` also drops a day that starts before the pruner's cutoff, so
# every job submitted on a checked day is still in the DB.
MAX_CHECK_DAYS = int(JOB_RETENTION.to_seconds() // 86400) - 1
# A finish this recent may still have its terminal row in flight, and a day is
# checked only once it ended this long ago.
SETTLE_MARGIN = timedelta(hours=1)


class RowTable(StrEnum):
    JOB = CONTROLLER_JOB_NAMESPACE
    JOB_CONFIG = CONTROLLER_JOB_CONFIG_NAMESPACE
    TASK = CONTROLLER_TASK_NAMESPACE
    TASK_ATTEMPT = CONTROLLER_TASK_ATTEMPT_NAMESPACE


@dataclass(frozen=True)
class DayWindow:
    """One UTC day to check, and the times that bound its queries.

    ``settled_before`` is the cutoff for "finished": an entity that finished
    later may still have its terminal row in flight. ``until`` is the upper
    bound on Finelog ``ts``.
    """

    day: date
    settled_before: datetime
    until: datetime

    @property
    def start(self) -> datetime:
        return datetime.combine(self.day, time(), tzinfo=UTC)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=1)


@dataclass(frozen=True)
class KeyRows:
    """The keys one side holds for a day, and the subset that is terminal."""

    keys: frozenset[Key]
    terminal: frozenset[Key]


@dataclass(frozen=True)
class TableCoverage:
    """How completely one table's keys for one day reached Finelog."""

    table: RowTable
    day: date
    controller: int
    finelog: int
    missing: tuple[Key, ...]
    extra: tuple[Key, ...]
    settled_terminal: int
    missing_terminal: tuple[Key, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.missing_terminal


def days_to_check(now: datetime, *, start: date, days: int) -> list[date]:
    """The UTC days to check at ``now``: up to ``days`` days, from ``start`` on.

    A day is checked once it ended ``SETTLE_MARGIN`` ago, and only if it starts
    after ``now - JOB_RETENTION``, when the pruner may begin deleting its jobs.
    """
    if not 1 <= days <= MAX_CHECK_DAYS:
        raise ValueError(f"days={days}: check 1 to {MAX_CHECK_DAYS} full days, inside the controller's job retention")
    end = (now - SETTLE_MARGIN).date()
    retained = (now - timedelta(seconds=JOB_RETENTION.to_seconds())).date() + timedelta(days=1)
    first = max(start, end - timedelta(days=days), retained)
    return [first + timedelta(days=i) for i in range((end - first).days)]


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _ts(value: datetime) -> str:
    return f"TIMESTAMP '{value.astimezone(UTC).strftime('%Y-%m-%d %H:%M:%S')}'"


def _int_list(values: frozenset[int]) -> str:
    return ", ".join(str(v) for v in sorted(values))


def _name_list(names: list[str]) -> str:
    return ", ".join(f"'{n}'" for n in sorted(names))


_TERMINAL_JOB_NAMES = [job_pb2.JobState.Name(s) for s in TERMINAL_JOB_STATES]
_TERMINAL_TASK_NAMES = [job_pb2.TaskState.Name(s) for s in TERMINAL_TASK_STATES]
_JOB_KEY = "root_job_id, root_submitted_at_ms, job_id, submitted_at_ms"


def _terminal_flag(state_names: list[str]) -> str:
    """A 0/1 aggregate: whether any of a key's Finelog rows has a terminal state."""
    return f"max(CASE WHEN state IN ({_name_list(state_names)}) THEN 1 ELSE 0 END) AS terminal"


def controller_sql(table: RowTable, window: DayWindow) -> str:
    """The controller DB query for ``table``'s keys on ``window.day``: key columns, then a 0/1 settled-terminal flag.

    Only jobs this controller runs (``cluster = 'local'``) are written to Finelog,
    so hand-offs and federation mirrors are left out.
    """
    lo, hi, settled = _ms(window.start), _ms(window.end), _ms(window.settled_before)
    if table is RowTable.TASK_ATTEMPT:
        return (
            "SELECT a.attempt_uid, "
            f"(a.state IN ({_int_list(TERMINAL_TASK_STATES)}) AND a.finished_at_ms < {settled}) AS settled "
            "FROM task_attempts a JOIN tasks t ON t.task_id = a.task_id JOIN jobs j ON j.job_id = t.job_id "
            f"WHERE j.cluster = '{LOCAL_CLUSTER}' AND a.created_at_ms >= {lo} AND a.created_at_ms < {hi}"
        )
    keys = ", ".join(f"j.{column}" for column in _JOB_KEY.split(", "))
    where = f"WHERE j.cluster = '{LOCAL_CLUSTER}' AND j.submitted_at_ms >= {lo} AND j.submitted_at_ms < {hi}"
    if table is RowTable.TASK:
        return (
            "SELECT t.task_id, "
            f"(t.state IN ({_int_list(TERMINAL_TASK_STATES)}) AND t.finished_at_ms < {settled}) AS settled "
            f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {where}"
        )
    if table is RowTable.JOB_CONFIG:
        return f"SELECT {keys}, 0 AS settled FROM job_config c JOIN jobs j ON j.job_id = c.job_id {where}"
    return (
        f"SELECT {keys}, "
        f"(j.state IN ({_int_list(TERMINAL_JOB_STATES)}) AND j.finished_at_ms < {settled}) AS settled "
        f"FROM jobs j {where}"
    )


def finelog_sql(table: RowTable, cluster: str, window: DayWindow) -> str:
    """The Finelog query for ``table``'s keys on ``window.day``: key columns, then a 0/1 has-terminal-row flag.

    ``ts`` is bounded to ``[day start, until)``: every row of an entity created
    that day is written at or after its creation.
    """
    if not _CLUSTER_NAME.match(cluster):
        raise ValueError(f"not a cluster name: {cluster!r}")
    bounds = f"cluster = '{cluster}' AND ts >= {_ts(window.start)} AND ts < {_ts(window.until)}"
    if table is RowTable.TASK_ATTEMPT:
        return (
            "SELECT attempt_uid, "
            f"{_terminal_flag(_TERMINAL_TASK_NAMES)} "
            f'FROM "{table}" WHERE {bounds} '
            f"AND created_at >= {_ts(window.start)} AND created_at < {_ts(window.end)} "
            "GROUP BY attempt_uid"
        )
    day_filter = f"submitted_at_ms >= {_ms(window.start)} AND submitted_at_ms < {_ms(window.end)}"
    if table is RowTable.TASK:
        return (
            "SELECT task_id, "
            f"{_terminal_flag(_TERMINAL_TASK_NAMES)} "
            f'FROM "{table}" WHERE {bounds} AND {day_filter} '
            "GROUP BY task_id"
        )
    if table is RowTable.JOB_CONFIG:
        return f'SELECT DISTINCT {_JOB_KEY}, 0 AS terminal FROM "{table}" WHERE {bounds} AND {day_filter}'
    return (
        f"SELECT {_JOB_KEY}, "
        f"{_terminal_flag(_TERMINAL_JOB_NAMES)} "
        f'FROM "{table}" WHERE {bounds} AND {day_filter} '
        f"GROUP BY {_JOB_KEY}"
    )


def key_rows(rows: Sequence[Sequence[object]]) -> KeyRows:
    """Split query rows (key columns, then a 0/1 flag) into keys and flagged keys."""
    keys: set[Key] = set()
    flagged: set[Key] = set()
    for row in rows:
        key = tuple(row[:-1])
        keys.add(key)
        if row[-1]:
            flagged.add(key)
    return KeyRows(frozenset(keys), frozenset(flagged))


def table_coverage(table: RowTable, day: date, controller: KeyRows, finelog: KeyRows) -> TableCoverage:
    """Compare one day's controller keys with its Finelog keys."""
    return TableCoverage(
        table=table,
        day=day,
        controller=len(controller.keys),
        finelog=len(finelog.keys),
        missing=tuple(sorted(controller.keys - finelog.keys, key=repr)),
        extra=tuple(sorted(finelog.keys - controller.keys, key=repr)),
        settled_terminal=len(controller.terminal),
        missing_terminal=tuple(sorted(controller.terminal - finelog.terminal, key=repr)),
    )


def day_coverage(
    controller_query: RowQuery, finelog_query: RowQuery, cluster: str, window: DayWindow
) -> list[TableCoverage]:
    """Check every controller row table for one cluster and day."""
    return [
        table_coverage(
            table,
            window.day,
            key_rows(controller_query(controller_sql(table, window))),
            key_rows(finelog_query(finelog_sql(table, cluster, window))),
        )
        for table in RowTable
    ]


def format_report(cluster: str, coverage: Sequence[TableCoverage], *, max_keys: int) -> str:
    """A table of counts per day and table, then up to ``max_keys`` missing keys of each failing line."""
    lines = [
        f"cluster {cluster}",
        f"{'day':<10}  {'table':<33}  {'controller':>10}  {'finelog':>8}  {'missing':>7}  {'extra':>5}"
        f"  {'finished':>8}  {'no finish row':>13}  status",
    ]
    for c in coverage:
        lines.append(
            f"{c.day.isoformat():<10}  {c.table.value:<33}  {c.controller:>10}  {c.finelog:>8}  {len(c.missing):>7}"
            f"  {len(c.extra):>5}  {c.settled_terminal:>8}  {len(c.missing_terminal):>13}  {'ok' if c.ok else 'MISSING'}"
        )
    for c in coverage:
        for label, keys in (("missing", c.missing), ("no finish row", c.missing_terminal), ("extra", c.extra)):
            if not keys:
                continue
            lines.append(f"{c.day.isoformat()} {c.table.value} {label} ({len(keys)}):")
            lines.extend(f"  {key}" for key in keys[:max_keys])
    return "\n".join(lines)
