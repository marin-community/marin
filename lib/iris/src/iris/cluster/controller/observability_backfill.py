# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-time backfill of the ``iris.controller_db__*`` Finelog rows from a copy of the controller DB.

The live path (:mod:`iris.cluster.controller.observability_rows`) writes a row
when a job, task or attempt changes, from the moment a controller with it
starts: ``live_start``. Jobs that ended before then, and entities that did not
change after it, have no row. This module reads a checkpoint copy of the
controller DB, taken after ``live_start``, and writes one row per such entity:
its state in the copy, built by the live path's own queries and row builders.

Two rules keep it from doubling the live rows:

- A key that already has a row in Finelog, live or from an earlier run, is not
  written again, so a second run writes nothing.
- Every backfilled row's ``ts`` is the entity's own latest time (finish, start,
  or submit/create), capped at ``live_start - 1 ms``. Every live row's ``ts`` is
  ``live_start`` or later, so a backfilled row never becomes an entity's newest
  row, even when a live row lands during the run.

Before writing, :func:`plan_backfill` refuses a ``live_start`` that is too
recent (G1), later than the checkpoint (G2), earlier than the real start of the
live rows (G3: a job submitted after it has no live config row), or later than
it (G4: rows before it already exist, outside a declared rerun).
"""

import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import batched
from typing import Protocol

import pyarrow as pa
from finelog.client.log_client import FlushResult, NamespaceInfo
from finelog.client.log_client import Table as FinelogTable
from finelog.errors import NamespaceNotFoundError, SchemaValidationError, StatsError
from finelog.policy import StoragePolicy
from rigging.timing import Timestamp

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.diagnostics import DiagnosticDependencies, probe_database
from iris.cluster.controller.observability_rows import (
    attempt_row,
    attempt_rows_query,
    job_config_row,
    job_config_rows_query,
    job_row,
    job_rows_query,
    task_row,
    task_rows_query,
)
from iris.cluster.stats.tables import (
    CONTROLLER_JOB_CONFIG_NAMESPACE,
    CONTROLLER_JOB_CONFIG_STORAGE_POLICY,
    CONTROLLER_JOB_NAMESPACE,
    CONTROLLER_JOB_STORAGE_POLICY,
    CONTROLLER_TASK_ATTEMPT_NAMESPACE,
    CONTROLLER_TASK_ATTEMPT_STORAGE_POLICY,
    CONTROLLER_TASK_NAMESPACE,
    CONTROLLER_TASK_STORAGE_POLICY,
    IrisJob,
    IrisJobConfig,
    IrisTask,
    IrisTaskAttempt,
)

logger = logging.getLogger(__name__)

# G1: live rows must have been running for a while before the backfill trusts the boundary.
LIVE_START_MARGIN = timedelta(minutes=15)
# G3: jobs submitted this long after live_start must already have their live config row.
SUBMIT_GRACE = timedelta(minutes=5)
# The key scan queries one day per table at a time, halving a failed window down to this.
MIN_KEY_WINDOW = timedelta(hours=1)
KEY_WINDOW = timedelta(days=1)
MAX_KEY_ROWS = 2_000_000
# Below the Finelog client's 10,000-row queue, which drops the oldest rows past it.
DEFAULT_BATCH_ROWS = 5_000
DEFAULT_PAUSE_SECONDS = 2.0
QUERY_PAUSE_SECONDS = 1.0
FLUSH_TIMEOUT_SECONDS = 60.0
_CLUSTER_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")

Row = IrisJob | IrisJobConfig | IrisTask | IrisTaskAttempt
Key = tuple


class BackfillRefused(Exception):
    """A guard failed or a step could not complete; nothing after it ran."""


class Querier(Protocol):
    def query(self, sql: str, *, max_rows: int) -> pa.Table: ...

    def list_namespaces(self) -> list[NamespaceInfo]: ...


@dataclass(frozen=True)
class Target:
    """One ``iris.controller_db__*`` table: its rows' key and the entity time a backfilled ``ts`` comes from."""

    namespace: str
    row_class: type
    storage_policy: StoragePolicy
    key_columns: tuple[str, ...]
    entity_time: Callable[[Row], datetime]

    def key(self, row: Row) -> Key:
        return tuple(getattr(row, column) for column in self.key_columns)


def _ms(value: int) -> datetime:
    return Timestamp.from_ms(value).as_naive_utc()


_JOB_KEY = ("root_job_id", "root_submitted_at_ms", "job_id", "submitted_at_ms")
JOBS = Target(
    CONTROLLER_JOB_NAMESPACE,
    IrisJob,
    CONTROLLER_JOB_STORAGE_POLICY,
    _JOB_KEY,
    lambda r: r.finished_at or r.started_at or _ms(r.submitted_at_ms),
)
JOB_CONFIGS = Target(
    CONTROLLER_JOB_CONFIG_NAMESPACE,
    IrisJobConfig,
    CONTROLLER_JOB_CONFIG_STORAGE_POLICY,
    _JOB_KEY,
    # The live path writes a config row once, at submit.
    lambda r: _ms(r.submitted_at_ms),
)
TASKS = Target(
    CONTROLLER_TASK_NAMESPACE,
    IrisTask,
    CONTROLLER_TASK_STORAGE_POLICY,
    (*_JOB_KEY, "task_id"),
    lambda r: r.finished_at or r.started_at or r.task_submitted_at,
)
ATTEMPTS = Target(
    CONTROLLER_TASK_ATTEMPT_NAMESPACE,
    IrisTaskAttempt,
    CONTROLLER_TASK_ATTEMPT_STORAGE_POLICY,
    ("attempt_uid",),
    lambda r: r.finished_at or r.started_at or r.created_at,
)
TARGETS = (JOBS, JOB_CONFIGS, TASKS, ATTEMPTS)


def db_rows(db: ControllerDB, cluster: str, live_start_ms: int) -> dict[Target, list[Row]]:
    """Every row the DB copy can give, one per entity, with its backfill ``ts``.

    A removed worker's attempts read ``worker_id`` as NULL: only the live path
    keeps the cleared value, in memory.
    """
    cap = _ms(live_start_ms - 1)
    with db.read_snapshot() as tx:
        rows: dict[Target, list[Row]] = {
            JOBS: [job_row(r, cluster, cap) for r in tx.execute(job_rows_query())],
            JOB_CONFIGS: [job_config_row(r, cluster, cap) for r in tx.execute(job_config_rows_query())],
            TASKS: [task_row(r, cluster, cap) for r in tx.execute(task_rows_query())],
            ATTEMPTS: [attempt_row(r, cluster, cap, r.worker_id) for r in tx.execute(attempt_rows_query())],
        }
    for target, target_rows in rows.items():
        for row in target_rows:
            row.ts = min(target.entity_time(row), cap)
    return rows


def _sql_time(value: datetime) -> str:
    return f"TIMESTAMP '{value.isoformat(sep=' ')}'"


def _scan_window(querier: Querier, target: Target, cluster: str, lo: datetime, hi: datetime) -> list[pa.Table]:
    columns = ", ".join(target.key_columns)
    sql = (
        f'SELECT {columns}, min(ts) AS first_ts FROM "{target.namespace}" '
        f"WHERE cluster = '{cluster}' AND ts >= {_sql_time(lo)} AND ts < {_sql_time(hi)} "
        f"GROUP BY {columns}"
    )
    try:
        return [querier.query(sql, max_rows=MAX_KEY_ROWS)]
    except (NamespaceNotFoundError, SchemaValidationError):
        raise
    except (StatsError, ConnectionError, OSError) as error:
        if hi - lo <= MIN_KEY_WINDOW:
            raise BackfillRefused(
                f"key scan of {target.namespace} [{lo}, {hi}) failed at the smallest window "
                f"({type(error).__name__}); nothing was written"
            ) from error
        mid = lo + (hi - lo) / 2
        logger.info("key scan of %s [%s, %s) failed (%s); halving", target.namespace, lo, hi, type(error).__name__)
        return [
            *_scan_window(querier, target, cluster, lo, mid),
            *_scan_window(querier, target, cluster, mid, hi),
        ]


def existing_keys(
    querier: Querier,
    target: Target,
    cluster: str,
    start: datetime,
    end: datetime,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[Key, datetime]:
    """Each key with a row for ``cluster`` in ``[start, end)``, and its earliest ``ts``; one day per query."""
    found: dict[Key, datetime] = {}
    lo = datetime(start.year, start.month, start.day)
    while lo < end:
        hi = lo + KEY_WINDOW
        for result in _scan_window(querier, target, cluster, lo, hi):
            for record in result.to_pylist():
                key = tuple(record[column] for column in target.key_columns)
                first = record["first_ts"]
                found[key] = min(first, found.get(key, first))
        lo = hi
        sleep(QUERY_PAUSE_SECONDS)
    return found


@dataclass
class TablePlan:
    target: Target
    db_rows: list[Row]
    existing: dict[Key, datetime]
    to_write: list[Row] = field(default_factory=list)

    @property
    def earliest_existing(self) -> datetime | None:
        return min(self.existing.values(), default=None)


def plan_backfill(
    db: ControllerDB,
    querier: Querier,
    *,
    cluster: str,
    live_start_ms: int,
    rerun: bool,
    now: datetime,
    sleep: Callable[[float], None] = time.sleep,
) -> list[TablePlan]:
    """Read the DB copy and Finelog's keys, check the guards, and return the rows to write.

    Raises :class:`BackfillRefused` when a guard fails. Nothing is written here.
    """
    if not _CLUSTER_NAME.fullmatch(cluster):
        raise BackfillRefused(f"cluster {cluster!r} is not a cluster config name")
    live_start = _ms(live_start_ms)
    if live_start > now - LIVE_START_MARGIN:
        raise BackfillRefused(f"G1: live_start {live_start} is less than {LIVE_START_MARGIN} before now ({now})")
    checkpoint_ms = probe_database(DiagnosticDependencies(db=db))
    if checkpoint_ms is None:
        raise BackfillRefused("G2: the DB copy records no checkpoint time; restore it from a checkpoint")
    if checkpoint_ms < live_start_ms:
        raise BackfillRefused(
            f"G2: the checkpoint ({_ms(checkpoint_ms)}) predates live_start ({live_start}); take one after it"
        )

    rows = db_rows(db, cluster, live_start_ms)
    oldest = min((row.ts for target_rows in rows.values() for row in target_rows), default=live_start)
    end = now + KEY_WINDOW
    # A table no row was ever written to is not registered yet, so it has no keys to scan.
    registered = {info.namespace for info in querier.list_namespaces()}
    plans = [
        TablePlan(
            target,
            rows[target],
            existing_keys(querier, target, cluster, oldest, end, sleep=sleep) if target.namespace in registered else {},
        )
        for target in TARGETS
    ]

    configs = next(plan for plan in plans if plan.target is JOB_CONFIGS)
    grace_ms = live_start_ms + int(SUBMIT_GRACE.total_seconds() * 1000)
    missing = [
        row
        for row in configs.db_rows
        if row.submitted_at_ms >= grace_ms and JOB_CONFIGS.key(row) not in configs.existing
    ]
    if missing:
        raise BackfillRefused(
            f"G3: {len(missing)} jobs submitted after live_start + {SUBMIT_GRACE} have no live config row: "
            "live_start is earlier than the live rows' start, or live rows are being lost"
        )
    before = {plan.target.namespace: sum(1 for first in plan.existing.values() if first < live_start) for plan in plans}
    if any(before.values()) and not rerun:
        raise BackfillRefused(
            f"G4: rows before live_start already exist ({before}): either live_start is later than the live rows' "
            "start, or an earlier backfill ran; pass --rerun only in the second case"
        )

    for plan in plans:
        plan.to_write = [row for row in plan.db_rows if plan.target.key(row) not in plan.existing]
        late = [row for row in plan.to_write if row.ts >= live_start]
        assert not late, f"{plan.target.namespace}: {len(late)} backfill rows at or after live_start"
        assert all(row.cluster == cluster for row in plan.to_write)
    return plans


def write_backfill(
    plans: Sequence[TablePlan],
    tables: dict[Target, FinelogTable],
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Write each table's rows in flushed batches, one table after another; stop at the first lost batch.

    A batch whose flush is not ``SUCCEEDED`` may be partly written; a rerun
    skips the keys that landed and writes the rest.
    """
    assert batch_rows < 10_000, "the Finelog client drops the oldest rows past 10,000 queued"
    for plan in plans:
        table = tables[plan.target]
        batches = list(batched(plan.to_write, batch_rows))
        written = 0
        for index, batch in enumerate(batches, start=1):
            table.write(batch)
            result = table.flush(timeout=FLUSH_TIMEOUT_SECONDS)
            if result != FlushResult.SUCCEEDED:
                raise BackfillRefused(
                    f"{plan.target.namespace}: batch {index}/{len(batches)} flush {result.value} after "
                    f"{written} of {len(plan.to_write)} rows; rerun with --rerun to write the rest"
                )
            written += len(batch)
            logger.info("%s: batch %d/%d written (%d rows)", plan.target.namespace, index, len(batches), written)
            sleep(pause_seconds)


def summary(plans: Iterable[TablePlan], *, batch_rows: int, pause_seconds: float) -> list[str]:
    """Counts per table and day (of the backfill ``ts``): rows in the DB, keys already written, rows to write."""
    lines = []
    for plan in plans:
        days: dict[str, list[int]] = {}
        for row in plan.db_rows:
            counts = days.setdefault(row.ts.date().isoformat(), [0, 0, 0])
            counts[0] += 1
            if plan.target.key(row) in plan.existing:
                counts[1] += 1
            else:
                counts[2] += 1
        batches = -(-len(plan.to_write) // batch_rows)
        lines.append(
            f"{plan.target.namespace}: {len(plan.db_rows)} in DB, {len(plan.to_write)} to write in {batches} batches "
            f"(~{batches * pause_seconds:.0f} s of pauses); earliest existing row "
            f"(the earliest live row only before the first backfill run): {plan.earliest_existing}"
        )
        for day, (in_db, present, to_write) in sorted(days.items()):
            lines.append(f"  {day}: {in_db} in DB, {present} already in Finelog, {to_write} to write")
    return lines
