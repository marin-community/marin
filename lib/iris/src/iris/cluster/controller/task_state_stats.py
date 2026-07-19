# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``iris.task_state`` finelog namespace: periodic per-root-job task-state rows.

Every :data:`TASK_STATE_INTERVAL` the controller aggregates its local tasks per
root job — counts by state, plus how long the oldest PENDING task has waited for
dispatch and how long the oldest dispatched task has sat without reaching
RUNNING — and appends one row per root job with active tasks, plus one cluster
rollup row (``root_job_id=""``). The rollup row is written even when the cluster
is idle, so silence in the table means the controller (or this emitter) is down.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from finelog.client.log_client import Table
from rigging.timing import Timestamp

from iris.cluster.controller import reads
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.task_state import DISPATCHED_TASK_STATES
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# finelog namespace for ``IrisTaskState`` rows.
TASK_STATE_NAMESPACE = "iris.task_state"

# Emission cadence. Coarser than the control tick: these rows feed fleet
# dashboards and stuck-task alerting, where 30s resolution is plenty.
TASK_STATE_INTERVAL = 30.0

# ``root_job_id`` of the per-cluster rollup row (the sum over every root job).
CLUSTER_ROLLUP_ROOT_JOB = ""


@dataclass
class IrisTaskState:
    """One task-state aggregate per root job per tick. Doubles as the finelog table schema.

    ``oldest_pending_age_ms`` measures the oldest PENDING task from its last
    requeue (or submission for a first attempt); ``oldest_building_age_ms``
    measures the oldest ASSIGNED-or-BUILDING task from its current attempt's
    creation — time since dispatch without reaching RUNNING, the "tasks stuck
    in BUILDING" alert quantity. Both are 0 when no task is in those states.
    Terminal counts cover only root jobs still carrying an active task; a fully
    finished job stops producing rows.
    """

    # Fleet queries slice one root job's history at a time; clustering parquet by
    # root_job_id lets row-group min/max prune the scan.
    key_column: ClassVar[str] = "root_job_id"

    root_job_id: str  # wire job id, or "" for the per-cluster rollup row
    ts: datetime
    pending: int
    assigned: int
    building: int
    running: int
    succeeded: int
    failed: int
    killed: int
    worker_failed: int
    unschedulable: int
    preempted: int
    cosched_failed: int
    oldest_pending_age_ms: int
    oldest_building_age_ms: int


def build_task_state_rows(
    active_rows: list[reads.ActiveTaskRollupRow],
    terminal_rows: list[reads.TerminalTaskCountRow],
    now: Timestamp,
) -> list[IrisTaskState]:
    """Fold the two DB aggregates into per-root-job rows plus the cluster rollup.

    ``active_rows`` carries (root_job_id, state, count, oldest anchor ms) for
    waiting/running tasks; ``terminal_rows`` carries terminal-state counts for
    the same root jobs. The rollup row sums every column and takes the fleet-wide
    oldest anchors, and is emitted even when both inputs are empty.
    """
    now_ms = now.epoch_ms()
    ts = now.as_naive_utc()

    counts: dict[str, dict[int, int]] = {}
    pending_anchor: dict[str, int] = {}
    dispatched_anchor: dict[str, int] = {}

    for row in active_rows:
        counts.setdefault(row.root_job_id, {})[row.state] = row.count
        if row.oldest_anchor_ms is None:
            continue
        if row.state == job_pb2.TASK_STATE_PENDING:
            pending_anchor[row.root_job_id] = row.oldest_anchor_ms
        elif row.state in DISPATCHED_TASK_STATES:
            prev = dispatched_anchor.get(row.root_job_id)
            dispatched_anchor[row.root_job_id] = (
                row.oldest_anchor_ms if prev is None else min(prev, row.oldest_anchor_ms)
            )
    for trow in terminal_rows:
        counts.setdefault(trow.root_job_id, {})[trow.state] = trow.count

    def age_ms(anchor: int | None) -> int:
        if anchor is None:
            return 0
        return max(0, now_ms - anchor)

    def build_row(
        root_job_id: str, by_state: dict[int, int], pending_ms: int | None, dispatched_ms: int | None
    ) -> IrisTaskState:
        return IrisTaskState(
            root_job_id=root_job_id,
            ts=ts,
            pending=by_state.get(job_pb2.TASK_STATE_PENDING, 0),
            assigned=by_state.get(job_pb2.TASK_STATE_ASSIGNED, 0),
            building=by_state.get(job_pb2.TASK_STATE_BUILDING, 0),
            running=by_state.get(job_pb2.TASK_STATE_RUNNING, 0),
            succeeded=by_state.get(job_pb2.TASK_STATE_SUCCEEDED, 0),
            failed=by_state.get(job_pb2.TASK_STATE_FAILED, 0),
            killed=by_state.get(job_pb2.TASK_STATE_KILLED, 0),
            worker_failed=by_state.get(job_pb2.TASK_STATE_WORKER_FAILED, 0),
            unschedulable=by_state.get(job_pb2.TASK_STATE_UNSCHEDULABLE, 0),
            preempted=by_state.get(job_pb2.TASK_STATE_PREEMPTED, 0),
            cosched_failed=by_state.get(job_pb2.TASK_STATE_COSCHED_FAILED, 0),
            oldest_pending_age_ms=age_ms(pending_ms),
            oldest_building_age_ms=age_ms(dispatched_ms),
        )

    rows = [
        build_row(root_job_id, by_state, pending_anchor.get(root_job_id), dispatched_anchor.get(root_job_id))
        for root_job_id, by_state in sorted(counts.items())
    ]

    rollup_counts: dict[int, int] = {}
    for by_state in counts.values():
        for state, count in by_state.items():
            rollup_counts[state] = rollup_counts.get(state, 0) + count
    rows.append(
        build_row(
            CLUSTER_ROLLUP_ROOT_JOB,
            rollup_counts,
            min(pending_anchor.values(), default=None),
            min(dispatched_anchor.values(), default=None),
        )
    )
    return rows


class TaskStateCollector:
    """Periodic emitter of ``iris.task_state`` rows from the controller DB.

    Runs on its own controller thread, off the control tick: each cycle takes a
    read snapshot, aggregates active tasks per root job, and writes the rows.
    A failed cycle logs and skips — the emitter never propagates into the
    controller.
    """

    def __init__(self, db: ControllerDB, table: Table, *, interval: float = TASK_STATE_INTERVAL) -> None:
        self._db = db
        self._table = table
        self._interval = interval

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.collect_once()
            except Exception:
                logger.warning("task-state collect cycle failed", exc_info=True)
            stop_event.wait(timeout=self._interval)

    def collect_once(self, now: Timestamp | None = None) -> None:
        if now is None:
            now = Timestamp.now()
        with self._db.read_snapshot() as tx:
            active = reads.active_task_rollup_by_root_job(tx)
            terminal = reads.terminal_task_counts_by_root_job(tx, sorted({r.root_job_id for r in active}))
        self._table.write(build_task_state_rows(active, terminal, now))
