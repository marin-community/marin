# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``iris.task_state`` emitter: periodic per-root-job task-state rows.

Every :data:`TASK_STATE_INTERVAL` the controller aggregates its waiting/running
local tasks per root job — counts by state, plus how long the oldest PENDING
task has waited for dispatch and how long the oldest dispatched task has sat
without reaching RUNNING — and appends one row per root job with active tasks,
plus one cluster rollup row (``root_job_id=""``). The rollup row is written even
when the cluster is idle, so silence in the table means the controller (or this
emitter) is down.
"""

from finelog.client.log_client import Table
from rigging.timing import Timestamp

from iris.cluster.controller import reads
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.task_state import DISPATCHED_TASK_STATES
from iris.cluster.stats.emitter import PeriodicEmitter
from iris.cluster.stats.tables import CLUSTER_ROLLUP_ROOT_JOB, IrisTaskState
from iris.rpc import job_pb2

# Emission cadence. Coarser than the control tick: these rows feed fleet
# dashboards and stuck-task alerting, where 30s resolution is plenty.
TASK_STATE_INTERVAL = 30.0


def build_task_state_rows(
    active_rows: list[reads.ActiveTaskRollupRow],
    now: Timestamp,
) -> list[IrisTaskState]:
    """Fold the active-task aggregate into per-root-job rows plus the cluster rollup.

    ``active_rows`` carries (root_job_id, state, count, oldest anchor ms) for
    waiting/running tasks. The rollup row sums every column and takes the
    fleet-wide oldest anchors, and is emitted even when the input is empty.
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

    Each cycle takes a read snapshot, aggregates active tasks per root job, and
    writes the rows. Runs on its own :class:`PeriodicEmitter` thread, off the
    control tick.
    """

    def __init__(self, db: ControllerDB, table: Table, *, interval: float = TASK_STATE_INTERVAL) -> None:
        self._db = db
        self._table = table
        self._emitter = PeriodicEmitter(self.collect_once, interval=interval, name="task-state-stats")

    def collect_once(self, now: Timestamp | None = None) -> None:
        if now is None:
            now = Timestamp.now()
        with self._db.read_snapshot() as tx:
            active = reads.active_task_rollup_by_root_job(tx)
        self._table.write(build_task_state_rows(active, now))

    def close(self) -> None:
        self._emitter.close()
