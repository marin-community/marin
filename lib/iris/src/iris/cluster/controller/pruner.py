# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Background pruning loop for the controller.

``prune_old_data`` is an incremental maintenance loop that deletes one
prunable job (or worker) per transaction, sleeping between iterations so
scheduling and heartbeat traffic can interleave. The actual row-level
deletes live in :mod:`iris.cluster.controller.writes`; this module owns
only the loop structure that drives them.
"""

import logging
import threading
import time
from dataclasses import dataclass

from rigging.timing import Duration, Timestamp
from sqlalchemy import bindparam, select

from iris.cluster.controller import writes
from iris.cluster.controller.audit import log_event
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.schema import jobs_table
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import TERMINAL_JOB_STATES, WorkerId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PruneResult:
    """Counts of rows deleted by prune_old_data."""

    jobs_deleted: int = 0
    workers_deleted: int = 0

    @property
    def total(self) -> int:
        return self.jobs_deleted + self.workers_deleted


def _find_prunable_worker(health: WorkerHealthTracker, before_ms: int) -> WorkerId | None:
    """Return one tracker-known worker that is unhealthy/inactive with a stale heartbeat.

    Every persisted ``workers`` row has a tracker entry by construction
    (seeded at boot/restore, registered on commit of ``upsert``, removed
    on commit of ``remove``), so scanning the tracker is sufficient.
    """
    for worker_id, liveness in health.all().items():
        if (not liveness.healthy or not liveness.active) and liveness.last_heartbeat_ms < before_ms:
            return worker_id
    return None


def prune_old_data(
    db: ControllerDB,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
    worker_attrs: WorkerAttrsProjection,
    *,
    job_retention: Duration,
    worker_retention: Duration,
    stop_event: threading.Event | None = None,
    pause_between_s: float = 1.0,
) -> PruneResult:
    """Incrementally delete old data, one row per transaction.

    Designed to run on a background thread. Each deletion holds the write
    lock for only one CASCADE delete (one job or one worker), then sleeps
    to let scheduling and heartbeats proceed.

    Args:
        db: Controller database handle.
        health: Worker health tracker used to find stale workers.
        endpoints: Endpoints projection invalidated before each job CASCADE.
        worker_attrs: Worker attributes projection invalidated on worker removal.
        job_retention: Delete terminal jobs whose finished_at is older than this.
        worker_retention: Delete inactive/unhealthy workers whose last heartbeat is older than this.
        stop_event: If set, abort early (e.g. during shutdown).
        pause_between_s: Sleep between individual deletes to reduce lock contention.
    """
    now_ms = Timestamp.now().epoch_ms()
    job_cutoff_ms = now_ms - job_retention.to_ms()
    worker_cutoff_ms = now_ms - worker_retention.to_ms()

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    # 1. Jobs: one at a time (CASCADE to tasks → attempts, endpoints)
    jobs_deleted = 0
    while not _stopped():
        with db.read_snapshot() as snap:
            _prunable_row = snap.execute(
                select(jobs_table.c.job_id)
                .where(
                    jobs_table.c.state.in_(bindparam("terminal_states", expanding=True)),
                    jobs_table.c.finished_at_ms.is_not(None),
                    jobs_table.c.finished_at_ms < bindparam("before_ts"),
                )
                .limit(1),
                {"terminal_states": list(TERMINAL_JOB_STATES), "before_ts": Timestamp.from_ms(job_cutoff_ms)},
            ).first()
            job_name = _prunable_row.job_id if _prunable_row is not None else None
        if job_name is None:
            break
        with db.transaction() as cur:
            # Invalidate endpoint cache BEFORE the CASCADE so the cache
            # drops rows SQLite is about to delete for us.
            endpoints.remove_by_job_ids(cur, [job_name])
            writes.delete_job(cur, job_name)
        log_event("job_pruned", job_name.to_wire())
        jobs_deleted += 1
        time.sleep(pause_between_s)

    # 2. Workers: one at a time (CASCADE to attributes).
    workers_deleted = 0
    while not _stopped():
        worker_id = _find_prunable_worker(health, worker_cutoff_ms)
        if worker_id is None:
            break
        with db.transaction() as cur:
            writes.remove_worker(cur, worker_id, health=health, worker_attrs=worker_attrs)
        log_event("worker_pruned", str(worker_id))
        workers_deleted += 1
        time.sleep(pause_between_s)

    result = PruneResult(
        jobs_deleted=jobs_deleted,
        workers_deleted=workers_deleted,
    )
    if result.total > 0:
        logger.info(
            "Pruned old data: %d jobs, %d workers",
            result.jobs_deleted,
            result.workers_deleted,
        )
        db.optimize()

    return result
