# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Background pruning loop for the controller.

``prune_old_data`` is an incremental maintenance loop that deletes one
prunable job, worker, or slice per transaction, sleeping between iterations
so scheduling and heartbeat traffic can interleave. The actual row-level
deletes live in :mod:`iris.cluster.controller.writes`; this module owns
only the loop structure that drives them.

Slice pruning is the garbage collector for *abandoned* scaling groups —
see :func:`_prune_orphan_slices`.
"""

import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

from rigging.timing import Duration, Timestamp

from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.backend import TaskBackend
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.types import TERMINAL_JOB_STATES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PruneResult:
    """Counts of rows deleted by prune_old_data."""

    jobs_deleted: int = 0
    workers_deleted: int = 0
    slices_deleted: int = 0
    endpoints_deleted: int = 0

    @property
    def total(self) -> int:
        return self.jobs_deleted + self.workers_deleted + self.slices_deleted + self.endpoints_deleted


def _stopped(stop_event: threading.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _prune_terminal_jobs(
    db: ControllerDB,
    cutoff_ms: int,
    stop_event: threading.Event | None,
    pause: float,
) -> int:
    """Delete terminal jobs finished before ``cutoff_ms``, one CASCADE (tasks → attempts) at a time."""
    deleted = 0
    while not _stopped(stop_event):
        with db.read_snapshot() as snap:
            job_name = reads.find_prunable_job(snap, TERMINAL_JOB_STATES, Timestamp.from_ms(cutoff_ms))
        if job_name is None:
            break
        with db.transaction() as cur:
            # Invalidate endpoint cache BEFORE the CASCADE so the cache
            # drops rows SQLite is about to delete for us. delete_job then drops
            # the derived-count and run-template memos its CASCADE would strand.
            cur.caches[EndpointsProjection].remove_by_job_ids(cur, [job_name])
            writes.delete_job(cur, job_name)
        log_event("job_pruned", job_name.to_wire())
        deleted += 1
        time.sleep(pause)
    return deleted


def _prune_dead_workers(
    backends: Iterable[TaskBackend], cutoff_ms: int, stop_event: threading.Event | None, pause: float
) -> int:
    """Delete each backend's DEAD workers whose last heartbeat predates ``cutoff_ms``.

    Each backend garbage-collects the dead workers in its own tracker (the trackers
    are disjoint by scale group); a backend that tracks no Iris workers prunes
    nothing. The per-worker delete cadence (one CASCADE per transaction, ``pause``
    between deletes, early stop on ``stop_event``) lives in the backend.
    """
    return sum(
        backend.prune_dead_workers(cutoff_ms=cutoff_ms, stop_event=stop_event, pause=pause) for backend in backends
    )


def _prune_orphan_slices(db: ControllerDB, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
    """Garbage-collect slice rows left behind by abandoned scaling groups.

    A ``slices`` row is orphaned once no ``workers`` row references it: nothing
    live sits behind it, so after it ages past ``slice_retention`` the row is
    pure garbage. These accumulate when a scale group is dropped from config
    *after* its VMs are already gone — neither owner of the table reaps them.
    The autoscaler's state mirror (``persist_autoscaler_state``) only deletes
    rows for groups it still tracks, and drain-mode
    (``recovery.restore_autoscaler_state``) only re-adopts a retired group while
    its VMs are still alive. An abandoned group is in neither set, so this loop
    is the only thing that clears its rows.
    """
    deleted = 0
    while not _stopped(stop_event):
        with db.read_snapshot() as snap:
            slice_id = reads.find_prunable_slice(snap, cutoff_ms)
        if slice_id is None:
            break
        with db.transaction() as cur:
            writes.delete_slice(cur, slice_id)
        log_event("slice_pruned", slice_id)
        deleted += 1
        time.sleep(pause)
    return deleted


def _sweep_expired_endpoints(db: ControllerDB, now: Timestamp) -> int:
    """Delete endpoints whose lease has expired. Reads already hide them; this
    reclaims storage so the lease — not the FK CASCADE — is the GC trigger."""
    with db.transaction() as cur:
        removed = cur.caches[EndpointsProjection].sweep_expired(cur, now)
    for endpoint_id in removed:
        log_event("endpoint_lease_expired", endpoint_id)
    return len(removed)


def prune_old_data(
    db: ControllerDB,
    backends: Iterable[TaskBackend],
    *,
    job_retention: Duration,
    worker_retention: Duration,
    slice_retention: Duration,
    stop_event: threading.Event | None = None,
    pause_between_s: float = 1.0,
) -> PruneResult:
    """Incrementally delete old data, one row per transaction.

    Designed to run on a background thread. Each deletion holds the write lock
    for only one delete (one job, worker, or slice), then sleeps to let
    scheduling and heartbeats proceed.

    Args:
        db: Controller DB — job pruning routes ``delete_job`` through a write cursor so
            each CASCADE also drops the job's derived-count memo (reached via
            ``cur.caches``); slice/endpoint prunes use the same handle and reach
            ``EndpointsProjection`` the same way.
        backends: The backends, each of which garbage-collects its own dead workers.
        job_retention: Delete terminal jobs whose finished_at is older than this.
        worker_retention: Delete inactive/unhealthy workers whose last heartbeat is older than this.
        slice_retention: Delete orphaned slices from abandoned scale groups (no backing worker row) older than this.
        stop_event: If set, abort early (e.g. during shutdown).
        pause_between_s: Sleep between individual deletes to reduce lock contention.
    """
    now = Timestamp.now()
    now_ms = now.epoch_ms()
    result = PruneResult(
        jobs_deleted=_prune_terminal_jobs(db, now_ms - job_retention.to_ms(), stop_event, pause_between_s),
        workers_deleted=_prune_dead_workers(backends, now_ms - worker_retention.to_ms(), stop_event, pause_between_s),
        slices_deleted=_prune_orphan_slices(db, now_ms - slice_retention.to_ms(), stop_event, pause_between_s),
        endpoints_deleted=_sweep_expired_endpoints(db, now),
    )
    if result.total > 0:
        logger.info(
            "Pruned old data: %d jobs, %d workers, %d slices, %d endpoints",
            result.jobs_deleted,
            result.workers_deleted,
            result.slices_deleted,
            result.endpoints_deleted,
        )
        db.optimize()

    return result
