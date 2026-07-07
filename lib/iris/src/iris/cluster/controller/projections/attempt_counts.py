# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AttemptCountsProjection — lazy per-job memo of derived retry counts.

The job-list dashboard reads a job's failure/preemption totals on every refresh.
Those totals are derived from ``task_attempts`` (see
:mod:`iris.cluster.controller.attempt_counts`), and re-aggregating the attempt log
on every read is wasteful for the many terminal jobs whose attempts never change
again. This projection memoizes the per-job :class:`AttemptCounts`, computed lazily
via :func:`reads.attempt_counts_for_jobs` and dropped after any commit that touches
a job's attempts — the ``watches``/invalidate-and-recompute half of the
:class:`~iris.cluster.controller.projections.base.Projection` concept.

It ``watches`` ``task_attempts`` rather than owning it: the table is written from
several sinks (reconcile commit, federation mirror) that hold no reference to this
projection. They reach it by type through their cursor —
``cur.caches[AttemptCountsProjection].invalidate_for_tasks(cur, task_ids)`` — and
the service reads via ``q.caches[AttemptCountsProjection].get_jobs(q, job_ids)``.
New attempts (ASSIGNED, no ``started_at``) do not change the counts but funnel
through the same chokepoints, so over-invalidating is harmless.

Consistency: entries drop via a post-commit hook that fires under the write lock,
and a :class:`~iris.cluster.controller.projections.base.LazyFillGuard` rejects a
fill recomputed from a snapshot older than the key's most recent invalidation — so
a reader can neither read a torn value nor write back a stale one (the stale set).
Callers that must read counts in lockstep with other rows of the *same* snapshot
(the reconcile loader, the ``list_jobs`` SQL sort) derive via SQL directly and do
not consult this projection.
"""

import logging
import threading
from collections.abc import Iterable, Mapping
from typing import ClassVar

from iris.cluster.controller import reads
from iris.cluster.controller.attempt_counts import AttemptCounts
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.base import LazyFillGuard, Projection
from iris.cluster.controller.schema import task_attempts_table
from iris.cluster.types import JobName

logger = logging.getLogger(__name__)


class AttemptCountsProjection(Projection):
    """Process-local memo of ``{job_id: AttemptCounts}`` derived from attempts."""

    watches: ClassVar = (task_attempts_table,)

    # A deleted job is never invalidated (its rows are gone and it is never queried
    # again), so entries for pruned jobs would otherwise accumulate. Clear the memo
    # once it grows past this bound; warm entries simply re-derive on next read.
    _MAX_ENTRIES: ClassVar[int] = 100_000

    def __init__(self, db: ControllerDB) -> None:
        self._lock = threading.Lock()
        self._by_job: dict[JobName, AttemptCounts] = {}
        self._guard: LazyFillGuard[JobName] = LazyFillGuard()
        # super().__init__ registers into db.caches, calls rehydrate() (which needs
        # the fields above), and wires the reopen hook.
        super().__init__(db)

    def rehydrate(self) -> None:
        """Drop the memo and reset the guard's floor to the current commit seq.

        Lazy start: begin empty and refill on demand. A checkpoint restore swaps
        the DB file, so this both clears stale entries and — via the guard floor —
        refuses any fill still in flight against the old file.
        """
        with self._lock:
            self._by_job.clear()
            self._guard.reset(self._db.commit_seq)

    def get_jobs(self, tx: Tx, job_ids: Iterable[JobName]) -> Mapping[JobName, AttemptCounts]:
        """Return ``{job_id: AttemptCounts}`` for ``job_ids``, filling misses from ``tx``.

        Jobs with no attempt rows resolve to a zero :class:`AttemptCounts`. A
        computed value is cached only if the guard confirms ``tx``'s snapshot is at
        least as fresh as the job's most recent invalidation; otherwise it is
        returned to the caller but not stored.
        """
        ids = list(job_ids)
        if not ids:
            return {}
        with self._lock:
            hits = {jid: self._by_job[jid] for jid in ids if jid in self._by_job}
        missing = [jid for jid in ids if jid not in hits]
        if not missing:
            return hits
        computed = reads.attempt_counts_for_jobs(tx, missing)
        filled = {jid: computed.get(jid, AttemptCounts()) for jid in missing}
        with self._lock:
            if len(self._by_job) + len(filled) > self._MAX_ENTRIES:
                self._by_job.clear()
                self._guard.reset(self._db.commit_seq)
            for jid, val in filled.items():
                if self._guard.may_store(tx.seq, jid):
                    self._by_job[jid] = val
        return {**hits, **filled}

    def invalidate_for_tasks(self, tx: Tx, task_ids: Iterable[JobName]) -> None:
        """Drop the memo for the jobs owning ``task_ids`` after ``tx`` commits."""
        self.invalidate_for_jobs(tx, jobs_of_tasks(task_ids))

    def invalidate_for_jobs(self, tx: Tx, job_ids: Iterable[JobName]) -> None:
        """Drop the memo for ``job_ids`` after ``tx`` commits.

        Used both when a job's attempts change and when a job is deleted (so a later
        job of the same id cannot serve the dead job's counts).
        """
        jobs = {jid for jid in job_ids if jid is not None}
        if not jobs:
            return

        def apply() -> None:
            # Fires post-commit under the write lock, so commit_seq is already
            # ticked: stamp the guard with it and drop the entries together. An
            # eager pop would let a concurrent reader refill from the pre-commit
            # snapshot; the deferred pop + guard stamp reject that refill instead.
            with self._lock:
                for jid in jobs:
                    self._by_job.pop(jid, None)
                self._guard.note_invalidated(self._db.commit_seq, jobs)

        tx.register(apply)


def jobs_of_tasks(task_ids: Iterable[JobName]) -> set[JobName]:
    """Map task ids to their owning job ids (a task id's parent)."""
    jobs: set[JobName] = set()
    for task_id in task_ids:
        parent = task_id.parent
        if parent is not None:
            jobs.add(parent)
    return jobs
