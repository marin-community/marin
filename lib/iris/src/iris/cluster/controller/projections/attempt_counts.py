# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AttemptCountsProjection — DB-scoped memo of per-job derived retry counts.

The job-list dashboard reads a job's failure/preemption totals on every refresh.
Those totals are derived from ``task_attempts`` (see
:mod:`iris.cluster.controller.attempt_counts`), and re-aggregating the attempt
log on every read is wasteful for the many terminal jobs whose attempts never
change again. This projection memoizes the per-job :class:`AttemptCounts`,
computed lazily via :func:`reads.attempt_counts_for_jobs` and dropped after any
commit that touches a job's attempts.

Unlike :class:`WorkerAttrsProjection` / :class:`EndpointsProjection`, this
projection owns no table and is not threaded to its writers. ``task_attempts`` is
written from several call sites (reconcile commit, federation mirror) that do not
hold the cache. Instead it is owned by
:class:`~iris.cluster.controller.scope.ControllerScope` and exposed on every
:class:`~iris.cluster.controller.scope.ScopedTx` cursor the scope hands out — so a
write chokepoint invalidates via ``cur.attempt_counts.invalidate_for_tasks(cur,
task_ids)`` and the service reads via ``cur.attempt_counts.get_jobs(cur,
job_ids)``, all without threading a cache reference or a global registry. New
attempts (ASSIGNED, no ``started_at``) do not change the counts, but they funnel
through the same chokepoints, so over-invalidating is harmless.

Consistency: entries are dropped by a post-commit hook that fires under the DB
write lock (via ``tx.register``), so a reader either sees the pre-commit memo or
recomputes from the post-commit attempt rows — never a torn value. Callers that
must read counts in lockstep with other rows of the *same* DB snapshot (the
reconcile loader, the ``list_jobs`` SQL sort) derive via SQL directly and do not
consult this cache.
"""

import logging
import threading
from collections.abc import Iterable, Mapping
from typing import ClassVar

from iris.cluster.controller import reads
from iris.cluster.controller.attempt_counts import AttemptCounts
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.types import JobName

logger = logging.getLogger(__name__)


class AttemptCountsProjection:
    """Process-local memo of ``{job_id: AttemptCounts}`` derived from attempts."""

    # A deleted job is never invalidated (its rows are gone and it is never queried
    # again), so entries for pruned jobs would otherwise accumulate. Clear the memo
    # once it grows past this bound; warm entries simply re-derive on next read.
    _MAX_ENTRIES: ClassVar[int] = 100_000

    def __init__(self, db: ControllerDB) -> None:
        self._lock = threading.Lock()
        self._by_job: dict[JobName, AttemptCounts] = {}
        # Lazy fill: start empty and populate on demand. A checkpoint restore
        # swaps the DB file, so drop everything and refill against the new rows.
        db.register_reopen_hook(self.clear)

    def clear(self) -> None:
        with self._lock:
            self._by_job.clear()

    def get_jobs(self, tx: Tx, job_ids: Iterable[JobName]) -> Mapping[JobName, AttemptCounts]:
        """Return ``{job_id: AttemptCounts}`` for ``job_ids``, filling misses from ``tx``.

        Jobs with no attempt rows resolve to a zero :class:`AttemptCounts`. The
        computed values for missing jobs are memoized; ``tx`` supplies the read
        snapshot for the aggregate.
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
            self._by_job.update(filled)
        return {**hits, **filled}

    def invalidate_for_tasks(self, tx: Tx, task_ids: Iterable[JobName]) -> None:
        """Drop the memo for the jobs owning ``task_ids`` after ``tx`` commits."""
        self.invalidate_for_jobs(tx, jobs_of_tasks(task_ids))

    def invalidate_for_jobs(self, tx: Tx, job_ids: Iterable[JobName]) -> None:
        """Drop the memo for ``job_ids`` after ``tx`` commits.

        Used both when a job's attempts change and when a job is deleted (so a
        later job of the same id cannot serve the dead job's counts). The pop is
        deferred to a post-commit hook (fires under the write lock): an eager pop
        would let a concurrent reader refill the entry from the pre-commit snapshot
        and go stale the moment the write lands.
        """
        jobs = {jid for jid in job_ids if jid is not None}
        if not jobs:
            return

        def apply() -> None:
            with self._lock:
                for jid in jobs:
                    self._by_job.pop(jid, None)

        tx.register(apply)


def jobs_of_tasks(task_ids: Iterable[JobName]) -> set[JobName]:
    """Map task ids to their owning job ids (a task id's parent)."""
    jobs: set[JobName] = set()
    for task_id in task_ids:
        parent = task_id.parent
        if parent is not None:
            jobs.add(parent)
    return jobs
