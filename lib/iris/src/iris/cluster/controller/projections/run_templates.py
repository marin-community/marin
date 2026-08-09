# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RunTemplatesProjection — lazy per-Job cache of native Attempt launch templates.

The worker-daemon reconcile path reads a Job's ``AttemptLaunchTemplate``
(carrying the immutable per-job fields: entrypoint, environment, resources,
constraints) on every tick for each ASSIGNED task. Re-building the template
from SQL on every read is wasteful for jobs with many in-flight tasks.  This
projection memoizes the per-Job native template, filled
lazily on the first read and invalidated after any commit that rewrites the job's
config — the ``watches``/invalidate-and-recompute half of the
:class:`~iris.cluster.controller.projections.base.Projection` concept.

It ``watches`` ``job_config`` and ``job_workdir_files``: both tables are written
during ``insert_job_and_config`` (the submit and handoff path), and a resubmit
with the same ``JobName`` writes a fresh row.  Callers reach the invalidation
point via ``cur.caches[RunTemplatesProjection].invalidate_for_job(cur, job_id)``
immediately after the write.

Consistency: entries drop via a post-commit hook that fires under the write lock,
and a :class:`~iris.cluster.controller.projections.base.LazyFillGuard` rejects a
fill recomputed from a snapshot older than the key's most recent invalidation —
so a reader cannot write back a stale value (the stale set) even if the
invalidation commit lands between the reader's snapshot open and its cache store.
"""

import threading
from typing import ClassVar

from iris.cluster.controller import reads
from iris.cluster.controller.codec import reconstruct_job_spec
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.lru_cache import LRUCache
from iris.cluster.controller.projections.base import LazyFillGuard, Projection
from iris.cluster.controller.schema import job_config_table, job_workdir_files_table
from iris.cluster.resources.attempt import AttemptLaunchTemplate
from iris.cluster.resources.job import JobSpec
from iris.cluster.types import JobName

# Size matches the worst-case concurrent job count expected in a single
# controller process.
ATTEMPT_LAUNCH_TEMPLATE_CACHE_SIZE = 4096


def attempt_launch_template_from_job_spec(spec: JobSpec) -> AttemptLaunchTemplate:
    """Project the immutable execution fields shared by a Job's Attempts."""
    return AttemptLaunchTemplate(
        num_tasks=spec.replicas,
        entrypoint=spec.entrypoint,
        environment=spec.environment,
        bundle_id=spec.bundle_id,
        resources=spec.resources,
        timeout=spec.timeout,
        ports=spec.ports,
        constraints=spec.constraints,
        task_image=spec.task_image,
        coscheduling=spec.coscheduling,
        priority_band=spec.priority_band,
        container_profile=spec.container_profile,
    )


class RunTemplatesProjection(Projection):
    """Process-local cache of ``{wire_job_id: AttemptLaunchTemplate}``."""

    watches: ClassVar = (job_config_table, job_workdir_files_table)

    def __init__(self, db: ControllerDB) -> None:
        self._lock = threading.Lock()
        self._cache: LRUCache[str, AttemptLaunchTemplate] = LRUCache(ATTEMPT_LAUNCH_TEMPLATE_CACHE_SIZE)
        self._guard: LazyFillGuard[str] = LazyFillGuard()
        # super().__init__ registers into db.caches, calls rehydrate() (which needs
        # the fields above), and wires the reopen hook.
        super().__init__(db)

    def rehydrate(self) -> None:
        """Drop the cache and reset the guard's floor to the current commit seq.

        Lazy start: begin empty and refill on demand. A checkpoint restore swaps
        the DB file, so this both clears stale entries and — via the guard floor —
        refuses any fill still in flight against the old file.
        """
        with self._lock:
            self._cache = LRUCache(ATTEMPT_LAUNCH_TEMPLATE_CACHE_SIZE)
            self._guard.reset(self._db.commit_seq)

    def get(self, tx: Tx, job_id: JobName) -> AttemptLaunchTemplate | None:
        """Return a cached per-Job launch template, filling from ``tx`` on miss.

        Returns ``None`` when the Job has no config row.

        A computed value is cached only if the guard confirms ``tx``'s snapshot is at
        least as fresh as the job's most recent invalidation; otherwise it is returned
        to the caller but not stored.
        """
        wire = job_id.to_wire()
        with self._lock:
            cached = self._cache.get(wire)
        if cached is not None:
            return cached

        job = reads.get_job_detail(tx, job_id)
        if job is None:
            return None

        spec = reconstruct_job_spec(
            job,
            workdir_files=reads.get_workdir_files(tx, job_id),
        )
        template = attempt_launch_template_from_job_spec(spec)

        with self._lock:
            if self._guard.may_store(tx.seq, wire):
                # put interns: returns the already-cached instance if one was
                # concurrently inserted, otherwise returns ``template``.
                return self._cache.put(wire, template)
        return template

    def invalidate_for_job(self, tx: Tx, job_id: JobName) -> None:
        """Drop the cached template for ``job_id`` after ``tx`` commits."""
        wire = job_id.to_wire()

        def apply() -> None:
            # Fires post-commit under the write lock (commit_seq already ticked):
            # dropping and stamping the guard here rather than eagerly pre-commit
            # stops a concurrent reader refilling from the pre-commit snapshot.
            with self._lock:
                self._cache.pop(wire)
                self._guard.note_invalidated(self._db.commit_seq, [wire])

        tx.register(apply)
