# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RunTemplatesProjection — lazy per-job cache of RunTaskRequest templates.

The worker-daemon reconcile path reads a job's ``RunTaskRequest`` template
(carrying the immutable per-job fields: entrypoint, environment, resources,
constraints) on every tick for each ASSIGNED task. Re-building the template
from SQL on every read is wasteful for jobs with many in-flight tasks.  This
projection memoizes the per-job :class:`~iris.rpc.job_pb2.RunTaskRequest`, filled
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
from iris.cluster.controller.codec import constraints_from_json, proto_from_json, resource_spec_from_scalars
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.lru_cache import LRUCache
from iris.cluster.controller.projections.base import LazyFillGuard, Projection
from iris.cluster.controller.schema import job_config_table, job_workdir_files_table
from iris.cluster.types import JobName
from iris.rpc import job_pb2

# Size matches the worst-case concurrent job count expected in a single
# controller process.
RUN_REQUEST_TEMPLATE_CACHE_SIZE = 4096


def build_run_request_fields(
    *,
    num_tasks: int,
    entrypoint_json: str,
    environment_json: str,
    bundle_id: str,
    resources: job_pb2.ResourceSpecProto,
    ports_json: list,
    constraints_json: str | None,
    task_image: str,
    task_id: str = "",
    attempt_id: int = 0,
    priority: int = 0,
    container_profile: int = 0,
) -> job_pb2.RunTaskRequest:
    """Build a RunTaskRequest carrying the per-job fields shared by the template
    and per-attempt construction paths.

    The template path leaves ``task_id``/``attempt_id``/``priority`` at their
    proto defaults; the per-attempt path stamps them. proto_from_json returns
    shared cached instances — set via constructor kwarg so RunTaskRequest
    copies them; callers then mutate the copy's workdir_files (never the cached
    source).
    """
    return job_pb2.RunTaskRequest(
        num_tasks=num_tasks,
        entrypoint=proto_from_json(entrypoint_json, job_pb2.RuntimeEntrypoint),
        environment=proto_from_json(environment_json, job_pb2.EnvironmentConfig),
        bundle_id=bundle_id,
        resources=resources,
        ports=ports_json,
        constraints=[c.to_proto() for c in constraints_from_json(constraints_json)],
        task_image=task_image,
        task_id=task_id,
        attempt_id=attempt_id,
        priority=priority,
        container_profile=container_profile,
    )


class RunTemplatesProjection(Projection):
    """Process-local cache of ``{wire_job_id: RunTaskRequest}`` templates."""

    watches: ClassVar = (job_config_table, job_workdir_files_table)

    def __init__(self, db: ControllerDB) -> None:
        self._lock = threading.Lock()
        self._cache: LRUCache[str, job_pb2.RunTaskRequest] = LRUCache(RUN_REQUEST_TEMPLATE_CACHE_SIZE)
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
            self._cache = LRUCache(RUN_REQUEST_TEMPLATE_CACHE_SIZE)
            self._guard.reset(self._db.commit_seq)

    def get(self, tx: Tx, job_id: JobName) -> job_pb2.RunTaskRequest | None:
        """Return a cached per-job ``RunTaskRequest`` template, filling from ``tx`` on miss.

        Per-attempt fields (``task_id``, ``attempt_id``) are left at proto defaults;
        callers stamp them onto a copy at fan-out time. Returns ``None`` when the job
        has no config row.

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

        resources = resource_spec_from_scalars(
            job.res_cpu_millicores,
            job.res_memory_bytes,
            job.res_disk_bytes,
            job.res_device_json,
        )
        template = build_run_request_fields(
            num_tasks=job.num_tasks,
            entrypoint_json=job.entrypoint_json,
            environment_json=job.environment_json,
            bundle_id=job.bundle_id,
            resources=resources,
            ports_json=job.ports_json,
            constraints_json=job.constraints_json,
            task_image=job.task_image,
            container_profile=job.container_profile,
        )
        for filename, data in reads.get_workdir_files(tx, job_id).items():
            template.entrypoint.workdir_files[filename] = data

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
