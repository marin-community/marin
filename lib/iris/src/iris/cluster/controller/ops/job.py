# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for jobs: submit, cancel, remove_finished."""

from dataclasses import dataclass

from rigging.timing import Timestamp
from sqlalchemy import Integer, bindparam, cast, func, insert, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.codec import (
    constraints_to_json,
    entrypoint_to_json,
    proto_to_json,
)
from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.reconcile import ReconcileState
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.policy import (
    MAX_REPLICAS_PER_JOB,
)
from iris.cluster.controller.schema import (
    job_workdir_files_table,
    jobs_table,
    users_table,
)
from iris.cluster.types import LOCAL_CLUSTER, TERMINAL_JOB_STATES, JobName
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_from_proto


def _extract_resource_cols(resources: job_pb2.ResourceSpecProto | None) -> tuple[int, int, int, str | None]:
    """Return ``(cpu_millicores, memory_bytes, disk_bytes, device_json)`` columns.

    Missing resources map to zeros and a NULL device json.
    """
    if resources is None:
        return 0, 0, 0, None
    return (
        int(resources.cpu_millicores),
        int(resources.memory_bytes),
        int(resources.disk_bytes),
        proto_to_json(resources.device),
    )


def _materialize_tasks(
    cur: Tx,
    *,
    job_id: JobName,
    num_tasks: int,
    submitted_at_ms: int,
    max_retries_failure: int,
    max_retries_preemption: int,
    priority_root_submitted_ms: int,
    priority_band: int,
) -> None:
    """Insert ``num_tasks`` PENDING task rows for ``job_id``.

    Reserves a contiguous ``priority_insertion`` block so the job's tasks sort together.
    """
    insertion_base = writes.reserve_priority_insertion_base(cur)
    rows = [
        writes.task_row(
            task_id=job_id.task(idx),
            job_id=job_id,
            task_index=idx,
            state=job_pb2.TASK_STATE_PENDING,
            submitted_at_ms=submitted_at_ms,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            priority_neg_depth=-job_id.depth,
            priority_root_submitted_ms=priority_root_submitted_ms,
            priority_insertion=insertion_base + idx,
            priority_band=priority_band,
        )
        for idx in range(num_tasks)
    ]
    writes.bulk_insert_tasks(cur, rows)


@dataclass(frozen=True)
class JobInsertResult:
    """What :func:`insert_job_and_config` computed, for the task-materialization
    and audit steps the caller runs next."""

    replicas: int
    effective_submission_ms: int
    root_submitted_ms: int
    band_sort_key: int
    validation_error: str | None


def submit(
    cur: Tx,
    *,
    job_id: JobName,
    request: controller_pb2.Controller.LaunchJobRequest,
    ts: Timestamp,
) -> None:
    """Insert the job row and expand its tasks. Caller owns the transaction."""
    inserted = insert_job_and_config(cur, job_id=job_id, request=request, ts=ts)
    if inserted.validation_error is None:
        _materialize_tasks(
            cur,
            job_id=job_id,
            num_tasks=inserted.replicas,
            submitted_at_ms=inserted.effective_submission_ms,
            max_retries_failure=int(request.max_retries_failure),
            max_retries_preemption=int(request.max_retries_preemption),
            priority_root_submitted_ms=inserted.root_submitted_ms,
            priority_band=inserted.band_sort_key,
        )
    cur.register(
        lambda: log_event(
            "job_submitted",
            job_id.to_wire(),
            num_tasks=inserted.replicas,
            error=inserted.validation_error,
        )
    )


def insert_job_and_config(
    cur: Tx,
    *,
    job_id: JobName,
    request: controller_pb2.Controller.LaunchJobRequest,
    ts: Timestamp,
    cluster: str = LOCAL_CLUSTER,
) -> JobInsertResult:
    """Insert the ``jobs`` + ``job_config`` (+ workdir file) rows for one job.

    Does NOT materialize tasks — :func:`submit` adds them for a local job; a
    federated handoff (``cluster`` set to a peer) has no local tasks (the peer
    creates them; the sync mirrors them back). Caller owns the transaction.
    """

    submitted_ms = ts.epoch_ms()

    # Derive monotone submission timestamp from MAX(jobs.submitted_at_ms)
    # instead of a separate meta-table key. O(log n) with the existing index.
    # Cast to Integer to bypass TimestampMsType's result-value hook so we get
    # a plain int (needed for arithmetic below).
    last_ms: int = cur.execute(
        select(func.coalesce(func.max(cast(jobs_table.c.submitted_at_ms, Integer)), 0))
    ).scalar_one()
    effective_submission_ms = max(submitted_ms, last_ms + 1)

    parent_job_id = job_id.parent.to_wire() if job_id.parent is not None else None
    root_submitted_ms = effective_submission_ms
    if job_id.parent is not None:
        parent_row = cur.execute(
            select(jobs_table.c.root_submitted_at_ms).where(jobs_table.c.job_id == bindparam("job_id")),
            {"job_id": job_id.parent},
        ).first()
        if parent_row is None:
            raise ValueError(f"Cannot submit job {job_id}: parent {parent_job_id} is absent from the database")
        root_submitted_ms = parent_row.root_submitted_at_ms.epoch_ms()

    deadline_epoch_ms: int | None = None
    if request.HasField("scheduling_timeout") and request.scheduling_timeout.milliseconds > 0:
        deadline_epoch_ms = (
            Timestamp.from_ms(effective_submission_ms).add(duration_from_proto(request.scheduling_timeout)).epoch_ms()
        )

    # Idempotently create a ``users`` row at submission time.
    cur.execute(
        sqlite_insert(users_table)
        .values(
            user_id=job_id.user,
            created_at_ms=Timestamp.from_ms(effective_submission_ms),
            role="user",
        )
        .on_conflict_do_nothing(index_elements=["user_id"])
    )

    requested_band = int(request.priority_band)
    if requested_band != job_pb2.PRIORITY_BAND_UNSPECIFIED:
        band_sort_key = requested_band
    else:
        band_sort_key = job_pb2.PRIORITY_BAND_INTERACTIVE

    replicas = int(request.replicas)
    validation_error: str | None = None
    if replicas < 1:
        validation_error = f"Job {job_id} has invalid replicas={replicas}; must be >= 1"
        replicas = 0
    elif replicas > MAX_REPLICAS_PER_JOB:
        validation_error = f"Job {job_id} replicas={replicas} exceeds max {MAX_REPLICAS_PER_JOB}"
        replicas = 0

    state = job_pb2.JOB_STATE_PENDING if validation_error is None else job_pb2.JOB_STATE_FAILED
    finished_ms = None if validation_error is None else effective_submission_ms

    res = request.resources if request.HasField("resources") else None
    res_cpu, res_mem, res_disk, res_device = _extract_resource_cols(res)
    constraints_json = constraints_to_json(list(request.constraints))
    has_cosched = 1 if request.HasField("coscheduling") else 0
    cosched_group = request.coscheduling.group_by if has_cosched else ""
    sched_timeout: int | None = (
        int(request.scheduling_timeout.milliseconds)
        if request.HasField("scheduling_timeout") and request.scheduling_timeout.milliseconds > 0
        else None
    )
    max_failures = int(request.max_task_failures)
    entrypoint_json = entrypoint_to_json(request.entrypoint)
    environment_json = proto_to_json(request.environment)
    ports_json = list(request.ports)
    timeout_ms: int | None = int(request.timeout.milliseconds) if request.timeout.milliseconds > 0 else None

    job_name_lower = request.name.lower()
    writes.insert_job(
        cur,
        job_id=job_id,
        user_id=job_id.user,
        parent_job_id=parent_job_id,
        root_job_id=job_id.root_job.to_wire(),
        depth=job_id.depth,
        state=state,
        submitted_at_ms=effective_submission_ms,
        root_submitted_at_ms=root_submitted_ms,
        started_at_ms=None,
        finished_at_ms=finished_ms,
        scheduling_deadline_epoch_ms=deadline_epoch_ms,
        error=validation_error,
        exit_code=None,
        num_tasks=replicas,
        name=job_name_lower,
        cluster=cluster,
    )
    writes.insert_job_config(
        cur,
        job_id=job_id,
        name=job_name_lower,
        res_cpu_millicores=res_cpu,
        res_memory_bytes=res_mem,
        res_disk_bytes=res_disk,
        res_device_json=res_device,
        constraints_json=constraints_json,
        has_coscheduling=bool(has_cosched),
        coscheduling_group_by=cosched_group,
        scheduling_timeout_ms=sched_timeout,
        max_task_failures=max_failures,
        entrypoint_json=entrypoint_json,
        environment_json=environment_json,
        bundle_id=request.bundle_id,
        ports_json=ports_json,
        max_retries_failure=int(request.max_retries_failure),
        max_retries_preemption=int(request.max_retries_preemption),
        timeout_ms=timeout_ms,
        preemption_policy=int(request.preemption_policy),
        existing_job_policy=int(request.existing_job_policy),
        priority_band=int(request.priority_band),
        task_image=request.task_image,
        container_profile=int(request.container_profile),
        submit_argv_json=list(request.submit_argv),
        fail_if_exists=bool(request.fail_if_exists),
    )

    workdir_files = dict(request.entrypoint.workdir_files)
    if workdir_files:
        cur.execute(
            insert(job_workdir_files_table),
            [{"job_id": job_id, "filename": name, "data": data} for name, data in workdir_files.items()],
        )

    # A received handoff runs as an ordinary local job, but is recorded as a
    # RECEIVED federated_jobs row (after the jobs row, per the FK) naming the
    # requester, so FederationSync reports it back only to that requester and the
    # changelog events below (and its tasks') resolve their requester from it.
    if request.HasField("federation"):
        writes.insert_received_handle(
            cur,
            job_id=job_id,
            requester_id=request.federation.requester_id,
            owner_principal=request.federation.owner_principal,
        )

    # Record the job-level creation for any requester federating with this peer (a
    # no-op unless this job's root was received via handoff).
    writes.record_federation_change(cur, job_id)

    # Invalidate post-commit so a concurrent reader cannot refill the template
    # cache from the pre-commit snapshot and have that stale value persist past
    # the new row's commit.
    cur.caches[RunTemplatesProjection].invalidate_for_job(cur, job_id)

    return JobInsertResult(
        replicas=replicas,
        effective_submission_ms=effective_submission_ms,
        root_submitted_ms=root_submitted_ms,
        band_sort_key=band_sort_key,
        validation_error=validation_error,
    )


def cancel(
    cur: Tx,
    *,
    job_id: JobName,
    reason: str,
) -> None:
    """Cancel ``job_id`` and its descendant subtree through the kernel.

    Loads a snapshot covering every job in the subtree and all their active
    tasks (so the kernel can fire coscheduled-peer cascades on killed tasks),
    runs :meth:`ReconcileState.cancel_job`, then applies the resulting effects.

    Fixes the latent coscheduled-skip bug in the legacy direct-SQL cancel:
    when one half of an atomic coscheduled group is cancelled, the kernel
    cascades termination to the surviving peers instead of stranding them.
    """
    now = Timestamp.now()
    # The slice closes the full descendant subtree (and every job's tasks /
    # active rows) so the kernel can cascade-kill children and fire
    # coscheduled-peer cascades on killed tasks.
    snapshot = load_closed_snapshot(cur, now=now, seed_job_ids=[job_id])
    if job_id not in snapshot.job_configs:
        return
    # No per-job state preload: the cascade-kill merge guard skips already-
    # terminal rows (excluding WORKER_FAILED, which cancel overwrites).
    effects = ReconcileState.open(snapshot).cancel_job(job_id, reason, now)
    commit_effects(cur, effects)
    # Sweep endpoints that survived because their owning task was already
    # terminal before cancel ran (kernel only emits EndpointDeletion for
    # tasks we actively killed). Derive the same subtree the kernel cancelled
    # from the snapshot's transitive descendants.
    subtree = [job_id, *snapshot.job_descendants[job_id].descendants]
    cur.caches[EndpointsProjection].remove_by_job_ids(cur, subtree)


def purge_job(cur: Tx, job_id: JobName) -> None:
    """Delete a job and drop its derived-count and run-template memos.

    Deletions route through here rather than :func:`writes.delete_job` so a later
    job minted with the same id cannot serve the dead job's cached counts or
    template.
    """
    writes.delete_job(cur, job_id)
    cur.caches[AttemptCountsProjection].invalidate_for_jobs(cur, [job_id])
    cur.caches[RunTemplatesProjection].invalidate_for_job(cur, job_id)


def remove_finished(
    cur: Tx,
    job_id: JobName,
) -> bool:
    """Remove a finished job and its tasks from state.

    Only removes jobs that are in a terminal state. Returns True if removed,
    False if the job does not exist or is not finished.
    """
    job_state = reads.get_job_state(cur, job_id)
    if job_state is None:
        return False
    if job_state not in TERMINAL_JOB_STATES:
        return False
    purge_job(cur, job_id)
    cur.register(
        lambda: log_event(
            "job_removed",
            job_id.to_wire(),
            state=job_state,
        )
    )
    return True
