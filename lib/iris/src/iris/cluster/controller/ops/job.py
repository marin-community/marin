# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for jobs: submit, cancel, remove_finished."""

from rigging.timing import Timestamp
from sqlalchemy import Integer, bindparam, cast, func, insert, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.constraints import Constraint, constraints_from_resources, merge_constraints
from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit import log_event
from iris.cluster.controller.codec import (
    constraints_to_json,
    entrypoint_to_json,
    proto_to_json,
    reservation_to_json,
)
from iris.cluster.controller.db import Tx
from iris.cluster.controller.direct_provider import RunTemplateCache
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.reconcile.batches import apply_cancel_job_batch
from iris.cluster.controller.reconcile.effects import apply_effects
from iris.cluster.controller.reconcile.loader import load_jobs_slice
from iris.cluster.controller.reconcile.policy import (
    DEFAULT_MAX_RETRIES_PREEMPTION,
    MAX_REPLICAS_PER_JOB,
    RESERVATION_HOLDER_JOB_NAME,
)
from iris.cluster.controller.schema import (
    job_workdir_files_table,
    jobs_table,
    users_table,
)
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import TERMINAL_JOB_STATES, JobName
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_from_proto


def request_has_reservation(request: controller_pb2.Controller.LaunchJobRequest) -> bool:
    """Return True if the request carries reservation entries, else False."""
    return request.HasField("reservation") and bool(request.reservation.entries)


def submit(
    cur: Tx,
    *,
    job_id: JobName,
    request: controller_pb2.Controller.LaunchJobRequest,
    ts: Timestamp,
    run_template_cache: RunTemplateCache,
) -> None:
    """Insert the job row and expand its tasks. Caller owns the transaction."""
    # Same-name replacement reuses ``job_id``; drop any stale cached
    # template before the new row's fields land in the DB.
    run_template_cache.pop(job_id.to_wire())

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
        _parent_row = cur.execute(
            select(jobs_table.c.root_submitted_at_ms).where(jobs_table.c.job_id == bindparam("job_id")),
            {"job_id": job_id.parent},
        ).first()
        if _parent_row is None:
            raise ValueError(f"Cannot submit job {job_id}: parent {parent_job_id} is absent from the database")
        root_submitted_ms = _parent_row.root_submitted_at_ms.epoch_ms()

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
    has_reservation = request_has_reservation(request)

    res = request.resources if request.HasField("resources") else None
    res_cpu = int(res.cpu_millicores) if res else 0
    res_mem = int(res.memory_bytes) if res else 0
    res_disk = int(res.disk_bytes) if res else 0
    res_device = proto_to_json(res.device) if res else None
    constraints_json = constraints_to_json(request.constraints)
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
    reservation_json = reservation_to_json(request)
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
        is_reservation_holder=False,
        name=job_name_lower,
        has_reservation=has_reservation,
    )
    writes.insert_job_config(
        cur,
        job_id=job_id,
        name=job_name_lower,
        has_reservation=has_reservation,
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
        submit_argv_json=list(request.submit_argv),
        reservation_json=reservation_json,
        fail_if_exists=bool(request.fail_if_exists),
    )

    _workdir_files = dict(request.entrypoint.workdir_files)
    if _workdir_files:
        cur.execute(
            insert(job_workdir_files_table),
            [{"job_id": job_id, "filename": name, "data": data} for name, data in _workdir_files.items()],
        )

    if validation_error is None:
        insertion_base = writes.reserve_priority_insertion_base(cur)
        replica_rows: list[dict] = []
        for idx in range(replicas):
            task_id = job_id.task(idx)
            replica_rows.append(
                writes.task_row(
                    task_id=task_id,
                    job_id=job_id,
                    task_index=idx,
                    state=job_pb2.TASK_STATE_PENDING,
                    submitted_at_ms=effective_submission_ms,
                    max_retries_failure=int(request.max_retries_failure),
                    max_retries_preemption=int(request.max_retries_preemption),
                    priority_neg_depth=-job_id.depth,
                    priority_root_submitted_ms=root_submitted_ms,
                    priority_insertion=insertion_base + idx,
                    priority_band=band_sort_key,
                )
            )
        writes.bulk_insert_tasks(cur, replica_rows)
        if request.HasField("reservation") and request.reservation.entries:
            holder_id = job_id.child(RESERVATION_HOLDER_JOB_NAME)
            entry = request.reservation.entries[0]
            holder_request = controller_pb2.Controller.LaunchJobRequest(
                name=holder_id.to_wire(),
                entrypoint=request.entrypoint,
                resources=entry.resources,
                environment=request.environment,
                replicas=len(request.reservation.entries),
                max_retries_preemption=DEFAULT_MAX_RETRIES_PREEMPTION,
            )
            merged = merge_constraints(
                constraints_from_resources(entry.resources),
                [Constraint.from_proto(c) for c in entry.constraints or request.constraints],
            )
            for constraint in merged:
                holder_request.constraints.append(constraint.to_proto())
            holder_res = holder_request.resources if holder_request.HasField("resources") else None
            holder_res_cpu = int(holder_res.cpu_millicores) if holder_res else 0
            holder_res_mem = int(holder_res.memory_bytes) if holder_res else 0
            holder_res_disk = int(holder_res.disk_bytes) if holder_res else 0
            holder_res_device = proto_to_json(holder_res.device) if holder_res else None
            holder_constraints_json = constraints_to_json(holder_request.constraints)
            holder_name_lower = holder_request.name.lower()
            writes.insert_job(
                cur,
                job_id=holder_id,
                user_id=holder_id.user,
                parent_job_id=job_id.to_wire(),
                root_job_id=holder_id.root_job.to_wire(),
                depth=holder_id.depth,
                state=job_pb2.JOB_STATE_PENDING,
                submitted_at_ms=effective_submission_ms,
                root_submitted_at_ms=root_submitted_ms,
                started_at_ms=None,
                finished_at_ms=None,
                scheduling_deadline_epoch_ms=None,
                error=None,
                exit_code=None,
                num_tasks=len(request.reservation.entries),
                is_reservation_holder=True,
                name=holder_name_lower,
                has_reservation=False,
            )
            holder_entrypoint_json = entrypoint_to_json(holder_request.entrypoint)
            holder_environment_json = proto_to_json(holder_request.environment)
            writes.insert_job_config(
                cur,
                job_id=holder_id,
                name=holder_name_lower,
                has_reservation=False,
                res_cpu_millicores=holder_res_cpu,
                res_memory_bytes=holder_res_mem,
                res_disk_bytes=holder_res_disk,
                res_device_json=holder_res_device,
                constraints_json=holder_constraints_json,
                has_coscheduling=False,
                coscheduling_group_by="",
                scheduling_timeout_ms=None,
                max_task_failures=0,
                entrypoint_json=holder_entrypoint_json,
                environment_json=holder_environment_json,
                bundle_id="",
                ports_json=[],
                max_retries_failure=0,
                max_retries_preemption=DEFAULT_MAX_RETRIES_PREEMPTION,
                timeout_ms=None,
                preemption_policy=0,
                existing_job_policy=0,
                priority_band=0,
                task_image="",
            )
            holder_base = writes.reserve_priority_insertion_base(cur)
            holder_rows: list[dict] = []
            for idx in range(len(request.reservation.entries)):
                holder_rows.append(
                    writes.task_row(
                        task_id=holder_id.task(idx),
                        job_id=holder_id,
                        task_index=idx,
                        state=job_pb2.TASK_STATE_PENDING,
                        submitted_at_ms=effective_submission_ms,
                        max_retries_failure=0,
                        max_retries_preemption=DEFAULT_MAX_RETRIES_PREEMPTION,
                        priority_neg_depth=-holder_id.depth,
                        priority_root_submitted_ms=root_submitted_ms,
                        priority_insertion=holder_base + idx,
                        priority_band=band_sort_key,
                    )
                )
            writes.bulk_insert_tasks(cur, holder_rows)

    cur.register(
        lambda: log_event(
            "job_submitted",
            job_id.to_wire(),
            num_tasks=replicas,
            error=validation_error,
        )
    )


def cancel(
    cur: Tx,
    *,
    job_id: JobName,
    reason: str,
    endpoints: EndpointsProjection,
    health: WorkerHealthTracker,
) -> None:
    """Cancel ``job_id`` and its descendant subtree through the kernel.

    Loads a snapshot covering every job in the subtree and all their active
    tasks (so the kernel can fire coscheduled-peer cascades on killed tasks),
    runs :func:`apply_cancel_job_batch`, then applies the resulting effects.

    Fixes the latent coscheduled-skip bug in the legacy direct-SQL cancel:
    when one half of an atomic coscheduled group is cancelled, the kernel
    cascades termination to the surviving peers instead of stranding them.
    """
    now = Timestamp.now()
    # The slice closes the full descendant subtree (and every job's tasks /
    # active rows) so the kernel can cascade-kill children and fire
    # coscheduled-peer cascades on killed tasks.
    snapshot = load_jobs_slice(cur, [job_id], now=now)
    if job_id not in snapshot.job_configs:
        return
    # No per-job state preload: the cascade-kill merge guard skips already-
    # terminal rows (excluding WORKER_FAILED, which cancel overwrites).
    effects = apply_cancel_job_batch(snapshot, job_id, reason, now)
    apply_effects(cur, effects, health=health, endpoints=endpoints, now=now)
    # Sweep endpoints that survived because their owning task was already
    # terminal before cancel ran (kernel only emits EndpointDeletion for
    # tasks we actively killed). Derive the same subtree the kernel cancelled
    # from the snapshot's transitive descendants.
    subtree = [job_id, *snapshot.job_descendants[job_id].descendants_full]
    endpoints.remove_by_job_ids(cur, subtree)


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
    writes.delete_job(cur, job_id)
    cur.register(
        lambda: log_event(
            "job_removed",
            job_id.to_wire(),
            state=job_state,
        )
    )
    return True
