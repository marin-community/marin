# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for jobs: submit, cancel, remove_finished."""

from dataclasses import dataclass

from rigging.timing import Timestamp
from sqlalchemy import Integer, bindparam, cast, func, insert, select

from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.codec import (
    constraints_to_json,
    entrypoint_to_json,
    proto_to_json,
)
from iris.cluster.controller.db import Tx
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
    meta_table,
)
from iris.cluster.resources.job import JobSpec
from iris.cluster.types import LOCAL_ADMIN_SUBMITTER, LOCAL_CLUSTER, TERMINAL_JOB_STATES, JobName, ResourceSpec
from iris.rpc import job_pb2

_JOB_SUBMISSION_EPOCH_KEY = "job_submission_epoch_ms"


def _extract_resource_cols(resources: ResourceSpec) -> tuple[int, int, int, str | None]:
    """Return ``(cpu_millicores, memory_bytes, disk_bytes, device_json)`` columns.

    Missing resources map to zeros and a NULL device json.
    """
    proto = resources.to_exact_proto()
    return (
        int(proto.cpu_millicores),
        int(proto.memory_bytes),
        int(proto.disk_bytes),
        proto_to_json(proto.device) if proto.HasField("device") else None,
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


def resolve_priority_band(requested_band: int, inherited_band: int | None) -> job_pb2.PriorityBand:
    """Resolve ``PRIORITY_BAND_INHERIT`` to a real band. Call at ingestion only.

    Args:
        requested_band: The band on the launch request; INHERIT means the client asked for none.
        inherited_band: The parent job's stored band, or ``None`` for a root job.
    """
    if requested_band != job_pb2.PRIORITY_BAND_INHERIT:
        return job_pb2.PriorityBand.ValueType(requested_band)
    if inherited_band:
        return job_pb2.PriorityBand.ValueType(inherited_band)
    return job_pb2.PRIORITY_BAND_INTERACTIVE


@dataclass(frozen=True)
class JobInsertResult:
    """What :func:`insert_job_and_config` computed, for the task-materialization
    and audit steps the caller runs next."""

    replicas: int
    effective_submission_ms: int
    root_submitted_ms: int
    validation_error: str | None


@dataclass(frozen=True, slots=True)
class ReceivedHandoff:
    requester_id: str
    owner_principal: str
    handoff_nonce: str


def submit(
    cur: Tx,
    *,
    job_id: JobName,
    spec: JobSpec,
    ts: Timestamp,
    priority_band: int,
    submitting_user: str | None = None,
    received_handoff: ReceivedHandoff | None = None,
) -> None:
    """Insert the job row and expand its tasks. Caller owns the transaction.

    ``priority_band`` must already be resolved — see :func:`resolve_priority_band`.
    ``submitting_user`` is the authenticated principal for a root submission; a
    child ignores it and inherits its root's value (see :func:`insert_job_and_config`).
    """
    inserted = insert_job_and_config(
        cur,
        job_id=job_id,
        spec=spec,
        ts=ts,
        priority_band=priority_band,
        submitting_user=submitting_user,
        received_handoff=received_handoff,
    )
    if inserted.validation_error is None:
        _materialize_tasks(
            cur,
            job_id=job_id,
            num_tasks=inserted.replicas,
            submitted_at_ms=inserted.effective_submission_ms,
            max_retries_failure=spec.max_retries_failure,
            max_retries_preemption=spec.max_retries_preemption,
            priority_root_submitted_ms=inserted.root_submitted_ms,
            priority_band=priority_band,
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
    spec: JobSpec,
    ts: Timestamp,
    priority_band: int,
    cluster: str = LOCAL_CLUSTER,
    submitting_user: str | None = None,
    received_handoff: ReceivedHandoff | None = None,
) -> JobInsertResult:
    """Insert the ``jobs`` + ``job_config`` (+ workdir file) rows for one job.

    Does NOT materialize tasks — :func:`submit` adds them for a local job; a
    federated handoff (``cluster`` set to a peer) has no local tasks (the peer
    creates them; the sync mirrors them back). Caller owns the transaction.

    ``priority_band`` must already be resolved by :func:`resolve_priority_band`, so
    ``job_config.priority_band`` never holds INHERIT and no reader re-derives a band.
    ``submitting_user`` — the authenticated principal — is required for a root and
    stored verbatim. A child ignores it and inherits its root's stored value, so a
    federated subtree keeps the root's submitter no matter who spawns each child.
    """
    assert (
        priority_band != job_pb2.PRIORITY_BAND_INHERIT
    ), f"Job {job_id} would store an unresolved priority band; resolve it at ingestion"

    submitted_ms = ts.epoch_ms()

    stored_epoch = cur.execute(select(meta_table.c.value).where(meta_table.c.key == _JOB_SUBMISSION_EPOCH_KEY)).scalar()
    if stored_epoch is None:
        # Lazy initialization keeps upgraded databases monotone without a schema
        # migration. Cast bypasses TimestampMsType so arithmetic uses a plain int.
        last_ms = int(
            cur.execute(select(func.coalesce(func.max(cast(jobs_table.c.submitted_at_ms, Integer)), 0))).scalar_one()
        )
    else:
        last_ms = int(stored_epoch)
    effective_submission_ms = max(submitted_ms, last_ms + 1)
    writes.meta_value_set(cur, _JOB_SUBMISSION_EPOCH_KEY, effective_submission_ms)

    parent_job_id = job_id.parent.to_wire() if job_id.parent is not None else None
    root_submitted_ms = effective_submission_ms
    if job_id.parent is not None:
        parent_row = cur.execute(
            select(jobs_table.c.root_submitted_at_ms, jobs_table.c.submitting_user).where(
                jobs_table.c.job_id == bindparam("job_id")
            ),
            {"job_id": job_id.parent},
        ).first()
        if parent_row is None:
            raise ValueError(f"Cannot submit job {job_id}: parent {parent_job_id} is absent from the database")
        root_submitted_ms = parent_row.root_submitted_at_ms.epoch_ms()
        # A child inherits its root's submitter, never re-resolving to the acting
        # caller: a federated subtree stays attributed to the principal that
        # launched the root.
        submitting_user = parent_row.submitting_user
    elif submitting_user is None:
        # A root with no resolved principal is an identity-less direct/loopback
        # submit — the same case the submit-time resolver attributes to local_admin.
        submitting_user = LOCAL_ADMIN_SUBMITTER

    deadline_epoch_ms: int | None = None
    if spec.scheduling_timeout is not None and spec.scheduling_timeout.to_ms() > 0:
        deadline_epoch_ms = Timestamp.from_ms(effective_submission_ms).add(spec.scheduling_timeout).epoch_ms()

    replicas = spec.replicas
    validation_error: str | None = None
    if replicas < 1:
        validation_error = f"Job {job_id} has invalid replicas={replicas}; must be >= 1"
        replicas = 0
    elif replicas > MAX_REPLICAS_PER_JOB:
        validation_error = f"Job {job_id} replicas={replicas} exceeds max {MAX_REPLICAS_PER_JOB}"
        replicas = 0

    state = job_pb2.JOB_STATE_PENDING if validation_error is None else job_pb2.JOB_STATE_FAILED
    finished_ms = None if validation_error is None else effective_submission_ms

    res_cpu, res_mem, res_disk, res_device = _extract_resource_cols(spec.resources)
    constraints_json = constraints_to_json(constraint.to_proto() for constraint in spec.constraints)
    has_cosched = spec.coscheduling is not None
    cosched_group = spec.coscheduling.group_by if spec.coscheduling is not None else ""
    sched_timeout: int | None = (
        spec.scheduling_timeout.to_ms()
        if spec.scheduling_timeout is not None and spec.scheduling_timeout.to_ms() > 0
        else None
    )
    entrypoint_json = entrypoint_to_json(spec.entrypoint)
    environment_json = proto_to_json(spec.environment)
    timeout_ms = spec.timeout.to_ms() if spec.timeout is not None else None

    job_name_lower = spec.name.lower()
    writes.insert_job(
        cur,
        job_id=job_id,
        user_id=job_id.user,
        submitting_user=submitting_user,
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
        has_coscheduling=has_cosched,
        coscheduling_group_by=cosched_group,
        scheduling_timeout_ms=sched_timeout,
        max_task_failures=spec.max_task_failures,
        entrypoint_json=entrypoint_json,
        environment_json=environment_json,
        bundle_id=spec.bundle_id,
        ports_json=list(spec.ports),
        max_retries_failure=spec.max_retries_failure,
        max_retries_preemption=spec.max_retries_preemption,
        timeout_ms=timeout_ms,
        preemption_policy=int(spec.preemption_policy),
        existing_job_policy=int(spec.existing_job_policy),
        priority_band=priority_band,
        task_image=spec.task_image,
        container_profile=int(spec.container_profile),
        submit_argv_json=list(spec.submit_argv),
        fail_if_exists=spec.fail_if_exists,
    )

    workdir_files = dict(spec.entrypoint.workdir_files)
    if workdir_files:
        cur.execute(
            insert(job_workdir_files_table),
            [{"job_id": job_id, "filename": name, "data": data} for name, data in workdir_files.items()],
        )

    # A received handoff runs as an ordinary local job, but is recorded as a
    # RECEIVED federated_jobs row (after the jobs row, per the FK) naming the
    # requester, so FederationSync reports it back only to that requester and the
    # changelog events below (and its tasks') resolve their requester from it.
    if received_handoff is not None:
        writes.insert_received_handle(
            cur,
            job_id=job_id,
            requester_id=received_handoff.requester_id,
            owner_principal=received_handoff.owner_principal,
            handoff_nonce=received_handoff.handoff_nonce,
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
    # Fast-path clear of the cancelled subtree's endpoints (the FK CASCADE
    # backstop): cancellation stops routing to these endpoints at once rather
    # than waiting out their lease. Derive the subtree from the snapshot's
    # transitive descendants.
    subtree = [job_id, *snapshot.job_descendants[job_id].descendants]
    cur.caches[EndpointsProjection].remove_by_job_ids(cur, subtree)


def remove_finished(
    cur: Tx,
    job_id: JobName,
    *,
    record_tombstone: bool = True,
) -> bool:
    """Remove a finished job and its tasks from state.

    Only removes jobs that are in a terminal state. Returns True if removed,
    False if the job does not exist or is not finished. ``record_tombstone=False``
    is for a federated resubmission replacing a finished run in place — the
    parent must see the fresh submission's changelog row, not a tombstone.
    """
    job_state = reads.get_job_state(cur, job_id)
    if job_state is None:
        return False
    if job_state not in TERMINAL_JOB_STATES:
        return False
    writes.delete_job(cur, job_id, record_tombstone=record_tombstone)
    cur.register(
        lambda: log_event(
            "job_removed",
            job_id.to_wire(),
            state=job_state,
        )
    )
    return True
