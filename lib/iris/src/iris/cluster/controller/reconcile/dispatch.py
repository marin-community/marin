# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller-side dispatch drain for cluster backends that own placement.

The counterpart to :mod:`reconcile.worker` (which builds per-worker plans for a
worker-daemon backend): this reads and writes the DB inside a controller
transaction to produce the :class:`DispatchBatch` a cluster backend (Kueue
today) reconciles against. It promotes PENDING tasks, builds per-job
``RunTaskRequest`` templates (LRU-cached) and per-attempt requests, and
snapshots the running set. Because it owns DB I/O it lives controller-side, not
in the DB-less backend; the controller rides its output on the reconcile
``ControlSnapshot``.
"""

from dataclasses import dataclass, field

from rigging.timing import Timestamp
from sqlalchemy import select

from iris.cluster.controller import reads, writes
from iris.cluster.controller.codec import constraints_from_json, proto_from_json, resource_spec_from_scalars
from iris.cluster.controller.db import Tx
from iris.cluster.controller.reads import (
    PENDING_DISPATCH_COLS,
    PendingDispatchRow,
    TaskScope,
    pending_dispatch_row,
)
from iris.cluster.controller.run_template import RunTemplateCache
from iris.cluster.controller.schema import job_config_table, jobs_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, RunningTaskEntry
from iris.cluster.types import JobName
from iris.rpc import job_pb2


@dataclass(frozen=True)
class DispatchBatch:
    """The dispatch drain a cluster backend's reconcile tick consumes.

    Rides on :class:`~iris.cluster.controller.reads.ControlSnapshot` as
    ``tasks_to_run`` / ``running_tasks``: tasks the controller promoted to
    ASSIGNED this tick plus the active null-worker roster to poll.
    """

    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)


DISPATCH_PROMOTION_RATE = 128
"""Token bucket capacity for task promotion (pods per minute).

The direct provider relies on the Kubernetes scheduler (and the cloud
autoscaler) for placement and capacity management.  Pods that cannot be
scheduled immediately stay Pending — that signal drives node provisioning.
This rate limit exists only to bound API server pressure."""


def _build_run_request_fields(
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
    )


def run_request_template(
    cache: RunTemplateCache,
    snap: Tx,
    job_id: JobName,
) -> job_pb2.RunTaskRequest | None:
    """Return a cached per-job ``RunTaskRequest`` template.

    Per-attempt fields (``task_id``, ``attempt_id``) are stamped onto a
    copy at fan-out time. Returns ``None`` for jobs with no row.
    """
    wire = job_id.to_wire()
    cached = cache.get(wire)
    if cached is not None:
        return cached

    job = reads.get_job_detail(snap, job_id)
    if job is None:
        return None

    resources = resource_spec_from_scalars(
        job.res_cpu_millicores,
        job.res_memory_bytes,
        job.res_disk_bytes,
        job.res_device_json,
    )
    template = _build_run_request_fields(
        num_tasks=job.num_tasks,
        entrypoint_json=job.entrypoint_json,
        environment_json=job.environment_json,
        bundle_id=job.bundle_id,
        resources=resources,
        ports_json=job.ports_json,
        constraints_json=job.constraints_json,
        task_image=job.task_image,
    )
    for filename, data in reads.get_workdir_files(snap, job_id).items():
        template.entrypoint.workdir_files[filename] = data
    # cache.put interns: it returns the already-cached instance for this key if
    # one exists, otherwise the template we just built. Callers must use the
    # returned value, not ``template``, to share a single canonical instance.
    return cache.put(wire, template)


def build_run_request(
    cur: Tx,
    row: PendingDispatchRow,
    attempt_id: int,
) -> job_pb2.RunTaskRequest:
    """Assemble a RunTaskRequest for a direct-provider dispatch row."""
    run_req = _build_run_request_fields(
        num_tasks=row.num_tasks,
        entrypoint_json=row.entrypoint_json,
        environment_json=row.environment_json,
        bundle_id=row.bundle_id,
        resources=row.resources,
        ports_json=row.ports_json,
        constraints_json=row.constraints_json,
        task_image=row.task_image,
        task_id=row.task_id.to_wire(),
        attempt_id=attempt_id,
        # Priority selects the Kueue WorkloadPriorityClass on the direct path.
        priority=row.priority_band,
    )
    # Load inline workdir files from the job_workdir_files table.
    for filename, data in reads.get_workdir_files(cur, row.job_id).items():
        run_req.entrypoint.workdir_files[filename] = data
    # Propagate timeout for K8s activeDeadlineSeconds (Kubernetes-native enforcement).
    if row.timeout_ms is not None and row.timeout_ms > 0:
        run_req.timeout.milliseconds = row.timeout_ms
    # Coscheduling drives Kueue gang admission on the direct path.
    if row.has_coscheduling:
        run_req.coscheduling.group_by = row.coscheduling_group_by
    return run_req


def _dispatch_query(
    cur: Tx,
    *predicates,
    order_by_job_id: bool = False,
    limit: int | None = None,
) -> list[PendingDispatchRow]:
    """Fetch :class:`PendingDispatchRow`s for the direct-provider drain.

    All drain queries select ``PENDING_DISPATCH_COLS`` over the
    tasks⋈jobs⋈job_config join; callers supply the distinct state /
    coscheduling predicates plus optional ordering and limit.
    """
    dispatch_join = tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id).join(
        job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
    )
    stmt = select(*PENDING_DISPATCH_COLS).select_from(dispatch_join).where(*predicates)
    if order_by_job_id:
        stmt = stmt.order_by(tasks_table.c.job_id)
    if limit is not None:
        stmt = stmt.limit(limit)
    return [pending_dispatch_row(r) for r in cur.execute(stmt).all()]


def drain_for_dispatch(
    cur: Tx,
    *,
    cache: RunTemplateCache,
    max_promotions: int = DISPATCH_PROMOTION_RATE,
) -> DispatchBatch:
    """Drain pending tasks and snapshot running tasks for a direct provider sync cycle.

    Builds RunTaskRequest for two row classes:
    - Up to ``max_promotions`` PENDING rows, each promoted to ASSIGNED
      with a fresh attempt_id.
    - All ASSIGNED+null_worker rows whose pod creation may not have landed
      (controller crashed between assign-commit and ``provider.reconcile``, or
      the prior ``_apply_pod`` errored). ``kubectl apply`` is idempotent;
      re-issuing for a row whose pod already exists is a no-op.

    Every active null-worker row (ASSIGNED/BUILDING/RUNNING) populates
    ``running_tasks`` so the poll observes the pod's current phase. For
    ASSIGNED rows the pod was applied earlier in this same sync (or
    falls through the K8s provider's ``Pod not found`` grace path), so
    the first poll after dispatch transitions the row out of ASSIGNED.

    Kill targets are not enqueued: producing transitions move
    ``tasks.state`` directly to terminal, and the K8s provider's pod
    diff against the desired set deletes the corresponding pod on the
    next sync.
    """
    now_ms = Timestamp.now().epoch_ms()
    tasks_to_run: list[job_pb2.RunTaskRequest] = []

    # Snapshot redrive set BEFORE the PENDING promotion loop so newly-
    # promoted rows (which become ASSIGNED+null_worker mid-transaction)
    # don't get dispatched twice.
    redrive_rows = _dispatch_query(
        cur,
        tasks_table.c.state == int(job_pb2.TASK_STATE_ASSIGNED),
        tasks_table.c.current_worker_id.is_(None),
    )

    def _promote(row: PendingDispatchRow) -> None:
        attempt_id = row.current_attempt_id + 1
        writes.promote_for_dispatch(cur, row.task_id, attempt_id, now_ms)
        tasks_to_run.append(build_run_request(cur, row, attempt_id))

    promoted_count = 0
    if max_promotions > 0:
        # Coscheduled gangs are promoted all-or-none in a single cycle.
        # Kueue only admits a pod group once it has observed every
        # ``pod-group-total-count`` pod, so a gang split across drain
        # cycles — or one larger than the per-cycle cap — would deadlock
        # waiting for pods Iris never created. Fetch the full coscheduled
        # PENDING set (no SQL limit) and promote each gang only once all
        # its tasks are PENDING together; this also keeps every sibling on
        # the same attempt_id, which is the pod-group generation key (see
        # _pod_group_name). Non-coscheduled rows keep the flat,
        # budget-bounded first-fit behavior.
        cosched_pending = _dispatch_query(
            cur,
            tasks_table.c.state == int(job_pb2.TASK_STATE_PENDING),
            job_config_table.c.has_coscheduling == True,  # noqa: E712
            order_by_job_id=True,
        )
        gangs: dict[JobName, list[PendingDispatchRow]] = {}
        for row in cosched_pending:
            gangs.setdefault(row.job_id, []).append(row)
        for gang in gangs.values():
            num_tasks = gang[0].num_tasks
            if len(gang) != num_tasks:
                # Gang not fully assembled yet (siblings still in flight,
                # e.g. mid-bounce); a later cycle promotes it whole once
                # they converge to PENDING.
                continue
            remaining = max_promotions - promoted_count
            if len(gang) > remaining and len(gang) <= max_promotions:
                # Gang fits within the per-cycle cap but not this cycle's
                # remaining budget — defer to a later cycle. (Oversized
                # gangs, larger than the cap itself, fall through and are
                # promoted whole to avoid a permanent deadlock.)
                continue
            for row in gang:
                _promote(row)
            promoted_count += len(gang)

        # Non-coscheduled first-fit, bounded by the remaining budget.
        remaining = max_promotions - promoted_count
        if remaining > 0:
            noncosched_pending = _dispatch_query(
                cur,
                tasks_table.c.state == int(job_pb2.TASK_STATE_PENDING),
                job_config_table.c.has_coscheduling == False,  # noqa: E712
                limit=remaining,
            )
            for row in noncosched_pending:
                _promote(row)

    # Redrive: pods for these rows may not exist yet (crash between
    # assign-commit and apply, or apply errored last cycle). `kubectl
    # apply` is idempotent so re-issuing for a row whose pod is already
    # there is a no-op.
    for row in redrive_rows:
        tasks_to_run.append(build_run_request(cur, row, row.current_attempt_id))

    # Poll every active row (including ASSIGNED) so a pod that just got
    # applied this cycle can transition out of ASSIGNED on the same sync.
    # Pods for ASSIGNED rows either exist (apply_pod ran above) or fall
    # through the K8s provider's "Pod not found" grace path.
    running_rows = reads.list_active_tasks(
        cur,
        TaskScope(null_worker=True),
        states=ACTIVE_TASK_STATES,
        order_by_task_id=True,
    )
    running_tasks = [
        RunningTaskEntry(
            task_id=row.task_id,
            attempt_id=row.current_attempt_id,
            coscheduled=row.has_coscheduling,
        )
        for row in running_rows
    ]

    return DispatchBatch(
        tasks_to_run=tasks_to_run,
        running_tasks=running_tasks,
    )
