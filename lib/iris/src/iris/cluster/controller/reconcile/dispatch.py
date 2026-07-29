# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller-side dispatch drain for cluster backends that own placement.

The counterpart to :mod:`reconcile.worker` (which builds per-worker plans for a
worker-daemon backend): this reads and writes the DB inside a controller
transaction to produce the :class:`DispatchBatch` a cluster backend (Kueue
today) reconciles against. It promotes PENDING tasks, builds per-attempt
``RunTaskRequest``s, and snapshots the running set. Because it owns DB I/O it
lives controller-side, not in the DB-less backend; the controller rides its
output on the reconcile ``ControlSnapshot``.
"""

from collections import defaultdict
from dataclasses import dataclass, field, replace

from rigging.timing import Timestamp
from sqlalchemy import select

from iris.cluster.controller import reads, writes
from iris.cluster.controller.budget import (
    UserTask,
    compute_effective_band,
    compute_user_spend,
    interleave_by_user,
)
from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.run_templates import build_run_request_fields
from iris.cluster.controller.reads import (
    PENDING_DISPATCH_COLS,
    PendingDispatchRow,
    TaskScope,
    pending_dispatch_row,
)
from iris.cluster.controller.schema import job_config_table, jobs_table, local_tasks
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, RunningTaskEntry
from iris.cluster.types import JobName, UserBudgetDefaults
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


def build_run_request(
    cur: Tx,
    row: PendingDispatchRow,
    attempt_id: int,
) -> job_pb2.RunTaskRequest:
    """Assemble a RunTaskRequest for a direct-provider dispatch row."""
    run_req = build_run_request_fields(
        num_tasks=row.num_tasks,
        entrypoint_json=row.entrypoint_json,
        workdir_files=reads.get_workdir_files(cur, row.job_id),
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
        container_profile=row.container_profile,
    )
    # Propagate timeout for K8s activeDeadlineSeconds (Kubernetes-native enforcement).
    if row.timeout_ms is not None and row.timeout_ms > 0:
        run_req.timeout.milliseconds = row.timeout_ms
    # Coscheduling drives Kueue gang admission on the direct path.
    if row.has_coscheduling:
        run_req.coscheduling.group_by = row.coscheduling_group_by
    # Stamp the attempt's uid so the K8s backend can label the pod with it and
    # tell this attempt's own pod apart from a stale pod a previous job left at
    # the same (task_hash, attempt_id) name. Visible in this tx for both paths:
    # promote inserted the attempt row above, redrive reads the current one.
    uid = reads.attempt_uid_for(cur, row.task_id, attempt_id)
    if uid is not None:
        run_req.attempt_uid = uid
    return run_req


def _dispatch_query(cur: Tx, *predicates) -> list[PendingDispatchRow]:
    """Fetch :class:`PendingDispatchRow`s for the direct-provider drain.

    Both drain queries (PENDING promotion, ASSIGNED redrive) select
    ``PENDING_DISPATCH_COLS`` over the tasks⋈jobs⋈job_config join and supply
    their distinct state predicate; ordering and the rate cap are applied in
    Python by :func:`_rank_promotion_units` and the drain loop.
    """
    dispatch_join = local_tasks.join(jobs_table, jobs_table.c.job_id == local_tasks.c.job_id).join(
        job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
    )
    stmt = select(*PENDING_DISPATCH_COLS).select_from(dispatch_join).where(*predicates)
    return [pending_dispatch_row(r) for r in cur.execute(stmt).all()]


def _unit_sort_key(unit: list[PendingDispatchRow]) -> tuple[int, int, int, int]:
    """Hierarchy/submission ordering key for a promotion unit.

    Mirrors the worker-daemon sort (``reads._PENDING_TASKS_STMT``): parent depth,
    then root submission, then own submission, then insertion. A coscheduled gang
    shares one job, so its members' keys agree except on insertion; the minimum
    anchors the whole gang at its earliest sibling.
    """
    return min(
        (row.priority_neg_depth, row.priority_root_submitted_ms, row.priority_submitted_ms, row.priority_insertion)
        for row in unit
    )


def _build_promotion_units(pending: list[PendingDispatchRow]) -> list[list[PendingDispatchRow]]:
    """Group PENDING dispatch rows into atomic promotion units.

    Non-coscheduled tasks are singleton units. Coscheduled tasks are grouped by
    job into gangs; a gang becomes a unit only once every sibling is PENDING
    together (``len == num_tasks``), keeping siblings on one attempt_id so Kueue
    never waits on pods Iris deferred. A partially-assembled gang is dropped this
    cycle and reconsidered next.
    """
    units: list[list[PendingDispatchRow]] = []
    gangs: dict[JobName, list[PendingDispatchRow]] = {}
    for row in pending:
        if row.has_coscheduling:
            gangs.setdefault(row.job_id, []).append(row)
        else:
            units.append([row])
    for gang in gangs.values():
        if len(gang) == gang[0].num_tasks:
            units.append(gang)
    return units


def _rank_promotion_units(
    units: list[list[PendingDispatchRow]],
    effective_bands: dict[JobName, int],
    user_spend: dict[str, int],
) -> list[list[PendingDispatchRow]]:
    """Order promotion units by effective band, then worker-daemon fairness.

    Units are bucketed by their job's effective band and emitted in ascending
    band order (PRODUCTION first). Within a band they are sorted by the
    hierarchy/submission key, then round-robined across users by ascending spend
    — matching ``scheduling.policy.compute_scheduling_order`` so the direct and
    worker-daemon paths agree on ordering.
    """
    by_band: dict[int, list[list[PendingDispatchRow]]] = defaultdict(list)
    for unit in units:
        by_band[effective_bands[unit[0].job_id]].append(unit)
    ranked: list[list[PendingDispatchRow]] = []
    for band in sorted(by_band):
        band_units = sorted(by_band[band], key=_unit_sort_key)
        user_units = [UserTask(user_id=unit[0].task_id.user, task=unit) for unit in band_units]
        ranked.extend(interleave_by_user(user_units, user_spend))
    return ranked


def drain_for_dispatch(
    cur: Tx,
    *,
    max_promotions: int = DISPATCH_PROMOTION_RATE,
    backend_id: str | None = None,
    defaults: UserBudgetDefaults | None = None,
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

    Candidates are ranked by *effective* band — the ancestor-resolved requested
    band after :func:`compute_effective_band` demotes over-budget users to
    BATCH — before the ``max_promotions`` rate cap applies, so a capped cycle
    never exposes a lower-band pod ahead of a higher-band one. The effective
    band is stamped on ``tasks.priority_band`` and drives the Kueue
    WorkloadPriorityClass; a redrive reuses that fixed band even if spend or
    budget configuration later changes.
    """
    defaults = defaults or UserBudgetDefaults()
    now_ms = Timestamp.now().epoch_ms()
    tasks_to_run: list[job_pb2.RunTaskRequest] = []

    # In a multi-backend cluster, scope the drain to this backend's tasks; a
    # single backend (``backend_id is None``) drains every pending task.
    backend_pred = () if backend_id is None else (local_tasks.c.backend_id == backend_id,)

    # Snapshot redrive set BEFORE the PENDING promotion loop so newly-
    # promoted rows (which become ASSIGNED+null_worker mid-transaction)
    # don't get dispatched twice.
    redrive_rows = _dispatch_query(
        cur,
        local_tasks.c.state == int(job_pb2.TASK_STATE_ASSIGNED),
        local_tasks.c.current_worker_id.is_(None),
        *backend_pred,
    )

    effective_bands: dict[JobName, int] = {}
    rows_to_promote: list[PendingDispatchRow] = []
    if max_promotions > 0:
        # Rank every PENDING candidate by effective band before the rate cap.
        # The whole set is fetched (no SQL limit) so band ordering, not row
        # insertion order, decides what the bounded budget promotes.
        pending = _dispatch_query(
            cur,
            local_tasks.c.state == int(job_pb2.TASK_STATE_PENDING),
            *backend_pred,
        )
        job_ids = {row.job_id for row in pending}
        resolved_bands = reads.get_priority_bands(cur, job_ids)
        user_spend = compute_user_spend(cur)
        user_budget_limits = reads.get_all_user_budget_limits(cur)
        effective_bands = {
            job_id: compute_effective_band(resolved_bands[job_id], job_id.user, user_spend, user_budget_limits, defaults)
            for job_id in job_ids
        }
        units = _rank_promotion_units(_build_promotion_units(pending), effective_bands, user_spend)

        promoted_count = 0
        for unit in units:
            remaining = max_promotions - promoted_count
            if remaining <= 0:
                break
            if len(unit) <= remaining or len(unit) > max_promotions:
                # Fits this cycle, or an oversized gang (larger than the cap
                # itself) promoted whole to avoid a permanent deadlock: Kueue
                # only admits a pod group once every sibling pod exists, so a
                # gang split across cycles would wait forever.
                rows_to_promote.extend(unit)
                promoted_count += len(unit)
            else:
                # A coscheduled gang that fits the cap but not this cycle's
                # remaining budget defers whole. Stop here rather than promote a
                # lower-ranked unit ahead of it — that would invert priority
                # across the band boundary. A later cycle with fuller budget
                # promotes the gang atomically.
                break

    for row in rows_to_promote:
        band = effective_bands[row.job_id]
        row = replace(row, priority_band=band)
        attempt_id = row.current_attempt_id + 1
        writes.promote_for_dispatch(cur, row.task_id, attempt_id, now_ms, priority_band=band)
        tasks_to_run.append(build_run_request(cur, row, attempt_id))

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
        backend_id=backend_id,
    )
    # The K8s provider rebuilds each pod name from (task_id, attempt_id, uid), so
    # poll must carry the current attempt's uid to target the right incarnation.
    uids = reads.attempt_uids_for(cur, [(row.task_id, row.current_attempt_id) for row in running_rows])
    running_tasks = [
        RunningTaskEntry(
            task_id=row.task_id,
            attempt_id=row.current_attempt_id,
            coscheduled=row.has_coscheduling,
            attempt_uid=uids.get((row.task_id, row.current_attempt_id), ""),
        )
        for row in running_rows
    ]

    return DispatchBatch(
        tasks_to_run=tasks_to_run,
        running_tasks=running_tasks,
    )
