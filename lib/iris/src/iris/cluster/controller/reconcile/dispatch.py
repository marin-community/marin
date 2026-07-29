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
from typing import NamedTuple

from rigging.timing import Timestamp
from sqlalchemy import exists, select

from iris.cluster.controller import reads, writes
from iris.cluster.controller.budget import (
    UserTask,
    compute_effective_band,
    compute_user_spend,
    interleave_by_user,
)
from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.run_templates import build_run_request_fields
from iris.cluster.controller.reads import PENDING_DISPATCH_COLS, PendingDispatchRow, pending_dispatch_row
from iris.cluster.controller.schema import job_config_table, jobs_table, local_tasks, task_attempts_table
from iris.cluster.controller.task_state import TaskAttemptEntry
from iris.cluster.types import JobName, UserBudgetDefaults
from iris.rpc import job_pb2


@dataclass(frozen=True)
class DispatchBatch:
    """The dispatch drain a cluster backend's reconcile tick consumes.

    Rides on :class:`~iris.cluster.controller.reads.ControlSnapshot` as
    ``tasks_to_run`` / ``task_attempts``: tasks the controller promoted to
    ASSIGNED this tick and the complete unfinished null-worker attempt roster.
    """

    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    task_attempts: list[TaskAttemptEntry] = field(default_factory=list)


DISPATCH_PROMOTION_RATE = 128
"""Token bucket capacity for task promotion (pods per minute).

The direct provider relies on the Kubernetes scheduler (and the cloud
autoscaler) for placement and capacity management.  Pods that cannot be
scheduled immediately stay Pending — that signal drives node provisioning.
This rate limit exists only to bound API server pressure."""


_CURRENT_ATTEMPT_DRAINED = ~exists().where(
    task_attempts_table.c.task_id == local_tasks.c.task_id,
    task_attempts_table.c.attempt_id == local_tasks.c.current_attempt_id,
    task_attempts_table.c.finished_at_ms.is_(None),
)


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
    """Fetch full :class:`PendingDispatchRow`s (with the runtime config blobs).

    Used for the ASSIGNED-redrive set and, keyed by the already-chosen task ids,
    for the PENDING rows the rate cap actually promotes. Ranking runs first on
    :func:`_ranking_rows`, which omits the heavy JSON columns.
    """
    dispatch_join = local_tasks.join(jobs_table, jobs_table.c.job_id == local_tasks.c.job_id).join(
        job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
    )
    stmt = select(*PENDING_DISPATCH_COLS).select_from(dispatch_join).where(*predicates)
    return [pending_dispatch_row(r) for r in cur.execute(stmt).all()]


class _PriorityKey(NamedTuple):
    """Within-band ordering key, mirroring the worker-daemon sort (``reads._PENDING_TASKS_STMT``).

    Compared field-by-field in this declared order, ascending — the earliest
    ancestor and submission win. A ``NamedTuple`` so ``min``/``sorted`` treat it
    as the plain tuple the comparison needs while each field stays named.
    """

    neg_depth: int
    root_submitted_ms: int
    submitted_ms: int
    insertion: int


@dataclass(frozen=True, slots=True)
class _RankRow:
    """Promotion-candidate fields needed only to order and cap the drain.

    Deliberately excludes the ``PENDING_DISPATCH_COLS`` runtime blobs so a capped
    cycle ranks every pending task cheaply and loads the full row for just the
    winners.
    """

    task_id: JobName
    job_id: JobName
    num_tasks: int
    has_coscheduling: bool
    sort_key: _PriorityKey


def _ranking_rows(cur: Tx, *predicates) -> list[_RankRow]:
    """Fetch lightweight :class:`_RankRow`s (no runtime blobs) for the given predicates."""
    rank_join = local_tasks.join(jobs_table, jobs_table.c.job_id == local_tasks.c.job_id).join(
        job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
    )
    stmt = (
        select(
            local_tasks.c.task_id,
            local_tasks.c.job_id,
            jobs_table.c.num_tasks,
            job_config_table.c.has_coscheduling,
            local_tasks.c.priority_neg_depth,
            local_tasks.c.priority_root_submitted_ms,
            local_tasks.c.submitted_at_ms,
            local_tasks.c.priority_insertion,
        )
        .select_from(rank_join)
        .where(*predicates)
    )
    return [
        _RankRow(
            task_id=r.task_id,
            job_id=r.job_id,
            num_tasks=int(r.num_tasks),
            has_coscheduling=bool(r.has_coscheduling),
            sort_key=_PriorityKey(
                neg_depth=int(r.priority_neg_depth),
                root_submitted_ms=int(r.priority_root_submitted_ms),
                submitted_ms=int(r.submitted_at_ms.epoch_ms()),
                insertion=int(r.priority_insertion),
            ),
        )
        for r in cur.execute(stmt).all()
    ]


def _build_promotion_units(candidates: list[_RankRow]) -> list[list[_RankRow]]:
    """Group PENDING candidates into atomic promotion units.

    Non-coscheduled tasks are singleton units. Coscheduled tasks are grouped by
    job into gangs; a gang becomes a unit only once every sibling is PENDING
    together (``len == num_tasks``), keeping siblings on one attempt_id so Kueue
    never waits on pods Iris deferred. A partially-assembled gang is dropped this
    cycle and reconsidered next.
    """
    units: list[list[_RankRow]] = []
    gangs: dict[JobName, list[_RankRow]] = {}
    for row in candidates:
        if row.has_coscheduling:
            gangs.setdefault(row.job_id, []).append(row)
        else:
            units.append([row])
    for gang in gangs.values():
        if len(gang) == gang[0].num_tasks:
            units.append(gang)
    return units


def _rank_promotion_units(
    units: list[list[_RankRow]],
    effective_bands: dict[JobName, int],
    user_spend: dict[str, int],
) -> list[list[_RankRow]]:
    """Order promotion units by effective band, then per-user fairness.

    Buckets by the job's effective band (ascending: PRODUCTION first); within a
    band, sorts by the hierarchy/submission key and round-robins across users by
    ascending spend. A gang is one atomic unit, so it takes a single round-robin
    turn rather than one per task.
    """
    by_band: dict[int, list[list[_RankRow]]] = defaultdict(list)
    for unit in units:
        by_band[effective_bands[unit[0].job_id]].append(unit)
    ranked: list[list[_RankRow]] = []
    for band in sorted(by_band):
        band_units = sorted(by_band[band], key=lambda unit: min(row.sort_key for row in unit))
        user_units = [UserTask(user_id=unit[0].task_id.user, task=unit) for unit in band_units]
        ranked.extend(interleave_by_user(user_units, user_spend))
    return ranked


def _select_within_cap(
    ranked: list[list[_RankRow]],
    effective_bands: dict[JobName, int],
    max_promotions: int,
) -> list[list[_RankRow]]:
    """Take units from the ranked list up to the ``max_promotions`` cap.

    A unit fits when it is no larger than the remaining budget. An oversized gang
    (larger than the cap itself) is promoted whole, since a partial pod group
    would leave Kueue waiting forever. A gang that fits the cap but not the
    remaining budget defers whole and records its band as a barrier: later units
    in the same band may still fill the budget, but nothing from a worse band may
    promote ahead of the deferred better-band gang — reaching such a unit ends the
    cycle. This bounds waste to same-band fill while keeping cross-band priority.
    """
    selected: list[list[_RankRow]] = []
    promoted = 0
    barrier_band: int | None = None
    for unit in ranked:
        if promoted >= max_promotions:
            break
        band = effective_bands[unit[0].job_id]
        if barrier_band is not None and band > barrier_band:
            break
        if len(unit) <= max_promotions - promoted or len(unit) > max_promotions:
            selected.append(unit)
            promoted += len(unit)
        elif barrier_band is None:
            barrier_band = band
    return selected


def _task_attempts(cur: Tx, backend_id: str | None) -> list[TaskAttemptEntry]:
    """Return every unfinished current attempt owned by a direct provider."""
    backend_pred = () if backend_id is None else (local_tasks.c.backend_id == backend_id,)
    stmt = (
        select(
            local_tasks.c.task_id,
            local_tasks.c.current_attempt_id,
            local_tasks.c.state.label("task_state"),
            job_config_table.c.has_coscheduling,
            task_attempts_table.c.attempt_uid,
        )
        .select_from(
            local_tasks.join(job_config_table, job_config_table.c.job_id == local_tasks.c.job_id).join(
                task_attempts_table,
                (task_attempts_table.c.task_id == local_tasks.c.task_id)
                & (task_attempts_table.c.attempt_id == local_tasks.c.current_attempt_id),
            )
        )
        .where(
            local_tasks.c.current_worker_id.is_(None),
            task_attempts_table.c.finished_at_ms.is_(None),
            *backend_pred,
        )
        .order_by(local_tasks.c.task_id)
    )
    return [
        TaskAttemptEntry(
            task_id=row.task_id,
            attempt_id=int(row.current_attempt_id),
            task_state=int(row.task_state),
            coscheduled=bool(row.has_coscheduling),
            attempt_uid=str(row.attempt_uid),
        )
        for row in cur.execute(stmt).all()
    ]


def drain_for_dispatch(
    cur: Tx,
    *,
    max_promotions: int = DISPATCH_PROMOTION_RATE,
    backend_id: str | None = None,
    defaults: UserBudgetDefaults | None = None,
) -> DispatchBatch:
    """Drain pending tasks and snapshot unfinished attempts for a direct provider sync cycle.

    Builds RunTaskRequest for two row classes:
    - Up to ``max_promotions`` PENDING rows, each promoted to ASSIGNED
      with a fresh attempt_id.
    - All ASSIGNED+null_worker rows whose pod creation may not have landed
      (controller crashed between assign-commit and ``provider.reconcile``, or
      the prior ``_apply_pod`` errored). ``kubectl apply`` is idempotent;
      re-issuing for a row whose pod already exists is a no-op.

    Every unfinished current null-worker attempt populates ``task_attempts``.
    Active tasks are desired pods and are polled normally. Attempts whose task
    has already left the active set remain in the same authoritative roster
    until the provider confirms that their old pod left the active phase.

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
    promote_units: list[list[_RankRow]] = []
    if max_promotions > 0:
        candidates = _ranking_rows(
            cur,
            local_tasks.c.state == int(job_pb2.TASK_STATE_PENDING),
            _CURRENT_ATTEMPT_DRAINED,
            *backend_pred,
        )
        if candidates:
            job_ids = {row.job_id for row in candidates}
            resolved_bands = reads.get_priority_bands(cur, job_ids)
            user_spend = compute_user_spend(cur)
            user_budget_limits = reads.get_all_user_budget_limits(cur)
            effective_bands = {
                job_id: compute_effective_band(
                    resolved_bands[job_id], job_id.user, user_spend, user_budget_limits, defaults
                )
                for job_id in job_ids
            }
            units = _rank_promotion_units(_build_promotion_units(candidates), effective_bands, user_spend)
            promote_units = _select_within_cap(units, effective_bands, max_promotions)

    # Load the runtime blobs for only the promoted rows, then stamp the effective
    # band and build a request per row, keeping the ranked order.
    promote_ids = [row.task_id for unit in promote_units for row in unit]
    if promote_ids:
        heavy = {row.task_id: row for row in _dispatch_query(cur, local_tasks.c.task_id.in_(promote_ids))}
        for task_id in promote_ids:
            band = effective_bands[heavy[task_id].job_id]
            row = replace(heavy[task_id], priority_band=band)
            attempt_id = row.current_attempt_id + 1
            writes.promote_for_dispatch(cur, row.task_id, attempt_id, now_ms, priority_band=band)
            tasks_to_run.append(build_run_request(cur, row, attempt_id))

    # Redrive: pods for these rows may not exist yet (crash between
    # assign-commit and apply, or apply errored last cycle). `kubectl
    # apply` is idempotent so re-issuing for a row whose pod is already
    # there is a no-op.
    for row in redrive_rows:
        tasks_to_run.append(build_run_request(cur, row, row.current_attempt_id))

    return DispatchBatch(
        tasks_to_run=tasks_to_run,
        task_attempts=_task_attempts(cur, backend_id),
    )
