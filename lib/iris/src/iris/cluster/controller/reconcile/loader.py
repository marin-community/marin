# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot loader: produces TransitionSnapshot instances for the kernel."""

from collections.abc import Iterable
from typing import Protocol

from rigging.timing import Timestamp
from sqlalchemy import bindparam, select

from iris.cluster.controller import reads
from iris.cluster.controller.attempt_counts import AttemptCounts
from iris.cluster.controller.db import Tx
from iris.cluster.controller.reads import TaskScope
from iris.cluster.controller.reconcile.policy import NON_TERMINAL_TASK_STATES
from iris.cluster.controller.reconcile.snapshot import (
    JobConfigRow,
    JobDescendants,
    JobStateBasis,
    TaskHistogramRow,
    TransitionSnapshot,
    pick_earliest_task_error,
)
from iris.cluster.controller.schema import (
    job_config_table,
    jobs_table,
    local_tasks,
)
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    AttemptUid,
    JobName,
    WorkerId,
)


class TransitionReader(Protocol):
    """A read surface that yields a closed :class:`TransitionSnapshot` for the kernel.

    Lets a backend author its task-state projection from a point-in-time snapshot
    without reading the controller database directly. Implementations seed by
    whichever entities their reconcile path starts from (workers + observation
    uids for the worker path, tasks + attempt keys for the direct path); the
    snapshot closes over everything the kernel may touch.
    """

    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: Iterable[WorkerId] = (),
        observation_uids: Iterable[AttemptUid] = (),
        seed_task_ids: Iterable[JobName] = (),
        extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    ) -> TransitionSnapshot:
        """Load a closed snapshot stamped with ``now``, seeded by the given entities."""
        ...


def _build_multi_root_descendants_stmt():
    """Build a single recursive CTE walking descendants from many roots.

    Seeds with all roots (each row carries its own ``root_id``) and walks down
    via ``parent_job_id``. Returns ``(root_id, descendant_id)`` pairs.
    """
    j_seed = jobs_table.alias("j_seed")
    base_q = select(
        j_seed.c.job_id.label("root_id"),
        j_seed.c.job_id.label("descendant_id"),
    ).where(j_seed.c.job_id.in_(bindparam("root_ids", expanding=True)))
    subtree = base_q.cte("subtree", recursive=True)

    j_child = jobs_table.alias("j_child")
    recursive_q = select(
        subtree.c.root_id,
        j_child.c.job_id.label("descendant_id"),
    ).join(subtree, j_child.c.parent_job_id == subtree.c.descendant_id)
    full = subtree.union_all(recursive_q)
    # Drop the seed rows themselves (root_id == descendant_id) — callers only
    # want strict descendants.
    return select(full.c.root_id, full.c.descendant_id).where(full.c.root_id != full.c.descendant_id)


_MULTI_ROOT_DESCENDANTS_STMT = _build_multi_root_descendants_stmt()


def _load_descendants_multi(cur: Tx, root_ids: Iterable[JobName]) -> dict[JobName, JobDescendants]:
    """Load descendants for all ``root_ids`` in one recursive-CTE query.

    Returns a :class:`JobDescendants` per root from a single ``(root_id,
    descendant_id)`` result set.
    """
    ids = list(root_ids)
    if not ids:
        return {}
    rows = cur.execute(_MULTI_ROOT_DESCENDANTS_STMT, {"root_ids": ids}).all()
    full_by_root: dict[JobName, list[JobName]] = {root_id: [] for root_id in ids}
    for row in rows:
        full_by_root.setdefault(row.root_id, []).append(row.descendant_id)
    return {
        root_id: JobDescendants(
            job_id=root_id,
            descendants=tuple(full_by_root.get(root_id, ())),
        )
        for root_id in ids
    }


def _bulk_load_job_state_basis(
    cur: Tx,
    job_ids: Iterable[JobName],
    all_tasks_by_job: dict[JobName, tuple[TaskHistogramRow, ...]],
) -> dict[JobName, JobStateBasis]:
    """Load job state/started_at/max_task_failures for all ``job_ids`` in one query.

    Builds :class:`JobStateBasis` per job using ``all_tasks_by_job`` for the
    task histogram and first error — no per-job SQL loop.
    """
    ids = list(job_ids)
    if not ids:
        return {}
    basis_rows = cur.execute(
        select(jobs_table.c.job_id, jobs_table.c.state, jobs_table.c.started_at_ms, job_config_table.c.max_task_failures)
        .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
        .where(jobs_table.c.job_id.in_(bindparam("job_ids", expanding=True))),
        {"job_ids": ids},
    ).all()

    result: dict[JobName, JobStateBasis] = {}
    for basis_row in basis_rows:
        job_id = basis_row.job_id
        state = int(basis_row.state)
        max_task_failures = int(basis_row.max_task_failures)
        started_at: Timestamp | None = basis_row.started_at_ms

        if state in TERMINAL_JOB_STATES:
            result[job_id] = JobStateBasis(
                job_id=job_id,
                state=state,
                started_at=started_at,
                max_task_failures=max_task_failures,
                task_state_counts={},
                total_failures=0,
                first_task_error=None,
            )
            continue

        rows = all_tasks_by_job.get(job_id, ())
        histogram: dict[int, int] = {}
        total_failures = 0
        for row in rows:
            histogram[row.state] = histogram.get(row.state, 0) + 1
            total_failures += row.failure_count

        result[job_id] = JobStateBasis(
            job_id=job_id,
            state=state,
            started_at=started_at,
            max_task_failures=max_task_failures,
            task_state_counts=histogram,
            total_failures=total_failures,
            first_task_error=pick_earliest_task_error(
                (row.task_index, row.state, row.finished_at, row.error) for row in rows
            ),
        )
    return result


def _load_all_tasks_for_jobs(cur: Tx, job_ids: Iterable[JobName]) -> dict[JobName, tuple[TaskHistogramRow, ...]]:
    """Load every task row for ``job_ids``, grouped by job and sorted by task index.

    ``failure_count`` is derived from each task's attempt rows (it feeds the
    job-level cumulative ``total_failures`` budget).
    """
    ids = list(job_ids)
    if not ids:
        return {}
    rows = cur.execute(
        select(
            local_tasks.c.task_id,
            local_tasks.c.job_id,
            local_tasks.c.task_index,
            local_tasks.c.state,
            local_tasks.c.error,
            local_tasks.c.finished_at_ms,
        ).where(local_tasks.c.job_id.in_(bindparam("job_ids", expanding=True))),
        {"job_ids": ids},
    ).all()
    counts = reads.attempt_counts_for_tasks(cur, [r.task_id for r in rows])
    grouped: dict[JobName, list[TaskHistogramRow]] = {jid: [] for jid in ids}
    for r in rows:
        grouped.setdefault(r.job_id, []).append(
            TaskHistogramRow(
                task_id=r.task_id,
                task_index=int(r.task_index),
                state=int(r.state),
                failure_count=counts.get(r.task_id, AttemptCounts()).failure_count,
                error=str(r.error) if r.error is not None else None,
                finished_at=r.finished_at_ms,
            )
        )
    return {jid: tuple(sorted(rows, key=lambda row: row.task_index)) for jid, rows in grouped.items()}


def _load_job_num_tasks(cur: Tx, job_ids: Iterable[JobName]) -> dict[JobName, int]:
    ids = list(job_ids)
    if not ids:
        return {}
    rows = cur.execute(
        select(jobs_table.c.job_id, jobs_table.c.num_tasks).where(
            jobs_table.c.job_id.in_(bindparam("job_ids", expanding=True))
        ),
        {"job_ids": ids},
    ).all()
    return {row.job_id: int(row.num_tasks) for row in rows}


def _build_job_configs(cur: Tx, job_ids: Iterable[JobName]) -> dict[JobName, JobConfigRow]:
    """Build :class:`JobConfigRow` per job, folding in ``num_tasks``."""
    ids = list(job_ids)
    num_tasks_by_job = _load_job_num_tasks(cur, ids)
    raw_configs = reads.bulk_get_job_configs(cur, ids)
    return {
        job_id: JobConfigRow(
            job_id=job_id,
            has_coscheduling=bool(config["has_coscheduling"]),
            max_task_failures=int(config["max_task_failures"]),
            preemption_policy=int(config["preemption_policy"]),
            num_tasks=int(num_tasks_by_job.get(job_id, 0)),
        )
        for job_id, config in raw_configs.items()
    }


def load_closed_snapshot(
    cur: Tx,
    *,
    now: Timestamp,
    seed_task_ids: Iterable[JobName] = (),
    seed_job_ids: Iterable[JobName] = (),
    seed_worker_ids: Iterable[WorkerId] = (),
    extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    observation_uids: Iterable[AttemptUid] = (),
) -> TransitionSnapshot:
    """Build a closed :class:`TransitionSnapshot` from arbitrary seeds.

    Callers seed by whichever entity their path starts from — tasks, jobs,
    and/or workers — and the snapshot closes over everything the kernel may
    touch. Seed semantics:

    * ``seed_job_ids`` expand to their full descendant subtree; every task of
      every job in the subtree joins the slice.
    * ``seed_worker_ids`` pull in the active tasks currently on those workers
      (the worker-failure path finalizes whatever they were running) and are
      recorded as ``active_workers`` if they still exist.
    * ``seed_task_ids`` join verbatim, as do ``extra_attempt_keys`` (non-current
      attempts, e.g. direct-provider stale-attempt validation) and the tasks /
      attempts resolved from ``observation_uids``.

    Closure invariants (every kernel-touched relation is present for *every*
    job in the slice, not just the roots):

    * ``tasks`` covers every seed task, every task resolved from an
      observation UID, every active task on a seeded worker, and every task
      of every job in the seeded-job subtrees.
    * ``attempts`` covers each task's current attempt plus ``extra_attempt_keys``
      and the attempts resolved from ``observation_uids``.
    * ``job_descendants``, ``job_state_basis``, ``all_tasks_by_job``,
      ``job_configs``, and ``active_tasks_by_job`` cover the *full* job set:
      the jobs of every loaded task, the seeded jobs, and the transitive
      descendant subtree of every job already in the set. Any of them may
      therefore act as a coscheduled-peer or cascade source.
    * ``active_workers`` is the subset of seeded workers that still exist.
    """
    seed_task_set: set[JobName] = set(seed_task_ids)
    requested_attempt_keys: set[tuple[JobName, int]] = set(extra_attempt_keys)

    # Resolve observation UIDs to (task_id, attempt_id); the resolved tasks +
    # their (possibly non-current) attempts join the slice.
    attempt_uid_index = reads.resolve_attempt_uids(cur, list(observation_uids))
    for resolved_task_id, resolved_attempt_id in attempt_uid_index.values():
        seed_task_set.add(resolved_task_id)
        requested_attempt_keys.add((resolved_task_id, int(resolved_attempt_id)))

    # Tasks currently running on the seeded workers join the slice so the
    # kernel can finalize them when the worker fails.
    seed_worker_list = list(seed_worker_ids)
    if seed_worker_list:
        for row in reads.list_active_tasks(cur, TaskScope(worker_ids=seed_worker_list), states=ACTIVE_TASK_STATES):
            seed_task_set.add(row.task_id)
            requested_attempt_keys.add((row.task_id, int(row.current_attempt_id)))

    # Expand seeded jobs to their full descendant subtree and pull in every
    # task of every job in those subtrees.
    job_set: set[JobName] = set(seed_job_ids)
    if job_set:
        for desc in _load_descendants_multi(cur, job_set).values():
            job_set.update(desc.descendants)
        for rows in _load_all_tasks_for_jobs(cur, job_set).values():
            seed_task_set.update(row.task_id for row in rows)

    tasks = reads.bulk_get_task_detail(cur, list(seed_task_set))
    # Fold each task's current_attempt_id into the attempt fetch so stale-
    # attempt lookups also resolve from the snapshot.
    for task_id, task in tasks.items():
        requested_attempt_keys.add((task_id, int(task.current_attempt_id)))
    attempts = reads.bulk_get_attempts(cur, list(requested_attempt_keys))

    # Full job set: jobs of every loaded task + seeded jobs + the transitive
    # descendant subtree of every job already in the set. Loading descendants
    # for jobs-of-tasks (not just seeded jobs) ensures cascade targets are
    # present for task- and worker-seeded slices too.
    job_set.update(task.job_id for task in tasks.values())
    job_descendants = _load_descendants_multi(cur, job_set)
    for desc in job_descendants.values():
        job_set.update(desc.descendants)

    # Re-walk so descendants pulled in above also expose their own subtrees as
    # cascade sources, then close the per-job relations over the full set.
    job_descendants = _load_descendants_multi(cur, job_set)
    job_configs = _build_job_configs(cur, job_set)
    all_tasks_by_job = _load_all_tasks_for_jobs(cur, job_set)
    job_state_basis = _bulk_load_job_state_basis(cur, job_set, all_tasks_by_job)
    active_tasks_by_job = reads.list_active_tasks_for_jobs(cur, job_set, states=NON_TERMINAL_TASK_STATES)

    existing_worker_strs = reads.filter_existing_workers(cur, seed_worker_list)
    requested_by_str = {str(w): w for w in seed_worker_list}
    active_workers = frozenset(requested_by_str[s] for s in existing_worker_strs if s in requested_by_str)

    return TransitionSnapshot(
        now=now,
        tasks=tasks,
        attempts=attempts,
        attempt_uid_index=attempt_uid_index,
        job_configs=job_configs,
        job_state_basis=job_state_basis,
        job_descendants=job_descendants,
        all_tasks_by_job=all_tasks_by_job,
        active_tasks_by_job=active_tasks_by_job,
        active_workers=active_workers,
    )
