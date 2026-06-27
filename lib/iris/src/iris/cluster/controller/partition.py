# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Slice a global per-tick snapshot into per-backend views.

The controller builds one global read snapshot per tick, then partitions it by
backend so each backend's ``schedule``/``reconcile``/``autoscale`` sees only its
own workers and tasks. Workers are routed to a backend by their scale group
(``backend_of_worker``); jobs by their pinned ``backend_id`` (``backend_of_job``,
which folds in the pins the meta-scheduler computed this tick). Running tasks
follow their worker's backend, since a task runs on exactly one worker.

With a single backend every predicate is constant, so each partition keeps the
whole snapshot — the scheduling decision is byte-for-byte identical to the
unpartitioned path.
"""

from collections.abc import Callable

from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.scheduling.scheduler import SchedulingContext
from iris.cluster.types import JobName, WorkerId


def _job_of_task(task_id: JobName) -> JobName:
    parent = task_id.parent
    return parent if parent is not None else task_id


def partition_scheduling_context(
    ctx: SchedulingContext,
    backend_id: str,
    backend_of_worker: Callable[[WorkerId], str],
    backend_of_job: Callable[[JobName], str],
    *,
    user_spend: dict[str, int] | None = None,
) -> SchedulingContext:
    """Restrict ``ctx`` to one backend's workers and tasks.

    Filters workers (and their building counts) by ``backend_of_worker`` and
    pending tasks/jobs/requested-bands by ``backend_of_job``; running tasks are
    kept when their worker belongs to this backend. Per-user budget fields pass
    through globally — ``user_spend`` overrides the running tally the controller
    threads across backends so one user's budget is not double-spent in a tick.
    Mirrors :meth:`SchedulingContext.evolve_with_workers`: the returned context
    re-derives capacities and the constraint index for its worker subset.
    """
    workers = [w for w in ctx.workers if backend_of_worker(w.worker_id) == backend_id]
    worker_ids = {w.worker_id for w in workers}
    building_counts = {wid: count for wid, count in ctx.building_counts.items() if wid in worker_ids}
    pending_task_rows = [t for t in ctx.pending_task_rows if backend_of_job(t.job_id) == backend_id]
    kept_job_ids = {t.job_id for t in pending_task_rows}
    requested_bands = {jid: band for jid, band in ctx.requested_bands.items() if jid in kept_job_ids}
    pending_tasks = [tid for tid in ctx.pending_tasks if backend_of_job(_job_of_task(tid)) == backend_id]
    jobs = {jid: req for jid, req in ctx.jobs.items() if backend_of_job(jid) == backend_id}
    running = [r for r in ctx.running_for_preemption if backend_of_worker(r.worker_id) == backend_id]
    return SchedulingContext(
        workers=workers,
        building_counts=building_counts,
        max_building_tasks=ctx.max_building_tasks,
        max_assignments_per_worker=ctx.max_assignments_per_worker,
        pending_tasks=pending_tasks,
        jobs=jobs,
        pending_task_rows=pending_task_rows,
        user_spend=ctx.user_spend if user_spend is None else user_spend,
        user_budget_limits=ctx.user_budget_limits,
        requested_bands=requested_bands,
        user_budget_defaults=ctx.user_budget_defaults,
        running_for_preemption=running,
    )


def partition_control_snapshot(
    control: ControlSnapshot,
    backend_id: str,
    backend_of_worker: Callable[[WorkerId], str],
) -> ControlSnapshot:
    """Restrict a reconcile/autoscale snapshot to one backend's workers.

    Keeps the worker addresses, reconcile rows, per-worker status, and the job
    templates referenced by the kept rows. ``timeout_rows`` and the dispatch
    drain fields are dropped: execution-timeout finalization stays controller-
    side and global, and a placement-owning backend's reconcile snapshot comes
    from its own dispatch drain, not from this partition.
    """
    worker_addresses = {
        wid: addr for wid, addr in control.worker_addresses.items() if backend_of_worker(wid) == backend_id
    }
    reconcile_rows = [r for r in control.reconcile_rows if backend_of_worker(r.worker_id) == backend_id]
    kept_jobs = {r.job_id for r in reconcile_rows}
    job_specs = {jid: spec for jid, spec in control.job_specs.items() if jid in kept_jobs}
    worker_status_map = {
        wid: status for wid, status in control.worker_status_map.items() if backend_of_worker(wid) == backend_id
    }
    return ControlSnapshot(
        worker_addresses=worker_addresses,
        reconcile_rows=reconcile_rows,
        timeout_rows=[],
        job_specs=job_specs,
        worker_status_map=worker_status_map,
    )
