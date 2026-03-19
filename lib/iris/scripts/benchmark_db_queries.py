#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Iris controller DB queries against a local checkpoint.

Usage:
    # Download a checkpoint
    gsutil cp gs://<bucket>/<prefix>/controller-state/latest.sqlite3 ./controller.sqlite3

    # Run all benchmarks
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3

    # Run specific benchmark group
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3 --only scheduling
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3 --only dashboard
"""

import time
from pathlib import Path

import click

from iris.cluster.controller.controller import (
    _building_counts,
    _jobs_by_id,
    _schedulable_tasks,
)
from iris.cluster.controller.db import (
    JOBS,
    ControllerDB,
    EndpointQuery,
    healthy_active_workers_with_attributes,
    running_tasks_by_worker,
    tasks_for_job_with_attempts,
)
from iris.cluster.controller.service import (
    USER_JOB_STATES,
    _child_jobs,
    _descendant_jobs,
    _descendants_for_roots,
    _jobs_paginated,
    _live_user_stats,
    _query_endpoints,
    _read_job,
    _read_task_with_attempts,
    _read_worker,
    _read_worker_detail,
    _task_summaries_for_jobs,
    _tasks_for_listing,
    _tasks_for_worker,
    _transaction_actions,
    _worker_addresses_for_tasks,
    _worker_roster,
)
from iris.rpc import cluster_pb2


def bench(name: str, fn: object, *, iterations: int = 20) -> tuple[float, float]:
    """Run fn() for iterations, return (p50, p95) latency in ms."""
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return p50, p95


def print_result(name: str, p50: float, p95: float) -> None:
    print(f"  {name:45s}  p50={p50:8.1f}ms  p95={p95:8.1f}ms")


def benchmark_scheduling(db: ControllerDB, iterations: int) -> list[tuple[str, float, float]]:
    """Benchmark scheduling-loop queries."""
    results: list[tuple[str, float, float]] = []

    p50, p95 = bench("_schedulable_tasks", lambda: _schedulable_tasks(db), iterations=iterations)
    results.append(("_schedulable_tasks", p50, p95))
    print_result("_schedulable_tasks", p50, p95)

    p50, p95 = bench(
        "healthy_active_workers_with_attributes",
        lambda: healthy_active_workers_with_attributes(db),
        iterations=iterations,
    )
    results.append(("healthy_active_workers_with_attributes", p50, p95))
    print_result("healthy_active_workers_with_attributes", p50, p95)

    # Pre-fetch workers for _building_counts
    workers = healthy_active_workers_with_attributes(db)
    p50, p95 = bench("_building_counts", lambda: _building_counts(db, workers), iterations=iterations)
    results.append(("_building_counts", p50, p95))
    print_result("_building_counts", p50, p95)

    # Pre-fetch job_ids for _jobs_by_id
    tasks = _schedulable_tasks(db)
    job_ids = {t.job_id for t in tasks}
    if job_ids:
        p50, p95 = bench("_jobs_by_id", lambda: _jobs_by_id(db, job_ids), iterations=iterations)
        results.append(("_jobs_by_id", p50, p95))
        print_result("_jobs_by_id", p50, p95)
    else:
        print("  _jobs_by_id                                  (skipped, no pending jobs)")

    return results


def benchmark_dashboard(db: ControllerDB, iterations: int) -> list[tuple[str, float, float]]:
    """Benchmark dashboard/service queries."""
    results: list[tuple[str, float, float]] = []

    def _bench_jobs_in_states(db):
        with db.read_snapshot() as q:
            return q.select(JOBS, where=JOBS.c.state.in_(list(USER_JOB_STATES)) & (JOBS.c.depth == 1))

    p50, p95 = bench("jobs_in_states (top-level)", lambda: _bench_jobs_in_states(db), iterations=iterations)
    results.append(("jobs_in_states (top-level)", p50, p95))
    print_result("jobs_in_states (top-level)", p50, p95)

    # Pre-fetch job_ids for _task_summaries_for_jobs
    jobs = _bench_jobs_in_states(db)
    job_ids = {j.job_id for j in jobs}

    p50, p95 = bench(
        "_task_summaries_for_jobs (all)", lambda: _task_summaries_for_jobs(db, job_ids), iterations=iterations
    )
    results.append(("_task_summaries_for_jobs (all)", p50, p95))
    print_result("_task_summaries_for_jobs (all)", p50, p95)

    # _jobs_paginated: the unified SQL pagination path
    p50, p95 = bench(
        "_jobs_paginated (date)",
        lambda: _jobs_paginated(db, USER_JOB_STATES, limit=50),
        iterations=iterations,
    )
    results.append(("_jobs_paginated (date)", p50, p95))
    print_result("_jobs_paginated (date)", p50, p95)

    p50, p95 = bench(
        "_jobs_paginated (name filter)",
        lambda: _jobs_paginated(db, USER_JOB_STATES, name_filter="test", limit=50),
        iterations=iterations,
    )
    results.append(("_jobs_paginated (name filter)", p50, p95))
    print_result("_jobs_paginated (name filter)", p50, p95)

    p50, p95 = bench(
        "_jobs_paginated (sort failures)",
        lambda: _jobs_paginated(
            db, USER_JOB_STATES, sort_field=cluster_pb2.Controller.JOB_SORT_FIELD_FAILURES, limit=50
        ),
        iterations=iterations,
    )
    results.append(("_jobs_paginated (sort failures)", p50, p95))
    print_result("_jobs_paginated (sort failures)", p50, p95)

    # _worker_addresses_for_tasks: use a representative job's tasks
    sample_job = jobs[0] if jobs else None
    if sample_job:
        sample_tasks = _tasks_for_listing(db, job_id=sample_job.job_id)
        p50, p95 = bench(
            "_worker_addresses_for_tasks", lambda: _worker_addresses_for_tasks(db, sample_tasks), iterations=iterations
        )
        results.append(("_worker_addresses_for_tasks", p50, p95))
        print_result("_worker_addresses_for_tasks", p50, p95)
    else:
        print("  _worker_addresses_for_tasks                  (skipped, no jobs)")

    p50, p95 = bench("_live_user_stats", lambda: _live_user_stats(db), iterations=iterations)
    results.append(("_live_user_stats", p50, p95))
    print_result("_live_user_stats", p50, p95)

    # _query_endpoints: unfiltered (all active endpoints) and with name prefix
    p50, p95 = bench("_query_endpoints (all)", lambda: _query_endpoints(db), iterations=iterations)
    results.append(("_query_endpoints (all)", p50, p95))
    print_result("_query_endpoints (all)", p50, p95)

    p50, p95 = bench(
        "_query_endpoints (prefix)",
        lambda: _query_endpoints(db, EndpointQuery(name_prefix="test")),
        iterations=iterations,
    )
    results.append(("_query_endpoints (prefix)", p50, p95))
    print_result("_query_endpoints (prefix)", p50, p95)

    # _transaction_actions: action log
    p50, p95 = bench("_transaction_actions", lambda: _transaction_actions(db), iterations=iterations)
    results.append(("_transaction_actions", p50, p95))
    print_result("_transaction_actions", p50, p95)

    # _worker_roster: list all workers
    p50, p95 = bench("_worker_roster", lambda: _worker_roster(db), iterations=iterations)
    results.append(("_worker_roster", p50, p95))
    print_result("_worker_roster", p50, p95)

    # Pre-fetch workers for running_tasks_by_worker
    workers = healthy_active_workers_with_attributes(db)
    worker_ids = {w.worker_id for w in workers}
    if worker_ids:
        p50, p95 = bench(
            "running_tasks_by_worker", lambda: running_tasks_by_worker(db, worker_ids), iterations=iterations
        )
        results.append(("running_tasks_by_worker", p50, p95))
        print_result("running_tasks_by_worker", p50, p95)
    else:
        print("  running_tasks_by_worker                      (skipped, no workers)")

    # _tasks_for_listing: list tasks for a single job
    if sample_job:
        p50, p95 = bench(
            "_tasks_for_listing (job)",
            lambda: _tasks_for_listing(db, job_id=sample_job.job_id),
            iterations=iterations,
        )
        results.append(("_tasks_for_listing (job)", p50, p95))
        print_result("_tasks_for_listing (job)", p50, p95)

    # _child_jobs: children of a job
    if sample_job:
        p50, p95 = bench("_child_jobs", lambda: _child_jobs(db, sample_job.job_id), iterations=iterations)
        results.append(("_child_jobs", p50, p95))
        print_result("_child_jobs", p50, p95)

    # _descendant_jobs: all descendants
    if sample_job:
        p50, p95 = bench("_descendant_jobs", lambda: _descendant_jobs(db, sample_job.job_id), iterations=iterations)
        results.append(("_descendant_jobs", p50, p95))
        print_result("_descendant_jobs", p50, p95)

    # _descendants_for_roots: batch descendant lookup (used by list_jobs RPC)
    root_job_ids = [j.job_id.to_wire() for j in jobs]
    if root_job_ids:
        p50, p95 = bench(
            "_descendants_for_roots",
            lambda: _descendants_for_roots(db, root_job_ids),
            iterations=iterations,
        )
        results.append(("_descendants_for_roots", p50, p95))
        print_result("_descendants_for_roots", p50, p95)
    else:
        print("  _descendants_for_roots                       (skipped, no jobs)")

    # _read_job: single job lookup
    if sample_job:
        p50, p95 = bench("_read_job", lambda: _read_job(db, sample_job.job_id), iterations=iterations)
        results.append(("_read_job", p50, p95))
        print_result("_read_job", p50, p95)

    # tasks_for_job_with_attempts: job detail page query
    if sample_job:
        p50, p95 = bench(
            "tasks_for_job_with_attempts",
            lambda: tasks_for_job_with_attempts(db, sample_job.job_id),
            iterations=iterations,
        )
        results.append(("tasks_for_job_with_attempts", p50, p95))
        print_result("tasks_for_job_with_attempts", p50, p95)

    # _read_task_with_attempts: single task lookup
    if sample_job:
        sample_tasks_for_read = _tasks_for_listing(db, job_id=sample_job.job_id)
        if sample_tasks_for_read:
            sample_task_id = sample_tasks_for_read[0].task_id
            p50, p95 = bench(
                "_read_task_with_attempts",
                lambda: _read_task_with_attempts(db, sample_task_id),
                iterations=iterations,
            )
            results.append(("_read_task_with_attempts", p50, p95))
            print_result("_read_task_with_attempts", p50, p95)

    # _read_worker: single worker lookup
    roster = _worker_roster(db)
    if roster:
        sample_worker_id = roster[0].worker_id
        p50, p95 = bench("_read_worker", lambda: _read_worker(db, sample_worker_id), iterations=iterations)
        results.append(("_read_worker", p50, p95))
        print_result("_read_worker", p50, p95)

        # _read_worker_detail: worker detail page
        p50, p95 = bench("_read_worker_detail", lambda: _read_worker_detail(db, sample_worker_id), iterations=iterations)
        results.append(("_read_worker_detail", p50, p95))
        print_result("_read_worker_detail", p50, p95)

        # _tasks_for_worker: worker detail task history
        p50, p95 = bench(
            "_tasks_for_worker",
            lambda: _tasks_for_worker(db, sample_worker_id),
            iterations=iterations,
        )
        results.append(("_tasks_for_worker", p50, p95))
        print_result("_tasks_for_worker", p50, p95)

    # Composite: simulate full list_jobs RPC path
    def _list_jobs_full(db):
        paginated_jobs, _total = _jobs_paginated(db, USER_JOB_STATES, limit=50)
        root_ids = [j.job_id.to_wire() for j in paginated_jobs]
        descendants = _descendants_for_roots(db, root_ids)
        all_jobs = paginated_jobs + descendants
        _task_summaries_for_jobs(db, {j.job_id for j in all_jobs})

    p50, p95 = bench("list_jobs_full (composite)", lambda: _list_jobs_full(db), iterations=iterations)
    results.append(("list_jobs_full (composite)", p50, p95))
    print_result("list_jobs_full (composite)", p50, p95)

    return results


def print_summary(results: list[tuple[str, float, float]]) -> None:
    print("\n" + "=" * 75)
    print(f"  {'Query':45s}  {'p50':>10s}  {'p95':>10s}")
    print("-" * 75)
    for name, p50, p95 in results:
        print(f"  {name:45s}  {p50:8.1f}ms  {p95:8.1f}ms")
    print("=" * 75)


def print_db_stats(db: ControllerDB) -> None:
    """Print basic DB size info for context."""
    row_counts = {}
    for table in ("jobs", "tasks", "task_attempts", "workers"):
        rows = db.fetchall(f"SELECT COUNT(*) as cnt FROM {table}")
        row_counts[table] = rows[0]["cnt"]
    print(f"  DB stats: {', '.join(f'{t}={c}' for t, c in row_counts.items())}")


@click.command()
@click.argument("db_path", type=click.Path(exists=True, path_type=Path))
@click.option("--iterations", "-n", default=20, help="Number of iterations per benchmark")
@click.option("--only", "only_group", type=click.Choice(["scheduling", "dashboard"]), help="Run only this group")
def main(db_path: Path, iterations: int, only_group: str | None) -> None:
    """Benchmark Iris controller DB queries against a local checkpoint."""
    db = ControllerDB(db_path)
    db.apply_migrations()
    print(f"Benchmarking {db_path} ({iterations} iterations per query)")
    print_db_stats(db)
    print()

    all_results: list[tuple[str, float, float]] = []

    if only_group is None or only_group == "scheduling":
        print("[scheduling]")
        all_results.extend(benchmark_scheduling(db, iterations))
        print()

    if only_group is None or only_group == "dashboard":
        print("[dashboard]")
        all_results.extend(benchmark_dashboard(db, iterations))

    print_summary(all_results)
    db.close()


if __name__ == "__main__":
    main()
