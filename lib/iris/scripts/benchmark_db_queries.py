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
    ControllerDB,
    healthy_active_workers_with_attributes,
    running_tasks_by_worker,
)
from iris.cluster.controller.service import (
    USER_JOB_STATES,
    _jobs_in_states,
    _live_user_stats,
    _task_summaries_for_jobs,
    _tasks_for_listing,
    _worker_addresses_for_tasks,
)


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

    p50, p95 = bench("_jobs_in_states (all)", lambda: _jobs_in_states(db, USER_JOB_STATES), iterations=iterations)
    results.append(("_jobs_in_states (all)", p50, p95))
    print_result("_jobs_in_states (all)", p50, p95)

    # Pre-fetch job_ids for _task_summaries_for_jobs
    jobs = _jobs_in_states(db, USER_JOB_STATES)
    job_ids = {j.job_id for j in jobs}

    p50, p95 = bench(
        "_task_summaries_for_jobs (all)", lambda: _task_summaries_for_jobs(db, job_ids), iterations=iterations
    )
    results.append(("_task_summaries_for_jobs (all)", p50, p95))
    print_result("_task_summaries_for_jobs (all)", p50, p95)

    # _worker_addresses_for_tasks requires a list of Task objects
    tasks = _tasks_for_listing(db)
    p50, p95 = bench(
        "_worker_addresses_for_tasks", lambda: _worker_addresses_for_tasks(db, tasks), iterations=iterations
    )
    results.append(("_worker_addresses_for_tasks", p50, p95))
    print_result("_worker_addresses_for_tasks", p50, p95)

    p50, p95 = bench("_live_user_stats", lambda: _live_user_stats(db), iterations=iterations)
    results.append(("_live_user_stats", p50, p95))
    print_result("_live_user_stats", p50, p95)

    # Pre-fetch worker_ids for running_tasks_by_worker
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
