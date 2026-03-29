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
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3 --only heartbeat

    # Compare with/without ANALYZE statistics
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3 --only heartbeat
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3 --only heartbeat --no-analyze
"""

import shutil
import tempfile
import time
from pathlib import Path

import click

from iris.cluster.controller.controller import (
    _building_counts,
    _jobs_by_id,
    _jobs_with_reservations,
    _schedulable_tasks,
)
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    ControllerDB,
    EndpointQuery,
    Job,
    decode_rows,
    healthy_active_workers_with_attributes,
    running_tasks_by_worker,
    tasks_for_job_with_attempts,
)
from iris.cluster.controller.transitions import (
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
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

    # Reservation queries: compare old (fetch all + filter) vs new (SQL filter)
    reservable_states = (
        cluster_pb2.JOB_STATE_PENDING,
        cluster_pb2.JOB_STATE_BUILDING,
        cluster_pb2.JOB_STATE_RUNNING,
    )

    def _reservation_jobs_old():
        placeholders = ",".join("?" for _ in reservable_states)
        with db.snapshot() as snapshot:
            all_jobs = decode_rows(
                Job, snapshot.fetchall(f"SELECT * FROM jobs WHERE state IN ({placeholders})", reservable_states)
            )
        return [j for j in all_jobs if j.request.HasField("reservation")]

    p50, p95 = bench("reservation_jobs (old: full scan)", _reservation_jobs_old, iterations=iterations)
    results.append(("reservation_jobs (old: full scan)", p50, p95))
    print_result("reservation_jobs (old: full scan)", p50, p95)

    p50, p95 = bench(
        "reservation_jobs (new: has_reservation)",
        lambda: _jobs_with_reservations(db, reservable_states),
        iterations=iterations,
    )
    results.append(("reservation_jobs (new: has_reservation)", p50, p95))
    print_result("reservation_jobs (new: has_reservation)", p50, p95)

    return results


def benchmark_dashboard(db: ControllerDB, iterations: int) -> list[tuple[str, float, float]]:
    """Benchmark dashboard/service queries."""
    results: list[tuple[str, float, float]] = []

    def _bench_jobs_in_states(db):
        placeholders = ",".join("?" for _ in USER_JOB_STATES)
        with db.read_snapshot() as q:
            return decode_rows(
                Job,
                q.fetchall(
                    f"SELECT * FROM jobs WHERE state IN ({placeholders}) AND depth = 1",
                    (*USER_JOB_STATES,),
                ),
            )

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


def benchmark_heartbeat(db: ControllerDB, iterations: int) -> list[tuple[str, float, float]]:
    """Benchmark heartbeat/provider-sync queries."""
    results: list[tuple[str, float, float]] = []

    workers = healthy_active_workers_with_attributes(db)
    worker_ids = {w.worker_id for w in workers}

    if not workers:
        print("  (skipped, no workers)")
        return results

    sample_worker_id = str(workers[0].worker_id)
    active_states = tuple(ACTIVE_TASK_STATES)

    # Single-worker running tasks query (simulates drain_dispatch inner query, 2-way JOIN)
    def _single_worker_running_tasks():
        with db.read_snapshot() as q:
            q.raw(
                "SELECT t.task_id, t.current_attempt_id, t.job_id "
                "FROM tasks t "
                "JOIN task_attempts ta ON t.task_id = ta.task_id AND t.current_attempt_id = ta.attempt_id "
                "WHERE ta.worker_id = ? AND t.state IN (?, ?, ?) "
                "ORDER BY t.task_id ASC",
                (sample_worker_id, *active_states),
            )

    p50, p95 = bench("drain_dispatch running_tasks (1 worker)", _single_worker_running_tasks, iterations=iterations)
    results.append(("drain_dispatch running_tasks (1 worker)", p50, p95))
    print_result("drain_dispatch running_tasks (1 worker)", p50, p95)

    # Full loop: running tasks for ALL workers (simulates phase 1, 2-way JOIN)
    def _all_workers_running_tasks():
        for w in workers:
            with db.read_snapshot() as q:
                q.raw(
                    "SELECT t.task_id, t.current_attempt_id, t.job_id "
                    "FROM tasks t "
                    "JOIN task_attempts ta ON t.task_id = ta.task_id AND t.current_attempt_id = ta.attempt_id "
                    "WHERE ta.worker_id = ? AND t.state IN (?, ?, ?) "
                    "ORDER BY t.task_id ASC",
                    (str(w.worker_id), *active_states),
                )

    p50, p95 = bench(f"drain_dispatch loop ({len(workers)} workers)", _all_workers_running_tasks, iterations=iterations)
    results.append((f"drain_dispatch loop ({len(workers)} workers)", p50, p95))
    print_result(f"drain_dispatch loop ({len(workers)} workers)", p50, p95)

    # Batch running_tasks_by_worker (the db.py helper)
    p50, p95 = bench("running_tasks_by_worker", lambda: running_tasks_by_worker(db, worker_ids), iterations=iterations)
    results.append(("running_tasks_by_worker", p50, p95))
    print_result("running_tasks_by_worker", p50, p95)

    # --- Phase 3 simulation: per-worker vs batched transactions ---
    # Collect (worker_id, task_id, attempt_id) tuples for running tasks.
    running_tasks_per_worker: dict[str, list[tuple[str, int]]] = {}
    for w in workers:
        wid = str(w.worker_id)
        rows = db.fetchall(
            "SELECT t.task_id, t.current_attempt_id "
            "FROM tasks t "
            "JOIN task_attempts ta ON t.task_id = ta.task_id AND t.current_attempt_id = ta.attempt_id "
            "WHERE ta.worker_id = ? AND t.state IN (?, ?, ?)",
            (wid, *active_states),
        )
        if rows:
            running_tasks_per_worker[wid] = [(str(r["task_id"]), int(r["current_attempt_id"])) for r in rows]

    total_tasks = sum(len(v) for v in running_tasks_per_worker.values())
    print(f"  (phase 3 simulation: {len(running_tasks_per_worker)} workers, {total_tasks} running tasks)")

    if running_tasks_per_worker:
        now_ms = int(time.time() * 1000)
        resource_blob = b"\x00" * 64  # dummy resource snapshot

        # Collect unique job_ids for recompute simulation
        job_ids_for_tasks: dict[str, str] = {}  # task_id -> job_id
        for _wid, task_list in running_tasks_per_worker.items():
            for task_id, _ in task_list:
                rows = db.fetchall("SELECT job_id FROM tasks WHERE task_id = ?", (task_id,))
                if rows:
                    job_ids_for_tasks[task_id] = str(rows[0]["job_id"])
        unique_job_ids = set(job_ids_for_tasks.values())

        def _apply_per_worker_old(copy_db: ControllerDB):
            """Old behavior: per-worker txns, per-task recompute, no skip."""
            for wid, task_list in running_tasks_per_worker.items():
                with copy_db.transaction() as cur:
                    cur.execute("SELECT * FROM workers WHERE worker_id = ?", (wid,))
                    cur.execute(
                        "UPDATE workers SET healthy = 1, active = 1, consecutive_failures = 0, "
                        "last_heartbeat_ms = ?, resource_snapshot_proto = ? WHERE worker_id = ?",
                        (now_ms, resource_blob, wid),
                    )
                    cur.execute(
                        "INSERT INTO worker_resource_history(worker_id, snapshot_proto, timestamp_ms) VALUES (?, ?, ?)",
                        (wid, resource_blob, now_ms),
                    )
                    for task_id, attempt_id in task_list:
                        cur.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
                        cur.execute(
                            "SELECT * FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                            (task_id, attempt_id),
                        )
                        cur.execute(
                            "UPDATE task_attempts SET state = ?, started_at_ms = COALESCE(started_at_ms, ?) "
                            "WHERE task_id = ? AND attempt_id = ?",
                            (cluster_pb2.TASK_STATE_RUNNING, now_ms, task_id, attempt_id),
                        )
                        cur.execute(
                            "UPDATE tasks SET state = ?, started_at_ms = COALESCE(started_at_ms, ?) "
                            "WHERE task_id = ?",
                            (cluster_pb2.TASK_STATE_RUNNING, now_ms, task_id),
                        )
                        # Per-task job recompute (old behavior)
                        jid = job_ids_for_tasks.get(task_id)
                        if jid:
                            cur.execute("SELECT state FROM jobs WHERE job_id = ?", (jid,))
                            cur.execute("SELECT state, COUNT(*) AS c FROM tasks WHERE job_id = ? GROUP BY state", (jid,))

        def _apply_batched_optimized(copy_db: ControllerDB):
            """New behavior: single txn, skip no-ops, deduplicated job recompute."""
            with copy_db.transaction() as cur:
                for wid, task_list in running_tasks_per_worker.items():
                    cur.execute("SELECT * FROM workers WHERE worker_id = ?", (wid,))
                    cur.execute(
                        "UPDATE workers SET healthy = 1, active = 1, consecutive_failures = 0, "
                        "last_heartbeat_ms = ?, resource_snapshot_proto = ? WHERE worker_id = ?",
                        (now_ms, resource_blob, wid),
                    )
                    cur.execute(
                        "INSERT INTO worker_resource_history(worker_id, snapshot_proto, timestamp_ms) VALUES (?, ?, ?)",
                        (wid, resource_blob, now_ms),
                    )
                    for task_id, _attempt_id in task_list:
                        # Skip: task already RUNNING, heartbeat reports RUNNING, no new data
                        cur.execute("SELECT state FROM tasks WHERE task_id = ?", (task_id,))
                # Deduplicated job recompute (once per job, not per task)
                for jid in unique_job_ids:
                    cur.execute("SELECT state FROM jobs WHERE job_id = ?", (jid,))
                    cur.execute("SELECT state, COUNT(*) AS c FROM tasks WHERE job_id = ? GROUP BY state", (jid,))

        # Create two directory copies — ControllerDB expects a directory, not a file.
        per_worker_dir = Path(tempfile.mkdtemp(prefix="iris_bench_per_worker_"))
        batched_dir = Path(tempfile.mkdtemp(prefix="iris_bench_batched_"))
        shutil.copy2(db.db_path, per_worker_dir / ControllerDB.DB_FILENAME)
        shutil.copy2(db.db_path, batched_dir / ControllerDB.DB_FILENAME)
        per_worker_db = ControllerDB(per_worker_dir)
        batched_db = ControllerDB(batched_dir)

        try:
            p50, p95 = bench(
                "phase3 old (per-worker, per-task recompute)",
                lambda: _apply_per_worker_old(per_worker_db),
                iterations=iterations,
            )
            results.append(("phase3 old (per-worker, per-task recompute)", p50, p95))
            print_result("phase3 old (per-worker, per-task recompute)", p50, p95)

            p50, p95 = bench(
                f"phase3 new (batched, skip, {len(unique_job_ids)} recomputes)",
                lambda: _apply_batched_optimized(batched_db),
                iterations=iterations,
            )
            results.append((f"phase3 new (batched, skip, {len(unique_job_ids)} recomputes)", p50, p95))
            print_result(f"phase3 new (batched, skip, {len(unique_job_ids)} recomputes)", p50, p95)
        finally:
            per_worker_db.close()
            batched_db.close()
            shutil.rmtree(per_worker_dir, ignore_errors=True)
            shutil.rmtree(batched_dir, ignore_errors=True)

        # --- Two-pass benchmark: real apply_heartbeats_batch with resource_usage ---
        # Builds real HeartbeatApplyRequest objects simulating the common
        # steady-state case: all tasks RUNNING, reporting RUNNING + resource_usage.
        resource_usage_proto = cluster_pb2.ResourceUsage()
        resource_usage_proto.cpu_millicores = 1000
        resource_usage_proto.memory_mb = 1024

        snapshot_proto = cluster_pb2.WorkerResourceSnapshot()

        heartbeat_requests: list[HeartbeatApplyRequest] = []
        for wid, task_list in running_tasks_per_worker.items():
            updates = []
            for task_id, attempt_id in task_list:
                updates.append(
                    TaskUpdate(
                        task_id=JobName.from_wire(task_id),
                        attempt_id=attempt_id,
                        new_state=cluster_pb2.TASK_STATE_RUNNING,
                        resource_usage=resource_usage_proto,
                    )
                )
            heartbeat_requests.append(
                HeartbeatApplyRequest(
                    worker_id=WorkerId(wid),
                    worker_resource_snapshot=snapshot_proto,
                    updates=updates,
                )
            )

        two_pass_dir = Path(tempfile.mkdtemp(prefix="iris_bench_two_pass_"))
        shutil.copy2(db.db_path, two_pass_dir / ControllerDB.DB_FILENAME)
        two_pass_db = ControllerDB(two_pass_dir)
        two_pass_transitions = ControllerTransitions(two_pass_db, log_store=None)  # type: ignore[arg-type]

        try:
            p50, p95 = bench(
                f"phase3 two-pass ({len(heartbeat_requests)} workers, {total_tasks} tasks)",
                lambda: two_pass_transitions.apply_heartbeats_batch(heartbeat_requests),
                iterations=iterations,
            )
            label = f"phase3 two-pass ({len(heartbeat_requests)}w, {total_tasks}t)"
            results.append((label, p50, p95))
            print_result(label, p50, p95)
        finally:
            two_pass_db.close()
            shutil.rmtree(two_pass_dir, ignore_errors=True)

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
@click.option(
    "--only",
    "only_group",
    type=click.Choice(["scheduling", "dashboard", "heartbeat"]),
    help="Run only this group",
)
@click.option("--no-analyze", is_flag=True, help="Skip ANALYZE to test unoptimized query plans")
def main(db_path: Path, iterations: int, only_group: str | None, no_analyze: bool) -> None:
    """Benchmark Iris controller DB queries against a local checkpoint."""
    db = ControllerDB(db_dir=db_path.parent)
    db.apply_migrations()
    if no_analyze:
        print("Dropping sqlite_stat1 to test unoptimized query plans...")
        db.fetchall("DROP TABLE IF EXISTS sqlite_stat1")
        print()
    else:
        print("ANALYZE statistics present (default). Use --no-analyze to compare without.")

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
        print()

    if only_group is None or only_group == "heartbeat":
        print("[heartbeat]")
        all_results.extend(benchmark_heartbeat(db, iterations))

    print_summary(all_results)
    db.close()


if __name__ == "__main__":
    main()
