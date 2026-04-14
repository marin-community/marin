#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Iris controller DB queries against a local checkpoint.

Usage:
    # Auto-download latest archive from the marin cluster and run all benchmarks
    uv run python lib/iris/scripts/benchmark_db_queries.py

    # Use a specific local checkpoint
    uv run python lib/iris/scripts/benchmark_db_queries.py ./controller.sqlite3

    # Re-download even if cached
    uv run python lib/iris/scripts/benchmark_db_queries.py --fresh

    # Run specific benchmark group
    uv run python lib/iris/scripts/benchmark_db_queries.py --only scheduling
    uv run python lib/iris/scripts/benchmark_db_queries.py --only dashboard
    uv run python lib/iris/scripts/benchmark_db_queries.py --only heartbeat
"""

import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable

import click

from iris.cluster.controller.checkpoint import download_checkpoint_to_local
from iris.cluster.controller.controller import (
    _building_counts,
    _find_reservation_ancestor,
    _jobs_by_id,
    _jobs_with_reservations,
    _read_reservation_claims,
    _schedulable_tasks,
)
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    TERMINAL_JOB_STATES,
    ControllerDB,
    EndpointQuery,
    healthy_active_workers_with_attributes,
    running_tasks_by_worker,
    tasks_for_job_with_attempts,
)
from iris.cluster.controller.schema import (
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
)
from iris.cluster.controller.service import (
    USER_JOB_STATES,
    _descendant_jobs,
    _live_user_stats,
    _parent_ids_with_children,
    _query_jobs,
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
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    ReservationClaim,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.rpc import controller_pb2

_results: list[tuple[str, float, float, int]] = []

# Tables needed for write-path benchmarks (queue_assignments, heartbeat, prune).
_CLONE_TABLES = [
    "jobs",
    "job_config",
    "job_workdir_files",
    "tasks",
    "task_attempts",
    "workers",
    "worker_attributes",
    "dispatch_queue",
    "worker_task_history",
    "worker_resource_history",
    "endpoints",
    "reservation_claims",
    "txn_log",
    "txn_actions",
    "meta",
    "schema_migrations",
    "logs",
]


def clone_db(source: ControllerDB) -> ControllerDB:
    """Create a lightweight writable clone via ATTACH + INSERT.

    Much faster than copying a multi-GB file — only copies the rows, not
    the free-page overhead. The clone gets its own ControllerDB with
    migrations already satisfied and ANALYZE stats.
    """
    clone_dir = Path(tempfile.mkdtemp(prefix="iris_bench_clone_"))
    clone_path = clone_dir / ControllerDB.DB_FILENAME
    conn = sqlite3.connect(str(clone_path))
    conn.execute("ATTACH DATABASE ? AS src", (str(source.db_path),))
    # Copy schema + data for each table
    for table in _CLONE_TABLES:
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM src.{table}")
    # Copy indexes from source schema
    rows = conn.execute("SELECT sql FROM src.sqlite_master WHERE type='index' AND sql IS NOT NULL").fetchall()
    for row in rows:
        try:
            conn.execute(row[0])
        except sqlite3.OperationalError:
            pass  # skip indexes on tables we didn't clone
    # Copy triggers
    rows = conn.execute("SELECT sql FROM src.sqlite_master WHERE type='trigger' AND sql IS NOT NULL").fetchall()
    for row in rows:
        try:
            conn.execute(row[0])
        except sqlite3.OperationalError:
            pass
    conn.execute("DETACH DATABASE src")
    conn.execute("ANALYZE")
    conn.close()
    return ControllerDB(clone_dir)


def bench(
    name: str,
    fn: Callable[[], Any],
    *,
    reset: Callable[[], Any] | None = None,
    min_time_s: float = 2.0,
    min_runs: int = 5,
    max_runs: int = 200,
) -> None:
    """Adaptive benchmark: runs fn() until min_time_s elapsed and at least min_runs done.

    If reset is provided, it's called after each iteration (untimed) to restore
    state for the next run. Useful for destructive write benchmarks.
    """
    print(f"  {name:50s}  ", end="", flush=True)
    fn()  # warmup
    if reset:
        reset()

    times: list[float] = []
    elapsed = 0.0
    while len(times) < min_runs or (elapsed < min_time_s and len(times) < max_runs):
        start = time.perf_counter()
        fn()
        dt = time.perf_counter() - start
        times.append(dt * 1000)
        elapsed += dt
        if reset:
            reset()
        if len(times) % 10 == 0:
            print(".", end="", flush=True)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    _results.append((name, p50, p95, len(times)))
    print(f"p50={p50:8.1f}ms  p95={p95:8.1f}ms  (n={len(times)})")


def benchmark_scheduling(db: ControllerDB) -> None:
    """Benchmark scheduling-loop queries."""
    # Create pending work so scheduling queries have realistic load.
    # Pick up to 50 running jobs and revert their first few tasks to PENDING.
    with db.read_snapshot() as snap:
        running_jobs = snap.fetchall(
            "SELECT job_id FROM jobs WHERE state = ? LIMIT 50",
            (job_pb2.JOB_STATE_RUNNING,),
        )
    pending_count = 0
    for job_row in running_jobs:
        jid = job_row["job_id"]
        db.execute(
            "UPDATE tasks SET state = ?, current_worker_id = NULL, current_worker_address = NULL "
            "WHERE job_id = ? AND state = ? AND rowid IN "
            "(SELECT rowid FROM tasks WHERE job_id = ? AND state = ? LIMIT 3)",
            (job_pb2.TASK_STATE_PENDING, jid, job_pb2.TASK_STATE_RUNNING, jid, job_pb2.TASK_STATE_RUNNING),
        )
        pending_count += db.fetchone("SELECT changes() as c")["c"]
    if pending_count:
        print(f"  (created {pending_count} pending tasks across {len(running_jobs)} jobs for scheduling benchmarks)")

    bench("_schedulable_tasks", lambda: _schedulable_tasks(db))

    bench(
        "healthy_active_workers_with_attributes",
        lambda: healthy_active_workers_with_attributes(db),
    )

    workers = healthy_active_workers_with_attributes(db)
    bench("_building_counts", lambda: _building_counts(db, workers))

    tasks = _schedulable_tasks(db)
    job_ids = {t.job_id for t in tasks}
    if job_ids:
        bench("_jobs_by_id", lambda: _jobs_by_id(db, job_ids))
    else:
        print("  _jobs_by_id                                       (skipped, no pending jobs)")

    bench("_read_reservation_claims", lambda: _read_reservation_claims(db))

    if job_ids:
        sample_job_id = next(iter(job_ids))
        bench(
            "_find_reservation_ancestor",
            lambda: _find_reservation_ancestor(db, sample_job_id),
        )
    else:
        print("  _find_reservation_ancestor                        (skipped, no pending jobs)")

    reservable_states = (
        job_pb2.JOB_STATE_PENDING,
        job_pb2.JOB_STATE_BUILDING,
        job_pb2.JOB_STATE_RUNNING,
    )
    bench(
        "_jobs_with_reservations",
        lambda: _jobs_with_reservations(db, reservable_states),
    )

    # --- Write-path benchmarks (use a lightweight clone) ---
    write_db = clone_db(db)
    write_txns = ControllerTransitions(write_db)

    try:
        # queue_assignments: the main write-lock holder in scheduling.
        if tasks and workers:
            worker_list = list(workers)
            sample_assignments: list[Assignment] = []
            for i, t in enumerate(tasks[:20]):
                w = worker_list[i % len(worker_list)]
                sample_assignments.append(Assignment(task_id=t.task_id, worker_id=w.worker_id))

            if sample_assignments:
                n_assign = len(sample_assignments)
                # Save task/attempt state for reset
                task_wires = [a.task_id.to_wire() for a in sample_assignments]
                placeholders_t = ",".join("?" for _ in task_wires)

                def _save_task_state():
                    """Snapshot the rows we're about to mutate."""
                    cols = "task_id, state, current_attempt_id, current_worker_id, current_worker_address, started_at_ms"
                    rows = write_db.fetchall(
                        f"SELECT {cols} FROM tasks WHERE task_id IN ({placeholders_t})",
                        tuple(task_wires),
                    )
                    return [
                        (
                            r["task_id"],
                            r["state"],
                            r["current_attempt_id"],
                            r["current_worker_id"],
                            r["current_worker_address"],
                            r["started_at_ms"],
                        )
                        for r in rows
                    ]

                saved = _save_task_state()

                def _reset_queue_assignments():
                    for tid, st, aid, wid, waddr, started in saved:
                        write_db.execute(
                            "UPDATE tasks SET state=?, current_attempt_id=?, current_worker_id=?, "
                            "current_worker_address=?, started_at_ms=? WHERE task_id=?",
                            (st, aid, wid, waddr, started, tid),
                        )
                        write_db.execute(
                            "DELETE FROM task_attempts WHERE task_id=? AND attempt_id > ?",
                            (tid, aid),
                        )
                    write_db.execute("DELETE FROM dispatch_queue")

                bench(
                    f"queue_assignments ({n_assign} tasks, WRITE)",
                    lambda: write_txns.queue_assignments(sample_assignments),
                    reset=_reset_queue_assignments,
                )
        else:
            print("  queue_assignments (WRITE)                         (skipped, no pending tasks or workers)")

        # replace_reservation_claims: atomic DELETE + INSERT.
        existing_claims = _read_reservation_claims(db)
        claims = existing_claims
        if not claims and workers:
            worker_list = list(workers)
            claims = {
                w.worker_id: ReservationClaim(job_id="synthetic/job", entry_idx=i)
                for i, w in enumerate(worker_list[:10])
            }
        if claims:
            n_claims = len(claims)
            bench(
                f"replace_reservation_claims ({n_claims} claims, WRITE)",
                lambda: write_txns.replace_reservation_claims(claims),
            )
        else:
            print("  replace_reservation_claims (WRITE)                (skipped, no workers)")

        # prune_old_data: single-job CASCADE delete (the unit of lock-holding work).
        terminal_states = tuple(TERMINAL_JOB_STATES)
        t_placeholders = ",".join("?" for _ in terminal_states)
        with write_db.read_snapshot() as snap:
            terminal_row = snap.fetchone(
                f"SELECT job_id FROM jobs WHERE state IN ({t_placeholders}) LIMIT 1",
                terminal_states,
            )
        if terminal_row:
            prune_job_id = terminal_row["job_id"]

            # Save the job + its tasks/attempts for reset
            def _save_prune_state():
                job = write_db.fetchall("SELECT * FROM jobs WHERE job_id = ?", (prune_job_id,))
                tasks_rows = write_db.fetchall("SELECT * FROM tasks WHERE job_id = ?", (prune_job_id,))
                task_ids = [r["task_id"] for r in tasks_rows]
                attempts = []
                if task_ids:
                    ph = ",".join("?" for _ in task_ids)
                    attempts = write_db.fetchall(f"SELECT * FROM task_attempts WHERE task_id IN ({ph})", tuple(task_ids))
                return job, tasks_rows, attempts

            prune_saved = _save_prune_state()

            def _do_prune():
                with write_db.transaction() as cur:
                    cur.execute("DELETE FROM jobs WHERE job_id = ?", (prune_job_id,))

            def _reset_prune():
                job_rows, task_rows, attempt_rows = prune_saved
                for r in job_rows:
                    cols = r.keys()
                    ph = ",".join("?" for _ in cols)
                    write_db.execute(f"INSERT OR REPLACE INTO jobs({','.join(cols)}) VALUES ({ph})", tuple(r))
                for r in task_rows:
                    cols = r.keys()
                    ph = ",".join("?" for _ in cols)
                    write_db.execute(f"INSERT OR REPLACE INTO tasks({','.join(cols)}) VALUES ({ph})", tuple(r))
                for r in attempt_rows:
                    cols = r.keys()
                    ph = ",".join("?" for _ in cols)
                    write_db.execute(f"INSERT OR REPLACE INTO task_attempts({','.join(cols)}) VALUES ({ph})", tuple(r))

            bench("prune_old_data (1 job CASCADE, WRITE)", _do_prune, reset=_reset_prune, min_runs=3, min_time_s=1.0)
        else:
            print("  prune_old_data (1 job CASCADE, WRITE)             (skipped, no terminal jobs)")
    finally:
        write_db.close()
        shutil.rmtree(write_db._db_dir, ignore_errors=True)


def benchmark_dashboard(db: ControllerDB) -> None:
    """Benchmark dashboard/service queries."""

    def _bench_jobs_in_states(db):
        placeholders = ",".join("?" for _ in USER_JOB_STATES)
        with db.read_snapshot() as q:
            return JOB_DETAIL_PROJECTION.decode(
                q.fetchall(
                    f"SELECT * FROM jobs j {JOB_CONFIG_JOIN} " f"WHERE j.state IN ({placeholders}) AND j.depth = 1",
                    (*USER_JOB_STATES,),
                ),
            )

    bench("jobs_in_states (top-level)", lambda: _bench_jobs_in_states(db))

    jobs = _bench_jobs_in_states(db)
    job_ids = {j.job_id for j in jobs}

    bench("_task_summaries_for_jobs (all)", lambda: _task_summaries_for_jobs(db, job_ids))

    roots_by_date = controller_pb2.Controller.JobQuery(
        scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS,
        limit=50,
    )
    bench(
        "_query_jobs (roots, by date)",
        lambda: _query_jobs(db, roots_by_date, USER_JOB_STATES),
    )

    roots_by_name = controller_pb2.Controller.JobQuery(
        scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS,
        name_filter="test",
        limit=50,
    )
    bench(
        "_query_jobs (roots, name filter)",
        lambda: _query_jobs(db, roots_by_name, USER_JOB_STATES),
    )

    roots_by_failures = controller_pb2.Controller.JobQuery(
        scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS,
        sort_field=controller_pb2.Controller.JOB_SORT_FIELD_FAILURES,
        limit=50,
    )
    bench(
        "_query_jobs (roots, sort failures)",
        lambda: _query_jobs(db, roots_by_failures, USER_JOB_STATES),
    )

    sample_job = jobs[0] if jobs else None
    if sample_job:
        sample_tasks = _tasks_for_listing(db, job_id=sample_job.job_id)
        bench("_worker_addresses_for_tasks", lambda: _worker_addresses_for_tasks(db, sample_tasks))
    else:
        print("  _worker_addresses_for_tasks                       (skipped, no jobs)")

    bench("_live_user_stats", lambda: _live_user_stats(db))

    bench("endpoint_registry.query (all)", lambda: db.endpoints.query())

    bench(
        "endpoint_registry.query (prefix)",
        lambda: db.endpoints.query(EndpointQuery(name_prefix="test")),
    )

    bench("_transaction_actions", lambda: _transaction_actions(db))

    bench("_worker_roster", lambda: _worker_roster(db))

    workers = healthy_active_workers_with_attributes(db)
    worker_ids = {w.worker_id for w in workers}
    if worker_ids:
        bench("running_tasks_by_worker", lambda: running_tasks_by_worker(db, worker_ids))
    else:
        print("  running_tasks_by_worker                           (skipped, no workers)")

    if sample_job:
        bench(
            "_tasks_for_listing (job)",
            lambda: _tasks_for_listing(db, job_id=sample_job.job_id),
        )

    if sample_job:
        bench("_descendant_jobs", lambda: _descendant_jobs(db, sample_job.job_id))

    # Use paginated roots (limit=50) like the real list_jobs RPC does, not all jobs
    roots_query = controller_pb2.Controller.JobQuery(
        scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS,
        limit=50,
    )
    paginated_jobs, _ = _query_jobs(db, roots_query, USER_JOB_STATES)
    root_job_ids = [j.job_id for j in paginated_jobs]
    if root_job_ids:
        bench(
            f"_parent_ids_with_children ({len(root_job_ids)} roots)",
            lambda: _parent_ids_with_children(db, root_job_ids),
        )
    else:
        print("  _parent_ids_with_children                         (skipped, no jobs)")

    if sample_job:
        bench("_read_job", lambda: _read_job(db, sample_job.job_id))

    if sample_job:
        bench(
            "tasks_for_job_with_attempts",
            lambda: tasks_for_job_with_attempts(db, sample_job.job_id),
        )

    if sample_job:
        sample_tasks_for_read = _tasks_for_listing(db, job_id=sample_job.job_id)
        if sample_tasks_for_read:
            sample_task_id = sample_tasks_for_read[0].task_id
            bench("_read_task_with_attempts", lambda: _read_task_with_attempts(db, sample_task_id))

    roster = _worker_roster(db)
    if roster:
        sample_worker_id = roster[0].worker_id
        bench("_read_worker", lambda: _read_worker(db, sample_worker_id))
        bench("_read_worker_detail", lambda: _read_worker_detail(db, sample_worker_id))
        bench("_tasks_for_worker", lambda: _tasks_for_worker(db, sample_worker_id))

    def _list_jobs_full(db):
        paginated_jobs, _total = _query_jobs(db, roots_query, USER_JOB_STATES)
        root_ids = [j.job_id for j in paginated_jobs]
        _task_summaries_for_jobs(db, {j.job_id for j in paginated_jobs})
        _parent_ids_with_children(db, root_ids)

    bench("list_jobs_full (composite)", lambda: _list_jobs_full(db))


def benchmark_heartbeat(db: ControllerDB) -> None:
    """Benchmark heartbeat/provider-sync queries."""
    workers = healthy_active_workers_with_attributes(db)
    worker_ids = {w.worker_id for w in workers}

    if not workers:
        print("  (skipped, no workers)")
        return

    sample_worker_id = str(workers[0].worker_id)
    active_states = tuple(ACTIVE_TASK_STATES)

    def _single_worker_running_tasks():
        with db.read_snapshot() as q:
            q.raw(
                "SELECT t.task_id, t.current_attempt_id, t.job_id "
                "FROM tasks t "
                "WHERE t.current_worker_id = ? AND t.state IN (?, ?, ?) "
                "ORDER BY t.task_id ASC",
                (sample_worker_id, *active_states),
            )

    bench("drain_dispatch (1 worker)", _single_worker_running_tasks)

    def _all_workers_running_tasks():
        with db.read_snapshot() as q:
            q.raw(
                "SELECT t.current_worker_id AS worker_id, t.task_id, t.current_attempt_id, t.job_id "
                "FROM tasks t "
                "WHERE t.state IN (?, ?, ?) AND t.current_worker_id IS NOT NULL "
                "ORDER BY t.task_id ASC",
                active_states,
            )

    bench(f"drain_dispatch ({len(workers)} workers)", _all_workers_running_tasks)

    bench("running_tasks_by_worker", lambda: running_tasks_by_worker(db, worker_ids))

    transitions = ControllerTransitions(db)
    bench(
        f"drain_dispatch_all ({len(workers)} workers)",
        lambda: transitions.drain_dispatch_all(),
    )

    # Collect running tasks per worker for apply_heartbeats_batch benchmark.
    running_tasks_per_worker: dict[str, list[tuple[str, int]]] = {}
    for w in workers:
        wid = str(w.worker_id)
        rows = db.fetchall(
            "SELECT t.task_id, t.current_attempt_id "
            "FROM tasks t "
            "WHERE t.current_worker_id = ? AND t.state IN (?, ?, ?)",
            (wid, *active_states),
        )
        if rows:
            running_tasks_per_worker[wid] = [(str(r["task_id"]), int(r["current_attempt_id"])) for r in rows]

    total_tasks = sum(len(v) for v in running_tasks_per_worker.values())
    print(f"  (heartbeat simulation: {len(running_tasks_per_worker)} workers, {total_tasks} running tasks)")

    if not running_tasks_per_worker:
        return

    resource_usage_proto = job_pb2.ResourceUsage()
    resource_usage_proto.cpu_millicores = 1000
    resource_usage_proto.memory_mb = 1024

    snapshot_proto = job_pb2.WorkerResourceSnapshot()

    heartbeat_requests: list[HeartbeatApplyRequest] = []
    for wid, task_list in running_tasks_per_worker.items():
        updates = []
        for task_id, attempt_id in task_list:
            updates.append(
                TaskUpdate(
                    task_id=JobName.from_wire(task_id),
                    attempt_id=attempt_id,
                    new_state=job_pb2.TASK_STATE_RUNNING,
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

    hb_db = clone_db(db)
    hb_transitions = ControllerTransitions(hb_db)

    try:
        bench(
            f"apply_heartbeats_batch ({len(heartbeat_requests)}w, {total_tasks}t)",
            lambda: hb_transitions.apply_heartbeats_batch(heartbeat_requests),
        )

        # prune_worker_resource_history runs in the background prune loop every
        # 10 minutes. It was previously inlined into apply_heartbeats_batch as
        # a per-worker SELECT+DELETE pair, adding ~N*2 queries to each sync
        # cycle's write transaction. Benchmark it here so we can track its cost
        # as a background operation.
        workers_over_limit = hb_db.fetchall(
            "SELECT COUNT(DISTINCT worker_id) as cnt FROM worker_resource_history "
            "GROUP BY worker_id HAVING COUNT(*) >= ?",
            (500,),
        )
        n_over = len(workers_over_limit)
        if n_over:
            bench(
                f"prune_worker_resource_history ({n_over} workers over limit)",
                lambda: hb_transitions.prune_worker_resource_history(),
            )
        else:
            print("  prune_worker_resource_history                     (skipped, no workers over limit)")
    finally:
        hb_db.close()
        shutil.rmtree(hb_db._db_dir, ignore_errors=True)


def print_summary() -> None:
    print("\n" + "=" * 80)
    print(f"  {'Query':50s}  {'p50':>10s}  {'p95':>10s}  {'n':>5s}")
    print("-" * 80)
    for name, p50, p95, n in _results:
        print(f"  {name:50s}  {p50:8.1f}ms  {p95:8.1f}ms  {n:5d}")
    print("=" * 80)


def print_db_stats(db: ControllerDB) -> None:
    """Print basic DB size info for context."""
    row_counts = {}
    for table in ("jobs", "tasks", "task_attempts", "workers"):
        rows = db.fetchall(f"SELECT COUNT(*) as cnt FROM {table}")
        row_counts[table] = rows[0]["cnt"]
    print(f"  DB stats: {', '.join(f'{t}={c}' for t, c in row_counts.items())}")


MARIN_REMOTE_STATE_DIR = "gs://marin-us-central2/iris/marin/state"
DEFAULT_DB_DIR = Path("/tmp/iris_benchmark")


def _ensure_db(db_path: Path | None) -> Path:
    """Download latest archive from the marin cluster if no local DB is provided."""
    if db_path is not None:
        return db_path

    db_dir = DEFAULT_DB_DIR
    db_file = db_dir / ControllerDB.DB_FILENAME
    if db_file.exists():
        print(f"Using cached DB at {db_file}")
        return db_file

    print(f"Downloading latest controller archive from {MARIN_REMOTE_STATE_DIR} ...")
    db_dir.mkdir(parents=True, exist_ok=True)
    ok = download_checkpoint_to_local(MARIN_REMOTE_STATE_DIR, db_dir)
    if not ok:
        raise click.ClickException("No checkpoint found in remote state dir")
    print(f"Downloaded to {db_file}\n")
    return db_file


@click.command()
@click.argument("db_path", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option(
    "--only",
    "only_group",
    type=click.Choice(["scheduling", "dashboard", "heartbeat"]),
    help="Run only this group",
)
@click.option("--no-analyze", is_flag=True, help="Skip ANALYZE to test unoptimized query plans")
@click.option("--fresh", is_flag=True, help="Re-download the archive even if cached")
def main(db_path: Path | None, only_group: str | None, no_analyze: bool, fresh: bool) -> None:
    """Benchmark Iris controller DB queries against a local checkpoint."""
    _results.clear()

    if fresh and db_path is None:
        cached = DEFAULT_DB_DIR / ControllerDB.DB_FILENAME
        if cached.exists():
            cached.unlink()

    db_path = _ensure_db(db_path)
    db = ControllerDB(db_dir=db_path.parent)
    db.apply_migrations()
    if no_analyze:
        print("Dropping sqlite_stat1 to test unoptimized query plans...")
        db.fetchall("DROP TABLE IF EXISTS sqlite_stat1")
        print()
    else:
        print("ANALYZE statistics present (default). Use --no-analyze to compare without.")

    print(f"Benchmarking {db_path}")
    print_db_stats(db)
    print()

    if only_group is None or only_group == "scheduling":
        print("[scheduling]")
        benchmark_scheduling(db)
        print()

    if only_group is None or only_group == "dashboard":
        print("[dashboard]")
        benchmark_dashboard(db)
        print()

    if only_group is None or only_group == "heartbeat":
        print("[heartbeat]")
        benchmark_heartbeat(db)

    print_summary()
    db.close()


if __name__ == "__main__":
    main()
