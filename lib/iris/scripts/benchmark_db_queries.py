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
    uv run python lib/iris/scripts/benchmark_db_queries.py --only endpoints
"""

import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
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
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.store import ControllerStores, EndpointQuery
from iris.cluster.controller.schema import (
    ACTIVE_TASK_STATES,
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
)
from iris.cluster.types import TERMINAL_JOB_STATES
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
from iris.cluster.controller.schema import EndpointRow
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    DispatchBatch,
    HeartbeatApplyRequest,
    ReservationClaim,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from rigging.timing import Timestamp

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
    "task_resource_history",
    "endpoints",
    "reservation_claims",
    "txn_log",
    "txn_actions",
    "meta",
    "schema_migrations",
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

    # Use the source's real CREATE TABLE DDL — CREATE TABLE AS SELECT drops
    # UNIQUE/PRIMARY KEY/CHECK constraints, which breaks UPSERT paths like
    # register_worker's INSERT ... ON CONFLICT.
    clone_tables = set(_CLONE_TABLES)
    table_ddl = conn.execute("SELECT name, sql FROM src.sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    for name, sql in table_ddl:
        if name not in clone_tables:
            continue
        conn.execute(sql)
        conn.execute(f"INSERT INTO {name} SELECT * FROM src.{name}")

    # Copy indexes from source schema (skip autoindexes — those come from
    # UNIQUE/PK constraints already in the CREATE TABLE).
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
    conn.commit()
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
    stores = ControllerStores.from_db(db)
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

    def _bench_healthy_active():
        with stores.read() as ctx:
            stores.workers.healthy_active_with_attributes(ctx.cur)

    bench("healthy_active_workers_with_attributes", _bench_healthy_active)

    with stores.read() as ctx:
        workers = stores.workers.healthy_active_with_attributes(ctx.cur)
    bench("_building_counts", lambda: _building_counts(db, workers))

    tasks = _schedulable_tasks(db)
    job_ids = {t.job_id for t in tasks}

    def _bench_jobs_by_id():
        with stores.read() as ctx:
            _jobs_by_id(stores, ctx.cur, job_ids)

    if job_ids:
        bench("_jobs_by_id", _bench_jobs_by_id)
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
    write_stores = ControllerStores.from_db(write_db)
    write_txns = ControllerTransitions(stores=write_stores)

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
    stores = ControllerStores.from_db(db)

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

    bench("_transaction_actions", lambda: _transaction_actions(db))

    bench("_worker_roster", lambda: _worker_roster(db))

    with stores.read() as ctx:
        workers = stores.workers.healthy_active_with_attributes(ctx.cur)
    worker_ids = {w.worker_id for w in workers}

    def _bench_running_by_worker():
        with stores.read() as ctx:
            stores.tasks.running_tasks_by_worker(ctx.cur, worker_ids)

    if worker_ids:
        bench("running_tasks_by_worker", _bench_running_by_worker)
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

    def _bench_tasks_for_job():
        with stores.read() as ctx:
            stores.tasks.tasks_for_job_with_attempts(ctx.cur, sample_job.job_id)

    if sample_job:
        bench("tasks_for_job_with_attempts", _bench_tasks_for_job)

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
    stores = ControllerStores.from_db(db)
    with stores.read() as ctx:
        workers = stores.workers.healthy_active_with_attributes(ctx.cur)
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

    def _bench_running_by_worker_heartbeat():
        with stores.read() as ctx:
            stores.tasks.running_tasks_by_worker(ctx.cur, worker_ids)

    bench("running_tasks_by_worker", _bench_running_by_worker_heartbeat)

    transitions = ControllerTransitions(stores=stores)
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
    hb_stores = ControllerStores.from_db(hb_db)
    hb_transitions = ControllerTransitions(stores=hb_stores)

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


def _healthy_workers(db: ControllerDB) -> list[Any]:
    """Bench-local shim over the post-refactor store API."""
    stores = ControllerStores.from_db(db)
    with stores.read() as ctx:
        return list(stores.workers.healthy_active_with_attributes(ctx.cur))


def _active_task_sample(db: ControllerDB, limit: int) -> list[tuple[JobName, int]]:
    """Return up to ``limit`` (task_id, current_attempt_id) pairs for non-terminal tasks.

    add_endpoint checks that the task is not TERMINAL, so we pick from
    ACTIVE_TASK_STATES to exercise the full RegisterEndpoint write path.
    """
    active_states = tuple(ACTIVE_TASK_STATES)
    placeholders = ",".join("?" for _ in active_states)
    rows = db.fetchall(
        f"SELECT task_id, current_attempt_id FROM tasks "
        f"WHERE state IN ({placeholders}) AND current_attempt_id IS NOT NULL LIMIT ?",
        (*active_states, limit),
    )
    return [(JobName.from_wire(str(r["task_id"])), int(r["current_attempt_id"])) for r in rows]


def _make_endpoint(task_id: JobName) -> EndpointRow:
    return EndpointRow(
        endpoint_id=str(uuid.uuid4()),
        name=f"/bench/endpoint/{uuid.uuid4().hex[:8]}",
        address="127.0.0.1:0",
        task_id=task_id,
        metadata={"bench": "true"},
        registered_at=Timestamp.now(),
    )


def _measure_tail_latency(
    *,
    name: str,
    write_db: ControllerDB,
    write_txns: ControllerTransitions,
    batch_fn: Callable[[], Any],
    endpoint_tasks: list[JobName],
    reset: Callable[[], Any],
) -> None:
    """Fire add_endpoint calls on a probe thread while ``batch_fn`` runs on
    another thread, and report the per-call latency distribution of the
    probe. This is the metric that matches the production symptom —
    RegisterEndpoint RPCs stalling for seconds while a large
    fail_heartbeats_batch holds the SQLite writer — not the batch's own
    wall time.
    """
    print(f"  {name:50s}  ", end="", flush=True)

    def _run() -> tuple[list[float], float]:
        probe_latencies: list[float] = []
        errors: list[BaseException] = []
        stop = threading.Event()

        def _batch() -> None:
            try:
                batch_fn()
            except BaseException as e:
                errors.append(e)
            finally:
                stop.set()

        def _probe() -> None:
            # Hammer add_endpoint back-to-back; record each call's latency.
            # Rotate through endpoint_tasks to avoid exhausting the list.
            i = 0
            try:
                while not stop.is_set():
                    t = endpoint_tasks[i % len(endpoint_tasks)]
                    start = time.perf_counter()
                    write_txns.add_endpoint(_make_endpoint(t))
                    probe_latencies.append((time.perf_counter() - start) * 1000)
                    i += 1
            except BaseException as e:
                errors.append(e)

        t_batch = threading.Thread(target=_batch)
        t_probe = threading.Thread(target=_probe)
        wall_start = time.perf_counter()
        t_batch.start()
        t_probe.start()
        t_batch.join()
        t_probe.join()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        if errors:
            raise errors[0]
        return probe_latencies, wall_ms

    # Single real run — no warmup; the batch itself is expensive.
    latencies, wall_ms = _run()
    reset()

    if not latencies:
        print("(no probe samples)")
        return

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    max_ms = latencies[-1]
    _results.append((name, p50, p95, len(latencies)))
    print(
        f"probes={len(latencies):4d} p50={p50:7.1f}ms  p95={p95:7.1f}ms  "
        f"max={max_ms:7.1f}ms  batch_wall={wall_ms:7.0f}ms"
    )


def _measure_drain_tail_latency(
    *,
    name: str,
    write_db: ControllerDB,
    write_txns: ControllerTransitions,
    batch_fn: Callable[[], Any],
    reset: Callable[[], Any],
) -> None:
    """Measure drain_dispatch_all latency while ``batch_fn`` holds the writer.

    Same pattern as ``_measure_tail_latency`` but probes drain_dispatch_all
    (provider-sync phase 1) instead of add_endpoint.
    """
    print(f"  {name:50s}  ", end="", flush=True)

    def _run() -> tuple[list[float], float]:
        drain_latencies: list[float] = []
        errors: list[BaseException] = []
        stop = threading.Event()

        def _batch() -> None:
            try:
                batch_fn()
            except BaseException as e:
                errors.append(e)
            finally:
                stop.set()

        def _probe() -> None:
            try:
                while not stop.is_set():
                    start = time.perf_counter()
                    write_txns.drain_dispatch_all()
                    drain_latencies.append((time.perf_counter() - start) * 1000)
            except BaseException as e:
                errors.append(e)

        t_batch = threading.Thread(target=_batch)
        t_probe = threading.Thread(target=_probe)
        wall_start = time.perf_counter()
        t_batch.start()
        t_probe.start()
        t_batch.join()
        t_probe.join()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        if errors:
            raise errors[0]
        return drain_latencies, wall_ms

    latencies, wall_ms = _run()
    reset()

    if not latencies:
        print("(no probe samples)")
        return

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    max_ms = latencies[-1]
    _results.append((name, p50, p95, len(latencies)))
    print(
        f"probes={len(latencies):4d} p50={p50:7.1f}ms  p95={p95:7.1f}ms  "
        f"max={max_ms:7.1f}ms  batch_wall={wall_ms:7.0f}ms"
    )


def benchmark_endpoints(db: ControllerDB) -> None:
    """Benchmark registration RPC hot paths: RegisterEndpoint and Register (worker).

    Covers:
      - add_endpoint: single write, burst-per-txn, burst-in-one-txn
      - fail_heartbeats_batch: provider-sync "apply results" failure path
      - endpoint burst contending with apply_heartbeats_batch on a thread
      - register_worker: single, burst of 100, and burst under heartbeat
        contention (matches the production Register p95 of 3-4s)
    """
    # Read-path queries run against the source DB (cheap, no clone needed).
    bench("endpoint_registry.query (all)", lambda: db.endpoints.query())
    bench(
        "endpoint_registry.query (prefix)",
        lambda: db.endpoints.query(EndpointQuery(name_prefix="test")),
    )

    write_db = clone_db(db)
    write_txns = ControllerTransitions(write_db)

    try:
        sample = _active_task_sample(write_db, limit=300)
        if not sample:
            print("  (skipped, no active tasks to attach endpoints to)")
            return

        # Single RegisterEndpoint write: one transaction = one fsync.
        single_task = sample[0][0]

        def _do_single():
            write_txns.add_endpoint(_make_endpoint(single_task))

        def _reset_single():
            write_db.execute("DELETE FROM endpoints WHERE name LIKE '/bench/endpoint/%'")
            write_db.endpoints._load_all()

        bench("add_endpoint (1 write)", _do_single, reset=_reset_single)

        # Burst: N endpoints each in their own transaction. This is what the
        # controller does today — every RegisterEndpoint RPC opens its own
        # write transaction, so N simultaneous callers serialize on the DB
        # write lock.
        for burst_n in (50, 200):
            if len(sample) < burst_n:
                print(f"  add_endpoint burst x{burst_n} (per-txn)                 (skipped, only {len(sample)} tasks)")
                continue
            tasks_for_burst = [t for t, _ in sample[:burst_n]]

            def _do_burst_per_txn(tasks=tasks_for_burst):
                for t in tasks:
                    write_txns.add_endpoint(_make_endpoint(t))

            bench(
                f"add_endpoint burst x{burst_n} (per-txn)",
                _do_burst_per_txn,
                reset=_reset_single,
                min_runs=3,
                min_time_s=1.0,
            )

            # Same N inserts, but coalesced into a single transaction — the
            # upper bound on what a batched RegisterEndpoint would cost.
            def _do_burst_one_txn(tasks=tasks_for_burst):
                with write_db.transaction() as cur:
                    for t in tasks:
                        write_db.endpoints.add(cur, _make_endpoint(t))

            bench(
                f"add_endpoint burst x{burst_n} (1 txn)",
                _do_burst_one_txn,
                reset=_reset_single,
                min_runs=3,
                min_time_s=1.0,
            )

        # Provider-sync "apply results" failure path: mark ~N workers failed
        # in one fail_heartbeats_batch call (force_remove=True matches slice
        # reaping). This is the exact code path the controller log attributes
        # the 29s "apply results" phase to.
        # x50 is enough to show the contention pattern (~5s batch, plenty of
        # headroom to observe starved RPCs). x300 is instructive but 6x longer
        # and doesn't change the conclusion.
        for fail_n in (50,):
            with write_db.read_snapshot() as snap:
                worker_rows = snap.fetchall(
                    "SELECT worker_id, address FROM workers WHERE active = 1 LIMIT ?",
                    (fail_n,),
                )
            if len(worker_rows) < fail_n:
                print(
                    f"  fail_heartbeats_batch x{fail_n}                      "
                    f"(skipped, only {len(worker_rows)} active workers)"
                )
                continue

            failures = [
                (
                    DispatchBatch(
                        worker_id=WorkerId(str(r["worker_id"])),
                        worker_address=str(r["address"]) if r["address"] is not None else None,
                        running_tasks=[],
                    ),
                    "benchmark: simulated provider-sync failure",
                )
                for r in worker_rows
            ]

            # Snapshot worker rows so we can restore between runs. force_remove
            # flips active=0 and clears current_worker_* on tasks, so the reset
            # has to restore the worker table and re-activate their tasks.
            target_ids = [str(r["worker_id"]) for r in worker_rows]
            placeholders_w = ",".join("?" for _ in target_ids)
            saved_workers = write_db.fetchall(
                f"SELECT * FROM workers WHERE worker_id IN ({placeholders_w})",
                tuple(target_ids),
            )
            saved_tasks = write_db.fetchall(
                f"SELECT task_id, state, current_attempt_id, current_worker_id, current_worker_address, "
                f"started_at_ms FROM tasks WHERE current_worker_id IN ({placeholders_w})",
                tuple(target_ids),
            )

            def _reset_fail(saved_w=saved_workers, saved_t=saved_tasks):
                for r in saved_w:
                    cols = r.keys()
                    ph = ",".join("?" for _ in cols)
                    write_db.execute(
                        f"INSERT OR REPLACE INTO workers({','.join(cols)}) VALUES ({ph})",
                        tuple(r),
                    )
                for r in saved_t:
                    write_db.execute(
                        "UPDATE tasks SET state=?, current_attempt_id=?, current_worker_id=?, "
                        "current_worker_address=?, started_at_ms=? WHERE task_id=?",
                        (
                            r["state"],
                            r["current_attempt_id"],
                            r["current_worker_id"],
                            r["current_worker_address"],
                            r["started_at_ms"],
                            r["task_id"],
                        ),
                    )

            bench(
                f"fail_heartbeats_batch x{fail_n} (force_remove)",
                lambda f=failures: write_txns.fail_heartbeats_batch(f, force_remove=True),
                reset=_reset_fail,
                min_runs=3,
                min_time_s=1.0,
            )

            # Tail-latency: what does a concurrent RegisterEndpoint see while
            # fail_heartbeats_batch is running? This is the metric that
            # matches the production symptom (2-6s RegisterEndpoint RPCs
            # during a zone-wide worker failure burst), not the batch's own
            # wall time. One thread runs the failure batch; another thread
            # fires add_endpoint calls back-to-back and records each one's
            # latency. Report p50/p95/max of the endpoint adds.
            endpoint_tasks = [t for t, _ in sample[:200]] if len(sample) >= 200 else None
            if endpoint_tasks:
                # A/B: simulate "one giant txn" (pre-fix) vs chunk_size=10
                # (current default) to show what chunking buys.
                _measure_tail_latency(
                    name=f"add_endpoint tail latency during fail x{fail_n} (1 txn)",
                    write_db=write_db,
                    write_txns=write_txns,
                    batch_fn=lambda f=failures, n=fail_n: write_txns.fail_heartbeats_batch(
                        f, force_remove=True, chunk_size=n
                    ),
                    endpoint_tasks=endpoint_tasks,
                    reset=_reset_fail,
                )
                _measure_tail_latency(
                    name=f"add_endpoint tail latency during fail x{fail_n} (chunk=10)",
                    write_db=write_db,
                    write_txns=write_txns,
                    batch_fn=lambda f=failures: write_txns.fail_heartbeats_batch(f, force_remove=True, chunk_size=10),
                    endpoint_tasks=endpoint_tasks,
                    reset=_reset_fail,
                )

            # Tail-latency for drain_dispatch_all — controller's
            # "provider sync phase 1 (snapshot)" path. Its phase-2
            # dispatch_queue drain opens a write transaction; if
            # fail_heartbeats_batch is holding the writer, phase 1 waits
            # for the full batch. This is the "81s phase 1" you see in the
            # log — it's a victim of writer starvation, not slow itself.
            _measure_drain_tail_latency(
                name=f"drain_dispatch_all tail latency during fail x{fail_n} (1 txn)",
                write_db=write_db,
                write_txns=write_txns,
                batch_fn=lambda f=failures, n=fail_n: write_txns.fail_heartbeats_batch(
                    f, force_remove=True, chunk_size=n
                ),
                reset=_reset_fail,
            )
            _measure_drain_tail_latency(
                name=f"drain_dispatch_all tail latency during fail x{fail_n} (chunk=10)",
                write_db=write_db,
                write_txns=write_txns,
                batch_fn=lambda f=failures: write_txns.fail_heartbeats_batch(f, force_remove=True, chunk_size=10),
                reset=_reset_fail,
            )

        # Contention: run an add_endpoint burst concurrently with an
        # apply_heartbeats_batch call on two Python threads sharing the clone
        # DB. SQLite serializes writers, so this measures write-lock wait.
        workers = _healthy_workers(write_db)
        if workers and len(sample) >= 200:
            active_states = tuple(ACTIVE_TASK_STATES)
            running_by_worker: dict[str, list[tuple[str, int]]] = {}
            for w in workers:
                wid = str(w.worker_id)
                rows = write_db.fetchall(
                    "SELECT task_id, current_attempt_id FROM tasks "
                    "WHERE current_worker_id = ? AND state IN (?, ?, ?)",
                    (wid, *active_states),
                )
                if rows:
                    running_by_worker[wid] = [(str(r["task_id"]), int(r["current_attempt_id"])) for r in rows]

            snapshot_proto = job_pb2.WorkerResourceSnapshot()
            resource_usage_proto = job_pb2.ResourceUsage(cpu_millicores=1000, memory_mb=1024)
            hb_requests: list[HeartbeatApplyRequest] = []
            for wid, task_list in running_by_worker.items():
                updates = [
                    TaskUpdate(
                        task_id=JobName.from_wire(tid),
                        attempt_id=aid,
                        new_state=job_pb2.TASK_STATE_RUNNING,
                        resource_usage=resource_usage_proto,
                    )
                    for tid, aid in task_list
                ]
                hb_requests.append(
                    HeartbeatApplyRequest(
                        worker_id=WorkerId(wid),
                        worker_resource_snapshot=snapshot_proto,
                        updates=updates,
                    )
                )

            burst_tasks = [t for t, _ in sample[:200]]

            def _run_contended():
                errors: list[BaseException] = []

                def _hb():
                    try:
                        write_txns.apply_heartbeats_batch(hb_requests)
                    except BaseException as e:  # surface thread errors
                        errors.append(e)

                def _reg():
                    try:
                        for t in burst_tasks:
                            write_txns.add_endpoint(_make_endpoint(t))
                    except BaseException as e:
                        errors.append(e)

                t1 = threading.Thread(target=_hb)
                t2 = threading.Thread(target=_reg)
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                if errors:
                    raise errors[0]

            bench(
                f"contention: add_endpoint x200 || apply_heartbeats_batch ({len(hb_requests)}w)",
                _run_contended,
                reset=_reset_single,
                min_runs=3,
                min_time_s=1.0,
            )
        else:
            print("  contention (endpoint burst || heartbeats)         (skipped, insufficient workers/tasks)")

        # --- Worker Register RPC benchmarks ---
        # Production report: Register RPC takes 3-4s under burst. Same
        # single-writer transaction path as add_endpoint, but register_worker
        # also writes to worker_attributes and touches the in-memory
        # attribute cache, so measure it separately.
        #
        # NOTE: clone_db() uses CREATE TABLE AS SELECT which drops the UNIQUE
        # constraint needed by register_worker's UPSERT. Skip when the clone
        # can't support it.
        try:
            _bench_register_worker(write_db, write_txns)
        except sqlite3.OperationalError as e:
            print(f"  register_worker benchmarks                        (skipped: {e})")
    finally:
        write_db.close()
        shutil.rmtree(write_db._db_dir, ignore_errors=True)


def _build_sample_worker_metadata() -> job_pb2.WorkerMetadata:
    """Minimal but representative WorkerMetadata for a CPU worker.

    Matches the shape of real worker metadata (device set, a handful of
    attributes) so the INSERT path hits the same column and attribute count
    as production.
    """
    device = job_pb2.DeviceConfig()
    device.cpu.CopyFrom(job_pb2.CpuDevice(variant="cpu"))
    meta = job_pb2.WorkerMetadata(
        hostname="bench-worker",
        ip_address="127.0.0.1",
        cpu_count=64,
        memory_bytes=256 * 1024**3,
        disk_bytes=2 * 1024**4,
        device=device,
    )
    meta.attributes["device_type"].string_value = "cpu"
    meta.attributes["device_variant"].string_value = "cpu"
    meta.attributes["pool"].string_value = "default"
    return meta


def _bench_register_worker(write_db: ControllerDB, write_txns: ControllerTransitions) -> None:
    """Benchmark the worker Register RPC hot path.

    Cases:
      (a) single fresh Register — per-call INSERT cost baseline.
      (b) sequential burst of 100 fresh Registers — what happens when a slice
          of 100 workers all come up at once and each hits its own write txn.
      (c) same 100-burst with apply_heartbeats_batch hammering the same DB on
          a background thread — exercises SQLite write-lock contention, which
          is our leading theory for the 3-4s p95 seen in prod.
    """
    sample_meta = _build_sample_worker_metadata()

    def _register_one(worker_id: WorkerId) -> None:
        write_txns.register_worker(
            worker_id=worker_id,
            address=f"tcp://{worker_id}:1234",
            metadata=sample_meta,
            ts=Timestamp.now(),
            slice_id="",
            scale_group="bench",
        )

    # (a) single Register, fresh worker each iteration.
    single_counter = {"n": 0}

    def _single():
        single_counter["n"] += 1
        _register_one(WorkerId(f"bench-single-{uuid.uuid4().hex[:8]}-{single_counter['n']}"))

    bench("register_worker (1 fresh, WRITE)", _single)

    # (b) burst: 100 sequential fresh Registers per bench iteration.
    burst_counter = {"n": 0}

    def _burst_100():
        burst_counter["n"] += 1
        base = f"bench-burst-{uuid.uuid4().hex[:8]}-{burst_counter['n']}"
        for i in range(100):
            _register_one(WorkerId(f"{base}-{i}"))

    bench("register_worker (burst 100, WRITE)", _burst_100, min_runs=3, min_time_s=2.0)

    # (c) burst 100 under concurrent apply_heartbeats_batch contention.
    active_states = tuple(ACTIVE_TASK_STATES)
    workers = _healthy_workers(write_db)
    running_tasks_per_worker: dict[str, list[tuple[str, int]]] = {}
    for w in workers:
        wid = str(w.worker_id)
        rows = write_db.fetchall(
            "SELECT task_id, current_attempt_id FROM tasks " "WHERE current_worker_id = ? AND state IN (?, ?, ?)",
            (wid, *active_states),
        )
        if rows:
            running_tasks_per_worker[wid] = [(str(r["task_id"]), int(r["current_attempt_id"])) for r in rows]

    if not running_tasks_per_worker:
        print("  register_worker (burst 100 + hb contention)       (skipped, no running tasks)")
        return

    resource_usage_proto = job_pb2.ResourceUsage(cpu_millicores=1000, memory_mb=1024)
    snapshot_proto = job_pb2.WorkerResourceSnapshot()
    heartbeat_requests: list[HeartbeatApplyRequest] = [
        HeartbeatApplyRequest(
            worker_id=WorkerId(wid),
            worker_resource_snapshot=snapshot_proto,
            updates=[
                TaskUpdate(
                    task_id=JobName.from_wire(tid),
                    attempt_id=aid,
                    new_state=job_pb2.TASK_STATE_RUNNING,
                    resource_usage=resource_usage_proto,
                )
                for tid, aid in task_list
            ],
        )
        for wid, task_list in running_tasks_per_worker.items()
    ]

    stop_flag = threading.Event()

    def _heartbeat_loop():
        while not stop_flag.is_set():
            write_txns.apply_heartbeats_batch(heartbeat_requests)

    contention_counter = {"n": 0}

    def _burst_100_contended():
        contention_counter["n"] += 1
        base = f"bench-contend-{uuid.uuid4().hex[:8]}-{contention_counter['n']}"
        for i in range(100):
            _register_one(WorkerId(f"{base}-{i}"))

    hb_thread = threading.Thread(target=_heartbeat_loop, name="bench-heartbeat", daemon=True)
    hb_thread.start()
    try:
        bench(
            "register_worker (burst 100 + hb contention, WRITE)",
            _burst_100_contended,
            min_runs=3,
            min_time_s=2.0,
        )
    finally:
        stop_flag.set()
        hb_thread.join(timeout=10.0)


def _build_heartbeat_requests(db: ControllerDB) -> list[HeartbeatApplyRequest]:
    """Build a heartbeat batch shaped like a live provider-sync round:
    one HeartbeatApplyRequest per active worker, with one RUNNING
    resource-usage update per task currently assigned to that worker.
    """
    workers = _healthy_workers(db)
    active_states = tuple(ACTIVE_TASK_STATES)
    snapshot_proto = job_pb2.WorkerResourceSnapshot()
    usage = job_pb2.ResourceUsage(cpu_millicores=1000, memory_mb=1024)
    requests: list[HeartbeatApplyRequest] = []
    for w in workers:
        wid = str(w.worker_id)
        rows = db.fetchall(
            "SELECT task_id, current_attempt_id FROM tasks " "WHERE current_worker_id = ? AND state IN (?, ?, ?)",
            (wid, *active_states),
        )
        updates = [
            TaskUpdate(
                task_id=JobName.from_wire(str(r["task_id"])),
                attempt_id=int(r["current_attempt_id"]),
                new_state=job_pb2.TASK_STATE_RUNNING,
                resource_usage=usage,
            )
            for r in rows
        ]
        requests.append(
            HeartbeatApplyRequest(
                worker_id=WorkerId(wid),
                worker_resource_snapshot=snapshot_proto,
                updates=updates,
            )
        )
    return requests


def _build_failure_batch(db: ControllerDB, n: int) -> list[tuple[DispatchBatch, str]]:
    rows = db.fetchall(
        "SELECT worker_id, address FROM workers WHERE active = 1 LIMIT ?",
        (n,),
    )
    return [
        (
            DispatchBatch(
                worker_id=WorkerId(str(r["worker_id"])),
                worker_address=str(r["address"]) if r["address"] is not None else None,
                running_tasks=[],
            ),
            "benchmark: simulated provider-sync failure",
        )
        for r in rows
    ]


def _print_latency_distribution(name: str, latencies: list[float]) -> None:
    if not latencies:
        print(f"  {name:60s}  (no samples)")
        return
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    max_ms = latencies[-1]
    _results.append((name, p50, p95, len(latencies)))
    print(
        f"  {name:60s}  n={len(latencies):3d}  "
        f"p50={p50:7.1f}ms  p95={p95:8.1f}ms  p99={p99:8.1f}ms  max={max_ms:8.1f}ms"
    )


def _run_apply_under_contention(
    *,
    name: str,
    write_db: ControllerDB,
    write_txns: ControllerTransitions,
    heartbeat_requests: list[HeartbeatApplyRequest],
    fail_threads: int = 0,
    fail_n: int = 50,
    fail_chunk: int = 50,
    fail_interval_s: float = 2.0,
    register_threads: int = 0,
    register_burst: int = 100,
    endpoint_threads: int = 0,
    checkpoint_thread: bool = False,
    synchronous_normal: bool = False,
    duration_s: float = 8.0,
) -> None:
    """Run apply_heartbeats_batch repeatedly on a victim thread while
    configurable write storms hammer the same clone DB. Report p50/p95/p99/max
    of the victim's per-call latency.
    """
    if synchronous_normal:
        # PRAGMA synchronous can't be changed mid-connection once a tx has run,
        # so issue it on a fresh raw connection to the clone file. It persists
        # for that connection only; our ControllerDB connection is unaffected,
        # which is the point — prod can't change synchronous mid-flight either.
        _raw = sqlite3.connect(str(write_db.db_path))
        _raw.execute("PRAGMA synchronous=NORMAL")
        _raw.close()

    endpoint_tasks_rows = write_db.fetchall(
        "SELECT task_id FROM tasks WHERE state IN (1,2,3,9) AND current_attempt_id IS NOT NULL LIMIT 200"
    )
    endpoint_tasks = [JobName.from_wire(str(r["task_id"])) for r in endpoint_tasks_rows]

    stop = threading.Event()
    victim_latencies: list[float] = []
    errors: list[BaseException] = []

    def _victim():
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                write_txns.apply_heartbeats_batch(heartbeat_requests)
                victim_latencies.append((time.perf_counter() - t0) * 1000)
        except BaseException as e:
            errors.append(e)

    def _fail_storm():
        try:
            while not stop.is_set():
                failures = _build_failure_batch(write_db, fail_n)
                if failures:
                    write_txns.fail_heartbeats_batch(failures, force_remove=True, chunk_size=fail_chunk)
                stop.wait(fail_interval_s)
        except BaseException as e:
            errors.append(e)

    def _register_storm():
        try:
            meta = _build_sample_worker_metadata()
            while not stop.is_set():
                base = f"bench-contend-{uuid.uuid4().hex[:8]}"
                for i in range(register_burst):
                    write_txns.register_worker(
                        worker_id=WorkerId(f"{base}-{i}"),
                        address=f"tcp://{base}-{i}:1234",
                        metadata=meta,
                        ts=Timestamp.now(),
                        slice_id="",
                        scale_group="bench",
                    )
                    if stop.is_set():
                        break
        except BaseException as e:
            errors.append(e)

    def _endpoint_storm():
        try:
            i = 0
            while not stop.is_set():
                t = endpoint_tasks[i % len(endpoint_tasks)]
                write_txns.add_endpoint(_make_endpoint(t))
                i += 1
        except BaseException as e:
            errors.append(e)

    def _checkpoint_loop():
        try:
            while not stop.is_set():
                try:
                    write_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.OperationalError:
                    pass
                stop.wait(1.0)
        except BaseException as e:
            errors.append(e)

    threads: list[threading.Thread] = [threading.Thread(target=_victim, name="victim")]
    for _ in range(fail_threads):
        threads.append(threading.Thread(target=_fail_storm, name="fail"))
    for _ in range(register_threads):
        threads.append(threading.Thread(target=_register_storm, name="register"))
    for _ in range(endpoint_threads):
        threads.append(threading.Thread(target=_endpoint_storm, name="endpoint"))
    if checkpoint_thread:
        threads.append(threading.Thread(target=_checkpoint_loop, name="checkpoint"))

    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=30.0)

    if errors:
        print(f"  {name}: background thread error: {errors[0]!r}")
    _print_latency_distribution(name, victim_latencies)


def benchmark_apply_contention(db: ControllerDB) -> None:
    """Reproduce the production 'apply results' multi-second tail by running
    apply_heartbeats_batch as the victim under concurrent write storms.
    """
    heartbeat_requests = _build_heartbeat_requests(db)
    total_tasks = sum(len(r.updates) for r in heartbeat_requests)
    print(f"  (victim heartbeat batch: {len(heartbeat_requests)} workers, {total_tasks} tasks)")

    if not heartbeat_requests:
        print("  (skipped, no workers)")
        return

    scenarios = [
        dict(name="apply @ baseline (no contention)"),
        dict(name="apply + 1x fail_heartbeats_batch", fail_threads=1),
        dict(name="apply + 1x register_worker burst", register_threads=1),
        dict(name="apply + 1x add_endpoint storm", endpoint_threads=1),
        dict(
            name="apply + prod-mix (fail + register + endpoint)",
            fail_threads=1,
            register_threads=1,
            endpoint_threads=1,
        ),
        dict(
            name="apply + heavy storm (2f/2r/2e, chunk=200, 0.5s)",
            fail_threads=2,
            fail_chunk=200,
            fail_interval_s=0.5,
            register_threads=2,
            endpoint_threads=2,
        ),
        dict(
            name="apply + heavy + forced WAL checkpoints",
            fail_threads=2,
            fail_chunk=200,
            fail_interval_s=0.5,
            register_threads=2,
            endpoint_threads=2,
            checkpoint_thread=True,
        ),
        dict(
            name="apply + heavy + synchronous=NORMAL",
            fail_threads=2,
            fail_chunk=200,
            fail_interval_s=0.5,
            register_threads=2,
            endpoint_threads=2,
            synchronous_normal=True,
        ),
    ]

    write_db = clone_db(db)
    write_txns = ControllerTransitions(write_db)
    try:
        for scenario in scenarios:
            _run_apply_under_contention(
                write_db=write_db,
                write_txns=write_txns,
                heartbeat_requests=heartbeat_requests,
                **scenario,
            )
    finally:
        write_db.close()
        shutil.rmtree(write_db._db_dir, ignore_errors=True)


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
    type=click.Choice(["scheduling", "dashboard", "heartbeat", "endpoints", "apply_contention"]),
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
        print()

    if only_group is None or only_group == "endpoints":
        print("[endpoints]")
        benchmark_endpoints(db)
        print()

    if only_group == "apply_contention":
        print("[apply_contention]")
        benchmark_apply_contention(db)

    print_summary()
    db.close()


if __name__ == "__main__":
    main()
