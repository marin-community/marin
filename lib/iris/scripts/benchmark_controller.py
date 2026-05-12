#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Iris controller hot paths against a local checkpoint.

The script is organized by **RPC** (or per-tick worker), in roughly the order
of their production importance, so output lines map directly back to a
production endpoint. Each benchmark wraps the service-layer entry point or
its dominant internal helper, not raw SQL.

Usage:
    # Auto-download the latest archive from the Marin cluster.
    uv run python lib/iris/scripts/benchmark_controller.py

    # Use a specific local checkpoint.
    uv run python lib/iris/scripts/benchmark_controller.py --db ./controller.sqlite3

    # Run a specific group: rpcs, scheduling, polling, dashboard, endpoints,
    # apply_contention.
    uv run python lib/iris/scripts/benchmark_controller.py --only polling
"""

import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import yaml
from iris.cluster.controller.checkpoint import download_checkpoint_to_local
from iris.cluster.controller.controller import (
    _schedulable_tasks,
    compute_demand_entries,
)
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    ControllerDB,
    EndpointQuery,
)
from iris.cluster.controller.projections.endpoints import EndpointRow, EndpointsProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.reads import jobs as reads_jobs
from iris.cluster.controller.reads import scheduler as reads_scheduler
from iris.cluster.controller.reads import workers as reads_workers
from iris.cluster.controller.reads.workers import SchedulableWorker, healthy_active_workers_with_attributes  # noqa: F401
from iris.cluster.controller.scheduler import Scheduler
from iris.cluster.controller.schema import (
    endpoints_table,
    job_config_table,
    jobs_table,
    task_attempts_table,
    tasks_table,
    worker_attributes_table,
    workers_table,
)
from iris.cluster.controller.service import (
    USER_JOB_STATES,
    _parent_ids_with_children,
    _query_jobs,
    _read_job,
    _task_summaries_for_jobs,
    _tasks_for_listing,
    _worker_addresses_for_tasks,
    _worker_roster,
)
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from sqlalchemy import func, select, text, update

# ---------------------------------------------------------------------------
# Result accumulation
# ---------------------------------------------------------------------------

_results: list[tuple[str, float, float, int]] = []

_MARIN_YAML = Path(__file__).resolve().parents[1] / "config" / "marin.yaml"
DEFAULT_DB_DIR = Path("/tmp/iris_benchmark")


def _marin_remote_state_dir() -> str:
    """Resolve the canonical remote_state_dir from config/marin.yaml."""
    with _MARIN_YAML.open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg["storage"]["remote_state_dir"]


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def bench(
    name: str,
    fn: Callable[[], Any],
    *,
    reset: Callable[[], Any] | None = None,
    min_time_s: float = 2.0,
    min_runs: int = 5,
    max_runs: int = 200,
) -> None:
    """Adaptive benchmark: run fn() until ``min_time_s`` and ``min_runs``.

    ``reset`` is called between iterations (untimed) to restore mutated state.
    """
    print(f"  {name:64s}  ", end="", flush=True)
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


# ---------------------------------------------------------------------------
# DB clone
# ---------------------------------------------------------------------------

_CLONE_TABLES = [
    "jobs",
    "job_config",
    "job_workdir_files",
    "tasks",
    "task_attempts",
    "workers",
    "worker_attributes",
    "endpoints",
    "reservation_claims",
    "meta",
    "users",
    "user_budgets",
    "scaling_groups",
    "slices",
    "schema_migrations",
]


def clone_db(source: ControllerDB) -> ControllerDB:
    """Create a lightweight writable clone via ATTACH + INSERT.

    Much faster than copying the multi-GB file: only rows are copied, not
    free-page overhead. The clone preserves UNIQUE/PK/CHECK constraints by
    reusing the source DDL (CREATE TABLE AS SELECT loses them, breaking UPSERT).
    """
    clone_dir = Path(tempfile.mkdtemp(prefix="iris_bench_clone_"))
    clone_path = clone_dir / ControllerDB.DB_FILENAME
    conn = sqlite3.connect(str(clone_path))
    conn.execute("ATTACH DATABASE ? AS src", (str(source.db_path),))

    clone_tables = set(_CLONE_TABLES)
    table_ddl = conn.execute("SELECT name, sql FROM src.sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    for name, sql in table_ddl:
        if name not in clone_tables:
            continue
        conn.execute(sql)
        conn.execute(f"INSERT INTO {name} SELECT * FROM src.{name}")

    rows = conn.execute("SELECT sql FROM src.sqlite_master WHERE type='index' AND sql IS NOT NULL").fetchall()
    for row in rows:
        try:
            conn.execute(row[0])
        except sqlite3.OperationalError:
            pass  # skip indexes on tables we didn't clone
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


# ---------------------------------------------------------------------------
# Helpers shared across groups
# ---------------------------------------------------------------------------


def _build_heartbeat_requests(db: ControllerDB) -> list[HeartbeatApplyRequest]:
    """One HeartbeatApplyRequest per active worker, RUNNING update per active task."""
    health = WorkerHealthTracker()
    _seed_health(db, health)
    with db.read_snapshot() as tx:
        workers = reads_workers.healthy_active_workers_with_attributes(tx, health, _NoAttrs())
    active_states = list(ACTIVE_TASK_STATES)
    requests: list[HeartbeatApplyRequest] = []
    for w in workers:
        with db.read_snapshot() as tx:
            rows = tx.execute(
                select(tasks_table.c.task_id, tasks_table.c.current_attempt_id).where(
                    tasks_table.c.current_worker_id == w.worker_id,
                    tasks_table.c.state.in_(active_states),
                )
            ).all()
        updates = [
            TaskUpdate(
                task_id=row.task_id,
                attempt_id=int(row.current_attempt_id),
                new_state=job_pb2.TASK_STATE_RUNNING,
            )
            for row in rows
        ]
        requests.append(HeartbeatApplyRequest(worker_id=w.worker_id, updates=updates))
    return requests


class _NoAttrs:
    """Empty attrs source for benchmarks where attributes are not needed."""

    def all(self):
        return {}


def _seed_health(db: ControllerDB, health: WorkerHealthTracker) -> None:
    """Mark every persisted worker as live + healthy.

    The ``workers.active`` column was removed; activity now lives entirely
    in :class:`WorkerHealthTracker`. For benchmarks against a frozen
    snapshot we treat every persisted row as a candidate active worker.
    """
    with db.read_snapshot() as tx:
        rows = tx.execute(select(workers_table.c.worker_id)).all()
    if not rows:
        return
    health.heartbeat([WorkerId(str(r.worker_id)) for r in rows], Timestamp.now().epoch_ms())


def _build_failure_batch(db: ControllerDB, n: int) -> list[tuple[WorkerId, str | None, str]]:
    with db.read_snapshot() as tx:
        rows = tx.fetchall(select(workers_table.c.worker_id, workers_table.c.address).limit(n))
    if not rows:
        return []
    return [
        (
            WorkerId(str(r.worker_id)),
            str(r.address) if r.address is not None else None,
            "benchmark: simulated provider-sync failure",
        )
        for r in rows
    ]


def _make_endpoint(task_id: JobName, _attempt_id: int = 0) -> EndpointRow:
    return EndpointRow(
        endpoint_id=str(uuid.uuid4()),
        name=f"/bench/endpoint/{uuid.uuid4().hex[:8]}",
        address="127.0.0.1:0",
        task_id=task_id,
        metadata={"bench": "true"},
        registered_at=Timestamp.now(),
    )


def _build_sample_worker_metadata() -> job_pb2.WorkerMetadata:
    """Minimal but representative WorkerMetadata for a CPU worker."""
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


def _active_task_sample(db: ControllerDB, limit: int) -> list[tuple[JobName, int]]:
    active_states = list(ACTIVE_TASK_STATES)
    with db.read_snapshot() as tx:
        rows = tx.fetchall(
            select(tasks_table.c.task_id, tasks_table.c.current_attempt_id)
            .where(
                tasks_table.c.state.in_(active_states),
                tasks_table.c.current_attempt_id.is_not(None),
            )
            .limit(limit)
        )
    return [(row.task_id, int(row.current_attempt_id)) for row in rows]


def _has_committed_columns(db: ControllerDB) -> bool:
    with db.read_snapshot() as tx:
        rows = tx.fetchall(text("PRAGMA table_info(workers)"))
    return "committed_cpu_millicores" in {r[1] for r in rows}


# ---------------------------------------------------------------------------
# Group: RPCs (high-frequency RPC handlers, weighted by production volume)
# ---------------------------------------------------------------------------


def benchmark_rpcs(db: ControllerDB) -> None:
    """Cover the highest-volume RPCs: GetJobState, ListJobs, GetJobStatus,
    Register, RegisterEndpoint, LaunchJob, TerminateJob, UpdateTaskStatus.
    """
    health = WorkerHealthTracker()
    _seed_health(db, health)

    # ---- GetJobState (172k/day) — batched job-state lookup. ----
    with db.read_snapshot() as tx:
        rows = tx.fetchall(select(jobs_table.c.job_id).limit(50))
    job_ids = [str(r.job_id) for r in rows]
    if job_ids:
        _jnames = [JobName.from_wire(j) for j in job_ids]

        def _get_job_state():
            with db.read_snapshot() as tx:
                tx.fetchall(select(jobs_table.c.job_id, jobs_table.c.state).where(jobs_table.c.job_id.in_(_jnames)))

        bench(f"RPC: GetJobState (batch={len(job_ids)})", _get_job_state)

    # Single-id lookup is the realistic worst-case shape — many dashboards poll one.
    if job_ids:
        _single = JobName.from_wire(job_ids[0])

        def _get_job_state_single():
            with db.read_snapshot() as tx:
                tx.fetchall(select(jobs_table.c.job_id, jobs_table.c.state).where(jobs_table.c.job_id == _single))

        bench("RPC: GetJobState (single id)", _get_job_state_single)

    # ---- ListJobs (3.2k/day, p95=2.86s — known hot path). ----
    with db.read_snapshot() as q:
        page, _ = _query_jobs(
            q,
            controller_pb2.Controller.JobQuery(scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS, limit=50),
            USER_JOB_STATES,
        )
    page_ids = [j.job_id for j in page]

    def _list_jobs_full():
        with db.read_snapshot() as q:
            page, _total = _query_jobs(
                q,
                controller_pb2.Controller.JobQuery(scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS, limit=50),
                USER_JOB_STATES,
            )
            ids = [j.job_id for j in page]
            if ids:
                _task_summaries_for_jobs(q, set(ids))
                _parent_ids_with_children(q, ids)

    bench(f"RPC: ListJobs (roots, limit=50, paged={len(page_ids)})", _list_jobs_full)

    # ---- GetJobStatus (10k/day) — single-job page. ----
    if page:
        sample_job_id = page[0].job_id

        def _get_job_status():
            with db.read_snapshot() as q:
                _read_job(q, sample_job_id)
                _task_summaries_for_jobs(q, {sample_job_id})
                _parent_ids_with_children(q, [sample_job_id])

        bench("RPC: GetJobStatus", _get_job_status)

    # ---- RegisterEndpoint (128/day, p95=245ms) — write txn through add_endpoint. ----
    write_db = clone_db(db)
    write_endpoints = EndpointsProjection(write_db)
    write_worker_attrs = WorkerAttrsProjection(write_db)
    write_txns = ControllerTransitions(write_db, endpoints=write_endpoints, worker_attrs=write_worker_attrs)
    try:
        sample = _active_task_sample(write_db, limit=300)
        if sample:
            single_task, single_attempt = sample[0]

            def _register_endpoint_one():
                with write_db.transaction() as cur:
                    write_txns.add_endpoint(cur, _make_endpoint(single_task, single_attempt))

            def _reset_endpoint():
                with write_db.transaction() as _tx:
                    _tx.execute(text("DELETE FROM endpoints WHERE name LIKE '/bench/endpoint/%'"))
                write_endpoints.rehydrate()

            bench("RPC: RegisterEndpoint (1 write)", _register_endpoint_one, reset=_reset_endpoint)

        # ---- Register (192/day, p95=340ms) — fresh worker UPSERT. ----
        sample_meta = _build_sample_worker_metadata()

        register_counter = {"n": 0}

        def _register_one():
            register_counter["n"] += 1
            wid = WorkerId(f"bench-reg-{uuid.uuid4().hex[:6]}-{register_counter['n']}")
            with write_db.transaction() as cur:
                write_txns.register_worker(
                    cur,
                    worker_id=wid,
                    address=f"tcp://{wid}:1234",
                    metadata=sample_meta,
                    ts=Timestamp.now(),
                    slice_id="",
                    scale_group="bench",
                )

        bench("RPC: Register (1 fresh worker, WRITE)", _register_one)

        burst_counter = {"n": 0}

        def _register_burst_100():
            burst_counter["n"] += 1
            base = f"bench-burst-{uuid.uuid4().hex[:6]}-{burst_counter['n']}"
            for i in range(100):
                with write_db.transaction() as cur:
                    write_txns.register_worker(
                        cur,
                        worker_id=WorkerId(f"{base}-{i}"),
                        address=f"tcp://{base}-{i}:1234",
                        metadata=sample_meta,
                        ts=Timestamp.now(),
                        slice_id="",
                        scale_group="bench",
                    )

        bench(
            "RPC: Register (burst x100, WRITE)",
            _register_burst_100,
            min_runs=3,
            min_time_s=2.0,
        )

        # ---- LaunchJob (1.4k/day) — submit_job + task expansion. ----
        # Skip the auth/budget surface; benchmark the dominant write path
        # (insert + per-task expansion) as transitions.submit_job.
        replicas_set = (1, 8)
        for replicas in replicas_set:
            launch_counter = {"n": 0}

            def _launch(_n=replicas, counter=launch_counter):
                counter["n"] += 1
                jid = JobName.from_wire(f"/bench/launch-{uuid.uuid4().hex[:6]}-{counter['n']}")
                req = controller_pb2.Controller.LaunchJobRequest(
                    name=jid.to_wire(),
                    replicas=_n,
                    entrypoint=job_pb2.RuntimeEntrypoint(
                        run_command=job_pb2.CommandEntrypoint(argv=["echo", "hi"]),
                    ),
                    environment=job_pb2.EnvironmentConfig(),
                    resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
                )
                with write_db.transaction() as cur:
                    write_txns.submit_job(cur, jid, req, Timestamp.now())

            bench(f"RPC: LaunchJob (replicas={replicas}, WRITE)", _launch)

        # ---- TerminateJob (73/day, p95=274ms) — cancel_job over a subtree. ----
        # Pre-clone a dedicated DB so cancel_job has many distinct running
        # jobs to chew through without exhausting them across reset cycles.
        cancel_db = clone_db(db)
        cancel_txns = ControllerTransitions(cancel_db)
        try:
            with cancel_db.read_snapshot() as _tx:
                running_rows = _tx.fetchall(
                    select(jobs_table.c.job_id)
                    .where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING, jobs_table.c.depth == 1)
                    .limit(50)
                )
            cancel_targets = [row.job_id for row in running_rows]
            if cancel_targets:
                cancel_idx = {"i": 0}

                def _terminate():
                    if cancel_idx["i"] >= len(cancel_targets):
                        return
                    jid = cancel_targets[cancel_idx["i"]]
                    cancel_idx["i"] += 1
                    with cancel_db.transaction() as cur:
                        cancel_txns.cancel_job(cur, jid, "benchmark")

                bench(
                    f"RPC: TerminateJob (cancel_job, n={len(cancel_targets)} jobs)",
                    _terminate,
                    min_runs=min(5, len(cancel_targets)),
                    max_runs=len(cancel_targets),
                )
            else:
                print("  RPC: TerminateJob                                                 (skipped, no running jobs)")
        finally:
            cancel_db.close()
            shutil.rmtree(cancel_db._db_dir, ignore_errors=True)

        # ---- UpdateTaskStatus (4.8k/day, p95=256ms) — apply_heartbeats_batch.
        hb_requests = _build_heartbeat_requests(write_db)
        total_t = sum(len(r.updates) for r in hb_requests)
        if hb_requests:
            print(f"  (heartbeat batch: {len(hb_requests)} workers, {total_t} task updates per call)")

            def _apply_hb():
                with write_db.transaction() as cur:
                    write_txns.apply_heartbeats_batch(cur, hb_requests)

            bench(
                f"RPC: UpdateTaskStatus (apply_heartbeats_batch, w={len(hb_requests)}, t={total_t})",
                _apply_hb,
            )
    finally:
        write_db.close()
        shutil.rmtree(write_db._db_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Group: scheduling (full tick + autoscaler demand)
# ---------------------------------------------------------------------------


def benchmark_scheduling(db: ControllerDB) -> None:
    """Per-tick scheduling cost.

    Includes the new derived ``resource_usage_by_worker`` query and its
    pre-Jumbo predecessor (``SELECT committed_* FROM workers``) when the
    legacy columns are still present.
    """
    health = WorkerHealthTracker()
    _seed_health(db, health)

    # Inject pending tasks if the production snapshot has none, so we exercise
    # the scheduler's main path. Pick up to 50 running jobs and revert their
    # first 3 tasks to PENDING.
    with db.read_snapshot() as _snap:
        running_jobs = _snap.fetchall(
            select(jobs_table.c.job_id).where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING).limit(50)
        )
    pending_count = 0
    for job_row in running_jobs:
        jid = job_row.job_id
        with db.transaction() as _tx:
            _tx.execute(
                text(
                    "UPDATE tasks SET state = :new_state, current_worker_id = NULL, "
                    "current_worker_address = NULL "
                    "WHERE job_id = :jid AND state = :run_state AND rowid IN "
                    "(SELECT rowid FROM tasks WHERE job_id = :jid AND state = :run_state LIMIT 3)"
                ),
                {
                    "new_state": job_pb2.TASK_STATE_PENDING,
                    "jid": str(jid),
                    "run_state": job_pb2.TASK_STATE_RUNNING,
                },
            )
            pending_count += int(_tx.execute(text("SELECT changes() AS c")).scalar() or 0)
    pending_tasks = _schedulable_tasks(db)
    with db.read_snapshot() as _wtx:
        workers = reads_workers.healthy_active_workers_with_attributes(_wtx, health, _NoAttrs())
    print(
        f"  (scheduling shape: {len(workers)} workers, {len(pending_tasks)} pending tasks "
        f"after injecting {pending_count})"
    )

    # ---- resource_usage_by_worker (NEW): full join over unfinished
    #      worker-bound attempts. Runs every scheduling tick. ----
    def _usage_new():
        from iris.cluster.controller import db

        with db.read_snapshot(db.sa_read_engine) as snap:
            reads_scheduler.resource_usage_by_worker(snap)

    bench("Scheduling: resource_usage_by_worker (NEW derived query)", _usage_new)

    # ---- Predecessor: SELECT committed_* FROM workers. Only available
    #      pre-migration. ----
    if _has_committed_columns(db):

        def _usage_old():
            with db.read_snapshot() as _tx:
                _tx.fetchall(
                    text(
                        "SELECT worker_id, committed_cpu_millicores, committed_mem_bytes, "
                        "committed_gpu, committed_tpu FROM workers"
                    )
                )

        bench("Scheduling: workers.committed_* read (pre-Jumbo)", _usage_old)
    else:
        print("  Scheduling: workers.committed_* read (pre-Jumbo)                  (skipped, columns dropped)")

    # ---- Full tick: _read_scheduling_state-style aggregate ----
    def _state_read():
        from iris.cluster.controller import db

        _schedulable_tasks(db)
        with db.read_snapshot() as _rtx:
            ws = reads_workers.healthy_active_workers_with_attributes(_rtx, health, _NoAttrs())
        with db.read_snapshot(db.sa_read_engine) as snap:
            usage = reads_scheduler.resource_usage_by_worker(snap)
        return ws, usage

    bench("Scheduling: state read (pending+workers+usage)", _state_read)

    # ---- Autoscaler demand path (compute_demand_entries) ----
    sched = Scheduler()

    def _demand():
        compute_demand_entries(db, scheduler=sched, workers=workers, reservation_claims={})

    bench(
        f"Scheduling: compute_demand_entries (w={len(workers)}, t={len(pending_tasks)})",
        _demand,
        min_runs=3,
        min_time_s=1.0,
    )

    # ---- queue_assignments WRITE path ----
    write_db = clone_db(db)
    write_txns = ControllerTransitions(write_db)
    try:
        if pending_tasks and workers:
            worker_list = list(workers)
            sample_assignments: list[Assignment] = [
                Assignment(task_id=t.task_id, worker_id=worker_list[i % len(worker_list)].worker_id)
                for i, t in enumerate(pending_tasks[:20])
            ]
            task_ids_jn = [a.task_id for a in sample_assignments]

            def _save_state():
                with write_db.read_snapshot() as _rtx:
                    rows = _rtx.fetchall(
                        select(
                            tasks_table.c.task_id,
                            tasks_table.c.state,
                            tasks_table.c.current_attempt_id,
                            tasks_table.c.current_worker_id,
                            tasks_table.c.current_worker_address,
                            tasks_table.c.started_at_ms,
                        ).where(tasks_table.c.task_id.in_(task_ids_jn))
                    )
                return [
                    (
                        row.task_id,
                        row.state,
                        row.current_attempt_id,
                        row.current_worker_id,
                        row.current_worker_address,
                        row.started_at_ms,
                    )
                    for row in rows
                ]

            saved = _save_state()

            def _reset():
                with write_db.transaction() as cur:
                    for tid, st, aid, wid, waddr, started in saved:
                        cur.execute(
                            update(tasks_table)
                            .where(tasks_table.c.task_id == tid)
                            .values(
                                state=st,
                                current_attempt_id=aid,
                                current_worker_id=wid,
                                current_worker_address=waddr,
                                started_at_ms=started,
                            )
                        )
                        cur.execute(
                            text("DELETE FROM task_attempts WHERE task_id = :tid AND attempt_id > :aid"),
                            {"tid": str(tid), "aid": aid},
                        )

            def _do_queue():
                with write_db.transaction() as cur:
                    write_txns.queue_assignments(cur, sample_assignments)

            bench(
                f"Scheduling: queue_assignments (n={len(sample_assignments)} tasks, WRITE)",
                _do_queue,
                reset=_reset,
            )
    finally:
        write_db.close()
        shutil.rmtree(write_db._db_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Group: polling (per-tick polling-loop reads — NEW in this PR)
# ---------------------------------------------------------------------------


def benchmark_polling(db: ControllerDB) -> None:
    """Per-tick polling-loop reads.

    These are the queries that run every 250 ms in production. The
    ``reconcile_rows_for_workers`` join replaces the pre-Jumbo
    ``_poll_all_workers`` query; benchmark the new shape against the old.
    """
    health = WorkerHealthTracker()
    _seed_health(db, health)
    txns = ControllerTransitions(db, health=health)

    with db.read_snapshot() as snap:
        addresses = reads_workers.list_active_healthy(snap, health)
    worker_ids = list(addresses)
    n_workers = len(worker_ids)
    print(f"  (polling shape: {n_workers} active+healthy workers)")

    # ---- list_active_healthy: the snapshot read that drives reconcile. ----
    def _list_active_healthy():
        with db.read_snapshot() as snap:
            reads_workers.list_active_healthy(snap, health)

    bench("Polling: list_active_healthy (next reconcile batch)", _list_active_healthy)

    # ---- reconcile_rows_for_workers: per-tick batch (up to 512 ids). ----
    for batch_size in (64, min(256, n_workers), min(512, n_workers)):
        if batch_size <= 0 or batch_size > n_workers:
            continue
        ids = worker_ids[:batch_size]

        def _reconcile(_ids=ids):
            from iris.cluster.controller import db

            with db.read_snapshot(db.sa_read_engine) as snap:
                reads_scheduler.reconcile_rows_for_workers(snap, _ids)

        bench(f"Polling: reconcile_rows_for_workers (batch={batch_size})", _reconcile)

    # ---- Predecessor: pre-Jumbo _poll_all_workers SQL. Only one trip per
    #      tick covering ALL workers, regardless of batching. ----

    def _poll_all_pre_jumbo():
        # The old reconcile loop scanned every active healthy worker's running
        # tasks in one query. Use the same shape (no current_attempt_id join).
        _active = [job_pb2.TASK_STATE_ASSIGNED, job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING]
        with db.read_snapshot() as _tx:
            _tx.fetchall(
                select(
                    tasks_table.c.current_worker_id,
                    tasks_table.c.task_id,
                    tasks_table.c.current_attempt_id,
                    tasks_table.c.state,
                ).where(
                    tasks_table.c.state.in_(_active),
                    tasks_table.c.current_worker_id.is_not(None),
                )
            )

    bench("Polling: poll_all_workers (pre-Jumbo single-shot read)", _poll_all_pre_jumbo)

    # ---- run_request_template (cached): dominates ASSIGNED rows in the tick. ----
    _active_states = [job_pb2.TASK_STATE_ASSIGNED, job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING]
    with db.read_snapshot() as _rtx:
        rows = _rtx.fetchall(
            select(tasks_table.c.job_id)
            .where(
                tasks_table.c.state.in_(_active_states),
                tasks_table.c.current_worker_id.is_not(None),
            )
            .limit(64)
        )
    sample_job_ids = list({row.job_id for row in rows})
    if sample_job_ids:
        first_job = sample_job_ids[0]

        def _template_first():
            # Fresh transitions to defeat the LRU cache on every call.
            local = ControllerTransitions(db, health=health)
            with db.read_snapshot() as snap:
                local.run_request_template(snap, first_job)

        bench("Polling: run_request_template (cold, per-job build)", _template_first)

        # Warm cache: same job repeatedly.
        with db.read_snapshot() as snap:
            txns.run_request_template(snap, first_job)

        def _template_warm():
            with db.read_snapshot() as snap:
                txns.run_request_template(snap, first_job)

        bench("Polling: run_request_template (cached hit)", _template_warm)

    # ---- has_unfinished_worker_attempts: drain gate for job replacement. ----
    # Walks the parent_job_id subtree. Pick a depth=1 job with active subtree
    # tasks if possible, otherwise just any job.
    _drain_active = [job_pb2.TASK_STATE_ASSIGNED, job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING]
    with db.read_snapshot() as _dtx:
        drain_row = _dtx.fetchone(
            select(jobs_table.c.job_id)
            .join(tasks_table, tasks_table.c.job_id == jobs_table.c.job_id)
            .where(
                tasks_table.c.state.in_(_drain_active),
                tasks_table.c.current_worker_id.is_not(None),
                jobs_table.c.depth == 1,
            )
            .limit(1)
        )
    if drain_row:
        drain_jid = drain_row.job_id

        def _has_unfinished():
            from iris.cluster.controller import db

            with db.read_snapshot(db.sa_read_engine) as snap:
                reads_jobs.has_unfinished_worker_attempts(snap, drain_jid)

        bench("Polling: has_unfinished_worker_attempts (drain gate)", _has_unfinished)


# ---------------------------------------------------------------------------
# Group: dashboard (lower-volume read RPCs powering the UI)
# ---------------------------------------------------------------------------


def benchmark_dashboard(db: ControllerDB) -> None:
    """Cover ListWorkers, ListTasks, GetSchedulerState, and dashboard reads."""
    health = WorkerHealthTracker()
    _seed_health(db, health)

    def _bench_jobs_in_states():
        with db.read_snapshot() as tx:
            return tx.execute(
                select(jobs_table, job_config_table)
                .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
                .where(jobs_table.c.state.in_(list(USER_JOB_STATES)), jobs_table.c.depth == 1)
            ).all()

    bench("Dashboard: jobs_in_states (top-level)", _bench_jobs_in_states)

    # Worker roster + running map drives ListWorkers.
    def _list_workers():
        roster = _worker_roster(db)
        if roster:
            with db.read_snapshot() as tx:
                reads_scheduler.running_tasks_by_worker(tx, {w.worker_id for w in roster})

    bench(f"RPC: ListWorkers (n={len(_worker_roster(db))})", _list_workers)

    # Sample job for ListTasks.
    with db.read_snapshot() as _tx:
        sample_row = _tx.execute(
            select(jobs_table.c.job_id)
            .where(
                jobs_table.c.depth == 1, jobs_table.c.state.in_([job_pb2.JOB_STATE_RUNNING, job_pb2.JOB_STATE_PENDING])
            )
            .limit(1)
        ).first()
    if sample_row:
        sample_job = sample_row.job_id  # already decoded to JobName by JobNameType

        def _list_tasks():
            tasks = _tasks_for_listing(db, job_id=sample_job)
            _worker_addresses_for_tasks(db, tasks)

        bench("RPC: ListTasks (one job)", _list_tasks)

    # GetSchedulerState: heavy aggregation over all PENDING + RUNNING tasks.
    def _get_scheduler_state():
        with db.read_snapshot() as snap:
            snap.execute(
                select(
                    tasks_table.c.task_id,
                    tasks_table.c.job_id,
                    tasks_table.c.state,
                    tasks_table.c.current_attempt_id,
                    tasks_table.c.failure_count,
                    tasks_table.c.preemption_count,
                    tasks_table.c.max_retries_failure,
                    tasks_table.c.max_retries_preemption,
                    tasks_table.c.submitted_at_ms,
                    tasks_table.c.priority_band,
                ).where(tasks_table.c.state == job_pb2.TASK_STATE_PENDING)
            ).all()
            snap.execute(
                select(tasks_table.c.task_id, tasks_table.c.priority_band, tasks_table.c.current_worker_id).where(
                    tasks_table.c.state == job_pb2.TASK_STATE_RUNNING,
                    tasks_table.c.current_worker_id.is_not(None),
                )
            ).all()

    bench("RPC: GetSchedulerState (pending+running aggregation)", _get_scheduler_state)

    # ExecuteRawQuery: representative SELECT used by `iris query`. Use a
    # COUNT(*) over jobs as a stand-in.
    def _execute_raw_query():
        with db.read_snapshot() as q:
            q.execute(
                select(func.count()).select_from(jobs_table).where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING)
            ).scalar()

    bench("RPC: ExecuteRawQuery (COUNT over jobs)", _execute_raw_query)


# ---------------------------------------------------------------------------
# Group: endpoints (RegisterEndpoint variations)
# ---------------------------------------------------------------------------


def benchmark_endpoints(db: ControllerDB) -> None:
    """Endpoint registration, listing, and the fail-burst write storm."""
    health = WorkerHealthTracker()
    _seed_health(db, health)
    endpoints = EndpointsProjection(db)

    bench("RPC: ListEndpoints (no prefix)", lambda: endpoints.query())
    bench(
        "RPC: ListEndpoints (prefix='test')",
        lambda: endpoints.query(EndpointQuery(name_prefix="test")),
    )

    write_db = clone_db(db)
    write_endpoints = EndpointsProjection(write_db)
    write_txns = ControllerTransitions(write_db, endpoints=write_endpoints)

    try:
        sample = _active_task_sample(write_db, limit=300)
        if not sample:
            print("  (skipped, no active tasks to attach endpoints to)")
            return

        def _reset_endpoints():
            with write_db.transaction() as _tx:
                _tx.execute(text("DELETE FROM endpoints WHERE name LIKE '/bench/endpoint/%'"))
            write_endpoints.rehydrate()

        for burst_n in (50, 200):
            if len(sample) < burst_n:
                continue
            tasks_for_burst = sample[:burst_n]

            def _per_txn(tasks=tasks_for_burst):
                for t, aid in tasks:
                    with write_db.transaction() as cur:
                        write_txns.add_endpoint(cur, _make_endpoint(t, aid))

            bench(
                f"Endpoints: add_endpoint burst x{burst_n} (per-txn, WRITE)",
                _per_txn,
                reset=_reset_endpoints,
                min_runs=3,
                min_time_s=1.0,
            )

            def _one_txn(tasks=tasks_for_burst):
                with write_db.transaction() as cur:
                    for t, aid in tasks:
                        write_endpoints.add(cur, _make_endpoint(t, aid))

            bench(
                f"Endpoints: add_endpoint burst x{burst_n} (1 txn, WRITE)",
                _one_txn,
                reset=_reset_endpoints,
                min_runs=3,
                min_time_s=1.0,
            )

        # fail_workers: slice-reaping path. The exact code path the controller
        # log attributes the 29s "apply results" phase to in #5470.
        fail_n = 50
        with write_db.read_snapshot() as _wtx:
            worker_rows = _wtx.fetchall(select(workers_table.c.worker_id, workers_table.c.address).limit(fail_n))
        if len(worker_rows) >= fail_n:
            target_wids = [WorkerId(str(r.worker_id)) for r in worker_rows]
            failures: list[tuple[WorkerId, str | None, str]] = [
                (
                    WorkerId(str(r.worker_id)),
                    str(r.address) if r.address is not None else None,
                    "benchmark: simulated provider-sync failure",
                )
                for r in worker_rows
            ]
            # Save full worker rows using the raw connection (SELECT *) so
            # INSERT OR REPLACE can restore all columns without listing them.
            raw_conn = write_db.sa_read_engine.raw_connection()
            try:
                raw_conn.execute("BEGIN")
                saved_workers_raw = raw_conn.execute(
                    "SELECT * FROM workers WHERE worker_id IN ({})".format(",".join("?" for _ in target_wids)),
                    [str(w) for w in target_wids],
                ).fetchall()
                saved_tasks_raw = raw_conn.execute(
                    "SELECT task_id, state, current_attempt_id, current_worker_id, "
                    "current_worker_address, started_at_ms FROM tasks "
                    "WHERE current_worker_id IN ({})".format(",".join("?" for _ in target_wids)),
                    [str(w) for w in target_wids],
                ).fetchall()
                raw_conn.execute("ROLLBACK")
            finally:
                raw_conn.close()

            def _reset_fail(saved_w=saved_workers_raw, saved_t=saved_tasks_raw):
                if not saved_w and not saved_t:
                    return
                raw = write_db.sa_write_engine.raw_connection()
                try:
                    raw.execute("BEGIN IMMEDIATE")
                    if saved_w:
                        n_cols = len(saved_w[0])
                        ph = ",".join("?" for _ in range(n_cols))
                        raw.executemany(
                            f"INSERT OR REPLACE INTO workers VALUES ({ph})",
                            saved_w,
                        )
                    if saved_t:
                        raw.executemany(
                            "UPDATE tasks SET state=?, current_attempt_id=?, current_worker_id=?, "
                            "current_worker_address=?, started_at_ms=? WHERE task_id=?",
                            [(r[1], r[2], r[3], r[4], r[5], r[0]) for r in saved_t],
                        )
                    raw.execute("COMMIT")
                except Exception:
                    raw.execute("ROLLBACK")
                    raise
                finally:
                    raw.close()

            bench(
                f"Endpoints: fail_workers x{fail_n} (slice-reap, WRITE)",
                lambda f=failures: write_txns.fail_workers(f),
                reset=_reset_fail,
                min_runs=3,
                min_time_s=1.0,
            )
    finally:
        write_db.close()
        shutil.rmtree(write_db._db_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Group: apply_contention (heartbeat tail latency under write storms)
# ---------------------------------------------------------------------------


def _print_latency_distribution(name: str, latencies: list[float]) -> None:
    if not latencies:
        print(f"  {name:64s}  (no samples)")
        return
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    max_ms = latencies[-1]
    _results.append((name, p50, p95, len(latencies)))
    print(
        f"  {name:64s}  n={len(latencies):3d}  "
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
    duration_s: float = 6.0,
) -> None:
    """Run apply_heartbeats_batch on a victim thread while configurable write
    storms hammer the same DB. Reports p50/p95/p99/max of the victim.
    """
    _active_states_contend = list(ACTIVE_TASK_STATES)
    with write_db.read_snapshot() as _ctx:
        endpoint_tasks_rows = _ctx.fetchall(
            select(tasks_table.c.task_id, tasks_table.c.current_attempt_id)
            .where(
                tasks_table.c.state.in_(_active_states_contend),
                tasks_table.c.current_attempt_id.is_not(None),
            )
            .limit(200)
        )
    endpoint_tasks = [(row.task_id, int(row.current_attempt_id)) for row in endpoint_tasks_rows]

    stop = threading.Event()
    victim_latencies: list[float] = []
    errors: list[BaseException] = []

    def _victim():
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                with write_db.transaction() as cur:
                    write_txns.apply_heartbeats_batch(cur, heartbeat_requests)
                victim_latencies.append((time.perf_counter() - t0) * 1000)
        except BaseException as e:
            errors.append(e)

    def _fail_storm():
        try:
            while not stop.is_set():
                failures = _build_failure_batch(write_db, fail_n)
                if failures:
                    write_txns.fail_workers(failures, chunk_size=fail_chunk)
                stop.wait(fail_interval_s)
        except BaseException as e:
            errors.append(e)

    def _register_storm():
        try:
            meta = _build_sample_worker_metadata()
            while not stop.is_set():
                base = f"bench-contend-{uuid.uuid4().hex[:8]}"
                for i in range(register_burst):
                    with write_db.transaction() as cur:
                        write_txns.register_worker(
                            cur,
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
                t, aid = endpoint_tasks[i % len(endpoint_tasks)]
                with write_db.transaction() as cur:
                    write_txns.add_endpoint(cur, _make_endpoint(t, aid))
                i += 1
        except BaseException as e:
            errors.append(e)

    threads: list[threading.Thread] = [threading.Thread(target=_victim, name="victim")]
    for _ in range(fail_threads):
        threads.append(threading.Thread(target=_fail_storm, name="fail"))
    for _ in range(register_threads):
        threads.append(threading.Thread(target=_register_storm, name="register"))
    for _ in range(endpoint_threads):
        threads.append(threading.Thread(target=_endpoint_storm, name="endpoint"))

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
    """Reproduce the production tail when apply_heartbeats_batch contends
    with provider-sync failure storms and other write RPCs.
    """
    heartbeat_requests = _build_heartbeat_requests(db)
    total_tasks = sum(len(r.updates) for r in heartbeat_requests)
    print(f"  (victim heartbeat batch: {len(heartbeat_requests)} workers, {total_tasks} tasks)")

    if not heartbeat_requests:
        print("  (skipped, no workers)")
        return

    scenarios = [
        dict(name="Contention: apply @ baseline (no contention)"),
        dict(name="Contention: apply + fail_workers", fail_threads=1),
        dict(name="Contention: apply + register burst", register_threads=1),
        dict(name="Contention: apply + add_endpoint storm", endpoint_threads=1),
        dict(
            name="Contention: apply + prod-mix (fail+reg+ep)",
            fail_threads=1,
            register_threads=1,
            endpoint_threads=1,
        ),
        dict(
            name="Contention: apply + heavy storm (2f/2r/2e, chunk=200)",
            fail_threads=2,
            fail_chunk=200,
            fail_interval_s=0.5,
            register_threads=2,
            endpoint_threads=2,
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


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------


def _count(db: ControllerDB, tbl) -> int:
    with db.read_snapshot() as tx:
        return int(tx.execute(select(func.count()).select_from(tbl)).scalar() or 0)


def print_db_stats(db: ControllerDB) -> None:
    """Print scale context once at the top so output is self-describing."""
    counts: dict[str, int] = {
        "jobs": _count(db, jobs_table),
        "tasks": _count(db, tasks_table),
        "task_attempts": _count(db, task_attempts_table),
        "workers": _count(db, workers_table),
        "worker_attributes": _count(db, worker_attributes_table),
        "endpoints": _count(db, endpoints_table),
    }

    with db.read_snapshot() as tx:
        running_jobs = int(
            tx.execute(
                select(func.count()).select_from(jobs_table).where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING)
            ).scalar()
            or 0
        )
        running_tasks = int(
            tx.execute(
                select(func.count()).select_from(tasks_table).where(tasks_table.c.state == job_pb2.TASK_STATE_RUNNING)
            ).scalar()
            or 0
        )
        pending_tasks = int(
            tx.execute(
                select(func.count()).select_from(tasks_table).where(tasks_table.c.state == job_pb2.TASK_STATE_PENDING)
            ).scalar()
            or 0
        )
        live_attempts = int(
            tx.execute(
                select(func.count())
                .select_from(task_attempts_table)
                .where(
                    task_attempts_table.c.worker_id.is_not(None),
                    task_attempts_table.c.finished_at_ms.is_(None),
                )
            ).scalar()
            or 0
        )
    state_counts = {
        "persisted workers": counts["workers"],
        "running jobs": running_jobs,
        "running tasks": running_tasks,
        "pending tasks": pending_tasks,
        "live attempts (worker-bound, unfinished)": live_attempts,
    }

    print("Scale context:")
    print("  rows:    " + ", ".join(f"{t}={c}" for t, c in counts.items()))
    print("  active:  " + ", ".join(f"{k}={v}" for k, v in state_counts.items()))
    print()


def print_summary() -> None:
    print("\n" + "=" * 100)
    print(f"  {'Benchmark':64s}  {'p50':>10s}  {'p95':>10s}  {'n':>5s}")
    print("-" * 100)
    for name, p50, p95, n in _results:
        print(f"  {name:64s}  {p50:8.1f}ms  {p95:8.1f}ms  {n:5d}")
    print("=" * 100)


def _ensure_db(db_path: Path | None) -> Path:
    """Download latest archive if no local DB is provided."""
    if db_path is not None:
        return db_path
    db_dir = DEFAULT_DB_DIR
    db_file = db_dir / ControllerDB.DB_FILENAME
    if db_file.exists():
        print(f"Using cached DB at {db_file}")
        return db_file
    remote = _marin_remote_state_dir()
    print(f"Downloading latest controller archive from {remote} ...")
    db_dir.mkdir(parents=True, exist_ok=True)
    if not download_checkpoint_to_local(remote, db_dir):
        raise click.ClickException("No checkpoint found in remote state dir")
    print(f"Downloaded to {db_file}\n")
    return db_file


_GROUPS = ("rpcs", "scheduling", "polling", "dashboard", "endpoints", "apply_contention")


@click.command()
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a controller.sqlite3 checkpoint. Omit to auto-download.",
)
@click.option(
    "--only",
    "only_group",
    type=click.Choice(_GROUPS),
    help="Run only this group.",
)
def main(db_path: Path | None, only_group: str | None) -> None:
    """Benchmark Iris controller hot paths against a local checkpoint."""
    _results.clear()
    db_path = _ensure_db(db_path)
    db = ControllerDB(db_dir=db_path.parent)

    # Note whether the legacy committed_* columns are present BEFORE we apply
    # this branch's migrations: if so, we want to benchmark the legacy column
    # read against the new derived query.
    legacy_columns = _has_committed_columns(db)

    db.apply_migrations()
    print(f"Benchmarking {db_path}")
    print(f"  legacy committed_* columns at open time: {legacy_columns} (post-migration: {_has_committed_columns(db)})")
    print()
    print_db_stats(db)

    groups: list[tuple[str, Callable[[ControllerDB], None]]] = [
        ("rpcs", benchmark_rpcs),
        ("scheduling", benchmark_scheduling),
        ("polling", benchmark_polling),
        ("dashboard", benchmark_dashboard),
        ("endpoints", benchmark_endpoints),
        ("apply_contention", benchmark_apply_contention),
    ]
    for name, fn in groups:
        if only_group is not None and only_group != name:
            continue
        print(f"[{name}]")
        fn(db)
        print()

    print_summary()
    db.close()


if __name__ == "__main__":
    main()
