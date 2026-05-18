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

import dataclasses
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import yaml
from iris.cluster.controller import reads
from iris.cluster.controller.checkpoint import download_checkpoint_to_local
from iris.cluster.controller.controller import (
    Controller,
    ControllerConfig,
    compute_demand_entries,
)
from iris.cluster.controller.controller import (
    _pending_tasks_with_jobs as _schedulable_tasks,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.db import Tx as _Tx
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.managed_thread import ThreadContainer

# Branch removed Tx.fetchall/fetchone; restore for this benchmark script.
if not hasattr(_Tx, "fetchall"):
    _Tx.fetchall = lambda self, stmt, params=None: self.execute(stmt, params).all()
    _Tx.fetchone = lambda self, stmt, params=None: self.execute(stmt, params).first()
from iris.cluster.controller.projections.endpoints import EndpointQuery, EndpointRow, EndpointsProjection
from iris.cluster.controller.reads import SchedulableWorker, healthy_active_workers_with_attributes  # noqa: F401
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
    _tasks_for_listing,
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
from iris.rpc import controller_pb2, job_pb2, query_pb2, worker_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.version import client_revision_date
from rigging.timing import Timestamp
from sqlalchemy import func, select, text, update


def _worker_addresses_for_tasks(db, tasks):
    """Inlined for the branch: service.py removed this helper."""
    worker_ids = {t.current_worker_id for t in tasks if getattr(t, "current_worker_id", None)}
    if not worker_ids:
        return {}
    with db.read_snapshot() as q:
        rows = q.execute(
            select(workers_table.c.worker_id, workers_table.c.address).where(
                workers_table.c.worker_id.in_(list(worker_ids))
            )
        ).all()
    return {r.worker_id: r.address for r in rows}


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
# RPC harness: real Controller(dry_run=True) + Connect sync client
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal TaskProvider that satisfies Controller's wiring without making
    real cluster calls. Mirrors tests/cluster/controller/conftest.py:FakeProvider.

    Used in combination with ``dry_run=True`` so the polling loop's reconcile
    short-circuits before it would ever call the provider — but we still need a
    real provider object to satisfy the constructor's type contract.
    """

    def get_process_status(self, worker_id, address, request):
        raise RuntimeError("fake provider")

    def on_worker_failed(self, worker_id, address):
        pass

    def profile_task(self, address, request, timeout_ms):
        raise RuntimeError("fake provider")

    def ping_workers(self, workers):
        return []

    def reconcile_workers(self, plans):
        # Same shape the test-suite FakeProvider returns. Only exercised if
        # someone disables dry_run on the harness.
        from iris.cluster.controller.worker_provider import WorkerReconcileResult

        return [
            WorkerReconcileResult(
                worker_id=plan.worker_id,
                start_response=worker_pb2.Worker.StartTasksResponse() if plan.start_tasks else None,
                start_error=None,
                poll_updates=[],
                poll_error=None,
            )
            for plan in plans
        ]

    def close(self):
        pass


class RpcHarness:
    """Out-of-process Controller(dry_run=True) + sync Connect clients.

    Spawns ``benchmark_controller.py serve --db-path X --state-dir Y`` as a
    child process. The child prints ``READY port=N`` to stdout once the HTTP
    server is up. Benchmark threads in the parent then hit the child over
    real TCP — no shared GIL, no shared event loop. Mirrors production
    where each task pushes from its own interpreter.

    ``dry_run=True`` in the child gates every side-effecting write the
    scheduler/polling/autoscaler loops would otherwise emit — so the cloned
    snapshot stays stable across benchmark iterations, but RPCs still
    contend with the scheduler's reads and the loop's CPU cost, matching
    production shape.

    ``on_loop`` toggles the SetTaskStatusText ``@on_loop`` attribute inside
    the child before the controller is built so the dispatch mode is locked
    in for the lifetime of the harness.
    """

    def __init__(
        self,
        db_path: Path,
        tmp: Path,
        *,
        startup_timeout_s: float = 30.0,
        on_loop: bool = False,
    ) -> None:
        tmp.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "serve",
            "--db-path",
            str(db_path),
            "--state-dir",
            str(tmp / "server-state"),
        ]
        if on_loop:
            cmd.append("--on-loop")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            # New session so SIGINT to the parent (Ctrl-C) doesn't immediately
            # SIGINT the child; we send SIGTERM explicitly in close().
            preexec_fn=os.setsid if os.name == "posix" else None,
        )
        port: int | None = None
        deadline = time.time() + startup_timeout_s
        while time.time() < deadline:
            line = self._proc.stdout.readline() if self._proc.stdout else ""
            if not line:
                rc = self._proc.poll()
                if rc is not None:
                    raise RuntimeError(f"serve subprocess exited before READY (rc={rc})")
                continue
            line = line.rstrip()
            if line.startswith("READY port="):
                port = int(line.split("=", 1)[1])
                break
            print(f"[serve] {line}")
        if port is None:
            self._proc.send_signal(signal.SIGTERM)
            raise RuntimeError(f"serve subprocess did not send READY in {startup_timeout_s}s")

        # Drain remaining stdout in a daemon thread so the pipe doesn't fill
        # and block the child mid-run.
        self._drain_thread = threading.Thread(target=self._drain_stdout, name="serve-stdout-drain", daemon=True)
        self._drain_thread.start()

        self.url = f"http://127.0.0.1:{port}"
        self.client = ControllerServiceClientSync(address=self.url, timeout_ms=30000)

    def _drain_stdout(self) -> None:
        if self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            sys.stdout.write(f"[serve] {line}")

    def make_client(self) -> ControllerServiceClientSync:
        return ControllerServiceClientSync(address=self.url, timeout_ms=30000)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Scenario abstractions
# ---------------------------------------------------------------------------

_MAX_PER_THREAD_RPS = 200.0

_SCENARIO_SCALE: float = 1.0
_SCENARIO_DURATION: float = 60.0


@dataclasses.dataclass
class RpcLoad:
    name: str
    target_rps: float
    invoke: Callable[[ControllerServiceClientSync], None]
    n_clients_min: int = 1
    """Minimum number of client threads / connections for this load.

    The runner picks ``max(n_clients_min, ceil(target_rps / MAX_PER_THREAD_RPS))``.
    Set this above 1 to model loads where the *connection count* matters
    independently of throughput — e.g. SetTaskStatusText in production is
    pushed by ~200 task processes, each holding its own connection. A single
    thread doing 316 rps doesn't expose the same concurrency on the server.
    """


@dataclasses.dataclass
class Scenario:
    name: str
    loads: list[RpcLoad]
    duration_s: float


class ScenarioRunner:
    def __init__(self, harness: "RpcHarness", scenario: Scenario) -> None:
        self.harness = harness
        self.scenario = scenario

    def run(self) -> dict[str, dict]:
        stop = threading.Event()
        per_load_results: dict[str, dict] = {}
        lock = threading.Lock()
        all_threads: list[threading.Thread] = []

        for load in self.scenario.loads:
            n_threads = max(load.n_clients_min, math.ceil(load.target_rps / _MAX_PER_THREAD_RPS))
            per_thread_rps = load.target_rps / n_threads
            thread_latencies: list[list[float]] = [[] for _ in range(n_threads)]
            thread_errors: list[list[Exception]] = [[] for _ in range(n_threads)]

            def make_worker(
                idx: int, tl: list[float], te: list[Exception], lload: RpcLoad, ptr: float
            ) -> threading.Thread:
                def worker() -> None:
                    client = self.harness.make_client()
                    interval = 1.0 / ptr
                    next_call = time.perf_counter()
                    try:
                        while not stop.is_set():
                            delay = next_call - time.perf_counter()
                            if delay > 0:
                                if stop.wait(min(delay, 0.5)):
                                    break
                                continue
                            t0 = time.perf_counter()
                            try:
                                lload.invoke(client)
                            except Exception as e:
                                te.append(e)
                            tl.append((time.perf_counter() - t0) * 1000.0)
                            next_call += interval
                    finally:
                        try:
                            client.close()
                        except Exception:
                            pass

                return threading.Thread(target=worker, name=f"scenario-{lload.name}-{idx}", daemon=True)

            threads_for_load = [
                make_worker(i, thread_latencies[i], thread_errors[i], load, per_thread_rps) for i in range(n_threads)
            ]
            all_threads.extend(threads_for_load)

            with lock:
                per_load_results[load.name] = {
                    "_latencies": thread_latencies,
                    "_errors": thread_errors,
                    "_n_threads": n_threads,
                }

        t_start = time.perf_counter()
        for t in all_threads:
            t.start()
        stop.wait(self.scenario.duration_s)
        stop.set()
        for t in all_threads:
            t.join(timeout=30.0)
        elapsed = time.perf_counter() - t_start

        results: dict[str, dict] = {}
        for load in self.scenario.loads:
            raw = per_load_results[load.name]
            all_latencies = [v for sub in raw["_latencies"] for v in sub]
            all_errors = [e for sub in raw["_errors"] for e in sub]
            n = len(all_latencies)
            if n == 0:
                results[load.name] = {
                    "n": 0,
                    "errors": len(all_errors),
                    "actual_rps": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "max": 0.0,
                }
            else:
                p50, p95, p99, _p999, mx = _percentiles_ms(all_latencies)
                results[load.name] = {
                    "n": n,
                    "errors": len(all_errors),
                    "actual_rps": n / elapsed,
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "max": mx,
                }
        return results


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
        workers = reads.healthy_active_workers_with_attributes(tx, health, _NoAttrs())
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
# === Section: RPC load factories ===
# ---------------------------------------------------------------------------


def load_set_task_status_text(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    task_ids = [f"/bench/sts/{i:04d}" for i in range(50)]
    counter = {"n": 0}
    payload = "x" * 256
    short = payload[:32]

    def invoke(client: ControllerServiceClientSync) -> None:
        tid = task_ids[counter["n"] % len(task_ids)]
        counter["n"] += 1
        client.set_task_status_text(
            job_pb2.SetTaskStatusTextRequest(
                task_id=tid,
                status_text_detail_md=payload,
                status_text_summary_md=short,
            )
        )

    # Production reality: ~200 task processes each push their own status,
    # so the connection count on the server matters as much as the throughput.
    return RpcLoad(name="SetTaskStatusText", target_rps=rps, invoke=invoke, n_clients_min=200)


def load_get_job_state(harness: RpcHarness, db: ControllerDB, rps: float, batch: int = 1) -> RpcLoad | None:
    with db.read_snapshot() as tx:
        rows = tx.fetchall(select(jobs_table.c.job_id).limit(50))
    if not rows:
        return None
    job_ids = [str(r.job_id) for r in rows[:batch]]
    req = controller_pb2.Controller.GetJobStateRequest(job_ids=job_ids)

    def invoke(client: ControllerServiceClientSync) -> None:
        client.get_job_state(req)

    return RpcLoad(name="GetJobState", target_rps=rps, invoke=invoke)


def load_update_task_status(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    hb_requests = _build_heartbeat_requests(db)
    if not hb_requests:
        return None
    mid = hb_requests[len(hb_requests) // 2]
    rpc_req = controller_pb2.Controller.UpdateTaskStatusRequest(
        worker_id=str(mid.worker_id),
        updates=[
            job_pb2.WorkerTaskStatus(
                task_id=u.task_id.to_wire(),
                attempt_id=u.attempt_id,
                state=u.new_state,
            )
            for u in mid.updates
        ],
    )

    def invoke(client: ControllerServiceClientSync) -> None:
        client.update_task_status(rpc_req)

    return RpcLoad(name="UpdateTaskStatus", target_rps=rps, invoke=invoke)


def load_register_endpoint(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    sample = _active_task_sample(db, limit=300)
    if not sample:
        return None
    task_id, attempt_id = sample[0]
    lock = threading.Lock()
    counter = {"n": 0}

    def invoke(client: ControllerServiceClientSync) -> None:
        with lock:
            n = counter["n"]
            counter["n"] += 1
        client.register_endpoint(
            controller_pb2.Controller.RegisterEndpointRequest(
                name=f"/bench/endpoint/{n:010d}",
                address="127.0.0.1:0",
                task_id=str(task_id),
                attempt_id=int(attempt_id),
                metadata={"bench": "true"},
            )
        )

    return RpcLoad(name="RegisterEndpoint", target_rps=rps, invoke=invoke)


def load_register(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    sample_meta = _build_sample_worker_metadata()
    lock = threading.Lock()
    counter = {"n": 0}

    def invoke(client: ControllerServiceClientSync) -> None:
        with lock:
            n = counter["n"]
            counter["n"] += 1
        wid = f"bench-reg-{n:010d}"
        client.register(
            controller_pb2.Controller.RegisterRequest(
                worker_id=wid,
                address=f"tcp://{wid}:1234",
                metadata=sample_meta,
                slice_id="",
                scale_group="bench",
            )
        )

    return RpcLoad(name="Register", target_rps=rps, invoke=invoke)


def load_launch_job(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    client_date = client_revision_date()
    lock = threading.Lock()
    counter = {"n": 0}

    def invoke(client: ControllerServiceClientSync) -> None:
        with lock:
            n = counter["n"]
            counter["n"] += 1
        jid = JobName.from_wire(f"/bench/launch-scenario-{n:010d}")
        client.launch_job(
            controller_pb2.Controller.LaunchJobRequest(
                name=jid.to_wire(),
                replicas=1,
                entrypoint=job_pb2.RuntimeEntrypoint(
                    run_command=job_pb2.CommandEntrypoint(argv=["echo", "hi"]),
                ),
                environment=job_pb2.EnvironmentConfig(),
                resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
                client_revision_date=client_date,
            )
        )

    return RpcLoad(name="LaunchJob", target_rps=rps, invoke=invoke)


def load_terminate_job(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    with db.read_snapshot() as tx:
        rows = tx.fetchall(
            select(jobs_table.c.job_id)
            .where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING, jobs_table.c.depth == 1)
            .limit(50)
        )
    if not rows:
        return None
    targets = [str(r.job_id) for r in rows]
    counter = {"i": 0}

    def invoke(client: ControllerServiceClientSync) -> None:
        jid = targets[counter["i"] % len(targets)]
        counter["i"] += 1
        client.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=jid))

    return RpcLoad(name="TerminateJob", target_rps=rps, invoke=invoke)


def load_list_jobs(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    req = controller_pb2.Controller.ListJobsRequest(
        query=controller_pb2.Controller.JobQuery(
            scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS,
            limit=50,
        )
    )

    def invoke(client: ControllerServiceClientSync) -> None:
        client.list_jobs(req)

    return RpcLoad(name="ListJobs", target_rps=rps, invoke=invoke)


def load_list_endpoints(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    req = controller_pb2.Controller.ListEndpointsRequest()

    def invoke(client: ControllerServiceClientSync) -> None:
        client.list_endpoints(req)

    return RpcLoad(name="ListEndpoints", target_rps=rps, invoke=invoke)


def load_list_workers(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    req = controller_pb2.Controller.ListWorkersRequest()

    def invoke(client: ControllerServiceClientSync) -> None:
        client.list_workers(req)

    return RpcLoad(name="ListWorkers", target_rps=rps, invoke=invoke)


def load_list_tasks(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    with db.read_snapshot() as tx:
        rows = tx.fetchall(select(jobs_table.c.job_id).limit(1))
    if not rows:
        return None
    job_id = str(rows[0].job_id)
    req = controller_pb2.Controller.ListTasksRequest(job_id=job_id)

    def invoke(client: ControllerServiceClientSync) -> None:
        client.list_tasks(req)

    return RpcLoad(name="ListTasks", target_rps=rps, invoke=invoke)


def load_get_job_status(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    with db.read_snapshot() as tx:
        rows = tx.fetchall(select(jobs_table.c.job_id).limit(1))
    if not rows:
        return None
    job_id = str(rows[0].job_id)
    req = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id)

    def invoke(client: ControllerServiceClientSync) -> None:
        client.get_job_status(req)

    return RpcLoad(name="GetJobStatus", target_rps=rps, invoke=invoke)


def load_execute_raw_query(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    req = query_pb2.RawQueryRequest(sql="SELECT COUNT(*) FROM jobs")

    def invoke(client: ControllerServiceClientSync) -> None:
        client.execute_raw_query(req)

    return RpcLoad(name="ExecuteRawQuery", target_rps=rps, invoke=invoke)


def load_get_scheduler_state(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    req = controller_pb2.Controller.GetSchedulerStateRequest()

    def invoke(client: ControllerServiceClientSync) -> None:
        client.get_scheduler_state(req)

    return RpcLoad(name="GetSchedulerState", target_rps=rps, invoke=invoke)


def load_list_users(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    req = controller_pb2.Controller.ListUsersRequest()

    def invoke(client: ControllerServiceClientSync) -> None:
        client.list_users(req)

    return RpcLoad(name="ListUsers", target_rps=rps, invoke=invoke)


def load_get_autoscaler_status(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    req = controller_pb2.Controller.GetAutoscalerStatusRequest()

    def invoke(client: ControllerServiceClientSync) -> None:
        client.get_autoscaler_status(req)

    return RpcLoad(name="GetAutoscalerStatus", target_rps=rps, invoke=invoke)


def load_get_process_status(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    roster = _worker_roster(db)
    if not roster:
        return None
    worker_id = str(next(iter(roster)))
    req = job_pb2.GetProcessStatusRequest(max_log_lines=10, target=worker_id)

    def invoke(client: ControllerServiceClientSync) -> None:
        client.get_process_status(req)

    return RpcLoad(name="GetProcessStatus", target_rps=rps, invoke=invoke)


PRODUCTION_MIX_RPS: dict[str, float] = {
    "SetTaskStatusText": 316.0,
    "GetJobState": 7.85,
    "UpdateTaskStatus": 1.71,
    "ListEndpoints": 0.90,
    "GetJobStatus": 0.73,
    "RegisterEndpoint": 0.45,
    "ListJobs": 0.29,
    "Register": 0.16,
    "ExecuteRawQuery": 0.10,
    "LaunchJob": 0.052,
    "ListWorkers": 0.032,
    "ListTasks": 0.018,
    "GetProcessStatus": 0.01,
    "TerminateJob": 0.01,
    "GetSchedulerState": 0.01,
    "ListUsers": 0.005,
    "GetAutoscalerStatus": 0.005,
}

_FACTORY_BY_NAME: dict[str, Callable[..., RpcLoad | None]] = {
    "SetTaskStatusText": load_set_task_status_text,
    "GetJobState": load_get_job_state,
    "UpdateTaskStatus": load_update_task_status,
    "ListEndpoints": load_list_endpoints,
    "GetJobStatus": load_get_job_status,
    "RegisterEndpoint": load_register_endpoint,
    "ListJobs": load_list_jobs,
    "Register": load_register,
    "ExecuteRawQuery": load_execute_raw_query,
    "LaunchJob": load_launch_job,
    "ListWorkers": load_list_workers,
    "ListTasks": load_list_tasks,
    "GetProcessStatus": load_get_process_status,
    "TerminateJob": load_terminate_job,
    "GetSchedulerState": load_get_scheduler_state,
    "ListUsers": load_list_users,
    "GetAutoscalerStatus": load_get_autoscaler_status,
}


def build_production_scenario(
    harness: RpcHarness,
    db: ControllerDB,
    *,
    scale: float = 1.0,
    duration_s: float = 60.0,
) -> Scenario:
    loads: list[RpcLoad] = []
    for name, base_rps in PRODUCTION_MIX_RPS.items():
        factory = _FACTORY_BY_NAME[name]
        target_rps = base_rps * scale
        load = factory(harness, db, rps=target_rps)
        if load is None:
            print(f"  WARNING: {name} skipped — factory returned None (no suitable snapshot sample)")
        else:
            loads.append(load)
    return Scenario(name="production_mix", loads=loads, duration_s=duration_s)


# ---------------------------------------------------------------------------
# Group: RPCs (high-frequency RPC handlers, weighted by production volume)
# ---------------------------------------------------------------------------


def _bench_load(name: str, harness: RpcHarness, load: RpcLoad | None, **bench_kwargs) -> None:
    if load is None:
        print(f"  {name:64s}  (skipped, no suitable snapshot sample)")
        return
    client = harness.make_client()
    bench(name, lambda: load.invoke(client), **bench_kwargs)


def benchmark_rpcs(db: ControllerDB) -> None:
    """Cover the highest-volume RPCs: GetJobState, ListJobs, GetJobStatus,
    Register, RegisterEndpoint, LaunchJob, TerminateJob, UpdateTaskStatus.

    All RPCs go through ``RpcHarness`` (real Controller in dry_run + Connect
    HTTP). Numbers include serialization, ASGI dispatch, AsyncServiceAdapter,
    and any contention from the controller's read-only scheduling loop.
    """
    write_db = clone_db(db)
    harness_dir = Path(tempfile.mkdtemp(prefix="iris_bench_harness_"))
    db_dir = write_db._db_dir
    db_path = db_dir / ControllerDB.DB_FILENAME
    # Hand the DB to the subprocess controller; reopen a read-only handle
    # in the parent for sample queries (WAL allows concurrent readers).
    write_db.close()
    harness = RpcHarness(db_path, harness_dir)
    write_db = ControllerDB(db_dir=db_dir)
    try:
        # ---- GetJobState (172k/day) — batched job-state lookup. ----
        with write_db.read_snapshot() as tx:
            rows = tx.fetchall(select(jobs_table.c.job_id).limit(50))
        job_ids = [str(r.job_id) for r in rows]
        if job_ids:
            _bench_load(
                f"RPC: GetJobState (batch={len(job_ids)})",
                harness,
                load_get_job_state(harness, write_db, rps=0, batch=len(job_ids)),
            )
            # Single-id lookup is the realistic worst-case shape — many dashboards poll one.
            _bench_load(
                "RPC: GetJobState (single id)",
                harness,
                load_get_job_state(harness, write_db, rps=0, batch=1),
            )

        # ---- ListJobs (3.2k/day, p95=2.86s — known hot path). ----
        list_load = load_list_jobs(harness, write_db, rps=0)
        if list_load is not None:
            tmp_client = harness.make_client()
            first_resp = tmp_client.list_jobs(
                controller_pb2.Controller.ListJobsRequest(
                    query=controller_pb2.Controller.JobQuery(
                        scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS,
                        limit=50,
                    )
                )
            )
            page_ids = [j.job_id for j in first_resp.jobs]
            bench_client = harness.make_client()
            bench(
                f"RPC: ListJobs (roots, limit=50, paged={len(page_ids)})",
                lambda: list_load.invoke(bench_client),
            )

            # ---- GetJobStatus (10k/day) — single-job page. ----
            if page_ids:
                status_load = load_get_job_status(harness, write_db, rps=0)
                if status_load is not None:
                    bench(
                        "RPC: GetJobStatus",
                        lambda: status_load.invoke(bench_client),
                    )

        # ---- RegisterEndpoint (128/day, p95=245ms). ----
        # The factory mints a unique endpoint name per call, so the endpoints
        # table grows during the bench (a few hundred rows for a short run —
        # not material to the measurement).
        sample = _active_task_sample(write_db, limit=300)
        if sample:
            ep_load = load_register_endpoint(harness, write_db, rps=0)
            _bench_load("RPC: RegisterEndpoint (1 write)", harness, ep_load)

        # ---- Register (192/day, p95=340ms) — fresh worker UPSERT. ----
        _bench_load("RPC: Register (1 fresh worker, WRITE)", harness, load_register(harness, write_db, rps=0))

        # _register_burst_100 is bespoke: it batches 100 sequential calls into one timed unit.
        sample_meta = _build_sample_worker_metadata()
        burst_client = harness.make_client()
        burst_counter = {"n": 0}

        def _register_burst_100():
            burst_counter["n"] += 1
            base = f"bench-burst-{uuid.uuid4().hex[:6]}-{burst_counter['n']}"
            for i in range(100):
                burst_client.register(
                    controller_pb2.Controller.RegisterRequest(
                        worker_id=f"{base}-{i}",
                        address=f"tcp://{base}-{i}:1234",
                        metadata=sample_meta,
                        slice_id="",
                        scale_group="bench",
                    )
                )

        bench(
            "RPC: Register (burst x100, WRITE)",
            _register_burst_100,
            min_runs=3,
            min_time_s=2.0,
        )

        # ---- LaunchJob (1.4k/day) — full handler path including budget guard. ----
        _bench_load("RPC: LaunchJob (replicas=1, WRITE)", harness, load_launch_job(harness, write_db, rps=0))

        # replicas=8 variant: factory always uses replicas=1; build this one bespoke.
        client_date = client_revision_date()
        launch8_client = harness.make_client()
        launch8_counter = {"n": 0}

        def _launch8():
            launch8_counter["n"] += 1
            jid = JobName.from_wire(f"/bench/launch8-{uuid.uuid4().hex[:6]}-{launch8_counter['n']}")
            launch8_client.launch_job(
                controller_pb2.Controller.LaunchJobRequest(
                    name=jid.to_wire(),
                    replicas=8,
                    entrypoint=job_pb2.RuntimeEntrypoint(
                        run_command=job_pb2.CommandEntrypoint(argv=["echo", "hi"]),
                    ),
                    environment=job_pb2.EnvironmentConfig(),
                    resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
                    client_revision_date=client_date,
                )
            )

        bench("RPC: LaunchJob (replicas=8, WRITE)", _launch8)

        # ---- TerminateJob (73/day, p95=274ms). ----
        terminate_load = load_terminate_job(harness, write_db, rps=0)
        if terminate_load is not None:
            with write_db.read_snapshot() as _tx:
                running_rows = _tx.fetchall(
                    select(jobs_table.c.job_id)
                    .where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING, jobs_table.c.depth == 1)
                    .limit(50)
                )
            n_cancel = len(running_rows)
            _bench_load(
                f"RPC: TerminateJob (cancel_job, n={n_cancel} jobs)",
                harness,
                terminate_load,
                min_runs=min(5, n_cancel),
                max_runs=n_cancel,
            )
        else:
            print("  RPC: TerminateJob                                                 (skipped, no running jobs)")

        # ---- UpdateTaskStatus (4.8k/day, p95=256ms) — one RPC per worker.
        hb_requests = _build_heartbeat_requests(write_db)
        if hb_requests:
            total_t = sum(len(r.updates) for r in hb_requests)
            print(f"  (UpdateTaskStatus batch: {len(hb_requests)} workers, {total_t} task updates total)")

            rpc_hb_requests = [
                controller_pb2.Controller.UpdateTaskStatusRequest(
                    worker_id=str(req.worker_id),
                    updates=[
                        job_pb2.WorkerTaskStatus(
                            task_id=u.task_id.to_wire(),
                            attempt_id=u.attempt_id,
                            state=u.new_state,
                        )
                        for u in req.updates
                    ],
                )
                for req in hb_requests
            ]
            mid = rpc_hb_requests[len(rpc_hb_requests) // 2]
            update_client = harness.make_client()

            bench(
                f"RPC: UpdateTaskStatus (1 worker, t={len(mid.updates)})",
                lambda: update_client.update_task_status(mid),
            )

            # _update_all_serial is bespoke: it issues one call per worker in sequence
            # so total latency is the sum — a different measurement than per-RPC.
            def _update_all_serial():
                for req in rpc_hb_requests:
                    update_client.update_task_status(req)

            bench(
                f"RPC: UpdateTaskStatus (w={len(rpc_hb_requests)} serial, t={total_t})",
                _update_all_serial,
                min_runs=3,
                min_time_s=2.0,
            )
    finally:
        harness.close()
        write_db.close()
        shutil.rmtree(db_dir, ignore_errors=True)
        shutil.rmtree(harness_dir, ignore_errors=True)


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
        workers = reads.healthy_active_workers_with_attributes(_wtx, health, _NoAttrs())
    print(
        f"  (scheduling shape: {len(workers)} workers, {len(pending_tasks)} pending tasks "
        f"after injecting {pending_count})"
    )

    # ---- resource_usage_by_worker (NEW): full join over unfinished
    #      worker-bound attempts. Runs every scheduling tick. ----
    def _usage_new():
        from iris.cluster.controller import db as db_mod

        with db_mod.read_snapshot(db.sa_read_engine) as snap:
            reads.resource_usage_by_worker(snap)

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
        from iris.cluster.controller import db as db_mod

        _schedulable_tasks(db)
        with db.read_snapshot() as _rtx:
            ws = reads.healthy_active_workers_with_attributes(_rtx, health, _NoAttrs())
        with db_mod.read_snapshot(db.sa_read_engine) as snap:
            usage = reads.resource_usage_by_worker(snap)
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
        addresses = reads.list_active_healthy_workers(snap, health)
    worker_ids = list(addresses)
    n_workers = len(worker_ids)
    print(f"  (polling shape: {n_workers} active+healthy workers)")

    # ---- list_active_healthy: the snapshot read that drives reconcile. ----
    def _list_active_healthy():
        with db.read_snapshot() as snap:
            reads.list_active_healthy_workers(snap, health)

    bench("Polling: list_active_healthy (next reconcile batch)", _list_active_healthy)

    # ---- reconcile_rows_for_workers: per-tick batch (up to 512 ids). ----
    for batch_size in (64, min(256, n_workers), min(512, n_workers)):
        if batch_size <= 0 or batch_size > n_workers:
            continue
        ids = worker_ids[:batch_size]

        def _reconcile(_ids=ids):
            from iris.cluster.controller import db as db_mod

            target_ids = set(_ids)
            with db_mod.read_snapshot(db.sa_read_engine) as snap:
                # Worker filter applied in Python to keep the partial index
                # ``idx_task_attempts_live_workerbound`` in play (a long IN
                # list on worker_id degrades to a scan).
                raw_rows = snap.execute(
                    select(
                        task_attempts_table.c.worker_id,
                        tasks_table.c.task_id,
                        task_attempts_table.c.attempt_id,
                        tasks_table.c.state.label("task_state"),
                        task_attempts_table.c.state.label("attempt_state"),
                        tasks_table.c.job_id,
                    )
                    .select_from(
                        task_attempts_table.join(
                            tasks_table,
                            (tasks_table.c.task_id == task_attempts_table.c.task_id)
                            & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
                        )
                    )
                    .where(
                        task_attempts_table.c.worker_id.is_not(None),
                        task_attempts_table.c.finished_at_ms.is_(None),
                        tasks_table.c.state.in_(
                            [
                                int(job_pb2.TASK_STATE_ASSIGNED),
                                int(job_pb2.TASK_STATE_BUILDING),
                                int(job_pb2.TASK_STATE_RUNNING),
                            ]
                        ),
                    ),
                ).all()
                _ = [row for row in raw_rows if row.worker_id in target_ids]

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
            from iris.cluster.controller import db as db_mod

            with db_mod.read_snapshot(db.sa_read_engine) as snap:
                reads.has_unfinished_worker_attempts(snap, drain_jid)

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
                reads.running_tasks_by_worker(tx, {w[0].worker_id for w in roster})

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
# Group: set_task_status_text (on_loop vs threadpool dispatch comparison)
# ---------------------------------------------------------------------------


def _percentiles_ms(latencies: list[float]) -> tuple[float, float, float, float, float]:
    latencies.sort()
    n = len(latencies)
    return (
        latencies[n // 2],
        latencies[int(n * 0.95)],
        latencies[int(n * 0.99)],
        latencies[min(int(n * 0.999), n - 1)],
        latencies[-1],
    )


def _set_status_text_probe(
    harness: "RpcHarness",
    *,
    n_threads: int,
    duration_s: float,
    payload_size: int = 256,
) -> dict:
    """Saturate SetTaskStatusText from ``n_threads`` client threads.

    Each thread runs a tight loop with its own pooled client — no rate
    limiting — so the result is the ceiling, not a hand-throttled target.
    """
    stop = threading.Event()
    payload = "x" * payload_size
    per_thread: list[list[float]] = [[] for _ in range(n_threads)]
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        client = harness.make_client()
        latencies = per_thread[idx]
        task_id = f"/u/probe/{uuid.uuid4().hex[:8]}-{idx}"
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                client.set_task_status_text(
                    job_pb2.SetTaskStatusTextRequest(
                        task_id=task_id,
                        status_text_detail_md=payload,
                        status_text_summary_md=payload[:32],
                    )
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)
        except BaseException as exc:
            errors.append(exc)
        finally:
            try:
                client.close()
            except Exception:
                pass

    threads = [threading.Thread(target=worker, args=(i,), name=f"sts-probe-{i}") for i in range(n_threads)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=30.0)
    elapsed = time.perf_counter() - t_start

    if errors:
        print(f"  probe worker errors: {errors[0]!r}")
    all_latencies = [v for sub in per_thread for v in sub]
    if not all_latencies:
        return {"n": 0, "rps": 0.0}
    p50, p95, p99, p999, mx = _percentiles_ms(all_latencies)
    return {
        "n": len(all_latencies),
        "rps": len(all_latencies) / elapsed,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "p999": p999,
        "max": mx,
    }


def _print_probe(label: str, r: dict) -> None:
    if r["n"] == 0:
        print(f"  {label:36s}  (no samples)")
        return
    print(
        f"  {label:36s}  n={r['n']:6d}  rps={r['rps']:8.1f}  "
        f"p50={r['p50']:6.2f}ms  p95={r['p95']:7.2f}ms  "
        f"p99={r['p99']:7.2f}ms  p99.9={r['p999']:7.2f}ms  max={r['max']:7.2f}ms"
    )
    _results.append((label, r["p50"], r["p95"], r["n"]))


def _print_scenario_table(scenario: Scenario, results: dict[str, dict]) -> None:
    header = (
        f"  {'RPC':<26}  {'target_rps':>10}  {'actual_rps':>10}  {'n':>6}  "
        f"{'errs':>5}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    empty_row_cells = "  ".join(
        [f"{'—':>10}", f"{'0':>6}", f"{'0':>5}", f"{'—':>7}", f"{'—':>7}", f"{'—':>7}", f"{'—':>7}"]
    )
    for load in scenario.loads:
        r = results.get(load.name, {})
        if not r or r["n"] == 0:
            print(f"  {load.name:<26}  {load.target_rps:>10.2f}  {empty_row_cells}")
            continue
        print(
            f"  {load.name:<26}  {load.target_rps:>10.2f}  {r['actual_rps']:>10.2f}  "
            f"{r['n']:>6}  {r['errors']:>5}  "
            f"{r['p50']:>6.1f}ms  {r['p95']:>6.1f}ms  {r['p99']:>6.1f}ms  {r['max']:>6.1f}ms"
        )
        _results.append((f"Scenario: {load.name}", r["p50"], r["p95"], r["n"]))


def _run_set_status_text_variant(
    db: ControllerDB,
    *,
    on_loop: bool,
    n_threads: int,
    duration_s: float,
) -> dict:
    """Boot a fresh harness with the requested dispatch mode and run the probe."""
    label = "on_loop=True" if on_loop else "on_loop=False (threadpool)"
    write_db = clone_db(db)
    db_dir = write_db._db_dir
    db_path = db_dir / ControllerDB.DB_FILENAME
    write_db.close()
    harness_dir = Path(tempfile.mkdtemp(prefix="iris_sts_probe_"))
    harness = RpcHarness(db_path, harness_dir, on_loop=on_loop)
    try:
        # Warmup: prime the connection / event loop with a single call.
        warmup = harness.make_client()
        try:
            warmup.set_task_status_text(
                job_pb2.SetTaskStatusTextRequest(
                    task_id="/u/warmup/0",
                    status_text_detail_md="",
                    status_text_summary_md="",
                )
            )
        finally:
            warmup.close()
        # Single-call baseline (sequential, low load) so the on_loop dispatch
        # cost is visible without saturation effects.
        baseline_client = harness.make_client()
        baseline_latencies: list[float] = []
        try:
            for _ in range(200):
                t0 = time.perf_counter()
                baseline_client.set_task_status_text(
                    job_pb2.SetTaskStatusTextRequest(
                        task_id="/u/baseline/0",
                        status_text_detail_md="x" * 256,
                        status_text_summary_md="x" * 32,
                    )
                )
                baseline_latencies.append((time.perf_counter() - t0) * 1000.0)
        finally:
            baseline_client.close()
        baseline_p50, baseline_p95, baseline_p99, _, _ = _percentiles_ms(baseline_latencies)
        print(
            f"  SetTaskStatusText [{label}] single-client baseline: "
            f"p50={baseline_p50:.2f}ms  p95={baseline_p95:.2f}ms  p99={baseline_p99:.2f}ms  (n=200)"
        )
        _results.append((f"RPC: SetTaskStatusText baseline [{label}]", baseline_p50, baseline_p95, 200))

        # Saturation probe.
        return _set_status_text_probe(harness, n_threads=n_threads, duration_s=duration_s)
    finally:
        harness.close()
        shutil.rmtree(db_dir, ignore_errors=True)
        shutil.rmtree(harness_dir, ignore_errors=True)


def _measure_update_task_status_under_storm(
    db: ControllerDB,
    *,
    on_loop: bool,
    storm_rps_target: int,
    storm_threads: int,
    victim_duration_s: float,
) -> tuple[dict, dict]:
    """Run a victim UpdateTaskStatus RPC alongside a SetTaskStatusText storm.

    Returns ``(baseline, under_storm)`` results so the caller can quantify the
    blast radius. Both phases share the same harness (and therefore the same
    background scheduler/polling/ping cadence) so the only changing variable
    is whether the storm is running.

    The victim is a real ``UpdateTaskStatus`` RPC built from the snapshot's
    live worker→task attribution. Sample a worker that actually has updates
    to push so we measure a write-touching call, not a no-op early-return.
    """
    label = "on_loop=True" if on_loop else "on_loop=False (threadpool)"
    write_db = clone_db(db)
    db_dir = write_db._db_dir
    db_path = db_dir / ControllerDB.DB_FILENAME
    write_db.close()
    harness_dir = Path(tempfile.mkdtemp(prefix="iris_sts_contend_"))
    harness = RpcHarness(db_path, harness_dir, on_loop=on_loop)
    try:
        # Build the victim request. Bias toward a worker with non-empty updates.
        hb_requests = _build_heartbeat_requests(db)
        candidates = [req for req in hb_requests if req.updates]
        if not candidates:
            print(f"  [{label}] (skipped: no worker with active tasks for victim)")
            return ({"n": 0, "rps": 0.0}, {"n": 0, "rps": 0.0})
        sample = candidates[len(candidates) // 2]
        victim_req = controller_pb2.Controller.UpdateTaskStatusRequest(
            worker_id=str(sample.worker_id),
            updates=[
                job_pb2.WorkerTaskStatus(
                    task_id=u.task_id.to_wire(),
                    attempt_id=u.attempt_id,
                    state=u.new_state,
                )
                for u in sample.updates
            ],
        )

        def _victim_run(seconds: float) -> dict:
            client = harness.make_client()
            latencies: list[float] = []
            try:
                deadline = time.perf_counter() + seconds
                while time.perf_counter() < deadline:
                    t0 = time.perf_counter()
                    client.update_task_status(victim_req)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
            finally:
                client.close()
            if not latencies:
                return {"n": 0, "rps": 0.0}
            p50, p95, p99, p999, mx = _percentiles_ms(latencies)
            return {
                "n": len(latencies),
                "rps": len(latencies) / seconds,
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "p999": p999,
                "max": mx,
            }

        # Warmup: prime connections + JIT paths.
        warmup_client = harness.make_client()
        try:
            warmup_client.update_task_status(victim_req)
            warmup_client.set_task_status_text(
                job_pb2.SetTaskStatusTextRequest(
                    task_id="/u/warmup/0", status_text_detail_md="", status_text_summary_md=""
                )
            )
        finally:
            warmup_client.close()

        # Phase 1: victim baseline (no storm).
        baseline = _victim_run(victim_duration_s)

        # Phase 2: storm + victim concurrently. The storm threads rate-limit
        # themselves to ``storm_rps_target`` total so we're testing the
        # production scenario (1000 RPS), not the saturation ceiling.
        stop = threading.Event()
        per_thread_target = storm_rps_target / storm_threads
        storm_payload = "x" * 256
        storm_short = storm_payload[:32]
        storm_latencies: list[list[float]] = [[] for _ in range(storm_threads)]

        def _storm_worker(idx: int) -> None:
            client = harness.make_client()
            latencies = storm_latencies[idx]
            task_id = f"/u/storm/{uuid.uuid4().hex[:8]}-{idx}"
            interval = 1.0 / per_thread_target if per_thread_target > 0 else 0.0
            next_send = time.perf_counter()
            try:
                while not stop.is_set():
                    now = time.perf_counter()
                    wait = next_send - now
                    if wait > 0:
                        # Sleep in small slices so stop signal is responsive.
                        time.sleep(min(wait, 0.05))
                        continue
                    t0 = time.perf_counter()
                    client.set_task_status_text(
                        job_pb2.SetTaskStatusTextRequest(
                            task_id=task_id,
                            status_text_detail_md=storm_payload,
                            status_text_summary_md=storm_short,
                        )
                    )
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    next_send += interval
                    # If we fell badly behind, don't try to catch up — reset the
                    # baseline so a slow tail doesn't compound.
                    if next_send < time.perf_counter() - interval:
                        next_send = time.perf_counter()
            finally:
                try:
                    client.close()
                except Exception:
                    pass

        threads = [
            threading.Thread(target=_storm_worker, args=(i,), name=f"sts-storm-{i}") for i in range(storm_threads)
        ]
        for t in threads:
            t.start()
        # Let the storm reach steady state before measuring the victim.
        time.sleep(0.5)
        under_storm = _victim_run(victim_duration_s)
        stop.set()
        for t in threads:
            t.join(timeout=10.0)

        storm_total = sum(len(s) for s in storm_latencies)
        storm_rps_actual = storm_total / (victim_duration_s + 0.5)
        print(
            f"  [{label}] storm actual: {storm_rps_actual:.0f} rps "
            f"(target {storm_rps_target} rps from {storm_threads} threads, sent {storm_total})"
        )
        return baseline, under_storm
    finally:
        harness.close()
        shutil.rmtree(db_dir, ignore_errors=True)
        shutil.rmtree(harness_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Group: scenario (production-mix concurrent RPC benchmark)
# ---------------------------------------------------------------------------


def benchmark_scenario(db: ControllerDB) -> None:
    """Drive a production-mix RPC scenario against an out-of-process controller.

    The controller runs in a separate Python process so the benchmark
    process's client threads don't share a GIL with the server. Mirrors
    production where each task pushes from its own interpreter.
    """
    write_db = clone_db(db)
    harness_dir = Path(tempfile.mkdtemp(prefix="iris_scenario_"))
    db_dir = write_db._db_dir
    db_path = db_dir / ControllerDB.DB_FILENAME
    # Close the parent's writer connection so the subprocess owns the DB:
    # we still read samples through ``sample_db.read_snapshot()`` (WAL handles
    # concurrent readers), but the parent doesn't need a writer.
    write_db.close()
    harness = RpcHarness(db_path, harness_dir)
    # Re-open for read-only sampling by the factories. Safe under WAL because
    # the subprocess controller has already migrated and opened the file.
    sample_db = ControllerDB(db_dir=db_dir)
    try:
        scenario = build_production_scenario(harness, sample_db, scale=_SCENARIO_SCALE, duration_s=_SCENARIO_DURATION)
        results = ScenarioRunner(harness, scenario).run()
        _print_scenario_table(scenario, results)
    finally:
        harness.close()
        sample_db.close()
        shutil.rmtree(db_dir, ignore_errors=True)
        shutil.rmtree(harness_dir, ignore_errors=True)


def benchmark_set_task_status_text(db: ControllerDB) -> None:
    """Compare SetTaskStatusText latency & throughput with @on_loop vs threadpool.

    Production sees ~1000 SetTaskStatusText/sec. The handler body is two
    in-memory dict writes (microseconds); what changes between the two
    dispatch modes is whether each call runs inline on the asyncio event loop
    (one-at-a-time, blocks every other on_loop handler) or hops into a thread
    via ``asyncio.to_thread`` (parallel-capable, pays a thread-hop per call).

    Two scenarios:

    1. **Saturation probe**: N client threads in a tight loop hitting
       SetTaskStatusText with no rate limiting — surfaces the ceiling and
       lets us compare modes head-to-head.

    2. **Production-shape contention**: a victim ``UpdateTaskStatus`` RPC
       runs alongside a 1000 RPS SetTaskStatusText storm. The delta between
       victim-baseline and victim-under-storm is the blast radius the prod
       traffic imposes on every other handler.

    Both scenarios run the controller's scheduler / polling / ping loops at
    their natural production cadence (10s / 1s / etc).
    """
    n_threads = 8
    duration_s = 5.0

    # ---- Saturation probe ----
    print(f"  (saturation probe: {n_threads} client threads, {duration_s:.0f}s per variant, no rate-limit)")
    threadpool_result = _run_set_status_text_variant(db, on_loop=False, n_threads=n_threads, duration_s=duration_s)
    _print_probe("RPC: SetTaskStatusText [threadpool]", threadpool_result)

    on_loop_result = _run_set_status_text_variant(db, on_loop=True, n_threads=n_threads, duration_s=duration_s)
    _print_probe("RPC: SetTaskStatusText [on_loop]", on_loop_result)

    if threadpool_result["n"] and on_loop_result["n"]:
        rps_ratio = on_loop_result["rps"] / threadpool_result["rps"]
        p99_ratio = on_loop_result["p99"] / threadpool_result["p99"]
        print(f"  saturation: on_loop / threadpool ratio  rps x{rps_ratio:.2f}, p99 x{p99_ratio:.2f}")

    # ---- Production-shape contention: victim UpdateTaskStatus + 1000 RPS storm. ----
    print()
    storm_rps = 1000
    storm_threads = 8
    victim_seconds = 5.0
    print(
        f"  (contention probe: victim=UpdateTaskStatus, storm={storm_rps} rps SetTaskStatusText "
        f"from {storm_threads} threads, {victim_seconds:.0f}s)"
    )
    tp_baseline, tp_under = _measure_update_task_status_under_storm(
        db, on_loop=False, storm_rps_target=storm_rps, storm_threads=storm_threads, victim_duration_s=victim_seconds
    )
    _print_probe("Victim: UpdateTaskStatus [storm=threadpool] baseline", tp_baseline)
    _print_probe("Victim: UpdateTaskStatus [storm=threadpool] +storm", tp_under)

    ol_baseline, ol_under = _measure_update_task_status_under_storm(
        db, on_loop=True, storm_rps_target=storm_rps, storm_threads=storm_threads, victim_duration_s=victim_seconds
    )
    _print_probe("Victim: UpdateTaskStatus [storm=on_loop] baseline", ol_baseline)
    _print_probe("Victim: UpdateTaskStatus [storm=on_loop] +storm", ol_under)

    # Blast-radius summary: how much does the storm cost the victim?
    def _ratio(under: dict, base: dict, key: str) -> str:
        if not under.get("n") or not base.get("n") or base[key] == 0:
            return "—"
        return f"x{under[key] / base[key]:.2f}"

    print()
    print("  blast radius (victim under storm / victim baseline):")
    print(
        f"    threadpool storm:  p50 {_ratio(tp_under, tp_baseline, 'p50')}  "
        f"p95 {_ratio(tp_under, tp_baseline, 'p95')}  "
        f"p99 {_ratio(tp_under, tp_baseline, 'p99')}  "
        f"max {_ratio(tp_under, tp_baseline, 'max')}"
    )
    print(
        f"    on_loop storm:     p50 {_ratio(ol_under, ol_baseline, 'p50')}  "
        f"p95 {_ratio(ol_under, ol_baseline, 'p95')}  "
        f"p99 {_ratio(ol_under, ol_baseline, 'p99')}  "
        f"max {_ratio(ol_under, ol_baseline, 'max')}"
    )


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


_GROUPS = (
    "rpcs",
    "scheduling",
    "polling",
    "dashboard",
    "endpoints",
    "apply_contention",
    "set_task_status_text",
    "scenario",
)


@click.group()
def main() -> None:
    """Benchmark the Iris controller against a local checkpoint."""


@main.command("run")
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
@click.option(
    "--scale",
    "scenario_scale",
    type=float,
    default=1.0,
    help="Scale all production_mix rates by this factor.",
)
@click.option(
    "--scenario-duration",
    "scenario_duration",
    type=float,
    default=60.0,
    help="Scenario runtime in seconds.",
)
def run_cmd(db_path: Path | None, only_group: str | None, scenario_scale: float, scenario_duration: float) -> None:
    """Run the benchmark groups."""
    global _SCENARIO_SCALE, _SCENARIO_DURATION
    _SCENARIO_SCALE = scenario_scale
    _SCENARIO_DURATION = scenario_duration
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
        ("set_task_status_text", benchmark_set_task_status_text),
        ("scenario", benchmark_scenario),
    ]
    for name, fn in groups:
        if only_group is not None and only_group != name:
            continue
        print(f"[{name}]")
        fn(db)
        print()

    print_summary()
    db.close()


@main.command("serve")
@click.option(
    "--db-path",
    "db_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the controller.sqlite3 file the dry-run server should open.",
)
@click.option(
    "--state-dir",
    "state_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Scratch directory for the controller's local state / remote-state-dir.",
)
@click.option(
    "--on-loop/--no-on-loop",
    "on_loop",
    default=False,
    help="Run SetTaskStatusText inline on the event loop (vs the default threadpool dispatch).",
)
def serve_cmd(db_path: Path, state_dir: Path, on_loop: bool) -> None:
    """Boot a Controller(dry_run=True) bound to ``db_path`` and serve over RPC.

    Prints ``READY port=N`` to stdout once the HTTP server is accepting
    connections, then blocks until SIGTERM. Used by ``RpcHarness`` so the
    benchmark process and the controller have independent GILs.

    ``--on-loop`` flips the ``@on_loop`` attribute on
    ``ControllerServiceImpl.set_task_status_text`` before the controller is
    built. ``AsyncServiceAdapter`` reads the attribute when it walks the
    service's methods, so the toggle is locked in for this process's
    lifetime.
    """
    if on_loop:
        from iris.cluster.controller.service import ControllerServiceImpl
        from iris.rpc.async_adapter import _ON_LOOP_ATTR

        setattr(ControllerServiceImpl.set_task_status_text, _ON_LOOP_ATTR, True)

    state_dir.mkdir(parents=True, exist_ok=True)
    db = ControllerDB(db_dir=db_path.parent)
    db.apply_migrations()

    config = ControllerConfig(
        host="127.0.0.1",
        port=0,
        remote_state_dir=f"file://{state_dir / 'remote'}",
        local_state_dir=state_dir / "local",
        dry_run=True,
        checkpoint_interval=None,
    )
    threads = ThreadContainer("bench-serve")
    controller = Controller(config=config, provider=_FakeProvider(), db=db, threads=threads)
    controller.start()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if controller._server is not None and controller._server.started:
            break
        time.sleep(0.02)
    else:
        controller.stop()
        raise click.ClickException("Controller server did not start within 10s")

    print(f"READY port={controller.port}", flush=True)

    stop = threading.Event()

    def _shutdown(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    stop.wait()
    controller.stop()
    db.close()


if __name__ == "__main__":
    main()
