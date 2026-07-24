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

    # Benchmark the Reconcile RPC dispatch + apply tick on a synthesized DB
    # (independent of the checkpoint-driven groups above).
    uv run python lib/iris/scripts/benchmark_controller.py reconcile \\
        --num-tasks 5000 --tasks-per-worker 64 --payload-bytes 100000

"""

import asyncio
import dataclasses
import math
import os
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import click
import uvicorn
import yaml
from connectrpc.request import RequestContext
from iris.cluster.backends.rpc.backend import (
    WORKER_RECONCILE_TEARDOWN_REASON,
    RpcTaskBackend,
    RpcWorkerStubFactory,
)
from iris.cluster.controller import ops, reads
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendRuntime,
    ReconcileRequest,
    ReconcileResult,
    ScheduleInput,
    ScheduleRequest,
    ScheduleResult,
    TaskBackend,
    assemble_scheduling_context,
    plans_from_snapshot,
    run_scheduling_decision,
)
from iris.cluster.controller.backend_store import BackendWorkerStore, DbBackendWorkerStore
from iris.cluster.controller.checkpoint import download_checkpoint_to_local
from iris.cluster.controller.controller import (
    _CONTROLLER_KEEPALIVE,
    Controller,
    ControllerConfig,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.ops.worker import apply_reconcile
from iris.cluster.controller.projections.endpoints import EndpointQuery, EndpointRow, EndpointsProjection
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.reads import (  # noqa: F401
    ControlSnapshot,
    SchedulableWorker,
    healthy_active_workers_with_attributes,
)
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileRow,
    WorkerReconcilePlan,
    WorkerReconcileResult,
    build_reconcile_plans,
)
from iris.cluster.controller.scheduling.policy import build_scheduling_context, compute_demand_entries
from iris.cluster.controller.scheduling.scheduler import Scheduler
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
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind, WorkerHealthTracker
from iris.cluster.types import DEFAULT_BACKEND_ID, AttemptUid, JobName, UserBudgetDefaults, WorkerId
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2, job_pb2, query_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.worker_connect import WorkerService, WorkerServiceASGIApplication
from iris.version import client_revision_date
from rigging.timing import Duration, ExponentialBackoff, Timestamp
from sqlalchemy import func, select, text, update
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import CursorTransitionReader


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
    """Minimal worker-daemon TaskBackend that satisfies Controller's wiring
    without making real cluster calls. Mirrors
    tests/cluster/controller/conftest.py:FakeProvider.

    Used in combination with ``dry_run=True`` so the polling loop's reconcile
    short-circuits before it would ever call the provider — but we still need a
    real provider object to satisfy the constructor's type contract.
    """

    name = "worker"
    capabilities = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
    autoscaler = None

    def __init__(self) -> None:
        self._scheduler = Scheduler()
        self._store: BackendWorkerStore | None = None
        self.health: WorkerHealthTracker = WorkerHealthTracker()
        self.advertised: dict[str, set[str]] = {}
        self._pending_dead: list[WorkerId] = []

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def configure_routing(self, advertised: dict[str, set[str]]) -> None:
        self.advertised = advertised

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        assert self._store is not None
        context = assemble_scheduling_context(self._store.scheduling_inputs(), request)
        return run_scheduling_decision(
            self._scheduler,
            ScheduleInput(
                context=context,
                max_tasks_per_job_per_cycle=request.max_tasks_per_job_per_cycle,
                trace=request.trace,
            ),
        )

    def get_process_status(self, target, request):
        raise RuntimeError("fake provider")

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        self._store = DbBackendWorkerStore(
            db=runtime.db,
            owns_scale_group=runtime.owns_scale_group,
            health=self.health,
            defaults=runtime.budget_defaults,
            autoscale=self.autoscale,
        )

    def seed_liveness(self) -> None:
        assert self._store is not None
        worker_ids = self._store.owned_worker_ids()
        if worker_ids:
            self.health.heartbeat(worker_ids, Timestamp.now().epoch_ms())

    def set_log_sink(self, *args, **kwargs):
        pass

    def profile_task(self, target, request, timeout_ms):
        raise RuntimeError("fake provider")

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        # Same projection the test-suite FakeProvider returns. Only exercised if
        # someone disables dry_run on the harness.
        assert self._store is not None
        snapshot = self._store.reconcile_snapshot()
        plans = plans_from_snapshot(snapshot)
        worker_results = [(p, WorkerReconcileResult(worker_id=p.worker_id, observations=[], error=None)) for p in plans]
        events = [WorkerHealthEvent(p.worker_id, WorkerHealthEventKind.REACHED) for p in plans]
        now = Timestamp.now()
        effects = apply_reconcile(self._store, worker_results, now=now)
        events += [WorkerHealthEvent(wid, WorkerHealthEventKind.BUILD_FAILED) for wid in effects.health.build_failed]
        self._pending_dead.extend(self.health.apply(events, now_ms=now.epoch_ms()))
        return ReconcileResult(effects=effects)

    def run_teardown(self) -> None:
        dead = self._pending_dead
        self._pending_dead = []
        self.teardown(dead, reason=WORKER_RECONCILE_TEARDOWN_REASON)

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        assert self._store is not None
        self._store.reap_workers(dead_workers, reason=reason)

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        return AutoscaleResult()

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
    """

    def __init__(
        self,
        db_path: Path,
        tmp: Path,
        *,
        startup_timeout_s: float = 30.0,
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
    independently of throughput — e.g. a hot RPC pushed by many task
    processes, each holding its own connection. A single thread doing N rps
    doesn't expose the same concurrency on the server.
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
                percentiles = _percentiles_ms(all_latencies)
                results[load.name] = {
                    "n": n,
                    "errors": len(all_errors),
                    "actual_rps": n / elapsed,
                    "p50": percentiles.p50,
                    "p95": percentiles.p95,
                    "p99": percentiles.p99,
                    "max": percentiles.maximum,
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


def _build_reconcile_inputs(
    db: ControllerDB,
) -> list[tuple[WorkerReconcilePlan, WorkerReconcileResult]]:
    """Per active worker: a (plan, result) pair where the plan lists the worker's
    attempts as desired and the result reports each RUNNING. Drives the
    production reconcile-observation verb (``ops.worker.apply_reconcile``).
    """
    health = WorkerHealthTracker()
    _seed_health(db, health)
    with db.read_snapshot() as tx:
        workers = reads.healthy_active_workers_with_attributes(tx, health, _NoAttrs())
    active_states = list(ACTIVE_TASK_STATES)
    plan_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = []
    for w in workers:
        with db.read_snapshot() as tx:
            rows = tx.execute(
                select(task_attempts_table.c.attempt_uid)
                .select_from(
                    task_attempts_table.join(
                        tasks_table,
                        (tasks_table.c.task_id == task_attempts_table.c.task_id)
                        & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
                    )
                )
                .where(
                    tasks_table.c.current_worker_id == w.worker_id,
                    tasks_table.c.state.in_(active_states),
                )
            ).all()
        uids = [row.attempt_uid for row in rows]
        desired = [
            worker_pb2.Worker.DesiredAttempt(attempt_uid=uid, run=worker_pb2.Worker.AttemptSpec()) for uid in uids
        ]
        observations = [
            worker_pb2.Worker.AttemptObservation(attempt_uid=uid, state=job_pb2.TASK_STATE_RUNNING) for uid in uids
        ]
        plan = WorkerReconcilePlan(
            worker_id=w.worker_id,
            request=worker_pb2.Worker.ReconcileRequest(worker_id=str(w.worker_id), desired=desired),
        )
        result = WorkerReconcileResult(worker_id=w.worker_id, observations=observations, error=None)
        plan_results.append((plan, result))
    return plan_results


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
        rows = tx.execute(select(workers_table.c.worker_id, workers_table.c.address).limit(n)).all()
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
        rows = tx.execute(
            select(tasks_table.c.task_id, tasks_table.c.current_attempt_id)
            .where(
                tasks_table.c.state.in_(active_states),
                tasks_table.c.current_attempt_id.is_not(None),
            )
            .limit(limit)
        ).all()
    return [(row.task_id, int(row.current_attempt_id)) for row in rows]


def _has_committed_columns(db: ControllerDB) -> bool:
    with db.read_snapshot() as tx:
        rows = tx.execute(text("PRAGMA table_info(workers)")).all()
    return "committed_cpu_millicores" in {r[1] for r in rows}


# ---------------------------------------------------------------------------
# === Section: RPC load factories ===
# ---------------------------------------------------------------------------


def load_get_job_state(harness: RpcHarness, db: ControllerDB, rps: float, batch: int = 1) -> RpcLoad | None:
    with db.read_snapshot() as tx:
        rows = tx.execute(select(jobs_table.c.job_id).limit(50)).all()
    if not rows:
        return None
    job_ids = [str(r.job_id) for r in rows[:batch]]
    req = controller_pb2.Controller.GetJobStateRequest(job_ids=job_ids)

    def invoke(client: ControllerServiceClientSync) -> None:
        client.get_job_state(req)

    return RpcLoad(name="GetJobState", target_rps=rps, invoke=invoke)


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
        rows = tx.execute(
            select(jobs_table.c.job_id)
            .where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING, jobs_table.c.depth == 1)
            .limit(50)
        ).all()
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
        rows = tx.execute(select(jobs_table.c.job_id).limit(1)).all()
    if not rows:
        return None
    job_id = str(rows[0].job_id)
    req = controller_pb2.Controller.ListTasksRequest(job_id=job_id)

    def invoke(client: ControllerServiceClientSync) -> None:
        client.list_tasks(req)

    return RpcLoad(name="ListTasks", target_rps=rps, invoke=invoke)


def load_get_job_status(harness: RpcHarness, db: ControllerDB, rps: float) -> RpcLoad | None:
    with db.read_snapshot() as tx:
        rows = tx.execute(select(jobs_table.c.job_id).limit(1)).all()
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
    "GetJobState": 7.85,
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
    "GetJobState": load_get_job_state,
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
    Register, RegisterEndpoint, LaunchJob, TerminateJob.

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
            rows = tx.execute(select(jobs_table.c.job_id).limit(50)).all()
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
                running_rows = _tx.execute(
                    select(jobs_table.c.job_id)
                    .where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING, jobs_table.c.depth == 1)
                    .limit(50)
                ).all()
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
        running_jobs = _snap.execute(
            select(jobs_table.c.job_id).where(jobs_table.c.state == job_pb2.JOB_STATE_RUNNING).limit(50)
        ).all()
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
    with db.read_snapshot() as _ptx:
        pending_tasks = reads.pending_tasks_with_jobs(_ptx)
    with db.read_snapshot() as _wtx:
        workers = reads.healthy_active_workers_with_attributes(_wtx, health, _NoAttrs())
    print(
        f"  (scheduling shape: {len(workers)} workers, {len(pending_tasks)} pending tasks "
        f"after injecting {pending_count})"
    )

    # ---- resource_usage_by_worker (NEW): full join over unfinished
    #      worker-bound attempts. Runs every scheduling tick. ----
    def _usage_new():
        with db.read_snapshot() as snap:
            reads.resource_usage_by_worker(snap)

    bench("Scheduling: resource_usage_by_worker (NEW derived query)", _usage_new)

    # ---- Predecessor: SELECT committed_* FROM workers. Only available
    #      pre-migration. ----
    if _has_committed_columns(db):

        def _usage_old():
            with db.read_snapshot() as _tx:
                _tx.execute(
                    text(
                        "SELECT worker_id, committed_cpu_millicores, committed_mem_bytes, "
                        "committed_gpu, committed_tpu FROM workers"
                    )
                ).all()

        bench("Scheduling: workers.committed_* read (pre-Jumbo)", _usage_old)
    else:
        print("  Scheduling: workers.committed_* read (pre-Jumbo)                  (skipped, columns dropped)")

    # ---- Full tick: _read_scheduling_state-style aggregate ----
    def _state_read():
        with db.read_snapshot() as _ptx:
            reads.pending_tasks_with_jobs(_ptx)
        with db.read_snapshot() as _rtx:
            ws = reads.healthy_active_workers_with_attributes(_rtx, health, _NoAttrs())
        with db.read_snapshot() as snap:
            usage = reads.resource_usage_by_worker(snap)
        return ws, usage

    bench("Scheduling: state read (pending+workers+usage)", _state_read)

    # ---- Autoscaler demand path (compute_demand_entries) ----
    # Demand is now a pure function over the per-tick scheduling context; build it
    # once (as the scheduling pass does) and benchmark only the demand computation.
    # SchedulingContext carries plain data built eagerly, so it is safe to use after
    # the snapshot closes.
    sched = Scheduler()
    with db.read_snapshot() as _sched_snap:
        demand_ctx = build_scheduling_context(_sched_snap, health, _NoAttrs(), UserBudgetDefaults())

    def _demand():
        compute_demand_entries(demand_ctx, sched)

    bench(
        f"Scheduling: compute_demand_entries (w={len(workers)}, t={len(pending_tasks)})",
        _demand,
        min_runs=3,
        min_time_s=1.0,
    )

    # ---- assign WRITE path ----
    write_db = clone_db(db)
    write_txns = ControllerTestState(write_db)
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
                    rows = _rtx.execute(
                        select(
                            tasks_table.c.task_id,
                            tasks_table.c.state,
                            tasks_table.c.current_attempt_id,
                            tasks_table.c.current_worker_id,
                            tasks_table.c.current_worker_address,
                            tasks_table.c.started_at_ms,
                        ).where(tasks_table.c.task_id.in_(task_ids_jn))
                    ).all()
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
                    ops.task.assign(cur, sample_assignments, health=write_txns._health)

            bench(
                f"Scheduling: assign (n={len(sample_assignments)} tasks, WRITE)",
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
    _ = ControllerTestState(db, health=health)

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
            target_ids = set(_ids)
            with db.read_snapshot() as snap:
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
            _tx.execute(
                select(
                    tasks_table.c.current_worker_id,
                    tasks_table.c.task_id,
                    tasks_table.c.current_attempt_id,
                    tasks_table.c.state,
                ).where(
                    tasks_table.c.state.in_(_active),
                    tasks_table.c.current_worker_id.is_not(None),
                )
            ).all()

    bench("Polling: poll_all_workers (pre-Jumbo single-shot read)", _poll_all_pre_jumbo)

    # ---- RunTemplatesProjection.get (cached): dominates ASSIGNED rows in the tick. ----
    _active_states = [job_pb2.TASK_STATE_ASSIGNED, job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING]
    with db.read_snapshot() as _rtx:
        rows = _rtx.execute(
            select(tasks_table.c.job_id)
            .where(
                tasks_table.c.state.in_(_active_states),
                tasks_table.c.current_worker_id.is_not(None),
            )
            .limit(64)
        ).all()
    sample_job_ids = list({row.job_id for row in rows})
    if sample_job_ids:
        first_job = sample_job_ids[0]

        def _template_first():
            # Invalidate to defeat the cache on every call.
            with db.transaction() as cur:
                db.caches[RunTemplatesProjection].invalidate_for_job(cur, first_job)
            with db.read_snapshot() as snap:
                snap.caches[RunTemplatesProjection].get(snap, first_job)

        bench("Polling: run_templates_projection.get (cold, per-job build)", _template_first)

        # Warm cache: same job repeatedly.
        with db.read_snapshot() as snap:
            snap.caches[RunTemplatesProjection].get(snap, first_job)

        def _template_warm():
            with db.read_snapshot() as snap:
                snap.caches[RunTemplatesProjection].get(snap, first_job)

        bench("Polling: run_templates_projection.get (cached hit)", _template_warm)

    # ---- has_unfinished_worker_attempts: drain gate for job replacement. ----
    # Walks the parent_job_id subtree. Pick a depth=1 job with active subtree
    # tasks if possible, otherwise just any job.
    _drain_active = [job_pb2.TASK_STATE_ASSIGNED, job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING]
    with db.read_snapshot() as _dtx:
        drain_row = _dtx.execute(
            select(jobs_table.c.job_id)
            .join(tasks_table, tasks_table.c.job_id == jobs_table.c.job_id)
            .where(
                tasks_table.c.state.in_(_drain_active),
                tasks_table.c.current_worker_id.is_not(None),
                jobs_table.c.depth == 1,
            )
            .limit(1)
        ).first()
    if drain_row:
        drain_jid = drain_row.job_id

        def _has_unfinished():
            with db.read_snapshot() as snap:
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
            with db.read_snapshot() as snap:
                tasks = _tasks_for_listing(snap, job_id=sample_job)
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
    write_txns = ControllerTestState(write_db)
    # ops.worker.fail only acts on workers the tracker considers active.
    _seed_health(write_db, write_txns._health)

    try:
        sample = _active_task_sample(write_db, limit=300)
        if not sample:
            print("  (skipped, no active tasks to attach endpoints to)")
            return

        def _reset_endpoints():
            with write_db.transaction() as _tx:
                _tx.execute(text("DELETE FROM endpoints WHERE name LIKE '/bench/endpoint/%'"))
            write_txns._endpoints.rehydrate()

        for burst_n in (50, 200):
            if len(sample) < burst_n:
                continue
            tasks_for_burst = sample[:burst_n]

            def _per_txn(tasks=tasks_for_burst):
                for t, aid in tasks:
                    with write_db.transaction() as cur:
                        write_txns._endpoints.add(cur, _make_endpoint(t, aid))

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
                        write_txns._endpoints.add(cur, _make_endpoint(t, aid))

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
            worker_rows = _wtx.execute(select(workers_table.c.worker_id, workers_table.c.address).limit(fail_n)).all()
        if len(worker_rows) >= fail_n:
            target_wids = [WorkerId(str(r.worker_id)) for r in worker_rows]
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
                # Re-activate restored workers: ops.worker.fail deactivated them
                # in the tracker, so the next timed call would otherwise no-op.
                _seed_health(write_db, write_txns._health)
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
                lambda wids=[str(w) for w in target_wids]: ops.worker.fail(
                    write_txns._db,
                    worker_ids=wids,
                    reason="benchmark: simulated provider-sync failure",
                    health=write_txns._health,
                ),
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
        f"  {name:64s}  n={len(latencies):3d}  p50={p50:7.1f}ms  p95={p95:8.1f}ms  p99={p99:8.1f}ms  max={max_ms:8.1f}ms"
    )


class _ContentionScenario(TypedDict):
    """Per-scenario overrides spread into :func:`_run_apply_under_contention` via ``**``."""

    name: str
    fail_threads: NotRequired[int]
    fail_chunk: NotRequired[int]
    fail_interval_s: NotRequired[float]
    register_threads: NotRequired[int]
    endpoint_threads: NotRequired[int]


def _run_apply_under_contention(
    *,
    name: str,
    write_db: ControllerDB,
    write_txns: ControllerTestState,
    plan_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]],
    fail_threads: int = 0,
    fail_n: int = 50,
    fail_chunk: int = 50,
    fail_interval_s: float = 2.0,
    register_threads: int = 0,
    register_burst: int = 100,
    endpoint_threads: int = 0,
    duration_s: float = 6.0,
) -> None:
    """Run apply_reconcile on a victim thread while configurable
    write storms hammer the same DB. Reports p50/p95/p99/max of the victim.
    """
    _active_states_contend = list(ACTIVE_TASK_STATES)
    with write_db.read_snapshot() as _ctx:
        endpoint_tasks_rows = _ctx.execute(
            select(tasks_table.c.task_id, tasks_table.c.current_attempt_id)
            .where(
                tasks_table.c.state.in_(_active_states_contend),
                tasks_table.c.current_attempt_id.is_not(None),
            )
            .limit(200)
        ).all()
    endpoint_tasks = [(row.task_id, int(row.current_attempt_id)) for row in endpoint_tasks_rows]

    stop = threading.Event()
    victim_latencies: list[float] = []
    errors: list[BaseException] = []

    def _victim():
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                with write_txns._db.transaction() as cur:
                    now = Timestamp.now()
                    effects = apply_reconcile(CursorTransitionReader(cur), plan_results, now=now)
                    commit_effects(cur, effects)
                victim_latencies.append((time.perf_counter() - t0) * 1000)
        except BaseException as e:
            errors.append(e)

    def _fail_storm():
        try:
            while not stop.is_set():
                failures = _build_failure_batch(write_db, fail_n)
                if failures:
                    # fail_chunk is ignored: chunking is now a fixed module
                    # constant (FAIL_WORKERS_CHUNK_SIZE) in ops.worker.fail.
                    ops.worker.fail(
                        write_txns._db,
                        worker_ids=[str(wid) for wid, _addr, _reason in failures],
                        reason="benchmark: simulated provider-sync failure",
                        health=write_txns._health,
                    )
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
                        ops.worker.register(
                            cur,
                            worker_id=WorkerId(f"{base}-{i}"),
                            address=f"tcp://{base}-{i}:1234",
                            metadata=meta,
                            ts=Timestamp.now(),
                            health=write_txns._health,
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
                    write_txns._endpoints.add(cur, _make_endpoint(t, aid))
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
    """Reproduce the production tail when apply_reconcile contends
    with provider-sync failure storms and other write RPCs.
    """
    plan_results = _build_reconcile_inputs(db)
    total_tasks = sum(len(result.observations) for _plan, result in plan_results)
    print(f"  (victim reconcile batch: {len(plan_results)} workers, {total_tasks} tasks)")

    if not plan_results:
        print("  (skipped, no workers)")
        return

    scenarios: list[_ContentionScenario] = [
        _ContentionScenario(name="Contention: apply @ baseline (no contention)"),
        _ContentionScenario(name="Contention: apply + fail_workers", fail_threads=1),
        _ContentionScenario(name="Contention: apply + register burst", register_threads=1),
        _ContentionScenario(name="Contention: apply + add_endpoint storm", endpoint_threads=1),
        _ContentionScenario(
            name="Contention: apply + prod-mix (fail+reg+ep)",
            fail_threads=1,
            register_threads=1,
            endpoint_threads=1,
        ),
        _ContentionScenario(
            name="Contention: apply + heavy storm (2f/2r/2e, chunk=200)",
            fail_threads=2,
            fail_chunk=200,
            fail_interval_s=0.5,
            register_threads=2,
            endpoint_threads=2,
        ),
    ]

    write_db = clone_db(db)
    write_txns = ControllerTestState(write_db)
    # The fail-storm path (ops.worker.fail) only acts on tracker-active workers.
    _seed_health(write_db, write_txns._health)
    try:
        for scenario in scenarios:
            _run_apply_under_contention(
                write_db=write_db,
                write_txns=write_txns,
                plan_results=plan_results,
                **scenario,
            )
    finally:
        write_db.close()
        shutil.rmtree(write_db._db_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Group: control_isolation (control-loop checkout wait under a dashboard storm)
# ---------------------------------------------------------------------------


def _measure_connection_checkout(engine) -> float:
    """Check out a read connection and return the wait in ms.

    This is exactly the ``QueuePool.get`` blocking seen in production controller
    stacks: when every shared connection is busy, a thread needing a read blocks
    here. The query cost after checkout is identical regardless of
    which pool is used, so the checkout wait is the only quantity the dedicated
    control pool changes. A trivial probe query keeps iterations cheap so the
    distribution has enough samples.
    """
    t0 = time.perf_counter()
    conn = engine.connect()
    checkout_ms = (time.perf_counter() - t0) * 1000.0
    try:
        conn.execute(text("SELECT 1")).all()
    finally:
        conn.close()
    return checkout_ms


def _dashboard_storm_read(db, slow: bool, sample_job_id: str | None) -> None:
    """One RPC-shaped read against the shared pool.

    ``slow`` runs the ``GetSchedulerState`` pending+running aggregation (seconds
    on the prod DB — these are what occupy every shared connection and head-of-
    line block the dashboard); otherwise a single-job ``GetJobState``-shape read.
    """
    with db.read_snapshot() as snap:
        if slow:
            snap.execute(
                select(tasks_table.c.task_id, tasks_table.c.priority_band, tasks_table.c.submitted_at_ms).where(
                    tasks_table.c.state == job_pb2.TASK_STATE_PENDING
                )
            ).all()
            snap.execute(
                select(tasks_table.c.task_id, tasks_table.c.current_worker_id).where(
                    tasks_table.c.state == job_pb2.TASK_STATE_RUNNING,
                    tasks_table.c.current_worker_id.is_not(None),
                )
            ).all()
        elif sample_job_id is not None:
            snap.execute(select(jobs_table.c.state).where(jobs_table.c.job_id == sample_job_id)).all()


def _run_control_read_under_storm(
    db: ControllerDB,
    *,
    victim_engine,
    sample_job_id: str | None,
    storm_threads: int,
    slow_every: int,
    duration_s: float,
) -> list[float]:
    """Sample the connection-checkout wait on a victim thread while ``storm_threads``
    hammer the shared read pool with a fast/slow RPC mix. Returns checkout waits (ms).
    """
    stop = threading.Event()
    victim_latencies: list[float] = []
    errors: list[BaseException] = []

    def _victim() -> None:
        try:
            while not stop.is_set():
                victim_latencies.append(_measure_connection_checkout(victim_engine))
                # Sample every 50ms: frequent enough for a tight distribution,
                # but not a busy-loop that would itself contend for connections.
                stop.wait(0.05)
        except BaseException as e:
            errors.append(e)

    def _storm() -> None:
        i = 0
        try:
            while not stop.is_set():
                _dashboard_storm_read(db, slow=(i % slow_every == 0), sample_job_id=sample_job_id)
                i += 1
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_victim, name="control-victim")]
    threads += [threading.Thread(target=_storm, name=f"dash-storm-{i}") for i in range(storm_threads)]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=30.0)
    if errors:
        print(f"    storm thread error: {errors[0]!r}")
    return victim_latencies


def benchmark_control_isolation(db: ControllerDB) -> None:
    """Control-loop connection-checkout wait under a dashboard read storm.

    The control loop's per-tick snapshot must not queue behind a slow dashboard
    read for a connection. This contrasts the victim (a control-loop thread
    needing a read connection) checking out from the **shared** RPC read pool vs
    the **dedicated** control pool while concurrent RPC-shaped reads saturate the
    shared pool — the before/after for the dedicated control read pool.
    """
    with db.read_snapshot() as tx:
        row = tx.execute(select(jobs_table.c.job_id).limit(1)).first()
    sample_job_id = str(row.job_id) if row else None

    # Saturate the shared pool with readers, ~1 in 3 of them a slow aggregation.
    storm_threads = 12
    slow_every = 3
    duration_s = 6.0
    print(f"  (storm: {storm_threads} threads on the shared pool, every {slow_every}th read = slow aggregation)")

    shared = _run_control_read_under_storm(
        db,
        victim_engine=db.sa_read_engine,
        sample_job_id=sample_job_id,
        storm_threads=storm_threads,
        slow_every=slow_every,
        duration_s=duration_s,
    )
    _print_latency_distribution("Control checkout: SHARED pool (contends w/ dashboard)", shared)

    dedicated = _run_control_read_under_storm(
        db,
        victim_engine=db.sa_control_read_engine,
        sample_job_id=sample_job_id,
        storm_threads=storm_threads,
        slow_every=slow_every,
        duration_s=duration_s,
    )
    _print_latency_distribution("Control checkout: DEDICATED pool (isolated)", dedicated)


# ---------------------------------------------------------------------------
# Shared latency / scenario helpers (used by the rpcs and scenario groups)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LatencyPercentiles:
    p50: float
    p95: float
    p99: float
    p999: float
    maximum: float


def _percentiles_ms(latencies: list[float]) -> LatencyPercentiles:
    latencies.sort()
    n = len(latencies)
    return LatencyPercentiles(
        p50=latencies[n // 2],
        p95=latencies[int(n * 0.95)],
        p99=latencies[int(n * 0.99)],
        p999=latencies[min(int(n * 0.999), n - 1)],
        maximum=latencies[-1],
    )


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
    "control_isolation",
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
        ("control_isolation", benchmark_control_isolation),
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


_SERVER_START_TIMEOUT = Duration.from_seconds(10)


def _wait_for_server_start(condition: Callable[[], bool], error_message: str) -> None:
    ExponentialBackoff(initial=0.01, maximum=0.1).wait_until_or_raise(
        condition,
        timeout=_SERVER_START_TIMEOUT,
        error_message=error_message,
    )


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
def serve_cmd(db_path: Path, state_dir: Path) -> None:
    """Boot a Controller(dry_run=True) bound to ``db_path`` and serve over RPC.

    Prints ``READY port=N`` to stdout once the HTTP server is accepting
    connections, then blocks until SIGTERM. Used by ``RpcHarness`` so the
    benchmark process and the controller have independent GILs.
    """
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
    log_stack = build_log_stack(
        log_service_address="",
        local_log_dir=config.local_state_dir / "log-server",
        host=config.host,
        worker_token=None,
    )
    controller = Controller(
        config=config,
        backends={DEFAULT_BACKEND_ID: cast(TaskBackend, _FakeProvider())},
        log_stack=log_stack,
        db=db,
        threads=threads,
    )
    controller.start()
    try:
        _wait_for_server_start(
            lambda: controller._server is not None and controller._server.started,
            "Controller server did not start within 10s",
        )
    except TimeoutError as exc:
        controller.stop()
        raise click.ClickException(str(exc)) from exc

    print(f"READY port={controller.port}", flush=True)

    stop = threading.Event()

    def _shutdown(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    stop.wait()
    controller.stop()
    db.close()


# ---------------------------------------------------------------------------
# Reconcile dispatch + apply tick benchmark (synthetic DB + fake worker)
#
# Unlike the checkpoint-driven groups above, this exercise builds its own DB
# representing a single zephyr-shaped job (one root job, ``num_tasks`` task
# replicas, all dispatched to ``num_workers`` healthy workers), then times the
# four stages of one polling tick:
#
#     1. ``_snapshot_reconcile_inputs`` (DB read + per-job RunTaskRequest
#        template build).
#     2. building the ``ControlSnapshot`` the backend reconciles against
#        (flattening per-worker rows + job specs).
#     3. ``RpcTaskBackend.reconcile(ControlSnapshot)`` (builds the per-worker
#        plans via ``plans_from_snapshot``, then async fanout over Connect RPC
#        to a single in-process fake worker that echoes observations back).
#     4. ``apply_reconcile`` (one DB transaction fanning all per-worker
#        results in as a single batched apply).
#
# A single uvicorn-backed fake worker is mounted on localhost; every
# ``worker_id`` in the DB resolves to that address, isolating the cost of the
# controller-side loop + Connect protocol + asyncio.gather without paying
# NxN port setup.
# ---------------------------------------------------------------------------


def _reconcile_worker_metadata() -> job_pb2.WorkerMetadata:
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
    return meta


def _reconcile_launch_request(
    job_id: JobName,
    *,
    replicas: int,
    payload_bytes: int,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Build a zephyr-shaped LaunchJobRequest with an inflated workdir file.

    The workdir file lives on the job (not the task), and the controller
    pulls it once per job and embeds it in the cached RunTaskRequest
    template. Each ASSIGNED row's DesiredAttempt then carries a full copy
    of that template — so ``payload_bytes`` directly multiplies into the
    wire bytes for every ASSIGNED dispatch.
    """
    entrypoint = job_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-m", "zephyr.worker", "--task-id", "$IRIS_TASK_ID"]
    if payload_bytes > 0:
        entrypoint.workdir_files["state.pkl"] = (b"x" * 64 + b"\x00" * 8) * (payload_bytes // 72 + 1)

    return controller_pb2.Controller.LaunchJobRequest(
        name=job_id.to_wire(),
        entrypoint=entrypoint,
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=replicas,
        task_image="bench:latest",
    )


@dataclasses.dataclass
class SyntheticReconcileState:
    db: ControllerDB
    txns: ControllerTestState
    health: WorkerHealthTracker
    job_id: JobName
    worker_ids: list[WorkerId]
    task_ids: list[JobName]
    address: str  # all workers share this loopback address


def _build_synthetic_reconcile_state(
    *,
    db_dir: Path,
    num_tasks: int,
    num_workers: int,
    payload_bytes: int,
    worker_address: str,
) -> SyntheticReconcileState:
    """Build a fresh DB with one job, ``num_tasks`` tasks, ``num_workers`` workers.

    Every task is ``ASSIGNED`` to a worker — exactly the state the dispatch
    tick has to handle when the controller has just finished scheduling but
    no worker has acknowledged yet. This is the worst case for the
    Reconcile dispatch because every plan carries the full inline spec.
    """
    db = ControllerDB(db_dir=db_dir)
    db.apply_migrations()
    health = WorkerHealthTracker()
    txns = ControllerTestState(db, health=health)

    job_id = JobName.root("bench", "zephyr")
    request = _reconcile_launch_request(job_id, replicas=num_tasks, payload_bytes=payload_bytes)

    meta = _reconcile_worker_metadata()
    worker_ids: list[WorkerId] = []
    chunk = 500
    now = Timestamp.now()
    for chunk_start in range(0, num_workers, chunk):
        with db.transaction() as cur:
            for i in range(chunk_start, min(chunk_start + chunk, num_workers)):
                wid = WorkerId(f"w-{i:05d}")
                worker_ids.append(wid)
                ops.worker.register(
                    cur,
                    worker_id=wid,
                    address=worker_address,
                    metadata=meta,
                    ts=now,
                    health=txns._health,
                    slice_id="",
                    scale_group="bench",
                )
    health.heartbeat(worker_ids, now.epoch_ms())

    with db.transaction() as cur:
        ops.job.submit(cur, job_id=job_id, request=request, ts=now)

    with db.read_snapshot() as tx:
        task_rows = tx.execute(
            select(tasks_table.c.task_id).where(tasks_table.c.job_id == job_id).order_by(tasks_table.c.task_index)
        ).all()
    task_ids = [r.task_id for r in task_rows]
    assert len(task_ids) == num_tasks, f"expected {num_tasks} tasks, got {len(task_ids)}"

    for chunk_start in range(0, num_tasks, chunk):
        slice_tasks = task_ids[chunk_start : chunk_start + chunk]
        assignments = [
            Assignment(task_id=tid, worker_id=worker_ids[(chunk_start + i) % num_workers])
            for i, tid in enumerate(slice_tasks)
        ]
        with db.transaction() as cur:
            ops.task.assign(cur, assignments, health=txns._health)

    return SyntheticReconcileState(
        db=db, txns=txns, health=health, job_id=job_id, worker_ids=worker_ids, task_ids=task_ids, address=worker_address
    )


class _EchoWorker:
    """WorkerService that echoes each desired run as RUNNING.

    Mirrors a worker that has accepted the spec and is about to launch the
    process. Stop intents emit a KILLED observation. The shape matches
    ``WorkerLifecycle.handle_reconcile`` closely enough that the controller's
    apply path exercises its real branches.
    """

    async def reconcile(
        self,
        request: worker_pb2.Worker.ReconcileRequest,
        _ctx: RequestContext,
    ) -> worker_pb2.Worker.ReconcileResponse:
        observed: list[worker_pb2.Worker.AttemptObservation] = []
        for desired in request.desired:
            if desired.HasField("run"):
                observed.append(
                    worker_pb2.Worker.AttemptObservation(
                        attempt_uid=desired.attempt_uid,
                        state=job_pb2.TASK_STATE_RUNNING,
                    )
                )
            elif desired.HasField("stop"):
                observed.append(
                    worker_pb2.Worker.AttemptObservation(
                        attempt_uid=desired.attempt_uid,
                        state=job_pb2.TASK_STATE_KILLED,
                    )
                )
        return worker_pb2.Worker.ReconcileResponse(
            worker_id=request.worker_id,
            observed=observed,
            health=worker_pb2.Worker.WorkerHealth(healthy=True),
        )

    async def get_task_status(self, request, ctx):
        raise NotImplementedError

    async def list_tasks(self, request, ctx):
        raise NotImplementedError

    async def health_check(self, request, ctx):
        return worker_pb2.Worker.HealthResponse(healthy=True)

    async def profile_task(self, request, ctx):
        raise NotImplementedError

    async def get_process_status(self, request, ctx):
        raise NotImplementedError

    async def exec_in_container(self, request, ctx):
        raise NotImplementedError

    async def start_tasks(self, request, ctx):
        raise NotImplementedError

    async def stop_tasks(self, request, ctx):
        raise NotImplementedError

    async def poll_tasks(self, request, ctx):
        raise NotImplementedError


@contextmanager
def _serve_fake_worker():
    app = WorkerServiceASGIApplication(
        cast(WorkerService, _EchoWorker()),
        compressions=IRIS_RPC_COMPRESSIONS,
    )
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="error",
        log_config=None,
        timeout_keep_alive=_CONTROLLER_KEEPALIVE,
    )
    server = uvicorn.Server(config)

    def _run():
        # uvicorn's run() installs signal handlers from the main thread only;
        # drive the asyncio loop manually instead.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()

    thread = threading.Thread(target=_run, name="fake-worker", daemon=True)
    thread.start()

    _wait_for_server_start(lambda: server.started, "fake worker did not start within 10s")

    port = None
    for sock in server.servers or []:
        for s in sock.sockets:
            port = s.getsockname()[1]
            break
        if port is not None:
            break
    if port is None:
        raise RuntimeError("fake worker did not bind a port")
    address = f"127.0.0.1:{port}"
    try:
        yield address
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@dataclasses.dataclass
class _StageTimings:
    snapshot_ms: list[float] = dataclasses.field(default_factory=list)
    reconcile_compute_ms: list[float] = dataclasses.field(default_factory=list)
    rpc_fanout_ms: list[float] = dataclasses.field(default_factory=list)
    apply_ms: list[float] = dataclasses.field(default_factory=list)
    total_ms: list[float] = dataclasses.field(default_factory=list)


def _reconcile_percentiles(xs: list[float]) -> tuple[float, float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2], xs[int(n * 0.95)], xs[int(n * 0.99)], xs[-1])


def _serialized_bytes(plans) -> int:
    return sum(p.request.ByteSize() for p in plans)


def _snapshot_reconcile_inputs(state: SyntheticReconcileState) -> tuple[ReconcileInputs, dict[WorkerId, str]]:
    """Replicate ``controller._snapshot_reconcile_inputs`` against the synthetic DB."""
    db = state.db
    health = state.health
    with db.read_snapshot() as snap:
        addresses = reads.list_active_healthy_workers(snap, health)
        if not addresses:
            return ReconcileInputs(job_specs={}, worker_ids=[], rows_by_worker={}), {}
        worker_ids = list(addresses)
        target_ids = set(worker_ids)
        raw_rows = snap.execute(
            select(
                task_attempts_table.c.worker_id,
                tasks_table.c.task_id,
                task_attempts_table.c.attempt_id,
                tasks_table.c.state.label("task_state"),
                task_attempts_table.c.state.label("attempt_state"),
                tasks_table.c.job_id,
                task_attempts_table.c.attempt_uid,
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
            ),
        ).all()
        rows = [row for row in raw_rows if row.worker_id in target_ids]
        templates_by_job: dict[JobName, job_pb2.RunTaskRequest | None] = {}
        for row in rows:
            if row.task_state != job_pb2.TASK_STATE_ASSIGNED:
                continue
            if row.job_id not in templates_by_job:
                templates_by_job[row.job_id] = snap.caches[RunTemplatesProjection].get(snap, row.job_id)

    rows_by_worker: dict[WorkerId, list[ReconcileRow]] = {wid: [] for wid in worker_ids}
    for row in rows:
        rows_by_worker[WorkerId(str(row.worker_id))].append(
            ReconcileRow(
                worker_id=WorkerId(str(row.worker_id)),
                task_id=row.task_id,
                attempt_id=int(row.attempt_id),
                task_state=int(row.task_state),
                attempt_state=int(row.attempt_state),
                job_id=row.job_id,
                attempt_uid=AttemptUid(str(row.attempt_uid)),
            )
        )
    job_specs = {jid: spec for jid, spec in templates_by_job.items() if spec is not None}
    inputs = ReconcileInputs(job_specs=job_specs, worker_ids=worker_ids, rows_by_worker=rows_by_worker)
    return inputs, addresses


@dataclasses.dataclass
class _PrebuiltWorkerSource:
    """Hands the backend a reconcile snapshot the benchmark built itself.

    The synthetic reconcile benchmark times snapshot assembly separately, then
    drives ``RpcTaskBackend.reconcile`` against the prebuilt snapshot; the backend
    sources its snapshot through this O(1) stub so only the RPC fan-out is timed.
    """

    snapshot: ControlSnapshot
    # The benchmark times only the RPC fan-out (``_observe_fleet``), which never
    # folds liveness; the tracker is present solely to satisfy ``BackendWorkerStore``.
    health: WorkerHealthTracker = dataclasses.field(default_factory=WorkerHealthTracker)

    def reconcile_snapshot(self) -> ControlSnapshot:
        return self.snapshot

    def scheduling_inputs(self):
        raise NotImplementedError

    def worker_status(self):
        raise NotImplementedError

    def transition_snapshot(self, **kwargs):
        raise NotImplementedError


def _one_reconcile_tick(state: SyntheticReconcileState, provider: RpcTaskBackend) -> tuple[float, float, float, float]:
    """Run one full reconcile tick. Returns (snapshot, compute, rpc, apply) ms."""
    t0 = time.perf_counter()
    inputs, addresses = _snapshot_reconcile_inputs(state)
    t1 = time.perf_counter()
    snapshot = ControlSnapshot(
        worker_addresses=addresses,
        reconcile_rows=[row for rows in inputs.rows_by_worker.values() for row in rows],
        timeout_rows=[],
        job_specs=inputs.job_specs,
    )
    t2 = time.perf_counter()
    # The backend sources its own reconcile snapshot; the benchmark prebuilds it
    # (measured above) and hands it back through a stub store so the RPC fan-out
    # is what t2..t3 times. The stub implements only the fan-out read surface (it
    # never folds liveness or tears down), so it is cast to the full BackendWorkerStore.
    provider._store = cast(BackendWorkerStore, _PrebuiltWorkerSource(snapshot))
    worker_results = provider._observe_fleet().worker_results
    t3 = time.perf_counter()
    now = Timestamp.now()
    with state.txns._db.transaction() as cur:
        effects = apply_reconcile(CursorTransitionReader(cur), worker_results, now=now)
        commit_effects(cur, effects)
    t4 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000


def _measure_full_tick(
    state: SyntheticReconcileState,
    provider: RpcTaskBackend,
    *,
    n_iters: int,
) -> tuple[_StageTimings, _StageTimings]:
    """Return ``(dispatch, steady_state)`` timings.

    The first reconcile tick runs against a fully-ASSIGNED DB and triggers
    the spec-carrying RPC dispatch — that's the **dispatch tick**, measured
    once. The fake worker then echoes RUNNING observations and the apply
    layer transitions every task. Subsequent ticks have nothing inline to
    send (the worker is already running) and exercise the **steady-state
    polling tick** that runs every 250 ms in production.
    """
    _snapshot_reconcile_inputs(state)

    dispatch = _StageTimings()
    s, c, r, a = _one_reconcile_tick(state, provider)
    dispatch.snapshot_ms.append(s)
    dispatch.reconcile_compute_ms.append(c)
    dispatch.rpc_fanout_ms.append(r)
    dispatch.apply_ms.append(a)
    dispatch.total_ms.append(s + c + r + a)

    steady = _StageTimings()
    for _ in range(n_iters):
        s, c, r, a = _one_reconcile_tick(state, provider)
        steady.snapshot_ms.append(s)
        steady.reconcile_compute_ms.append(c)
        steady.rpc_fanout_ms.append(r)
        steady.apply_ms.append(a)
        steady.total_ms.append(s + c + r + a)
    return dispatch, steady


def _measure_compute_only(state: SyntheticReconcileState, *, n_iters: int) -> tuple[list[float], int]:
    """Time just ``build_reconcile_plans`` (pure compute) — no RPC, no DB transaction."""
    inputs, _addresses = _snapshot_reconcile_inputs(state)
    plans = build_reconcile_plans(inputs)  # warmup
    total_bytes = _serialized_bytes(plans)

    times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _plans = build_reconcile_plans(inputs)
        times.append((time.perf_counter() - t0) * 1000)
    return times, total_bytes


def _summarize_reconcile(label: str, xs: list[float], unit: str = "ms") -> str:
    if not xs:
        return f"{label}: (no samples)"
    p50, p95, p99, mx = _reconcile_percentiles(xs)
    return (
        f"{label:<28s} mean={statistics.mean(xs):8.1f}{unit}  "
        f"p50={p50:7.1f}{unit}  p95={p95:7.1f}{unit}  p99={p99:7.1f}{unit}  max={mx:7.1f}{unit}  (n={len(xs)})"
    )


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def _run_reconcile_scenario(
    num_tasks: int,
    tasks_per_worker: int,
    payload_bytes: int,
    n_iters: int,
    parallelism: int,
) -> dict:
    """Run one reconcile scenario and return a dict of measurements."""
    num_workers = max(1, (num_tasks + tasks_per_worker - 1) // tasks_per_worker)
    tmp = Path(tempfile.mkdtemp(prefix="bench_reconcile_"))
    print(
        f"\n=== num_tasks={num_tasks}  num_workers={num_workers}  "
        f"tasks_per_worker={tasks_per_worker}  payload={_human_bytes(payload_bytes)} ==="
    )
    sys.stdout.flush()
    state = None
    try:
        with _serve_fake_worker() as address:
            t0 = time.perf_counter()
            state = _build_synthetic_reconcile_state(
                db_dir=tmp,
                num_tasks=num_tasks,
                num_workers=num_workers,
                payload_bytes=payload_bytes,
                worker_address=address,
            )
            build_s = time.perf_counter() - t0
            print(f"  build:                       {build_s * 1000:.0f} ms")

            stub_factory = RpcWorkerStubFactory()
            provider = RpcTaskBackend(stub_factory=stub_factory, parallelism=parallelism)
            try:
                compute_ms, total_bytes = _measure_compute_only(state, n_iters=n_iters)
                per_worker_avg = total_bytes / max(1, len(state.worker_ids))
                print(
                    f"  ReconcileRequest payload:    total={_human_bytes(total_bytes)}  "
                    f"avg/worker={_human_bytes(int(per_worker_avg))}"
                )
                print("  " + _summarize_reconcile("compute (build_reconcile_plans)", compute_ms))

                dispatch, steady = _measure_full_tick(state, provider, n_iters=n_iters)
                print("  -- dispatch tick (all ASSIGNED, full spec on the wire) --")
                for label, xs in [
                    ("snapshot", dispatch.snapshot_ms),
                    ("compute", dispatch.reconcile_compute_ms),
                    ("RPC fanout", dispatch.rpc_fanout_ms),
                    ("apply", dispatch.apply_ms),
                    ("total tick", dispatch.total_ms),
                ]:
                    print("  " + _summarize_reconcile(label, xs))
                print("  -- steady-state tick (all RUNNING, no inline spec) --")
                for label, xs in [
                    ("snapshot", steady.snapshot_ms),
                    ("compute", steady.reconcile_compute_ms),
                    ("RPC fanout", steady.rpc_fanout_ms),
                    ("apply", steady.apply_ms),
                    ("total tick", steady.total_ms),
                ]:
                    print("  " + _summarize_reconcile(label, xs))

                return {
                    "num_tasks": num_tasks,
                    "num_workers": num_workers,
                    "tasks_per_worker": tasks_per_worker,
                    "payload_bytes": payload_bytes,
                    "total_request_bytes": total_bytes,
                    "compute_ms": compute_ms,
                    "dispatch": dataclasses.asdict(dispatch),
                    "steady": dataclasses.asdict(steady),
                }
            finally:
                provider.close()
    finally:
        if state is not None:
            state.db.close()
        shutil.rmtree(tmp, ignore_errors=True)


@main.command("reconcile")
@click.option("--num-tasks", "num_tasks", type=int, default=5000, help="Number of zephyr tasks.")
@click.option("--tasks-per-worker", type=int, default=64, help="Round-robin density (smaller = more workers).")
@click.option("--payload-bytes", type=int, default=100_000, help="Per-job workdir-file size.")
@click.option("--n-iters", type=int, default=5, help="Number of full reconcile ticks to time.")
@click.option("--parallelism", type=int, default=128, help="Async RPC concurrency cap.")
@click.option("--scale-sweep", is_flag=True, help="Sweep across multiple scales for the final report.")
def reconcile_cmd(
    num_tasks: int,
    tasks_per_worker: int,
    payload_bytes: int,
    n_iters: int,
    parallelism: int,
    scale_sweep: bool,
) -> None:
    """Benchmark the Reconcile dispatch + apply tick at zephyr scale.

    Builds a synthetic controller DB with one zephyr-shaped job
    (``num_tasks`` task replicas dispatched across enough workers to hit
    ``tasks_per_worker`` round-robin density), spins up a single in-process
    fake worker on localhost, and times each stage of the polling tick.
    """
    if scale_sweep:
        scenarios: list[tuple[int, int, int]] = [
            # (num_tasks, tasks_per_worker, payload_bytes)
            (500, 64, 1_000),
            (500, 64, 100_000),
            (2000, 64, 1_000),
            (2000, 64, 100_000),
            (5000, 64, 1_000),
            (5000, 64, 10_000),
            (5000, 64, 100_000),
            (5000, 1, 100_000),  # zero-fanout density: 1 task per worker
            (5000, 8, 100_000),  # intermediate density
        ]
    else:
        scenarios = [(num_tasks, tasks_per_worker, payload_bytes)]

    for nt, tpw, pb in scenarios:
        _run_reconcile_scenario(nt, tpw, pb, n_iters=n_iters, parallelism=parallelism)


if __name__ == "__main__":
    main()
