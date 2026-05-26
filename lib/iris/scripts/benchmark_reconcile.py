#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the Reconcile RPC dispatch + apply loop.

Synthesizes a fresh controller DB representing a single zephyr-shaped job
(one root job, ``num_tasks`` task replicas, all dispatched to ``num_workers``
healthy workers), then exercises the four stages of one polling tick:

    1. ``_snapshot_reconcile_inputs``  (DB read + per-job RunTaskRequest
       template build).
    2. ``reconcile_workers(inputs)``  (pure-compute: copies the spec proto
       into a ``DesiredAttempt`` for every ASSIGNED row).
    3. ``WorkerProvider.reconcile_workers(plans, addresses)``  (async fanout
       over Connect RPC to a single in-process fake worker that echoes
       observations back).
    4. ``ControllerTransitions.apply_reconcile_result``  (one DB transaction
       fanning all per-worker results in).

The fake worker is mounted once on localhost; every "worker_id" in the DB
resolves to the same address. That isolates the cost of the controller-side
loop + the Connect protocol + asyncio.gather, without paying NxN port setup.

Usage:
    uv run python lib/iris/scripts/benchmark_reconcile.py \\
        --num-tasks 5000 --tasks-per-worker 64 --payload-bytes 100000
    uv run python lib/iris/scripts/benchmark_reconcile.py --scale-sweep
"""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import statistics
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import click
import uvicorn
from connectrpc.request import RequestContext
from iris.cluster.controller import reads
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.reconcile import ReconcileInputs, ReconcileRow, reconcile_workers
from iris.cluster.controller.schema import task_attempts_table, tasks_table
from iris.cluster.controller.transitions import Assignment, ControllerTransitions
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.controller.worker_provider import RpcWorkerStubFactory, WorkerProvider
from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.worker_connect import WorkerServiceASGIApplication
from rigging.timing import Timestamp
from sqlalchemy import select

# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------


def _make_worker_metadata() -> job_pb2.WorkerMetadata:
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


def _make_launch_request(
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
        # Single big workdir file. The proto field is bytes; pad with a
        # repeating pattern so gzip-style compressors can still see structure
        # (matches realistic pickled state).
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
class SyntheticState:
    db: ControllerDB
    txns: ControllerTransitions
    health: WorkerHealthTracker
    job_id: JobName
    worker_ids: list[WorkerId]
    task_ids: list[JobName]
    address: str  # all workers share this loopback address


def build_synthetic_state(
    *,
    db_dir: Path,
    num_tasks: int,
    num_workers: int,
    payload_bytes: int,
    worker_address: str,
) -> SyntheticState:
    """Build a fresh DB with one job, ``num_tasks`` tasks, and ``num_workers`` workers.

    Every task is ``ASSIGNED`` to a worker — exactly the state the dispatch
    tick has to handle when the controller has just finished scheduling but
    no worker has acknowledged yet. This is the worst case for the
    Reconcile dispatch because every plan carries the full inline spec.
    """
    db = ControllerDB(db_dir=db_dir)
    db.apply_migrations()
    health = WorkerHealthTracker()
    txns = ControllerTransitions(db, health=health)

    job_id = JobName.root("bench", "zephyr")
    request = _make_launch_request(job_id, replicas=num_tasks, payload_bytes=payload_bytes)

    meta = _make_worker_metadata()
    worker_ids: list[WorkerId] = []
    # Register workers in chunks to keep each transaction small.
    chunk = 500
    now = Timestamp.now()
    for chunk_start in range(0, num_workers, chunk):
        with db.transaction() as cur:
            for i in range(chunk_start, min(chunk_start + chunk, num_workers)):
                wid = WorkerId(f"w-{i:05d}")
                worker_ids.append(wid)
                txns.register_or_refresh_worker(
                    cur,
                    worker_id=wid,
                    address=worker_address,
                    metadata=meta,
                    ts=now,
                    slice_id="",
                    scale_group="bench",
                )
    health.heartbeat(worker_ids, now.epoch_ms())

    # Submit the job, then assign each task to a worker round-robin.
    with db.transaction() as cur:
        txns.submit_job(cur, job_id, request, now)

    with db.read_snapshot() as tx:
        task_rows = tx.execute(
            select(tasks_table.c.task_id).where(tasks_table.c.job_id == job_id).order_by(tasks_table.c.task_index)
        ).all()
    task_ids = [r.task_id for r in task_rows]
    assert len(task_ids) == num_tasks, f"expected {num_tasks} tasks, got {len(task_ids)}"

    # Assign in chunks to keep transactions bounded.
    for chunk_start in range(0, num_tasks, chunk):
        slice_tasks = task_ids[chunk_start : chunk_start + chunk]
        assignments = [
            Assignment(task_id=tid, worker_id=worker_ids[(chunk_start + i) % num_workers])
            for i, tid in enumerate(slice_tasks)
        ]
        with db.transaction() as cur:
            txns.queue_assignments(cur, assignments)

    return SyntheticState(
        db=db, txns=txns, health=health, job_id=job_id, worker_ids=worker_ids, task_ids=task_ids, address=worker_address
    )


# ---------------------------------------------------------------------------
# Fake worker (single ASGI server; all addresses point here)
# ---------------------------------------------------------------------------


class _EchoWorker:
    """Implements WorkerService.reconcile by echoing each desired run as RUNNING.

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

    # Unused, but the WorkerService protocol enumerates them all.
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

    async def ping(self, request, ctx):
        return worker_pb2.Worker.PingResponse(healthy=True)

    # Legacy StartTasks/StopTasks/PollTasks endpoints still exist on the
    # WorkerService protocol on main; provide stubs so endpoint resolution
    # in ConnectASGIApplication succeeds even though we never call them.
    async def start_tasks(self, request, ctx):
        raise NotImplementedError

    async def stop_tasks(self, request, ctx):
        raise NotImplementedError

    async def poll_tasks(self, request, ctx):
        raise NotImplementedError


@contextmanager
def serve_fake_worker():
    app = WorkerServiceASGIApplication(
        _EchoWorker(),
        # Reuse the same compressions the production worker accepts so the
        # client side of WorkerProvider negotiates an apples-to-apples
        # transport.
        compressions=IRIS_RPC_COMPRESSIONS,
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", log_config=None, timeout_keep_alive=120)
    server = uvicorn.Server(config)

    def _run():
        # uvicorn's run() installs signal handlers from the main thread only.
        # Bypass that by manually driving the asyncio loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()

    thread = threading.Thread(target=_run, name="fake-worker", daemon=True)
    thread.start()

    # Wait for the server to bind.
    deadline = time.time() + 10
    while time.time() < deadline:
        if server.started:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("fake worker did not start in 10s")

    # Resolve the bound port.
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


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StageTimings:
    snapshot_ms: list[float] = dataclasses.field(default_factory=list)
    reconcile_compute_ms: list[float] = dataclasses.field(default_factory=list)
    rpc_fanout_ms: list[float] = dataclasses.field(default_factory=list)
    apply_ms: list[float] = dataclasses.field(default_factory=list)
    total_ms: list[float] = dataclasses.field(default_factory=list)


def _percentiles(xs: list[float]) -> tuple[float, float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2], xs[int(n * 0.95)], xs[int(n * 0.99)], xs[-1])


def _serialized_bytes(plans) -> int:
    return sum(p.request.ByteSize() for p in plans)


def _run_snapshot(state: SyntheticState) -> tuple[ReconcileInputs, dict[WorkerId, str]]:
    """Replicate controller._snapshot_reconcile_inputs without touching the live controller."""
    db = state.db
    health = state.health
    txns = state.txns
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
                templates_by_job[row.job_id] = txns.run_request_template(snap, row.job_id)

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


def _one_tick(state: SyntheticState, provider: WorkerProvider) -> tuple[float, float, float, float]:
    """Run one full reconcile tick. Returns (snapshot, compute, rpc, apply) ms."""
    t0 = time.perf_counter()
    inputs, addresses = _run_snapshot(state)
    t1 = time.perf_counter()
    plans = reconcile_workers(inputs)
    t2 = time.perf_counter()
    results = provider.reconcile_workers(plans, addresses, use_reconcile_rpc=True)
    t3 = time.perf_counter()
    plan_by_worker = {p.worker_id: p for p in plans}
    now = Timestamp.now()
    with state.db.transaction() as cur:
        for result in results:
            plan = plan_by_worker[result.worker_id]
            state.txns.apply_reconcile_result(cur, plan, result, now)
    t4 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000


def measure_full_tick(
    state: SyntheticState,
    provider: WorkerProvider,
    *,
    n_iters: int,
) -> tuple[StageTimings, StageTimings]:
    """Return ``(dispatch, steady_state)`` timings.

    The first reconcile tick runs against a fully-ASSIGNED DB and triggers
    the spec-carrying RPC dispatch — that's the **dispatch tick**, measured
    once. The fake worker then echoes RUNNING observations and the apply
    layer transitions every task. Subsequent ticks have nothing inline to
    send (the worker is already running) and exercise the **steady-state
    polling tick** that runs every 250 ms in production.
    """
    # Warm up DB caches with a no-op snapshot (without consuming state).
    _run_snapshot(state)

    dispatch = StageTimings()
    s, c, r, a = _one_tick(state, provider)
    dispatch.snapshot_ms.append(s)
    dispatch.reconcile_compute_ms.append(c)
    dispatch.rpc_fanout_ms.append(r)
    dispatch.apply_ms.append(a)
    dispatch.total_ms.append(s + c + r + a)

    steady = StageTimings()
    for _ in range(n_iters):
        s, c, r, a = _one_tick(state, provider)
        steady.snapshot_ms.append(s)
        steady.reconcile_compute_ms.append(c)
        steady.rpc_fanout_ms.append(r)
        steady.apply_ms.append(a)
        steady.total_ms.append(s + c + r + a)
    return dispatch, steady


def measure_compute_only(state: SyntheticState, *, n_iters: int) -> tuple[list[float], int]:
    """Time just ``reconcile_workers`` (pure compute) — no RPC, no DB transaction."""
    inputs, _addresses = _run_snapshot(state)
    plans = reconcile_workers(inputs)  # warmup
    total_bytes = _serialized_bytes(plans)

    times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _plans = reconcile_workers(inputs)
        times.append((time.perf_counter() - t0) * 1000)
    return times, total_bytes


def _summarize(label: str, xs: list[float], unit: str = "ms") -> str:
    if not xs:
        return f"{label}: (no samples)"
    p50, p95, p99, mx = _percentiles(xs)
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


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------


def run_scenario(
    num_tasks: int,
    tasks_per_worker: int,
    payload_bytes: int,
    n_iters: int,
    parallelism: int,
) -> dict:
    """Run one scenario and return a dict of measurements."""
    num_workers = max(1, (num_tasks + tasks_per_worker - 1) // tasks_per_worker)
    tmp = Path(tempfile.mkdtemp(prefix="bench_reconcile_"))
    print(
        f"\n=== num_tasks={num_tasks}  num_workers={num_workers}  "
        f"tasks_per_worker={tasks_per_worker}  payload={_human_bytes(payload_bytes)} ==="
    )
    sys.stdout.flush()
    state = None
    try:
        with serve_fake_worker() as address:
            t0 = time.perf_counter()
            state = build_synthetic_state(
                db_dir=tmp,
                num_tasks=num_tasks,
                num_workers=num_workers,
                payload_bytes=payload_bytes,
                worker_address=address,
            )
            build_s = time.perf_counter() - t0
            print(f"  build:                       {build_s * 1000:.0f} ms")

            stub_factory = RpcWorkerStubFactory()
            provider = WorkerProvider(stub_factory=stub_factory, parallelism=parallelism)
            try:
                # Pure compute (lets us isolate proto build from DB/RPC cost).
                compute_ms, total_bytes = measure_compute_only(state, n_iters=n_iters)
                per_worker_avg = total_bytes / max(1, len(state.worker_ids))
                print(
                    f"  ReconcileRequest payload:    total={_human_bytes(total_bytes)}  "
                    f"avg/worker={_human_bytes(int(per_worker_avg))}"
                )
                print("  " + _summarize("compute (reconcile_workers)", compute_ms))

                dispatch, steady = measure_full_tick(state, provider, n_iters=n_iters)
                print("  -- dispatch tick (all ASSIGNED, full spec on the wire) --")
                for label, xs in [
                    ("snapshot", dispatch.snapshot_ms),
                    ("compute", dispatch.reconcile_compute_ms),
                    ("RPC fanout", dispatch.rpc_fanout_ms),
                    ("apply", dispatch.apply_ms),
                    ("total tick", dispatch.total_ms),
                ]:
                    print("  " + _summarize(label, xs))
                print("  -- steady-state tick (all RUNNING, no inline spec) --")
                for label, xs in [
                    ("snapshot", steady.snapshot_ms),
                    ("compute", steady.reconcile_compute_ms),
                    ("RPC fanout", steady.rpc_fanout_ms),
                    ("apply", steady.apply_ms),
                    ("total tick", steady.total_ms),
                ]:
                    print("  " + _summarize(label, xs))

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


@click.command()
@click.option("--num-tasks", "num_tasks", type=int, default=5000, help="Number of zephyr tasks.")
@click.option("--tasks-per-worker", type=int, default=64, help="Round-robin density (smaller = more workers).")
@click.option("--payload-bytes", type=int, default=100_000, help="Per-job workdir-file size.")
@click.option("--n-iters", type=int, default=5, help="Number of full reconcile ticks to time.")
@click.option("--parallelism", type=int, default=128, help="Async RPC concurrency cap.")
@click.option("--scale-sweep", is_flag=True, help="Sweep across multiple scales for the final report.")
def main(
    num_tasks: int,
    tasks_per_worker: int,
    payload_bytes: int,
    n_iters: int,
    parallelism: int,
    scale_sweep: bool,
) -> None:
    scenarios: list[tuple[int, int, int]]
    if scale_sweep:
        scenarios = [
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
        run_scenario(nt, tpw, pb, n_iters=n_iters, parallelism=parallelism)


if __name__ == "__main__":
    main()
