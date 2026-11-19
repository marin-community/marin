# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the cost of copying controller rows to Finelog, on synthetic data.

Runs controller write paths against a fresh synthetic controller DB, once with
no row tables attached (``none``) and once with them attached (``rows``), and
reports per-transaction latency and the rows each transaction captured. The
write lock is held for a whole transaction, so in this single-threaded run the
transaction time is the write-lock hold time.

Every value is made up. The background DB holds 7 days of finished work at the
fleet-wide rate (8,000 jobs, 15,000 tasks and 17,000 attempts a day: about
56,000 jobs and configs, 105,000 tasks and 119,000 attempts, ~336,000 rows),
so index lookups run against a realistically sized DB.

The script only uses APIs shared by every capture implementation, so the same
file can be run from different checkouts and the results compared::

    uv run python lib/iris/scripts/benchmark_observability_rows.py run --out triggers.json
    uv run python lib/iris/scripts/benchmark_observability_rows.py compare hook.json triggers.json

``run`` also writes the captured rows next to ``--out`` (``*.rows.json``);
``compare`` reports per-scenario latency side by side and any rows one run
captured that the other did not.
"""

import dataclasses
import json
import shutil
import sqlite3
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, cast

import click
from finelog.client.log_client import Table as FinelogTable
from iris.cluster.controller import db as db_module
from iris.cluster.controller import observability_rows, ops, writes
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.controller.observability_rows import ObservabilityRowTables
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.test_util import FakeStatsTable
from iris.testing.controller import (
    make_job_request,
    make_worker_metadata,
    query_tasks_for_job,
    register_worker,
    submit_job,
)
from iris.testing.controller_state import ControllerTestState
from iris.testing.transitions import WorkerTaskUpdates, apply_task_observations
from rigging.timing import Timestamp

DAYS = 7
JOBS_PER_DAY = 8_000
NUM_WORKERS = 80
CHUNK = 500  # tasks per scheduling or reconcile transaction
START_MS = 1_790_000_000_000
# Fields that differ between runs whatever the capture code does: the row write time
# and the randomly minted attempt uid.
_UNSTABLE_FIELDS = ("ts", "attempt_uid")


class FakeClock:
    """``Timestamp.now`` for the run: advances only when the workload says so, so
    both runs see the same times however often capture code reads the clock."""

    def __init__(self) -> None:
        self.ms = START_MS

    def now(self) -> Timestamp:
        return Timestamp.from_ms(self.ms)

    def tick(self) -> None:
        self.ms += 1_000


@dataclass
class Measurement:
    txn_ms: list[float] = field(default_factory=list)
    capture_ms: float = 0.0


@contextmanager
def measured(db: ControllerDB, m: Measurement) -> Iterator[None]:
    """Time every write transaction and the capture step inside it."""
    original = db.transaction

    @contextmanager
    def timed():
        start = time.perf_counter()
        with original() as cur:
            yield cur
        m.txn_ms.append((time.perf_counter() - start) * 1000)

    def time_capture(module, name: str) -> Callable[[], None]:
        inner = getattr(module, name)

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                m.capture_ms += (time.perf_counter() - start) * 1000

        setattr(module, name, wrapper)
        return lambda: setattr(module, name, inner)

    # This file is also run from checkouts that mark changes at each write site instead of
    # using triggers; there the capture step is ``observability_rows._snapshot_or_drop``.
    if hasattr(db_module, "capture_changes"):
        restore = time_capture(db_module, "capture_changes")
    else:
        restore = time_capture(observability_rows, "_snapshot_or_drop")
    db.transaction = timed
    try:
        yield
    finally:
        db.transaction = original
        restore()


# ---------------------------------------------------------------------------
# Background DB
# ---------------------------------------------------------------------------


def _template_rows(conn: sqlite3.Connection, table: str, where: str, args: tuple) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(f"SELECT * FROM {table} WHERE {where}", args)]


def build_background_db(path: Path) -> dict[str, int]:
    """Write ``DAYS`` of finished jobs: seeded through controller code, then copied in bulk."""
    db = ControllerDB(db_dir=path)
    db.apply_migrations()
    state = ControllerTestState(db, health=WorkerHealthTracker())
    worker = register_worker(state, "w-seed", "10.0.0.1:1", make_worker_metadata())
    seed = submit_job(state, "/bench/seed", make_job_request("/bench/seed", replicas=2, max_retries_failure=1))
    for task in seed:
        _assign(state, [task.task_id], [worker])
        _observe(state, [task.task_id], job_pb2.TASK_STATE_SUCCEEDED, worker_of={task.task_id: worker})
    db.close()

    conn = sqlite3.connect(path / ControllerDB.DB_FILENAME)
    seed_id = "/bench/seed"
    job = _template_rows(conn, "jobs", "job_id = ?", (seed_id,))[0]
    config = _template_rows(conn, "job_config", "job_id = ?", (seed_id,))[0]
    task = _template_rows(conn, "tasks", "task_id = ?", (f"{seed_id}/0",))[0]
    attempt = _template_rows(conn, "task_attempts", "task_id = ?", (f"{seed_id}/0",))[0]

    def renamed(row: dict, old: str, new: str, **overrides) -> dict:
        out = {k: v.replace(old, new) if isinstance(v, str) else v for k, v in row.items()}
        out.update(overrides)
        return out

    jobs, configs, tasks, attempts = [], [], [], []
    for i in range(DAYS * JOBS_PER_DAY):
        job_id = f"/bench/bg-{i:06d}"
        jobs.append(renamed(job, seed_id, job_id))
        configs.append(renamed(config, seed_id, job_id))
        # 7 jobs in 8 have two tasks; one in four has a retried first task.
        for index in range(2 if i % 8 else 1):
            task_id = f"{job_id}/{index}"
            tasks.append(renamed(task, seed_id, job_id, task_id=task_id, task_index=index))
            retries = 2 if (index == 0 and i % 4 == 0) else 1
            for attempt_id in range(retries):
                attempts.append(
                    renamed(
                        attempt,
                        seed_id,
                        job_id,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        attempt_uid=f"bg-{i}-{index}-{attempt_id}",
                        worker_id=None,
                    )
                )
    for table, rows in (("jobs", jobs), ("job_config", configs), ("tasks", tasks), ("task_attempts", attempts)):
        columns = list(rows[0])
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            [tuple(r[c] for c in columns) for r in rows],
        )
    conn.execute("DELETE FROM workers")
    conn.commit()
    counts = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("jobs", "job_config", "tasks", "task_attempts")
    }
    conn.close()
    return counts


# ---------------------------------------------------------------------------
# Workload helpers (all through controller write paths)
# ---------------------------------------------------------------------------


def _assign(state: ControllerTestState, task_ids: list[JobName], workers: list[WorkerId]) -> None:
    for start in range(0, len(task_ids), CHUNK):
        chunk = task_ids[start : start + CHUNK]
        assignments = [Assignment(task_id=t, worker_id=workers[(start + i) % len(workers)]) for i, t in enumerate(chunk)]
        with state._db.transaction() as cur:
            ops.task.assign(cur, assignments, health=state._health)


def _observe(
    state: ControllerTestState,
    task_ids: list[JobName],
    new_state: int,
    *,
    worker_of: dict[JobName, WorkerId],
    status_message: str | None = None,
) -> None:
    for start in range(0, len(task_ids), CHUNK):
        chunk = task_ids[start : start + CHUNK]
        attempts = {
            r.task_id: r.current_attempt_id
            for r in query_tasks_for_job(state, _job_of(chunk[0]))
            if r.task_id in set(chunk)
        }
        by_worker: dict[WorkerId, list[TaskUpdate]] = {}
        for t in chunk:
            by_worker.setdefault(worker_of[t], []).append(
                TaskUpdate(task_id=t, attempt_id=attempts[t], new_state=new_state, status_message=status_message)
            )
        with state._db.transaction() as cur:
            apply_task_observations(
                cur,
                [WorkerTaskUpdates(worker_id=w, updates=u) for w, u in by_worker.items()],
                health=state._health,
                now=Timestamp.now(),
            )


def _job_of(task_id: JobName) -> JobName:
    job_id = task_id.parent
    assert job_id is not None
    return job_id


class RunningJob(NamedTuple):
    job_id: JobName
    tasks: list[JobName]
    worker_of: dict[JobName, WorkerId]


def _running_job(
    state: ControllerTestState, workers: list[WorkerId], name: str, replicas: int, **request_kwargs
) -> RunningJob:
    tasks = [t.task_id for t in submit_job(state, name, make_job_request(name, replicas=replicas, **request_kwargs))]
    worker_of = {t: workers[i % len(workers)] for i, t in enumerate(tasks)}
    _assign(state, tasks, workers)
    _observe(state, tasks, job_pb2.TASK_STATE_RUNNING, worker_of=worker_of)
    return RunningJob(_job_of(tasks[0]), tasks, worker_of)


# ---------------------------------------------------------------------------
# Scenarios: ``setup`` is untimed and returns the timed step.
# ---------------------------------------------------------------------------


def scenario_submit(state, workers, clock):
    def run():
        for i in range(1_000):
            submit_job(state, f"/bench/small-{i:04d}", make_job_request(f"/bench/small-{i:04d}", replicas=4))
            clock.tick()
        submit_job(state, "/bench/wide", make_job_request("/bench/wide", replicas=5_000))

    return run


def scenario_schedule(state, workers, clock):
    tasks = [t.task_id for t in submit_job(state, "/bench/wide", make_job_request("/bench/wide", replicas=5_000))]
    clock.tick()
    return lambda: _assign(state, tasks, workers)


def scenario_finish(state, workers, clock):
    job = _running_job(state, workers, "/bench/wide", 5_000)
    clock.tick()
    return lambda: _observe(state, job.tasks, job_pb2.TASK_STATE_SUCCEEDED, worker_of=job.worker_of)


def scenario_status_only(state, workers, clock):
    job = _running_job(state, workers, "/bench/wide", 5_000)
    clock.tick()
    return lambda: _observe(
        state, job.tasks, job_pb2.TASK_STATE_RUNNING, worker_of=job.worker_of, status_message="made-up pending reason"
    )


def scenario_remove_worker(state, workers, clock):
    for k in range(20):
        _running_job(state, [workers[k]], f"/bench/on-{k:02d}", 64)
    clock.tick()

    def run():
        for k in range(20):
            with state._db.transaction() as cur:
                writes.remove_worker(cur, workers[k], state._health)
            clock.tick()

    return run


def _cancel(state, job_id: JobName) -> None:
    with state._db.transaction() as cur:
        ops.job.cancel(cur, job_id=job_id, reason="made-up cancel")


def scenario_cancel_5k(state, workers, clock):
    job = _running_job(state, workers, "/bench/wide", 5_000)
    clock.tick()
    return lambda: _cancel(state, job.job_id)


def scenario_cancel_12k(state, workers, clock):
    # A job holds at most 10,000 tasks (MAX_REPLICAS_PER_JOB), so the 12,000 come from
    # two 6,000-task children that the parent's cancel cascades to.
    parent = _running_job(state, workers, "/bench/huge", 1).job_id
    for child in ("a", "b"):
        _running_job(state, workers, f"/bench/huge/{child}", 6_000)
    clock.tick()
    return lambda: _cancel(state, parent)


def scenario_preempt_retry(state, workers, clock):
    job = _running_job(state, workers, "/bench/preempt", 1_000, max_retries_preemption=1)
    clock.tick()

    def run():
        _observe(state, job.tasks, job_pb2.TASK_STATE_PREEMPTED, worker_of=job.worker_of)
        clock.tick()
        _assign(state, job.tasks, workers)
        clock.tick()
        _observe(state, job.tasks, job_pb2.TASK_STATE_RUNNING, worker_of=job.worker_of)

    return run


def scenario_worker_reregister(state, workers, clock):
    _running_job(state, workers, "/bench/wide", 5_000)
    clock.tick()

    def run():
        for _ in range(10):
            for k, worker in enumerate(workers):
                register_worker(state, str(worker), f"10.0.{k // 250}.{k % 250}:1", make_worker_metadata(cpu=1_000))
            clock.tick()

    return run


# The scenario also run against real Finelog tables, to count rows past the client buffer.
OVER_BUFFER_SCENARIO = "cancel_12000"

SCENARIOS: dict[str, Callable] = {
    "submit_1000x4_and_5000": scenario_submit,
    "schedule_5000": scenario_schedule,
    "finish_5000": scenario_finish,
    "status_only_5000": scenario_status_only,
    "remove_worker_64_live_x20": scenario_remove_worker,
    "cancel_5000": scenario_cancel_5k,
    OVER_BUFFER_SCENARIO: scenario_cancel_12k,
    "preempt_and_retry_1000": scenario_preempt_retry,
    "worker_reregister_80x10": scenario_worker_reregister,
}
# The number of rows a capture of each scenario's timed step should write, where it is fixed.
EXPECTED_ROWS = {
    "cancel_5000": {"jobs": 1, "tasks": 5_000, "attempts": 5_000},
    OVER_BUFFER_SCENARIO: {"jobs": 3, "tasks": 12_001, "attempts": 12_001},
    "status_only_5000": {"jobs": 0, "tasks": 0, "attempts": 0},
    "worker_reregister_80x10": {"jobs": 0, "tasks": 0, "attempts": 0},
}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _row_json(row) -> dict:
    out = {}
    for k, v in dataclasses.asdict(row).items():
        if k in _UNSTABLE_FIELDS:
            continue
        out[k] = v.isoformat() if isinstance(v, datetime) else v
    return out


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def _fresh_controller(
    template: Path, run_dir: Path, clock: FakeClock
) -> tuple[ControllerDB, ControllerTestState, list[WorkerId]]:
    """A copy of the background DB with ``NUM_WORKERS`` registered workers, the clock reset."""
    shutil.copytree(template, run_dir)
    clock.ms = START_MS
    db = ControllerDB(db_dir=run_dir)
    db.apply_migrations()
    state = ControllerTestState(db, health=WorkerHealthTracker())
    workers = [
        register_worker(state, f"w-{k:03d}", f"10.0.{k // 250}.{k % 250}:1", make_worker_metadata(cpu=1_000))
        for k in range(NUM_WORKERS)
    ]
    return db, state, workers


def run_scenario(name: str, attach: bool, template: Path, workdir: Path, clock: FakeClock) -> dict:
    run_dir = workdir / f"{name}-{'rows' if attach else 'none'}"
    db, state, workers = _fresh_controller(template, run_dir, clock)
    step = SCENARIOS[name](state, workers, clock)
    # FakeStatsTable keeps what it is given; capture only calls ``write``.
    tables = ObservabilityRowTables(
        cluster="bench",
        job_table=cast(FinelogTable, FakeStatsTable()),
        job_config_table=cast(FinelogTable, FakeStatsTable()),
        task_table=cast(FinelogTable, FakeStatsTable()),
        task_attempt_table=cast(FinelogTable, FakeStatsTable()),
    )
    if attach:
        db.attach_observability_rows(tables)
    m = Measurement()
    with measured(db, m):
        step()
    db.close()
    shutil.rmtree(run_dir)
    captured = {
        kind: [row for write in cast(FakeStatsTable, table).writes for row in write]
        for kind, table in (
            ("jobs", tables.job_table),
            ("configs", tables.job_config_table),
            ("tasks", tables.task_table),
            ("attempts", tables.task_attempt_table),
        )
    }
    return {
        "transactions": len(m.txn_ms),
        "p50_ms": _percentile(m.txn_ms, 50),
        "p95_ms": _percentile(m.txn_ms, 95),
        "p99_ms": _percentile(m.txn_ms, 99),
        "lock_total_ms": sum(m.txn_ms),
        "capture_ms": m.capture_ms,
        "captured": {k: len(v) for k, v in captured.items()},
        "rows": {k: [_row_json(r) for r in v] for k, v in captured.items()} if attach else {},
    }


def finelog_cancel_check(template: Path, workdir: Path, clock: FakeClock) -> dict:
    """Run the 12,000-task cancel with real Finelog client tables attached; count what the server keeps."""
    run_dir = workdir / "cancel-finelog"
    db, state, workers = _fresh_controller(template, run_dir, clock)
    step = scenario_cancel_12k(state, workers, clock)
    stack = build_log_stack(
        log_service_address="", local_log_dir=run_dir / "finelog", host="127.0.0.1", worker_token_provider=None
    )
    try:
        db.attach_observability_rows(stack.observability_row_tables("bench"))
        step()
        result = {}
        for kind, table in (("tasks", stack.task_table), ("attempts", stack.task_attempt_table)):
            verdict = table.flush(timeout=120)
            kept = table.query(f'SELECT count(*) AS n FROM "{table.namespace}"').column("n")[0].as_py()
            expected = EXPECTED_ROWS[OVER_BUFFER_SCENARIO][kind]
            result[kind] = {"expected": expected, "kept_by_server": int(kept), "flush": str(verdict)}
        return result
    finally:
        stack.close()
        db.close()
        shutil.rmtree(run_dir, ignore_errors=True)


@click.group()
def cli() -> None:
    pass


@cli.command("run")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Results JSON to write.")
@click.option("--repeat", type=int, default=3, help="Runs per scenario and variant; transaction times are pooled.")
@click.option("--scenario", "only", multiple=True, help="Run only these scenarios.")
def run_cmd(out: Path, repeat: int, only: tuple[str, ...]) -> None:
    """Run every scenario with no row tables and with them attached."""
    clock = FakeClock()
    original_now = Timestamp.now
    Timestamp.now = classmethod(lambda cls: clock.now())  # type: ignore[method-assign]
    workdir = Path(tempfile.mkdtemp(prefix="bench_obs_rows_"))
    results: dict = {"scenarios": {}}
    rows_out: dict = {}
    try:
        template = workdir / "template"
        start = time.perf_counter()
        results["background_rows"] = build_background_db(template)
        print(f"background DB built in {time.perf_counter() - start:.1f}s: {results['background_rows']}")
        for name in only or SCENARIOS:
            results["scenarios"][name] = {}
            for attach in (False, True):
                variant = "rows" if attach else "none"
                runs = [run_scenario(name, attach, template, workdir, clock) for _ in range(repeat)]
                merged = runs[-1]
                results["scenarios"][name][variant] = {k: merged[k] for k in ("transactions", "captured")} | {
                    "p50_ms": statistics.median(r["p50_ms"] for r in runs),
                    "p95_ms": statistics.median(r["p95_ms"] for r in runs),
                    "p99_ms": statistics.median(r["p99_ms"] for r in runs),
                    "lock_total_ms": statistics.median(r["lock_total_ms"] for r in runs),
                    "capture_ms": statistics.median(r["capture_ms"] for r in runs),
                }
                if attach:
                    rows_out[name] = merged["rows"]
                print(name, variant, json.dumps(results["scenarios"][name][variant]))
        if not only or OVER_BUFFER_SCENARIO in only:
            key = f"{OVER_BUFFER_SCENARIO}_real_finelog"
            results[key] = finelog_cancel_check(template, workdir, clock)
            print(key, results[key])
    finally:
        Timestamp.now = original_now  # type: ignore[method-assign]
        shutil.rmtree(workdir, ignore_errors=True)
    results["expected_rows"] = EXPECTED_ROWS
    out.write_text(json.dumps(results, indent=2))
    out.with_suffix(".rows.json").write_text(json.dumps(rows_out))


def _row_key(row: dict) -> str:
    return json.dumps(row, sort_keys=True)


@cli.command("compare")
@click.argument("left", type=click.Path(path_type=Path, exists=True))
@click.argument("right", type=click.Path(path_type=Path, exists=True))
def compare_cmd(left: Path, right: Path) -> None:
    """Print latency side by side and the rows captured by only one run."""
    a, b = json.loads(left.read_text()), json.loads(right.read_text())
    ra = json.loads(left.with_suffix(".rows.json").read_text())
    rb = json.loads(right.with_suffix(".rows.json").read_text())
    print(
        "| scenario | variant | txns | p50 ms | p95 ms | p99 ms | lock ms | capture ms | jobs/configs/tasks/attempts |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for name in a["scenarios"]:
        for label, res in (
            ("none", a["scenarios"][name]["none"]),
            (left.stem, a["scenarios"][name]["rows"]),
            (right.stem, b["scenarios"][name]["rows"]),
        ):
            c = res["captured"]
            print(
                f"| {name} | {label} | {res['transactions']} | {res['p50_ms']:.2f} | {res['p95_ms']:.2f} | "
                f"{res['p99_ms']:.2f} | {res['lock_total_ms']:.0f} | {res['capture_ms']:.0f} | "
                f"{c['jobs']}/{c['configs']}/{c['tasks']}/{c['attempts']} |"
            )
    print()
    for name in ra:
        for kind in ra[name]:
            ca = Counter(_row_key(r) for r in ra[name][kind])
            cb = Counter(_row_key(r) for r in rb[name][kind])
            only_a, only_b = ca - cb, cb - ca
            if only_a or only_b:
                counts = f"{sum(only_a.values())} only in {left.stem}, {sum(only_b.values())} only in {right.stem}"
                print(f"{name} {kind}: {counts}")
                for key in list(only_a)[:2]:
                    print(f"  {left.stem}: {key[:400]}")
                for key in list(only_b)[:2]:
                    print(f"  {right.stem}: {key[:400]}")


if __name__ == "__main__":
    cli()
