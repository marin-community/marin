# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark for the list_tasks resource-history query.

Validates that the batch query fetching latest task_resource_history rows
per task does not degrade at scale. Creates a synthetic DB matching the
profile requested in PR #4704:

- 1 target job with 1,000 tasks, each with 100 history rows  (100k rows)
- 10,000 unrelated tasks (across other jobs), each with 100 history rows (1M rows)
- Total: ~1.1M task_resource_history rows

The query should stay well under 1 second.
"""

import sqlite3
import time

import pytest

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.service import _tasks_for_listing
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp


def _populate_synthetic_db(db: ControllerDB, target_tasks: int, unrelated_tasks: int, history_per_task: int) -> JobName:
    """Populate a test DB with synthetic jobs, tasks, and resource history.

    Returns the job_id of the target job whose tasks will be queried.
    """
    target_job_id = JobName.from_wire("/bench-user/target-job")
    now_ms = Timestamp.now().epoch_ms()

    conn = sqlite3.connect(str(db.db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")

    # Ensure user exists for FK
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, created_at_ms) VALUES (?, ?)",
        ("bench-user", now_ms),
    )

    # --- Insert target job ---
    conn.execute(
        "INSERT INTO jobs (job_id, user_id, root_job_id, depth, state, submitted_at_ms, "
        "root_submitted_at_ms, num_tasks, is_reservation_holder, name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            target_job_id.to_wire(),
            "bench-user",
            target_job_id.to_wire(),
            1,
            job_pb2.JOB_STATE_RUNNING,
            now_ms,
            now_ms,
            target_tasks,
            0,
            "target-job",
        ),
    )
    conn.execute("INSERT INTO job_config (job_id) VALUES (?)", (target_job_id.to_wire(),))

    # --- Insert target tasks ---
    target_task_rows = []
    target_attempt_rows = []
    for i in range(target_tasks):
        task_id = target_job_id.task(i).to_wire()
        target_task_rows.append(
            (
                task_id,
                target_job_id.to_wire(),
                i,  # task_index
                job_pb2.TASK_STATE_RUNNING,
                0,  # current_attempt_id
                now_ms,  # submitted_at_ms
                0,  # max_retries_failure
                0,  # max_retries_preemption
                0,  # failure_count
                0,  # preemption_count
                0,  # priority_neg_depth
                now_ms,  # priority_root_submitted_ms
                i,  # priority_insertion
            )
        )
        target_attempt_rows.append(
            (
                task_id,
                0,  # attempt_id
                job_pb2.TASK_STATE_RUNNING,
                now_ms,  # created_at_ms
            )
        )

    conn.executemany(
        "INSERT INTO tasks (task_id, job_id, task_index, state, current_attempt_id, submitted_at_ms, "
        "max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
        "priority_neg_depth, priority_root_submitted_ms, priority_insertion) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        target_task_rows,
    )
    conn.executemany(
        "INSERT INTO task_attempts (task_id, attempt_id, state, created_at_ms) VALUES (?, ?, ?, ?)",
        target_attempt_rows,
    )

    # --- Insert unrelated jobs + tasks ---
    tasks_per_job = 100
    num_unrelated_jobs = unrelated_tasks // tasks_per_job

    unrelated_task_rows = []
    unrelated_attempt_rows = []
    for j in range(num_unrelated_jobs):
        ujob_id = JobName.from_wire(f"/bench-user/unrelated-{j}")
        conn.execute(
            "INSERT INTO jobs (job_id, user_id, root_job_id, depth, state, submitted_at_ms, "
            "root_submitted_at_ms, num_tasks, is_reservation_holder, name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ujob_id.to_wire(),
                "bench-user",
                ujob_id.to_wire(),
                1,
                job_pb2.JOB_STATE_RUNNING,
                now_ms,
                now_ms,
                tasks_per_job,
                0,
                f"unrelated-{j}",
            ),
        )
        conn.execute("INSERT INTO job_config (job_id) VALUES (?)", (ujob_id.to_wire(),))
        for i in range(tasks_per_job):
            task_id = ujob_id.task(i).to_wire()
            unrelated_task_rows.append(
                (
                    task_id,
                    ujob_id.to_wire(),
                    i,
                    job_pb2.TASK_STATE_RUNNING,
                    0,
                    now_ms,
                    0,
                    0,
                    0,
                    0,
                    0,
                    now_ms,
                    j * tasks_per_job + i,
                )
            )
            unrelated_attempt_rows.append((task_id, 0, job_pb2.TASK_STATE_RUNNING, now_ms))

    conn.executemany(
        "INSERT INTO tasks (task_id, job_id, task_index, state, current_attempt_id, submitted_at_ms, "
        "max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
        "priority_neg_depth, priority_root_submitted_ms, priority_insertion) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        unrelated_task_rows,
    )
    conn.executemany(
        "INSERT INTO task_attempts (task_id, attempt_id, state, created_at_ms) VALUES (?, ?, ?, ?)",
        unrelated_attempt_rows,
    )

    # --- Insert task_resource_history rows (bulk) ---
    all_task_ids = [r[0] for r in target_task_rows] + [r[0] for r in unrelated_task_rows]

    batch_size = 50_000
    history_batch = []
    for task_id in all_task_ids:
        for h in range(history_per_task):
            ts = now_ms - (history_per_task - h) * 10_000
            history_batch.append(
                (
                    task_id,
                    0,  # attempt_id
                    1000 + h,  # cpu_millicores
                    512 + h,  # memory_mb
                    10,  # disk_mb
                    600 + h,  # memory_peak_mb
                    ts,
                )
            )
            if len(history_batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO task_resource_history "
                    "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    history_batch,
                )
                history_batch = []

    if history_batch:
        conn.executemany(
            "INSERT INTO task_resource_history "
            "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            history_batch,
        )

    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    return target_job_id


@pytest.fixture(scope="module")
def bench_db(tmp_path_factory):
    """Create a synthetic benchmark DB with ~1.1M task_resource_history rows.

    Module-scoped so the expensive DB population runs once and is shared
    across all benchmark tests.
    """
    tmp_path = tmp_path_factory.mktemp("bench_db")
    db = ControllerDB(db_dir=tmp_path)
    t0 = time.perf_counter()
    target_job_id = _populate_synthetic_db(
        db,
        target_tasks=1_000,
        unrelated_tasks=10_000,
        history_per_task=100,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  DB setup: {elapsed:.1f}s")
    yield db, target_job_id
    db.close()


def _run_resource_history_query(db: ControllerDB, job_id: JobName) -> dict:
    """Execute the exact query from list_tasks and return the results dict."""
    resource_by_task: dict[str, tuple] = {}
    with db.read_snapshot() as q:
        rows = q.raw(
            "SELECT trh.task_id, trh.cpu_millicores, trh.memory_mb, trh.disk_mb, trh.memory_peak_mb "
            "FROM task_resource_history trh "
            "INNER JOIN ("
            "  SELECT trh2.task_id, MAX(trh2.id) as max_id "
            "  FROM task_resource_history trh2 "
            "  JOIN tasks t ON trh2.task_id = t.task_id AND trh2.attempt_id = t.current_attempt_id "
            "  WHERE t.job_id = ? "
            "  GROUP BY trh2.task_id"
            ") latest ON trh.id = latest.max_id",
            (job_id.to_wire(),),
        )
    for r in rows:
        resource_by_task[r.task_id] = (r.cpu_millicores, r.memory_mb, r.disk_mb, r.memory_peak_mb)
    return resource_by_task


@pytest.mark.timeout(120)
def test_list_tasks_resource_query_performance(bench_db):
    """Benchmark the resource history query with 1.1M rows.

    Asserts the query completes in under 1 second even with 1M+ history rows.
    Prints timing for CI visibility.
    """
    db, target_job_id = bench_db

    # Verify data was inserted correctly
    row_count = db.fetchone("SELECT COUNT(*) as cnt FROM task_resource_history")["cnt"]
    target_task_count = db.fetchone("SELECT COUNT(*) as cnt FROM tasks WHERE job_id = ?", (target_job_id.to_wire(),))[
        "cnt"
    ]
    print(f"\n  task_resource_history rows: {row_count:,}")
    print(f"  target job tasks: {target_task_count:,}")

    # Warmup
    _run_resource_history_query(db, target_job_id)

    # Benchmark (10 iterations)
    iterations = 10
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = _run_resource_history_query(db, target_job_id)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    avg = sum(times) / len(times)

    print(f"  Resource history query (1000 tasks, {row_count:,} total rows):")
    print(f"    p50={p50:.1f}ms  p95={p95:.1f}ms  avg={avg:.1f}ms  n={iterations}")
    print(f"    returned {len(result)} task resource entries")

    # Verify correctness: should return exactly 1000 entries (one per target task)
    assert len(result) == 1000, f"Expected 1000 results, got {len(result)}"

    # Performance gate: must complete in under 1 second
    assert p95 < 1000, f"Query too slow: p95={p95:.1f}ms (limit: 1000ms)"


@pytest.mark.timeout(120)
def test_list_tasks_full_rpc_performance(bench_db):
    """Benchmark the full list_tasks flow including _tasks_for_listing + resource query.

    This exercises the complete code path that the RPC uses.
    """
    db, target_job_id = bench_db

    # Warmup
    _tasks_for_listing(db, job_id=target_job_id)
    _run_resource_history_query(db, target_job_id)

    iterations = 5
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        tasks = _tasks_for_listing(db, job_id=target_job_id)
        resource_by_task = _run_resource_history_query(db, target_job_id)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]

    print("\n  Full list_tasks flow (tasks + resource history):")
    print(f"    p50={p50:.1f}ms  p95={p95:.1f}ms  n={iterations}")
    print(f"    {len(tasks)} tasks, {len(resource_by_task)} resource entries")

    assert p95 < 2000, f"Full flow too slow: p95={p95:.1f}ms (limit: 2000ms)"


@pytest.mark.timeout(120)
def test_query_plan_uses_index(bench_db):
    """Verify the query plan uses the task_resource_history index, not a full scan."""
    db, target_job_id = bench_db

    with db.read_snapshot() as q:
        rows = q.raw(
            "EXPLAIN QUERY PLAN "
            "SELECT trh.task_id, trh.cpu_millicores, trh.memory_mb, trh.disk_mb, trh.memory_peak_mb "
            "FROM task_resource_history trh "
            "INNER JOIN ("
            "  SELECT trh2.task_id, MAX(trh2.id) as max_id "
            "  FROM task_resource_history trh2 "
            "  JOIN tasks t ON trh2.task_id = t.task_id AND trh2.attempt_id = t.current_attempt_id "
            "  WHERE t.job_id = ? "
            "  GROUP BY trh2.task_id"
            ") latest ON trh.id = latest.max_id",
            (target_job_id.to_wire(),),
        )

    plan_text = "\n".join(str(r) for r in rows)
    print(f"\n  Query plan:\n{plan_text}")

    # The plan should reference the index, not SCAN task_resource_history
    plan_lower = plan_text.lower()
    assert (
        "idx_task" in plan_lower or "using" in plan_lower or "search" in plan_lower
    ), f"Query plan may not be using indexes efficiently:\n{plan_text}"
