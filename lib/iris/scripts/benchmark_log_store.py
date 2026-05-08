#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark LogStore write and query performance.

Simulates a realistic workload: 100 jobs x 100 tasks x 1000 lines = 10M entries.
Measures ingest throughput and query latency across the patterns used by the
dashboard and log-tailing RPCs.

Usage:
    uv run python lib/iris/scripts/benchmark_log_store.py
    uv run python lib/iris/scripts/benchmark_log_store.py --jobs 50 --tasks 50 --lines 500
    uv run python lib/iris/scripts/benchmark_log_store.py --only ingest
    uv run python lib/iris/scripts/benchmark_log_store.py --only query

Corpus mode runs read-only queries against a pre-populated log directory
(e.g. downloaded from GCS), sampling real keys from the parquet files:

    uv run python lib/iris/scripts/benchmark_log_store.py \\
        --corpus-dir /tmp/cross-region/logs --segment-caps 50,200,1000
"""

import random
import time
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import duckdb
from finelog.rpc import logging_pb2
from finelog.store.duckdb_store import DuckDBLogStore as LogStore
from iris.cluster.log_store_helpers import task_log_key
from iris.cluster.types import JobName, TaskAttempt


def _make_entries(count: int, prefix: str, start_ms: int = 0) -> list[logging_pb2.LogEntry]:
    entries = []
    for i in range(count):
        level = 20 if i % 50 == 0 else 0  # ~2% WARNING entries
        entry = logging_pb2.LogEntry(source="stdout", data=f"{prefix} line {i}", level=level)
        entry.timestamp.epoch_ms = start_ms + i
        entries.append(entry)
    return entries


def _job_name(job_idx: int) -> str:
    return f"/user/bench-job-{job_idx:04d}"


def _task_key(job_idx: int, task_idx: int, attempt: int = 0) -> str:
    job = JobName.from_wire(f"{_job_name(job_idx)}/task/{task_idx}")
    return task_log_key(TaskAttempt(task_id=job, attempt_id=attempt))


def bench(name: str, fn: object, *, iterations: int = 20) -> tuple[float, float]:
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
    print(f"  {name:55s}  p50={p50:8.2f}ms  p95={p95:8.2f}ms")


def ingest(
    store: LogStore,
    num_jobs: int,
    num_tasks: int,
    lines_per_task: int,
) -> float:
    """Write all log entries. Returns total ingest time in seconds."""
    total_lines = num_jobs * num_tasks * lines_per_task
    print(f"  Ingesting {total_lines:,} entries ({num_jobs} jobs x {num_tasks} tasks x {lines_per_task} lines)...")

    batch_size = 50
    start = time.perf_counter()

    for job_idx in range(num_jobs):
        batch: list[tuple[str, list]] = []
        for task_idx in range(num_tasks):
            key = _task_key(job_idx, task_idx)
            entries = _make_entries(
                lines_per_task,
                prefix=f"[job={job_idx} task={task_idx}]",
                start_ms=job_idx * 1_000_000 + task_idx * 10_000,
            )
            batch.append((key, entries))

            if len(batch) >= batch_size:
                store.append_batch(batch)
                batch = []

        if batch:
            store.append_batch(batch)

        if (job_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - start
            rate = ((job_idx + 1) * num_tasks * lines_per_task) / elapsed
            print(f"    ... {job_idx + 1}/{num_jobs} jobs ingested ({rate:,.0f} lines/s)")

    elapsed = time.perf_counter() - start
    rate = total_lines / elapsed
    print(f"  Ingest complete: {elapsed:.1f}s ({rate:,.0f} lines/s)")
    return elapsed


def benchmark_queries(
    store: LogStore,
    num_jobs: int,
    num_tasks: int,
    lines_per_task: int,
    iterations: int,
) -> list[tuple[str, float, float]]:
    results: list[tuple[str, float, float]] = []

    # --- Exact key queries (single task) ---

    mid_job = num_jobs // 2
    mid_task = num_tasks // 2
    key = _task_key(mid_job, mid_task)

    p50, p95 = bench("get_logs(exact key, no limit)", lambda: store.get_logs(key), iterations=iterations)
    results.append(("get_logs(exact key, no limit)", p50, p95))
    print_result("get_logs(exact key, no limit)", p50, p95)

    p50, p95 = bench("get_logs(exact key, limit=100)", lambda: store.get_logs(key, max_lines=100), iterations=iterations)
    results.append(("get_logs(exact key, limit=100)", p50, p95))
    print_result("get_logs(exact key, limit=100)", p50, p95)

    p50, p95 = bench(
        "get_logs(exact key, tail=True, limit=100)",
        lambda: store.get_logs(key, max_lines=100, tail=True),
        iterations=iterations,
    )
    results.append(("get_logs(exact key, tail=True, limit=100)", p50, p95))
    print_result("get_logs(exact key, tail=True, limit=100)", p50, p95)

    # --- Cursor-based pagination ---

    result_first = store.get_logs(key, max_lines=500)
    cursor = result_first.cursor

    p50, p95 = bench(
        "get_logs(exact key, cursor=mid, limit=100)",
        lambda: store.get_logs(key, cursor=cursor, max_lines=100),
        iterations=iterations,
    )
    results.append(("get_logs(exact key, cursor=mid, limit=100)", p50, p95))
    print_result("get_logs(exact key, cursor=mid, limit=100)", p50, p95)

    # --- Substring filter ---

    p50, p95 = bench(
        "get_logs(exact key, substring='line 42')",
        lambda: store.get_logs(key, substring_filter="line 42"),
        iterations=iterations,
    )
    results.append(("get_logs(exact key, substring='line 42')", p50, p95))
    print_result("get_logs(exact key, substring='line 42')", p50, p95)

    # --- Min level filter ---

    p50, p95 = bench(
        "get_logs(exact key, min_level=WARNING)",
        lambda: store.get_logs(key, min_level="WARNING"),
        iterations=iterations,
    )
    results.append(("get_logs(exact key, min_level=WARNING)", p50, p95))
    print_result("get_logs(exact key, min_level=WARNING)", p50, p95)

    # --- since_ms filter ---

    since = mid_job * 1_000_000 + mid_task * 10_000 + lines_per_task // 2

    p50, p95 = bench(
        "get_logs(exact key, since_ms=mid)",
        lambda: store.get_logs(key, since_ms=since),
        iterations=iterations,
    )
    results.append(("get_logs(exact key, since_ms=mid)", p50, p95))
    print_result("get_logs(exact key, since_ms=mid)", p50, p95)

    # --- Prefix queries (all tasks in a job) ---

    job_prefix = f"{_job_name(mid_job)}/"

    p50, p95 = bench(
        f"get_logs(regex(job, ~{num_tasks} tasks, limit=100)",
        lambda: store.get_logs(job_prefix + ".*", max_lines=100),
        iterations=iterations,
    )
    results.append((f"get_logs(regex(job, ~{num_tasks} tasks, limit=100)", p50, p95))
    print_result(f"get_logs(regex(job, ~{num_tasks} tasks, limit=100)", p50, p95)

    p50, p95 = bench(
        f"get_logs(regex(job, ~{num_tasks} tasks, no limit)",
        lambda: store.get_logs(job_prefix + ".*"),
        iterations=iterations,
    )
    results.append((f"get_logs(regex(job, ~{num_tasks} tasks, no limit)", p50, p95))
    print_result(f"get_logs(regex(job, ~{num_tasks} tasks, no limit)", p50, p95)

    p50, p95 = bench(
        "get_logs(regex(job, tail=True, limit=100)",
        lambda: store.get_logs(job_prefix + ".*", max_lines=100, tail=True),
        iterations=iterations,
    )
    results.append(("get_logs(regex(job, tail=True, limit=100)", p50, p95))
    print_result("get_logs(regex(job, tail=True, limit=100)", p50, p95)

    p50, p95 = bench(
        "get_logs(regex(job, substring='line 999')",
        lambda: store.get_logs(job_prefix + ".*", substring_filter="line 999"),
        iterations=iterations,
    )
    results.append(("get_logs(regex(job, substring='line 999')", p50, p95))
    print_result("get_logs(regex(job, substring='line 999')", p50, p95)

    # --- Prefix with cursor continuation ---

    prefix_result = store.get_logs(job_prefix + ".*", max_lines=500)
    prefix_cursor = prefix_result.cursor

    p50, p95 = bench(
        "get_logs(regex(job, cursor=mid, limit=100)",
        lambda: store.get_logs(job_prefix + ".*", cursor=prefix_cursor, max_lines=100),
        iterations=iterations,
    )
    results.append(("get_logs(regex(job, cursor=mid, limit=100)", p50, p95))
    print_result("get_logs(regex(job, cursor=mid, limit=100)", p50, p95)

    # --- Broad prefix (all jobs under /user/) ---

    p50, p95 = bench(
        "get_logs(regex('/user/', limit=100)",
        lambda: store.get_logs("/user/.*", max_lines=100),
        iterations=iterations,
    )
    results.append(("get_logs(regex('/user/', limit=100)", p50, p95))
    print_result("get_logs(regex('/user/', limit=100)", p50, p95)

    p50, p95 = bench(
        "get_logs(regex('/user/', tail, limit=100)",
        lambda: store.get_logs("/user/.*", max_lines=100, tail=True),
        iterations=iterations,
    )
    results.append(("get_logs(regex('/user/', tail, limit=100)", p50, p95))
    print_result("get_logs(regex('/user/', tail, limit=100)", p50, p95)

    # --- has_logs ---

    p50, p95 = bench("has_logs(existing key)", lambda: store.has_logs(key), iterations=iterations)
    results.append(("has_logs(existing key)", p50, p95))
    print_result("has_logs(existing key)", p50, p95)

    missing_key = _task_key(num_jobs + 1, 0)
    p50, p95 = bench("has_logs(missing key)", lambda: store.has_logs(missing_key), iterations=iterations)
    results.append(("has_logs(missing key)", p50, p95))
    print_result("has_logs(missing key)", p50, p95)

    # --- Combined filters ---

    p50, p95 = bench(
        "get_logs(key, cursor+since_ms+substring)",
        lambda: store.get_logs(key, cursor=cursor, since_ms=since, substring_filter="line"),
        iterations=iterations,
    )
    results.append(("get_logs(key, cursor+since_ms+substring)", p50, p95))
    print_result("get_logs(key, cursor+since_ms+substring)", p50, p95)

    return results


def sample_corpus_keys(log_dir: Path, n_exact: int = 5) -> tuple[list[str], list[str]]:
    """Pick real exact keys and prefix families from an existing log_dir.

    Samples from the 5 newest parquets to reflect what the log server
    typically serves (recent tasks, current jobs).
    """
    files = sorted(log_dir.glob("logs_*.parquet"))
    if not files:
        raise click.ClickException(f"no logs_*.parquet files in {log_dir}")
    probe = [str(p) for p in files[-5:]]
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"SELECT key, COUNT(*) c FROM read_parquet({probe!r}) GROUP BY key ORDER BY c DESC LIMIT {n_exact * 4}"
        ).fetchall()
        keys = [r[0] for r in rows]
        rng = random.Random(0)
        rng.shuffle(keys)
        exact = keys[:n_exact]
        prefixes = sorted({k.rsplit("/", 1)[0] + "/" for k in exact})
    finally:
        con.close()
    return exact, prefixes


def benchmark_corpus(
    store: LogStore, exact_keys: list[str], prefixes: list[str], iterations: int
) -> list[tuple[str, float, float]]:
    """Run the read-only query workload against pre-populated segments."""
    results: list[tuple[str, float, float]] = []

    for i, key in enumerate(exact_keys):
        p50, p95 = bench(
            f"get_logs(exact #{i}, limit=100)", lambda k=key: store.get_logs(k, max_lines=100), iterations=iterations
        )
        results.append((f"get_logs(exact #{i}, limit=100)", p50, p95))
        print_result(f"get_logs(exact #{i}, limit=100)", p50, p95)

        p50, p95 = bench(
            f"get_logs(exact #{i}, tail, limit=100)",
            lambda k=key: store.get_logs(k, max_lines=100, tail=True),
            iterations=iterations,
        )
        results.append((f"get_logs(exact #{i}, tail, limit=100)", p50, p95))
        print_result(f"get_logs(exact #{i}, tail, limit=100)", p50, p95)

        p50, p95 = bench(
            f"get_logs(exact #{i}, substring='error')",
            lambda k=key: store.get_logs(k, substring_filter="error", max_lines=100),
            iterations=iterations,
        )
        results.append((f"get_logs(exact #{i}, substring='error')", p50, p95))
        print_result(f"get_logs(exact #{i}, substring='error')", p50, p95)

        p50, p95 = bench(
            f"has_logs(exact #{i})",
            lambda k=key: store.has_logs(k),
            iterations=iterations,
        )
        results.append((f"has_logs(exact #{i})", p50, p95))
        print_result(f"has_logs(exact #{i})", p50, p95)

    missing_key = "/no-such-user/no-such-job-xxxxxxxx/0:0"
    p50, p95 = bench(
        "get_logs(missing key, limit=100)", lambda: store.get_logs(missing_key, max_lines=100), iterations=iterations
    )
    results.append(("get_logs(missing key, limit=100)", p50, p95))
    print_result("get_logs(missing key, limit=100)", p50, p95)

    p50, p95 = bench("has_logs(missing key)", lambda: store.has_logs(missing_key), iterations=iterations)
    results.append(("has_logs(missing key)", p50, p95))
    print_result("has_logs(missing key)", p50, p95)

    for i, prefix in enumerate(prefixes[:3]):
        pattern = prefix + ".*"
        p50, p95 = bench(
            f"get_logs(regex(prefix #{i}), tail, limit=100)",
            lambda p=pattern: store.get_logs(p, max_lines=100, tail=True),
            iterations=iterations,
        )
        results.append((f"get_logs(regex(prefix #{i}), tail, limit=100)", p50, p95))
        print_result(f"get_logs(regex(prefix #{i}), tail, limit=100)", p50, p95)

    return results


def run_corpus_mode(corpus_dir: Path, caps: list[int], iterations: int) -> None:
    """Open the existing log_dir at each cap value, run the read workload."""
    if not corpus_dir.exists():
        raise click.ClickException(f"corpus dir does not exist: {corpus_dir}")
    all_files = sorted(corpus_dir.glob("logs_*.parquet")) + sorted(corpus_dir.glob("tmp_*.parquet"))
    total_bytes = sum(f.stat().st_size for f in all_files)
    print(f"Corpus: {corpus_dir}")
    print(f"  {len(all_files)} parquet files, {total_bytes / 1024 / 1024 / 1024:.1f} GB on disk")

    print("  Sampling real keys...")
    t0 = time.perf_counter()
    exact, prefixes = sample_corpus_keys(corpus_dir, n_exact=3)
    print(f"  Sampled in {time.perf_counter() - t0:.2f}s")
    print("  Exact keys:")
    for k in exact:
        print(f"    {k}")
    print("  Prefixes:")
    for p in prefixes[:3]:
        print(f"    {p}.*")

    summary: list[tuple[int, list[tuple[str, float, float]], float]] = []
    for cap in caps:
        print(f"\n[cap={cap}]")
        print(f"  Opening store with max_local_segments={cap}, max_local_bytes=100GB...")
        init_start = time.perf_counter()
        store = LogStore(
            log_dir=corpus_dir,
            max_local_segments=cap,
            max_local_bytes=100 * 1024**3,
        )
        init_elapsed = time.perf_counter() - init_start

        # Simulate post-GC state: _gc_local_segments only fires after flushes,
        # which a read-only corpus never triggers. Trim the in-memory deque to
        # the newest `cap` files by min_seq to reflect what the store would
        # actually retain at steady state under this cap.
        segs = sorted(store._local_segments, key=lambda s: s.min_seq)  # type: ignore[attr-defined]
        if len(segs) > cap:
            segs = segs[-cap:]
        store._local_segments = deque(segs)  # type: ignore[attr-defined]

        n_segments = len(store._local_segments)  # type: ignore[attr-defined]
        seg_bytes = sum(s.size_bytes for s in store._local_segments)  # type: ignore[attr-defined]
        seg_gb = seg_bytes / 1024 / 1024 / 1024
        print(f"  Init: {init_elapsed * 1000:.0f}ms ({n_segments} segments, {seg_gb:.1f} GB retained)")

        try:
            print("  Warmup...")
            for k in exact:
                store.get_logs(k, max_lines=100)
            print("  Benchmarking queries...")
            results = benchmark_corpus(store, exact, prefixes, iterations)
            summary.append((cap, results, init_elapsed * 1000))
        finally:
            store.close()

    print("\n" + "=" * 110)
    print(f"  Corpus benchmark summary ({corpus_dir})")
    print("=" * 110)
    caps_hdr = "  ".join(f"cap={c:<5d}" for c in caps)
    print(f"  {'Query':52s}  {caps_hdr}  (p50 / p95 ms)")
    print("-" * 110)
    per_cap = {cap: {name: (p50, p95) for name, p50, p95 in res} for cap, res, _ in summary}
    query_order = [name for name, _, _ in summary[0][1]]
    for name in query_order:
        parts = []
        for cap in caps:
            p50, p95 = per_cap[cap].get(name, (float("nan"), float("nan")))
            parts.append(f"{p50:5.1f}/{p95:<5.1f}")
        print(f"  {name:52s}  " + "  ".join(f"{p:<11s}" for p in parts))
    print("-" * 110)
    init_parts = []
    for _, _, init_ms in summary:
        init_parts.append(f"{init_ms:5.0f}      ")
    print(f"  {'[init time ms]':52s}  " + "  ".join(f"{p:<11s}" for p in init_parts))
    print("=" * 110)


def print_summary(results: list[tuple[str, float, float]]) -> None:
    print("\n" + "=" * 85)
    print(f"  {'Query':55s}  {'p50':>10s}  {'p95':>10s}")
    print("-" * 85)
    for name, p50, p95 in results:
        print(f"  {name:55s}  {p50:8.2f}ms  {p95:8.2f}ms")
    print("=" * 85)


@click.command()
@click.option("--jobs", default=100, help="Number of jobs")
@click.option("--tasks", default=100, help="Number of tasks per job")
@click.option("--lines", default=1000, help="Log lines per task")
@click.option("--iterations", "-n", default=20, help="Query benchmark iterations")
@click.option("--only", "only_group", type=click.Choice(["ingest", "query"]), help="Run only this phase")
@click.option(
    "--log-dir", type=click.Path(path_type=Path), default=None, help="Persist logs to this dir (default: tmpdir)"
)
@click.option(
    "--corpus-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Read-only: benchmark queries against an existing log dir (skips ingest)",
)
@click.option(
    "--segment-caps",
    default="50,1000",
    help="Comma-separated max_local_segments values to compare in corpus mode",
)
def main(
    jobs: int,
    tasks: int,
    lines: int,
    iterations: int,
    only_group: str | None,
    log_dir: Path | None,
    corpus_dir: Path | None,
    segment_caps: str,
) -> None:
    """Benchmark LogStore write and query performance."""
    if corpus_dir is not None:
        caps = [int(x) for x in segment_caps.split(",") if x.strip()]
        run_corpus_mode(corpus_dir, caps, iterations)
        return

    total = jobs * tasks * lines
    print(f"LogStore benchmark: {jobs} jobs x {tasks} tasks x {lines} lines = {total:,} entries")
    print(f"  query_iterations={iterations}")

    tmp = None
    if log_dir is None:
        tmp = TemporaryDirectory(prefix="bench_log_store_")
        log_dir = Path(tmp.name) / "logs"

    store = LogStore(log_dir=log_dir)

    try:
        if only_group is None or only_group == "ingest":
            print("\n[ingest]")
            ingest(store, jobs, tasks, lines)

            parquet_files = list(log_dir.glob("tmp_*.parquet")) + list(log_dir.glob("logs_*.parquet"))
            total_size = sum(f.stat().st_size for f in parquet_files)
            print(f"  {len(parquet_files)} Parquet segments, {total_size / 1024 / 1024:.1f} MB on disk")

        if only_group is None or only_group == "query":
            if only_group == "query":
                # If only running queries, we still need data
                print("\n[ingest (for query benchmark)]")
                ingest(store, jobs, tasks, lines)

            print("\n[query]")
            results = benchmark_queries(store, jobs, tasks, lines, iterations)
            print_summary(results)
    finally:
        store.close()
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
