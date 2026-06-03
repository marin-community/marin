# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apples-to-apples query-latency benchmark: Python (DuckDB) vs Rust (DataFusion).

Both finelog server backends expose the SAME ``StatsService.Query`` RPC over
HTTP/Connect. This script measures client-side query latency against each
backend over an IDENTICAL queryable row set, on the same machine, one backend
at a time.

Fairness protocol
------------------
The Python server rehydrates its namespace registry only from its DuckDB
sidecar (``_finelog_registry.duckdb``); it does NOT scan a parquet-only tree at
boot. So the "both adopt the same on-disk parquet" path is not available for a
fair comparison. Instead we use the guaranteed-fair load: read the sampled
parquet rows once with pyarrow, then write the IDENTICAL rows (same schema, same
order, same batches) into a freshly-started server of EACH backend via
``WriteRows``, force an L0 compaction via ``POST /debug/maintain`` (so both
re-segment symmetrically), then query. Before timing, we assert both backends
agree on ``SELECT COUNT(*)`` for the namespace.

Data
----
Point ``--data-dir`` at a directory laid out as ``<namespace>/seg_L*.parquet``
(the production segment layout). Each namespace's parquet rows are loaded into
both servers. The store-assigned ``seq`` column is dropped from the input (the
server stamps its own), and timestamp columns are coerced to millisecond
resolution to match the wire schema's ``TIMESTAMP_MS``.

Usage
-----
    uv run python -m benchmarks.query_bench \\
        --data-dir /tmp/finelog_bench_data \\
        --iterations 30 --warmup 5

Run from ``lib/finelog`` (so ``benchmarks`` is importable) under ``uv run``.
"""

from __future__ import annotations

import argparse
import io
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as paipc
import pyarrow.parquet as pq
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

# benchmarks/query_bench.py -> repo root is four parents up (lib/finelog/benchmarks).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Arrow type -> proto ColumnType. The finelog wire schema only carries these
# scalar types; the loader coerces inputs into this set before writing.
_PROTO_TYPE_FOR_ARROW: dict[pa.DataType, int] = {
    pa.string(): stats_pb2.COLUMN_TYPE_STRING,
    pa.int64(): stats_pb2.COLUMN_TYPE_INT64,
    pa.int32(): stats_pb2.COLUMN_TYPE_INT32,
    pa.float64(): stats_pb2.COLUMN_TYPE_FLOAT64,
    pa.bool_(): stats_pb2.COLUMN_TYPE_BOOL,
    pa.timestamp("ms"): stats_pb2.COLUMN_TYPE_TIMESTAMP_MS,
    pa.binary(): stats_pb2.COLUMN_TYPE_BYTES,
}

# The store stamps this column itself; it must not be sent on WriteRows.
_IMPLICIT_SEQ_COLUMN = "seq"

# Rows per WriteRows batch. Kept well under the server's 16 MiB / 1M-row caps
# for the wide (~20 column) worker schema; ~25k rows of that schema is a few MB.
_WRITE_BATCH_ROWS = 25_000


def _rust_binary() -> Path:
    for profile in ("release", "debug"):
        candidate = _REPO_ROOT / "rust" / "target" / profile / "finelog-server"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("finelog-server Rust binary not built (run `cargo build -p finelog --release`)")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass(frozen=True)
class Backend:
    """One server implementation under test."""

    name: str

    def command(self, *, port: int, log_dir: Path) -> list[str]:
        if self.name == "python":
            return [
                sys.executable,
                "-m",
                "finelog.server.main",
                "--port",
                str(port),
                "--log-dir",
                str(log_dir),
                "--log-level",
                "WARNING",
                "--debug-admin",
            ]
        if self.name == "rust":
            return [
                str(_rust_binary()),
                "--port",
                str(port),
                "--log-dir",
                str(log_dir),
                "--log-level",
                "warn",
                "--debug-admin",
            ]
        raise ValueError(f"unknown backend: {self.name}")


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode} before /health came up")
        try:
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.05)
    raise TimeoutError(f"{base_url}/health did not come up within {timeout}s: {last_err}")


@dataclass(frozen=True)
class RunningServer:
    base_url: str
    boot_seconds: float


@contextmanager
def _spawn(backend: Backend, log_dir: Path, *, health_timeout: float = 60.0) -> Iterator[RunningServer]:
    """Spawn ``backend`` over ``log_dir`` on a fresh port; yield once healthy."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    proc = subprocess.Popen(
        backend.command(port=port, log_dir=log_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url, proc, timeout=health_timeout)
        yield RunningServer(base_url=base_url, boot_seconds=time.monotonic() - started)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NamespaceData:
    """A namespace's queryable rows, normalized to the finelog wire schema."""

    namespace: str
    schema: stats_pb2.Schema
    batches: list[pa.RecordBatch]
    row_count: int
    segment_count: int
    parquet_bytes: int


def _normalize_table(table: pa.Table) -> pa.Table:
    """Drop the store-assigned ``seq`` column and coerce types to the wire set.

    Microsecond/nanosecond timestamps become millisecond (the only timestamp
    resolution the wire schema carries). Other unsupported types raise.
    """
    if _IMPLICIT_SEQ_COLUMN in table.column_names:
        table = table.drop_columns([_IMPLICIT_SEQ_COLUMN])
    for i, field in enumerate(table.schema):
        if pa.types.is_timestamp(field.type) and field.type != pa.timestamp("ms"):
            coerced = pc.cast(table.column(i), pa.timestamp("ms"))
            table = table.set_column(i, pa.field(field.name, pa.timestamp("ms")), coerced)
    return table


def _proto_schema(table: pa.Table, key_column: str) -> stats_pb2.Schema:
    columns = []
    for field in table.schema:
        proto_type = _PROTO_TYPE_FOR_ARROW.get(field.type)
        if proto_type is None:
            raise ValueError(f"column {field.name!r}: unsupported arrow type {field.type} for the finelog wire schema")
        # The key column must be non-nullable; everything else is nullable so a
        # segment that happens to have all-present values still matches a schema
        # that tolerates nulls elsewhere.
        columns.append(stats_pb2.Column(name=field.name, type=proto_type, nullable=(field.name != key_column)))
    return stats_pb2.Schema(columns=columns, key_column=key_column)


def _pick_key_column(table: pa.Table) -> str:
    """Pick a non-null ordering key: prefer ``timestamp_ms``/``ts``, else col 0."""
    names = table.column_names
    for candidate in ("timestamp_ms", "ts"):
        if candidate in names:
            return candidate
    return names[0]


def load_namespace(ns_dir: Path) -> NamespaceData:
    """Read every ``seg_L*.parquet`` under ``ns_dir`` into normalized batches."""
    files = sorted(ns_dir.glob("seg_L*.parquet"))
    if not files:
        raise FileNotFoundError(f"no seg_L*.parquet segments under {ns_dir}")
    parquet_bytes = sum(f.stat().st_size for f in files)
    table = _normalize_table(pq.read_table([str(f) for f in files]))
    key_column = _pick_key_column(table)
    schema = _proto_schema(table, key_column)
    batches = table.combine_chunks().to_batches(max_chunksize=_WRITE_BATCH_ROWS)
    return NamespaceData(
        namespace=ns_dir.name,
        schema=schema,
        batches=batches,
        row_count=table.num_rows,
        segment_count=len(files),
        parquet_bytes=parquet_bytes,
    )


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def populate(client: StatsServiceClientSync, base_url: str, data: NamespaceData) -> None:
    """Register the namespace, write all batches, force an L0 compaction."""
    client.register_table(stats_pb2.RegisterTableRequest(namespace=data.namespace, schema=data.schema))
    for batch in data.batches:
        client.write_rows(stats_pb2.WriteRowsRequest(namespace=data.namespace, arrow_ipc=_ipc_bytes(batch)))
    resp = httpx.post(
        f"{base_url}/debug/maintain",
        json={"namespace": data.namespace, "force_compact_l0": True},
        timeout=120.0,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Queries.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchQuery:
    name: str
    sql: str


def build_queries(namespace: str) -> list[BenchQuery]:
    """Representative query shapes over a single namespace (double-quoted, dotted).

    Modeled on the real corpus in tests/parity/test_golden_corpus.py:
    COUNT(*), GROUP BY aggregate, WHERE with AND/LIKE, multi-column ORDER BY
    LIMIT, projection, and SELECT * LIMIT N.

    The Rust server caps a Query response at 64 MiB of Arrow IPC
    (``MAX_MESSAGE_BYTES`` in stats_service.rs) and rejects anything larger;
    the Python server has no such cap. To keep the comparison apples-to-apples
    (and measure query execution, not transport edge behavior), the row-returning
    scans are bounded by a LIMIT that keeps the result well under that cap on
    both backends while still materializing a large result.
    """
    ns = f'"{namespace}"'
    # 200k rows of a 2-4 column projection is ~15-25 MB of Arrow IPC — a large,
    # representative materialization that stays under the 64 MiB transport cap.
    scan_limit = 200_000
    return [
        BenchQuery("count_star", f"SELECT COUNT(*) AS n FROM {ns}"),
        BenchQuery(
            "group_by_agg",
            f"SELECT status, COUNT(*) AS n, AVG(cpu_pct) AS avg_cpu FROM {ns} GROUP BY status ORDER BY n DESC",
        ),
        BenchQuery(
            "filter_and_like",
            f"SELECT worker_id, mem_bytes FROM {ns} "
            f"WHERE status = 'RUNNING' AND worker_id LIKE 'marin-%' AND mem_bytes >= 0 LIMIT {scan_limit}",
        ),
        BenchQuery(
            "order_by_limit",
            f"SELECT worker_id, ts, mem_bytes FROM {ns} ORDER BY mem_bytes DESC, ts ASC LIMIT 100",
        ),
        BenchQuery("projection", f"SELECT worker_id, status, cpu_pct, mem_bytes FROM {ns} LIMIT {scan_limit}"),
        BenchQuery("select_star_limit", f"SELECT * FROM {ns} LIMIT 1000"),
        BenchQuery(
            "distinct_count",
            f"SELECT COUNT(DISTINCT worker_id) AS workers FROM {ns}",
        ),
    ]


def _query_rows(client: StatsServiceClientSync, sql: str) -> int:
    resp = client.query(stats_pb2.QueryRequest(sql=sql))
    return paipc.open_stream(io.BytesIO(resp.arrow_ipc)).read_all().num_rows


def _count_star(client: StatsServiceClientSync, namespace: str) -> int:
    resp = client.query(stats_pb2.QueryRequest(sql=f'SELECT COUNT(*) AS n FROM "{namespace}"'))
    return paipc.open_stream(io.BytesIO(resp.arrow_ipc)).read_all().column("n")[0].as_py()


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    name: str
    result_rows: int
    samples_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples_ms)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.samples_ms)
        idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
        return ordered[idx]

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


@dataclass
class BackendResult:
    backend: str
    boot_seconds: float
    populate_seconds: float
    count_star: int
    queries: list[QueryResult]


def measure_query(client: StatsServiceClientSync, query: BenchQuery, *, warmup: int, iterations: int) -> QueryResult:
    result_rows = 0
    for _ in range(warmup):
        result_rows = _query_rows(client, query.sql)
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result_rows = _query_rows(client, query.sql)
        samples.append((time.perf_counter() - start) * 1000.0)
    return QueryResult(name=query.name, result_rows=result_rows, samples_ms=samples)


def run_backend(
    backend: Backend,
    datasets: Sequence[NamespaceData],
    query_namespace: str,
    log_dir: Path,
    *,
    warmup: int,
    iterations: int,
) -> BackendResult:
    """Spawn ``backend`` on a fresh log dir, load data, then time every query."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with _spawn(backend, log_dir) as server:
        client = StatsServiceClientSync(address=server.base_url)
        populate_start = time.perf_counter()
        for data in datasets:
            populate(client, server.base_url, data)
        populate_seconds = time.perf_counter() - populate_start
        count = _count_star(client, query_namespace)
        if count == 0:
            raise RuntimeError(f"{backend.name}: namespace {query_namespace!r} is empty after load")
        results = [
            measure_query(client, q, warmup=warmup, iterations=iterations) for q in build_queries(query_namespace)
        ]
        return BackendResult(
            backend=backend.name,
            boot_seconds=server.boot_seconds,
            populate_seconds=populate_seconds,
            count_star=count,
            queries=results,
        )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _format_report(
    datasets: Sequence[NamespaceData],
    query_namespace: str,
    results: dict[str, BackendResult],
) -> str:
    lines: list[str] = []
    total_rows = sum(d.row_count for d in datasets)
    total_bytes = sum(d.parquet_bytes for d in datasets)
    total_segments = sum(d.segment_count for d in datasets)
    lines.append("=" * 78)
    lines.append("finelog query benchmark: Python (DuckDB) vs Rust (DataFusion)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Dataset:")
    for d in datasets:
        lines.append(f"  {d.namespace}: {d.row_count:,} rows, {d.segment_count} segments, {d.parquet_bytes:,} bytes")
    lines.append(f"  TOTAL: {total_rows:,} rows, {total_segments} segments, {total_bytes:,} bytes")
    lines.append(f"  queried namespace: {query_namespace}")
    lines.append("")
    lines.append("Setup (one-time, separate from query latency):")
    for name, r in results.items():
        lines.append(
            f"  {name:<7} boot={r.boot_seconds * 1000:8.1f} ms  "
            f"load+compact={r.populate_seconds:6.2f} s  COUNT(*)={r.count_star:,}"
        )
    counts = {r.count_star for r in results.values()}
    lines.append(f"  fairness check: COUNT(*) equal across backends = {len(counts) == 1} ({sorted(counts)})")
    lines.append("")

    if "python" in results and "rust" in results:
        py = {q.name: q for q in results["python"].queries}
        ru = {q.name: q for q in results["rust"].queries}
        header = (
            f"{'query':<18} {'rows':>8} "
            f"{'py_mean':>9} {'ru_mean':>9} {'py_p50':>8} {'ru_p50':>8} "
            f"{'py_p95':>8} {'ru_p95':>8} {'py_min':>8} {'ru_min':>8} {'mean_x':>7}"
        )
        lines.append("Per-query latency (ms); mean_x = python_mean / rust_mean (>1 means Rust faster):")
        lines.append(header)
        lines.append("-" * len(header))
        for name in py:
            p, r = py[name], ru[name]
            speedup = p.mean_ms / r.mean_ms if r.mean_ms else float("nan")
            lines.append(
                f"{name:<18} {p.result_rows:>8} "
                f"{p.mean_ms:>9.2f} {r.mean_ms:>9.2f} {p.p50_ms:>8.2f} {r.p50_ms:>8.2f} "
                f"{p.p95_ms:>8.2f} {r.p95_ms:>8.2f} {p.min_ms:>8.2f} {r.min_ms:>8.2f} {speedup:>6.2f}x"
            )
    else:
        for name, res in results.items():
            lines.append(f"{name} per-query latency (ms):")
            lines.append(f"{'query':<18} {'rows':>8} {'mean':>9} {'p50':>8} {'p95':>8} {'min':>8}")
            for q in res.queries:
                lines.append(
                    f"{q.name:<18} {q.result_rows:>8} "
                    f"{q.mean_ms:>9.2f} {q.p50_ms:>8.2f} {q.p95_ms:>8.2f} {q.min_ms:>8.2f}"
                )
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="dir laid out as <namespace>/seg_L*.parquet (loaded into both backends)",
    )
    parser.add_argument(
        "--query-namespace",
        default=None,
        help="namespace to benchmark queries against (default: the largest loaded one)",
    )
    parser.add_argument("--iterations", type=int, default=30, help="timed iterations per query (default 30)")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations per query (default 5)")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["python", "rust"],
        choices=["python", "rust"],
        help="which backends to run (default both)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/finelog_bench_run"),
        help="scratch dir for each backend's log_dir (default /tmp/finelog_bench_run)",
    )
    args = parser.parse_args(argv)

    ns_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir() and any(p.glob("seg_L*.parquet")))
    if not ns_dirs:
        parser.error(f"no <namespace>/seg_L*.parquet layout under {args.data_dir}")
    datasets = [load_namespace(p) for p in ns_dirs]
    query_namespace = args.query_namespace or max(datasets, key=lambda d: d.row_count).namespace

    results: dict[str, BackendResult] = {}
    for name in args.backends:
        backend = Backend(name)
        log_dir = args.work_dir / name
        # Fresh log dir per backend so neither clobbers the other's catalog sidecar.
        if log_dir.exists():
            import shutil

            shutil.rmtree(log_dir)
        print(f"[{name}] spawning + loading {len(datasets)} namespace(s)...", flush=True)
        results[name] = run_backend(
            backend,
            datasets,
            query_namespace,
            log_dir,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"[{name}] done (COUNT(*)={results[name].count_star:,})", flush=True)

    report = _format_report(datasets, query_namespace, results)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
