# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog StatsService query, planning, pruning, memory, and load measurement."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import io
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

import httpx
import pyarrow as pa
import pyarrow.ipc as paipc

from finelog.benchmarks.telemetry_workload_corpus import (
    DatasetSpec,
    Workload,
    WorkloadName,
    build_workloads,
)
from finelog.embedded import require_embedded_server
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

_DURATION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ns|µs|us|ms|s)")
# Every scan metric below is a DataFusion counter, printed by
# `human_readable_count`: decimal scaling with a bare ` K`/` M`/` B`/` T`
# suffix. `bytes_scanned` is a counter too, so it scales by 1000, not 1024.
_COUNT = r"[0-9]+(?:\.[0-9]+)?\s*[KMBT]?"
_COUNT_PARTS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([KMBT]?)")
_FILES_RE = re.compile(rf"files_ranges_pruned_statistics=({_COUNT}) total → ({_COUNT}) matched")
_ROW_GROUPS_RE = re.compile(rf"row_groups_pruned_statistics=({_COUNT}) total → ({_COUNT}) matched")
_BYTES_RE = re.compile(rf"bytes_scanned=({_COUNT})")
_METADATA_RE = re.compile(r"metadata_load_time=([0-9.]+(?:ns|µs|us|ms|s))")
_OPEN_RE = re.compile(r"time_elapsed_opening=([0-9.]+(?:ns|µs|us|ms|s))")
_PUSHDOWN_MATCHED_RE = re.compile(rf"pushdown_rows_matched=({_COUNT})")
_PUSHDOWN_PRUNED_RE = re.compile(rf"pushdown_rows_pruned=({_COUNT})")
_PUSHDOWN_TIME_RE = re.compile(r"row_pushdown_eval_time=([0-9.]+(?:ns|µs|us|ms|s))")
_COUNT_MULTIPLIER = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}


# Budget for a benchmark-owned server to open its store and answer /health.
# Adoption footer-scans every segment, so a copied production log directory is
# far slower to open than a generated one.
SERVER_STARTUP_TIMEOUT = 300.0


class WorkerMode(StrEnum):
    QUERY = "query"
    PLAN = "plan"
    EXPLAIN = "explain"


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def local_server(
    binary: Path,
    log_dir: Path,
    *,
    query_timeout_ms: int,
    extra_args: Sequence[str] = (),
) -> Iterator[str]:
    """Run ``finelog-server`` over ``log_dir`` and yield its base address.

    The log directory must be a disposable local copy: starting Finelog activates
    normal maintenance, including compaction, layout rewrites, and index backfill.
    """
    port = unused_port()
    address = f"http://127.0.0.1:{port}"
    log_path = log_dir / "finelog-benchmark-server.log"
    with log_path.open("a") as log_file:
        process = subprocess.Popen(
            [
                str(binary),
                "--port",
                str(port),
                "--log-dir",
                str(log_dir),
                "--log-level",
                "warn",
                *extra_args,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "FINELOG_QUERY_TIMEOUT_MS": str(query_timeout_ms)},
        )
        try:
            deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text())
                try:
                    if httpx.get(f"{address}/health", timeout=1).is_success:
                        yield address
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            raise TimeoutError(f"Finelog did not become healthy; see {log_path}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def server_info(address: str) -> dict[str, object]:
    """Return the server's build, store, and cache diagnostics."""
    response = httpx.get(f"{address}/api/server", timeout=10)
    response.raise_for_status()
    return cast(dict[str, object], response.json())


@dataclass(frozen=True)
class ExplainMetrics:
    """Scan metrics from an analyzed plan's `DataSourceExec` nodes.

    The `pushdown_*` fields count rows the parquet decoder evaluated the filter
    against, which is where a predicate on a cheap column spends its time: a
    scan can read a few hundred KiB and still evaluate the whole namespace.
    """

    files_total: int
    files_matched: int
    row_groups_total: int
    row_groups_matched: int
    bytes_scanned: int
    metadata_load_ms: float
    file_open_ms: float
    pushdown_rows_matched: int
    pushdown_rows_pruned: int
    pushdown_eval_ms: float


@dataclass(frozen=True)
class QuerySample:
    latency_ms: float
    rows: int
    rss_before_bytes: int
    rss_after_bytes: int
    rss_peak_bytes: int


def ipc_bytes(batch: pa.RecordBatch) -> bytes:
    """Serialize one Arrow batch for the current StatsService write API."""
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def stats_client(address: str) -> StatsServiceClientSync:
    """Return a synchronous client for a benchmark-owned embedded server."""
    return StatsServiceClientSync(address=address)


def query_table(client: StatsServiceClientSync, sql: str) -> pa.Table:
    """Issue SQL through StatsService and decode its Arrow result."""
    response = client.query(stats_pb2.QueryRequest(sql=sql))
    return paipc.open_stream(io.BytesIO(response.arrow_ipc)).read_all()


def write_batch(
    client: StatsServiceClientSync,
    namespace: str,
    batch: pa.RecordBatch,
) -> int:
    """Write one generated batch through StatsService."""
    response = client.write_rows(stats_pb2.WriteRowsRequest(namespace=namespace, arrow_ipc=ipc_bytes(batch)))
    return response.rows_written


def maintain(address: str, namespace: str, *, force_compact_l0: bool) -> None:
    """Run one maintenance tick on a benchmark-owned server.

    A tick compacts and backfills a bounded number of index bundles.
    `force_compact_l0` additionally pushes sealed L0 files through the real
    compactor instead of waiting for its byte thresholds.
    """
    response = httpx.post(
        f"{address}/debug/maintain",
        json={"namespace": namespace, "force_compact_l0": force_compact_l0},
        timeout=3_600,
    )
    response.raise_for_status()


def rss_bytes() -> int:
    """Return resident memory for the harness process."""
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status has no VmRSS")


@contextmanager
def peak_rss_sampler() -> Iterator[list[int]]:
    """Sample process RSS every five milliseconds within a measured region."""
    stop = threading.Event()
    samples = [rss_bytes()]

    def sample() -> None:
        while not stop.wait(0.005):
            samples.append(rss_bytes())

    thread = threading.Thread(target=sample, name="finelog-benchmark-rss", daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join()
        samples.append(rss_bytes())


def measure_query(client: StatsServiceClientSync, sql: str) -> QuerySample:
    """Measure one StatsService request and the harness process RSS."""
    rss_before = rss_bytes()
    with peak_rss_sampler() as rss_samples:
        started = time.perf_counter()
        result = query_table(client, sql)
        latency_ms = (time.perf_counter() - started) * 1000
    return QuerySample(
        latency_ms=latency_ms,
        rows=result.num_rows,
        rss_before_bytes=rss_before,
        rss_after_bytes=rss_samples[-1],
        rss_peak_bytes=max(rss_samples),
    )


def _count(raw: str) -> int:
    """Decode a counter metric such as `19`, `4.10 K`, or `1.16 B`."""
    match = _COUNT_PARTS_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(f"invalid count: {raw!r}")
    return round(float(match.group(1)) * _COUNT_MULTIPLIER[match.group(2)])


def _duration_ms(raw: str) -> float:
    match = _DURATION_RE.fullmatch(raw)
    if match is None:
        raise ValueError(f"invalid duration: {raw!r}")
    value = float(match.group(1))
    multiplier = {
        "ns": 1e-6,
        "µs": 1e-3,
        "us": 1e-3,
        "ms": 1.0,
        "s": 1_000.0,
    }[match.group(2)]
    return value * multiplier


def parse_explain_metrics(plan: str) -> ExplainMetrics:
    """Aggregate DataFusion DataSourceExec metrics from an analyzed plan."""
    files = _FILES_RE.findall(plan)
    row_groups = _ROW_GROUPS_RE.findall(plan)
    return ExplainMetrics(
        files_total=sum(_count(total) for total, _matched in files),
        files_matched=sum(_count(matched) for _total, matched in files),
        row_groups_total=sum(_count(total) for total, _matched in row_groups),
        row_groups_matched=sum(_count(matched) for _total, matched in row_groups),
        bytes_scanned=sum(_count(value) for value in _BYTES_RE.findall(plan)),
        metadata_load_ms=sum(_duration_ms(value) for value in _METADATA_RE.findall(plan)),
        file_open_ms=sum(_duration_ms(value) for value in _OPEN_RE.findall(plan)),
        pushdown_rows_matched=sum(_count(value) for value in _PUSHDOWN_MATCHED_RE.findall(plan)),
        pushdown_rows_pruned=sum(_count(value) for value in _PUSHDOWN_PRUNED_RE.findall(plan)),
        pushdown_eval_ms=sum(_duration_ms(value) for value in _PUSHDOWN_TIME_RE.findall(plan)),
    )


def explain_metrics(client: StatsServiceClientSync, sql: str) -> ExplainMetrics:
    """Run EXPLAIN ANALYZE and return its scan metrics."""
    explained = query_table(client, f"EXPLAIN ANALYZE {sql}")
    plans = explained.column("plan").to_pylist()
    if len(plans) != 1:
        raise RuntimeError(f"expected one analyzed plan, received {len(plans)}")
    return parse_explain_metrics(str(plans[0]))


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a nearest-rank percentile."""
    ordered = sorted(samples)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize_samples(samples: Sequence[QuerySample]) -> dict[str, float | int]:
    """Summarize query samples without discarding extrema or RSS."""
    latencies = [sample.latency_ms for sample in samples]
    return {
        "iterations": len(samples),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "rows": samples[-1].rows,
        "rss_peak_bytes": max(sample.rss_peak_bytes for sample in samples),
        "rss_growth_peak_bytes": max(sample.rss_peak_bytes - sample.rss_before_bytes for sample in samples),
    }


def single_process_measurement(
    log_dir: Path,
    workload: Workload,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Measure warmed StatsService queries and planning in one server process."""
    server = require_embedded_server()(log_dir=str(log_dir))
    try:
        client = stats_client(server.address)
        for _ in range(warmup):
            query_table(client, workload.sql)
        samples = [measure_query(client, workload.sql) for _ in range(iterations)]
        planning_samples = [measure_query(client, f"EXPLAIN {workload.sql}") for _ in range(iterations)]
        metrics = explain_metrics(client, workload.sql)
        return {
            "workload": workload.name.value,
            "query": summarize_samples(samples),
            "planning_proxy": summarize_samples(planning_samples),
            "explain": asdict(metrics),
        }
    finally:
        server.stop()


def worker_measure(
    log_dir: Path,
    workload_name: WorkloadName,
    spec: DatasetSpec,
    mode: WorkerMode,
    sql_override: str | None,
) -> dict[str, object]:
    """Execute one process-cold worker request."""
    workloads = {workload.name: workload for workload in build_workloads(spec)}
    if sql_override is not None and workload_name not in workloads:
        workload = Workload(workload_name, sql_override)
    else:
        workload = workloads[workload_name]
        if sql_override is not None:
            workload = replace(workload, sql=sql_override)
    server = require_embedded_server()(log_dir=str(log_dir))
    try:
        client = stats_client(server.address)
        if mode == WorkerMode.EXPLAIN:
            return {
                "workload": workload.name.value,
                "explain": asdict(explain_metrics(client, workload.sql)),
            }
        sql = workload.sql if mode == WorkerMode.QUERY else f"EXPLAIN {workload.sql}"
        return {
            "workload": workload.name.value,
            "sample": asdict(measure_query(client, sql)),
        }
    finally:
        server.stop()


def run_cold_worker(
    log_dir: Path,
    workload: Workload,
    spec: DatasetSpec,
    mode: WorkerMode,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "finelog.benchmarks.telemetry_layout_bench",
        "_worker",
        "--log-dir",
        str(log_dir),
        "--rows",
        str(spec.rows),
        "--duration-days",
        str(spec.duration_days),
        "--batch-rows",
        str(spec.batch_rows),
        "--segments",
        str(spec.segments),
        "--decoy-namespaces",
        str(spec.decoy_namespaces),
        "--distinct-run-ids",
        str(spec.distinct_run_ids),
        "--workload",
        workload.name.value,
        "--mode",
        mode.value,
        "--sql",
        workload.sql,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return cast(dict[str, object], json.loads(completed.stdout))


def cold_samples(
    log_dir: Path,
    workload: Workload,
    spec: DatasetSpec,
    iterations: int,
    mode: WorkerMode,
) -> list[QuerySample]:
    """Collect process-cold samples, one embedded server per request."""
    samples: list[QuerySample] = []
    for _ in range(iterations):
        payload = run_cold_worker(log_dir, workload, spec, mode)
        sample = cast(dict[str, float | int], payload["sample"])
        samples.append(QuerySample(**sample))
    return samples


def concurrency_measurement(
    log_dir: Path,
    workloads: Sequence[Workload],
    levels: Sequence[int],
    queries_per_worker: int,
) -> dict[str, object]:
    """Measure warm mixed-workload StatsService concurrency."""
    server = require_embedded_server()(log_dir=str(log_dir))
    try:
        warm_client = stats_client(server.address)
        for workload in workloads:
            query_table(warm_client, workload.sql)
        results: dict[str, object] = {}
        for level in levels:
            sqls = [workloads[index % len(workloads)].sql for index in range(level * queries_per_worker)]
            rss_before = rss_bytes()
            with peak_rss_sampler() as rss_samples:
                started = time.perf_counter()

                def query(sql: str) -> float:
                    query_started = time.perf_counter()
                    query_table(stats_client(server.address), sql)
                    return (time.perf_counter() - query_started) * 1000

                with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                    latencies = list(executor.map(query, sqls))
                wall_seconds = time.perf_counter() - started
            results[str(level)] = {
                "concurrency": level,
                "queries": len(sqls),
                "p50_ms": statistics.median(latencies),
                "p95_ms": percentile(latencies, 0.95),
                "throughput_queries_per_second": len(sqls) / wall_seconds,
                "rss_peak_bytes": max(rss_samples),
                "rss_growth_peak_bytes": max(rss_samples) - rss_before,
            }
        return results
    finally:
        server.stop()


def runtime_environment() -> dict[str, object]:
    """Capture versions and definitions needed to reproduce a result."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "pyarrow": pa.__version__,
        "marin_finelog": importlib.metadata.version("marin-finelog"),
        "marin_finelog_server": importlib.metadata.version("marin-finelog-server"),
        "process_cold_definition": (
            "new Python process, embedded server, and DataFusion metadata cache; " "host page cache unchanged"
        ),
        "planning_proxy_definition": "StatsService wall time for EXPLAIN without ANALYZE",
    }
