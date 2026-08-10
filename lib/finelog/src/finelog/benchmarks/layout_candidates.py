# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build current-layout stores and local telemetry layout candidates.

The harness uses the native embedded Finelog server and its public StatsService
RPC. It does not contact Iris, GCS, or another remote store. The generated rows
match the flattened ``TelltaleMetric`` shape, with the forwarded ``cluster``
column included because fleet queries depend on it.

The initial ``baseline`` command intentionally benchmarks only the current
single ``telltale`` namespace keyed by metric name. It records:

* process-cold and in-process warm query latency;
* an ``EXPLAIN`` RPC wall-time proxy for binding and planning;
* DataFusion's file-open, metadata, pruning, and byte-scan metrics from
  ``EXPLAIN ANALYZE``;
* process RSS before and after each measured query.

Process-cold means a new Python process, embedded server, and DataFusion
metadata cache. It does not drop the host page cache.

Run from ``lib/finelog``:

    uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench baseline \
        --rows 100000 --work-dir /tmp/finelog-telemetry-baseline \
        --output /tmp/finelog-telemetry-baseline.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from finelog.benchmarks.query_measurement import (
    WorkerMode,
    cold_samples,
    concurrency_measurement,
    maintain,
    peak_rss_sampler,
    percentile,
    query_table,
    rss_bytes,
    run_cold_worker,
    runtime_environment,
    single_process_measurement,
    stats_client,
    summarize_samples,
    write_batch,
)
from finelog.benchmarks.telemetry_workload_corpus import (
    CLUSTERS,
    MILLISECONDS_PER_DAY,
    RUN_METRIC_NAMES,
    SERIES_COUNT,
    DatasetSpec,
    SignalGroup,
    Workload,
    WorkloadName,
    build_workloads,
    dataset_facts,
    generate_batch,
    generate_batches,
    telltale_schema,
)
from finelog.embedded import require_embedded_server
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

ROLLUP_DIMENSIONS = (
    "bucket",
    "name",
    "source",
    "run",
    "job_id",
    "worker",
    "region",
    "cluster",
    "le",
    "method",
    "status",
    "error_code",
    "device",
    "user_name",
    "project",
    "model",
    "component",
)
RUN_CATALOG_NAMESPACE = "telemetry.run.v1"
WORKLOAD_METRIC_NAMES = {
    WorkloadName.RUN_DRILLDOWN_15M: RUN_METRIC_NAMES,
    WorkloadName.RUN_DRILLDOWN_24H: RUN_METRIC_NAMES,
    WorkloadName.RUN_DRILLDOWN_7D: RUN_METRIC_NAMES,
    WorkloadName.WORKER_OUTLIERS: (
        "process_resident_memory_bytes",
        "process_cpu_seconds_total",
    ),
    WorkloadName.VLLM_HISTOGRAM: ("vllm_time_to_first_token_seconds_bucket",),
    WorkloadName.FLEET_FAILURES: (
        "iris_task_failures_total",
        "iris_worker_state",
    ),
    WorkloadName.NODE_DEVICE_HISTORY: (
        "iris_device_temperature_celsius",
        "iris_device_memory_used_bytes",
        "iris_device_health",
    ),
    WorkloadName.ERRORS: (
        "iris_rpc_errors_total",
        "iris_error_log_entries_total",
    ),
    WorkloadName.ALERT_EVALUATION: (
        "telemetry_progress_age_seconds",
        "telemetry_worker_heartbeat_age_seconds",
        "telemetry_vllm_queue_saturation",
        "telemetry_gpu_health",
        "telemetry_forwarding_gap",
        "telemetry_producer_heartbeat",
    ),
    WorkloadName.ACCOUNTING: ("iris_accelerator_seconds_total",),
}


@dataclass(frozen=True)
class SegmentManifestEntry:
    path: str
    min_ts_ms: int
    max_ts_ms: int
    min_name: str
    max_name: str
    rows: int
    row_groups: int
    bytes: int


@dataclass(frozen=True)
class PartitionedSegment:
    entry: SegmentManifestEntry
    day: int
    bucket: int


def _initialize_namespace(log_dir: Path, namespace: str, decoy_namespaces: int) -> None:
    _initialize_layout(
        log_dir,
        {namespace: telltale_schema()},
        decoy_namespaces=decoy_namespaces,
    )


def _initialize_layout(
    log_dir: Path,
    schemas: dict[str, stats_pb2.Schema],
    *,
    decoy_namespaces: int,
) -> None:
    server = require_embedded_server()(log_dir=str(log_dir))
    try:
        client = stats_client(server.address)
        for namespace, schema in schemas.items():
            client.register_table(stats_pb2.RegisterTableRequest(namespace=namespace, schema=schema))
        for decoy_index in range(decoy_namespaces):
            client.register_table(
                stats_pb2.RegisterTableRequest(
                    namespace=f"benchmark.decoy.{decoy_index:04d}",
                    schema=telltale_schema(),
                )
            )
    finally:
        server.stop()


def _principal(group: str, run: str, worker: str, cluster: str) -> str:
    if group in {
        SignalGroup.RUN.value,
        SignalGroup.WORKER.value,
        SignalGroup.INFERENCE.value,
        SignalGroup.ACCOUNTING.value,
    }:
        return run
    if group == SignalGroup.NODE.value:
        return worker
    return cluster


def _numeric_suffix(value: str) -> int:
    return int(value.rsplit("-", 1)[1])


def _partition_bucket(group: str, run: str, worker: str, cluster: str, buckets: int) -> int:
    principal = _principal(group, run, worker, cluster)
    if principal.startswith("cw-") or principal == "marin":
        return CLUSTERS.index(principal) % buckets
    return _numeric_suffix(principal) % buckets


def _filter_candidate_batch(
    batch: pa.RecordBatch,
    *,
    group: SignalGroup | None,
    bucket: int | None,
    buckets: int,
) -> pa.RecordBatch:
    groups = batch.column(batch.schema.get_field_index("source")).to_pylist()
    runs = batch.column(batch.schema.get_field_index("run")).to_pylist()
    workers = batch.column(batch.schema.get_field_index("worker")).to_pylist()
    clusters = batch.column(batch.schema.get_field_index("cluster")).to_pylist()
    keep = [
        (group is None or row_group == group.value)
        and (bucket is None or _partition_bucket(row_group, run, worker, cluster, buckets) == bucket)
        for row_group, run, worker, cluster in zip(
            groups,
            runs,
            workers,
            clusters,
            strict=True,
        )
    ]
    return batch.filter(pa.array(keep))


def _add_layout_key(batch: pa.RecordBatch) -> pa.RecordBatch:
    groups = batch.column(batch.schema.get_field_index("source")).to_pylist()
    runs = batch.column(batch.schema.get_field_index("run")).to_pylist()
    workers = batch.column(batch.schema.get_field_index("worker")).to_pylist()
    clusters = batch.column(batch.schema.get_field_index("cluster")).to_pylist()
    names = batch.column(batch.schema.get_field_index("name")).to_pylist()
    timestamps = batch.column(batch.schema.get_field_index("ts")).to_pylist()
    keys = [
        (f"{group}|{_principal(group, run, worker, cluster)}|{name}|" f"{int(timestamp.timestamp() * 1000):013d}")
        for group, run, worker, cluster, name, timestamp in zip(
            groups,
            runs,
            workers,
            clusters,
            names,
            timestamps,
            strict=True,
        )
    ]
    return batch.append_column(
        pa.field("layout_key", pa.string(), nullable=False),
        pa.array(keys, type=pa.string()),
    )


def _write_candidate_segment(
    client: StatsServiceClientSync,
    address: str,
    namespace: str,
    schema: stats_pb2.Schema,
    first: pa.RecordBatch,
    second: pa.RecordBatch,
) -> Path:
    if first.num_rows == 0 or second.num_rows == 0:
        raise RuntimeError(f"candidate segment {namespace} has an empty compaction half")
    client.register_table(stats_pb2.RegisterTableRequest(namespace=namespace, schema=schema))
    write_batch(client, namespace, first)
    write_batch(client, namespace, second)
    maintain(address, namespace, force_compact_l0=True)
    return Path(namespace)


def _link_layout_store(
    log_dir: Path,
    namespace_entries: dict[str, Sequence[SegmentManifestEntry]],
    schemas: dict[str, stats_pb2.Schema],
) -> None:
    if log_dir.exists():
        raise FileExistsError(f"benchmark layout already exists: {log_dir}")
    _initialize_layout(log_dir, schemas, decoy_namespaces=0)
    for namespace, entries in namespace_entries.items():
        namespace_dir = log_dir / namespace
        next_seq = 1
        for entry in entries:
            os.link(entry.path, namespace_dir / f"seg_L3_{next_seq:019d}.parquet")
            next_seq += entry.rows
    server = require_embedded_server()(log_dir=str(log_dir))
    server.stop()


def populate_current_store(
    log_dir: Path,
    staging_dir: Path,
    spec: DatasetSpec,
) -> dict[str, float | int]:
    """Populate a current-layout store with a controlled number of segments.

    Each staging namespace receives two adjacent L0 writes, then Finelog's real
    compactor sorts them by ``(name, seq)`` into one Parquet file. The files are
    hard-linked into one logical ``telltale`` namespace at terminal level L3.
    Startup reconciliation adopts them into the current catalog. This preserves
    current encoding and query behavior while controlling file count without
    waiting for production-sized compaction thresholds.
    """
    if log_dir.exists():
        raise FileExistsError(f"benchmark log dir already exists: {log_dir}")
    if staging_dir.exists():
        raise FileExistsError(f"benchmark staging dir already exists: {staging_dir}")
    if spec.segments > spec.rows // 2:
        raise ValueError("segments must leave at least two rows per segment")
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    server = require_embedded_server()(log_dir=str(staging_dir), debug_admin=True)
    rows_written = 0
    write_seconds = 0.0
    compact_seconds = 0.0
    try:
        client = stats_client(server.address)
        for segment_index in range(spec.segments):
            segment_start = segment_index * spec.rows // spec.segments
            segment_stop = (segment_index + 1) * spec.rows // spec.segments
            midpoint = segment_start + (segment_stop - segment_start) // 2
            staging_namespace = f"benchmark_segment_{segment_index:04d}"
            client.register_table(
                stats_pb2.RegisterTableRequest(
                    namespace=staging_namespace,
                    schema=telltale_schema(),
                )
            )
            write_started = time.perf_counter()
            rows_written += write_batch(
                client,
                staging_namespace,
                generate_batch(spec, segment_start, midpoint),
            )
            rows_written += write_batch(
                client,
                staging_namespace,
                generate_batch(spec, midpoint, segment_stop),
            )
            write_seconds += time.perf_counter() - write_started
            compact_started = time.perf_counter()
            maintain(server.address, staging_namespace, force_compact_l0=True)
            compact_seconds += time.perf_counter() - compact_started
    finally:
        server.stop()

    _initialize_namespace(log_dir, spec.namespace, spec.decoy_namespaces)
    namespace_dir = log_dir / spec.namespace
    parquet_files: list[Path] = []
    for segment_index in range(spec.segments):
        staging_namespace = f"benchmark_segment_{segment_index:04d}"
        sources = list((staging_dir / staging_namespace).glob("seg_L1_*.parquet"))
        if len(sources) != 1:
            raise RuntimeError(f"expected one compacted file for {staging_namespace}, found {len(sources)}")
        logical_min_seq = segment_index * spec.rows // spec.segments + 1
        destination = namespace_dir / f"seg_L3_{logical_min_seq:019d}.parquet"
        os.link(sources[0], destination)
        parquet_files.append(destination)

    server = require_embedded_server()(log_dir=str(log_dir), debug_admin=True)
    try:
        client = stats_client(server.address)
        count = query_table(client, f'SELECT count(*) AS rows FROM "{spec.namespace}"')
        observed_rows = int(count.column("rows")[0].as_py())
        if rows_written != spec.rows or observed_rows != spec.rows:
            raise RuntimeError(
                f"row-count mismatch: generated={spec.rows}, written={rows_written}, queried={observed_rows}"
            )
    finally:
        server.stop()
    parquet_bytes = sum(path.stat().st_size for path in parquet_files)
    return {
        "rows": spec.rows,
        "parquet_files": spec.segments,
        "parquet_bytes": parquet_bytes,
        "write_seconds": write_seconds,
        "compact_seconds": compact_seconds,
        "prepare_seconds": time.perf_counter() - started,
    }


def _parquet_stat_ms(value: object) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    if isinstance(value, int):
        return value
    raise TypeError(f"unsupported parquet timestamp statistic: {value!r}")


def build_segment_manifest(log_dir: Path, namespace: str) -> tuple[SegmentManifestEntry, ...]:
    """Read local Parquet footers into a deterministic segment manifest."""
    entries: list[SegmentManifestEntry] = []
    for path in sorted((log_dir / namespace).glob("seg_L*.parquet")):
        parquet_file = pq.ParquetFile(path)
        arrow_names = parquet_file.schema_arrow.names
        ts_index = arrow_names.index("ts")
        name_index = arrow_names.index("name")
        min_ts: int | None = None
        max_ts: int | None = None
        min_name: str | None = None
        max_name: str | None = None
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            ts_stats = row_group.column(ts_index).statistics
            name_stats = row_group.column(name_index).statistics
            if ts_stats is None or name_stats is None:
                raise RuntimeError(f"segment lacks required statistics: {path}")
            row_min_ts = _parquet_stat_ms(ts_stats.min)
            row_max_ts = _parquet_stat_ms(ts_stats.max)
            row_min_name = str(name_stats.min)
            row_max_name = str(name_stats.max)
            min_ts = row_min_ts if min_ts is None else min(min_ts, row_min_ts)
            max_ts = row_max_ts if max_ts is None else max(max_ts, row_max_ts)
            min_name = row_min_name if min_name is None else min(min_name, row_min_name)
            max_name = row_max_name if max_name is None else max(max_name, row_max_name)
        if min_ts is None or max_ts is None or min_name is None or max_name is None:
            raise RuntimeError(f"empty segment is not benchmarkable: {path}")
        entries.append(
            SegmentManifestEntry(
                path=str(path.resolve()),
                min_ts_ms=min_ts,
                max_ts_ms=max_ts,
                min_name=min_name,
                max_name=max_name,
                rows=parquet_file.metadata.num_rows,
                row_groups=parquet_file.metadata.num_row_groups,
                bytes=path.stat().st_size,
            )
        )
    if not entries:
        raise FileNotFoundError(f"no Finelog segments under {log_dir / namespace}")
    return tuple(entries)


def select_manifest_entries(
    manifest: Sequence[SegmentManifestEntry],
    workload: Workload,
) -> tuple[SegmentManifestEntry, ...]:
    """Select files whose timestamp and name bounds overlap a workload."""
    return tuple(
        entry
        for entry in manifest
        if entry.max_ts_ms >= workload.start_ms
        and entry.min_ts_ms < workload.end_ms
        and any(entry.min_name <= name <= entry.max_name for name in WORKLOAD_METRIC_NAMES[workload.name])
    )


def _create_mirror(
    log_dir: Path,
    namespace: str,
    entries: Sequence[SegmentManifestEntry],
    *,
    decoy_namespaces: int,
) -> None:
    if log_dir.exists():
        raise FileExistsError(f"benchmark mirror already exists: {log_dir}")
    _initialize_namespace(log_dir, namespace, decoy_namespaces)
    namespace_dir = log_dir / namespace
    next_seq = 1
    for entry in entries:
        destination = namespace_dir / f"seg_L3_{next_seq:019d}.parquet"
        os.link(entry.path, destination)
        next_seq += entry.rows
    server = require_embedded_server()(log_dir=str(log_dir))
    try:
        observed = query_table(
            stats_client(server.address),
            f'SELECT count(*) AS rows FROM "{namespace}"',
        )
        expected_rows = sum(entry.rows for entry in entries)
        if int(observed.column("rows")[0].as_py()) != expected_rows:
            raise RuntimeError(f"mirror row count differs from manifest: {log_dir}")
    finally:
        server.stop()


def table_digest(table: pa.Table) -> str:
    def normalize(value: object) -> object:
        if isinstance(value, float):
            return round(value, 9)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    rows = sorted(
        json.dumps(
            normalize(row),
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in table.to_pylist()
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _query_digest(log_dir: Path, workload: Workload) -> str:
    server = require_embedded_server()(log_dir=str(log_dir))
    try:
        return table_digest(query_table(stats_client(server.address), workload.sql))
    finally:
        server.stop()


def _selection_latency(
    manifest: Sequence[SegmentManifestEntry],
    workload: Workload,
    iterations: int = 1_000,
) -> dict[str, float | int]:
    latencies: list[float] = []
    selected: tuple[SegmentManifestEntry, ...] = ()
    for _ in range(iterations):
        started = time.perf_counter()
        selected = select_manifest_entries(manifest, workload)
        latencies.append((time.perf_counter() - started) * 1_000)
    return {
        "iterations": iterations,
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "files_total": len(manifest),
        "files_matched": len(selected),
        "row_groups_total": sum(entry.row_groups for entry in manifest),
        "row_groups_matched_upper_bound": sum(entry.row_groups for entry in selected),
        "bytes_total": sum(entry.bytes for entry in manifest),
        "bytes_in_matched_files": sum(entry.bytes for entry in selected),
    }


def _one_staged_entry(staging_dir: Path, namespace: str) -> SegmentManifestEntry:
    entries = build_segment_manifest(staging_dir, namespace)
    if len(entries) != 1:
        raise RuntimeError(f"expected one staged segment for {namespace}, found {len(entries)}")
    return entries[0]


def prepare_split_layout(
    staging_dir: Path,
    log_dir: Path,
    spec: DatasetSpec,
    *,
    files_per_group: int,
) -> dict[str, float | int]:
    """Build eight logical signal namespaces with fewer files per namespace."""
    started = time.perf_counter()
    schema = telltale_schema()
    entries_by_namespace: dict[str, list[SegmentManifestEntry]] = {
        f"telemetry.{group.value}": [] for group in SignalGroup
    }
    server = require_embedded_server()(log_dir=str(staging_dir), debug_admin=True)
    try:
        client = stats_client(server.address)
        for group in SignalGroup:
            for file_index in range(files_per_group):
                start = file_index * spec.rows // files_per_group
                stop = (file_index + 1) * spec.rows // files_per_group
                midpoint = start + (stop - start) // 2
                first = _filter_candidate_batch(
                    generate_batch(spec, start, midpoint),
                    group=group,
                    bucket=None,
                    buckets=1,
                )
                second = _filter_candidate_batch(
                    generate_batch(spec, midpoint, stop),
                    group=group,
                    bucket=None,
                    buckets=1,
                )
                staging_namespace = f"split_{group.value}_{file_index:04d}"
                _write_candidate_segment(
                    client,
                    server.address,
                    staging_namespace,
                    schema,
                    first,
                    second,
                )
                entries_by_namespace[f"telemetry.{group.value}"].append(
                    _one_staged_entry(staging_dir, staging_namespace)
                )
    finally:
        server.stop()
    schemas = {namespace: schema for namespace in entries_by_namespace}
    _link_layout_store(log_dir, entries_by_namespace, schemas)
    entries = [entry for namespace_entries in entries_by_namespace.values() for entry in namespace_entries]
    return {
        "prepare_seconds": time.perf_counter() - started,
        "parquet_files": len(entries),
        "parquet_bytes": sum(entry.bytes for entry in entries),
        "files_per_group": files_per_group,
    }


def prepare_clustered_layout(
    staging_dir: Path,
    log_dir: Path,
    spec: DatasetSpec,
) -> dict[str, float | int]:
    """Build a single namespace sorted by a workload-aware composite key."""
    started = time.perf_counter()
    schema = telltale_schema(layout_key=True)
    entries: list[SegmentManifestEntry] = []
    server = require_embedded_server()(log_dir=str(staging_dir), debug_admin=True)
    try:
        client = stats_client(server.address)
        for file_index in range(spec.segments):
            start = file_index * spec.rows // spec.segments
            stop = (file_index + 1) * spec.rows // spec.segments
            midpoint = start + (stop - start) // 2
            staging_namespace = f"clustered_{file_index:04d}"
            _write_candidate_segment(
                client,
                server.address,
                staging_namespace,
                schema,
                _add_layout_key(generate_batch(spec, start, midpoint)),
                _add_layout_key(generate_batch(spec, midpoint, stop)),
            )
            entries.append(_one_staged_entry(staging_dir, staging_namespace))
    finally:
        server.stop()
    _link_layout_store(
        log_dir,
        {spec.namespace: entries},
        {spec.namespace: schema},
    )
    return {
        "prepare_seconds": time.perf_counter() - started,
        "parquet_files": len(entries),
        "parquet_bytes": sum(entry.bytes for entry in entries),
    }


def prepare_partitioned_layout(
    staging_dir: Path,
    spec: DatasetSpec,
    *,
    buckets: int,
) -> tuple[tuple[PartitionedSegment, ...], dict[str, float | int]]:
    """Build day/hash files using hidden transforms over workload principals."""
    started = time.perf_counter()
    schema = telltale_schema()
    num_samples = math.ceil(spec.rows / SERIES_COUNT)
    partitioned: list[PartitionedSegment] = []
    server = require_embedded_server()(log_dir=str(staging_dir), debug_admin=True)
    try:
        client = stats_client(server.address)
        for day in range(spec.duration_days):
            sample_start = day * num_samples // spec.duration_days
            sample_stop = (day + 1) * num_samples // spec.duration_days
            start = min(sample_start * SERIES_COUNT, spec.rows)
            stop = min(sample_stop * SERIES_COUNT, spec.rows)
            if start >= stop:
                continue
            midpoint = start + (stop - start) // 2
            for bucket in range(buckets):
                first = _filter_candidate_batch(
                    generate_batch(spec, start, midpoint),
                    group=None,
                    bucket=bucket,
                    buckets=buckets,
                )
                second = _filter_candidate_batch(
                    generate_batch(spec, midpoint, stop),
                    group=None,
                    bucket=bucket,
                    buckets=buckets,
                )
                staging_namespace = f"partition_day_{day:03d}_bucket_{bucket:03d}"
                _write_candidate_segment(
                    client,
                    server.address,
                    staging_namespace,
                    schema,
                    first,
                    second,
                )
                partitioned.append(
                    PartitionedSegment(
                        entry=_one_staged_entry(staging_dir, staging_namespace),
                        day=day,
                        bucket=bucket,
                    )
                )
    finally:
        server.stop()
    return (
        tuple(partitioned),
        {
            "prepare_seconds": time.perf_counter() - started,
            "parquet_files": len(partitioned),
            "parquet_bytes": sum(item.entry.bytes for item in partitioned),
            "days": spec.duration_days,
            "hash_buckets": buckets,
        },
    )


def select_partitioned_entries(
    entries: Sequence[PartitionedSegment],
    workload: Workload,
) -> tuple[SegmentManifestEntry, ...]:
    buckets = max(item.bucket for item in entries) + 1
    selected_bucket = None if workload.hash_bucket is None else workload.hash_bucket % buckets
    return tuple(
        item.entry
        for item in entries
        if item.entry.max_ts_ms >= workload.start_ms
        and item.entry.min_ts_ms < workload.end_ms
        and (selected_bucket is None or item.bucket == selected_bucket)
    )


def _retarget_split_workload(workload: Workload) -> Workload:
    namespace = f"telemetry.{workload.group.value}"
    return replace(
        workload,
        sql=workload.sql.replace('"telltale"', f'"{namespace}"'),
    )


def _run_catalog_schema() -> stats_pb2.Schema:
    return stats_pb2.Schema(
        columns=[
            stats_pb2.Column(
                name="root_run_uid",
                type=stats_pb2.COLUMN_TYPE_STRING,
                nullable=False,
            ),
            stats_pb2.Column(
                name="event_ts",
                type=stats_pb2.COLUMN_TYPE_TIMESTAMP_MS,
                nullable=False,
            ),
            stats_pb2.Column(
                name="owner",
                type=stats_pb2.COLUMN_TYPE_STRING,
                nullable=False,
            ),
            stats_pb2.Column(
                name="status",
                type=stats_pb2.COLUMN_TYPE_STRING,
                nullable=False,
            ),
        ],
        key_column="root_run_uid",
    )


def _run_catalog_batch(start: int, stop: int, spec: DatasetSpec) -> pa.RecordBatch:
    duration_ms = spec.duration_days * MILLISECONDS_PER_DAY
    indices = range(start, stop)
    timestamps = [spec.start_ms + (index * duration_ms // max(spec.rows - 1, 1)) for index in indices]
    return pa.RecordBatch.from_arrays(
        [
            pa.array([f"run-{index:06d}" for index in indices], type=pa.string()),
            pa.array(timestamps, type=pa.timestamp("ms")),
            pa.array([f"owner-{index % 20:02d}" for index in indices], type=pa.string()),
            pa.array(
                ["running" if index % 4 else "failed" for index in indices],
                type=pa.string(),
            ),
        ],
        names=["root_run_uid", "event_ts", "owner", "status"],
    )


def run_catalog_benchmark(
    work_dir: Path,
    output: Path,
    *,
    run_ids: int,
    cold_iterations: int,
    warm_iterations: int,
    warmup_iterations: int,
    concurrency_levels: Sequence[int],
    concurrency_queries_per_worker: int,
) -> dict[str, object]:
    """Measure bounded prefix lookup over a 100,000-row run catalog."""
    if work_dir.exists():
        raise FileExistsError(f"benchmark work dir already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    log_dir = work_dir / "catalog"
    spec = DatasetSpec(
        rows=run_ids,
        duration_days=365,
        batch_rows=max(run_ids // 2, 1),
        segments=1,
        distinct_run_ids=run_ids,
    )
    server = require_embedded_server()(log_dir=str(log_dir), debug_admin=True)
    started = time.perf_counter()
    try:
        client = stats_client(server.address)
        client.register_table(
            stats_pb2.RegisterTableRequest(
                namespace=RUN_CATALOG_NAMESPACE,
                schema=_run_catalog_schema(),
            )
        )
        midpoint = run_ids // 2
        rows_written = write_batch(
            client,
            RUN_CATALOG_NAMESPACE,
            _run_catalog_batch(0, midpoint, spec),
        )
        rows_written += write_batch(
            client,
            RUN_CATALOG_NAMESPACE,
            _run_catalog_batch(midpoint, run_ids, spec),
        )
        maintain(server.address, RUN_CATALOG_NAMESPACE, force_compact_l0=True)
    finally:
        server.stop()
    sql = f"""
SELECT root_run_uid, event_ts, owner, status
FROM "{RUN_CATALOG_NAMESPACE}"
WHERE root_run_uid LIKE 'run-0999%'
  AND event_ts >= to_timestamp_millis({spec.start_ms})
  AND event_ts < to_timestamp_millis({spec.end_ms})
ORDER BY root_run_uid
LIMIT 100
""".strip()
    workload = Workload(WorkloadName.RUN_PREFIX_SEARCH, sql)
    measurement = _measure_workload_layout(
        log_dir,
        workload,
        spec,
        cold_iterations=cold_iterations,
        warm_iterations=warm_iterations,
        warmup_iterations=warmup_iterations,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_kind": "finelog_stats_service",
        "environment": runtime_environment(),
        "dataset": {
            "run_ids": run_ids,
            "namespace": RUN_CATALOG_NAMESPACE,
            "rows_written": rows_written,
            "parquet_files": len(list((log_dir / RUN_CATALOG_NAMESPACE).glob("seg_L*.parquet"))),
            "prepare_seconds": time.perf_counter() - started,
        },
        "workload": {"name": workload.name.value, "sql": sql},
        "measurement": measurement,
        "concurrency": concurrency_measurement(
            log_dir,
            [workload],
            concurrency_levels,
            concurrency_queries_per_worker,
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_baseline(
    work_dir: Path,
    output: Path,
    spec: DatasetSpec,
    *,
    cold_iterations: int,
    warm_iterations: int,
    warmup_iterations: int,
    concurrency_levels: Sequence[int],
    concurrency_queries_per_worker: int,
) -> dict[str, object]:
    if work_dir.exists():
        raise FileExistsError(f"benchmark work dir already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    log_dir = work_dir / "current"
    preparation = populate_current_store(log_dir, work_dir / "staging-current", spec)
    workloads = build_workloads(spec)
    warm = {
        workload.name.value: single_process_measurement(
            log_dir,
            workload,
            warmup=warmup_iterations,
            iterations=warm_iterations,
        )
        for workload in workloads
    }
    cold: dict[str, object] = {}
    for workload in workloads:
        cold[workload.name.value] = {
            "query": summarize_samples(
                cold_samples(
                    log_dir,
                    workload,
                    spec,
                    cold_iterations,
                    WorkerMode.QUERY,
                )
            ),
            "planning_proxy": summarize_samples(
                cold_samples(
                    log_dir,
                    workload,
                    spec,
                    cold_iterations,
                    WorkerMode.PLAN,
                )
            ),
            "explain": run_cold_worker(
                log_dir,
                workload,
                spec,
                WorkerMode.EXPLAIN,
            )["explain"],
        }
    concurrency = concurrency_measurement(
        log_dir,
        workloads,
        concurrency_levels,
        concurrency_queries_per_worker,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_kind": "finelog_stats_service",
        "environment": runtime_environment(),
        "dataset": dataset_facts(spec),
        "preparation": preparation,
        "workloads": {workload.name.value: workload.sql for workload in workloads},
        "current": {"cold": cold, "warm": warm, "concurrency": concurrency},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _measure_workload_layout(
    log_dir: Path,
    workload: Workload,
    spec: DatasetSpec,
    *,
    cold_iterations: int,
    warm_iterations: int,
    warmup_iterations: int,
) -> dict[str, object]:
    return {
        "cold": {
            "query": summarize_samples(
                cold_samples(
                    log_dir,
                    workload,
                    spec,
                    cold_iterations,
                    WorkerMode.QUERY,
                )
            ),
            "planning_proxy": summarize_samples(
                cold_samples(
                    log_dir,
                    workload,
                    spec,
                    cold_iterations,
                    WorkerMode.PLAN,
                )
            ),
            "explain": run_cold_worker(
                log_dir,
                workload,
                spec,
                WorkerMode.EXPLAIN,
            )["explain"],
        },
        "warm": single_process_measurement(
            log_dir,
            workload,
            warmup=warmup_iterations,
            iterations=warm_iterations,
        ),
    }


def run_manifest_candidate(
    source_log_dir: Path,
    work_dir: Path,
    output: Path,
    spec: DatasetSpec,
    *,
    cold_iterations: int,
    warm_iterations: int,
    warmup_iterations: int,
    concurrency_levels: Sequence[int],
    concurrency_queries_per_worker: int,
) -> dict[str, object]:
    """Benchmark referenced-only binding and manifest-selected file lists."""
    if work_dir.exists():
        raise FileExistsError(f"benchmark work dir already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    manifest = build_segment_manifest(source_log_dir, spec.namespace)
    workloads = build_workloads(spec)
    source_digests = {workload.name.value: _query_digest(source_log_dir, workload) for workload in workloads}

    referenced_log_dir = work_dir / "referenced-only"
    _create_mirror(
        referenced_log_dir,
        spec.namespace,
        manifest,
        decoy_namespaces=0,
    )
    referenced_only = _measure_shared_layout(
        referenced_log_dir,
        workloads,
        source_digests,
        replace(spec, decoy_namespaces=0),
        cold_iterations=cold_iterations,
        warm_iterations=warm_iterations,
        warmup_iterations=warmup_iterations,
        concurrency_levels=concurrency_levels,
        concurrency_queries_per_worker=concurrency_queries_per_worker,
    )

    results: dict[str, object] = {}
    mirror_dirs: dict[WorkloadName, Path] = {}
    for workload in workloads:
        selected = select_manifest_entries(manifest, workload)
        if not selected:
            raise RuntimeError(f"manifest selected no files for {workload.name.value}")
        mirror_dir = work_dir / workload.name.value
        _create_mirror(
            mirror_dir,
            spec.namespace,
            selected,
            decoy_namespaces=0,
        )
        mirror_dirs[workload.name] = mirror_dir
        candidate_digest = _query_digest(mirror_dir, workload)
        if candidate_digest != source_digests[workload.name.value]:
            raise RuntimeError(f"manifest mirror changed {workload.name.value} results")
        results[workload.name.value] = {
            "selection": _selection_latency(manifest, workload),
            "result_digest": candidate_digest,
            "measurement": _measure_workload_layout(
                mirror_dir,
                workload,
                replace(spec, decoy_namespaces=0),
                cold_iterations=cold_iterations,
                warm_iterations=warm_iterations,
                warmup_iterations=warmup_iterations,
            ),
        }
    alert_workload = next(workload for workload in workloads if workload.name == WorkloadName.ALERT_EVALUATION)
    concurrency = concurrency_measurement(
        mirror_dirs[WorkloadName.ALERT_EVALUATION],
        [alert_workload],
        concurrency_levels,
        concurrency_queries_per_worker,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_kind": "finelog_stats_service",
        "result_digest_float_round_digits": 9,
        "environment": runtime_environment(),
        "dataset": dataset_facts(spec),
        "source_log_dir": str(source_log_dir),
        "manifest": [
            {
                **asdict(entry),
                "path": Path(entry.path).name,
            }
            for entry in manifest
        ],
        "candidates": {
            "all_files_referenced_namespace_control": {
                "implementation": (
                    "local prototype: bind only the referenced namespace while retaining "
                    "the current shared DataFusion metadata cache and all source files"
                ),
                **referenced_only,
            },
            "segment_manifest_timestamp_pruning": {
                "implementation": (
                    "local prototype: footer-derived timestamp manifest selects files before "
                    "the unchanged Finelog NamespaceProvider is built"
                ),
                "workloads": results,
                "concurrency_alert_evaluation": concurrency,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _create_nonempty_decoy_store(
    log_dir: Path,
    namespace: str,
    entries: Sequence[SegmentManifestEntry],
    *,
    decoy_namespaces: int,
) -> None:
    if log_dir.exists():
        raise FileExistsError(f"benchmark binding store already exists: {log_dir}")
    _initialize_namespace(log_dir, namespace, decoy_namespaces)
    namespace_dir = log_dir / namespace
    next_seq = 1
    for entry in entries:
        os.link(entry.path, namespace_dir / f"seg_L3_{next_seq:019d}.parquet")
        next_seq += entry.rows
    decoy_entry = entries[0]
    for decoy_index in range(decoy_namespaces):
        decoy_dir = log_dir / f"benchmark.decoy.{decoy_index:04d}"
        os.link(
            decoy_entry.path,
            decoy_dir / "seg_L3_0000000000000000001.parquet",
        )
    server = require_embedded_server()(log_dir=str(log_dir))
    server.stop()


def run_nonempty_binding_benchmark(
    source_log_dir: Path,
    work_dir: Path,
    output: Path,
    spec: DatasetSpec,
    *,
    decoy_counts: Sequence[int],
    cold_iterations: int,
    warm_iterations: int,
    warmup_iterations: int,
    concurrency_levels: Sequence[int],
    concurrency_queries_per_worker: int,
) -> dict[str, object]:
    """Measure referenced-only binding against nonempty unrelated namespaces."""
    if work_dir.exists():
        raise FileExistsError(f"benchmark work dir already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    manifest = build_segment_manifest(source_log_dir, spec.namespace)
    representative_names = {
        WorkloadName.RUN_DRILLDOWN_15M,
        WorkloadName.ALERT_EVALUATION,
    }
    workloads = tuple(workload for workload in build_workloads(spec) if workload.name in representative_names)
    source_digests = {workload.name.value: _query_digest(source_log_dir, workload) for workload in workloads}
    referenced_dir = work_dir / "referenced-only"
    _create_mirror(
        referenced_dir,
        spec.namespace,
        manifest,
        decoy_namespaces=0,
    )
    variants: dict[str, object] = {
        "referenced_only": {
            "namespace_count": 1,
            "telltale_segments": len(manifest),
            "unrelated_segments": 0,
            **_measure_shared_layout(
                referenced_dir,
                workloads,
                source_digests,
                replace(spec, decoy_namespaces=0),
                cold_iterations=cold_iterations,
                warm_iterations=warm_iterations,
                warmup_iterations=warmup_iterations,
                concurrency_levels=concurrency_levels,
                concurrency_queries_per_worker=concurrency_queries_per_worker,
            ),
        }
    }
    for decoy_count in decoy_counts:
        decoy_dir = work_dir / f"nonempty-decoys-{decoy_count}"
        _create_nonempty_decoy_store(
            decoy_dir,
            spec.namespace,
            manifest,
            decoy_namespaces=decoy_count,
        )
        variants[f"nonempty_decoys_{decoy_count}"] = {
            "namespace_count": decoy_count + 1,
            "telltale_segments": len(manifest),
            "unrelated_segments": decoy_count,
            "total_segments_across_namespaces": len(manifest) + decoy_count,
            **_measure_shared_layout(
                decoy_dir,
                workloads,
                source_digests,
                replace(spec, decoy_namespaces=decoy_count),
                cold_iterations=cold_iterations,
                warm_iterations=warm_iterations,
                warmup_iterations=warmup_iterations,
                concurrency_levels=concurrency_levels,
                concurrency_queries_per_worker=concurrency_queries_per_worker,
            ),
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_kind": "finelog_stats_service",
        "environment": runtime_environment(),
        "dataset": dataset_facts(spec),
        "source": {
            "telltale_segments": len(manifest),
            "telltale_rows": sum(entry.rows for entry in manifest),
            "decoy_segment_rows_each": manifest[0].rows,
            "decoy_segment_bytes_each": manifest[0].bytes,
        },
        "workloads": {workload.name.value: workload.sql for workload in workloads},
        "variants": variants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _measure_shared_layout(
    log_dir: Path,
    workloads: Sequence[Workload],
    source_digests: dict[str, str],
    spec: DatasetSpec,
    *,
    cold_iterations: int,
    warm_iterations: int,
    warmup_iterations: int,
    concurrency_levels: Sequence[int],
    concurrency_queries_per_worker: int,
) -> dict[str, object]:
    candidate_digests = {workload.name.value: _query_digest(log_dir, workload) for workload in workloads}
    for workload in workloads:
        if candidate_digests[workload.name.value] != source_digests[workload.name.value]:
            raise RuntimeError(f"candidate changed {workload.name.value} results")
    results: dict[str, object] = {}
    for workload in workloads:
        digest = candidate_digests[workload.name.value]
        results[workload.name.value] = {
            "result_digest": digest,
            "measurement": _measure_workload_layout(
                log_dir,
                workload,
                spec,
                cold_iterations=cold_iterations,
                warm_iterations=warm_iterations,
                warmup_iterations=warmup_iterations,
            ),
        }
    return {
        "workloads": results,
        "concurrency": concurrency_measurement(
            log_dir,
            workloads,
            concurrency_levels,
            concurrency_queries_per_worker,
        ),
    }


def _partition_selection_latency(
    entries: Sequence[PartitionedSegment],
    workload: Workload,
    iterations: int = 1_000,
) -> dict[str, float | int]:
    latencies: list[float] = []
    selected: tuple[SegmentManifestEntry, ...] = ()
    for _ in range(iterations):
        started = time.perf_counter()
        selected = select_partitioned_entries(entries, workload)
        latencies.append((time.perf_counter() - started) * 1_000)
    return {
        "iterations": iterations,
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "files_total": len(entries),
        "files_matched": len(selected),
        "row_groups_total": sum(item.entry.row_groups for item in entries),
        "row_groups_matched_upper_bound": sum(entry.row_groups for entry in selected),
        "bytes_total": sum(item.entry.bytes for item in entries),
        "bytes_in_matched_files": sum(entry.bytes for entry in selected),
    }


def run_layout_candidates(
    source_log_dir: Path,
    work_dir: Path,
    output: Path,
    spec: DatasetSpec,
    *,
    split_files_per_group: int,
    partition_buckets: int,
    cold_iterations: int,
    warm_iterations: int,
    warmup_iterations: int,
    concurrency_levels: Sequence[int],
    concurrency_queries_per_worker: int,
) -> dict[str, object]:
    """Build and measure split, clustered, and hidden-partition prototypes."""
    if work_dir.exists():
        raise FileExistsError(f"benchmark work dir already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    workloads = build_workloads(spec)
    source_digests = {workload.name.value: _query_digest(source_log_dir, workload) for workload in workloads}

    split_log_dir = work_dir / "split"
    split_preparation = prepare_split_layout(
        work_dir / "staging-split",
        split_log_dir,
        spec,
        files_per_group=split_files_per_group,
    )
    split_workloads = tuple(_retarget_split_workload(workload) for workload in workloads)
    split = _measure_shared_layout(
        split_log_dir,
        split_workloads,
        source_digests,
        spec,
        cold_iterations=cold_iterations,
        warm_iterations=warm_iterations,
        warmup_iterations=warmup_iterations,
        concurrency_levels=concurrency_levels,
        concurrency_queries_per_worker=concurrency_queries_per_worker,
    )
    split["preparation"] = split_preparation

    clustered_log_dir = work_dir / "clustered"
    clustered_preparation = prepare_clustered_layout(
        work_dir / "staging-clustered",
        clustered_log_dir,
        spec,
    )
    clustered = _measure_shared_layout(
        clustered_log_dir,
        workloads,
        source_digests,
        replace(spec, decoy_namespaces=0),
        cold_iterations=cold_iterations,
        warm_iterations=warm_iterations,
        warmup_iterations=warmup_iterations,
        concurrency_levels=concurrency_levels,
        concurrency_queries_per_worker=concurrency_queries_per_worker,
    )
    clustered["preparation"] = clustered_preparation

    partitioned_entries, partitioned_preparation = prepare_partitioned_layout(
        work_dir / "staging-partitioned",
        spec,
        buckets=partition_buckets,
    )
    partitioned_results: dict[str, object] = {}
    partitioned_dirs: dict[WorkloadName, Path] = {}
    for workload in workloads:
        selected = select_partitioned_entries(partitioned_entries, workload)
        if not selected:
            raise RuntimeError(f"partition transforms selected no files for {workload.name.value}")
        mirror_dir = work_dir / "partitioned" / workload.name.value
        _create_mirror(
            mirror_dir,
            spec.namespace,
            selected,
            decoy_namespaces=0,
        )
        partitioned_dirs[workload.name] = mirror_dir
        digest = _query_digest(mirror_dir, workload)
        if digest != source_digests[workload.name.value]:
            raise RuntimeError(f"partition pruning changed {workload.name.value} results")
        partitioned_results[workload.name.value] = {
            "selection": _partition_selection_latency(partitioned_entries, workload),
            "result_digest": digest,
            "measurement": _measure_workload_layout(
                mirror_dir,
                workload,
                replace(spec, decoy_namespaces=0),
                cold_iterations=cold_iterations,
                warm_iterations=warm_iterations,
                warmup_iterations=warmup_iterations,
            ),
        }
    alert_workload = next(workload for workload in workloads if workload.name == WorkloadName.ALERT_EVALUATION)
    partitioned: dict[str, object] = {
        "preparation": partitioned_preparation,
        "workloads": partitioned_results,
        "concurrency_alert_evaluation": concurrency_measurement(
            partitioned_dirs[WorkloadName.ALERT_EVALUATION],
            [alert_workload],
            concurrency_levels,
            concurrency_queries_per_worker,
        ),
        "partitions": [
            {
                "path": Path(item.entry.path).name,
                "day": item.day,
                "bucket": item.bucket,
                "min_ts_ms": item.entry.min_ts_ms,
                "max_ts_ms": item.entry.max_ts_ms,
                "rows": item.entry.rows,
                "bytes": item.entry.bytes,
            }
            for item in partitioned_entries
        ],
    }

    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_kind": "finelog_stats_service",
        "environment": runtime_environment(),
        "dataset": dataset_facts(spec),
        "source_log_dir": str(source_log_dir),
        "candidates": {
            "logical_signal_namespace_split": split,
            "multi_column_clustering": clustered,
            "hidden_day_hash_partitioning": partitioned,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _partition_counts(
    spec: DatasetSpec,
    buckets: int,
) -> tuple[Counter[int], Counter[tuple[int, int]]]:
    day_counts: Counter[int] = Counter()
    hash_counts: Counter[tuple[int, int]] = Counter()
    for batch in generate_batches(spec):
        timestamps = batch.column(batch.schema.get_field_index("ts")).cast(pa.int64()).to_pylist()
        groups = batch.column(batch.schema.get_field_index("source")).to_pylist()
        runs = batch.column(batch.schema.get_field_index("run")).to_pylist()
        workers = batch.column(batch.schema.get_field_index("worker")).to_pylist()
        clusters = batch.column(batch.schema.get_field_index("cluster")).to_pylist()
        for timestamp, group, run, worker, cluster in zip(
            timestamps,
            groups,
            runs,
            workers,
            clusters,
            strict=True,
        ):
            day = min(
                spec.duration_days - 1,
                (timestamp - spec.start_ms) // MILLISECONDS_PER_DAY,
            )
            bucket = _partition_bucket(group, run, worker, cluster, buckets)
            day_counts[day] += 1
            hash_counts[(day, bucket)] += 1
    return day_counts, hash_counts


def _projected_selection(
    partition_keys: Sequence[tuple[int, int]],
    workload: Workload,
    spec: DatasetSpec,
    buckets: int,
    *,
    iterations: int = 1_000,
) -> dict[str, float | int]:
    selected: list[tuple[int, int]] = []
    latencies: list[float] = []
    workload_bucket = None if workload.hash_bucket is None else workload.hash_bucket % buckets
    for _ in range(iterations):
        started = time.perf_counter()
        selected = [
            (day, bucket)
            for day, bucket in partition_keys
            if spec.start_ms + (day + 1) * MILLISECONDS_PER_DAY > workload.start_ms
            and spec.start_ms + day * MILLISECONDS_PER_DAY < workload.end_ms
            and (workload_bucket is None or bucket == workload_bucket)
        ]
        latencies.append((time.perf_counter() - started) * 1_000)
    return {
        "files_selected_before_segment_multiplicity": len(selected),
        "selection_p50_ms": statistics.median(latencies),
        "selection_p95_ms": percentile(latencies, 0.95),
    }


def run_partition_projection(
    source_log_dir: Path,
    output: Path,
    spec: DatasetSpec,
    bucket_counts: Sequence[int],
) -> dict[str, object]:
    """Compute exact nonempty day/hash counts without building partition files."""
    manifest = build_segment_manifest(source_log_dir, spec.namespace)
    source_bytes = sum(entry.bytes for entry in manifest)
    workloads = build_workloads(spec)
    variants: dict[str, object] = {}
    for buckets in bucket_counts:
        started = time.perf_counter()
        day_counts, hash_counts = _partition_counts(spec, buckets)
        row_counts = list(hash_counts.values())
        variants[f"day_hash_{buckets}"] = {
            "bucket_count": buckets,
            "theoretical_partitions_before_segment_multiplicity": spec.duration_days * buckets,
            "nonempty_partitions_before_segment_multiplicity": len(hash_counts),
            "empty_partitions": spec.duration_days * buckets - len(hash_counts),
            "rows_per_nonempty_partition": {
                "min": min(row_counts),
                "p50": statistics.median(row_counts),
                "p95": percentile(row_counts, 0.95),
                "max": max(row_counts),
            },
            "proportional_source_payload_bytes_per_nonempty_partition": source_bytes / len(hash_counts),
            "projection_seconds": time.perf_counter() - started,
            "workload_selection": {
                workload.name.value: _projected_selection(
                    tuple(hash_counts),
                    workload,
                    spec,
                    buckets,
                )
                for workload in workloads
            },
        }
    day_rows = list(day_counts.values())
    variants["day_only"] = {
        "theoretical_partitions_before_segment_multiplicity": spec.duration_days,
        "nonempty_partitions_before_segment_multiplicity": len(day_counts),
        "empty_partitions": spec.duration_days - len(day_counts),
        "rows_per_nonempty_partition": {
            "min": min(day_rows),
            "p50": statistics.median(day_rows),
            "p95": percentile(day_rows, 0.95),
            "max": max(day_rows),
        },
        "proportional_source_payload_bytes_per_nonempty_partition": source_bytes / len(day_counts),
        "workload_selection": {
            workload.name.value: {
                **_projected_selection(
                    tuple((day, 0) for day in day_counts),
                    replace(workload, hash_bucket=None),
                    spec,
                    1,
                )
            }
            for workload in workloads
        },
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_kind": "exact_local_selection_projection_not_query_latency",
        "environment": runtime_environment(),
        "dataset": dataset_facts(spec),
        "source": {
            "files": len(manifest),
            "rows": sum(entry.rows for entry in manifest),
            "parquet_bytes": source_bytes,
        },
        "assumptions": {
            "files_per_nonempty_partition": 1,
            "segment_multiplicity_excluded": True,
            "proportional_payload_excludes_parquet_footer_and_small_file_overhead": True,
        },
        "variants": variants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _rollup_sql(
    table: str,
    resolution: str,
    *,
    partial_states: bool,
    datafusion: bool,
    time_bounds: tuple[int, int] | None = None,
) -> str:
    interval = {"minute": "1 minute", "hour": "1 hour"}[resolution]
    bucket = f"date_bin(INTERVAL '{interval}', ts)" if datafusion else f"time_bucket(INTERVAL '{interval}', ts)"
    label = (
        (lambda name: f"json_get(labels, '{name}')")
        if datafusion
        else (lambda name: f"map_extract_value(labels, '{name}')")
    )
    if partial_states:
        states = """
       sum(sum_value) AS sum_value,
       min(min_value) AS min_value,
       max(max_value) AS max_value,
       sum(sample_count) AS sample_count
"""
        source = table
    else:
        states = """
       sum(value) AS sum_value,
       min(value) AS min_value,
       max(value) AS max_value,
       count(*) AS sample_count
"""
        source = table
    dimensions = f"""
       {bucket} AS bucket,
       name,
       source,
       run,
       job_id,
       worker,
       region,
       "cluster",
       {label("le")} AS le,
       {label("method")} AS method,
       {label("status")} AS status,
       {label("error_code")} AS error_code,
       {label("device")} AS device,
       {label("user")} AS user_name,
       {label("project")} AS project,
       {label("model")} AS model,
       {label("component")} AS component
"""
    if partial_states:
        dimensions = ",\n       ".join(ROLLUP_DIMENSIONS)
    where_clauses = [] if partial_states else ["source != 'errors'"]
    if time_bounds is not None:
        start_ms, end_ms = time_bounds
        where_clauses.append(f"ts >= to_timestamp_millis({start_ms}) " f"AND ts < to_timestamp_millis({end_ms})")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return f"""
SELECT {dimensions},
       {states.strip()}
FROM {source}
{where}
GROUP BY {", ".join(str(index) for index in range(1, len(ROLLUP_DIMENSIONS) + 1))}
""".strip()


def _normalize_rollup_table(table: pa.Table) -> pa.Table:
    bucket_index = table.schema.get_field_index("bucket")
    bucket = table.column(bucket_index).cast(pa.timestamp("ms"))
    return table.set_column(
        bucket_index,
        pa.field("bucket", pa.timestamp("ms")),
        bucket,
    )


def _materialize_rollup(
    source_log_dir: Path,
    namespace: str,
    resolution: str,
    spec: DatasetSpec,
) -> tuple[pa.Table, dict[str, float | int]]:
    rss_before = rss_bytes()
    server = require_embedded_server()(log_dir=str(source_log_dir))
    try:
        client = stats_client(server.address)
        with peak_rss_sampler() as rss_samples:
            started = time.perf_counter()
            tables = []
            for day in range(spec.duration_days):
                start_ms = spec.start_ms + day * MILLISECONDS_PER_DAY
                end_ms = spec.end_ms if day == spec.duration_days - 1 else start_ms + MILLISECONDS_PER_DAY
                tables.append(
                    query_table(
                        client,
                        _rollup_sql(
                            f'"{namespace}"',
                            resolution,
                            partial_states=False,
                            datafusion=True,
                            time_bounds=(start_ms, end_ms),
                        ),
                    )
                )
            table = pa.concat_tables(tables)
            elapsed = time.perf_counter() - started
    finally:
        server.stop()
    normalized = _normalize_rollup_table(table)
    return normalized, {
        "seconds": elapsed,
        "bounded_day_batches": spec.duration_days,
        "rss_peak_bytes": max(rss_samples),
        "rss_growth_peak_bytes": max(rss_samples) - rss_before,
        "output_rows": normalized.num_rows,
        "output_arrow_bytes": normalized.nbytes,
    }


def _aggregate_rollup_batch(batch: pa.RecordBatch, resolution: str) -> pa.Table:
    connection = duckdb.connect()
    try:
        connection.register("raw", batch)
        return _normalize_rollup_table(
            connection.sql(
                _rollup_sql(
                    "raw",
                    resolution,
                    partial_states=False,
                    datafusion=False,
                )
            ).to_arrow_table()
        )
    finally:
        connection.close()


def _merge_rollup_states(partials: Sequence[pa.Table], resolution: str) -> pa.Table:
    connection = duckdb.connect()
    try:
        connection.register("partials", pa.concat_tables(partials))
        return _normalize_rollup_table(
            connection.sql(
                _rollup_sql(
                    "partials",
                    resolution,
                    partial_states=True,
                    datafusion=False,
                )
            ).to_arrow_table()
        )
    finally:
        connection.close()


def _compaction_time_rollup(
    source_log_dir: Path,
    namespace: str,
    resolution: str,
) -> tuple[pa.Table, dict[str, float | int]]:
    paths = sorted((source_log_dir / namespace).glob("seg_L*.parquet"))
    rss_before = rss_bytes()
    with peak_rss_sampler() as rss_samples:
        started = time.perf_counter()
        partials = [
            _aggregate_rollup_batch(batch, resolution) for path in paths for batch in pq.ParquetFile(path).iter_batches()
        ]
        table = _merge_rollup_states(partials, resolution)
        elapsed = time.perf_counter() - started
    return table, {
        "seconds": elapsed,
        "rss_peak_bytes": max(rss_samples),
        "rss_growth_peak_bytes": max(rss_samples) - rss_before,
        "input_files": len(paths),
        "input_parquet_bytes": sum(path.stat().st_size for path in paths),
        "partial_rows": sum(partial.num_rows for partial in partials),
        "output_rows": table.num_rows,
        "output_arrow_bytes": table.nbytes,
    }


def _rollup_difference_count(materialized: pa.Table, compacted: pa.Table) -> int:
    connection = duckdb.connect()
    try:
        connection.register("materialized", materialized)
        connection.register("compacted", compacted)
        dimensions = ", ".join(f'"{column}"' for column in ROLLUP_DIMENSIONS)
        query = f"""
SELECT count(*) AS differences
FROM materialized
FULL OUTER JOIN compacted USING ({dimensions})
WHERE materialized.sample_count IS NULL
   OR compacted.sample_count IS NULL
   OR materialized.sample_count != compacted.sample_count
   OR abs(materialized.sum_value - compacted.sum_value) > 1e-9
   OR abs(materialized.min_value - compacted.min_value) > 1e-9
   OR abs(materialized.max_value - compacted.max_value) > 1e-9
"""
        row = connection.sql(query).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


def run_rollup_candidates(
    source_log_dir: Path,
    work_dir: Path,
    output: Path,
    spec: DatasetSpec,
    resolutions: Sequence[str],
) -> dict[str, object]:
    """Compare external materialization with a per-segment compaction proxy."""
    if work_dir.exists():
        raise FileExistsError(f"benchmark work dir already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    input_manifest = build_segment_manifest(source_log_dir, spec.namespace)
    results: dict[str, object] = {}
    for resolution in resolutions:
        materialized, materializer_metrics = _materialize_rollup(
            source_log_dir,
            spec.namespace,
            resolution,
            spec,
        )
        compacted, compaction_metrics = _compaction_time_rollup(
            source_log_dir,
            spec.namespace,
            resolution,
        )
        differences = _rollup_difference_count(materialized, compacted)
        if differences:
            raise RuntimeError(
                f"{resolution} rollup methods differ in {differences} " "fixed-dimension aggregate states"
            )
        materializer_path = work_dir / f"{resolution}-materializer.parquet"
        compaction_path = work_dir / f"{resolution}-compaction.parquet"
        pq.write_table(
            materialized,
            materializer_path,
            compression="zstd",
            row_group_size=16_384,
        )
        pq.write_table(
            compacted,
            compaction_path,
            compression="zstd",
            row_group_size=16_384,
        )
        materializer_metrics["output_parquet_bytes"] = materializer_path.stat().st_size
        compaction_metrics["output_parquet_bytes"] = compaction_path.stat().st_size
        results[resolution] = {
            "difference_count": differences,
            "materializer": materializer_metrics,
            "compaction_time_proxy": compaction_metrics,
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_kind": "local_rollup_proxy_not_query_latency",
        "environment": runtime_environment(),
        "dataset": dataset_facts(spec),
        "source_log_dir": str(source_log_dir),
        "input": {
            "files": len(input_manifest),
            "rows": sum(entry.rows for entry in input_manifest),
            "parquet_bytes": sum(entry.bytes for entry in input_manifest),
        },
        "candidate": {
            "name": "minute_hour_rollups",
            "state": "sum/min/max/count over the same fixed-dimension projection",
            "fixed_dimension_projection": {
                "included_dimensions": list(ROLLUP_DIMENSIONS),
                "omitted_raw_dimensions": [
                    "task_index",
                    "attempt",
                    "process_index",
                ],
                "omitted_labels": (
                    "all labels except le, method, status, error_code, device, " "user, project, model, and component"
                ),
            },
            "materializer_definition": "one day-bounded Finelog query per input day over the completed " "source store",
            "compaction_time_proxy_definition": (
                "aggregate each existing local Parquet segment independently, then merge "
                "associative states; this is not production compactor instrumentation"
            ),
            "materializer_scan_path": "finelog_stats_service",
            "compaction_time_scan_path": "duckdb_local_parquet",
            "unbounded_materializer_negative_result": {
                "resolution": "minute",
                "query_time_bounds": "none",
                "response_bytes": 123_825_416,
                "rpc_cap_bytes": 67_108_864,
                "result": "failed_response_exceeds_rpc_cap",
            },
            "resolutions": results,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
