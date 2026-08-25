# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Telltale-shaped data and bounded telemetry query corpus."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum

import pyarrow as pa

from finelog.rpc import finelog_stats_pb2 as stats_pb2

TELLTALE_NAMESPACE = "telltale"
DEFAULT_BATCH_ROWS = 25_000
DEFAULT_SEGMENTS = 24
DEFAULT_DURATION_DAYS = 14
DEFAULT_COLD_ITERATIONS = 3
DEFAULT_WARM_ITERATIONS = 7
DEFAULT_WARMUP_ITERATIONS = 1
START_TIME = datetime(2026, 1, 1, tzinfo=UTC)
MILLISECONDS_PER_DAY = 86_400_000
ENTITIES_PER_SCRAPE = 64
HOT_RUN_ENTITIES = 16


class SignalGroup(StrEnum):
    RUN = "run"
    WORKER = "worker"
    INFERENCE = "inference"
    FLEET = "fleet"
    NODE = "node"
    ERRORS = "errors"
    ALERTS = "alerts"
    ACCOUNTING = "accounting"


class WorkloadName(StrEnum):
    RUN_PREFIX_SEARCH = "run_prefix_search"
    RUN_DRILLDOWN_15M = "run_drilldown_15m"
    RUN_DRILLDOWN_24H = "run_drilldown_24h"
    RUN_DRILLDOWN_7D = "run_drilldown_7d"
    WORKER_OUTLIERS = "worker_outliers"
    VLLM_HISTOGRAM = "vllm_histogram"
    FLEET_FAILURES = "fleet_failures"
    NODE_DEVICE_HISTORY = "node_device_history"
    ERRORS = "errors"
    ALERT_EVALUATION = "alert_evaluation"
    ACCOUNTING = "accounting"


@dataclass(frozen=True)
class Signal:
    name: str
    group: SignalGroup
    kind: str


SIGNALS = (
    Signal("levanter_train_loss", SignalGroup.RUN, "gauge"),
    Signal("levanter_throughput", SignalGroup.RUN, "gauge"),
    Signal("levanter_step", SignalGroup.RUN, "gauge"),
    Signal("levanter_learning_rate", SignalGroup.RUN, "gauge"),
    Signal("levanter_gradient_norm", SignalGroup.RUN, "gauge"),
    Signal("levanter_parameter_norm", SignalGroup.RUN, "gauge"),
    Signal("levanter_optimizer_step_seconds", SignalGroup.RUN, "gauge"),
    Signal("levanter_checkpoint_seconds", SignalGroup.RUN, "gauge"),
    Signal("levanter_input_queue_depth", SignalGroup.RUN, "gauge"),
    Signal("levanter_tokens_total", SignalGroup.RUN, "counter"),
    Signal("process_resident_memory_bytes", SignalGroup.WORKER, "gauge"),
    Signal("process_cpu_seconds_total", SignalGroup.WORKER, "counter"),
    Signal("iris_worker_heartbeat", SignalGroup.WORKER, "gauge"),
    Signal("iris_rank_progress", SignalGroup.WORKER, "gauge"),
    Signal("vllm_time_to_first_token_seconds_bucket", SignalGroup.INFERENCE, "histogram"),
    Signal("vllm_time_per_output_token_seconds_bucket", SignalGroup.INFERENCE, "histogram"),
    Signal("vllm_num_requests_waiting", SignalGroup.INFERENCE, "gauge"),
    Signal("vllm_kv_cache_usage_ratio", SignalGroup.INFERENCE, "gauge"),
    Signal("iris_task_failures_total", SignalGroup.FLEET, "counter"),
    Signal("iris_worker_state", SignalGroup.FLEET, "gauge"),
    Signal("iris_device_temperature_celsius", SignalGroup.NODE, "gauge"),
    Signal("iris_device_memory_used_bytes", SignalGroup.NODE, "gauge"),
    Signal("iris_device_health", SignalGroup.NODE, "gauge"),
    Signal("iris_rpc_errors_total", SignalGroup.ERRORS, "counter"),
    Signal("iris_error_log_entries_total", SignalGroup.ERRORS, "counter"),
    Signal("iris_rpc_request_duration_seconds_bucket", SignalGroup.ALERTS, "histogram"),
    Signal("iris_rpc_requests_total", SignalGroup.ALERTS, "counter"),
    Signal("telemetry_progress_age_seconds", SignalGroup.ALERTS, "gauge"),
    Signal("telemetry_worker_heartbeat_age_seconds", SignalGroup.ALERTS, "gauge"),
    Signal("telemetry_vllm_queue_saturation", SignalGroup.ALERTS, "gauge"),
    Signal("telemetry_gpu_health", SignalGroup.ALERTS, "gauge"),
    Signal("telemetry_forwarding_gap", SignalGroup.ALERTS, "gauge"),
    Signal("telemetry_producer_heartbeat", SignalGroup.ALERTS, "gauge"),
    Signal("iris_accelerator_seconds_total", SignalGroup.ACCOUNTING, "counter"),
)
SERIES_COUNT = len(SIGNALS) * ENTITIES_PER_SCRAPE
RUN_METRIC_NAMES = tuple(signal.name for signal in SIGNALS if signal.group == SignalGroup.RUN)

REGIONS = ("us-central2", "us-east5", "us-east-02a", "us-east-08a")
CLUSTERS = ("marin", "cw-rno2a", "cw-us-east-02a", "cw-us-east-08a")
HISTOGRAM_BUCKETS = ("0.01", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "+Inf")
RPC_METHODS = ("CreateTask", "GetJob", "Heartbeat", "ListTasks")
RPC_STATUSES = ("ok", "cancelled", "internal", "unavailable")
ERROR_CODES = ("none", "preempted", "oom", "timeout", "infrastructure")


@dataclass(frozen=True)
class DatasetSpec:
    rows: int
    duration_days: int
    batch_rows: int
    segments: int = DEFAULT_SEGMENTS
    decoy_namespaces: int = 0
    distinct_run_ids: int = 100_000
    namespace: str = TELLTALE_NAMESPACE

    @property
    def start_ms(self) -> int:
        return int(START_TIME.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_days * MILLISECONDS_PER_DAY + 1


def dataset_facts(spec: DatasetSpec) -> dict[str, object]:
    """Return requested and achieved corpus dimensions for benchmark output."""
    scrapes = math.ceil(spec.rows / SERIES_COUNT)
    full_scrapes, partial_rows = divmod(spec.rows, SERIES_COUNT)
    partial_entities = math.ceil(partial_rows / len(SIGNALS))
    rotating_runs = full_scrapes * (ENTITIES_PER_SCRAPE - HOT_RUN_ENTITIES)
    rotating_runs += max(partial_entities - HOT_RUN_ENTITIES, 0)
    achieved_runs = int(spec.rows > 0) + min(rotating_runs, max(spec.distinct_run_ids - 8, 1))
    scrape_interval_minutes = spec.duration_days * 24 * 60 / (scrapes - 1) if scrapes > 1 else 0.0
    return {
        **asdict(spec),
        "signal_count": len(SIGNALS),
        "series_per_scrape": SERIES_COUNT,
        "achieved_scrapes": scrapes,
        "scrape_interval_minutes": scrape_interval_minutes,
        "achieved_distinct_run_ids": achieved_runs,
    }


@dataclass(frozen=True)
class Workload:
    name: WorkloadName
    sql: str
    group: SignalGroup = SignalGroup.RUN
    start_ms: int = 0
    end_ms: int = 0
    hash_bucket: int | None = None


def telltale_schema(*, layout_key: bool = False) -> stats_pb2.Schema:
    """Return the synthetic schema registered with the current Finelog API."""
    columns = [
        stats_pb2.Column(name="name", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        stats_pb2.Column(name="value", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=False),
        stats_pb2.Column(name="kind", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        stats_pb2.Column(name="ts", type=stats_pb2.COLUMN_TYPE_TIMESTAMP_MS, nullable=False),
        stats_pb2.Column(name="source", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        stats_pb2.Column(name="run", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True),
        stats_pb2.Column(name="job_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True),
        stats_pb2.Column(name="task_index", type=stats_pb2.COLUMN_TYPE_INT32, nullable=True),
        stats_pb2.Column(name="attempt", type=stats_pb2.COLUMN_TYPE_INT32, nullable=True),
        stats_pb2.Column(name="worker", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True),
        stats_pb2.Column(name="region", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True),
        stats_pb2.Column(name="process_index", type=stats_pb2.COLUMN_TYPE_INT32, nullable=True),
        stats_pb2.Column(name="labels", type=stats_pb2.COLUMN_TYPE_MAP, nullable=False),
        stats_pb2.Column(name="cluster", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
    ]
    if layout_key:
        columns.append(
            stats_pb2.Column(
                name="layout_key",
                type=stats_pb2.COLUMN_TYPE_STRING,
                nullable=False,
            )
        )
    return stats_pb2.Schema(
        columns=columns,
        key_column="layout_key" if layout_key else "name",
    )


def _row_labels(signal: Signal, index: int) -> list[tuple[str, str]]:
    return [
        (
            "le",
            HISTOGRAM_BUCKETS[index % len(HISTOGRAM_BUCKETS)] if signal.kind == "histogram" else "",
        ),
        ("method", RPC_METHODS[index % len(RPC_METHODS)]),
        ("status", RPC_STATUSES[index % len(RPC_STATUSES)]),
        ("error_code", ERROR_CODES[index % len(ERROR_CODES)]),
        ("device", f"device-{index % 8:02d}"),
        ("user", f"user-{index % 12:02d}"),
        ("project", f"project-{index % 6:02d}"),
        ("model", f"model-{index % 4:02d}"),
        ("component", f"component-{index % 6:02d}"),
        (
            "body",
            "CUDA OOM while allocating device buffer" if index % 5 == 0 else "retryable transport error",
        ),
    ]


def _run_index(sample: int, entity: int, distinct_run_ids: int) -> int:
    if entity < HOT_RUN_ENTITIES:
        return 7
    rotating_run_ids = max(distinct_run_ids - 8, 1)
    return 8 + ((sample * (ENTITIES_PER_SCRAPE - HOT_RUN_ENTITIES) + entity) % rotating_run_ids)


def generate_batch(spec: DatasetSpec, start: int, stop: int) -> pa.RecordBatch:
    """Build one deterministic Telltale-shaped Arrow batch for ``[start, stop)``."""
    duration_ms = spec.duration_days * MILLISECONDS_PER_DAY
    arrow_schema = pa.schema(
        [
            pa.field("name", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("kind", pa.string(), nullable=False),
            pa.field("ts", pa.timestamp("ms"), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("run", pa.string()),
            pa.field("job_id", pa.string()),
            pa.field("task_index", pa.int32()),
            pa.field("attempt", pa.int32()),
            pa.field("worker", pa.string()),
            pa.field("region", pa.string()),
            pa.field("process_index", pa.int32()),
            pa.field("labels", pa.map_(pa.string(), pa.string()), nullable=False),
            pa.field("cluster", pa.string(), nullable=False),
        ]
    )
    indices = range(start, stop)
    series_indices = [index % SERIES_COUNT for index in indices]
    entity_indices = [series_index // len(SIGNALS) for series_index in series_indices]
    sample_indices = [index // SERIES_COUNT for index in indices]
    signal_indices = [series_index % len(SIGNALS) for series_index in series_indices]
    signals = [SIGNALS[signal_index] for signal_index in signal_indices]
    run_indices = [
        _run_index(sample, entity, spec.distinct_run_ids)
        for sample, entity in zip(sample_indices, entity_indices, strict=True)
    ]
    num_samples = math.ceil(spec.rows / SERIES_COUNT)
    timestamps = [
        spec.start_ms + (sample_index * duration_ms // max(num_samples - 1, 1)) for sample_index in sample_indices
    ]
    arrays = [
        pa.array([signal.name for signal in signals], type=pa.string()),
        pa.array(
            [float(sample_indices[offset] + (signal_indices[offset] * 0.01)) for offset in range(stop - start)],
            type=pa.float64(),
        ),
        pa.array([signal.kind for signal in signals], type=pa.string()),
        pa.array(timestamps, type=pa.timestamp("ms")),
        pa.array([signal.group.value for signal in signals], type=pa.string()),
        pa.array([f"run-{run:06d}" for run in run_indices], type=pa.string()),
        pa.array([f"job-{run:06d}" for run in run_indices], type=pa.string()),
        pa.array([signal_index for signal_index in signal_indices], type=pa.int32()),
        pa.array([sample % 3 for sample in sample_indices], type=pa.int32()),
        pa.array(
            [
                f"worker-{entity:04d}" if entity < HOT_RUN_ENTITIES else f"worker-{run:06d}-{entity:02d}"
                for run, entity in zip(run_indices, entity_indices, strict=True)
            ],
            type=pa.string(),
        ),
        pa.array([REGIONS[entity % len(REGIONS)] for entity in entity_indices], type=pa.string()),
        pa.array([signal_index for signal_index in signal_indices], type=pa.int32()),
        pa.array(
            [_row_labels(signal, series) for signal, series in zip(signals, series_indices, strict=True)],
            type=pa.map_(pa.string(), pa.string()),
        ),
        pa.array([CLUSTERS[entity % len(CLUSTERS)] for entity in entity_indices], type=pa.string()),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=arrow_schema)


def generate_batches(spec: DatasetSpec) -> Iterator[pa.RecordBatch]:
    """Yield deterministic Telltale-shaped Arrow batches in timestamp order."""
    for start in range(0, spec.rows, spec.batch_rows):
        yield generate_batch(spec, start, min(start + spec.batch_rows, spec.rows))


def _timestamp_literal(epoch_ms: int) -> str:
    return f"to_timestamp_millis({epoch_ms})"


def build_workloads(spec: DatasetSpec) -> tuple[Workload, ...]:
    """Return the bounded query matrix required by the storage decision."""
    table = f'"{spec.namespace}"'
    end = spec.end_ms

    def lower(hours: float) -> int:
        return max(spec.start_ms, int(end - hours * 3_600_000))

    def bounded(hours: float) -> str:
        return f"ts >= {_timestamp_literal(lower(hours))} AND ts < {_timestamp_literal(end)}"

    metric_names = ", ".join(f"'{name}'" for name in RUN_METRIC_NAMES)

    def run_drilldown(name: WorkloadName, hours: float) -> Workload:
        return Workload(
            name,
            f"""
SELECT date_bin(INTERVAL '1 minute', ts) AS t, name, avg(value) AS value
FROM {table}
WHERE run = 'run-000007'
  AND name IN ({metric_names})
  AND {bounded(hours)}
GROUP BY 1, 2
ORDER BY 1, 2
""".strip(),
        )

    workloads = (
        run_drilldown(WorkloadName.RUN_DRILLDOWN_15M, 0.25),
        run_drilldown(WorkloadName.RUN_DRILLDOWN_24H, 24),
        run_drilldown(WorkloadName.RUN_DRILLDOWN_7D, 24 * 7),
        Workload(
            WorkloadName.WORKER_OUTLIERS,
            f"""
WITH latest AS (
  SELECT worker, name, value, ts,
         row_number() OVER (PARTITION BY worker, name ORDER BY ts DESC) AS row_num
  FROM {table}
  WHERE run = 'run-000007'
    AND name IN ('process_resident_memory_bytes', 'process_cpu_seconds_total')
    AND {bounded(1)}
)
SELECT worker,
       max(CASE WHEN name = 'process_resident_memory_bytes' THEN value END) AS rss_bytes,
       max(CASE WHEN name = 'process_cpu_seconds_total' THEN value END) AS cpu_seconds
FROM latest
WHERE row_num = 1
GROUP BY worker
ORDER BY rss_bytes DESC
LIMIT 20
""".strip(),
        ),
        Workload(
            WorkloadName.VLLM_HISTOGRAM,
            f"""
SELECT date_bin(INTERVAL '1 minute', ts) AS t,
       json_get(labels, 'le') AS le,
       max(value) - min(value) AS observations
FROM {table}
WHERE name = 'vllm_time_to_first_token_seconds_bucket'
  AND run = 'run-000007'
  AND {bounded(1)}
GROUP BY 1, 2
ORDER BY 1, 2
""".strip(),
        ),
        Workload(
            WorkloadName.FLEET_FAILURES,
            f"""
SELECT "cluster", region,
       max(CASE WHEN name = 'iris_worker_state' THEN value END) AS latest_state,
       sum(CASE WHEN name = 'iris_task_failures_total' THEN value ELSE 0 END) AS failures
FROM {table}
WHERE name IN ('iris_task_failures_total', 'iris_worker_state')
  AND {bounded(6)}
GROUP BY "cluster", region
ORDER BY failures DESC
""".strip(),
        ),
        Workload(
            WorkloadName.NODE_DEVICE_HISTORY,
            f"""
SELECT date_bin(INTERVAL '5 minutes', ts) AS t,
       run,
       name,
       json_get(labels, 'device') AS device,
       avg(value) AS value
FROM {table}
WHERE worker = 'worker-0007'
  AND name IN (
    'iris_device_temperature_celsius',
    'iris_device_memory_used_bytes',
    'iris_device_health'
  )
  AND {bounded(24)}
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2, 3, 4
""".strip(),
        ),
        Workload(
            WorkloadName.ERRORS,
            f"""
SELECT "cluster", name, json_get(labels, 'error_code') AS error_code,
       json_get(labels, 'body') AS body, sum(value) AS errors
FROM {table}
WHERE (
    name = 'iris_rpc_errors_total'
    OR (
      name = 'iris_error_log_entries_total'
      AND run = 'run-000007'
      AND json_get(labels, 'body') LIKE '%OOM%'
    )
  )
  AND {bounded(6)}
GROUP BY "cluster", name, error_code, body
ORDER BY errors DESC
""".strip(),
        ),
        Workload(
            WorkloadName.ALERT_EVALUATION,
            f"""
SELECT name, "cluster", region, value, ts, labels
FROM {table}
WHERE name IN (
    'telemetry_progress_age_seconds',
    'telemetry_worker_heartbeat_age_seconds',
    'telemetry_vllm_queue_saturation',
    'telemetry_gpu_health',
    'telemetry_forwarding_gap',
    'telemetry_producer_heartbeat'
  )
  AND {bounded(1)}
QUALIFY row_number() OVER (
  PARTITION BY name, "cluster", region, run, job_id, worker, process_index,
               json_get(labels, 'method'),
               json_get(labels, 'status'),
               json_get(labels, 'le')
  ORDER BY ts DESC
) = 1
""".strip(),
        ),
        Workload(
            WorkloadName.ACCOUNTING,
            f"""
SELECT "cluster",
       json_get(labels, 'user') AS user_name,
       json_get(labels, 'project') AS project,
       max(value) - min(value) AS accelerator_seconds
FROM {table}
WHERE name = 'iris_accelerator_seconds_total'
  AND {bounded(24 * 7)}
GROUP BY "cluster", user_name, project
ORDER BY accelerator_seconds DESC
""".strip(),
        ),
    )
    metadata = {
        WorkloadName.RUN_DRILLDOWN_15M: (SignalGroup.RUN, 0.25, 7),
        WorkloadName.RUN_DRILLDOWN_24H: (SignalGroup.RUN, 24, 7),
        WorkloadName.RUN_DRILLDOWN_7D: (SignalGroup.RUN, 24 * 7, 7),
        WorkloadName.WORKER_OUTLIERS: (SignalGroup.WORKER, 1, 7),
        WorkloadName.VLLM_HISTOGRAM: (SignalGroup.INFERENCE, 1, 7),
        WorkloadName.FLEET_FAILURES: (SignalGroup.FLEET, 6, None),
        WorkloadName.NODE_DEVICE_HISTORY: (SignalGroup.NODE, 24, 7),
        WorkloadName.ERRORS: (SignalGroup.ERRORS, 6, None),
        WorkloadName.ALERT_EVALUATION: (SignalGroup.ALERTS, 1, None),
        WorkloadName.ACCOUNTING: (SignalGroup.ACCOUNTING, 24 * 7, None),
    }
    return tuple(
        replace(
            workload,
            group=metadata[workload.name][0],
            start_ms=lower(metadata[workload.name][1]),
            end_ms=end,
            hash_bucket=metadata[workload.name][2],
        )
        for workload in workloads
    )
