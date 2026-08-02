# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog stats schemas and counter-key constants for Zephyr pipelines.

Two namespaces are written:

- ``zephyr.stage`` — one row per stage at completion, emitted by the
  coordinator. Contains throughput and aggregated resource usage.
- ``zephyr.worker`` — one row per shard at START, each sample interval
  (RUNNING), and END, emitted directly by each runner.

Resource counters (cpu, memory) are sampled by runner background threads
via :func:`zephyr.runners._sample_process_stats` and stored with
:meth:`~zephyr.runners._InProcessWorkerContext.set_counter`. Counters are also
sent to the coordinator via heartbeats for aggregation into stage stats.
"""

import enum
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

from finelog.client import LogClient, Table
from iris.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from rigging.timing import RateLimiter

logger = logging.getLogger(__name__)

ZEPHYR_STAGE_STATS_NAMESPACE = "zephyr.stage"
ZEPHYR_WORKER_STATS_NAMESPACE = "zephyr.worker"
MAX_METRIC_STAGE_SERIES = 256

ZEPHYR_STAGE_ITEM_COUNT_KEY = "zephyr/item_count"
"""Counter key for items processed"""
ZEPHYR_STAGE_BYTES_PROCESSED_KEY = "zephyr/bytes_processed"
"""Counter key for bytes processed"""

# Counter keys written by runner sampler threads using set_counter().
# Read by the coordinator from completed task snapshots for stage stat aggregation.
ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY = "zephyr/worker/cpu_pct_current"
"""Current CPU percentage"""
ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY = "zephyr/worker/cpu_pct_average"
"""Average CPU percentage"""
ZEPHYR_WORKER_CPU_TIME_KEY = "zephyr/worker/cpu_time"
"""Cumulative CPU time (user + system) since shard start, in seconds"""
ZEPHYR_WORKER_MEM_CURRENT_KEY = "zephyr/worker/mem_current_bytes"
"""Current resident-set size of the runner process in bytes"""
ZEPHYR_WORKER_MEM_AVERAGE_KEY = "zephyr/worker/mem_average_bytes"
"""Average resident-set size of the runner process in bytes"""
ZEPHYR_WORKER_MEM_PEAK_KEY = "zephyr/worker/mem_peak_bytes"
"""Monotonically increasing peak RSS seen across all sampling intervals"""


def _push_iris_task_status(
    rate_limiter: RateLimiter,
    build_md: Callable[[], tuple[str, str]],
) -> None:
    """Send status text to the active Iris task."""
    iris_ctx = get_iris_ctx()
    if iris_ctx is None or iris_ctx.client is None:
        return
    job_info = get_job_info()
    if job_info is None or not rate_limiter.should_run():
        return
    detail_md, summary_md = build_md()
    try:
        iris_ctx.client.report_task_status_text(job_info.task_id, job_info.attempt_id, detail_md, summary_md)
    except Exception:
        logger.warning("Failed to report task status text to Iris controller", exc_info=True)


def per_second(total: float, elapsed: float) -> float:
    """Rate of ``total`` over ``elapsed`` seconds, or 0.0 when no time has passed."""
    return total / elapsed if elapsed > 0 else 0.0


class ZephyrWorkerStatStatus(enum.StrEnum):
    """Lifecycle status of a ZephyrWorkerStat or ZephyrStageStat row."""

    START = "START"
    RUNNING = "RUNNING"
    END = "END"
    FAILED = "FAILED"


@dataclass
class ZephyrStageStat:
    """One row per stage at completion (or failure), written by the coordinator."""

    key_column: ClassVar[str] = "execution_id"

    execution_id: str
    stage_name: str
    status: str  # ZephyrWorkerStatStatus value; str because LogClient cannot serialize StrEnum
    ts: datetime
    elapsed: float  # seconds
    items: int
    bytes_processed: int
    item_rate: float
    byte_rate: float
    total_shards: int
    # Resource usage aggregated across all completed shard tasks for this stage.
    cpu_pct_avg: float
    cpu_time_total: float  # seconds
    mem_bytes_avg: int
    mem_peak_bytes_max: int


@dataclass
class ZephyrWorkerStat:
    """One row per shard per sample interval, written by each runner."""

    key_column: ClassVar[str] = "execution_id"

    execution_id: str
    stage_name: str
    shard_idx: int
    status: str  # ZephyrWorkerStatStatus value; str because LogClient cannot serialize StrEnum
    ts: datetime
    items: int
    bytes_processed: int
    item_rate: float
    byte_rate: float
    cpu_time_total: float  # seconds
    cpu_current_pct: float
    cpu_avg_pct: float
    mem_current_bytes: int
    mem_avg_bytes: int
    mem_peak_bytes: int


@dataclass(frozen=True)
class PipelineMetricPoint:
    """One time-series point for the coordinator dashboard."""

    timestamp_ms: int
    stage: str
    item_rate: float
    byte_rate: float
    cpu_cores: float
    memory_bytes: int
    active_shards: int


@dataclass(frozen=True)
class PipelineMetricsResult:
    """Dashboard points and a user-visible data-source warning."""

    points: tuple[PipelineMetricPoint, ...]
    warning: str = ""


class StatsWriter:
    """Manages finelog connections and emits Zephyr stat rows.

    Call ``connect()`` to get a live instance; pass a pre-resolved URL when
    an Iris context is not available (e.g. in a subprocess).  All emit
    methods are no-ops when the log client is unavailable.
    """

    def __init__(self, log_client: LogClient | None) -> None:
        self._log_client = log_client
        self._stage_table: Table | None = None
        self._worker_table: Table | None = None
        if log_client is not None:
            with suppress(Exception):
                self._stage_table = log_client.get_table(ZEPHYR_STAGE_STATS_NAMESPACE, ZephyrStageStat)
            with suppress(Exception):
                self._worker_table = log_client.get_table(ZEPHYR_WORKER_STATS_NAMESPACE, ZephyrWorkerStat)

    @classmethod
    def connect(cls, url: str | None = None) -> "StatsWriter":
        """Connect to finelog; resolves the URL via Iris if not provided.

        Returns a no-op instance if the URL cannot be determined or the
        connection fails.
        """
        resolved = url or cls.resolve_url()
        if resolved is None:
            return cls(None)
        try:
            return cls(LogClient.connect(resolved))
        except Exception:
            logger.warning("Could not connect to finelog at %s; stats disabled", resolved, exc_info=True)
            return cls(None)

    @staticmethod
    def resolve_url() -> str | None:
        """Resolve the finelog endpoint URL via the Iris controller registry."""
        iris_ctx = get_iris_ctx()
        if iris_ctx is None or iris_ctx.client is None:
            return None
        try:
            return iris_ctx.client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
        except Exception:
            logger.warning("Could not resolve finelog endpoint", exc_info=True)
            return None

    def emit_stage_stat(
        self,
        stage_counters: dict[str, int | float],
        stage_name: str,
        execution_id: str,
        elapsed: float,
        total_shards: int,
        status: ZephyrWorkerStatStatus,
    ) -> None:
        """Build and emit a ZephyrStageStat row from aggregated stage counters.

        ``stage_counters`` is the coordinator's reduced view of all shard
        counters for this stage (throughput plus resource usage). Pass
        ``status=ZephyrWorkerStatStatus.FAILED`` when emitting for a failed stage.
        """
        if self._stage_table is None:
            return

        total_items = stage_counters.get(ZEPHYR_STAGE_ITEM_COUNT_KEY, 0)
        total_bytes = stage_counters.get(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, 0)
        item_rate = per_second(total_items, elapsed)
        byte_rate = per_second(total_bytes, elapsed)

        stat = ZephyrStageStat(
            execution_id=execution_id,
            stage_name=stage_name,
            status=status,
            ts=datetime.now(UTC).replace(tzinfo=None),
            elapsed=elapsed,
            items=total_items,
            bytes_processed=total_bytes,
            item_rate=item_rate,
            byte_rate=byte_rate,
            total_shards=total_shards,
            cpu_pct_avg=stage_counters.get(ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, 0),
            cpu_time_total=stage_counters.get(ZEPHYR_WORKER_CPU_TIME_KEY, 0),
            mem_bytes_avg=stage_counters.get(ZEPHYR_WORKER_MEM_AVERAGE_KEY, 0),
            mem_peak_bytes_max=stage_counters.get(ZEPHYR_WORKER_MEM_PEAK_KEY, 0),
        )
        try:
            self._stage_table.write([stat])
        except Exception:
            logger.warning("Failed to write stage stat to finelog", exc_info=True)

    def emit_worker_stat(
        self,
        stage_name: str,
        shard_idx: int,
        execution_id: str,
        status: ZephyrWorkerStatStatus,
        start_time: float,
        counters: dict[str, int | float],
    ) -> None:
        """Build and emit a ZephyrWorkerStat row from the runner's counter dict."""
        if self._worker_table is None:
            return
        elapsed = time.monotonic() - start_time
        items = counters.get(ZEPHYR_STAGE_ITEM_COUNT_KEY, 0)
        bytes_processed = counters.get(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, 0)
        item_rate = per_second(items, elapsed)
        byte_rate = per_second(bytes_processed, elapsed)
        stat = ZephyrWorkerStat(
            execution_id=execution_id,
            stage_name=stage_name,
            shard_idx=shard_idx,
            status=status,
            ts=datetime.now(UTC).replace(tzinfo=None),
            items=items,
            bytes_processed=bytes_processed,
            item_rate=item_rate,
            byte_rate=byte_rate,
            cpu_time_total=counters.get(ZEPHYR_WORKER_CPU_TIME_KEY, 0),
            cpu_current_pct=counters.get(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, 0),
            cpu_avg_pct=counters.get(ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, 0),
            mem_current_bytes=counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY, 0),
            mem_avg_bytes=counters.get(ZEPHYR_WORKER_MEM_AVERAGE_KEY, 0),
            mem_peak_bytes=counters.get(ZEPHYR_WORKER_MEM_PEAK_KEY, 0),
        )
        try:
            self._worker_table.write([stat])
        except Exception:
            logger.warning("Failed to write worker stat to finelog", exc_info=True)

    def query_pipeline_metrics(self, execution_id: str, max_points: int) -> PipelineMetricsResult:
        """Return bounded 15-second pipeline metrics from worker samples."""
        if self._log_client is None:
            return PipelineMetricsResult((), "Finelog is not available for this coordinator.")

        escaped_execution_id = execution_id.replace("'", "''")
        sql = f"""
WITH recent_bins AS (
  SELECT DISTINCT date_bin(INTERVAL '15 seconds', ts) AS time_bin
  FROM "{ZEPHYR_WORKER_STATS_NAMESPACE}"
  WHERE execution_id = '{escaped_execution_id}'
    AND status = '{ZephyrWorkerStatStatus.RUNNING}'
  ORDER BY 1 DESC
  LIMIT {max_points}
), per_shard AS (
  SELECT date_bin(INTERVAL '15 seconds', samples.ts) AS time_bin,
         samples.stage_name,
         samples.shard_idx,
         avg(samples.item_rate) AS item_rate,
         avg(samples.byte_rate) AS byte_rate,
         avg(samples.cpu_current_pct) / 100.0 AS cpu_cores,
         avg(samples.mem_current_bytes) AS memory_bytes
  FROM "{ZEPHYR_WORKER_STATS_NAMESPACE}" AS samples
  INNER JOIN recent_bins
    ON date_bin(INTERVAL '15 seconds', samples.ts) = recent_bins.time_bin
  WHERE samples.execution_id = '{escaped_execution_id}'
    AND samples.status = '{ZephyrWorkerStatStatus.RUNNING}'
  GROUP BY 1, 2, 3
)
SELECT time_bin,
       stage_name,
       sum(item_rate) AS item_rate,
       sum(byte_rate) AS byte_rate,
       sum(cpu_cores) AS cpu_cores,
       sum(memory_bytes) AS memory_bytes,
       count(*) AS active_shards
FROM per_shard
GROUP BY 1, 2
ORDER BY 1 DESC, 2
""".strip()
        try:
            rows = self._log_client.query(sql, max_rows=max_points * MAX_METRIC_STAGE_SERIES).to_pylist()
        except Exception:
            logger.warning("Failed to query Zephyr pipeline metrics", exc_info=True)
            return PipelineMetricsResult((), "Finelog metrics are temporarily unavailable.")

        points = [
            PipelineMetricPoint(
                timestamp_ms=_timestamp_ms(row["time_bin"]),
                stage=str(row["stage_name"]),
                item_rate=float(row["item_rate"] or 0),
                byte_rate=float(row["byte_rate"] or 0),
                cpu_cores=float(row["cpu_cores"] or 0),
                memory_bytes=int(row["memory_bytes"] or 0),
                active_shards=int(row["active_shards"] or 0),
            )
            for row in rows
        ]
        return PipelineMetricsResult(_complete_metric_bins(points, max_points))

    def close(self) -> None:
        if self._log_client is not None:
            with suppress(Exception):
                self._log_client.close()


def _timestamp_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def _complete_metric_bins(points: list[PipelineMetricPoint], max_points: int) -> tuple[PipelineMetricPoint, ...]:
    by_timestamp: dict[int, list[PipelineMetricPoint]] = {}
    for point in points:
        by_timestamp.setdefault(point.timestamp_ms, []).append(point)

    selected: list[PipelineMetricPoint] = []
    for timestamp in sorted(by_timestamp, reverse=True):
        time_bin = by_timestamp[timestamp]
        if selected and len(selected) + len(time_bin) > max_points:
            break
        selected.extend(time_bin)
    return tuple(sorted(selected, key=lambda point: (point.timestamp_ms, point.stage)))
