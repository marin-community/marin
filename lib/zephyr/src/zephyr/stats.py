# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog stats schemas and counter-key constants for Zephyr pipelines.

Two namespaces are written:

- ``zephyr.stage`` — one row per stage at completion, emitted by the
  coordinator. Contains throughput and aggregated resource usage.
- ``zephyr.worker`` — one row per shard at START, each sample interval
  (RUNNING), and END, emitted directly by each runner.

Resource counters (cpu, memory, io) are sampled by runner background threads
via :func:`zephyr.runners._sample_process_stats` and stored with
:meth:`~zephyr.runners._InProcessWorkerContext.set_counter`. Counters are also
sent to the coordinator via heartbeats for aggregation into stage stats.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

ZEPHYR_STAGE_STATS_NAMESPACE = "zephyr.stage"
ZEPHYR_WORKER_STATS_NAMESPACE = "zephyr.worker"

ZEPHYR_STAGE_ITEM_COUNT_KEY = "zephyr/stage/{stage_name}/item_count"
"""Counter key template for items processed by a stage; format with ``stage_name``."""
ZEPHYR_STAGE_BYTES_PROCESSED_KEY = "zephyr/stage/{stage_name}/bytes_processed"
"""Counter key template for bytes processed by a stage; format with ``stage_name``."""

# Counter keys written by runner sampler threads using set_counter().
# Read by the coordinator from completed task snapshots for stage stat aggregation.
ZEPHYR_WORKER_CPU_MILLI_KEY = "zephyr/worker/{stage_name}/cpu_millipct"
"""cpu_percent * 1000, stored as an integer millipercent; format with ``stage_name``."""
ZEPHYR_WORKER_CPU_TIME_MS_KEY = "zephyr/worker/{stage_name}/cpu_time_ms"
"""Cumulative CPU time (user + system) since shard start, in milliseconds; format with ``stage_name``."""
ZEPHYR_WORKER_MEM_CURRENT_KEY = "zephyr/worker/{stage_name}/mem_current_bytes"
"""Current resident-set size of the runner process in bytes; format with ``stage_name``."""
ZEPHYR_WORKER_MEM_PEAK_KEY = "zephyr/worker/{stage_name}/mem_peak_bytes"
"""Monotonically increasing peak RSS seen across all sampling intervals; format with ``stage_name``."""
ZEPHYR_WORKER_IO_READ_KEY = "zephyr/worker/{stage_name}/io_read_bytes"
"""Cumulative bytes read by the runner process (best-effort; 0 if unavailable); format with ``stage_name``."""
ZEPHYR_WORKER_IO_WRITE_KEY = "zephyr/worker/{stage_name}/io_write_bytes"
"""Cumulative bytes written by the runner process (best-effort; 0 if unavailable); format with ``stage_name``."""


class ZephyrWorkerStatStatus(enum.StrEnum):
    """Lifecycle status of a ZephyrWorkerStat row."""

    START = "START"
    RUNNING = "RUNNING"
    END = "END"


@dataclass
class ZephyrStageStat:
    """One row per stage at completion, written by the coordinator."""

    key_column: ClassVar[str] = "execution_id"

    execution_id: str
    stage_name: str
    ts: datetime
    elapsed_s: float
    items: int
    bytes_processed: int
    item_rate: float
    byte_rate: float
    total_shards: int
    # Resource usage aggregated across all completed shard tasks for this stage.
    avg_cpu_pct: float
    total_cpu_s: float
    mem_avg_bytes: int
    mem_max_bytes: int
    mem_peak_bytes_max: int
    io_read_bytes_sum: int
    io_write_bytes_sum: int


@dataclass
class ZephyrWorkerStat:
    """One row per shard per sample interval, written by each runner."""

    key_column: ClassVar[str] = "execution_id"

    execution_id: str
    stage_name: str
    shard_idx: int
    status: str  # ZephyrWorkerStatStatus value
    ts: datetime
    items: int
    bytes_processed: int
    item_rate: float
    byte_rate: float
    cumulative_cpu_s: float
    avg_cpu_pct: float
    mem_current_bytes: int
    mem_peak_bytes: int
    io_read_bytes: int
    io_write_bytes: int
