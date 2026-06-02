# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pluggable shard execution strategies for ZephyrWorker.

A ``StageRunner`` is the strategy a worker uses to execute one ``ShardTask``.
Two implementations ship here:

* ``InlineRunner`` (default) — runs the stage in the worker actor's own
  process. Cheapest; appropriate for tests and pipelines whose user code is
  trusted not to corrupt the worker.
* ``SubprocessRunner`` — runs the stage in a fresh
  ``python -m zephyr.shard_subprocess`` subprocess. Each shard gets a clean
  Python heap, Arrow pool, and file descriptors; native crashes (SIGSEGV from
  Arrow/JAX, OOM kill) surface as deterministic ``returncode != 0`` task errors
  instead of bringing down the worker actor. Slower (~700ms of cold-import
  overhead per task).

Pick the runner pipeline-wide via ``ZephyrContext(stage_runner_factory=...)``.

The child-process entry point lives in ``zephyr.shard_subprocess``, kept
separate so the ``python -m`` target is not also imported during ``zephyr``
package initialization (which would trip a ``runpy`` re-execution warning).
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess as sp
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, TypeVar

import cloudpickle
import psutil
from finelog.client import LogClient, Table
from iris.client import get_iris_ctx
from rigging.filesystem import open_url

from zephyr.plan import Scatter, StageContext, run_stage
from zephyr.stage_io import (
    ShardTask,
    StageRunner,
    TaskResult,
    _shared_data_path,
    _stage_throughput,
    _write_stage_output,
)
from zephyr.stats import (
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    ZEPHYR_WORKER_CPU_MILLI_KEY,
    ZEPHYR_WORKER_CPU_TIME_MS_KEY,
    ZEPHYR_WORKER_IO_READ_KEY,
    ZEPHYR_WORKER_IO_WRITE_KEY,
    ZEPHYR_WORKER_MEM_CURRENT_KEY,
    ZEPHYR_WORKER_MEM_PEAK_KEY,
    ZEPHYR_WORKER_STATS_NAMESPACE,
    ZephyrWorkerStat,
    ZephyrWorkerStatStatus,
)
from zephyr.worker_context import CounterSnapshot, _worker_ctx_var

logger = logging.getLogger(__name__)


__all__ = ["InlineRunner", "StageRunner", "SubprocessRunner"]


SUBPROCESS_STATS_INTERVAL = 5.0
"""How often the subprocess child samples and emits its stats to finelog and
flushes its counters.

Matches the parent's heartbeat cadence so each beat reads at most one stale
snapshot before a fresh flush lands.
"""


# ---------------------------------------------------------------------------
# Shared worker context + stats wrapping (used by both runners)
# ---------------------------------------------------------------------------


class _InProcessWorkerContext:
    """WorkerContext satisfied by an in-memory counter dict.

    Used both by ``InlineRunner`` (in the worker actor process) and by the
    ``SubprocessRunner`` child (in the forked subprocess). Loads shared data
    lazily from the chunk store on first access and caches it for the rest
    of the task.
    """

    def __init__(self, chunk_prefix: str, execution_id: str, num_workers: int = 1):
        self._chunk_prefix = chunk_prefix
        self._execution_id = execution_id
        self._shared_data_cache: dict[str, Any] = {}
        self._counters: dict[str, int] = {}
        self._generation = 0
        self.num_workers = num_workers

    def get_shared(self, name: str) -> Any:
        if name not in self._shared_data_cache:
            path = _shared_data_path(self._chunk_prefix, self._execution_id, name)
            logger.info("Loading shared data '%s' from %s", name, path)
            with open_url(path, "rb") as f:
                self._shared_data_cache[name] = cloudpickle.loads(f.read())
        return self._shared_data_cache[name]

    def increment_counter(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def set_counter(self, name: str, value: int) -> None:
        self._counters[name] = value

    def get_counter_snapshot(self) -> CounterSnapshot:
        self._generation += 1
        return CounterSnapshot(counters=dict(self._counters), generation=self._generation)


_T = TypeVar("_T")


def _wrap_stage_stats(gen: Iterator[_T], stage_name: str, ctx: _InProcessWorkerContext) -> Iterator[_T]:
    """Yield items from ``gen`` while recording item count and byte size into ``ctx``."""
    item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)
    byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)
    for item in gen:
        ctx.increment_counter(item_key, 1)
        ctx.increment_counter(byte_key, sys.getsizeof(item))
        yield item


def _sample_process_stats(
    ctx: _InProcessWorkerContext, cpu_s_at_start: float, stage_name: str, proc: psutil.Process
) -> None:
    """Sample the current process's resource usage into ``ctx`` counters.

    Uses set_counter (not increment) because these are point-in-time metrics.
    Peak memory is tracked as a monotonically increasing max across calls.
    IO counters are cumulative totals from the OS; unavailable on some platforms.
    ``cpu_s_at_start`` is subtracted from cumulative CPU time to give per-shard delta.
    ``proc`` must be the same object across calls so cpu_percent() has a
    prior measurement to diff against; prime it once before the first sample.
    """
    mem = proc.memory_info()
    cpu_pct = proc.cpu_percent()
    cpu_times = proc.cpu_times()
    cpu_time_delta_ms = int(max(0, (cpu_times.user + cpu_times.system - cpu_s_at_start) * 1000))
    peak_key = ZEPHYR_WORKER_MEM_PEAK_KEY.format(stage_name=stage_name)
    ctx.set_counter(ZEPHYR_WORKER_CPU_MILLI_KEY.format(stage_name=stage_name), int(cpu_pct * 1000))
    ctx.set_counter(ZEPHYR_WORKER_CPU_TIME_MS_KEY.format(stage_name=stage_name), cpu_time_delta_ms)
    ctx.set_counter(ZEPHYR_WORKER_MEM_CURRENT_KEY.format(stage_name=stage_name), mem.rss)
    ctx.set_counter(peak_key, max(ctx._counters.get(peak_key, 0), mem.rss))
    with suppress(AttributeError, psutil.AccessDenied):
        io = proc.io_counters()
        ctx.set_counter(ZEPHYR_WORKER_IO_READ_KEY.format(stage_name=stage_name), io.read_bytes)
        ctx.set_counter(ZEPHYR_WORKER_IO_WRITE_KEY.format(stage_name=stage_name), io.write_bytes)


def _emit_runner_stat(
    log_table: Any,
    task: ShardTask,
    execution_id: str,
    status: ZephyrWorkerStatStatus,
    start_time: float,
    ctx: _InProcessWorkerContext,
    proc: psutil.Process,
    cpu_s_at_start: float,
) -> None:
    """Emit one ZephyrWorkerStat row to finelog from inside the runner."""
    elapsed = time.monotonic() - start_time
    counters = ctx._counters
    throughput = _stage_throughput(counters, task.stage_name, elapsed)
    try:
        current_cpu = proc.cpu_times()
        cumulative_cpu_s = max(0.0, (current_cpu.user + current_cpu.system) - cpu_s_at_start)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        cumulative_cpu_s = 0.0
    avg_cpu_pct = (cumulative_cpu_s / elapsed * 100) if elapsed > 0 else 0.0
    stat = ZephyrWorkerStat(
        execution_id=execution_id,
        stage_name=task.stage_name,
        shard_idx=task.shard_idx,
        status=status,
        ts=datetime.now(timezone.utc).replace(tzinfo=None),
        items=throughput.items if throughput else 0,
        bytes_processed=throughput.bytes_processed if throughput else 0,
        item_rate=throughput.item_rate if throughput else 0.0,
        byte_rate=throughput.byte_rate if throughput else 0.0,
        cumulative_cpu_s=cumulative_cpu_s,
        avg_cpu_pct=avg_cpu_pct,
        mem_current_bytes=counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY.format(stage_name=task.stage_name), 0),
        mem_peak_bytes=counters.get(ZEPHYR_WORKER_MEM_PEAK_KEY.format(stage_name=task.stage_name), 0),
        io_read_bytes=counters.get(ZEPHYR_WORKER_IO_READ_KEY.format(stage_name=task.stage_name), 0),
        io_write_bytes=counters.get(ZEPHYR_WORKER_IO_WRITE_KEY.format(stage_name=task.stage_name), 0),
    )
    try:
        log_table.write([stat])
    except Exception:
        logger.warning("Failed to write runner worker stat to finelog", exc_info=True)


def _periodic_sampler(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    interval: float,
    *,
    cpu_s_at_start: float = 0.0,
    log_table: Any = None,
    task: ShardTask | None = None,
    execution_id: str = "",
    start_time: float = 0.0,
    proc: psutil.Process | None = None,
) -> None:
    """Periodically sample process stats and optionally emit RUNNING rows to finelog."""
    while not stop_event.wait(timeout=interval):
        try:
            if task is not None and proc is not None:
                _sample_process_stats(ctx, cpu_s_at_start, task.stage_name, proc)

            if log_table is not None and task is not None and proc is not None:
                _emit_runner_stat(
                    log_table,
                    task,
                    execution_id,
                    ZephyrWorkerStatStatus.RUNNING,
                    start_time,
                    ctx,
                    proc,
                    cpu_s_at_start,
                )
        except Exception:
            logger.warning("Failed to sample/emit process stats", exc_info=True)


def _resolve_finelog_url() -> str | None:
    """Resolve the finelog endpoint URL via the Iris controller registry."""
    iris_ctx = get_iris_ctx()
    if iris_ctx is None or iris_ctx.client is None:
        return None
    try:
        return iris_ctx.client.resolve_endpoint("/system/log-server")
    except Exception:
        logger.warning("Could not resolve finelog endpoint for runner stats", exc_info=True)
        return None


def _make_log_client() -> LogClient | None:
    """Return a LogClient connected to the finelog service.

    Returns ``None`` when not running inside an Iris context. Callers treat
    ``None`` as a signal to skip stats emission silently.
    """
    url = _resolve_finelog_url()
    if url is None:
        return None
    try:
        return LogClient.connect(url)
    except Exception:
        logger.warning("Could not connect to finelog stats service; runner stats disabled", exc_info=True)
        return None


def _run_stage_with_ctx(
    task: ShardTask,
    chunk_prefix: str,
    execution_id: str,
    ctx: _InProcessWorkerContext,
) -> TaskResult:
    """Run one ShardTask inside the given worker context, writing stage output to disk.

    Shared between ``InlineRunner.execute`` and the subprocess child entry —
    once the right ctx is in place (and ``_worker_ctx_var`` is set), the
    actual per-shard work is identical.
    """
    stage_ctx = StageContext(
        shard=task.shard,
        shard_idx=task.shard_idx,
        total_shards=task.total_shards,
        aux_shards=task.aux_shards,
    )
    output_stage_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task.stage_name).strip("-")
    stage_dir = f"{chunk_prefix}/{execution_id}/{output_stage_name}"
    external_sort_dir = f"{stage_dir}-external-sort/shard-{task.shard_idx:04d}"
    scatter_op = next((op for op in task.operations if isinstance(op, Scatter)), None)
    return _write_stage_output(
        _wrap_stage_stats(
            run_stage(stage_ctx, task.operations, external_sort_dir=external_sort_dir),
            task.stage_name,
            ctx,
        ),
        source_shard=task.shard_idx,
        stage_dir=stage_dir,
        shard_idx=task.shard_idx,
        scatter_op=scatter_op,
        total_shards=task.total_shards,
    )


# ---------------------------------------------------------------------------
# InlineRunner — default
# ---------------------------------------------------------------------------


class InlineRunner:
    """Run shard work in the worker actor's own process.

    Cheap and observable (counters live in shared memory; the heartbeat just
    reads them) but does not isolate native crashes or per-shard memory
    growth. Default for ``ZephyrContext`` because most pipelines are fine
    here, and tests run dramatically faster than under ``SubprocessRunner``.
    """

    def __init__(self, num_workers: int = 1) -> None:
        self._num_workers = num_workers
        self._ctx: _InProcessWorkerContext | None = None
        self._log_client: LogClient | None = None
        self._worker_stats_table: Table | None = None
        self._log_client_initialized: bool = False

    def _get_worker_stats_table(self) -> Any:
        if not self._log_client_initialized:
            self._log_client_initialized = True
            self._log_client = _make_log_client()
            if self._log_client is not None:
                try:
                    self._worker_stats_table = self._log_client.get_table(
                        ZEPHYR_WORKER_STATS_NAMESPACE, ZephyrWorkerStat
                    )
                except Exception:
                    logger.warning(
                        "Could not initialize finelog worker stats table; worker stats disabled", exc_info=True
                    )
                    self._log_client = None
        return self._worker_stats_table

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, int]]:
        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, num_workers=self._num_workers)
        self._ctx = ctx
        worker_token = _worker_ctx_var.set(ctx)
        stop_event = threading.Event()
        log_table = self._get_worker_stats_table()
        proc = psutil.Process()
        start_time = time.monotonic()
        cpu_times_at_start = proc.cpu_times()
        cpu_s_at_start = cpu_times_at_start.user + cpu_times_at_start.system
        proc.cpu_percent()  # prime so subsequent calls have a baseline
        if log_table is not None:
            _emit_runner_stat(
                log_table, task, execution_id, ZephyrWorkerStatStatus.START, start_time, ctx, proc, cpu_s_at_start
            )
        sampler = threading.Thread(
            target=_periodic_sampler,
            kwargs={
                "stop_event": stop_event,
                "ctx": ctx,
                "interval": SUBPROCESS_STATS_INTERVAL,
                "cpu_s_at_start": cpu_s_at_start,
                "log_table": log_table,
                "task": task,
                "execution_id": execution_id,
                "start_time": start_time,
                "proc": proc,
            },
            daemon=True,
            name="zephyr-inline-stats-sampler",
        )
        sampler.start()
        try:
            result = _run_stage_with_ctx(task, chunk_prefix, execution_id, ctx)
        finally:
            stop_event.set()
            sampler.join(timeout=2.0)
            _sample_process_stats(ctx, cpu_s_at_start, task.stage_name, proc)
            if log_table is not None:
                _emit_runner_stat(
                    log_table, task, execution_id, ZephyrWorkerStatStatus.END, start_time, ctx, proc, cpu_s_at_start
                )
            _worker_ctx_var.reset(worker_token)
            self._ctx = None
        return result, dict(ctx._counters)

    def live_counters(self) -> dict[str, int]:
        ctx = self._ctx
        return dict(ctx._counters) if ctx is not None else {}


# ---------------------------------------------------------------------------
# SubprocessRunner — opt-in isolation
# ---------------------------------------------------------------------------


class SubprocessRunner:
    """Run each shard in a fresh ``python -m zephyr.shard_subprocess`` subprocess.

    Provides full memory and crash isolation: native crashes (Arrow/JAX
    SIGSEGV, OOM) terminate only the child and surface as deterministic
    ``returncode != 0`` task errors. Costs ~700ms per task in cold Python
    imports plus pickle round-trip; reserve for stages with leak-prone or
    crash-prone user code.

    Args:
        num_workers: Total number of concurrent subprocess workers sharing this
            actor's RAM. Passed to child processes via ``ZEPHYR_NUM_WORKERS_PER_ACTOR``
            so each child scales its scatter-write buffer budget proportionally.
    """

    def __init__(self, num_workers: int = 1) -> None:
        self._num_workers = num_workers
        self._counter_file: str | None = None

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, int]]:
        finelog_url = _resolve_finelog_url()  # Requires Iris context, so called here and passed to subprocess
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cloudpickle.dump((task, chunk_prefix, execution_id, finelog_url), f)
            task_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result_file = f.name
        counter_file = f"{result_file}.counters"
        self._counter_file = counter_file

        try:
            # ``-u`` keeps the child's stdout/stderr unbuffered so any
            # faulthandler traceback reaches the parent's log before the
            # process dies.
            proc = sp.run(
                [sys.executable, "-u", "-m", "zephyr.shard_subprocess", task_file, result_file, str(self._num_workers)],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

            if proc.returncode != 0:
                # Linux OOM-killer sends SIGKILL → returncode == -9. Distinguish
                # so callers/retries can react to memory pressure specifically.
                if proc.returncode == -signal.SIGKILL:
                    raise MemoryError(
                        f"Subprocess for shard {task.shard_idx} was killed by SIGKILL "
                        f"(returncode {proc.returncode}); most likely OOM-killed by the kernel."
                    )
                raise RuntimeError(
                    f"Subprocess for shard {task.shard_idx} exited with code {proc.returncode}; "
                    "see worker stderr above for the faulthandler traceback."
                )

            with open(result_file, "rb") as f:
                result_or_error, child_counters = cloudpickle.load(f)

            # Clear counter pointer BEFORE returning so a heartbeat racing
            # this and ``report_result`` reads {} rather than re-shipping
            # values the caller is about to send as final.
            self._counter_file = None

            if isinstance(result_or_error, Exception):
                raise result_or_error

            return result_or_error, dict(child_counters)
        finally:
            self._counter_file = None
            for p in (task_file, result_file, counter_file, f"{counter_file}.tmp"):
                with suppress(FileNotFoundError):
                    os.unlink(p)

    def live_counters(self) -> dict[str, int]:
        cf = self._counter_file
        if cf is None:
            return {}
        try:
            with open(cf, "rb") as f:
                return cloudpickle.load(f)
        except (FileNotFoundError, EOFError):
            # Race against atomic rename, or task already cleaned up its file.
            return {}
        except Exception:
            logger.warning("Failed to read counter file %s", cf, exc_info=True)
            return {}
