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
from typing import Any, TypeVar

import cloudpickle
import psutil
from rigging.filesystem import StoragePath

from zephyr import counters
from zephyr.plan import Scatter, StageContext, run_stage
from zephyr.stage_io import (
    ShardTask,
    StageRunner,
    TaskResult,
    _shared_data_path,
    _write_stage_output,
)
from zephyr.stats import (
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY,
    ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY,
    ZEPHYR_WORKER_CPU_TIME_KEY,
    ZEPHYR_WORKER_MEM_AVERAGE_KEY,
    ZEPHYR_WORKER_MEM_CURRENT_KEY,
    ZEPHYR_WORKER_MEM_PEAK_KEY,
    StatsWriter,
    ZephyrWorkerStatStatus,
)
from zephyr.worker_context import Aggregation, CounterEntry, CounterSnapshot, _worker_ctx_var

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

    def __init__(self, chunk_prefix: str, execution_id: str, stage_name: str, task_memory_bytes: int = 0):
        self._chunk_prefix = chunk_prefix
        self._execution_id = execution_id
        self._stage_name = stage_name
        self._shared_data_cache: dict[str, Any] = {}
        self._counters: dict[str, CounterEntry] = {}
        self._generation = 0
        self.task_memory_bytes = task_memory_bytes

    def get_shared(self, name: str) -> Any:
        if name not in self._shared_data_cache:
            path = _shared_data_path(self._chunk_prefix, self._execution_id, name)
            logger.info("Loading shared data '%s' from %s", name, path)
            self._shared_data_cache[name] = cloudpickle.loads(StoragePath(path).read_bytes())
        return self._shared_data_cache[name]

    def set_counter(self, name: str, value: int | float, stage: str | None = None) -> None:
        if name in self._counters:
            entry = self._counters[name]
            entry.value = value
            entry.count = 1
            entry.stage = stage
        else:
            self._counters[name] = CounterEntry(value, stage=stage)

    def update_counter(self, name: str, value: int | float, stage: str | None = None) -> None:
        entry = self._counters.get(name)
        if entry is None or entry.count == 0:
            # First real observation: initialise the value regardless of aggregation.
            if entry is None:
                self._counters[name] = CounterEntry(value, stage=stage)
            else:
                entry.value = value
                entry.stage = stage
                entry.count = 1
            return
        entry.merge(CounterEntry(value, entry.aggregation, stage, count=1))

    def set_aggregation(self, name: str, agg: Aggregation) -> None:
        if name in self._counters:
            self._counters[name].aggregation = agg
        else:
            # count=0 marks the entry as uninitialised so update_counter sets the
            # first value directly rather than applying MIN/MAX/AVERAGE to 0.
            self._counters[name] = CounterEntry(0, aggregation=agg, count=0)

    def current_stage_name(self) -> str:
        return self._stage_name

    def get_counters(self, stage: str | None = None) -> dict[str, int | float]:
        """Flat view of counter values, for use by stats emission code."""
        return {k: e.value for k, e in self._counters.items() if stage is None or e.stage == stage}

    def get_counter_snapshot(self) -> CounterSnapshot:
        self._generation += 1
        return CounterSnapshot(
            counters={k: CounterEntry(e.value, e.aggregation, e.stage, e.count) for k, e in self._counters.items()},
            generation=self._generation,
        )


_T = TypeVar("_T")


def _wrap_stage_stats(gen: Iterator[_T]) -> Iterator[_T]:
    """Yield items from ``gen`` while recording item count and byte size into the current stage's counters."""
    stage_counters = counters.current_stage()
    for item in gen:
        stage_counters.update_counter(ZEPHYR_STAGE_ITEM_COUNT_KEY, 1)
        stage_counters.update_counter(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, sys.getsizeof(item))
        yield item


def _sample_process_stats(cpu_s_at_start: float, proc: psutil.Process) -> None:
    """Sample the current process's resource usage into the current stage's counters.

    Uses set_counter (not increment) because these are point-in-time metrics.
    Peak memory is tracked as a monotonically increasing max across calls.
    ``cpu_s_at_start`` is subtracted from cumulative CPU time to give per-shard delta.
    ``proc`` must be the same object across calls so cpu_percent() has a
    prior measurement to diff against; prime it once before the first sample.
    """
    rss = proc.memory_info().rss
    cpu_times = proc.cpu_times()
    cpu_pct = proc.cpu_percent()
    stage_counters = counters.current_stage()
    stage_counters.set_counter(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, cpu_pct)
    stage_counters.update_counter(ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, cpu_pct)
    stage_counters.set_counter(ZEPHYR_WORKER_CPU_TIME_KEY, cpu_times.user + cpu_times.system - cpu_s_at_start)
    stage_counters.set_counter(ZEPHYR_WORKER_MEM_CURRENT_KEY, rss)
    stage_counters.update_counter(ZEPHYR_WORKER_MEM_AVERAGE_KEY, rss)
    stage_counters.update_counter(ZEPHYR_WORKER_MEM_PEAK_KEY, rss)


def _set_counter_aggregations() -> None:
    """Register aggregation modes for resource-usage counters on the current stage.

    Must be called once per task before the first ``_sample_process_stats``
    so that AVERAGE/MAX counters are reduced correctly.  SUM is the default
    and listed only for documentation.
    """
    sc = counters.current_stage()
    sc.set_aggregation(ZEPHYR_STAGE_ITEM_COUNT_KEY, Aggregation.SUM)
    sc.set_aggregation(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, Aggregation.SUM)
    sc.set_aggregation(ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, Aggregation.AVERAGE)
    sc.set_aggregation(ZEPHYR_WORKER_CPU_TIME_KEY, Aggregation.SUM)
    sc.set_aggregation(ZEPHYR_WORKER_MEM_AVERAGE_KEY, Aggregation.AVERAGE)
    sc.set_aggregation(ZEPHYR_WORKER_MEM_PEAK_KEY, Aggregation.MAX)


def _periodic_sampler(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    interval: float,
    *,
    cpu_s_at_start: float,
    stats_writer: StatsWriter,
    task: ShardTask,
    execution_id: str,
    start_time: float,
    proc: psutil.Process,
) -> None:
    """Periodically sample process stats and emit RUNNING rows to finelog.

    ``stats_writer.emit_worker_stat`` is itself a no-op when finelog is
    unavailable, so no gating is needed here.
    """
    while not stop_event.wait(timeout=interval):
        try:
            _sample_process_stats(cpu_s_at_start, proc)
            stats_writer.emit_worker_stat(
                task.stage_name,
                task.shard_idx,
                execution_id,
                ZephyrWorkerStatStatus.RUNNING,
                start_time,
                ctx.get_counters(),
            )
        except Exception:
            logger.warning("Failed to sample/emit process stats", exc_info=True)


def _run_stage_with_ctx(
    task: ShardTask,
    chunk_prefix: str,
    execution_id: str,
) -> TaskResult:
    """Run one ShardTask in the active worker context, writing stage output to disk.

    Shared between ``InlineRunner.execute`` and the subprocess child entry. The
    caller must set ``_worker_ctx_var`` first; counter recording reads it via
    ``counters.current_stage()``. Once that ctx is in place the actual per-shard
    work is identical.
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
        _wrap_stage_stats(run_stage(stage_ctx, task.operations, external_sort_dir=external_sort_dir)),
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

    def __init__(self) -> None:
        self._ctx: _InProcessWorkerContext | None = None

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, task.stage_name, task_memory_bytes=task.cost.memory)
        self._ctx = ctx
        worker_token = _worker_ctx_var.set(ctx)
        _set_counter_aggregations()
        stop_event = threading.Event()
        stats_writer = StatsWriter.connect()
        proc = psutil.Process()
        start_time = time.monotonic()
        cpu_times_at_start = proc.cpu_times()
        cpu_s_at_start = cpu_times_at_start.user + cpu_times_at_start.system
        proc.cpu_percent()  # prime so subsequent calls have a baseline
        stats_writer.emit_worker_stat(
            task.stage_name, task.shard_idx, execution_id, ZephyrWorkerStatStatus.START, start_time, ctx.get_counters()
        )
        sampler = threading.Thread(
            target=_periodic_sampler,
            kwargs={
                "stop_event": stop_event,
                "ctx": ctx,
                "interval": SUBPROCESS_STATS_INTERVAL,
                "cpu_s_at_start": cpu_s_at_start,
                "stats_writer": stats_writer,
                "task": task,
                "execution_id": execution_id,
                "start_time": start_time,
                "proc": proc,
            },
            daemon=True,
            name="zephyr-inline-stats-sampler",
        )
        sampler.start()
        _task_failed = False
        try:
            result = _run_stage_with_ctx(task, chunk_prefix, execution_id)
        except Exception:
            _task_failed = True
            raise
        finally:
            stop_event.set()
            sampler.join(timeout=2.0)
            if not sampler.is_alive():
                _sample_process_stats(cpu_s_at_start, proc)
            _status = ZephyrWorkerStatStatus.FAILED if _task_failed else ZephyrWorkerStatStatus.END
            stats_writer.emit_worker_stat(
                task.stage_name, task.shard_idx, execution_id, _status, start_time, ctx.get_counters()
            )
            stats_writer.close()
            _worker_ctx_var.reset(worker_token)
            self._ctx = None
        return result, dict(ctx._counters)

    def live_counters(self) -> dict[str, CounterEntry]:
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

    """

    def __init__(self) -> None:
        self._counter_file: str | None = None

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        finelog_url = StatsWriter.resolve_url()  # Requires Iris context, so called here and passed to subprocess
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
                [sys.executable, "-u", "-m", "zephyr.shard_subprocess", task_file, result_file],
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

    def live_counters(self) -> dict[str, CounterEntry]:
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
