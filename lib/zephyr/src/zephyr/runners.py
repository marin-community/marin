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
import math
import os
import re
import signal
import subprocess as sp
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, TypeVar

import cloudpickle
import psutil
from rigging.filesystem.storage_path import StoragePath

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
    WORKER_STATS_INTERVAL,
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY,
    ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY,
    ZEPHYR_WORKER_CPU_TIME_KEY,
    ZEPHYR_WORKER_MEM_AVERAGE_KEY,
    ZEPHYR_WORKER_MEM_CURRENT_KEY,
    ZEPHYR_WORKER_MEM_PEAK_KEY,
)
from zephyr.worker_context import Aggregation, CounterEntry, CounterSnapshot, _worker_ctx_var, merge_counter_entries

logger = logging.getLogger(__name__)


__all__ = ["InlineRunner", "StageRunner", "SubprocessRunner"]


_ACCUMULATED_RESOURCE_COUNTER_KEYS = (
    ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY,
    ZEPHYR_WORKER_MEM_AVERAGE_KEY,
    ZEPHYR_WORKER_MEM_PEAK_KEY,
)
_RESOURCE_COUNTER_AGGREGATIONS = (
    (ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, Aggregation.AVERAGE),
    (ZEPHYR_WORKER_CPU_TIME_KEY, Aggregation.SUM),
    (ZEPHYR_WORKER_MEM_AVERAGE_KEY, Aggregation.AVERAGE),
    (ZEPHYR_WORKER_MEM_PEAK_KEY, Aggregation.MAX),
)

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
        # fold() rather than merge(): this runs twice per pipeline item, so avoid
        # allocating the throwaway CounterEntry that merge() would take.
        entry.fold(value)

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


def _sample_process_stats(
    cpu_time_at_start: float,
    proc: psutil.Process,
    ctx: _InProcessWorkerContext,
) -> None:
    """Sample the current process's resource usage into the shard context.

    Uses set_counter (not increment) because these are point-in-time metrics.
    Peak memory is tracked as a monotonically increasing max across calls.
    ``cpu_time_at_start`` is subtracted from cumulative CPU time to give per-shard delta.
    ``proc`` must be the same object across calls so cpu_percent() has a
    prior measurement to diff against; prime it once before the first sample.
    The context is explicit because sampler threads do not inherit ContextVars.
    """
    rss = proc.memory_info().rss
    cpu_times = proc.cpu_times()
    cpu_pct = proc.cpu_percent()
    stage = ctx.current_stage_name()
    ctx.set_counter(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, cpu_pct, stage=stage)
    ctx.update_counter(ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, cpu_pct, stage=stage)
    ctx.set_counter(
        ZEPHYR_WORKER_CPU_TIME_KEY,
        cpu_times.user + cpu_times.system - cpu_time_at_start,
        stage=stage,
    )
    ctx.set_counter(ZEPHYR_WORKER_MEM_CURRENT_KEY, rss, stage=stage)
    ctx.update_counter(ZEPHYR_WORKER_MEM_AVERAGE_KEY, rss, stage=stage)
    ctx.update_counter(ZEPHYR_WORKER_MEM_PEAK_KEY, rss, stage=stage)


def _set_counter_aggregations() -> None:
    """Register aggregation modes for resource-usage counters on the current stage.

    Must be called once per task before the first ``_sample_process_stats``
    so that AVERAGE/MAX counters are reduced correctly.  SUM is the default
    and listed only for documentation.
    """
    stage_counters = counters.current_stage()
    stage_counters.set_aggregation(ZEPHYR_STAGE_ITEM_COUNT_KEY, Aggregation.SUM)
    stage_counters.set_aggregation(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, Aggregation.SUM)
    for name, aggregation in _RESOURCE_COUNTER_AGGREGATIONS:
        stage_counters.set_aggregation(name, aggregation)


def _periodic_sampler(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    interval: float,
    *,
    cpu_time_at_start: float,
    proc: psutil.Process,
) -> None:
    """Periodically sample process stats into the shard context."""
    while not stop_event.wait(timeout=interval):
        try:
            _sample_process_stats(cpu_time_at_start, proc, ctx)
        except Exception:
            logger.warning("Failed to sample process stats", exc_info=True)


@contextmanager
def _shard_counter_session(
    ctx: _InProcessWorkerContext,
    *,
    sample_interval: float | None,
    sampler_thread_name: str | None,
) -> Iterator[float]:
    """Sample shard resource counters periodically and once at completion."""
    proc = psutil.Process()
    proc.cpu_percent()  # prime so subsequent calls have a baseline
    cpu_times_at_start = proc.cpu_times()
    cpu_time_at_start = cpu_times_at_start.user + cpu_times_at_start.system
    start_time = time.monotonic()

    stop_event = threading.Event()
    sampler: threading.Thread | None = None
    if sample_interval is not None:
        assert sampler_thread_name is not None
        sampler = threading.Thread(
            target=_periodic_sampler,
            kwargs={
                "stop_event": stop_event,
                "ctx": ctx,
                "interval": sample_interval,
                "cpu_time_at_start": cpu_time_at_start,
                "proc": proc,
            },
            daemon=True,
            name=sampler_thread_name,
        )
        sampler.start()

    try:
        yield start_time
    finally:
        stop_event.set()
        if sampler is not None:
            sampler.join(timeout=2.0)
        if sampler is None or not sampler.is_alive():
            try:
                _sample_process_stats(cpu_time_at_start, proc, ctx)
            except Exception:
                logger.warning("Failed to take final process stats sample", exc_info=True)


def _run_stage_with_ctx(
    task: ShardTask,
    chunk_prefix: str,
    execution_id: str,
    external_sort_dir: str | None = None,
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
    if external_sort_dir is None:
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
        self._last_counters: dict[str, CounterEntry] = {}

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, task.stage_name, task_memory_bytes=task.cost.memory)
        self._ctx = ctx
        self._last_counters = {}
        worker_token = _worker_ctx_var.set(ctx)
        _set_counter_aggregations()
        try:
            with _shard_counter_session(
                ctx,
                sample_interval=WORKER_STATS_INTERVAL,
                sampler_thread_name="zephyr-inline-stats-sampler",
            ):
                result = _run_stage_with_ctx(task, chunk_prefix, execution_id)
        finally:
            self._last_counters = dict(ctx._counters)
            _worker_ctx_var.reset(worker_token)
            self._ctx = None
        return result, dict(ctx._counters)

    def live_counters(self) -> dict[str, CounterEntry]:
        ctx = self._ctx
        return dict(ctx._counters) if ctx is not None else dict(self._last_counters)


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
        self._process: psutil.Process | None = None
        self._process_stats: _InProcessWorkerContext | None = None
        self._cpu_time_at_start = 0.0
        self._last_counters: dict[str, CounterEntry] = {}
        self._state_lock = threading.Lock()

    def _child_returncode(
        self,
        command: list[str],
        child_env: dict[str, str],
        execution_id: str,
        stage_name: str,
    ) -> int:
        with sp.Popen(
            command,
            env=child_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        ) as proc:
            process = psutil.Process(proc.pid)
            process.cpu_percent()
            cpu_times_at_start = process.cpu_times()
            process_stats = _InProcessWorkerContext("", execution_id, stage_name)
            for name, aggregation in _RESOURCE_COUNTER_AGGREGATIONS:
                process_stats.set_aggregation(name, aggregation)
            with self._state_lock:
                self._process = process
                self._process_stats = process_stats
                self._cpu_time_at_start = cpu_times_at_start.user + cpu_times_at_start.system
            try:
                return proc.wait()
            finally:
                with self._state_lock:
                    self._process = None

    def _final_counters(self, child_counters: dict[str, CounterEntry]) -> dict[str, CounterEntry]:
        with self._state_lock:
            process_counters = dict(self._process_stats._counters) if self._process_stats is not None else {}
        final_counters = dict(child_counters)
        for name in _ACCUMULATED_RESOURCE_COUNTER_KEYS:
            entries = [(name, entry) for entry in (process_counters.get(name), final_counters.get(name)) if entry]
            if entries:
                merged, _ = merge_counter_entries(entries)
                final_counters[name] = merged[name]
        return final_counters

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cloudpickle.dump((task, chunk_prefix, execution_id), f)
            task_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result_file = f.name
        counter_file = f"{result_file}.counters"
        self._counter_file = counter_file
        self._last_counters = {}

        try:
            # ``-u`` keeps the child's stdout/stderr unbuffered so any
            # faulthandler traceback reaches the parent's log before the
            # process dies.
            child_env = os.environ.copy()
            child_env["POLARS_MAX_THREADS"] = str(max(1, math.ceil(task.cost.cpu)))
            with tempfile.TemporaryDirectory(prefix=f"zephyr-external-sort-{task.shard_idx:04d}-") as sort_dir:
                returncode = self._child_returncode(
                    [sys.executable, "-u", "-m", "zephyr.shard_subprocess", task_file, result_file, sort_dir],
                    child_env,
                    execution_id,
                    task.stage_name,
                )

            if returncode != 0:
                # Linux OOM-killer sends SIGKILL → returncode == -9. Distinguish
                # so callers/retries can react to memory pressure specifically.
                if returncode == -signal.SIGKILL:
                    raise MemoryError(
                        f"Subprocess for shard {task.shard_idx} was killed by SIGKILL "
                        f"(returncode {returncode}); most likely OOM-killed by the kernel."
                    )
                raise RuntimeError(
                    f"Subprocess for shard {task.shard_idx} exited with code {returncode}; "
                    "see worker stderr above for the faulthandler traceback."
                )

            with open(result_file, "rb") as f:
                result_or_error, child_counters = cloudpickle.load(f)

            final_counters = self._final_counters(child_counters)
            self._last_counters = final_counters

            # Switch heartbeat reads to the final snapshot before returning.
            self._counter_file = None

            if isinstance(result_or_error, Exception):
                raise result_or_error

            return result_or_error, dict(final_counters)
        finally:
            self._counter_file = None
            with self._state_lock:
                self._process = None
                self._process_stats = None
            for p in (task_file, result_file, counter_file, f"{counter_file}.tmp"):
                with suppress(FileNotFoundError):
                    os.unlink(p)

    def live_counters(self) -> dict[str, CounterEntry]:
        cf = self._counter_file
        if cf is None:
            return dict(self._last_counters)
        counters: dict[str, CounterEntry] = {}
        try:
            with open(cf, "rb") as f:
                counters = cloudpickle.load(f)
        except (FileNotFoundError, EOFError):
            pass
        except Exception:
            logger.warning("Failed to read counter file %s", cf, exc_info=True)

        with self._state_lock:
            if self._process is not None and self._process_stats is not None:
                try:
                    _sample_process_stats(self._cpu_time_at_start, self._process, self._process_stats)
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error:
                    logger.warning("Failed to sample subprocess resource use", exc_info=True)
                counters.update(self._process_stats._counters)
        if counters:
            self._last_counters = dict(counters)
        return counters
