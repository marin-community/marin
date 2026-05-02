# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pluggable shard execution strategies for ZephyrWorker.

A ``StageRunner`` is the strategy a worker uses to execute one ``ShardTask``.
Two implementations ship here:

* ``InlineRunner`` (default) — runs the stage in the worker actor's own
  process. Cheapest; appropriate for tests and pipelines whose user code is
  trusted not to corrupt the worker.
* ``SubprocessRunner`` — runs the stage in a fresh ``python -m zephyr.runners``
  subprocess. Each shard gets a clean Python heap, Arrow pool, and file
  descriptors; native crashes (SIGSEGV from Arrow/JAX, OOM kill) surface as
  deterministic ``returncode != 0`` task errors instead of bringing down the
  worker actor. Slower (~700ms of cold-import overhead per task).

Pick the runner pipeline-wide via ``ZephyrContext(stage_runner_factory=...)``.
The factory is invoked once per worker actor; each worker holds its own
runner instance. The runner's ``live_counters()`` is polled by the worker's
heartbeat thread, so per-runner state (counter file pointer, in-memory ctx)
stays encapsulated.
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
import traceback
from collections.abc import Iterator
from contextlib import suppress
from typing import Any, TypeVar

import cloudpickle
from rigging.filesystem import open_url

from zephyr.execution import (
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    CounterSnapshot,
    ShardTask,
    StageRunner,
    TaskResult,
    _shared_data_path,
    _worker_ctx_var,
    _write_stage_output,
)
from zephyr.plan import Scatter, StageContext, run_stage

logger = logging.getLogger(__name__)


__all__ = ["InlineRunner", "StageRunner", "SubprocessRunner"]


SUBPROCESS_COUNTER_FLUSH_INTERVAL = 5.0
"""How often the subprocess child flushes its counter dict to disk.

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

    def __init__(self, chunk_prefix: str, execution_id: str):
        self._chunk_prefix = chunk_prefix
        self._execution_id = execution_id
        self._shared_data_cache: dict[str, Any] = {}
        self._counters: dict[str, int] = {}
        self._generation = 0

    def get_shared(self, name: str) -> Any:
        if name not in self._shared_data_cache:
            path = _shared_data_path(self._chunk_prefix, self._execution_id, name)
            logger.info("Loading shared data '%s' from %s", name, path)
            with open_url(path, "rb") as f:
                self._shared_data_cache[name] = cloudpickle.loads(f.read())
        return self._shared_data_cache[name]

    def increment_counter(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def get_counter_snapshot(self) -> CounterSnapshot:
        self._generation += 1
        return CounterSnapshot(counters=dict(self._counters), generation=self._generation)


_T = TypeVar("_T")


class _StageStatsGenerator:
    """Wraps a generator and records item count + byte size into the worker context."""

    def __init__(self, stage_name: str, ctx: _InProcessWorkerContext) -> None:
        self._item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)
        self._byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)
        self._ctx = ctx

    def wrap(self, gen: Iterator[_T]) -> Iterator[_T]:
        for item in gen:
            self._ctx.increment_counter(self._item_key, 1)
            self._ctx.increment_counter(self._byte_key, sys.getsizeof(item))
            yield item


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
    stats = _StageStatsGenerator(task.stage_name, ctx)
    return _write_stage_output(
        stats.wrap(run_stage(stage_ctx, task.operations, external_sort_dir=external_sort_dir)),
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
    ) -> tuple[TaskResult, dict[str, int]]:
        ctx = _InProcessWorkerContext(chunk_prefix, execution_id)
        self._ctx = ctx
        token = _worker_ctx_var.set(ctx)
        try:
            result = _run_stage_with_ctx(task, chunk_prefix, execution_id, ctx)
            return result, dict(ctx._counters)
        finally:
            _worker_ctx_var.reset(token)
            self._ctx = None

    def live_counters(self) -> dict[str, int]:
        ctx = self._ctx
        return dict(ctx._counters) if ctx is not None else {}


# ---------------------------------------------------------------------------
# SubprocessRunner — opt-in isolation
# ---------------------------------------------------------------------------


def _periodic_counter_writer(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    counter_file: str,
    interval: float,
) -> None:
    """Atomic temp-write + rename so the parent never reads a half-written file."""
    while not stop_event.wait(timeout=interval):
        try:
            tmp_path = f"{counter_file}.tmp"
            with open(tmp_path, "wb") as f:
                cloudpickle.dump(dict(ctx._counters), f)
            os.rename(tmp_path, counter_file)
        except Exception:
            logger.warning("Failed to flush counter file to %s", counter_file, exc_info=True)


def _periodic_status_logger(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    stage_name: str,
    execution_id: str,
    shard_idx: int,
    total_shards: int,
    monotonic_start: float,
    interval: float,
) -> None:
    """Per-shard items/bytes rate log line (mirrors coordinator ``_log_status``)."""
    item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)
    byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)
    while not stop_event.wait(timeout=interval):
        if sys.is_finalizing():
            return
        # Map-only stages never populate these counters; logging zeros is misleading.
        if item_key not in ctx._counters and byte_key not in ctx._counters:
            continue
        items = ctx._counters.get(item_key, 0)
        bytes_processed = ctx._counters.get(byte_key, 0)
        elapsed = time.monotonic() - monotonic_start
        item_rate = items / elapsed if elapsed > 0 else 0.0
        byte_rate = bytes_processed / elapsed if elapsed > 0 else 0.0
        logger.info(
            "[%s] [%s] [%s] shard %d/%d; items=%d (%.1f/s), bytes_processed=%.1fMiB (%.1fMiB/s)",
            execution_id,
            stage_name,
            threading.current_thread().name,
            shard_idx,
            total_shards,
            items,
            item_rate,
            bytes_processed / (1024 * 1024),
            byte_rate / (1024 * 1024),
        )


class SubprocessRunner:
    """Run each shard in a fresh ``python -m zephyr.runners`` subprocess.

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
    ) -> tuple[TaskResult, dict[str, int]]:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cloudpickle.dump((task, chunk_prefix, execution_id), f)
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
                [sys.executable, "-u", "-m", "zephyr.runners", task_file, result_file],
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


# ---------------------------------------------------------------------------
# Subprocess child entry point: `python -m zephyr.runners <task_file> <result_file>`
# ---------------------------------------------------------------------------


def _execute_shard_subprocess(task_file: str, result_file: str) -> None:
    """Subprocess child body: runs one ShardTask and writes the result file."""
    import pyarrow as pa
    from rigging.log_setup import configure_logging

    # Each shard already runs in its own subprocess; redundant Arrow thread
    # pools just compete with the parent's shard-level parallelism.
    pa.set_io_thread_count(1)
    pa.set_cpu_count(1)

    # configure_logging installs faulthandler so SIGSEGV / SIGABRT / SIGBUS
    # / SIGFPE / SIGILL in a C extension produces a Python traceback on
    # stderr instead of a bare ``returncode < 0``.
    configure_logging(level=logging.INFO)

    counter_file = f"{result_file}.counters"
    stop_event = threading.Event()
    flusher: threading.Thread | None = None
    status_logger: threading.Thread | None = None
    result_or_error: Any
    ctx: _InProcessWorkerContext | None = None
    try:
        with open(task_file, "rb") as f:
            task, chunk_prefix, execution_id = cloudpickle.load(f)

        ctx = _InProcessWorkerContext(chunk_prefix, execution_id)
        _worker_ctx_var.set(ctx)

        shard_monotonic_start = time.monotonic()

        flusher = threading.Thread(
            target=_periodic_counter_writer,
            args=(stop_event, ctx, counter_file, SUBPROCESS_COUNTER_FLUSH_INTERVAL),
            daemon=True,
            name="zephyr-subprocess-counter-flusher",
        )
        flusher.start()

        status_logger = threading.Thread(
            target=_periodic_status_logger,
            args=(
                stop_event,
                ctx,
                task.stage_name,
                execution_id,
                task.shard_idx,
                task.total_shards,
                shard_monotonic_start,
                SUBPROCESS_COUNTER_FLUSH_INTERVAL,
            ),
            daemon=True,
            name="zephyr-subprocess-status-logger",
        )
        status_logger.start()

        result_or_error = _run_stage_with_ctx(task, chunk_prefix, execution_id, ctx)
    except Exception as e:
        # Cloudpickling an exception drops ``__traceback__``, so a naive
        # parent re-raise would otherwise show only the parent stack at the
        # re-raise site. ``__notes__`` survives pickling and Python prints
        # it inline when the exception eventually propagates.
        logger.exception("Subprocess shard execution failed")
        e.add_note(f"--- subprocess traceback ---\n{traceback.format_exc().rstrip()}")
        result_or_error = e
    finally:
        stop_event.set()
        if flusher is not None and flusher.is_alive():
            flusher.join(timeout=2.0)
        if status_logger is not None and status_logger.is_alive():
            status_logger.join(timeout=2.0)

    with open(result_file, "wb") as f:
        counters_out = dict(ctx._counters) if ctx is not None else {}
        cloudpickle.dump((result_or_error, counters_out), f)


def _subprocess_main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m zephyr.runners <task_file> <result_file>", file=sys.stderr)
        os._exit(1)
    # Bypass interpreter shutdown: PyArrow GCS/Azure filesystem background
    # threads can race with module GC and fire ``std::terminate`` → SIGABRT,
    # poisoning the parent's returncode check. The result file is already
    # on disk and the counter flusher has been joined, so nothing in this
    # one-shot child needs ``atexit`` / ``__del__`` to run.
    exit_code = 0
    try:
        _execute_shard_subprocess(sys.argv[1], sys.argv[2])
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        with suppress(Exception):
            sys.stdout.flush()
        with suppress(Exception):
            sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    _subprocess_main()
