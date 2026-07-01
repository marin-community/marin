# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess child entry point for ``SubprocessRunner``.

``SubprocessRunner`` runs each shard as ``python -m zephyr.shard_subprocess``.
This module is the child body: it loads one ``ShardTask``, runs the stage in a
clean process, and writes the result file the parent reads back.

It is kept separate from ``zephyr.runners`` (which holds the runner classes and
the shared in-process helpers) so the module executed via ``python -m`` is never
*also* imported during ``zephyr`` package initialization. ``zephyr.runners`` is
pulled in transitively by ``zephyr/__init__.py`` (``__init__`` → ``execution``
→ ``runners``); if it were the ``-m`` target, ``runpy`` would find it already
in ``sys.modules`` and warn while re-executing its body a second time. Nothing
in the package-init graph imports ``shard_subprocess``, so no such warning.
"""

import logging
import os
import sys
import threading
import time
import traceback
from contextlib import suppress
from typing import Any

import cloudpickle
import psutil
import pyarrow as pa
from rigging.log_setup import configure_logging

from zephyr.runners import (
    SUBPROCESS_STATS_INTERVAL,
    _InProcessWorkerContext,
    _periodic_sampler,
    _run_stage_with_ctx,
    _sample_process_stats,
    _set_counter_aggregations,
)
from zephyr.stage_io import _ensure_picklable_exception, _stage_throughput
from zephyr.stats import StatsWriter, ZephyrWorkerStatStatus
from zephyr.worker_context import _worker_ctx_var

logger = logging.getLogger(__name__)


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
    while not stop_event.wait(timeout=interval):
        if sys.is_finalizing():
            return
        elapsed = time.monotonic() - monotonic_start
        # Map-only stages never populate these counters; logging zeros is misleading.
        throughput = _stage_throughput(ctx.get_counters(stage_name), elapsed)
        if throughput is None:
            continue
        logger.info(
            "[%s] [%s] [%s] shard %d/%d; %s",
            execution_id,
            stage_name,
            threading.current_thread().name,
            shard_idx,
            total_shards,
            throughput,
        )


def _execute_shard_subprocess(task_file: str, result_file: str) -> None:
    """Subprocess child body: runs one ShardTask and writes the result file."""
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
    sampler: threading.Thread | None = None
    result_or_error: Any
    ctx: _InProcessWorkerContext | None = None
    stats_writer: StatsWriter = StatsWriter(None)
    proc = psutil.Process()
    proc.cpu_percent()
    start_time = 0.0
    cpu_s_at_start = 0.0
    _task_failed = True  # cleared to False only on clean _run_stage_with_ctx completion
    try:
        with open(task_file, "rb") as f:
            task, chunk_prefix, execution_id, finelog_url = cloudpickle.load(f)

        start_time = time.monotonic()
        _cpu = proc.cpu_times()
        cpu_s_at_start = _cpu.user + _cpu.system

        if finelog_url:
            stats_writer = StatsWriter.connect(finelog_url)

        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, task.stage_name, task_memory_bytes=task.cost.memory)
        _worker_ctx_var.set(ctx)
        _set_counter_aggregations()

        shard_monotonic_start = time.monotonic()
        stats_writer.emit_worker_stat(
            task.stage_name, task.shard_idx, execution_id, ZephyrWorkerStatStatus.START, start_time, ctx.get_counters()
        )

        flusher = threading.Thread(
            target=_periodic_counter_writer,
            args=(stop_event, ctx, counter_file, SUBPROCESS_STATS_INTERVAL),
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
                SUBPROCESS_STATS_INTERVAL,
            ),
            daemon=True,
            name="zephyr-subprocess-status-logger",
        )
        status_logger.start()

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
            name="zephyr-subprocess-stats-sampler",
        )
        sampler.start()

        result_or_error = _run_stage_with_ctx(task, chunk_prefix, execution_id)
        _task_failed = False
    except Exception as e:
        # Cloudpickling an exception drops ``__traceback__``, so a naive
        # parent re-raise would otherwise show only the parent stack at the
        # re-raise site. ``__notes__`` survives pickling and Python prints
        # it inline when the exception eventually propagates.
        logger.exception("Subprocess shard execution failed")
        e.add_note(f"--- subprocess traceback ---\n{traceback.format_exc().rstrip()}")
        # Normalize before handing the error to the parent: a subclass that
        # cannot round-trip through pickle (e.g. an __init__ whose signature
        # does not match its args) would otherwise revive into a TypeError when
        # the parent loads the result file, masking the real failure.
        result_or_error = _ensure_picklable_exception(e)
    finally:
        stop_event.set()
        if flusher is not None and flusher.is_alive():
            flusher.join(timeout=2.0)
        if status_logger is not None and status_logger.is_alive():
            status_logger.join(timeout=2.0)
        if sampler is not None and sampler.is_alive():
            sampler.join(timeout=2.0)
        if ctx is not None:
            if sampler is None or not sampler.is_alive():
                with suppress(Exception):
                    _sample_process_stats(cpu_s_at_start, proc)
            _status = ZephyrWorkerStatStatus.FAILED if _task_failed else ZephyrWorkerStatStatus.END
            with suppress(Exception):
                stats_writer.emit_worker_stat(
                    task.stage_name, task.shard_idx, execution_id, _status, start_time, ctx.get_counters()
                )
        stats_writer.close()

    with open(result_file, "wb") as f:
        counters_out = dict(ctx._counters) if ctx is not None else {}
        cloudpickle.dump((result_or_error, counters_out), f)


def _subprocess_main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m zephyr.shard_subprocess <task_file> <result_file>", file=sys.stderr)
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
