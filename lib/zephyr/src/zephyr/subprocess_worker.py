# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point for Zephyr shard execution.

Invoked as ``python -m zephyr.subprocess_worker <task_file> <result_file>``.
Provides full memory isolation — all allocations (page cache, Arrow pool,
Python heap, leaked file descriptors) are reclaimed when the subprocess
exits, so successive tasks on the same parent worker actor do not
accumulate state.

Everything in this module runs inside the *child* process. The parent
``ZephyrWorker`` only spawns the subprocess and reads back its result file
via ``execution.ZephyrWorker._execute_shard``.
"""

import logging
import os
import re
import sys
import threading
import traceback
from contextlib import suppress
from typing import Any, TypeVar
from collections.abc import Iterator

import cloudpickle
from rigging.filesystem import open_url

from zephyr import counters
from zephyr.execution import (
    CounterSnapshot,
    _shared_data_path,
    _worker_ctx_var,
    _write_stage_output,
)
from zephyr.plan import Scatter, StageContext, run_stage

logger = logging.getLogger(__name__)

SUBPROCESS_COUNTER_FLUSH_INTERVAL = 5.0
"""How often the subprocess flushes its counter snapshot to the counter file.

Matches the parent worker's heartbeat interval so each heartbeat reads at most
one stale snapshot before the next flush lands.
"""


T = TypeVar("T")


class StatisticsGenerator:
    """Wraps a generator and counts and sizes yielded items."""

    def __init__(self, stage_name: str) -> None:
        self._stage_name = stage_name

    def wrap(self, gen: Iterator[T]) -> Iterator[T]:
        for item in gen:
            counters.increment(f"zephyr/stage/{self._stage_name}/item_count", 1)
            counters.increment(f"zephyr/stage/{self._stage_name}/bytes_processed", sys.getsizeof(item))
            yield item


class _SubprocessWorkerContext:
    """Lightweight WorkerContext for subprocess shard execution.

    Provides ``get_shared`` (loads from GCS on demand) and counter tracking.
    Counters are collected after the subprocess exits via the result file.
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


def _write_counter_file(counter_file: str, counters: dict[str, int]) -> None:
    """Atomically replace ``counter_file`` with a pickled counters dict.

    Writing to a temp file then renaming guarantees the parent never reads a
    half-written file: ``os.rename`` is atomic on POSIX.
    """
    tmp_path = f"{counter_file}.tmp"
    with open(tmp_path, "wb") as f:
        cloudpickle.dump(counters, f)
    os.rename(tmp_path, counter_file)


def _periodic_counter_writer(
    stop_event: threading.Event,
    ctx: _SubprocessWorkerContext,
    counter_file: str,
    interval: float,
) -> None:
    """Background loop that flushes ``ctx._counters`` to ``counter_file``.

    The parent worker's heartbeat thread reads the counter file to forward
    live counter updates to the coordinator while the subprocess is still
    running. Exits when ``stop_event`` is set.
    """
    while not stop_event.wait(timeout=interval):
        try:
            _write_counter_file(counter_file, dict(ctx._counters))
        except Exception:
            logger.warning("Failed to flush counter file to %s", counter_file, exc_info=True)


def execute_shard(task_file: str, result_file: str) -> None:
    """Entry point for subprocess shard execution.

    Reads ``(task, chunk_prefix, execution_id)`` from *task_file*, executes the
    shard inline, and writes ``(result_or_error, counters)`` to *result_file*.
    The counters dict is what the user-supplied operation accumulated via
    ``zephyr.counters.increment``; the parent worker hands it straight to
    ``coordinator.report_result`` so per-stage counter views stay accurate.
    While the shard is running, a background thread also flushes the live
    counter dict to ``<result_file>.counters`` every
    ``SUBPROCESS_COUNTER_FLUSH_INTERVAL`` seconds so the parent can forward
    live updates via heartbeats.
    """
    import pyarrow as pa
    from rigging.log_setup import configure_logging

    # Each shard already runs in its own subprocess, so PyArrow's internal
    # I/O and CPU thread pools provide redundant parallelism — we get
    # shard-level parallelism from the parent spawning multiple subprocesses.
    pa.set_io_thread_count(1)
    pa.set_cpu_count(1)

    # configure_logging installs faulthandler for us, so SIGSEGV / SIGABRT /
    # SIGBUS / SIGFPE / SIGILL in a C extension (Arrow, NumPy, ...) produces
    # a Python traceback on stderr instead of a bare ``returncode < 0``.
    configure_logging(level=logging.INFO)

    counter_file = f"{result_file}.counters"
    stop_event = threading.Event()
    flusher: threading.Thread | None = None
    result_or_error: Any
    ctx: _SubprocessWorkerContext | None = None
    try:
        with open(task_file, "rb") as f:
            task, chunk_prefix, execution_id = cloudpickle.load(f)

        ctx = _SubprocessWorkerContext(chunk_prefix, execution_id)
        _worker_ctx_var.set(ctx)

        flusher = threading.Thread(
            target=_periodic_counter_writer,
            args=(stop_event, ctx, counter_file, SUBPROCESS_COUNTER_FLUSH_INTERVAL),
            daemon=True,
            name="zephyr-subprocess-counter-flusher",
        )
        flusher.start()

        stage_ctx = StageContext(
            shard=task.shard,
            shard_idx=task.shard_idx,
            total_shards=task.total_shards,
            aux_shards=task.aux_shards,
        )

        # Sanitize for use as a path component: replace non-alphanumeric runs with '-'
        output_stage_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task.stage_name).strip("-")
        stage_dir = f"{chunk_prefix}/{execution_id}/{output_stage_name}"
        external_sort_dir = f"{stage_dir}-external-sort/shard-{task.shard_idx:04d}"
        scatter_op = next((op for op in task.operations if isinstance(op, Scatter)), None)

        output_counter = StatisticsGenerator(task.stage_name)

        result_or_error = _write_stage_output(
            output_counter.wrap(run_stage(stage_ctx, task.operations, external_sort_dir=external_sort_dir)),
            source_shard=task.shard_idx,
            stage_dir=stage_dir,
            shard_idx=task.shard_idx,
            scatter_op=scatter_op,
            total_shards=task.total_shards,
        )
    except Exception as e:
        # Cloudpickling an exception drops ``__traceback__``, so the parent
        # re-raise would otherwise show only the parent's stack with no hint
        # at the subprocess origin. Attach the formatted traceback as a note
        # — ``__notes__`` survives pickling and Python prints it inline when
        # the exception eventually propagates.
        logger.exception("Subprocess shard execution failed")
        e.add_note(f"--- subprocess traceback ---\n{traceback.format_exc().rstrip()}")
        result_or_error = e
    finally:
        stop_event.set()
        if flusher is not None and flusher.is_alive():
            flusher.join(timeout=2.0)

    with open(result_file, "wb") as f:
        counters_out = dict(ctx._counters) if ctx is not None else {}
        cloudpickle.dump((result_or_error, counters_out), f)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m zephyr.subprocess_worker <task_file> <result_file>", file=sys.stderr)
        os._exit(1)
    # Bypass interpreter shutdown to avoid PyArrow GCS/Azure filesystem
    # background threads racing with module GC, which fires
    # ``std::terminate`` ("terminate called without an active exception")
    # → SIGABRT and poisons the parent's returncode check. The result file
    # is already on disk and the counter flusher has been joined, so there
    # is nothing in this one-shot worker that needs ``atexit`` / ``__del__``
    # to run. PyArrow exposes ``finalize_s3`` but no equivalent for GCS,
    # so ``os._exit`` is the only reliable way to skip the racy teardown.
    #
    # Run inside try/finally so any escape path from ``execute_shard``
    # (BaseException, a failure writing the result file, ...) still bypasses
    # interpreter shutdown — otherwise the abort poisons the returncode and
    # the parent fails the shard even though the work itself succeeded.
    exit_code = 0
    try:
        execute_shard(sys.argv[1], sys.argv[2])
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
    main()
