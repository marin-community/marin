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
from collections.abc import Iterator
from contextlib import suppress
from typing import Any, TypeVar

import cloudpickle
from rigging.filesystem import open_url

from zephyr.plan import Scatter, StageContext, run_stage
from zephyr.stage_io import (
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    ShardTask,
    StageRunner,
    TaskResult,
    _shared_data_path,
    _write_stage_output,
)
from zephyr.worker_context import CounterSnapshot, _worker_ctx_var

logger = logging.getLogger(__name__)


__all__ = ["InlineRunner", "StageRunner", "SubprocessRunner"]


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

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, int]]:
        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, num_workers=self._num_workers)
        self._ctx = ctx
        worker_token = _worker_ctx_var.set(ctx)
        try:
            result = _run_stage_with_ctx(task, chunk_prefix, execution_id, ctx)
            return result, dict(ctx._counters)
        finally:
            _worker_ctx_var.reset(worker_token)
            self._ctx = None

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
