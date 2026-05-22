# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types, worker-facing protocol surface, and formatting helpers.

This module collects three closely-related concerns:

* **Types** — the small dataclasses, enums, and exception classes that the
  coordinator, worker, and runner share (``ShardTask``, ``CounterSnapshot``,
  ``PickleDiskChunk``, ``TaskResult``, ``ShardFailureKind``, ``PullStatus``,
  ``WorkerState``, ``ZephyrWorkerError``).
* **Worker context** — the ``WorkerContext`` / ``StageRunner`` protocols and
  the ``_worker_ctx_var`` ``ContextVar`` plumbing exposed via
  ``zephyr_worker_ctx()``. ``zephyr.counters`` and the runners read this.
* **Formatting / IO helpers** — humanized count and byte formatting, status
  text pushing to Iris, execution-id generation, chunk path computation, and
  the disk-backed stage-output writers (``_write_pickle_chunks`` /
  ``_write_stage_output``).

It is a leaf module from the perspective of the other ``execution`` package
modules: ``coordinator``, ``worker``, and ``context`` all import from here,
never the other way around. ``zephyr.shuffle`` is a pure dependency below
this module — the scatter write buffer budget is computed here (using the
worker context) and passed into ``_write_scatter``, so ``shuffle`` no longer
needs to reach back up into ``execution`` at all.
"""

from __future__ import annotations

import enum
import itertools
import logging
import pickle
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import humanfriendly
from iris.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.env_resources import TaskResources
from rigging.filesystem import open_url, url_to_fs
from rigging.timing import RateLimiter, log_time

from zephyr.plan import PhysicalOp, Scatter, Shard
from zephyr.shuffle import ListShard, _write_scatter
from zephyr.writers import INTERMEDIATE_CHUNK_SIZE, batchify, ensure_parent_dir, unique_temp_path

logger = logging.getLogger(__name__)

# Max explicit task errors (report_error) per shard before aborting.
MAX_SHARD_FAILURES = 3

# Max infra failures observed *while the same shard was in flight* on the
# crashing worker before treating that shard as a deterministic crasher and
# aborting. Genuine preemption distributes across all in-flight shards, so
# the same shard hitting this cap is strong evidence the shard payload is
# what's killing the worker (e.g. native SIGSEGV from Arrow / JAX, or an
# OOM that brings the host down). Set well above realistic preemption
# storms for any one shard in a multi-shard pipeline.
MAX_SHARD_INFRA_FAILURES = 20

ZEPHYR_STAGE_ITEM_COUNT_KEY = "zephyr/stage/{stage_name}/item_count"
ZEPHYR_STAGE_BYTES_PROCESSED_KEY = "zephyr/stage/{stage_name}/bytes_processed"

# Typical status text for a 6-stage pipeline is ~300 chars.
MAX_STATUS_TEXT_LENGTH = 1000


class ShardFailureKind(enum.StrEnum):
    """TASK failures count toward MAX_SHARD_FAILURES; INFRA failures (preemption) do not."""

    TASK = enum.auto()
    INFRA = enum.auto()


class WorkerState(enum.Enum):
    INIT = "init"
    READY = "ready"
    BUSY = "busy"
    FAILED = "failed"
    DEAD = "dead"


class PullStatus(enum.StrEnum):
    """Control signals returned by ``ZephyrCoordinator.pull_task`` in place of a task.

    Three explicit states make the slot's required response unambiguous:

    - ``NO_WORK_BACKOFF``: queue is empty mid-stage; slot sleeps and retries.
    - ``STAGE_COMPLETED``: this stage's pool is done (boundary or epoch-stale slot);
      slot exits and the worker re-pools for the next stage.
    - ``SHUTDOWN``: pipeline is finished or the coordinator is shutting down;
      slot exits and the worker breaks its outer loop.
    """

    NO_WORK_BACKOFF = enum.auto()
    STAGE_COMPLETED = enum.auto()
    SHUTDOWN = enum.auto()


@dataclass(frozen=True)
class PickleDiskChunk:
    """Reference to a pickle chunk stored on disk.

    Each write goes to a UUID-unique path to avoid collisions when multiple
    workers race on the same shard.  No coordinator-side rename is needed;
    the winning result's paths are used directly and the entire execution
    directory is cleaned up after the pipeline completes.
    """

    path: str
    count: int

    def __iter__(self) -> Iterator:
        return iter(self.read())

    @classmethod
    def write(cls, path: str, data: list) -> PickleDiskChunk:
        """Write *data* to a UUID-unique path derived from *path*.

        The UUID suffix avoids collisions when multiple workers race on
        the same shard.  The resulting path is used directly for reads —
        no rename step is required.
        """
        ensure_parent_dir(path)
        data = list(data)
        count = len(data)

        unique_path = unique_temp_path(path)
        with open_url(unique_path, "wb") as f:
            pickle.dump(data, f)
        return cls(path=unique_path, count=count)

    def read(self) -> list:
        """Load chunk data from disk."""
        with open_url(self.path, "rb") as f:
            return pickle.load(f)


@dataclass
class CounterSnapshot:
    """Bundled counter values and monotonically increasing generation tag.

    The generation increments on every snapshot, so each heartbeat and
    report_result carries a unique tag.  The coordinator uses strict
    ordering (>) to discard stale or out-of-order updates.
    """

    counters: dict[str, int]
    generation: int

    @staticmethod
    def empty(generation: int = 0) -> CounterSnapshot:
        return CounterSnapshot(counters={}, generation=generation)


@dataclass
class TaskResult:
    """Result of a single worker task.

    Always contains a ListShard. For non-scatter stages, refs are
    PickleDiskChunks. For scatter stages, refs contain file paths
    (the actual metadata lives in ``.scatter_meta`` sidecar files
    read lazily by reducers).
    """

    shard: ListShard


@dataclass
class ShardTask:
    """Describes a unit of work for a worker: one shard through one stage."""

    shard_idx: int
    total_shards: int
    shard: Shard
    operations: list[PhysicalOp]
    stage_name: str = "output"
    aux_shards: dict[int, Shard] | None = None


class ZephyrWorkerError(RuntimeError):
    """Raised when a worker encounters a fatal (non-transient) error."""


class CoordinatorUnreachable(RuntimeError):
    """Worker lost contact with the coordinator. Retryable at the iris task level."""


# Application errors that should never be retried by the execute() retry loop.
# These are deterministic errors (bad plan, invalid config, programming bugs)
# that would fail identically on every attempt. Infrastructure errors (OSError,
# RuntimeError from dead actors, backend actor errors) are NOT listed here so they
# remain retryable.
_NON_RETRYABLE_ERRORS = (ZephyrWorkerError, ValueError, TypeError, KeyError, AttributeError, MemoryError)


# ---------------------------------------------------------------------------
# WorkerContext protocol — the public interface exposed to user task code
# ---------------------------------------------------------------------------


class WorkerContext(Protocol):
    def get_shared(self, name: str) -> Any: ...
    def increment_counter(self, name: str, value: int = 1) -> None: ...
    def get_counter_snapshot(self) -> CounterSnapshot: ...


_worker_ctx_var: ContextVar[WorkerContext | None] = ContextVar("zephyr_worker_ctx", default=None)


def zephyr_worker_ctx() -> WorkerContext:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _worker_ctx_var.get()
    if ctx is None:
        raise RuntimeError("zephyr_worker_ctx() called outside of a worker task")
    return ctx


class StageRunner(Protocol):
    """Strategy a worker uses to execute a single shard. See ``zephyr.runners``."""

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, int]]: ...

    def live_counters(self) -> dict[str, int]: ...


# ---------------------------------------------------------------------------
# Scatter write buffer sizing
# ---------------------------------------------------------------------------

# Fraction of cgroup memory allocated to scatter write buffers.
_SCATTER_WRITE_BUFFER_FRACTION = 0.25
# Static fallback used when the cgroup memory limit cannot be determined.
_SCATTER_WRITE_BUFFER_BYTES_FALLBACK = 256 * 1024 * 1024  # 256 MB


def _scatter_write_buffer_bytes() -> int:
    """Return the scatter write buffer budget for the current worker.

    Uses 25% of the container memory limit so the budget scales with the
    worker size, divided by the number of concurrent workers sharing this
    actor's RAM. Falls back to 256 MB (divided by the same factor) when the
    cgroup limit cannot be read.

    Lives here (not in ``zephyr.shuffle``) so that ``shuffle`` does not have
    to import from ``zephyr.execution`` to read ``_worker_ctx_var`` — the
    budget is computed at the call site in ``_write_stage_output`` and passed
    into ``_write_scatter`` as an explicit parameter.
    """
    num_workers = _worker_ctx_var.get().num_workers
    memory = TaskResources.from_environment().memory_bytes
    if memory > 0:
        return int(memory * _SCATTER_WRITE_BUFFER_FRACTION / num_workers)
    return _SCATTER_WRITE_BUFFER_BYTES_FALLBACK // max(1, num_workers)


# ---------------------------------------------------------------------------
# Formatting / IO helpers
# ---------------------------------------------------------------------------


def _generate_execution_id() -> str:
    """Generate unique ID for this execution to avoid conflicts."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _format_count(n: float) -> str:
    """Format a count with SI-style suffixes (K/M/B/T) once it grows past 1k."""
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if abs_n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs_n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs_n >= 1e3:
        return f"{n / 1e3:.2f}K"
    if n == int(n):
        return f"{int(n):,}"
    return f"{n:.1f}"


def _format_bytes(n: float) -> str:
    """Format a byte count with binary (IEC) prefixes."""
    return humanfriendly.format_size(int(n), binary=True)


def _stage_throughput(
    counters: Mapping[str, int],
    stage_name: str,
    elapsed: float,
) -> tuple[int, int, float, float] | None:
    """Return ``(items, bytes_processed, item_rate, byte_rate)`` for *stage_name*.

    Returns ``None`` when neither the item nor the byte counter has been
    recorded for this stage. Map-only stages and stages still in run_stage
    setup never populate these counters; ``None`` distinguishes that case
    from a real zero count so callers can suppress misleading "0 items"
    status lines.
    """
    item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)
    byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)
    if item_key not in counters and byte_key not in counters:
        return None
    items = counters.get(item_key, 0)
    bytes_processed = counters.get(byte_key, 0)
    item_rate = items / elapsed if elapsed > 0 else 0.0
    byte_rate = bytes_processed / elapsed if elapsed > 0 else 0.0
    return items, bytes_processed, item_rate, byte_rate


def _shared_data_path(prefix: str, execution_id: str, name: str) -> str:
    """Path for a shared data object: {prefix}/{execution_id}/shared/{name}.pkl"""
    return f"{prefix}/{execution_id}/shared/{name}.pkl"


def _push_iris_task_status(
    rate_limiter: RateLimiter,
    build_md: Callable[[], tuple[str, str]],
) -> None:
    """Push ``(detail, summary)`` markdown to the active Iris task's status, if any.

    No-op when not running inside an Iris task or when ``rate_limiter`` declines
    this tick. ``build_md`` is invoked lazily after the gating checks so the
    formatting work is skipped on the no-op path.
    """
    iris_client = ctx.client if (ctx := get_iris_ctx()) is not None else None
    if iris_client is None:
        return
    job_info = get_job_info()
    if job_info is None:
        return
    if not rate_limiter.should_run():
        return
    detail_md, summary_md = build_md()
    try:
        iris_client.report_task_status_text(job_info.task_id, detail_md, summary_md)
    except Exception:
        logger.warning("Failed to report task status text to Iris controller", exc_info=True)


def _cleanup_execution(prefix: str, execution_id: str) -> None:
    """Remove all chunk files for an execution."""
    exec_dir = f"{prefix}/{execution_id}"
    fs = url_to_fs(exec_dir)[0]

    with log_time(f"Cleaning up execution directory {exec_dir}"):
        if fs.exists(exec_dir):
            try:
                fs.rm(exec_dir, recursive=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup chunks at {exec_dir}: {e}")


def _write_pickle_chunks(
    items: Iterator,
    source_shard: int,
    chunk_path_fn: Callable[[int], str],
) -> ListShard:
    """Batch a plain item stream into pickle chunk files.

    Returns a ListShard containing PickleDiskChunk references.
    """
    chunks: list[Iterable] = []
    for pidx, batch in enumerate(batchify(items, n=INTERMEDIATE_CHUNK_SIZE)):
        chunk_ref = PickleDiskChunk.write(chunk_path_fn(pidx), batch)
        chunks.append(chunk_ref)
        written = pidx + 1
        if written % 10 == 0:
            logger.info(
                "[shard %d] Wrote %d pickle chunks so far (latest: %d items)",
                source_shard,
                written,
                chunk_ref.count,
            )

    return ListShard(refs=chunks)


def _write_stage_output(
    stage_gen: Iterator,
    source_shard: int,
    stage_dir: str,
    shard_idx: int,
    scatter_op: Scatter | None,
    total_shards: int,
) -> TaskResult:
    """Write stage output to disk.

    For scatter stages (``scatter_op`` is set), writes Parquet with envelope
    wrapping and ``.scatter_meta`` sidecars. Returns TaskResult with compact
    scatter metadata.

    For non-scatter stages, batches items into pickle chunk files. Returns
    TaskResult with a ListShard.
    """
    if scatter_op is not None:
        first_item = next(stage_gen, None)
        if first_item is None:
            return TaskResult(shard=ListShard(refs=[]))

        full_gen = itertools.chain([first_item], stage_gen)

        num_output_shards = scatter_op.num_output_shards if scatter_op.num_output_shards > 0 else total_shards
        data_path = f"{stage_dir}/shard-{shard_idx:04d}.shuffle"
        shard = _write_scatter(
            full_gen,
            source_shard,
            data_path,
            key_fn=scatter_op.key_fn,
            num_output_shards=num_output_shards,
            buffer_limit_bytes=_scatter_write_buffer_bytes(),
            sort_fn=scatter_op.sort_fn,
            combiner_fn=scatter_op.combiner_fn,
        )
        return TaskResult(shard=shard)

    def chunk_path_fn(idx: int) -> str:
        return f"{stage_dir}/shard-{shard_idx:04d}/chunk-{idx:04d}.pkl"

    return TaskResult(shard=_write_pickle_chunks(stage_gen, source_shard, chunk_path_fn))
