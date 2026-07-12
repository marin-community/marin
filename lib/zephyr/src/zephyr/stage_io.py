# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side stage IO: task/result types, throughput stats, and output writers.

These helpers run on the worker side of a Zephyr stage — describing a unit of
work (``ShardTask``), its result (``TaskResult``), the strategy that executes it
(``StageRunner``), and the routines that materialise a stage's item stream to
disk (pickle chunks for map stages, scattered shuffle files for reduce stages).
"""

import itertools
import logging
import pickle
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import cloudpickle
import humanfriendly
from fray.types import ResourceConfig
from rigging.filesystem import open_url, unique_temp_path

from zephyr.plan import PhysicalOp, Scatter, Shard
from zephyr.shuffle import ListShard, _write_scatter
from zephyr.stats import ZEPHYR_STAGE_BYTES_PROCESSED_KEY, ZEPHYR_STAGE_ITEM_COUNT_KEY, per_second
from zephyr.worker_context import CounterEntry
from zephyr.writers import INTERMEDIATE_CHUNK_SIZE, batchify, ensure_parent_dir

logger = logging.getLogger(__name__)


class ZephyrWorkerError(RuntimeError):
    """Raised when a worker encounters a fatal (non-transient) error."""


def _ensure_picklable_exception(error: BaseException) -> BaseException:
    """Return *error*, or a picklable stand-in if it cannot survive pickling.

    Exceptions cross process boundaries (shard subprocess → worker, coordinator
    job → driver) by cloudpickle. A subclass whose ``__init__`` signature is
    incompatible with the ``args`` the default reduce replays revives into a
    ``TypeError`` at *unpickle* time — which would crash the process that is only
    trying to re-raise a child's failure, and mask the real error behind an
    unrelated ``TypeError``. We detect that here and substitute a
    ``ZephyrWorkerError`` carrying the original type and message, preserving any
    traceback notes. ``ZephyrWorkerError`` is non-retryable on purpose: an error
    we cannot even deserialize must not be retried blindly.
    """
    try:
        cloudpickle.loads(cloudpickle.dumps(error))
        return error
    except Exception:
        wrapped = ZephyrWorkerError(f"{type(error).__module__}.{type(error).__qualname__}: {error}")
        for note in getattr(error, "__notes__", ()):
            wrapped.add_note(note)
        return wrapped


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
    def write(cls, path: str, data: list) -> "PickleDiskChunk":
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
class TaskResult:
    """Result of a single worker task.

    Always contains a ListShard. For non-scatter stages, refs are
    PickleDiskChunks. For scatter stages, refs contain file paths
    (the actual metadata lives in ``.scatter_meta`` sidecar files
    read lazily by reducers).
    """

    shard: ListShard


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


@dataclass(frozen=True, repr=False)
class StageThroughput:
    """Items and bytes processed by a stage with their per-second rates.

    Item counts/rates use SI suffixes (K/M/B/T); byte totals/rates use binary
    (IEC) prefixes. The default ``%s`` / ``repr`` rendering is the single
    canonical text form, used by all coordinator, worker, and per-shard log
    messages:

        ``items=7 (3.5/s), bytes_processed=1 KiB (512 bytes/s)``
    """

    items: int
    bytes_processed: int
    item_rate: float
    byte_rate: float

    def __repr__(self) -> str:
        return (
            f"items={_format_count(self.items)} ({_format_count(self.item_rate)}/s), "
            f"bytes_processed={_format_bytes(self.bytes_processed)} ({_format_bytes(self.byte_rate)}/s)"
        )


def _stage_throughput(
    counters: Mapping[str, int | float],
    elapsed: float,
) -> StageThroughput | None:
    """Return throughput stats from *counters*, or ``None`` if uninstrumented.

    Returns ``None`` when neither the item nor the byte counter is present.
    Map-only stages and stages still in run_stage setup never populate these
    counters; ``None`` distinguishes that case from a real zero count so
    callers can suppress misleading "0 items" status lines.

    Callers are responsible for passing stage-filtered counters when needed.
    """
    if ZEPHYR_STAGE_ITEM_COUNT_KEY not in counters and ZEPHYR_STAGE_BYTES_PROCESSED_KEY not in counters:
        return None
    items = int(counters.get(ZEPHYR_STAGE_ITEM_COUNT_KEY, 0))
    bytes_processed = int(counters.get(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, 0))
    item_rate = per_second(items, elapsed)
    byte_rate = per_second(bytes_processed, elapsed)
    return StageThroughput(
        items=items,
        bytes_processed=bytes_processed,
        item_rate=item_rate,
        byte_rate=byte_rate,
    )


def _shared_data_path(prefix: str, execution_id: str, name: str) -> str:
    """Path for a shared data object: {prefix}/{execution_id}/shared/{name}.pkl"""
    return f"{prefix}/{execution_id}/shared/{name}.pkl"


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
            sort_fn=scatter_op.sort_fn,
            combiner_fn=scatter_op.combiner_fn,
        )
        return TaskResult(shard=shard)

    def chunk_path_fn(idx: int) -> str:
        return f"{stage_dir}/shard-{shard_idx:04d}/chunk-{idx:04d}.pkl"

    return TaskResult(shard=_write_pickle_chunks(stage_gen, source_shard, chunk_path_fn))


@dataclass
class ZephyrTaskResources:
    """CPU and memory budget for a single task or worker resource pool."""

    cpu: float
    memory: int

    def __add__(self, other: "ZephyrTaskResources") -> "ZephyrTaskResources":
        return ZephyrTaskResources(cpu=self.cpu + other.cpu, memory=self.memory + other.memory)

    def __sub__(self, other: "ZephyrTaskResources") -> "ZephyrTaskResources":
        return ZephyrTaskResources(cpu=self.cpu - other.cpu, memory=self.memory - other.memory)

    def can_fit(self, cost: "ZephyrTaskResources") -> bool:
        return self.cpu >= cost.cpu and (cost.memory == 0 or self.memory >= cost.memory)

    @classmethod
    def from_resource_config(cls, config: ResourceConfig) -> "ZephyrTaskResources":
        return cls(
            cpu=config.cpu,
            memory=int(humanfriendly.parse_size(config.ram, binary=True)),
        )


@dataclass
class ShardTask:
    """Describes a unit of work for a worker: one shard through one stage."""

    shard_idx: int
    total_shards: int
    shard: Shard
    operations: list[PhysicalOp]
    cost: ZephyrTaskResources
    stage_name: str = "output"
    aux_shards: dict[int, Shard] | None = None


class StageRunner(Protocol):
    """Strategy a worker uses to execute a single shard. See ``zephyr.runners``."""

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]: ...
    def live_counters(self) -> dict[str, CounterEntry]: ...
