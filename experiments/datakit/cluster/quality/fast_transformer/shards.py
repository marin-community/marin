# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-shard Zephyr steps over one normalized source and its co-partitioned leaves.

Every leaf of a source (Harrier embeddings, fusion scores, content types) holds the
normalized shard's documents under the same basename and in the same row order. The
quality steps each walk a normalized shard, read the same-basename shard of one or
more such leaves beside it, and write one output shard under that basename:
:func:`map_normalized_shards` is that driver, and :class:`AlignedColumn` is how a
step takes a side leaf's rows while checking they belong to the documents it walks.
"""

import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

import numpy as np
import polars as pl
import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.coordinator import ZephyrExecutionResult
from zephyr.dataset import Dataset, ShardInfo

logger = logging.getLogger(__name__)

COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)

ShardFn = Callable[[Iterator[pa.RecordBatch], ShardInfo, tuple[str, ...]], Iterator[pa.RecordBatch]]
"""``(normalized batches, shard, side shard paths) -> output batches`` for one shard."""


@dataclass(frozen=True)
class ShardPool:
    """Where a per-shard step runs: ``task``-sized tasks packed onto up to ``max_workers`` ``worker``-sized workers."""

    worker: ResourceConfig
    task: ResourceConfig
    max_workers: int


def paired_basenames(*dirs: str) -> list[str]:
    """The parquet basenames every directory holds, refusing any asymmetry.

    Co-partitioned leaves of one source share their complete basename sets. A leaf
    that carries a basename another lacks came from a different normalize run, and
    its documents would otherwise leave no trace in the output.
    """
    sets = {d: {os.path.basename(str(p)) for p in (StoragePath(d) / "*.parquet").glob()} for d in dirs}
    first_dir, first = next(iter(sets.items()))
    if not first:
        raise FileNotFoundError(f"no parquet shards under {first_dir}")
    for other_dir, other in sets.items():
        if other != first:
            missing = sorted(first - other)[:3]
            extra = sorted(other - first)[:3]
            raise ValueError(
                f"{other_dir} is not co-partitioned with {first_dir}: {len(first - other)} basenames missing "
                f"(e.g. {missing}) and {len(other - first)} unexpected (e.g. {extra})"
            )
    return sorted(first)


@dataclass
class AlignedColumn:
    """One co-partitioned shard's column, taken in step with the normalized shard's rows.

    Each :meth:`take` checks the ids of the rows it returns against the documents they
    are paired with, and :meth:`require_consumed` checks that no row was left over, so a
    side written from a different normalize run fails the shard rather than misaligning it.
    """

    ids: np.ndarray
    values: np.ndarray
    where: str
    consumed: int = 0

    def take(self, doc_ids: np.ndarray) -> np.ndarray:
        """Return the values of the next ``len(doc_ids)`` rows, which must carry ``doc_ids``."""
        start, end = self.consumed, self.consumed + len(doc_ids)
        if end > len(self.ids) or not np.array_equal(self.ids[start:end], doc_ids):
            raise ValueError(
                f"{self.where}: rows {start}..{end} do not carry the normalized shard's ids; "
                f"the two sides did not come from one normalize run"
            )
        self.consumed = end
        return self.values[start:end]

    def require_consumed(self) -> None:
        if self.consumed != len(self.ids):
            raise ValueError(
                f"{self.where}: {len(self.ids)} rows against {self.consumed} documents; "
                f"the two sides did not come from one normalize run"
            )


def read_aligned_column(path: str, column: str, where: str) -> AlignedColumn:
    """Read ``id`` and ``column`` from one co-partitioned parquet shard, whole."""
    # polars types a fixed-width list column as an Array with no offsets buffer,
    # so the int32 offset ceiling that fails a whole-column pyarrow read of the
    # largest Harrier shards (2,682,446 documents x 1,024 values > 2^31-1) does
    # not apply, and to_numpy hands back one contiguous [n, width] block.
    with StoragePath(path).open("rb") as fh:
        frame = pl.read_parquet(fh, columns=["id", column])
    return AlignedColumn(frame.get_column("id").to_numpy(), frame.get_column(column).to_numpy(), where)


def rebatch(batches: Iterator[pa.RecordBatch], rows_per_batch: int) -> Iterator[pa.RecordBatch]:
    """Regroup record batches into batches of exactly ``rows_per_batch`` rows, plus a tail."""
    pending: list[pa.RecordBatch] = []
    rows = 0
    for batch in batches:
        pending.append(batch)
        rows += batch.num_rows
        if rows < rows_per_batch:
            continue
        table = pa.Table.from_batches(pending)
        full = rows - rows % rows_per_batch
        for start in range(0, full, rows_per_batch):
            yield pa.concat_batches(table.slice(start, rows_per_batch).to_batches())
        pending = table.slice(full).to_batches()
        rows -= full
    if rows:
        yield pa.concat_batches(pa.Table.from_batches(pending).to_batches())


def _with_side_paths(
    batches: Iterator[pa.RecordBatch],
    shard: ShardInfo,
    *,
    shard_fn: ShardFn,
    side_paths: tuple[tuple[str, ...], ...],
) -> Iterator[pa.RecordBatch]:
    return shard_fn(batches, shard, side_paths[shard.shard_idx])


def map_normalized_shards(
    *,
    name: str,
    text_dir: str,
    side_dirs: Sequence[str],
    columns: list[str],
    shard_fn: ShardFn,
    output_path: str,
    schema: pa.Schema,
    pool: ShardPool,
    zephyr_context: ZephyrContext | None = None,
    stage_runner_factory: Callable | None = None,
    shared: Mapping[str, object] = MappingProxyType({}),
) -> ZephyrExecutionResult:
    """Run ``shard_fn`` over every shard of a normalized source, one Zephyr task per shard.

    ``shard_fn`` receives the shard's normalized ``columns`` as record batches and the
    same-basename shard path under each of ``side_dirs``; what it yields is written
    under that basename in ``output_path``. ``shared`` values are put on the context
    for tasks to read. Output shards that already exist are skipped, so a rerun after
    a partial failure does only the remainder.
    """
    basenames = tuple(paired_basenames(text_dir, *side_dirs))
    side_paths = tuple(tuple(prefix_join(d, basename) for d in side_dirs) for basename in basenames)
    logger.info("%s: %d shards of %s beside %s -> %s", name, len(basenames), text_dir, list(side_dirs), output_path)

    def output_file(shard_idx: int, _total: int) -> str:
        return prefix_join(output_path, basenames[shard_idx])

    pipeline = (
        Dataset.from_list([prefix_join(text_dir, basename) for basename in basenames])
        .load_parquet(columns=columns, batch_mode=True)
        .map_shard(partial(_with_side_paths, shard_fn=shard_fn, side_paths=side_paths))
        .write_parquet(output_file, schema=schema, skip_existing=True)
    )
    ctx = zephyr_context or ZephyrContext(
        name=f"{name}-{os.path.basename(text_dir.rstrip('/'))[:8]}",
        resources=pool.worker,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=min(pool.max_workers, len(basenames)),
        stage_runner_factory=stage_runner_factory,
    )
    for key, value in shared.items():
        ctx.put(key, value)
    return ctx.execute(pipeline, verbose=True, map_task_resources=pool.task)
