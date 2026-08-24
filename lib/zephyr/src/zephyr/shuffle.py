# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a set of zstd-compressed Parquet files,
one combined file per flush (``c{chunk:04d}.parquet``) containing all target
shards' data sorted by ``(_SHARD_COL, _SORT_KEY_COL)``.  A msgpack sidecar
(``metadata.msgpack``) records ``files -> [path, ...]``, a global
``avg_item_bytes`` estimate, and exact per-target-shard payload bytes
(``shard_bytes``) used by reducers to size the external-sort decision.

On the read side, each reducer uses DataFusion predicate pushdown to scan only
its target shard. Row-group statistics skip non-matching groups via byte-range
GETs, so each reducer reads roughly 1/N of each file. DataFusion preserves the
declared file ordering and performs the global k-way merge. Large merges use a
bounded multi-pass merge with Parquet runs on the stage filesystem.

Write-side memory is bounded by buffer size: when the sum of Arrow table
buffers exceeds ``_SCATTER_FLUSH_MEMORY_FRACTION`` of available task memory,
all buffers are flushed together into one combined file.

Routing columns (``__zephyr_shard__``, ``__zephyr_sort_key__``) are added
in ``_items_to_table``; ``__zephyr_shard__`` is stripped on read,
``__zephyr_sort_key__`` is consumed by the merge and stripped after.
"""

import concurrent.futures
import gc
import io
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import cloudpickle
import humanfriendly
import msgspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datafusion import DataFrame, SessionContext, col, lit
from iris.env_resources import TaskResources
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import RateLimiter, log_time

from zephyr.external_sort import external_sort_merge, merged_frame
from zephyr.parquet_scan import datafusion_context, register_object_stores, scan_parquet
from zephyr.shard_keys import encode_key, hash_encoded_key
from zephyr.worker_context import _worker_ctx_var
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core shard data types
# ---------------------------------------------------------------------------


@dataclass
class MemChunk:
    """In-memory chunk."""

    items: list[Any]

    def __iter__(self) -> Iterator:
        return iter(self.items)


@dataclass
class ListShard:
    """Shard backed by a list of iterable references (PickleDiskChunk, MemChunk, etc.)."""

    refs: list[Iterable]

    def __iter__(self) -> Iterator:
        for ref in self.refs:
            yield from ref


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCATTER_METADATA_FILENAME = "metadata.msgpack"

# Number of parallel small-file reads (sidecars, parquet schema footers) each
# reducer issues while building its ScatterReader. These reads are GCS
# GET-bound, so a modest pool keeps latency low without thrashing. The bound is
# per task, and a wave multiplies it: 2,048 reducers at 32 offered ~65,000
# simultaneous connections, more than a pod has ephemeral ports (#8402).
_SIDECAR_READ_CONCURRENCY = 8
# Fraction of available memory available for merging.
_SCATTER_READ_MEMORY_FRACTION = 0.4

# Memory overhead multiple per row in the DataFusion/Arrow representation.
_SCATTER_READ_DATAFUSION_ROW_OVERHEAD = 2
# Memory overhead multiple per row in the deserialized Python iterator.
_SCATTER_READ_PYTHON_ROW_OVERHEAD = 2

_PROGRESS_LOG_INTERVAL_SECONDS = 60.0
# Maximum run files that DataFusion merges at one time during an external sort.
_EXTERNAL_SORT_MAX_MERGE_FAN_IN = 32
# Bound Parquet footer size when a shuffle has thousands of target shards. One
# row group per target gives ideal predicate pruning, but makes every reducer
# read multi-megabyte footers from every mapper chunk before it can read data.
_SCATTER_MAX_ROW_GROUPS_PER_CHUNK = 512

# Internal routing columns injected by _items_to_table.
_SHARD_COL = "__zephyr_shard__"
_SORT_KEY_COL = "__zephyr_sort_key__"
# A cloudpickle-serialized Python object representing the item
_PAYLOAD_COL = "__payload__"
_PAYLOAD_BYTES_COL = "payload_bytes"
_PAYLOAD_BYTES_SUM_COL = f"{_PAYLOAD_BYTES_COL}_sum"

# Python items consumed before creating an Arrow table.
_TABLE_ROW_COUNT = 1000
# Preserve the prior 12% effective flush point. A scatter flush has reached
# 2.27x the buffered size at peak RSS: https://echo.oa.dev/wiki/68.
_SCATTER_FLUSH_MEMORY_FRACTION = 0.12
# Threshold for triggering a gc.collect() after a flush.
_GC_FLUSH_SIZE_THRESHOLD_BYTES = 8 * 1024 * 1024


def _task_memory_bytes() -> int:
    ctx = _worker_ctx_var.get()
    if ctx is not None and ctx.task_memory_bytes > 0:
        return ctx.task_memory_bytes

    memory_bytes = TaskResources.from_environment().memory_bytes
    if memory_bytes <= 0:
        logger.warning("No task memory is available. Using a 1 GiB memory budget.")
        return 1024 * 1024 * 1024
    return memory_bytes


def _table_to_items(table: pa.Table | pa.RecordBatch) -> Iterator[Any]:
    """Yield deserialized Python items from an Arrow table or record batch."""
    for payload in table[_PAYLOAD_COL].to_pylist():
        yield cloudpickle.loads(payload)


def _columns_to_table(
    payloads: list[bytes],
    shards: list[int],
    key_bytes: list[bytes],
    sort_values: list[Any],
) -> pa.Table:
    """Build a scatter Arrow table from pre-computed flat columns.

    Field order in the sort-key struct (key first) drives the
    ``(key, sort_value)`` ordering used by DataFusion.

    ``key_bytes`` must be pre-encoded via :func:`~zephyr.shard_keys.encode_key` so that
    the key field is always ``Binary`` — preventing struct schema mismatches
    when different mapper shards produce keys of different Python types.
    """
    try:
        sort_key = pa.StructArray.from_arrays(
            [pa.array(key_bytes, type=pa.binary()), pa.array(sort_values)],
            names=["key", "sort_value"],
        )
        return pa.table(
            {
                _PAYLOAD_COL: pa.array(payloads, type=pa.binary()),
                _SHARD_COL: pa.array(shards, type=pa.int32()),
                _SORT_KEY_COL: sort_key,
            }
        )
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as err:
        raise ValueError("sort_fn must return an Arrow-serializable object.") from err


def _items_to_table(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
) -> pa.Table:
    """Convert Python items to an Arrow table with routing columns.

    Cloudpickle-serializes items into ``_PAYLOAD_COL`` and adds ``_SHARD_COL``
    (int32 target shard index) and ``_SORT_KEY_COL``. This is the adapter
    between Python-item pipelines and the Arrow-based :class:`ScatterWriter`.

    ``num_output_shards=0`` means the caller assigns ``_SHARD_COL`` itself (the
    combiner path in :meth:`ScatterWriter._flush`, whose rows are already
    routed); routing is then skipped rather than computed and discarded.
    """
    shards: list[int] = []
    key_bytes: list[bytes] = []
    sort_values: list[Any] = []
    for item in items:
        key = key_fn(item)
        try:
            kb = encode_key(key)
        except TypeError as err:
            raise ValueError(f"key_fn must return a msgpack-serializable object; got {type(key).__name__!r}.") from err
        # Route from the bytes we just encoded: deterministic_hash(key) would
        # msgpack-encode the same key a second time for every scattered item.
        shards.append(hash_encoded_key(kb) % num_output_shards if num_output_shards > 0 else 0)
        key_bytes.append(kb)
        sort_values.append(sort_fn(item) if sort_fn is not None else None)
    payloads = [cloudpickle.dumps(item) for item in items]
    return _columns_to_table(payloads, shards, key_bytes, sort_values)


class _SidecarFilesystem(Protocol):
    """A protocol because ``url_to_fs`` returns a bare fsspec filesystem for
    ``s3://`` and a ``CrossRegionGuardedFS`` wrapper for ``gs://``, which share
    no base class."""

    def _strip_protocol(self, path: str) -> str: ...

    def cat_file(self, path: str) -> bytes: ...


@dataclass(frozen=True)
class _Sidecar:
    """One mapper's scatter metadata (``metadata.msgpack``).

    ``files`` lists the combined Parquet paths written during flushes; each file
    contains data for all target shards sorted by ``(_SHARD_COL, _SORT_KEY_COL)``.
    ``shard_bytes`` maps target shard index to exact payload bytes written for
    that shard across all files (used by reducers for the external-sort decision).
    ``path`` is the mapper output directory the sidecar lives under.
    """

    path: str
    files: list[str]
    avg_item_bytes: float
    shard_bytes: dict[int, int]

    _encoder: ClassVar[msgspec.msgpack.Encoder] = msgspec.msgpack.Encoder()
    _decoder: ClassVar[msgspec.msgpack.Decoder] = msgspec.msgpack.Decoder()
    _files_field: ClassVar[str] = "files"
    _avg_item_bytes_field: ClassVar[str] = "avg_item_bytes"
    _shard_bytes_field: ClassVar[str] = "shard_bytes"

    @staticmethod
    def meta_path(data_path: str) -> str:
        return f"{data_path}{_SCATTER_METADATA_FILENAME}"

    def target_bytes(self, target_shard: int) -> int:
        return self.shard_bytes.get(target_shard, 0)

    def write(self) -> None:
        """Serialize this sidecar to ``path/metadata.msgpack``."""
        meta_path = self.meta_path(self.path)
        payload = self._encoder.encode(
            {
                self._files_field: self.files,
                self._avg_item_bytes_field: self.avg_item_bytes,
                self._shard_bytes_field: {str(k): v for k, v in self.shard_bytes.items()},
            }
        )
        with log_time(f"Writing scatter meta for {self.path} to {meta_path}", level=logging.DEBUG):
            StoragePath(meta_path).write_bytes(payload)

    @classmethod
    def read(cls, fs: _SidecarFilesystem, data_path: str) -> "_Sidecar | None":
        """Load one non-empty sidecar, returning ``None`` for an empty writer."""
        meta_path = fs._strip_protocol(cls.meta_path(data_path))
        # Avoid buffered-file overhead for these small payloads.
        data = cls._decoder.decode(fs.cat_file(meta_path))
        files = data.get(cls._files_field, [])
        if not files:
            return None
        raw_shard_bytes = data.get(cls._shard_bytes_field, {})
        return cls(
            path=data_path,
            files=[str(f) for f in files],
            avg_item_bytes=float(data.get(cls._avg_item_bytes_field, 0)),
            shard_bytes={int(k): int(v) for k, v in raw_shard_bytes.items()},
        )

    @classmethod
    def read_all(cls, scatter_paths: list[str]) -> list["_Sidecar"]:
        """Read every non-empty sidecar concurrently in input order."""
        if not scatter_paths:
            return []
        ordered: list[_Sidecar | None] = [None] * len(scatter_paths)
        # Resolve before creating the pool so worker threads share one client.
        fs, _ = url_to_fs(cls.meta_path(scatter_paths[0]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=_SIDECAR_READ_CONCURRENCY) as pool:
            futures = {pool.submit(cls.read, fs, p): i for i, p in enumerate(scatter_paths)}
            for fut in concurrent.futures.as_completed(futures):
                ordered[futures[fut]] = fut.result()
        return [s for s in ordered if s is not None]


# ---------------------------------------------------------------------------
# ScatterReader: built from manifest, fed to Reduce
# ---------------------------------------------------------------------------


class ScatterReader:
    """All scatter chunks for one target shard, across all source shards.

    ``_chunk_paths`` lists every combined Parquet file the mappers wrote. Each
    file holds rows for *all* target shards, so :meth:`get_frames` filters each
    scan down to ``_target_shard``.

    Construct via :meth:`from_sidecars` for production use, or pass fields
    directly for testing.
    """

    def __init__(
        self,
        chunk_paths: list[str],
        target_shard: int,
        avg_item_bytes: float,
        shard_payload_bytes: float = 0.0,
    ) -> None:
        self._chunk_paths = chunk_paths
        self._target_shard = target_shard
        self.avg_item_bytes = avg_item_bytes
        self.shard_payload_bytes = shard_payload_bytes

    @classmethod
    def from_sidecars(cls, scatter_paths: list[str], target_shard: int) -> "ScatterReader":
        """Build a ScatterReader by reading per-mapper sidecars directly.

        Each reducer reads every mapper's ``metadata.msgpack`` sidecar in parallel
        and filters for its own ``target_shard``. No coordinator-written manifest
        is needed, which eliminates a serialization bottleneck when there are
        thousands of mappers.
        """
        chunk_paths: list[str] = []
        weighted_bytes = 0.0
        shard_payload_bytes = 0.0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            sidecars = _Sidecar.read_all(scatter_paths)
            for sidecar in sidecars:
                chunk_paths.extend(sidecar.files)
                weighted_bytes += sidecar.avg_item_bytes * len(sidecar.files)
                shard_payload_bytes += sidecar.target_bytes(target_shard)

        avg_item_bytes = weighted_bytes / len(chunk_paths) if chunk_paths else 0.0

        logger.info(
            "ScatterReader for shard %d: %d source shards, %d total chunks, "
            "avg_item_bytes=%.1f, shard_payload_bytes=%.0f",
            target_shard,
            len(sidecars),
            len(chunk_paths),
            avg_item_bytes,
            shard_payload_bytes,
        )
        return cls(
            chunk_paths=chunk_paths,
            target_shard=target_shard,
            avg_item_bytes=avg_item_bytes,
            shard_payload_bytes=shard_payload_bytes,
        )

    def get_frames(self, context: SessionContext) -> list[DataFrame]:
        """Build ordered DataFusion scans for this reducer's target shard."""
        paths = list(self._chunk_paths)
        if not paths:
            return []

        register_object_stores(context, paths)

        def scans(schema: pa.Schema | None = None) -> list[DataFrame]:
            def scan(path: str) -> DataFrame:
                return scan_parquet(
                    context,
                    path,
                    schema=schema,
                    sorted_by=(_SHARD_COL, _SORT_KEY_COL),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=_SIDECAR_READ_CONCURRENCY) as pool:
                return list(pool.map(scan, paths))

        frames = scans()
        schemas = [frame.schema() for frame in frames]
        if not all(schema.equals(schemas[0]) for schema in schemas[1:]):
            unified = pa.unify_schemas(schemas, promote_options="permissive")
            frames = scans(unified)

        return [frame.filter(col(_SHARD_COL) == lit(self._target_shard)).drop(_SHARD_COL) for frame in frames]

    @property
    def total_chunks(self) -> int:
        return len(self._chunk_paths)

    def merge_sorted_chunks(self, external_sort_dir: str) -> Iterator[Any]:
        """Merge sorted chunks using k-way merge, yielding items in global sort order.

        Each chunk file is assumed to be sorted by ``_SORT_KEY_COL`` (key plus optional
        secondary sort). Performs a k-way merge across all chunks.
        Args:
            external_sort_dir: If set and the shard exceeds the memory budget,
                spill intermediate runs.

        Yields:
            Deserialized Python items in merged sort order.
        """

        if self.total_chunks == 0:
            return

        estimated_merge_memory_bytes = self.shard_payload_bytes
        overhead = _SCATTER_READ_DATAFUSION_ROW_OVERHEAD * _SCATTER_READ_PYTHON_ROW_OVERHEAD
        memory_bytes = _task_memory_bytes()
        merge_memory_bytes = int(memory_bytes * _SCATTER_READ_MEMORY_FRACTION)

        context = datafusion_context(memory_limit_bytes=merge_memory_bytes)
        frames = self.get_frames(context)

        if estimated_merge_memory_bytes * overhead > merge_memory_bytes:
            fan_in = math.ceil(math.sqrt(self.total_chunks))
            logger.info(
                "[shard %d] Merging %d chunks via external sort " "(%s memory needed > %s memory available); fan_in=%d",
                self._target_shard,
                self.total_chunks,
                humanfriendly.format_size(estimated_merge_memory_bytes * overhead, binary=True),
                humanfriendly.format_size(merge_memory_bytes, binary=True),
                fan_in,
            )
            batches = external_sort_merge(
                context=context,
                input_frames=frames,
                sort_key=_SORT_KEY_COL,
                external_sort_dir=external_sort_dir,
                fan_in=fan_in,
                max_merge_fan_in=_EXTERNAL_SORT_MAX_MERGE_FAN_IN,
                shard=self._target_shard,
            )
        else:
            logger.info(
                "[shard %d] Merging %d chunks in memory (%s memory needed < %s memory available)",
                self._target_shard,
                self.total_chunks,
                humanfriendly.format_size(estimated_merge_memory_bytes * overhead, binary=True),
                humanfriendly.format_size(merge_memory_bytes, binary=True),
            )
            batches = (batch.to_pyarrow() for batch in merged_frame(frames, _SORT_KEY_COL).execute_stream())

        for batch in batches:
            yield from _table_to_items(batch)


# ---------------------------------------------------------------------------
# Scatter writer
# ---------------------------------------------------------------------------


def _apply_combiner(buffer: list, key_fn: Callable, combiner_fn: Callable) -> list:
    """Group buffer by key and reduce locally."""
    by_key: dict[object, list] = defaultdict(list)
    with log_time(f"Applying combiner to buffer of size {len(buffer)}", level=logging.DEBUG):
        for item in buffer:
            by_key[key_fn(item)].append(item)
        combined: list = []
        for key, items in by_key.items():
            combined.extend(combiner_fn(key, iter(items)))
    return combined


class ScatterWriter:
    """Write sorted, multiplexed scatter chunks and their reducer sidecar.

    Input tables contain payload, target-shard, and sort-key columns. The writer
    buffers them against the shard task's memory budget and writes target-local
    Parquet row groups for reducer predicate pushdown.
    """

    def __init__(
        self,
        data_path: str,
        key_fn: Callable,
        source_shard: int,
        sort_fn: Callable | None = None,
        combiner_fn: Callable | None = None,
    ) -> None:
        self._data_path = data_path if data_path.endswith("/") else f"{data_path}/"
        self._key_fn = key_fn
        self._sort_fn = sort_fn

        self._source_shard = source_shard
        self._combiner_fn = combiner_fn
        self._memory_available_bytes = _task_memory_bytes()
        self._flush_threshold_bytes = int(self._memory_available_bytes * _SCATTER_FLUSH_MEMORY_FRACTION)

        # Buffered Arrow tables, combined into one file per flush.
        self._tables: list[pa.Table] = []
        self._chunk_paths: list[str] = []
        # Payload bytes written per target shard, recorded in the sidecar so
        # reducers know their shard's exact data size for the external-sort
        # decision without opening any chunk files.
        self._shard_bytes: defaultdict[int, int] = defaultdict(int)
        self._avg_item_bytes: float = 0.0
        self._total_bytes_written: int = 0
        self._total_rows_written: int = 0
        self._n_chunks_written = 0
        # Throttles the per-flush progress log so high-fanout workloads don't log too often
        self._progress_log_limiter = RateLimiter(interval_seconds=_PROGRESS_LOG_INTERVAL_SECONDS)
        # Running Arrow buffer-size total of unflushed tables; reset to 0 on flush.
        self._buffer_estimated_bytes: int = 0

        ensure_parent_dir(self._data_path)
        self._result: ListShard | None = None

    def _flush(self) -> None:
        """Flush the accumulated buffer into one combined Parquet file sorted by [_SHARD_COL, _SORT_KEY_COL]."""
        if not self._tables:
            return

        buffer = pa.concat_tables(self._tables, promote_options="permissive")
        self._tables = []
        self._buffer_estimated_bytes = 0

        if self._combiner_fn is not None:
            by_shard: defaultdict[int, list[Any]] = defaultdict(list)
            for shard_val, payload in zip(
                buffer[_SHARD_COL].to_pylist(),
                buffer[_PAYLOAD_COL].to_pylist(),
                strict=True,
            ):
                by_shard[int(shard_val)].append(cloudpickle.loads(payload))

            tables: list[pa.Table] = []
            for shard_val, rows in by_shard.items():
                rows = _apply_combiner(rows, self._key_fn, self._combiner_fn)
                if not rows:
                    continue
                table = _items_to_table(rows, self._key_fn, self._sort_fn, num_output_shards=0)
                shard_index = table.schema.get_field_index(_SHARD_COL)
                table = table.set_column(
                    shard_index,
                    _SHARD_COL,
                    pa.array([shard_val] * len(table), type=pa.int32()),
                )
                tables.append(table)
            if not tables:
                return
            buffer = pa.concat_tables(tables, promote_options="permissive")

        sort_context = datafusion_context(memory_limit_bytes=self._memory_available_bytes)
        buffer_sorted = sort_context.from_arrow(buffer).sort(_SHARD_COL, _SORT_KEY_COL).to_arrow_table()
        del buffer

        flushed_bytes = int(buffer_sorted.nbytes)
        self._total_bytes_written += flushed_bytes
        self._total_rows_written += len(buffer_sorted)
        shard_sizes = (
            pa.table(
                {
                    _SHARD_COL: buffer_sorted[_SHARD_COL],
                    _PAYLOAD_BYTES_COL: pc.binary_length(buffer_sorted[_PAYLOAD_COL]),
                }
            )
            .group_by(_SHARD_COL, use_threads=False)
            .aggregate([(_PAYLOAD_BYTES_COL, "sum")])
        )
        for row in shard_sizes.to_pylist():
            self._shard_bytes[int(row[_SHARD_COL])] += int(row[_PAYLOAD_BYTES_SUM_COL])

        # Keep target shards local to as few row groups as practical so DataFusion
        # predicate pushdown can skip unrelated data. Cap the group count because
        # every reducer must read every chunk footer before applying that filter.
        num_targets = int(pc.count_distinct(buffer_sorted[_SHARD_COL]).as_py())
        num_row_groups = min(num_targets, _SCATTER_MAX_ROW_GROUPS_PER_CHUNK)
        row_group_size = max(1, math.ceil(len(buffer_sorted) / num_row_groups))
        chunk_path = f"{self._data_path}c{self._n_chunks_written:04d}.parquet"
        # Buffer through fsspec because direct native GCS writes have occasionally
        # failed with an uninformative Arrow error.
        buf = io.BytesIO()
        pq.write_table(buffer_sorted, buf, compression="zstd", row_group_size=row_group_size)
        with open_url(chunk_path, "wb") as f:
            f.write(buf.getvalue())

        self._chunk_paths.append(chunk_path)
        self._n_chunks_written += 1

        if self._progress_log_limiter.should_run():
            logger.info(
                "[shard %d] Wrote %d scatter chunks so far (latest chunk size: %d items, %d targets)",
                self._source_shard,
                self._n_chunks_written,
                len(buffer_sorted),
                num_targets,
            )

        del buffer_sorted
        if flushed_bytes >= _GC_FLUSH_SIZE_THRESHOLD_BYTES:
            gc.collect()

    def write(self, table: pa.Table) -> None:
        """Buffer an Arrow table, flushing on memory pressure.

        The table must contain ``_PAYLOAD_COL``, ``_SHARD_COL`` (int32), and
        ``_SORT_KEY_COL`` columns as produced by ``_items_to_table``.
        """
        if len(table) == 0:
            return

        self._tables.append(table)
        self._buffer_estimated_bytes += int(table.nbytes)

        if self._buffer_estimated_bytes > self._flush_threshold_bytes:
            logger.info(
                "[shard %d] Buffer estimated at %s (threshold %s); flushing scatter buffers",
                self._source_shard,
                humanfriendly.format_size(self._buffer_estimated_bytes, binary=True),
                humanfriendly.format_size(self._flush_threshold_bytes, binary=True),
            )
            self._flush()

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar, return ListShard.

        Idempotent: the committed result is cached, so a second call (e.g. the
        context-manager exit after an explicit ``close()``) is a no-op.
        """
        if self._result is not None:
            return self._result
        pre_close_flushes = self._n_chunks_written
        with log_time(f"Flushing remaining buffer for {self._data_path}"):
            self._flush()

        self._avg_item_bytes = (
            self._total_bytes_written / self._total_rows_written if self._total_rows_written > 0 else 0.0
        )

        logger.info(
            "[shard %d] scatter write done: %d pre-close flushes + %d at close = %d total; avg_item_bytes=%.0f B",
            self._source_shard,
            pre_close_flushes,
            self._n_chunks_written - pre_close_flushes,
            self._n_chunks_written,
            self._avg_item_bytes,
        )

        with log_time(f"Writing scatter meta for {self._data_path}"):
            _Sidecar(
                path=self._data_path,
                files=list(self._chunk_paths),
                avg_item_bytes=round(self._avg_item_bytes, 1),
                shard_bytes=dict(self._shard_bytes),
            ).write()

        self._result = ListShard(refs=[MemChunk(items=[self._data_path])])
        return self._result

    def cleanup(self) -> None:
        """Discard the output instead of committing it."""
        try:
            path = StoragePath(self._data_path)
            if path.exists():
                path.rmtree()
        except Exception as e:
            logger.warning(f"Failed to cleanup scatter directory {self._data_path}: {e}")

    def __enter__(self) -> "ScatterWriter":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: Any) -> None:
        if exc_type is not None:
            self.cleanup()
            return
        try:
            self.close()
        except BaseException:
            # A failure while committing (e.g. a dead R2 connection) would
            # otherwise leave the multipart upload open. Discard, then re-raise.
            self.cleanup()
            raise


def _write_scatter(
    items: Iterator,
    source_shard: int,
    data_path: str,
    key_fn: Callable,
    num_output_shards: int,
    sort_fn: Callable | None = None,
    combiner_fn: Callable | None = None,
) -> ListShard:
    """Write items as sorted Parquet chunks routed across target shards.

    Returns:
        A shard containing the scatter data path.
    """
    with ScatterWriter(
        data_path=data_path,
        key_fn=key_fn,
        source_shard=source_shard,
        sort_fn=sort_fn,
        combiner_fn=combiner_fn,
    ) as writer:
        pending: list[Any] = []
        for item in items:
            pending.append(item)
            if len(pending) >= _TABLE_ROW_COUNT:
                writer.write(_items_to_table(pending, key_fn, sort_fn, num_output_shards))
                pending.clear()
        if pending:
            writer.write(_items_to_table(pending, key_fn, sort_fn, num_output_shards))
        return writer.close()
