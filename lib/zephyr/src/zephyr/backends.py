# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend implementations for distributed execution."""

from __future__ import annotations

import logging
import os
import pickle
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

import msgspec
import numpy as np
import ray
import zstandard as zstd
from tqdm import tqdm

from zephyr.dataset import (
    BatchOp,
    Dataset,
    FilterOp,
    FlatMapOp,
    FusedMapOp,
    GroupByLocalOp,
    GroupByShuffleReduceOp,
    MapOp,
    ReshardOp,
    WriteJsonlOp,
    WriteParquetOp,
)

from .writers import write_jsonl_file, write_parquet_file

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def msgpack_encode(obj: Any) -> bytes:
    """Serialize with msgpack and compress with zstd."""
    serialized = msgspec.msgpack.encode(obj)
    cctx = zstd.ZstdCompressor(level=-10, threads=1)
    return cctx.compress(serialized)


def msgpack_decode(data: bytes) -> Any:
    """Decompress zstd and deserialize msgpack."""
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(data)
    return msgspec.msgpack.decode(decompressed)


@dataclass
class BackendConfig:
    """Configuration for backend creation.

    Attributes:
        backend_type: Type of backend (ray, threadpool, or sync)
        max_parallelism: Maximum number of concurrent tasks
        memory: Memory requirement per task in bytes
        num_cpus: Number of CPUs per task for Ray backend
        num_gpus: Number of GPUs per task for Ray backend
        chunk_size: Number of items per chunk in Shard
        ray_options: Additional Ray remote options
        dry_run: If True, show optimization plan without executing
    """

    backend_type: Literal["ray", "threadpool", "sync"]
    max_parallelism: int = 1000000
    chunk_size: int = 1000
    memory: int | None = None
    num_cpus: float | None = None
    num_gpus: float | None = None
    ray_options: dict = field(default_factory=dict)
    dry_run: bool = False


class ExecutionContext(Protocol):
    """Protocol for execution contexts that abstract put/put_many/get/run/wait primitives.

    This allows different backends (Ray, ThreadPool, Sync) to share the same
    Shard-based execution logic while using different execution strategies.
    """

    def put(self, obj: Any) -> Any:
        """Store an object and return a reference to it."""
        ...

    def put_many(self, objs: list[Any]) -> list[Any]:
        """Store multiple objects and return references to them."""
        ...

    def get(self, ref: Any) -> Any:
        """Retrieve an object from its reference."""
        ...

    def run(self, fn: Callable, *args) -> Any:
        """Execute a function with arguments and return a future.

        Args:
            fn: Function to execute
            *args: Arguments to pass to function

        Returns:
            Future representing the execution (type depends on context)
        """
        ...

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete.

        Args:
            futures: List of futures to wait on
            num_returns: Number of futures to wait for

        Returns:
            Tuple of (ready_futures, pending_futures)
        """
        ...


class _ImmediateFuture:
    """Wrapper for immediately available results to match Future interface."""

    def __init__(self, result: Any):
        self._result = result

    def result(self):
        return self._result


@ray.remote(enable_task_events=False)
def _put_on_worker(serialized: bytes) -> ray.ObjectRef:
    """Put serialized bytes in Ray's object store on a worker node.

    This is used to scatter data across the cluster with Ray.
    """
    return ray.put(serialized)


class RayContext:
    """Execution context using Ray for distributed execution."""

    def __init__(self, ray_options: dict | None = None):
        """Initialize Ray context.

        Args:
            ray_options: Options to pass to ray.remote() (e.g., memory, num_cpus, num_gpus)
        """
        self.ray_options = ray_options or {}

    def put(self, obj: Any) -> ray.ObjectRef:
        """Store a single object in Ray object store."""
        return self.put_many([obj])[0]

    def put_many(self, objs: list[Any]) -> list[ray.ObjectRef]:
        """Store multiple objects in Ray object store via remote workers."""
        if not objs:
            return []

        compressed_objs = [msgpack_encode(obj) for obj in objs]

        # Submit all put tasks asynchronously
        ref_futures = [
            _put_on_worker.options(scheduling_strategy="SPREAD", enable_task_events=False).remote(compressed)
            for compressed in compressed_objs
        ]

        # Wait for all puts to complete in one batch
        return ray.get(ref_futures)

    def get(self, ref: ray.ObjectRef) -> Any:
        """Retrieve and decode object from Ray object store."""
        result = ray.get(ref)
        if isinstance(result, bytes):
            return msgpack_decode(result)
        return result

    def run(self, fn: Callable, *args) -> ray.ObjectRef:
        """Execute function remotely with configured Ray options.

        Uses SPREAD scheduling strategy to distribute work across worker nodes.
        """
        if self.ray_options:
            remote_fn = ray.remote(**self.ray_options)(fn)
        else:
            remote_fn = ray.remote(fn)
        return remote_fn.options(scheduling_strategy="SPREAD").remote(*args)

    def wait(self, futures: list[ray.ObjectRef], num_returns: int = 1) -> tuple[list, list]:
        ready, pending = ray.wait(futures, num_returns=num_returns)
        return list(ready), list(pending)


class LocalContext:
    """Base class for local execution contexts (sync and thread-based)."""

    def put(self, obj: Any) -> Any:
        """No-op for local contexts - objects are already in memory."""
        return obj

    def put_many(self, objs: list[Any]) -> list[Any]:
        """No-op for local contexts - objects are already in memory."""
        return objs

    def get(self, ref: Any) -> Any:
        """Get result, unwrapping Future/_ImmediateFuture if needed."""
        if isinstance(ref, (Future, _ImmediateFuture)):
            return ref.result()
        return ref


class SyncContext(LocalContext):
    """Execution context for synchronous (single-threaded) execution."""

    def run(self, fn: Callable, *args) -> _ImmediateFuture:
        """Execute function immediately and wrap result."""
        fn_copy = pickle.loads(pickle.dumps(fn))
        result = fn_copy(*args)
        return _ImmediateFuture(result)

    def wait(self, futures: list[_ImmediateFuture], num_returns: int = 1) -> tuple[list, list]:
        """All futures are immediately ready."""
        return futures[:num_returns], futures[num_returns:]


class ThreadContext(LocalContext):
    """Execution context using ThreadPoolExecutor for parallel execution."""

    def __init__(self, max_workers: int):
        """Initialize thread pool context.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(self, fn: Callable, *args) -> Future:
        """Submit function to thread pool."""
        fn_copy = pickle.loads(pickle.dumps(fn))
        return self.executor.submit(fn_copy, *args)

    def wait(self, futures: list[Future], num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete."""
        if num_returns >= len(futures):
            done, pending = wait(futures, return_when="ALL_COMPLETED")
        else:
            done, pending = wait(futures, return_when="FIRST_COMPLETED")

        done_list = list(done)[:num_returns]
        pending_list = list(pending) + done_list[num_returns:]
        return done_list, pending_list


@dataclass
class Chunk:
    """A single chunk of data with count metadata."""

    count: int
    data: Any  # The actual ref (ObjectRef, Future, or object)


@dataclass
class Shard:
    """Container for a logically grouped set of items split across multiple chunks."""

    idx: int  # Shard index (e.g., 0 of 50)
    chunks: list[Chunk]
    context: ExecutionContext

    @property
    def count(self) -> int:
        """Total number of items across all chunks."""
        return sum(c.count for c in self.chunks)

    def iter_chunks(self) -> Iterator[list]:
        """Iterate over chunks (each chunk is a list of items)."""
        for chunk in self.chunks:
            yield self.context.get(chunk.data)

    def __iter__(self):
        """Flat map over all chunks."""
        for chunk_data in self.iter_chunks():
            yield from chunk_data

    @staticmethod
    def from_items(items: Iterable, chunk_size: int, context: ExecutionContext, idx: int = 0) -> Shard:
        """Create a Shard from items by chunking and storing via context.

        Args:
            items: Items to chunk and store
            chunk_size: Number of items per chunk
            context: Execution context for put/get operations
            idx: Shard index (default 0)

        Returns:
            Shard containing chunked refs
        """
        chunks = []
        chunk = []
        for item in items:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                ref = context.put(chunk)
                chunks.append(Chunk(count=len(chunk), data=ref))
                chunk = []
        if chunk:
            ref = context.put(chunk)
            chunks.append(Chunk(count=len(chunk), data=ref))
        return Shard(idx=idx, chunks=chunks, context=context)

    @staticmethod
    def from_single_ref(ref: Any, context: ExecutionContext, idx: int, count: int) -> Shard:
        """Wrap a single ref as a Shard.

        Args:
            ref: Reference to wrap (type depends on context)
            context: Execution context for get operations
            idx: Shard index
            count: Number of items in the ref

        Returns:
            Shard containing the single ref
        """
        return Shard(idx=idx, chunks=[Chunk(count=count, data=ref)], context=context)


def make_batches(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Batch items into groups of batch_size.

    Args:
        items: Items to batch
        batch_size: Maximum size of each batch

    Yields:
        Batches of items (last batch may be smaller than batch_size)
    """
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def format_shard_path(pattern: str, shard_idx: int, total: int) -> str:
    """Format output path with shard information.

    Args:
        pattern: Path pattern with {shard}, {total}, {basename} placeholders
        shard_idx: Index of this shard
        total: Total number of shards

    Returns:
        Formatted path with double slashes normalized

    Raises:
        ValueError: If multiple shards will write to the same file (pattern missing {shard})
    """
    import re

    if total > 1 and "{shard" not in pattern:
        raise ValueError(
            f"Output pattern must contain '{{shard}}' placeholder when writing {total} shards. Got pattern: {pattern}"
        )

    basename = f"shard_{shard_idx}"
    formatted = pattern.format(shard=shard_idx, total=total, basename=basename)

    # Normalize double slashes while preserving protocol (e.g., gs://, s3://, http://)
    normalized = re.sub(r"(?<!:)//+", "/", formatted)

    return normalized


def process_shard_flat_map(shard: Shard, shard_idx: int, total: int, fn: Callable, chunk_size: int) -> list[Shard]:
    """Process shard with flat_map, returning chunked results.

    Supports callable classes for stateful operations - each shard gets a fresh
    instance via serialization (pickle/ray).
    """
    logger.info(f"flat_map: shard {shard_idx + 1}/{total}")

    def result_generator():
        for item in shard:
            yield from fn(item)

    return [
        Shard.from_items(
            tqdm(result_generator(), desc=f"flat_map shard {shard_idx + 1}/{total}", mininterval=10),
            chunk_size,
            shard.context,
            idx=shard_idx,
        )
    ]


def process_shard_map(shard: Shard, shard_idx: int, total: int, fn: Callable, chunk_size: int) -> list[Shard]:
    """Process shard with map, returning chunked results."""
    logger.info(f"map: shard {shard_idx + 1}/{total}")

    def result_generator():
        for item in shard:
            yield fn(item)

    return [
        Shard.from_items(
            tqdm(result_generator(), desc=f"map shard {shard_idx + 1}/{total}", mininterval=10),
            chunk_size,
            shard.context,
            idx=shard_idx,
        )
    ]


def process_shard_filter(shard: Shard, shard_idx: int, total: int, predicate: Callable, chunk_size: int) -> list[Shard]:
    """Process shard with filter, returning chunked results."""
    logger.info(f"filter: shard {shard_idx + 1}/{total}")

    def result_generator():
        for item in shard:
            if predicate(item):
                yield item

    return [
        Shard.from_items(
            tqdm(result_generator(), desc=f"filter shard {shard_idx + 1}/{total}", mininterval=10),
            chunk_size,
            shard.context,
            idx=shard_idx,
        )
    ]


def process_shard_batch(shard: Shard, shard_idx: int, total: int, batch_size: int, chunk_size: int) -> list[Shard]:
    """Process shard with batching, returning chunked results."""

    def result_generator():
        yield from make_batches(shard, batch_size)

    return [
        Shard.from_items(
            tqdm(result_generator(), desc=f"batch shard {shard_idx + 1}/{total}", mininterval=10),
            chunk_size,
            shard.context,
            idx=shard_idx,
        )
    ]


def process_shard_write_jsonl(
    shard: Shard, shard_idx: int, total: int, output_pattern: str, chunk_size: int
) -> list[Shard]:
    """Write shard to JSONL file and return Shard containing the output filename."""
    logger.info(f"write_jsonl: shard {shard_idx + 1}/{total}")
    output_path = format_shard_path(output_pattern, shard_idx, total)

    result = write_jsonl_file(shard, output_path)["path"]
    logger.info(f"write_jsonl: shard {shard_idx + 1}/{total} → {output_path}")
    return [Shard.from_items([result], chunk_size, shard.context, idx=shard_idx)]


def process_shard_write_parquet(
    shard: Shard,
    shard_idx: int,
    total: int,
    output_pattern: str,
    schema: object | None,
    batch_size: int,
    chunk_size: int,
) -> list[Shard]:
    """Write shard to Parquet file and return Shard containing the output filename."""
    logger.info(f"write_parquet: shard {shard_idx + 1}/{total}")
    output_path = format_shard_path(output_pattern, shard_idx, total)

    result = write_parquet_file(shard, output_path, schema, batch_size)["path"]
    logger.info(f"write_parquet: shard {shard_idx + 1}/{total} → {output_path}")
    return [Shard.from_items([result], chunk_size, shard.context, idx=shard_idx)]


def process_shard_fused(
    shard: Shard, shard_idx: int, total_shards: int, operations: list, chunk_size: int
) -> list[Shard]:
    """Process shard with fused operations pipeline.

    Chains operations into a single pass over the data, avoiding intermediate
    materialization in the object store.

    Args:
        shard: Input shard
        shard_idx: Index of this shard
        total: Total number of shards
        operations: List of operations to apply in sequence
        chunk_size: Number of items per chunk in output

    Returns:
        List of output shards (single shard for most ops, multiple for GroupByLocalOp)
    """
    op_names = "|".join(op.__class__.__name__.replace("Op", "") for op in operations)
    logger.info(f"fused[{len(operations)}]: shard {shard_idx + 1}/{total_shards}")

    def build_stream(stream_input, ops):
        """Recursively build the generator pipeline."""
        if not ops:
            yield from stream_input
            return

        op, *rest = ops

        if isinstance(op, MapOp):
            yield from build_stream((op.fn(item) for item in stream_input), rest)
        elif isinstance(op, FlatMapOp):
            yield from build_stream((result_item for item in stream_input for result_item in op.fn(item)), rest)
        elif isinstance(op, FilterOp):
            yield from build_stream((item for item in stream_input if op.predicate(item)), rest)
        elif isinstance(op, BatchOp):
            yield from build_stream(make_batches(stream_input, op.batch_size), rest)
        elif isinstance(op, WriteJsonlOp):
            output_path = format_shard_path(op.output_pattern, shard_idx, total_shards)
            result = write_jsonl_file(stream_input, output_path)["path"]
            yield from build_stream(iter([result]), rest)
        elif isinstance(op, WriteParquetOp):
            output_path = format_shard_path(op.output_pattern, shard_idx, total_shards)
            result = write_parquet_file(stream_input, output_path, op.schema, op.batch_size)["path"]
            yield from build_stream(iter([result]), rest)

    if operations and isinstance(operations[-1], GroupByLocalOp):
        group_by_local_op = operations[-1]
        pre_ops = operations[:-1]
        num_output_shards = group_by_local_op.num_output_shards

        # Handle sentinel value (-1) - use total as default
        if num_output_shards <= 0:
            num_output_shards = total_shards

        # Build streaming pipeline through fused ops, then group
        stream = tqdm(
            build_stream(shard, pre_ops),
            desc=f"fused[{op_names}]: shard {shard_idx + 1}/{total_shards}",
            mininterval=10,
        )

        output_chunks = _group_items_by_hash(
            stream, group_by_local_op.key_fn, num_output_shards, chunk_size, shard.context
        )

        return [Shard(idx=idx, chunks=output_chunks[idx], context=shard.context) for idx in range(num_output_shards)]

    return [
        Shard.from_items(
            tqdm(
                build_stream(shard, operations),
                desc=f"fused[{op_names}]: shard {shard_idx + 1}/{total_shards}",
                mininterval=10,
            ),
            chunk_size,
            shard.context,
            idx=shard_idx,
        )
    ]


def deterministic_hash(obj: object) -> int:
    s = msgspec.msgpack.encode(obj, order="deterministic")
    return zlib.adler32(s)


def _group_items_by_hash(
    items: Iterable,
    key_fn: Callable,
    num_output_shards: int,
    chunk_size: int,
    context: ExecutionContext,
) -> dict[int, list[Chunk]]:
    """Core grouping logic: hash items and distribute into chunked buckets.

    Items within each chunk are sorted by key to enable streaming reduction.

    Args:
        items: Items to group
        key_fn: Function to extract grouping key from item
        num_output_shards: Number of output shards to distribute across
        chunk_size: Number of items per chunk
        context: Execution context for storing chunks

    Returns:
        Dict mapping shard index to list of chunks for that shard
    """
    output_chunks = defaultdict(list)
    output_tmp = defaultdict(list)

    for item in items:
        key = key_fn(item)
        target_shard = deterministic_hash(key) % num_output_shards
        output_tmp[target_shard].append(item)
        if len(output_tmp[target_shard]) >= chunk_size:
            # Sort items by key before storing chunk
            sorted_items = sorted(output_tmp[target_shard], key=key_fn)
            output_chunks[target_shard].append(Chunk(count=len(sorted_items), data=context.put(sorted_items)))
            output_tmp[target_shard] = []

    # Batch put all remaining chunks together
    flush_data = [(target_shard, sorted(items, key=key_fn)) for target_shard, items in output_tmp.items() if items]
    if flush_data:
        sorted_items_list = [sorted_items for _, sorted_items in flush_data]
        refs = context.put_many(sorted_items_list)
        for (target_shard, sorted_items), ref in zip(flush_data, refs, strict=True):
            output_chunks[target_shard].append(Chunk(count=len(sorted_items), data=ref))

    return output_chunks


def _merge_sorted_chunks(shard: Shard, key_fn: Callable) -> Iterator[tuple[object, Iterator]]:
    """Merge sorted chunks using k-way merge, yielding (key, items_iterator) groups.

    Each chunk is assumed to be sorted by key. This function performs a k-way merge
    across all chunks and groups consecutive items with the same key.

    Args:
        shard: Shard containing sorted chunks
        key_fn: Function to extract key from item

    Yields:
        Tuples of (key, iterator_of_items) for each unique key
    """
    import heapq
    from itertools import groupby

    # Create iterators for each chunk
    chunk_iterators = []
    for chunk_data in shard.iter_chunks():
        chunk_iterators.append(iter(chunk_data))

    # Use heapq.merge to k-way merge sorted streams
    # heapq.merge requires a key function that extracts the sort key
    merged_stream = heapq.merge(*chunk_iterators, key=key_fn)

    # Group consecutive items by key
    for key, group_iter in groupby(merged_stream, key=key_fn):  # noqa: UP028
        yield key, group_iter


def process_shard_group_by_local(
    shard: Shard, shard_idx: int, total: int, key_fn: Callable, num_output_shards: int, chunk_size: int
) -> list[Shard]:
    """Phase 1: Local grouping per shard, distributing by hash(key) % num_output_shards."""

    logger.info(f"group_by_local: shard {shard_idx + 1}/{total}")

    items = tqdm(shard, desc=f"group_by_local shard {shard_idx + 1}/{total}", mininterval=10)
    output_chunks = _group_items_by_hash(items, key_fn, num_output_shards, chunk_size, shard.context)

    return [Shard(idx=idx, chunks=output_chunks[idx], context=shard.context) for idx in range(num_output_shards)]


def process_shard_group_by_reduce(
    shard: Shard, shard_idx: int, total: int, key_fn: Callable, reducer_fn: Callable, chunk_size: int
) -> list[Shard]:
    """Phase 2: Global reduction per shard, applying reducer to each key group.

    Uses streaming k-way merge to avoid materializing all items for each key.
    Chunks are assumed to be sorted by key from Phase 1.

    Args:
        shard: Input shard (contains items for one output shard)
        shard_idx: Index of this output shard
        total: Total number of output shards
        key_fn: Function from item -> key
        reducer_fn: Function from (key, Iterator[items]) -> result
        chunk_size: Number of items per chunk in output

    Returns:
        List containing single output shard with reduced results
    """
    logger.info(f"group_by_reduce: shard {shard_idx + 1}/{total}")

    # Use streaming merge to group items by key without full materialization
    def result_generator():
        for key, items_iter in _merge_sorted_chunks(shard, key_fn):
            yield reducer_fn(key, items_iter)

    return [Shard.from_items(result_generator(), chunk_size, shard.context, idx=shard_idx)]


def reshard_refs(shards: list[Shard], num_shards: int) -> list[Shard]:
    """Redistribute chunks across target number of shards.

    Does not split chunks - simply reassigns chunks to new shards in round-robin fashion."""
    if not shards:
        return []

    context = shards[0].context
    all_chunks = [chunk for shard in shards for chunk in shard.chunks]

    if not all_chunks:
        return []

    chunk_groups = np.array_split(all_chunks, num_shards)  # type: ignore
    return [
        Shard(idx=idx, chunks=list(group), context=context) for idx, group in enumerate(chunk_groups) if len(group) > 0
    ]


def recompact_shards(shard_lists: list[list[Shard]]) -> list[Shard]:
    """Group a list of shard lists into a list of shards by shard index."""
    # Group chunks by output shard index
    chunks_by_idx = defaultdict(list)
    max_idx = -1

    for shard_list in shard_lists:
        for shard in shard_list:
            chunks_by_idx[shard.idx].extend(shard.chunks)
            max_idx = max(max_idx, shard.idx)

    if max_idx < 0:
        return []

    # Rebuild shards with all chunks for each index
    # Include all shard indices from 0 to max_idx, even if some are empty
    context = shard_lists[0][0].context if shard_lists[0] else None
    return [Shard(idx=idx, chunks=chunks_by_idx[idx], context=context) for idx in range(max_idx + 1)]


class Backend:
    """Base class for execution backends using Shard-based execution.

    All backends share the same execution logic and differ only in their
    ExecutionContext (Ray, ThreadPool, or Sync).
    """

    def __init__(self, context: ExecutionContext, config: BackendConfig):
        """Initialize backend with execution context and configuration.

        Args:
            context: Execution context providing put/get/run/wait primitives
            config: Backend configuration
        """
        self.context = context
        self.config = config
        self.dry_run = config.dry_run

    @property
    def chunk_size(self) -> int:
        return self.config.chunk_size

    def execute(self, dataset: Dataset) -> Iterator:
        """Execute a dataset and return an iterator over results.

        Args:
            dataset: Dataset to execute

        Returns:
            Iterator over dataset results
        """
        yield from self.execute_operations(dataset.source, dataset.operations)

    def execute_operations(self, source: Iterable, operations: list) -> Iterator:
        """Execute a chain of operations on a data source.

        Backend interprets the operation chain and can apply optimizations
        before execution.

        Args:
            source: Source data iterable
            operations: List of operation dataclasses (MapOp, FilterOp, BatchOp)

        Yields:
            Results after applying all operations
        """
        optimized_ops = self._optimize_operations(operations)
        self._print_optimization_plan(operations, optimized_ops)

        if self.dry_run:
            return

        yield from self._execute_optimized(source, optimized_ops)

    def _optimize_operations(self, operations: list) -> list:
        """Optimize operation chain by fusing consecutive operations.

        Fuses consecutive operations (MapOp, FlatMapOp, FilterOp, BatchOp, Write*Op)
        into a single FusedMapOp to avoid materializing intermediate results.
        ReshardOp and GroupByShuffleReduceOp break fusion as they change parallelism
        structure or require a shuffle boundary.

        Args:
            operations: Original operation chain

        Returns:
            Optimized operation chain with fused operations
        """
        if not operations:
            return operations

        optimized = []
        fusible_buffer = []

        def is_fusible(op):
            """Check if operation can be fused.

            Fusible operations can be chained together in a single pass over the data.
            Non-fusible operations require a shuffle boundary.

            Non-fusible: ReshardOp (just reshuffles refs), GroupByShuffleReduceOp (shuffle boundary).
            Fusible: MapOp, FlatMapOp, FilterOp, BatchOp, Write*Op, GroupByLocalOp.
            """
            return not isinstance(op, (ReshardOp, GroupByShuffleReduceOp))

        def flush_buffer():
            """Flush accumulated fusible operations."""
            if not fusible_buffer:
                return

            # don't create a fused op for a single operation
            if len(fusible_buffer) == 1:
                optimized.append(fusible_buffer[0])
            else:
                optimized.append(FusedMapOp(operations=fusible_buffer[:]))

            fusible_buffer.clear()

        for op in operations:
            if is_fusible(op):
                fusible_buffer.append(op)
            else:
                flush_buffer()
                optimized.append(op)

        flush_buffer()
        return optimized

    def _print_optimization_plan(self, original_ops: list, optimized_ops: list) -> None:
        """Print the optimization plan showing how operations are fused.

        Args:
            original_ops: Original operation chain
            optimized_ops: Optimized operation chain with fused operations
        """
        print("\n=== ML-Flow Optimization Plan ===\n")
        print(f"Original operations: {len(original_ops)}")
        print(f"Optimized operations: {len(optimized_ops)}")
        print(f"Fusion savings: {len(original_ops) - len(optimized_ops)} operations fused\n")

        print("Original pipeline:")
        for i, op in enumerate(original_ops, 1):
            op_name = op.__class__.__name__.replace("Op", "")
            print(f"  {i}. {op_name}")

        print("\nOptimized pipeline:")
        for i, op in enumerate(optimized_ops, 1):
            if isinstance(op, FusedMapOp):
                fused_names = [fused_op.__class__.__name__.replace("Op", "") for fused_op in op.operations]
                print(f"  {i}. FusedMap[{' → '.join(fused_names)}] ({len(op.operations)} operations)")
            else:
                op_name = op.__class__.__name__.replace("Op", "")
                print(f"  {i}. {op_name}")

        print("\n=== End Optimization Plan ===\n")

    def _execute_optimized(self, source: Iterable, operations: list) -> Iterator:
        """Execute operations using chunked shards.

        Each input item becomes a Shard (initially with one chunk). Operations produce
        Shards with chunked results to prevent large object overhead.

        Args:
            source: Source data iterable
            operations: Operation chain

        Yields:
            Results after applying all operations
        """
        # Convert source items to Shards (one item per shard initially)
        # Batch all puts together to avoid sequential round-trips
        source_items = [[item] for item in source]
        refs = self.context.put_many(source_items)
        shards = [Shard.from_single_ref(ref, self.context, idx=idx, count=1) for idx, ref in enumerate(refs)]

        for op in operations:
            if isinstance(op, FlatMapOp):
                shards = self._execute_on_shards(process_shard_flat_map, (op.fn, self.chunk_size), shards)
            elif isinstance(op, MapOp):
                shards = self._execute_on_shards(process_shard_map, (op.fn, self.chunk_size), shards)
            elif isinstance(op, FilterOp):
                shards = self._execute_on_shards(process_shard_filter, (op.predicate, self.chunk_size), shards)
            elif isinstance(op, FusedMapOp):
                shards = self._execute_on_shards(process_shard_fused, (op.operations, self.chunk_size), shards)
            elif isinstance(op, BatchOp):
                shards = self._execute_on_shards(process_shard_batch, (op.batch_size, self.chunk_size), shards)
            elif isinstance(op, GroupByLocalOp):
                # Auto-detect num_output_shards if using sentinel value
                num_output_shards = op.num_output_shards if op.num_output_shards > 0 else len(shards)
                shards = self._execute_on_shards(
                    process_shard_group_by_local, (op.key_fn, num_output_shards, self.chunk_size), shards
                )
            elif isinstance(op, GroupByShuffleReduceOp):
                shards = self._execute_on_shards(
                    process_shard_group_by_reduce, (op.key_fn, op.reducer_fn, self.chunk_size), shards
                )
            elif isinstance(op, ReshardOp):
                shards = reshard_refs(shards, op.num_shards)
            elif isinstance(op, WriteJsonlOp):
                shards = self._execute_on_shards(process_shard_write_jsonl, (op.output_pattern, self.chunk_size), shards)
            elif isinstance(op, WriteParquetOp):
                shards = self._execute_on_shards(
                    process_shard_write_parquet, (op.output_pattern, op.schema, op.batch_size, self.chunk_size), shards
                )

        op_names = [op.__class__.__name__.replace("Op", "") for op in operations]
        desc = f"Materialize [{' → '.join(op_names)}]"

        def materialize_all():
            for shard in shards:
                yield from shard

        yield from tqdm(materialize_all(), desc=desc, unit="items")

    def _execute_on_shards(self, process_fn: Callable, fn_args: tuple, shards: list[Shard]) -> list[Shard]:
        """Execute a processing function on shards, handling both 1:1 and 1:N operations.

        All process functions now return list[Shard], making the interface uniform.
        After execution, results are recompacted by shard index.

        Args:
            process_fn: Function that takes (Shard, int, int, *fn_args) -> list[Shard]
            fn_args: Additional arguments to pass to process_fn
            shards: List of input Shards

        Returns:
            Recompacted list of output Shards
        """
        print(
            f"Running {process_fn.__name__} on {len(shards)} shards with max parallelism {self.config.max_parallelism}"
        )
        total = len(shards)
        result_futures = [None] * total
        pending = {}

        for shard_idx, shard in enumerate(shards):
            future = self.context.run(process_fn, shard, shard_idx, total, *fn_args)
            pending[future] = shard_idx

            if len(pending) >= self.config.max_parallelism:
                ready, _ = self.context.wait(list(pending.keys()), num_returns=1)
                ready_future = ready[0]
                result_futures[pending[ready_future]] = ready_future
                del pending[ready_future]

        for future, idx in pending.items():
            result_futures[idx] = future

        self.context.wait(result_futures, num_returns=len(result_futures))

        # Recompact into final output shards
        shard_lists = [self.context.get(future) for future in result_futures]
        return recompact_shards(shard_lists)


class RayBackend(Backend):
    """Ray-based distributed execution backend.

    Uses Ray remote tasks with bounded parallelism to prevent OOM.

    Example:
        >>> from zephyr import create_backend, Dataset
        >>> backend = create_backend("ray", max_parallelism=10, memory="2GB")
        >>> ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
        >>> results = list(backend.execute(ds))
    """

    def __init__(self, config: BackendConfig):
        # Build ray_options dict
        options = {}
        if config.memory is not None:
            options["memory"] = config.memory
        if config.num_cpus is not None:
            options["num_cpus"] = config.num_cpus
        if config.num_gpus is not None:
            options["num_gpus"] = config.num_gpus
        options.update(config.ray_options)

        # Create Ray context and initialize base backend
        context = RayContext(ray_options=options)
        super().__init__(context, config)


class ThreadPoolBackend(Backend):
    """ThreadPoolExecutor-based backend for I/O-bound parallelism.

    Example:
        >>> from zephyr import create_backend, Dataset
        >>> backend = create_backend("threadpool", max_parallelism=4)
        >>> ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
        >>> results = list(backend.execute(ds))
    """

    def __init__(self, config: BackendConfig):
        context = ThreadContext(max_workers=min(config.max_parallelism, os.cpu_count()))
        super().__init__(context, config)


class SyncBackend(Backend):
    """Synchronous backend for testing and debugging.

    Executes all operations sequentially in the current thread.

    Example:
        >>> from zephyr import create_backend, Dataset
        >>> backend = create_backend("sync")
        >>> ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
        >>> results = list(backend.execute(ds))
    """

    def __init__(self, config: BackendConfig):
        context = SyncContext()
        super().__init__(context, config)
