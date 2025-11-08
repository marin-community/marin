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
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

import msgspec
import numpy as np
import ray
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
    MapShardOp,
    ReshardOp,
    WriteJsonlOp,
    WriteParquetOp,
)

from .writers import ensure_parent_dir, write_jsonl_file

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


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
    """Protocol for execution contexts that abstract put/get/run/wait primitives.

    This allows different backends (Ray, ThreadPool, Sync) to share the same
    Shard-based execution logic while using different execution strategies.
    """

    def put(self, obj: Any) -> Any:
        """Store an object and return a reference to it.

        Args:
            obj: Object to store

        Returns:
            Reference to the stored object (type depends on context)
        """
        ...

    def get(self, ref: Any) -> Any:
        """Retrieve an object from its reference.

        Args:
            ref: Reference to retrieve

        Returns:
            The stored object
        """
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


@ray.remote
def _put_on_worker(serialized: bytes) -> ray.ObjectRef:
    """Put serialized bytes in Ray's object store on a worker node.

    This function is executed remotely to ensure objects are placed on worker
    nodes rather than the head node.
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
        """Store object on a worker node using SPREAD strategy.

        Uses a remote task to place the object on a worker node's object store
        rather than the head node, distributing data across the cluster.
        Serializes data using msgpack for efficiency.
        """
        # Serialize with msgpack
        serialized = msgspec.msgpack.encode(obj)
        # Submit task to worker with SPREAD strategy to distribute across nodes
        ref_future = _put_on_worker.options(scheduling_strategy="SPREAD").remote(serialized)
        # Get the ObjectRef that the worker created
        return ray.get(ref_future)

    def get(self, ref: ray.ObjectRef) -> Any:
        """Retrieve object, decoding msgpack if applicable.

        Handles both msgpack-encoded chunk data and Ray-pickled function results.
        """
        result = ray.get(ref)
        # If it's bytes, it's msgpack-encoded chunk data
        if isinstance(result, bytes):
            return msgspec.msgpack.decode(result)
        # Otherwise it's a Ray-pickled object (e.g., Shard from remote function)
        return result

    def run(self, fn: Callable, *args) -> ray.ObjectRef:
        """Execute function remotely with configured Ray options.

        Uses SPREAD scheduling strategy to avoid running on head node and
        distribute work across worker nodes.
        """
        if self.ray_options:
            remote_fn = ray.remote(**self.ray_options)(fn)
        else:
            remote_fn = ray.remote(fn)
        return remote_fn.options(scheduling_strategy="SPREAD").remote(*args)

    def wait(self, futures: list[ray.ObjectRef], num_returns: int = 1) -> tuple[list, list]:
        ready, pending = ray.wait(futures, num_returns=num_returns)
        return list(ready), list(pending)


class SyncContext:
    """Execution context for synchronous (single-threaded) execution."""

    def put(self, obj: Any) -> Any:
        """Identity operation - no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, unwrapping _ImmediateFuture if needed."""
        if isinstance(ref, _ImmediateFuture):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> _ImmediateFuture:
        """Execute function immediately and wrap result."""
        result = fn(*args)
        return _ImmediateFuture(result)

    def wait(self, futures: list[_ImmediateFuture], num_returns: int = 1) -> tuple[list, list]:
        """All futures are immediately ready."""
        return futures[:num_returns], futures[num_returns:]


class ThreadContext:
    """Execution context using ThreadPoolExecutor for parallel execution."""

    def __init__(self, max_workers: int):
        """Initialize thread pool context.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def put(self, obj: Any) -> Any:
        """Identity operation - in-process, no serialization needed."""
        return obj

    def get(self, ref: Any) -> Any:
        """Get result, handling both Future objects and plain values."""
        if isinstance(ref, Future):
            return ref.result()
        return ref

    def run(self, fn: Callable, *args) -> Future:
        """Submit function to thread pool."""
        return self.executor.submit(fn, *args)

    def wait(self, futures: list[Future], num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete."""
        if num_returns >= len(futures):
            # Wait for all
            done, pending = wait(futures, return_when="ALL_COMPLETED")
        else:
            # Wait for first to complete
            done, pending = wait(futures, return_when="FIRST_COMPLETED")

        # Return requested number of completed futures
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
    """Container for a logically grouped set of items split across multiple chunks.

    Each chunk contains a portion of items to prevent large object overhead.
    Works with any ExecutionContext (Ray, ThreadPool, Sync) via context.put/get operations.
    """

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
    def from_single_ref(ref: Any, context: ExecutionContext, idx: int = 0, count: int | None = None) -> Shard:
        """Wrap a single ref as a Shard.

        Args:
            ref: Reference to wrap (type depends on context)
            context: Execution context for get operations
            idx: Shard index (default 0)
            count: Number of items in the ref (if known, otherwise will be computed lazily)

        Returns:
            Shard containing the single ref
        """
        if count is None:
            # Need to get the ref to count items
            data = context.get(ref)
            count = len(data) if hasattr(data, "__len__") else sum(1 for _ in data)
            ref = context.put(data) if not hasattr(data, "__len__") else ref

        return Shard(idx=idx, chunks=[Chunk(count=count, data=ref)], context=context)


def infer_parquet_type(value):
    """Recursively infer PyArrow type from a Python value.

    Args:
        value: Python value

    Returns:
        PyArrow type
    """
    import pyarrow as pa

    if isinstance(value, bool):
        # Check bool before int since bool is a subclass of int
        return pa.bool_()
    elif isinstance(value, str):
        return pa.string()
    elif isinstance(value, int):
        return pa.int64()
    elif isinstance(value, float):
        return pa.float64()
    elif isinstance(value, dict):
        # Recursively build struct type for nested dict
        nested_fields = []
        for k, v in value.items():
            nested_fields.append((k, infer_parquet_type(v)))
        return pa.struct(nested_fields)
    elif isinstance(value, list):
        # Simple list of strings for now
        return pa.list_(pa.string())
    else:
        return pa.string()


def infer_parquet_schema(record: dict):
    """Infer PyArrow schema from a dictionary record.

    Supports nested dictionaries as struct types.

    Args:
        record: Dictionary record

    Returns:
        PyArrow schema
    """
    import pyarrow as pa

    fields = []
    for key, value in record.items():
        fields.append((key, infer_parquet_type(value)))

    return pa.schema(fields)


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

    # Validate that pattern contains {shard} when there are multiple shards
    if total > 1 and "{shard" not in pattern:
        raise ValueError(
            f"Output pattern must contain '{{shard}}' placeholder when writing {total} shards. Got pattern: {pattern}"
        )

    basename = f"shard_{shard_idx}"
    formatted = pattern.format(shard=shard_idx, total=total, basename=basename)

    # Normalize double slashes while preserving protocol (e.g., gs://, s3://, http://)
    # Replace multiple slashes with single slash, except after protocol
    normalized = re.sub(r"(?<!:)//+", "/", formatted)

    return normalized


def write_records_to_jsonl(records: Iterable, output_path: str) -> str:
    """Write records to a JSONL file.

    Internal wrapper around write_jsonl_file that returns just the path for backward compatibility.

    Args:
        records: Records to write (each should be JSON-serializable)
        output_path: Path to output file

    Returns:
        Output path written
    """
    result = write_jsonl_file(records, output_path)
    return result["path"]


def write_records_to_parquet(records: Iterable, output_path: str, schema: object | None, batch_size: int = 1000) -> str:
    """Write records to a Parquet file using streaming batched writes.

    Args:
        records: Records to write (iterable of dicts)
        output_path: Path to output file
        schema: PyArrow schema (optional, will be inferred from first record if None)
        batch_size: Number of records per batch (default: 1000)

    Returns:
        Output path written
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ensure_parent_dir(output_path)

    # Convert to iterator to allow peeking
    record_iter = iter(records)

    # Peek at first record to infer schema if not provided
    first_record = None
    try:
        first_record = next(record_iter)
    except StopIteration:
        # No records - write empty file
        actual_schema = schema or pa.schema([])
        table = pa.Table.from_pylist([], schema=actual_schema)
        pq.write_table(table, output_path)
        return output_path

    # Infer schema if not provided
    actual_schema = schema or infer_parquet_schema(first_record)

    # Write records in batches using ParquetWriter
    with pq.ParquetWriter(output_path, actual_schema) as writer:
        # Process first batch including the peeked record
        batch = [first_record]
        for record in record_iter:
            batch.append(record)
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=actual_schema)
                writer.write_table(table)
                batch = []

        # Write remaining records
        if batch:
            table = pa.Table.from_pylist(batch, schema=actual_schema)
            writer.write_table(table)

    return output_path


def process_shard_flat_map(shard: Shard, shard_idx: int, total: int, fn: Callable, chunk_size: int) -> list[Shard]:
    """Process shard with flat_map, returning chunked results."""
    logger.info(f"flat_map: shard {shard_idx + 1}/{total}")

    def result_generator():
        for chunk in tqdm(shard.iter_chunks(), desc=f"flat_map shard {shard_idx + 1}/{total}"):
            for item in chunk:
                yield from fn(item)

    return [Shard.from_items(result_generator(), chunk_size, shard.context, idx=shard_idx)]


def process_shard_map(shard: Shard, shard_idx: int, total: int, fn: Callable, chunk_size: int) -> list[Shard]:
    """Process shard with map, returning chunked results."""
    logger.info(f"map: shard {shard_idx + 1}/{total}")

    def result_generator():
        for chunk in tqdm(shard.iter_chunks(), desc=f"map shard {shard_idx + 1}/{total}"):
            for item in chunk:
                yield fn(item)

    return [Shard.from_items(result_generator(), chunk_size, shard.context, idx=shard_idx)]


def process_shard_filter(shard: Shard, shard_idx: int, total: int, predicate: Callable, chunk_size: int) -> list[Shard]:
    """Process shard with filter, returning chunked results."""
    logger.info(f"filter: shard {shard_idx + 1}/{total}")

    def result_generator():
        for chunk in tqdm(shard.iter_chunks(), desc=f"filter shard {shard_idx + 1}/{total}"):
            for item in chunk:
                if predicate(item):
                    yield item

    return [Shard.from_items(result_generator(), chunk_size, shard.context, idx=shard_idx)]


def process_shard_batch(shard: Shard, shard_idx: int, total: int, batch_size: int, chunk_size: int) -> list[Shard]:
    """Process shard with batching, returning chunked results."""

    def result_generator():
        yield from make_batches(shard, batch_size)

    return [Shard.from_items(result_generator(), chunk_size, shard.context, idx=shard_idx)]


def process_shard_write_jsonl(
    shard: Shard, shard_idx: int, total: int, output_pattern: str, chunk_size: int
) -> list[Shard]:
    """Write shard to JSONL file and return Shard containing the output filename."""
    logger.info(f"write_jsonl: shard {shard_idx + 1}/{total}")
    output_path = format_shard_path(output_pattern, shard_idx, total)

    result = write_records_to_jsonl(shard, output_path)
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

    # Stream records from shard
    result = write_records_to_parquet(shard, output_path, schema, batch_size)
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
    from zephyr.dataset import BatchOp, FilterOp, FlatMapOp, GroupByLocalOp, MapOp, WriteJsonlOp, WriteParquetOp

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
            result = write_records_to_jsonl(stream_input, output_path)
            yield from build_stream(iter([result]), rest)
        elif isinstance(op, WriteParquetOp):
            output_path = format_shard_path(op.output_pattern, shard_idx, total_shards)
            result = write_records_to_parquet(stream_input, output_path, op.schema, op.batch_size)
            yield from build_stream(iter([result]), rest)

    # If the last operation is a group by, handle the resharding
    if operations and isinstance(operations[-1], GroupByLocalOp):
        group_by_local_op = operations[-1]
        pre_ops = operations[:-1]
        num_output_shards = group_by_local_op.num_output_shards

        # Handle sentinel value (-1) - use total as default
        if num_output_shards <= 0:
            num_output_shards = total_shards

        # Accumulate items into output shards
        output_chunks = defaultdict(list)
        output_tmp = defaultdict(list)

        for chunk in tqdm(
            shard.iter_chunks(), desc=f"fused[{len(operations)}]: shard {shard_idx + 1}/{total_shards}", mininterval=10
        ):
            for item in build_stream(chunk, pre_ops):
                key = group_by_local_op.key_fn(item)
                target_shard = deterministic_hash(key) % num_output_shards
                output_tmp[target_shard].append(item)
                if len(output_tmp[target_shard]) >= chunk_size:
                    output_chunks[target_shard].append(
                        Chunk(count=len(output_tmp[target_shard]), data=shard.context.put(output_tmp[target_shard]))
                    )
                    output_tmp[target_shard] = []

        # Flush remaining items
        for target_shard, items in output_tmp.items():
            if items:
                output_chunks[target_shard].append(Chunk(count=len(items), data=shard.context.put(items)))

        return [Shard(idx=idx, chunks=output_chunks[idx], context=shard.context) for idx in range(num_output_shards)]

    # Otherwise, just stream and return a single output shard
    def _generator():
        for chunk in shard.iter_chunks():
            yield from build_stream(chunk, operations)

    return [
        Shard.from_items(
            tqdm(_generator(), desc=f"fused[{len(operations)}]: shard {shard_idx + 1}/{total_shards}", mininterval=10),
            chunk_size,
            shard.context,
            idx=shard_idx,
        )
    ]


def process_shard_map_shard(shard: Shard, shard_idx: int, total: int, fn: Callable, chunk_size: int) -> list[Shard]:
    """Process shard with map_shard, applying function to all items at once.

    Args:
        shard: Input shard
        shard_idx: Index of this shard
        total: Total number of shards
        fn: Function from Iterator[T] -> Iterator[R]
        chunk_size: Number of items per chunk in output

    Returns:
        List containing single output shard with chunked results
    """
    logger.info(f"map_shard: shard {shard_idx + 1}/{total}")
    result_items = fn(shard)
    return [Shard.from_items(result_items, chunk_size, shard.context, idx=shard_idx)]


def deterministic_hash(obj: object) -> int:
    """Compute a cheap deterministic hash that's consistent across processes.

    Python's built-in hash() is randomized per-process for security, which breaks
    distributed groupby operations. This function uses zlib.adler32 for a fast
    deterministic hash.

    Args:
        obj: Object to hash (must be JSON-serializable)

    Returns:
        Non-negative integer hash value
    """
    import json
    import zlib

    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return zlib.adler32(s)


def process_shard_group_by_local(
    shard: Shard, shard_idx: int, total: int, key_fn: Callable, num_output_shards: int, chunk_size: int
) -> list[Shard]:
    """Phase 1: Local grouping per shard, distributing by hash(key) % num_output_shards."""

    logger.info(f"group_by_local: shard {shard_idx + 1}/{total}")

    # accumulate output into chunks, grouped by target shard idx
    output_chunks = defaultdict(list)
    output_tmp = defaultdict(list)

    for item in tqdm(shard, desc=f"group_by_local shard {shard_idx + 1}/{total}", mininterval=10):
        key = key_fn(item)
        target_shard = deterministic_hash(key) % num_output_shards
        output_tmp[target_shard].append(item)
        if len(output_tmp[target_shard]) >= chunk_size:
            output_chunks[target_shard].append(
                Chunk(count=len(output_tmp[target_shard]), data=shard.context.put(output_tmp[target_shard]))
            )
            output_tmp[target_shard] = []

    for target_shard, items in output_tmp.items():
        output_chunks[target_shard].append(Chunk(count=len(items), data=shard.context.put(items)))

    return [Shard(idx=idx, chunks=output_chunks[idx], context=shard.context) for idx in range(num_output_shards)]


def process_shard_group_by_reduce(
    shard: Shard, shard_idx: int, total: int, key_fn: Callable, reducer_fn: Callable, chunk_size: int
) -> list[Shard]:
    """Phase 2: Global reduction per shard, applying reducer to each key group.

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
    from collections import defaultdict

    logger.info(f"group_by_reduce: shard {shard_idx + 1}/{total}")

    # Group items by key
    groups = defaultdict(list)
    for item in shard:
        key = key_fn(item)
        groups[key].append(item)

    # Apply reducer to each group
    def result_generator():
        for key, items in groups.items():
            yield reducer_fn(key, iter(items))

    return [Shard.from_items(result_generator(), chunk_size, shard.context, idx=shard_idx)]


def reshard_refs(shards: list[Shard], num_shards: int) -> list[Shard]:
    """Redistribute chunks across target number of shards (best-effort).

    Args:
        shards: Input shards
        num_shards: Target number of output shards

    Returns:
        New list of shards with redistributed chunks
    """
    if not shards:
        return []

    context = shards[0].context
    all_chunks = [chunk for shard in shards for chunk in shard.chunks]

    if not all_chunks:
        return []

    chunk_groups = np.array_split(all_chunks, num_shards)
    return [
        Shard(idx=idx, chunks=list(group), context=context) for idx, group in enumerate(chunk_groups) if len(group) > 0
    ]


def recompact_shards(shard_lists: list[list[Shard]]) -> list[Shard]:
    """Recompact shards after an operation.

    Takes list of lists (one list per input shard) and merges chunks by target shard index.
    This enables a unified architecture where all operations return list[Shard].

    Args:
        shard_lists: List of shard lists, one from each input shard

    Returns:
        Compacted list of output shards, ordered by shard index
    """
    if not shard_lists:
        return []

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
        # Optimize operation chain
        optimized_ops = self._optimize_operations(operations)

        # Print optimization plan in dry-run mode
        if self.dry_run:
            self._print_optimization_plan(operations, optimized_ops)
            return

        # Execute optimized chain
        yield from self._execute_optimized(source, optimized_ops)

    def _optimize_operations(self, operations: list) -> list:
        """Optimize operation chain by fusing consecutive operations.

        Fuses consecutive operations (MapOp, FlatMapOp, FilterOp, BatchOp, Write*Op)
        into a single FusedMapOp to avoid materializing intermediate results.
        ReshardOp, MapShardOp, and GroupByOp break fusion as they change parallelism
        structure or need to see all items at once.

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
            Non-fusible operations require a shuffle boundary or need to see all items at once.

            Non-fusible: ReshardOp (just reshuffles refs), MapShardOp (needs all items),
                        GroupByShuffleReduceOp (shuffle boundary).
            Fusible: MapOp, FlatMapOp, FilterOp, BatchOp, Write*Op, GroupByLocalOp.
            """
            return not isinstance(op, (ReshardOp, MapShardOp, GroupByShuffleReduceOp))

        def flush_buffer():
            """Flush accumulated fusible operations."""
            if not fusible_buffer:
                return

            if len(fusible_buffer) == 1:
                # Single operation - no need to fuse
                optimized.append(fusible_buffer[0])
            else:
                # Multiple operations - create fused op
                optimized.append(FusedMapOp(operations=fusible_buffer[:]))

            fusible_buffer.clear()

        for op in operations:
            if is_fusible(op):
                fusible_buffer.append(op)
            else:
                # Non-fusible operation (ReshardOp) - flush buffer and add this op
                flush_buffer()
                optimized.append(op)

        # Flush any remaining operations
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
        shards = []
        for idx, item in enumerate(source):
            ref = self.context.put([item])
            shards.append(Shard.from_single_ref(ref, self.context, idx=idx, count=1))

        for op in operations:
            if isinstance(op, FlatMapOp):
                shards = self._execute_on_shards_unified(process_shard_flat_map, (op.fn, self.chunk_size), shards)
            elif isinstance(op, MapOp):
                shards = self._execute_on_shards_unified(process_shard_map, (op.fn, self.chunk_size), shards)
            elif isinstance(op, FilterOp):
                shards = self._execute_on_shards_unified(process_shard_filter, (op.predicate, self.chunk_size), shards)
            elif isinstance(op, FusedMapOp):
                shards = self._execute_on_shards_unified(process_shard_fused, (op.operations, self.chunk_size), shards)
            elif isinstance(op, BatchOp):
                shards = self._execute_on_shards_unified(process_shard_batch, (op.batch_size, self.chunk_size), shards)
            elif isinstance(op, MapShardOp):
                shards = self._execute_on_shards_unified(process_shard_map_shard, (op.fn, self.chunk_size), shards)
            elif isinstance(op, GroupByLocalOp):
                # Auto-detect num_output_shards if using sentinel value
                num_output_shards = op.num_output_shards if op.num_output_shards > 0 else len(shards)
                shards = self._execute_on_shards_unified(
                    process_shard_group_by_local, (op.key_fn, num_output_shards, self.chunk_size), shards
                )
            elif isinstance(op, GroupByShuffleReduceOp):
                shards = self._execute_on_shards_unified(
                    process_shard_group_by_reduce, (op.key_fn, op.reducer_fn, self.chunk_size), shards
                )
            elif isinstance(op, ReshardOp):
                shards = reshard_refs(shards, op.num_shards)
            elif isinstance(op, WriteJsonlOp):
                shards = self._execute_on_shards_unified(
                    process_shard_write_jsonl, (op.output_pattern, self.chunk_size), shards
                )
            elif isinstance(op, WriteParquetOp):
                shards = self._execute_on_shards_unified(
                    process_shard_write_parquet, (op.output_pattern, op.schema, op.batch_size, self.chunk_size), shards
                )

        # Materialize shards with progress bar
        op_names = [op.__class__.__name__.replace("Op", "") for op in operations]
        desc = f"Pipeline [{' → '.join(op_names)}]"

        def materialize_all():
            for shard in shards:
                yield from shard

        yield from tqdm(materialize_all(), desc=desc, unit="items")

    def _execute_on_shards_unified(self, process_fn: Callable, fn_args: tuple, shards: list[Shard]) -> list[Shard]:
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

        # Get all list[Shard] results
        shard_lists = [self.context.get(future) for future in result_futures]

        # Recompact into final output shards
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
