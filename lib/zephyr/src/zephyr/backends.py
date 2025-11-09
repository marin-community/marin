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
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

import numpy as np
import ray
from tqdm import tqdm

from zephyr.dataset import (
    BatchOp,
    Dataset,
    FilterOp,
    FlatMapOp,
    FusedMapOp,
    MapOp,
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
def _put_on_worker(obj: Any) -> ray.ObjectRef:
    """Put object in Ray's object store on a worker node.

    This function is executed remotely to ensure objects are placed on worker
    nodes rather than the head node.
    """
    return ray.put(obj)


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
        """
        # Submit task to worker with SPREAD strategy to distribute across nodes
        ref_future = _put_on_worker.options(scheduling_strategy="SPREAD").remote(obj)
        # Get the ObjectRef that the worker created
        return ray.get(ref_future)

    def get(self, ref: ray.ObjectRef) -> Any:
        return ray.get(ref)

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
class Shard:
    """Container for a logically grouped set of items split across multiple refs.

    Each ref contains a chunk of items (e.g., 1000 items) to prevent large object overhead.
    Works with any ExecutionContext (Ray, ThreadPool, Sync) via context.put/get operations.
    """

    refs: list[Any]  # Context-specific refs (ray.ObjectRef, Future, or plain objects)
    context: ExecutionContext

    def iter_chunks(self) -> Iterator[list]:
        """Iterate over chunks (each chunk is a list of items)."""
        for ref in self.refs:
            yield self.context.get(ref)

    def materialize(self) -> Iterator:
        """Iterate over all individual items across all chunks."""
        for chunk in self.iter_chunks():
            yield from chunk

    def __iter__(self):
        return self.materialize()

    @staticmethod
    def from_items(items: Iterable, chunk_size: int, context: ExecutionContext) -> Shard:
        """Create a Shard from items by chunking and storing via context.

        Args:
            items: Items to chunk and store
            chunk_size: Number of items per chunk
            context: Execution context for put/get operations

        Returns:
            Shard containing chunked refs
        """
        refs = []
        chunk = []
        for item in items:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                refs.append(context.put(chunk))
                chunk = []
        if chunk:
            refs.append(context.put(chunk))
        return Shard(refs=refs, context=context)

    @staticmethod
    def from_single_ref(ref: Any, context: ExecutionContext) -> Shard:
        """Wrap a single ref as a Shard.

        Args:
            ref: Reference to wrap (type depends on context)
            context: Execution context for get operations

        Returns:
            Shard containing the single ref
        """
        return Shard(refs=[ref], context=context)


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


def process_shard_flat_map(shard: Shard, shard_idx: int, total: int, fn: Callable, chunk_size: int) -> Shard:
    """Process shard with flat_map, returning chunked results."""
    logger.info(f"flat_map: shard {shard_idx + 1}/{total}")

    def result_generator():
        for chunk in tqdm(shard.iter_chunks(), desc=f"flat_map shard {shard_idx + 1}/{total}"):
            for item in chunk:
                yield from fn(item)

    return Shard.from_items(result_generator(), chunk_size, shard.context)


def process_shard_map(shard: Shard, shard_idx: int, total: int, fn: Callable, chunk_size: int) -> Shard:
    """Process shard with map, returning chunked results."""
    logger.info(f"map: shard {shard_idx + 1}/{total}")

    def result_generator():
        for chunk in tqdm(shard.iter_chunks(), desc=f"map shard {shard_idx + 1}/{total}"):
            for item in chunk:
                yield fn(item)

    return Shard.from_items(result_generator(), chunk_size, shard.context)


def process_shard_filter(shard: Shard, shard_idx: int, total: int, predicate: Callable, chunk_size: int) -> Shard:
    """Process shard with filter, returning chunked results."""
    logger.info(f"filter: shard {shard_idx + 1}/{total}")

    def result_generator():
        for chunk in tqdm(shard.iter_chunks(), desc=f"filter shard {shard_idx + 1}/{total}"):
            for item in chunk:
                if predicate(item):
                    yield item

    return Shard.from_items(result_generator(), chunk_size, shard.context)


def process_shard_batch(shard: Shard, shard_idx: int, total: int, batch_size: int, chunk_size: int) -> Shard:
    """Process shard with batching, returning chunked results."""

    def result_generator():
        yield from make_batches(shard.materialize(), batch_size)

    return Shard.from_items(result_generator(), chunk_size, shard.context)


def process_shard_write_jsonl(shard: Shard, shard_idx: int, total: int, output_pattern: str, chunk_size: int) -> Shard:
    """Write shard to JSONL file and return Shard containing the output filename."""
    logger.info(f"write_jsonl: shard {shard_idx + 1}/{total}")
    output_path = format_shard_path(output_pattern, shard_idx, total)

    result = write_records_to_jsonl(shard.materialize(), output_path)
    logger.info(f"write_jsonl: shard {shard_idx + 1}/{total} → {output_path}")
    return Shard.from_items([result], chunk_size, shard.context)


def process_shard_write_parquet(
    shard: Shard,
    shard_idx: int,
    total: int,
    output_pattern: str,
    schema: object | None,
    batch_size: int,
    chunk_size: int,
) -> Shard:
    """Write shard to Parquet file and return Shard containing the output filename."""
    logger.info(f"write_parquet: shard {shard_idx + 1}/{total}")
    output_path = format_shard_path(output_pattern, shard_idx, total)

    # Stream records from shard
    result = write_records_to_parquet(shard.materialize(), output_path, schema, batch_size)
    logger.info(f"write_parquet: shard {shard_idx + 1}/{total} → {output_path}")
    return Shard.from_items([result], chunk_size, shard.context)


def process_shard_fused(shard: Shard, shard_idx: int, total: int, operations: list, chunk_size: int) -> Shard:
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
        Output shard with chunked results
    """
    from zephyr.dataset import BatchOp, FilterOp, FlatMapOp, MapOp, WriteJsonlOp, WriteParquetOp

    logger.info(f"fused[{len(operations)}]: shard {shard_idx + 1}/{total}")

    def apply_pipeline(items):
        """Apply all operations to an iterable of items using generators."""

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
                # Write stream to file, yield filename
                output_path = format_shard_path(op.output_pattern, shard_idx, total)
                result = write_records_to_jsonl(stream_input, output_path)
                logger.info(f"fused write_jsonl: shard {shard_idx + 1}/{total} → {output_path}")
                yield from build_stream(iter([result]), rest)
            elif isinstance(op, WriteParquetOp):
                # Write stream to file, yield filename
                output_path = format_shard_path(op.output_pattern, shard_idx, total)
                result = write_records_to_parquet(stream_input, output_path, op.schema, op.batch_size)
                logger.info(f"fused write_parquet: shard {shard_idx + 1}/{total} → {output_path}")
                yield from build_stream(iter([result]), rest)

        yield from build_stream(items, operations)

    def result_generator():
        """Generate results by processing chunks through the pipeline."""
        for chunk in tqdm(shard.iter_chunks(), desc=f"fused[{len(operations)}] shard {shard_idx + 1}/{total}"):
            yield from apply_pipeline(chunk)

    return Shard.from_items(result_generator(), chunk_size, shard.context)


def reshard_refs(shards: list[Shard], num_shards: int) -> list[Shard]:
    """Redistribute refs across target number of shards (best-effort).

    Args:
        shards: Input shards
        num_shards: Target number of output shards

    Returns:
        New list of shards with redistributed refs
    """
    if not shards:
        return []

    context = shards[0].context
    all_refs = [ref for shard in shards for ref in shard.refs]

    if not all_refs:
        return []

    ref_chunks = np.array_split(all_refs, num_shards)
    return [Shard(refs=list(chunk), context=context) for chunk in ref_chunks if len(chunk) > 0]


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
        Only ReshardOp breaks fusion as it changes parallelism structure.

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
            """Check if operation can be fused (everything except ReshardOp)."""
            return not isinstance(op, ReshardOp)

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
        for item in source:
            ref = self.context.put([item])
            shards.append(Shard.from_single_ref(ref, self.context))

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
            elif isinstance(op, ReshardOp):
                shards = reshard_refs(shards, op.num_shards)
            elif isinstance(op, WriteJsonlOp):
                shards = self._execute_on_shards(process_shard_write_jsonl, (op.output_pattern, self.chunk_size), shards)
            elif isinstance(op, WriteParquetOp):
                shards = self._execute_on_shards(
                    process_shard_write_parquet, (op.output_pattern, op.schema, op.batch_size, self.chunk_size), shards
                )

        # Materialize shards with progress bar
        op_names = [op.__class__.__name__.replace("Op", "") for op in operations]
        desc = f"Pipeline [{' → '.join(op_names)}]"

        def materialize_all():
            for shard in shards:
                yield from shard.materialize()

        yield from tqdm(materialize_all(), desc=desc, unit="items")

    def _execute_on_shards(self, process_fn: Callable, fn_args: tuple, shards: list[Shard]) -> list[Shard]:
        """Execute a processing function on shards with bounded parallelism.

        Args:
            process_fn: Function that takes (Shard, int, int, *fn_args) -> Shard
            fn_args: Additional arguments to pass to process_fn
            shards: List of input Shards

        Returns:
            List of output Shards in original order
        """
        total = len(shards)
        result_futures = [None] * total
        pending = {}

        for shard_idx, shard in enumerate(shards):
            # Execute via context (Ray remote, thread pool submit, or immediate)
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

        # Get all Shard results
        return [self.context.get(future) for future in result_futures]


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
