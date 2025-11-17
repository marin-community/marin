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

import heapq
import logging
import os
import re
import struct
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass, field
from hashlib import blake2b
from itertools import groupby, islice
from typing import Any, TypeVar
from collections.abc import Sequence

import fsspec
import msgspec
import numpy as np
from fray import ExecutionContext
from tqdm_loggable.auto import tqdm

from zephyr.dataset import (
    Dataset,
    FilterOp,
    FlatMapOp,
    FusedMapOp,
    GroupByLocalOp,
    GroupByShuffleReduceOp,
    MapOp,
    MapShardOp,
    ReduceGlobalOp,
    ReduceLocalOp,
    ReshardOp,
    SortedMergeJoinOp,
    TakePerShardOp,
    WindowOp,
    WriteDataOp,
)

from .writers import write_binary_file, write_jsonl_file, write_levanter_cache, write_parquet_file

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ShardMonitor:
    """Monitor progress across multiple pipeline stages within a shard.

    Periodically logs a single line showing progress across all active stages.
    Example: "shard 3/10: FlatMap[process] (1234) Map[transform] (5678) Filter[quality] (234)"
    """

    def __init__(self, shard_idx: int, total_shards: int, interval: float = 10.0):
        """Initialize shard monitor.

        Args:
            shard_idx: Index of this shard (0-based)
            total_shards: Total number of shards
            interval: Logging interval in seconds
        """
        self.shard_idx = shard_idx
        self.total_shards = total_shards
        self.interval = interval
        self.stage_counts: dict[str, int] = {}
        self.last_log_time = 0

    def _maybe_log(self):
        """Log status if interval has elapsed."""
        now = time.time()
        if now - self.last_log_time >= self.interval:
            self._log_status()
            self.last_log_time = now

    def _log_status(self):
        """Log current progress across all stages."""
        stages = " ".join(f"{stage} ({count})" for stage, count in self.stage_counts.items())
        logger.info(f"shard {self.shard_idx + 1}/{self.total_shards}: {stages}")

    def wrap(self, items: Iterable[T], stage_name: str) -> Iterator[T]:
        """Wrap an iterable to track progress for a specific stage.

        Args:
            items: Items to iterate over
            stage_name: Name of the stage (e.g., "FlatMap[process]")

        Yields:
            Items from the input iterable
        """
        self.stage_counts[stage_name] = 0
        try:
            for item in items:
                self.stage_counts[stage_name] += 1
                self._maybe_log()
                yield item
        finally:
            # Stage complete - remove from active stages
            self.stage_counts.pop(stage_name, None)


@dataclass
class BackendConfig:
    """Configuration for backend execution.

    Attributes:
        max_parallelism: Maximum number of concurrent tasks
        chunk_size: Number of items per chunk in Shard
        dry_run: If True, show optimization plan without executing
    """

    max_parallelism: int = 1000000
    chunk_size: int = 1000
    dry_run: bool = False


@dataclass
class ChunkHeader:
    """Metadata for a chunk being streamed from a worker."""

    shard_idx: int
    chunk_idx: int
    count: int


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
            data = self.context.get(chunk.data)
            yield data

    def __iter__(self):
        """Flat map over all chunks."""
        for chunk_data in self.iter_chunks():
            yield from chunk_data

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


@dataclass
class ApplyShardCtx:
    """Context for applying operations to a shard.

    Encapsulates all the metadata and auxiliary data needed to process a shard.
    """

    shard: Shard
    shard_idx: int
    total_shards: int
    chunk_size: int
    aux_shards: dict[int, list[Shard]] = field(default_factory=dict)


def make_windows(
    items: Iterable[T],
    folder_fn: Callable[[object, T], tuple[bool, object]],
    initial_state: object,
) -> Iterator[list[T]]:
    """Window items using a folder function.

    Args:
        items: Items to window
        folder_fn: Function (state, item) -> (should_continue, new_state)
        initial_state: Initial state for the folder function

    Yields:
        Windows of items (window closes when folder returns False)
    """
    window = []
    state = initial_state

    for item in items:
        should_continue, new_state = folder_fn(state, item)

        if not should_continue and window:
            # Close current window and start new one with this item
            yield window
            window = [item]
            state = initial_state
            # Re-apply folder with the item in the new window
            _, state = folder_fn(state, item)
        else:
            # Add item to current window
            window.append(item)
            state = new_state

    if window:
        yield window


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
    if total > 1 and "{shard" not in pattern:
        raise ValueError(
            f"Output pattern must contain '{{shard}}' placeholder when writing {total} shards. Got pattern: {pattern}"
        )

    basename = f"shard_{shard_idx}"
    formatted = pattern.format(shard=shard_idx, total=total, basename=basename)

    # Normalize double slashes while preserving protocol (e.g., gs://, s3://, http://)
    normalized = re.sub(r"(?<!:)//+", "/", formatted)

    return normalized


# Helper to stream chunks from an iterator
# N.B. All shard operations yield header & chunk separately.
# This is so that we can resolve the header (which indicates which shard a chunk goes to)
# separately from the chunk data. This allows the controller to coalesce chunks without
# copying data directly.
def _stream_chunks(items: Iterator, shard_idx: int, chunk_size: int) -> Generator[ChunkHeader | list[Any], None, None]:
    """Stream chunks from an iterator, yielding header/data pairs."""
    chunk = []
    chunk_idx = 0
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            header = ChunkHeader(shard_idx=shard_idx, chunk_idx=chunk_idx, count=len(chunk))
            yield header
            yield chunk
            chunk = []
            chunk_idx += 1
    # Yield final partial chunk
    if chunk:
        header = ChunkHeader(shard_idx=shard_idx, chunk_idx=chunk_idx, count=len(chunk))
        yield header
        yield chunk


def deterministic_hash(obj: object) -> int:
    s = msgspec.msgpack.encode(obj, order="deterministic")
    return zlib.adler32(s)


def _group_items_by_hash(
    items: Iterable,
    key_fn: Callable,
    num_output_shards: int,
    chunk_size: int,
) -> dict[int, list[Chunk]]:
    """Group items by hash of key into output shards with sorted chunks.

    Args:
        items: Items to group
        key_fn: Function to extract grouping key from item
        num_output_shards: Number of output shards to distribute across
        chunk_size: Number of items per chunk

    Returns:
        Dict mapping shard index to list of chunks for that shard

    Note:
        Chunks contain raw data lists (not ObjectRefs). When used in worker
        functions, the controller will store yielded data in the object store.
    """
    output_chunks = defaultdict(list)
    output_tmp = defaultdict(list)

    for item in items:
        key = key_fn(item)
        target_shard = deterministic_hash(key) % num_output_shards
        output_tmp[target_shard].append(item)
        if len(output_tmp[target_shard]) >= chunk_size:
            sorted_items = sorted(output_tmp[target_shard], key=key_fn)
            output_chunks[target_shard].append(Chunk(count=len(sorted_items), data=sorted_items))
            output_tmp[target_shard] = []

    # Add all remaining chunks
    for target_shard, items in output_tmp.items():
        if items:
            sorted_items = sorted(items, key=key_fn)
            output_chunks[target_shard].append(Chunk(count=len(sorted_items), data=sorted_items))

    return output_chunks


def process_shard_fused(
    ctx: ApplyShardCtx,
    operations: list,
) -> Generator[ChunkHeader | list[Any], None, None]:
    """Process shard with fused operations pipeline, streaming chunks as produced.

    Chains operations into a single pass over the data, avoiding intermediate
    materialization in the object store. Yields ChunkHeader followed by list of items
    for each chunk produced.

    Args:
        ctx: Context containing shard, metadata, and auxiliary data
        operations: List of operations to apply in sequence

    Yields:
        ChunkHeader followed by list of items for each chunk
    """
    op_names = "|".join([str(op) for op in operations])
    logger.info(f"fused[{op_names}]: shard {ctx.shard_idx}/{ctx.total_shards}")

    monitor = ShardMonitor(ctx.shard_idx, ctx.total_shards)

    # we first build a stream of all of the fused operations.
    # if we have a group by or reduction, we then apply them at the end
    # if not, we yield our final shard directly.
    def build_stream(stream_input, ops, op_index=0):
        """Recursively build the generator pipeline.

        Args:
            stream_input: Input stream
            ops: Operations still to process
            op_index: Index of the current operation
        """
        if not ops:
            yield from stream_input
            return

        op, *rest = ops

        # Wrap stream with monitor for this operation
        stream_input = monitor.wrap(stream_input, str(op))

        if isinstance(op, MapOp):
            yield from build_stream((op.fn(item) for item in stream_input), rest, op_index + 1)
        elif isinstance(op, FlatMapOp):
            # flatten the mapped iterators
            flat_map_iter = (result_item for item in stream_input for result_item in op.fn(item))
            yield from build_stream(flat_map_iter, rest, op_index + 1)
        elif isinstance(op, MapShardOp):
            yield from build_stream(op.fn(stream_input), rest, op_index + 1)
        elif isinstance(op, FilterOp):
            filter_iter = (item for item in stream_input if op.predicate(item))
            yield from build_stream(filter_iter, rest, op_index + 1)
        elif isinstance(op, TakePerShardOp):
            logger.info(f"Taking first {op.n} items from shard {ctx.shard_idx}")
            take_iter = islice(stream_input, op.n)
            yield from build_stream(take_iter, rest, op_index + 1)
        elif isinstance(op, WindowOp):
            yield from build_stream(make_windows(stream_input, op.folder_fn, op.initial_state), rest, op_index + 1)
        elif isinstance(op, WriteDataOp):
            output_path = format_shard_path(op.output_pattern, ctx.shard_idx, ctx.total_shards)

            # Check if we should skip writing because file already exists
            if op.skip_existing:
                fs = fsspec.core.url_to_fs(output_path)[0]
                # levanter writes directories, so use a sentinel file to check for existence
                if op.writer_type == "levanter_cache":
                    test_path = os.path.join(output_path, ".success")
                else:
                    test_path = output_path

                if fs.exists(test_path):
                    logger.info(f"Skipping write, output exists: {output_path}")
                    # Don't consume stream - lazy evaluation means upstream processing is skipped
                    yield from build_stream(iter([output_path]), rest, op_index + 1)
                    return

            # Write the file
            if op.writer_type == "jsonl":
                result = write_jsonl_file(stream_input, output_path)["path"]
            elif op.writer_type == "parquet":
                result = write_parquet_file(stream_input, output_path, op.schema, op.batch_size)["path"]
            elif op.writer_type == "levanter_cache":
                result = write_levanter_cache(stream_input, output_path, op.levanter_metadata)["path"]
            elif op.writer_type == "binary":
                result = write_binary_file(stream_input, output_path)["path"]
            else:
                raise ValueError(f"Unknown writer_type: {op.writer_type}")
            yield from build_stream(iter([result]), rest, op_index + 1)
        elif isinstance(op, SortedMergeJoinOp):
            # Get right shard from aux_shards
            right_shards = ctx.aux_shards.get(op_index, [])

            if len(right_shards) != 1:
                raise ValueError(f"Expected 1 right shard for join at op {op_index}, got {len(right_shards)}")

            right_shard = right_shards[0]
            joined = _sorted_merge_join(stream_input, right_shard, op)
            yield from build_stream(joined, rest, op_index + 1)

    if operations and isinstance(operations[-1], GroupByLocalOp):
        group_by_local_op = operations[-1]
        pre_ops = operations[:-1]
        num_output_shards = group_by_local_op.num_output_shards

        # Handle sentinel value (-1) - use total as default
        if num_output_shards <= 0:
            num_output_shards = ctx.total_shards

        stream = monitor.wrap(build_stream(ctx.shard, pre_ops), str(group_by_local_op))

        output_chunks = _group_items_by_hash(stream, group_by_local_op.key_fn, num_output_shards, ctx.chunk_size)

        # Stream chunks for each output shard
        for idx in range(num_output_shards):
            if output_chunks[idx]:
                for chunk in output_chunks[idx]:
                    header = ChunkHeader(shard_idx=idx, chunk_idx=0, count=chunk.count)
                    yield header
                    yield chunk.data
            else:
                # Yield empty chunk so controller knows this shard exists
                header = ChunkHeader(shard_idx=idx, chunk_idx=0, count=0)
                yield header
                yield []

    elif operations and isinstance(operations[-1], ReduceLocalOp):
        reduce_local_op = operations[-1]
        pre_ops = operations[:-1]

        stream = monitor.wrap(build_stream(ctx.shard, pre_ops), str(reduce_local_op))

        result = reduce_local_op.local_reducer(stream)
        # Stream single-item chunk
        yield from _stream_chunks([result], ctx.shard_idx, ctx.chunk_size)

    else:
        # No grouping or reduction at the end, stream the results directly
        stream = build_stream(ctx.shard, operations)
        yield from _stream_chunks(stream, ctx.shard_idx, ctx.chunk_size)


def _sorted_merge_join(left_stream: Iterable, right_stream: Iterable, join_op) -> Iterator:
    """Perform a sorted merge join between two streams.

    Args:
        left_stream: Iterator of items from left side
        right_stream: Iterator of items from right side
        join_op: SortedMergeJoinOp containing join configuration

    Yields:
        Joined items according to join_type (inner or left)
    """
    # Materialize left stream and tag both streams
    left_items = list(left_stream)
    left_tagged = (("left", join_op.left_key_fn(item), item) for item in left_items)
    right_tagged = (("right", join_op.right_key_fn(item), item) for item in right_stream)

    # Merge both sorted streams by key
    merged = heapq.merge(left_tagged, right_tagged, key=lambda x: x[1])

    # Group by key and apply join logic
    for _key, group in groupby(merged, key=lambda x: x[1]):
        left_group, right_group = [], []
        for side, _, item in group:
            (left_group if side == "left" else right_group).append(item)

        if join_op.join_type == "inner":
            if left_group and right_group:
                for left_item in left_group:
                    for right_item in right_group:
                        yield join_op.combiner_fn(left_item, right_item)
        elif join_op.join_type == "left":
            for left_item in left_group:
                if right_group:
                    for right_item in right_group:
                        yield join_op.combiner_fn(left_item, right_item)
                else:
                    yield join_op.combiner_fn(left_item, None)


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
    # Create iterators for each chunk
    chunk_iterators = []
    for chunk_data in shard.iter_chunks():
        chunk_iterators.append(iter(chunk_data))

    # Use heapq.merge to k-way merge sorted streams
    merged_stream = heapq.merge(*chunk_iterators, key=key_fn)
    for key, group_iter in groupby(merged_stream, key=key_fn):  # noqa: UP028
        yield key, group_iter


def process_shard_group_by_reduce(ctx: ApplyShardCtx, key_fn: Callable, reducer_fn: Callable) -> Generator:
    """Global reduction per shard, applying reducer to each key group.
    Uses streaming k-way merge to avoid materializing all items for each key.
    Chunks are assumed to be sorted by key.

    Args:
        ctx: Context containing shard and metadata
        key_fn: Function from item -> key
        reducer_fn: Function from (key, Iterator[items]) -> result

    Returns:
        List containing single output shard with reduced results
    """
    logger.info(f"group_by_reduce: shard {ctx.shard_idx + 1}/{ctx.total_shards}")

    def result_generator():
        for key, items_iter in _merge_sorted_chunks(ctx.shard, key_fn):
            yield reducer_fn(key, items_iter)

    has_data = False
    for i, item in enumerate(result_generator()):
        has_data = True
        yield ChunkHeader(shard_idx=ctx.shard_idx, chunk_idx=i, count=len(item))
        yield [item]

    # Yield empty chunk if shard has no data, so controller knows this shard exists
    if not has_data:
        yield ChunkHeader(shard_idx=ctx.shard_idx, chunk_idx=0, count=0)
        yield []


def process_shard_reduce_global(
    shards: list[Shard], global_reducer: Callable, context: ExecutionContext, chunk_size: int
) -> list[Shard]:
    logger.info(f"reduce_global: reducing {len(shards)} shard results")

    if not shards:
        return []

    def get_shard_results():
        for shard in shards:
            yield from shard

    final_result = global_reducer(get_shard_results())
    return [Shard.from_single_ref(context.put([final_result]), context, idx=0, count=1)]


def reshard_refs(shards: list[Shard], num_shards: int) -> list[Shard]:
    """Redistribute chunks across target number of shards. No data shuffling."""
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


class Backend:
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

    def execute(self, dataset: Dataset, verbose: bool = False) -> Sequence:
        """Execute a dataset and return an iterator over results.

        Args:
            dataset: Dataset to execute
            verbose: Print additional logging and optimization stats

        Returns:
            Iterator over dataset results
        """
        optimized_ops = self._optimize_operations(dataset.operations)

        if verbose:
            self._print_optimization_plan(dataset.operations, optimized_ops)

        if self.dry_run:
            return

        return list(self._execute_optimized(dataset.source, optimized_ops))

    def _optimize_operations(self, operations: list) -> list:
        if not operations:
            return operations

        optimized = []
        fusible_buffer = []

        def is_fusible(op):
            return not isinstance(op, (ReshardOp, GroupByShuffleReduceOp, ReduceGlobalOp))

        def flush_buffer():
            if not fusible_buffer:
                return

            # Always create a FusedMapOp, even for single operations, this simplifies downstream logic
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
        logger.info("\n=== ML-Flow Optimization Plan ===\n")
        logger.info(f"Original operations: {len(original_ops)}")
        logger.info(f"Optimized operations: {len(optimized_ops)}")
        logger.info(f"Fusion savings: {len(original_ops) - len(optimized_ops)} operations fused\n")

        logger.info("Original pipeline:")
        for i, op in enumerate(original_ops, 1):
            logger.info(f"  {i}. {op}")

        logger.info("\nOptimized pipeline:")
        for i, op in enumerate(optimized_ops, 1):
            if isinstance(op, FusedMapOp):
                fused_names = [str(op) for op in op.operations]
                logger.info(f"  {i}. FusedMap[{' → '.join(fused_names)}] ({len(op.operations)} operations)")
            else:
                logger.info(f"  {i}. {op}")

        logger.info("\n=== End Optimization Plan ===\n")

    def _run_operations_on_shards(self, source: Iterable, optimized_ops: list) -> list[Shard]:
        """Core execution logic - executes already-optimized operations on shards.

        Args:
            source: Source data iterable
            optimized_ops: Already-optimized operation chain

        Returns:
            List of Shards after applying all operations
        """
        # Convert source items to Shards
        shards = [
            Shard.from_single_ref(self.context.put([item]), self.context, idx=idx, count=1)
            for idx, item in enumerate(source)
        ]

        for op in optimized_ops:
            if isinstance(op, FusedMapOp):
                all_right_shards = {}
                # If we find any joins, we compute the intermediate right shards first
                # and then pass them as aux_shards to the processing function. This
                # currently materializes the right side fully before processing the left side,
                # we can probably be improved.
                for i, join_op in enumerate(op.operations):
                    if not isinstance(join_op, SortedMergeJoinOp):
                        continue
                    right_ops = self._optimize_operations(join_op.right_dataset.operations)
                    right_shards = self._run_operations_on_shards(join_op.right_dataset.source, right_ops)

                    if len(shards) != len(right_shards):
                        raise ValueError(
                            f"Sorted merge join requires equal shard counts. "
                            f"Left has {len(shards)} shards, right has {len(right_shards)} shards."
                        )

                    all_right_shards[i] = right_shards

                aux_shards_per_left = []
                for shard_idx in range(len(shards)):
                    shard_aux = {}
                    for join_idx, right_shards in all_right_shards.items():
                        shard_aux[join_idx] = [right_shards[shard_idx]]
                    aux_shards_per_left.append(shard_aux)

                shards = self._execute_on_shards(process_shard_fused, (op.operations,), shards, aux_shards_per_left)
            elif isinstance(op, GroupByShuffleReduceOp):
                shards = self._execute_on_shards(process_shard_group_by_reduce, (op.key_fn, op.reducer_fn), shards)
            elif isinstance(op, ReshardOp):
                shards = reshard_refs(shards, op.num_shards)
            elif isinstance(op, ReduceGlobalOp):
                shards = process_shard_reduce_global(shards, op.global_reducer, self.context, self.chunk_size)

        return shards

    def _execute_optimized(self, source: Iterable, optimized_ops: list) -> Iterator:
        """Execute already-optimized operations and materialize results.

        Args:
            source: Source data iterable
            optimized_ops: Already-optimized operation chain

        Yields:
            Results after applying all operations
        """
        # Execute operations to get shards
        shards = self._run_operations_on_shards(source, optimized_ops)

        # Materialize results
        op_names = [op.__class__.__name__.replace("Op", "") for op in optimized_ops]
        desc = f"Materialize [{' → '.join(op_names)}]"

        def materialize_all():
            for shard in shards:
                yield from shard

        yield from tqdm(materialize_all(), desc=desc, unit="items")

    def _execute_on_shards(
        self, process_fn: Callable, fn_args: tuple, shards: list[Shard], aux_shards_per_shard: list[dict] | None = None
    ) -> list[Shard]:
        """Execute a processing function on shards with optional per-shard auxiliary data.

        Args:
            process_fn: Function that takes (ApplyShardCtx, *fn_args) and yields (ChunkHeader, list[Any]) pairs
            fn_args: Additional arguments to pass to process_fn
            shards: List of input Shards
            aux_shards_per_shard: Optional list of aux_shards dicts, one per input shard.
                                 If None, creates empty dicts for each shard.

        Returns:
            List of output Shards assembled from streamed chunks
        """
        # If no aux_shards provided, create a list of empty dicts
        if aux_shards_per_shard is None:
            aux_shards_per_shard = [{} for _ in range(len(shards))]
        elif len(shards) != len(aux_shards_per_shard):
            raise ValueError(f"Mismatch: {len(shards)} shards but {len(aux_shards_per_shard)} aux_shards entries")

        total = len(shards)

        # Build list of (shard_idx, shard, aux_shards) tuples to process
        shard_tasks = [
            (shard_idx, shard, aux_shards)
            for shard_idx, (shard, aux_shards) in enumerate(zip(shards, aux_shards_per_shard, strict=True))
        ]

        chunks_by_shard_idx = defaultdict(list)

        if shard_tasks:
            active_gens = []
            queued_tasks = shard_tasks[:]
            finished_gens = []

            while len(active_gens) < self.config.max_parallelism and queued_tasks:
                shard_idx, shard, aux_shards = queued_tasks.pop(0)
                ctx = ApplyShardCtx(
                    shard=shard,
                    shard_idx=shard_idx,
                    total_shards=total,
                    chunk_size=self.chunk_size,
                    aux_shards=aux_shards,
                )
                active_gens.append(self.context.run(process_fn, ctx, *fn_args))

            while active_gens or queued_tasks:
                ready, _ = self.context.wait(active_gens, num_returns=1)

                for ready_gen in ready:
                    try:
                        header = self.context.get(next(ready_gen))
                        data = next(ready_gen)  # ref to header data
                        chunks_by_shard_idx[header.shard_idx].append(Chunk(header.count, data))

                    except StopIteration:
                        finished_gens.append(ready_gen)
                        active_gens.remove(ready_gen)
                        if queued_tasks:
                            shard_idx, shard, aux_shards = queued_tasks.pop(0)
                            ctx = ApplyShardCtx(
                                shard=shard,
                                shard_idx=shard_idx,
                                total_shards=total,
                                chunk_size=self.chunk_size,
                                aux_shards=aux_shards,
                            )
                            active_gens.append(self.context.run(process_fn, ctx, *fn_args))

        # Reconstruct shards from collected chunks
        if len(chunks_by_shard_idx) == 0:
            return []

        max_shard_idx = max(chunks_by_shard_idx.keys())
        return [
            Shard(idx=idx, chunks=chunks_by_shard_idx.get(idx, []), context=self.context)
            for idx in range(max_shard_idx + 1)
        ]
