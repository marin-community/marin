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

"""Physical execution plan for zephyr pipelines.

This module separates the logical plan (Dataset operations) from the physical
execution plan. The compute_plan() function transforms a Dataset into a
PhysicalPlan that can be executed by the Backend.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zephyr.dataset import (
    FlatMapOp,
    FusedMapOp,
    GroupByShuffleReduceOp,
    Operation,
    ReduceGlobalOp,
    ReshardOp,
)

if TYPE_CHECKING:
    from zephyr.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass
class ShardSpec:
    """Specification for a shard's data source with optional chunking.

    When source_chunk_bytes is enabled in ExecutionHint, the planner may create
    multiple ShardSpecs from a single source item (e.g., chunking a large parquet
    file into multiple shards by row range).

    Attributes:
        source_item: Original source item (e.g., file path string)
        chunk_index: Which chunk of this source (0 if not chunked)
        chunk_count: Total chunks for this source (1 if not chunked)
        row_start: For parquet: starting row index (None if not chunked)
        row_end: For parquet: ending row index exclusive (None if not chunked)
        shard_idx: Logical shard index. When intra-shard parallelism is enabled,
            multiple ShardSpecs with different chunk_index but same shard_idx
            belong to the same logical shard and will be aggregated together.
    """

    source_item: Any
    chunk_index: int = 0
    chunk_count: int = 1
    row_start: int | None = None
    row_end: int | None = None
    shard_idx: int = 0


@dataclass
class PhysicalStage:
    """A stage in the physical plan (operations between shuffle boundaries).

    A stage contains a sequence of operations that can be fused together and
    executed without requiring a shuffle. Stages are separated by shuffle
    boundaries like GroupByShuffleReduceOp or ReduceGlobalOp.

    Attributes:
        operations: List of operations to apply in sequence
        is_shuffle_boundary: True if this stage ends with an operation that
            requires shuffling data across shards
    """

    operations: list[Operation] = field(default_factory=list)
    is_shuffle_boundary: bool = False


@dataclass
class PhysicalPlan:
    """Executable plan computed from a Dataset.

    The physical plan contains:
    - shard_specs: How to partition the source data into shards
    - stages: Sequence of operation stages to execute

    The plan can be inspected before execution to understand parallelism,
    shard count, and operation fusion.

    Attributes:
        shard_specs: List of ShardSpecs defining initial shard boundaries
        stages: List of PhysicalStages to execute in order
    """

    shard_specs: list[ShardSpec]
    stages: list[PhysicalStage]

    @property
    def num_shards(self) -> int:
        """Total number of logical shards in the plan.

        This counts unique shard_idx values, not total ShardSpecs. When
        intra-shard parallelism is enabled, multiple ShardSpecs may belong
        to the same logical shard.
        """
        if not self.shard_specs:
            return 0
        return len({spec.shard_idx for spec in self.shard_specs})

    @property
    def num_chunks(self) -> int:
        """Total number of chunks across all shards."""
        return len(self.shard_specs)


@dataclass(frozen=True)
class ExecutionHint:
    """Hints for pipeline execution.

    Attributes:
        chunk_size: Number of items per chunk during streaming. Use -1 for
            1 chunk per shard.
        source_chunk_bytes: Target bytes per shard when chunking source files.
            Set to -1 (default) to disable source chunking. When enabled,
            large files (e.g., parquet) will be split into multiple shards.
        intra_shard_parallelism: Controls parallel processing of chunks within
            a shard. Set to -1 (default) for auto (parallel when chunks > 1),
            0 to disable, or N to limit max parallel chunks per shard.
    """

    chunk_size: int = 100_000
    source_chunk_bytes: int = -1
    intra_shard_parallelism: int = -1


# Registry of introspectors for different file types
# Each introspector takes (path, target_bytes) and returns list of ShardSpecs
INTROSPECTORS: dict[str, Callable[[str, int], list[ShardSpec]]] = {}


def register_introspector(extension: str, introspector: Callable[[str, int], list[ShardSpec]]) -> None:
    """Register an introspector for a file extension.

    Args:
        extension: File extension including dot (e.g., ".parquet")
        introspector: Function that takes (path, target_bytes) and returns ShardSpecs
    """
    INTROSPECTORS[extension] = introspector


def introspect_parquet(path: str, target_bytes: int) -> list[ShardSpec]:
    """Introspect a parquet file and return ShardSpecs for chunks.

    Chunks are created at row group boundaries to avoid reading partial row groups.

    Args:
        path: Path to parquet file
        target_bytes: Target bytes per chunk

    Returns:
        List of ShardSpecs, one per chunk
    """
    import pyarrow.parquet as pq

    meta = pq.read_metadata(path)

    if meta.num_row_groups == 0:
        return [ShardSpec(source_item=path)]

    chunks: list[tuple[int, int]] = []
    current_start_row = 0
    current_bytes = 0
    current_rows = 0

    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        rg_bytes = rg.total_byte_size
        rg_rows = rg.num_rows

        if current_bytes + rg_bytes > target_bytes and current_bytes > 0:
            # Close current chunk before this row group
            chunks.append((current_start_row, current_start_row + current_rows))
            current_start_row = current_start_row + current_rows
            current_bytes = rg_bytes
            current_rows = rg_rows
        else:
            current_bytes += rg_bytes
            current_rows += rg_rows

    # Final chunk
    if current_rows > 0:
        chunks.append((current_start_row, current_start_row + current_rows))

    # Convert to ShardSpecs
    chunk_count = len(chunks)
    return [
        ShardSpec(
            source_item=path,
            chunk_index=i,
            chunk_count=chunk_count,
            row_start=start,
            row_end=end,
        )
        for i, (start, end) in enumerate(chunks)
    ]


# Register built-in introspectors
register_introspector(".parquet", introspect_parquet)


def detect_introspector(source_item: Any) -> Callable[[str, int], list[ShardSpec]] | None:
    """Detect appropriate introspector based on file extension.

    Args:
        source_item: Source item (typically a file path string)

    Returns:
        Introspector function if detected, None otherwise
    """
    if not isinstance(source_item, str):
        return None

    for ext, introspector in INTROSPECTORS.items():
        if source_item.endswith(ext):
            return introspector

    return None


def _fuse_operations(operations: list[Operation]) -> list[PhysicalStage]:
    """Fuse operations into physical stages.

    Operations are fused together when they can be executed without a shuffle
    boundary. Shuffle boundaries occur at:
    - ReshardOp
    - GroupByShuffleReduceOp
    - ReduceGlobalOp

    Args:
        operations: List of logical operations

    Returns:
        List of PhysicalStages with fused operations
    """
    if not operations:
        return []

    stages: list[PhysicalStage] = []
    fusible_buffer: list[Operation] = []

    def is_fusible(op: Operation) -> bool:
        return not isinstance(op, (ReshardOp, GroupByShuffleReduceOp, ReduceGlobalOp))

    def flush_buffer() -> None:
        if not fusible_buffer:
            return
        # Always create a FusedMapOp, even for single operations
        stages.append(PhysicalStage(operations=[FusedMapOp(operations=fusible_buffer[:])]))
        fusible_buffer.clear()

    for op in operations:
        if is_fusible(op):
            fusible_buffer.append(op)
        else:
            flush_buffer()
            # Non-fusible operations get their own stage
            is_shuffle = isinstance(op, (GroupByShuffleReduceOp, ReduceGlobalOp))
            stages.append(PhysicalStage(operations=[op], is_shuffle_boundary=is_shuffle))

    flush_buffer()
    return stages


def _expand_source_shards(
    source: Iterable[Any],
    operations: list[Operation],
    hints: ExecutionHint,
) -> list[ShardSpec]:
    """Create ShardSpecs from source, potentially expanding via introspection.

    If the first operation is FlatMapOp and source items are file paths with
    a registered introspector, AND source_chunk_bytes is set, the source will
    be expanded into multiple chunks per source file. Each source file becomes
    one logical shard (same shard_idx) with potentially multiple chunks.

    Args:
        source: Source iterable
        operations: List of operations
        hints: Execution hints

    Returns:
        List of ShardSpecs with shard_idx properly assigned
    """
    source_list = list(source)

    # Check if we should expand shards
    should_expand = (
        hints.source_chunk_bytes > 0
        and operations
        and isinstance(operations[0], FlatMapOp)
        and source_list
        and detect_introspector(source_list[0]) is not None
    )

    if not should_expand:
        # No expansion: 1:1 source item to shard, assign sequential shard_idx
        return [ShardSpec(source_item=item, shard_idx=idx) for idx, item in enumerate(source_list)]

    # Expand shards via introspection
    # Each source item gets its own shard_idx, but may have multiple chunks
    introspector = detect_introspector(source_list[0])
    assert introspector is not None  # We checked above

    expanded_specs: list[ShardSpec] = []
    current_shard_idx = 0
    total_chunks = 0

    for item in source_list:
        item_introspector = detect_introspector(item)
        if item_introspector is not None:
            item_specs = item_introspector(item, hints.source_chunk_bytes)
            # All chunks from this source get the same shard_idx
            for spec in item_specs:
                spec.shard_idx = current_shard_idx
            expanded_specs.extend(item_specs)
            total_chunks += len(item_specs)
        else:
            # No introspector for this item, use as-is
            expanded_specs.append(ShardSpec(source_item=item, shard_idx=current_shard_idx))
            total_chunks += 1
        current_shard_idx += 1

    logger.info(
        f"Expanded {len(source_list)} source items into {total_chunks} chunks "
        f"across {current_shard_idx} logical shards "
        f"(target {hints.source_chunk_bytes} bytes per chunk)"
    )

    return expanded_specs


def compute_plan(dataset: Dataset, hints: ExecutionHint = ExecutionHint()) -> PhysicalPlan:
    """Compute physical execution plan from logical dataset.

    This function transforms the logical Dataset (source + operations) into a
    PhysicalPlan that can be executed by the Backend. The planning process:

    1. Inspects source items to determine shard boundaries
    2. If source_chunk_bytes is set and first op is FlatMapOp on files,
       expands source items into multiple shards via introspection
    3. Fuses operations into stages separated by shuffle boundaries

    Args:
        dataset: Dataset to plan
        hints: Execution hints controlling chunking behavior

    Returns:
        PhysicalPlan ready for execution
    """
    # Create shard specs, potentially expanding via introspection
    shard_specs = _expand_source_shards(dataset.source, dataset.operations, hints)

    # Fuse operations into stages
    stages = _fuse_operations(dataset.operations)

    return PhysicalPlan(shard_specs=shard_specs, stages=stages)
