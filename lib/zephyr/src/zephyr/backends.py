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
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from fray.job import JobContext
from tqdm_loggable.auto import tqdm

from zephyr.dataset import Dataset
from zephyr.plan import (
    Chunk,
    ChunkHeader,
    ExecutionHint,
    Join,
    PhysicalOp,
    PhysicalPlan,
    SourceItem,
    StageContext,
    StageType,
    compute_plan,
    run_stage,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BackendConfig:
    """Configuration for backend execution.

    Attributes:
        max_parallelism: Maximum number of concurrent tasks
        dry_run: If True, show optimization plan without executing
    """

    max_parallelism: int = 1024
    dry_run: bool = False


@dataclass
class Shard:
    """Container for a logically grouped set of items split across multiple chunks."""

    idx: int  # Shard index (e.g., 0 of 50)
    chunks: list[Chunk]
    context: JobContext

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
    def from_single_ref(ref: Any, context: JobContext, idx: int, count: int) -> Shard:
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
    def __init__(self, context: JobContext, config: BackendConfig):
        """Initialize backend with execution context and configuration.

        Args:
            context: Execution context providing put/get/run/wait primitives
            config: Backend configuration
        """
        self.context = context
        self.config = config
        self.dry_run = config.dry_run

    def execute(self, dataset: Dataset[T], hints: ExecutionHint = ExecutionHint(), verbose: bool = False) -> Sequence[T]:
        """Execute a dataset, returning a sequence of results.

        Args:
            dataset: Dataset to execute
            hints: Execution hints (chunk_size, source_chunk_bytes, etc.)
            verbose: Print additional logging and optimization stats

        Returns:
            Sequence of results
        """
        plan = compute_plan(dataset, hints)

        if verbose:
            self._print_plan(dataset.operations, plan)

        if self.dry_run:
            return []

        return self.execute_plan(plan, hints)

    def execute_plan(self, plan: PhysicalPlan, hints: ExecutionHint = ExecutionHint()) -> Sequence:
        """Execute a pre-computed physical plan.

        Args:
            plan: Physical plan to execute
            hints: Execution hints

        Returns:
            Sequence of results
        """
        return list(self._execute_plan_impl(plan, hints))

    def _print_plan(self, original_ops: list, plan: PhysicalPlan) -> None:
        """Print the physical plan showing shard count and operation fusion.

        Args:
            original_ops: Original operation chain
            plan: Computed physical plan
        """
        total_physical_ops = sum(len(stage.operations) for stage in plan.stages)

        logger.info("\n=== Physical Execution Plan ===\n")
        logger.info(f"Shards: {plan.num_shards}")
        logger.info(f"Original operations: {len(original_ops)}")
        logger.info(f"Stages: {len(plan.stages)}")
        logger.info(f"Physical ops: {total_physical_ops}\n")

        logger.info("Original pipeline:")
        for i, op in enumerate(original_ops, 1):
            logger.info(f"  {i}. {op}")

        logger.info("\nPhysical stages:")
        for i, stage in enumerate(plan.stages, 1):
            op_names = [type(op).__name__ for op in stage.operations]
            stage_desc = " → ".join(op_names)

            hints = []
            if stage.stage_type == StageType.RESHARD:
                hints.append(f"reshard→{stage.output_shards}")
            if any(isinstance(op, Join) for op in stage.operations):
                hints.append("join")

            hint_str = f" [{', '.join(hints)}]" if hints else ""
            logger.info(f"  {i}. {stage_desc}{hint_str}")

        logger.info("\n=== End Plan ===\n")

    def _shards_from_source_items(self, source_items: list[SourceItem]) -> list[Shard]:
        """Create Shards from SourceItems, grouping by shard_idx.

        The `data` field of each SourceItem is passed directly to the first operation.
        For LoadFileOp pipelines, data is an InputFileSpec.
        For from_list pipelines, data is the raw source item.
        For flat_map(load_parquet) pattern, data is the file path string.

        Multiple SourceItems with the same shard_idx are grouped into a single Shard
        with multiple chunks (for intra-shard parallelism).

        Args:
            source_items: List of SourceItems from the plan

        Returns:
            List of Shards ready for processing, one per unique shard_idx
        """
        # Group by shard_idx
        items_by_shard: dict[int, list[SourceItem]] = defaultdict(list)
        for item in source_items:
            items_by_shard[item.shard_idx].append(item)

        # Create Shards, grouping chunks by shard_idx
        shards = []
        for shard_idx in sorted(items_by_shard.keys()):
            items = items_by_shard[shard_idx]

            chunks = []
            for item in items:
                # Pass the data field directly to the first operation
                chunks.append(Chunk(count=1, data=self.context.put([item.data])))

            shards.append(Shard(idx=shard_idx, chunks=chunks, context=self.context))

        return shards

    def _execute_plan_stages(self, shards: list[Shard], plan: PhysicalPlan, hints: ExecutionHint) -> list[Shard]:
        """Execute plan stages on shards.

        Args:
            shards: Input shards
            plan: Physical plan containing stages
            hints: Execution hints

        Returns:
            List of Shards after applying all stages
        """
        for stage in plan.stages:
            logger.info(f"Executing stage {stage}")
            shards = self._execute_stage(stage, shards, hints)
        return shards

    def _execute_stage(
        self,
        stage,
        shards: list[Shard],
        hints: ExecutionHint,
    ) -> list[Shard]:
        """Execute a single stage on shards."""
        # Reshard: just redistribute refs
        if stage.stage_type == StageType.RESHARD:
            return reshard_refs(shards, stage.output_shards or len(shards))

        # Compute aux shards for joins
        aux_shards_per_shard = self._compute_join_aux_shards(stage, shards, hints)

        # Single execution path - ForkChunks handles parallelism internally
        return self._execute_shard_parallel(stage.operations, shards, aux_shards_per_shard, hints)

    def _compute_join_aux_shards(
        self,
        stage,
        shards: list[Shard],
        hints: ExecutionHint,
    ) -> list[dict[int, list[Shard]]]:
        """Compute auxiliary shards for join operations in a stage."""
        all_right_shards: dict[int, list[Shard]] = {}

        for i, op in enumerate(stage.operations):
            if isinstance(op, Join) and op.right_plan is not None:
                right_shards = self._shards_from_source_items(op.right_plan.source_items)
                right_shards = self._execute_plan_stages(right_shards, op.right_plan, hints)

                if len(shards) != len(right_shards):
                    raise ValueError(
                        f"Sorted merge join requires equal shard counts. "
                        f"Left has {len(shards)} shards, right has {len(right_shards)} shards."
                    )
                all_right_shards[i] = right_shards

        # Build aux_shards for each left shard
        return [
            {join_idx: [right_shards[shard_idx]] for join_idx, right_shards in all_right_shards.items()}
            for shard_idx in range(len(shards))
        ]

    def _execute_plan_impl(self, plan: PhysicalPlan, hints: ExecutionHint) -> Iterator:
        """Execute a physical plan and materialize results.

        Args:
            plan: Physical plan to execute
            hints: Execution hints

        Yields:
            Results after applying all operations
        """
        # Create shards from source items
        shards = self._shards_from_source_items(plan.source_items)

        # Execute all stages
        shards = self._execute_plan_stages(shards, plan, hints)

        # Materialize results
        stage_names = []
        for stage in plan.stages:
            stage_names.extend(op.__class__.__name__ for op in stage.operations)
        desc = f"Materialize [{' → '.join(stage_names)}]"

        def materialize_all():
            for shard in shards:
                yield from shard

        yield from tqdm(materialize_all(), desc=desc, unit="shards", total=len(shards))

    def _run_tasks(
        self,
        contexts: list[StageContext],
        operations: list[PhysicalOp],
    ) -> dict[int, list[tuple[ChunkHeader, Any]]]:
        """Run stage tasks for contexts, return results grouped by output shard_idx.

        Args:
            contexts: List of StageContext to process
            operations: Physical operations to execute

        Returns:
            Dict mapping shard_idx -> list of (header, data_ref) tuples.
        """
        results_by_shard: dict[int, list[tuple[ChunkHeader, Any]]] = defaultdict(list)

        if not contexts:
            return results_by_shard

        active_gens: list[tuple[Any, StageContext]] = []
        queued = list(contexts)

        # Start initial batch
        while len(active_gens) < self.config.max_parallelism and queued:
            ctx = queued.pop(0)
            active_gens.append((self.context.run(run_stage, ctx, operations), ctx))

        # Process results
        while active_gens or queued:
            gen_objs = [g for g, _ in active_gens]
            ready, _ = self.context.wait(gen_objs, num_returns=1)

            for ready_gen in ready:
                # Find matching entry
                for g, ctx in active_gens:
                    if g is ready_gen:
                        try:
                            header = self.context.get(next(ready_gen))
                            data_ref = next(ready_gen)
                            results_by_shard[header.shard_idx].append((header, data_ref))
                        except StopIteration:
                            active_gens.remove((g, ctx))
                            if queued:
                                next_ctx = queued.pop(0)
                                active_gens.append((self.context.run(run_stage, next_ctx, operations), next_ctx))
                        break

        return results_by_shard

    def _execute_shard_parallel(
        self,
        operations: list[PhysicalOp],
        shards: list[Shard],
        aux_shards_per_shard: list[dict] | None,
        hints: ExecutionHint,
    ) -> list[Shard]:
        """Execute operations on shards with one task per shard.

        Args:
            operations: Physical operations to execute
            shards: List of input Shards
            aux_shards_per_shard: Optional list of aux_shards dicts, one per input shard.
            hints: Execution hints

        Returns:
            List of output Shards assembled from streamed chunks
        """
        if aux_shards_per_shard is None:
            aux_shards_per_shard = [{} for _ in range(len(shards))]
        elif len(shards) != len(aux_shards_per_shard):
            raise ValueError(f"Mismatch: {len(shards)} shards but {len(aux_shards_per_shard)} aux_shards entries")

        total = len(shards)

        contexts = [
            StageContext(
                shard=shard,
                shard_idx=shard_idx,
                total_shards=total,
                chunk_size=hints.chunk_size,
                aux_shards=aux_shards,
                execution_context=self.context,
            )
            for shard_idx, (shard, aux_shards) in enumerate(zip(shards, aux_shards_per_shard, strict=True))
        ]

        results = self._run_tasks(contexts, operations)

        # Use input shard count to preserve empty shards (important for joins)
        # TODO: the planner should just inform the controller about the number of output shards
        num_output_shards = total
        if results:
            # If scatter/groupby produces more shards, use that count
            max_result_idx = max(results.keys())
            if max_result_idx >= num_output_shards:
                num_output_shards = max_result_idx + 1

        shards = []
        for idx in range(num_output_shards):
            if idx not in results:
                shards.append(Shard(idx=idx, chunks=[], context=self.context))
            else:
                shards.append(
                    Shard(
                        idx=idx,
                        chunks=[Chunk(header.count, data_ref) for header, data_ref in results[idx]],
                        context=self.context,
                    )
                )

        return shards
