# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legalize projected relation selection into bounded SM100 skeletons."""

from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.routed_attention import ProjectedBlockSelectionProgram


class SM100SelectionStrategy(StrEnum):
    """Where block scores cross the maximum-Fold/Selection boundary."""

    MATERIALIZED_BLOCK_SCORES = "materialized_block_scores"
    FUSED_STREAMING_TOP_K = "fused_streaming_top_k"


@dataclass(frozen=True)
class SM100SelectionSchedule:
    """Physical candidate for Contract, block maximum, and top-k selection."""

    strategy: SM100SelectionStrategy
    left_tile_size: int
    right_block_size: int
    feature_tile_size: int
    pipeline_stages: int
    matrix_threads: int
    selection_threads_per_row: int

    def __post_init__(self) -> None:
        if (self.left_tile_size, self.right_block_size, self.feature_tile_size) != (128, 128, 128):
            raise ValueError("the initial SM100 projected-selection skeleton requires 128x128x128 tiles")
        if self.pipeline_stages not in (1, 2, 3):
            raise ValueError("the initial SM100 projected-selection pipeline supports one to three stages")
        if self.matrix_threads <= 0 or self.matrix_threads % 32:
            raise ValueError("matrix worker count must be a positive number of warps")
        if self.selection_threads_per_row != 32:
            raise ValueError("the initial register top-k template uses one warp per logical row")


@dataclass(frozen=True)
class SM100ProjectedSelectionLowering:
    """Workload-neutral physical contract for relation-index generation."""

    program: ProjectedBlockSelectionProgram
    schedule: SM100SelectionSchedule
    right_block_count: int
    token_score_materialization_bytes: int
    block_score_materialization_bytes: int
    relation_index_bytes: int

    def dump(self) -> str:
        """Render the semantic and materialization decisions."""
        return "\n".join(
            (
                "SM100 projected relation selection",
                "  semantics: Contract(index Q/K) -> scale Map -> DomainRestriction "
                "-> block maximum Fold -> top-k Selection -> Relation",
                f"  strategy: {self.schedule.strategy.value}",
                (
                    f"  domains: left={self.program.source_count} groups={self.program.group_count} "
                    f"right_blocks={self.right_block_count} feature={self.program.relation_feature_count}"
                ),
                (
                    f"  materialization: token_scores={self.token_score_materialization_bytes} "
                    f"block_scores={self.block_score_materialization_bytes} "
                    f"relation_indices={self.relation_index_bytes}"
                ),
                f"  local block forced: {str(self.program.force_local_block).lower()}",
                "  external semantics: none",
            )
        )


def default_sm100_selection_schedules() -> tuple[SM100SelectionSchedule, ...]:
    """Return materialized and fused candidates without choosing by workload name."""
    common = {
        "left_tile_size": 128,
        "right_block_size": 128,
        "feature_tile_size": 128,
        "pipeline_stages": 2,
        "matrix_threads": 384,
        "selection_threads_per_row": 32,
    }
    return tuple(
        SM100SelectionSchedule(strategy=strategy, **common)
        for strategy in (
            SM100SelectionStrategy.MATERIALIZED_BLOCK_SCORES,
            SM100SelectionStrategy.FUSED_STREAMING_TOP_K,
        )
    )


def lower_sm100_projected_selection(
    program: ProjectedBlockSelectionProgram,
    schedule: SM100SelectionSchedule,
) -> SM100ProjectedSelectionLowering:
    """Prove generic projected block selection legal for one SM100 candidate."""
    if program.accumulation_dtype != "fp32":
        raise ValueError("the initial SM100 index Contract and maximum Fold accumulate in FP32")
    if program.relation_feature_count != schedule.feature_tile_size:
        raise ValueError("the initial SM100 index Contract requires relation feature dimension 128")
    if program.right_block_size != schedule.right_block_size:
        raise ValueError("the semantic Fold block must match the physical right tile")
    if program.selected_count not in (4, 8, 16, 32):
        raise ValueError("the initial SM100 top-k templates support k=4/8/16/32")
    if not program.force_local_block:
        raise ValueError("the primary selection candidate requires an explicit forced-local relation edge")

    right_block_count = program.right_block_count
    token_score_bytes = program.source_count * program.group_count * program.resolved_right_count * 4
    block_score_bytes = program.source_count * program.group_count * right_block_count * 4
    if schedule.strategy is SM100SelectionStrategy.FUSED_STREAMING_TOP_K:
        token_score_bytes = 0
    relation_index_bytes = program.source_count * program.group_count * program.selected_count * 4
    return SM100ProjectedSelectionLowering(
        program=program,
        schedule=schedule,
        right_block_count=right_block_count,
        token_score_materialization_bytes=token_score_bytes,
        block_score_materialization_bytes=block_score_bytes,
        relation_index_bytes=relation_index_bytes,
    )
