# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legalize generic relation-driven streaming reductions for SM100.

This module is deliberately CUDA independent.  It checks that a generic
``Relation -> Contract -> Map -> Fold -> Contract`` program can be implemented
by the first bounded SM100 templates.  The optional CuTe emitter consumes the
result; it does not receive a workload or architecture name.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.h100_streaming_lowering import LoweredScoreMap, lower_score_map
from tile_lifetime.relation import RelationPlan
from tile_lifetime.streaming_attention import AttentionScoreAxis, StreamingAttentionProgram
from tile_lifetime.tensor_program import ScalarExpressionKind


class SM100RelationOrientation(StrEnum):
    """Side of the binary relation that owns physical task traversal."""

    LEFT_MAJOR = "left_major"
    RIGHT_MAJOR = "right_major"


@dataclass(frozen=True)
class SM100RoutedSchedule:
    """One finite SM100 tensor-core and producer-consumer schedule."""

    orientation: SM100RelationOrientation
    packed_left_rows: int
    right_block_size: int
    query_stages: int
    score_stages: int
    output_stages: int
    right_stages: int
    relation_metadata_stages: int
    threads: int
    split_probability_rows: int
    partial_merge_tile_rows: int
    partial_merge_threads: int
    partial_state_representation: str

    def __post_init__(self) -> None:
        if self.packed_left_rows != 128 or self.right_block_size != 128:
            raise ValueError("the initial SM100 templates require 128x128 logical score tiles")
        if self.threads != 512:
            raise ValueError("the initial SM100 contraction/fold template requires 512 threads")
        if (
            min(
                self.query_stages,
                self.score_stages,
                self.output_stages,
                self.right_stages,
                self.relation_metadata_stages,
            )
            <= 0
        ):
            raise ValueError("all bounded pipeline depths must be positive")
        if self.split_probability_rows <= 0 or self.split_probability_rows >= self.right_block_size:
            raise ValueError("the probability split must lie inside the right tile")
        if self.split_probability_rows % 32:
            raise ValueError("the probability split must be aligned to 32 rows")
        if self.partial_state_representation not in {
            "max_sum_weighted_value",
            "log_normalizer_normalized_value",
        }:
            raise ValueError("unsupported attention partial-state representation")


@dataclass(frozen=True)
class SM100RoutedStreamingLowering:
    """Backend-neutral contract for a generated SM100 relation program."""

    score_map: LoweredScoreMap
    schedule: SM100RoutedSchedule
    head_group_size: int
    query_length: int
    key_length: int
    key_value_heads: int
    right_block_count: int
    selected_count: int
    output_scale: float
    relation: RelationPlan

    def edge_group(self, route_slot: int) -> int:
        """Return the relation group encoded by one generic edge slot."""
        if route_slot < 0 or route_slot >= self.relation.route_slots:
            raise ValueError(f"route slot {route_slot} is outside the relation domain")
        return route_slot // self.selected_count

    def edge_selected_slot(self, route_slot: int) -> int:
        """Return the selected-right-item position within an edge group."""
        if route_slot < 0 or route_slot >= self.relation.route_slots:
            raise ValueError(f"route slot {route_slot} is outside the relation domain")
        return route_slot % self.selected_count

    def right_task_key(self, route_slot: int, right_block: int) -> tuple[int, int]:
        """Build the physical right-major grouping key from generic edge metadata."""
        if right_block < 0 or right_block >= self.right_block_count:
            raise ValueError(f"right block {right_block} is outside the relation domain")
        return self.edge_group(route_slot), right_block

    def canonical_right_indices(self) -> np.ndarray:
        """Return destination-sorted edges as ``[group, left, selected]``.

        This is the canonical input to a right-major relation scheduler. It is
        derived only from the generic relation edge arrays; no attention or
        workload identity is required.
        """
        destination = self.relation.destination_item.reshape(
            self.query_length,
            self.key_value_heads,
            self.selected_count,
        )
        valid = self.relation.edge_valid.reshape(destination.shape)
        canonical = np.full(destination.shape, -1, dtype=np.int32)
        for left_item, group in np.ndindex(self.query_length, self.key_value_heads):
            selected = destination[left_item, group, valid[left_item, group]]
            canonical[left_item, group, : selected.size] = np.sort(selected, kind="stable")
        return np.transpose(canonical, (1, 0, 2))

    @property
    def query_tokens_per_task(self) -> int:
        """Number of query tokens packed with their GQA heads in one task."""
        return self.schedule.packed_left_rows // self.head_group_size

    def dump(self) -> str:
        """Explain the generic semantics and physical choices."""
        return "\n".join(
            (
                "SM100 routed streaming reduction",
                "  semantics: Relation -> QK Contract -> score Map -> DomainRestriction "
                "-> normalized-exp Fold -> PV Contract",
                f"  orientation: {self.schedule.orientation.value}",
                (
                    f"  domains: query={self.query_length} key={self.key_length} "
                    f"kv_heads={self.key_value_heads} right_blocks={self.right_block_count}"
                ),
                (
                    f"  relation: sources={self.relation.source_item_count} "
                    f"groups={self.key_value_heads} selected_per_group={self.selected_count} "
                    f"edge_slots={self.relation.route_slots} edges={self.relation.route_count}"
                ),
                (
                    f"  tile: packed_left={self.schedule.packed_left_rows} "
                    f"right={self.schedule.right_block_size} "
                    f"query_tokens_per_task={self.query_tokens_per_task}"
                ),
                (
                    f"  pipelines: query={self.schedule.query_stages} "
                    f"score={self.schedule.score_stages} output={self.schedule.output_stages} "
                    f"right={self.schedule.right_stages} metadata={self.schedule.relation_metadata_stages}"
                ),
                (
                    f"  score: scale={self.score_map.scale} causal={str(self.score_map.causal).lower()} "
                    f"softcap={self.score_map.softcap}"
                ),
                f"  partial state: {self.schedule.partial_state_representation}",
                "  external semantics: none",
            )
        )


def default_sm100_routed_schedules() -> tuple[SM100RoutedSchedule, ...]:
    """Return the bounded initial candidate set without selecting a winner."""
    common = {
        "packed_left_rows": 128,
        "right_block_size": 128,
        "query_stages": 2,
        "score_stages": 2,
        "output_stages": 2,
        "right_stages": 1,
        "relation_metadata_stages": 16,
        "threads": 512,
        "split_probability_rows": 96,
        "partial_merge_tile_rows": 8,
        "partial_merge_threads": 256,
        "partial_state_representation": "log_normalizer_normalized_value",
    }
    return tuple(
        SM100RoutedSchedule(orientation=orientation, **common)
        for orientation in (SM100RelationOrientation.LEFT_MAJOR, SM100RelationOrientation.RIGHT_MAJOR)
    )


def lower_sm100_routed_streaming_program(
    program: StreamingAttentionProgram,
    relation: RelationPlan,
    schedule: SM100RoutedSchedule,
) -> SM100RoutedStreamingLowering:
    """Prove one generic routed program legal for the initial SM100 skeleton."""
    query, key = program.qk.inputs
    value = program.pv.inputs[1]
    query_length = _axis_extent(query, AttentionScoreAxis.QUERY.value)
    key_length = _axis_extent(key, AttentionScoreAxis.KEY.value)
    query_heads = _axis_extent(query, AttentionScoreAxis.HEAD.value)
    key_value_heads = _axis_extent(key, "key_value_head")
    head_group_size = _head_group_size(program)

    if query.dtype.value != "bf16" or key.dtype.value != "bf16" or value.dtype.value != "bf16":
        raise ValueError("the initial SM100 contraction skeleton accepts BF16 operands")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("the initial natural-route prototype supports batch size one")
    if query.shape[-1] != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        raise ValueError("the initial SM100 contraction skeleton requires Q/K/V dimension 128")
    if head_group_size not in (1, 2, 4, 8, 16):
        raise ValueError("the initial SM100 packed-GQA skeleton supports head-group sizes 1/2/4/8/16")
    if query_heads != key_value_heads * head_group_size:
        raise ValueError("the recovered GQA index map does not cover all query heads")
    if key_length % schedule.right_block_size:
        raise ValueError("the initial SM100 relation requires a whole number of right blocks")
    if schedule.packed_left_rows % head_group_size:
        raise ValueError("packed left rows must contain a whole number of query tokens")

    right_block_count = key_length // schedule.right_block_size
    expected_sources = query_length
    expected_destinations = right_block_count
    if relation.source_item_count != expected_sources:
        raise ValueError(
            "the relation left domain must contain query tokens: "
            f"expected {expected_sources}, found {relation.source_item_count}"
        )
    if relation.destination_count != expected_destinations:
        raise ValueError(
            "the relation right domain must contain KV blocks: "
            f"expected {expected_destinations}, found {relation.destination_count}"
        )
    if relation.route_slots % key_value_heads:
        raise ValueError("relation edge slots must divide evenly into KV-head groups")
    selected_count = relation.route_slots // key_value_heads
    if selected_count not in (4, 8, 16, 32):
        raise ValueError("the initial bounded SM100 templates support selected counts 4/8/16/32")

    score_map = lower_score_map(program)
    if score_map.softcap is not None:
        raise ValueError("the first SM100 candidate does not yet legalize a score softcap")
    return SM100RoutedStreamingLowering(
        score_map=score_map,
        schedule=schedule,
        head_group_size=head_group_size,
        query_length=query_length,
        key_length=key_length,
        key_value_heads=key_value_heads,
        right_block_count=right_block_count,
        selected_count=selected_count,
        output_scale=_output_scale(program),
        relation=relation,
    )


def _axis_extent(value, label: str) -> int:
    matches = tuple(axis.extent for axis in value.axes if axis.label == label)
    if len(matches) != 1:
        raise ValueError(f"expected one {label!r} axis, found {len(matches)}")
    return matches[0]


def _head_group_size(program: StreamingAttentionProgram) -> int:
    key_maps = program.qk.index_maps_for_input(1)
    value_maps = program.pv.index_maps_for_input(1)
    if len(key_maps) != 1 or value_maps != key_maps:
        raise ValueError("SM100 GQA requires one shared query-head to KV-head index map")
    mapping = key_maps[0]
    if mapping.offset != 0 or mapping.modulus is not None:
        raise ValueError("SM100 packed GQA requires a floor-division head map")
    return mapping.divisor


def _output_scale(program: StreamingAttentionProgram) -> float:
    expression = program.finalize.expression
    output_scale = 1.0
    if expression.kind is ScalarExpressionKind.MULTIPLY:
        left, right = expression.operands
        constant = left if left.kind is ScalarExpressionKind.CONSTANT else right
        expression = right if constant is left else left
        if constant.kind is not ScalarExpressionKind.CONSTANT or constant.constant is None:
            raise ValueError("the SM100 finalizer only supports one scalar output multiplier")
        output_scale = float(constant.constant)
    if expression.kind is not ScalarExpressionKind.DIVIDE:
        raise ValueError("the SM100 finalizer must divide the weighted state by its normalized-exp sum")
    numerator, denominator = expression.operands
    expected = (program.state.weighted_value_accumulator.name, program.state.row_sum_exp.name)
    if (numerator.input_name, denominator.input_name) != expected:
        raise ValueError("the SM100 finalizer does not consume the recovered normalized-exp state")
    if not np.isfinite(output_scale):
        raise ValueError("the output scale must be finite")
    return output_scale
