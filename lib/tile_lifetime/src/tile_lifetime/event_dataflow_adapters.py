# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive generic task dataflow from scheduled tensor programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventMemoryScope,
    MemoryVisibility,
    TaskAxis,
    TaskDependence,
    TaskFamily,
    TaskRelation,
    derive_event_tensor_plan,
)
from tile_lifetime.streaming_attention import AttentionScoreAxis, StreamingAttentionProgram


@dataclass(frozen=True)
class StreamingFoldTaskDataflow:
    """Task graph derived from one Contract/normalized-Fold/Contract program."""

    program: EventDataflowProgram
    qk_contract: TaskFamily
    fold_partial: TaskFamily
    pv_contract: TaskFamily
    finalize: TaskFamily
    row_tile_count: int
    fold_partition_count: int
    pipeline_depth: int


def streaming_fold_task_dataflow(
    program: StreamingAttentionProgram,
    *,
    visibility_scope: EventMemoryScope = EventMemoryScope.CTA,
) -> StreamingFoldTaskDataflow:
    """Mechanically tile a streaming normalized weighted Fold into task families."""
    row_axes = program.state.row_max.axes
    query_axes = tuple(axis for axis in row_axes if axis.label == AttentionScoreAxis.QUERY.value)
    if len(query_axes) != 1:
        raise ValueError("streaming Fold task derivation requires one query axis")
    fold_axes = tuple(axis for axis in program.qk.output.axes if axis not in row_axes)
    if len(fold_axes) != 1:
        raise ValueError("streaming Fold task derivation requires one Fold axis")
    query_axis = query_axes[0]
    fold_axis = fold_axes[0]
    row_tile_count = prod(
        ceil(axis.extent / program.schedule.query_tile_size) if axis == query_axis else axis.extent for axis in row_axes
    )
    fold_partition_count = ceil(fold_axis.extent / program.schedule.key_value_tile_size)
    tiled_axes = (TaskAxis("row_tile", row_tile_count), TaskAxis("fold_partition", fold_partition_count))
    row_axis = (TaskAxis("row_tile", row_tile_count),)
    qk_contract = TaskFamily(program.qk.name, tiled_axes, placement="matrix_workers")
    fold_partial = TaskFamily("normalized_exp_fold_partial", tiled_axes, placement="reduction_workers")
    pv_contract = TaskFamily(program.pv.name, tiled_axes, placement="matrix_workers")
    finalize = TaskFamily(program.finalize.name, row_axis, placement="reduction_workers")
    visibility = MemoryVisibility(visibility_scope)
    pointwise = tuple(
        ((row_tile, partition), (row_tile, partition))
        for row_tile in range(row_tile_count)
        for partition in range(fold_partition_count)
    )
    qk_to_fold = TaskDependence(TaskRelation.from_pairs(qk_contract, fold_partial, pointwise), visibility)
    fold_to_pv = TaskDependence(TaskRelation.from_pairs(fold_partial, pv_contract, pointwise), visibility)
    pv_to_finalize = TaskDependence(
        TaskRelation.from_pairs(
            pv_contract,
            finalize,
            tuple(
                ((row_tile, partition), (row_tile,))
                for row_tile in range(row_tile_count)
                for partition in range(fold_partition_count)
            ),
        ),
        visibility,
    )
    dependences = (qk_to_fold, fold_to_pv, pv_to_finalize)
    event_plans = tuple(
        derive_event_tensor_plan(
            dependence,
            name=f"{dependence.relation.source.name}_to_{dependence.relation.target.name}",
        )
        for dependence in dependences
    )
    dataflow = EventDataflowProgram(
        (qk_contract, fold_partial, pv_contract, finalize),
        dependences,
        event_plans,
    )
    return StreamingFoldTaskDataflow(
        program=dataflow,
        qk_contract=qk_contract,
        fold_partial=fold_partial,
        pv_contract=pv_contract,
        finalize=finalize,
        row_tile_count=row_tile_count,
        fold_partition_count=fold_partition_count,
        pipeline_depth=program.schedule.pipeline_depth,
    )
