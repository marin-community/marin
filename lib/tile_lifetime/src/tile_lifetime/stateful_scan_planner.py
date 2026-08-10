# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Physical candidate generation for recovered affine state scans."""

from shuttle.ir import DType
from tile_lifetime.plan import (
    ChunkSummaryRepresentation,
    ScanNumericalContract,
    StatefulScanExecutionForm,
    StatefulScanSkeleton,
    StateTransitionStructure,
)
from tile_lifetime.stateful_scan_recovery import RecoveredAffineStateUpdate


def compile_affine_scan_candidates(
    recovery: RecoveredAffineStateUpdate,
    *,
    ordered_axis: str,
    length: int,
    state: str,
    state_shape: tuple[int, ...],
    state_dtype: DType,
    output: str,
    state_layout: str,
    chunk_sizes: tuple[int, ...],
) -> tuple[StatefulScanSkeleton, ...]:
    """Create a bounded schedule set from state-affine structure, not a model name."""
    if recovery.transition_structure is not StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK:
        raise ValueError("the first generated scan templates require diagonal-plus-low-rank state structure")
    if length <= 0 or any(dimension <= 0 for dimension in state_shape):
        raise ValueError("scan length and state dimensions must be positive")
    if not chunk_sizes or any(chunk_size <= 0 for chunk_size in chunk_sizes):
        raise ValueError("chunk sizes must be a non-empty tuple of positive integers")

    recurrent = StatefulScanSkeleton(
        name="affine_scan_recurrent",
        ordered_axis=ordered_axis,
        length=length,
        state=state,
        state_shape=state_shape,
        state_dtype=state_dtype,
        output=output,
        execution_form=StatefulScanExecutionForm.RECURRENT,
        chunk_size=1,
        summary_representation=ChunkSummaryRepresentation.NONE,
        transition_structure=recovery.transition_structure,
        maximum_update_rank=recovery.maximum_low_rank,
        backend="shuttle_affine_scan_recurrent_template",
        backend_revision=None,
        state_layout=state_layout,
        materialized_values=("persistent_state", "output"),
        numerical_contract=ScanNumericalContract.SOURCE_ORDERED,
        numerical_effect="Executes recovered diagonal and bounded-rank updates in source order with FP32 state.",
    )
    chunkwise = tuple(
        StatefulScanSkeleton(
            name=f"affine_scan_chunk_{chunk_size}",
            ordered_axis=ordered_axis,
            length=length,
            state=state,
            state_shape=state_shape,
            state_dtype=state_dtype,
            output=output,
            execution_form=StatefulScanExecutionForm.CHUNKWISE,
            chunk_size=chunk_size,
            summary_representation=ChunkSummaryRepresentation.FACTORED_AFFINE,
            transition_structure=recovery.transition_structure,
            maximum_update_rank=recovery.maximum_low_rank,
            backend="shuttle_factored_affine_scan_template",
            backend_revision=None,
            state_layout=state_layout,
            materialized_values=("persistent_state", "factored_chunk_summary", "output"),
            numerical_contract=ScanNumericalContract.BOUNDED_REASSOCIATION,
            numerical_effect=(
                "Builds bounded diagonal-plus-low-rank chunk factors and scans chunks in order; "
                "FP32 contractions are reassociated relative to token recurrence."
            ),
        )
        for chunk_size in chunk_sizes
    )
    return (recurrent, *chunkwise)
