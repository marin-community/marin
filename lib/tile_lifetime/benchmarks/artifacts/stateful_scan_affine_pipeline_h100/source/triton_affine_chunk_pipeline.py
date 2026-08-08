# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic three-stage H100 pipeline for factored affine state scans.

The physical interface is intentionally recurrence-neutral.  It consumes the
diagonal, left/right low-rank, additive, residual-scale, and read factors
recovered from tensor algebra.  The implementation is organized as:

``AffineIntraChunkPrepare -> AffineStateScan -> AffineReadout``.

The stages are structurally inspired by the preparation, state-scan, and
readout split in flash-linear-attention commit
``9c8e42e762fce087c27b673af4922795d9edb85e``.  No code in this module imports,
recognizes, or dispatches through that project or a named recurrent model.
Unlike a model-specific kernel, the same emitter accepts scalar or per-key
diagonals and any bounded update rank supported by the summary tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl

from tile_lifetime.plan import StateTransitionStructure
from tile_lifetime.stateful_scan_recovery import RecoveredAffineStateUpdate

NUMERICAL_CONTRACT = "bounded_reassociation"
UPSTREAM_INSPIRATION_REVISION = "9c8e42e762fce087c27b673af4922795d9edb85e"
UPSTREAM_INSPIRATION_FILES = (
    "fla/ops/gated_delta_rule/chunk_fwd.py",
    "fla/ops/common/chunk_delta_h.py",
    "fla/ops/common/chunk_o.py",
)


class AffinePhysicalStage(StrEnum):
    """Compiler-visible stages in the reusable affine chunk skeleton."""

    INTRA_CHUNK_PREPARE = "affine_intra_chunk_prepare"
    STATE_SCAN = "affine_state_scan"
    READOUT = "affine_readout"


@dataclass(frozen=True)
class AffineChunkInputs:
    """Canonical padded factors shared by all three physical stages."""

    read: torch.Tensor
    diagonal: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    additive: torch.Tensor
    residual_scale: torch.Tensor
    original_length: int
    chunk_size: int
    update_rank: int

    @property
    def chunk_count(self) -> int:
        return self.read.shape[1] // self.chunk_size


@dataclass(frozen=True)
class AffineChunkBuffers:
    """Bounded materializations forwarded between physical stages."""

    triangular_inverse: torch.Tensor
    transformed_left: torch.Tensor
    transformed_right: torch.Tensor
    final_diagonal: torch.Tensor
    solved_right: torch.Tensor
    solved_additive: torch.Tensor
    chunk_state: torch.Tensor
    projected_residual: torch.Tensor
    output: torch.Tensor
    numerical_contract: str = NUMERICAL_CONTRACT

    @property
    def preparation_bytes(self) -> int:
        tensors = (
            self.triangular_inverse,
            self.transformed_left,
            self.transformed_right,
            self.final_diagonal,
            self.solved_right,
            self.solved_additive,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def forwarded_bytes(self) -> int:
        tensors = (self.chunk_state, self.projected_residual)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def canonicalize_affine_chunk_inputs(
    recovery: RecoveredAffineStateUpdate,
    read: torch.Tensor,
    diagonal: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    additive: torch.Tensor,
    residual_scale: torch.Tensor,
    *,
    chunk_size: int,
) -> AffineChunkInputs:
    """Validate and pad generic recovered affine factors for chunk execution."""
    if recovery.transition_structure is not StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK:
        raise ValueError("affine chunk execution requires diagonal-plus-low-rank state structure")
    tensors = (read, diagonal, left, right, additive, residual_scale)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("affine chunk execution requires CUDA tensors")
    if chunk_size not in (16, 32, 64):
        raise ValueError("the first physical skeleton supports chunk sizes 16, 32, and 64")
    if read.ndim != 4 or diagonal.shape != read.shape:
        raise ValueError("read and diagonal must have shape [B,T,H,K]")
    batch, length, heads, key_dimension = read.shape
    update_rank = recovery.maximum_low_rank
    if update_rank not in (1, 2):
        raise ValueError("the first physical skeleton supports update rank 1 or 2")
    summary_rank = chunk_size * update_rank
    if summary_rank > 64:
        raise ValueError("the first physical skeleton supports at most 64 summary factors")
    vector_shape = (batch, length, heads, update_rank, key_dimension)
    if left.shape != vector_shape or right.shape != vector_shape:
        raise ValueError(f"left and right factors must have shape {vector_shape}")
    if additive.ndim != 5 or additive.shape[:4] != (batch, length, heads, update_rank):
        raise ValueError("additive factors must have shape [B,T,H,R,V]")
    if residual_scale.shape != (batch, length, heads, update_rank):
        raise ValueError("residual scale must have shape [B,T,H,R]")
    if read.dtype is not torch.bfloat16 or left.dtype is not torch.bfloat16 or right.dtype is not torch.bfloat16:
        raise ValueError("read and low-rank vector factors must use BF16 physical storage")

    padded_length = triton.cdiv(length, chunk_size) * chunk_size
    sequence_padding = padded_length - length
    if sequence_padding:
        read = functional.pad(read, (0, 0, 0, 0, 0, sequence_padding))
        diagonal = functional.pad(diagonal, (0, 0, 0, 0, 0, sequence_padding), value=1.0)
        left = functional.pad(left, (0, 0, 0, 0, 0, 0, 0, sequence_padding))
        right = functional.pad(right, (0, 0, 0, 0, 0, 0, 0, sequence_padding))
        additive = functional.pad(additive, (0, 0, 0, 0, 0, 0, 0, sequence_padding))
        residual_scale = functional.pad(residual_scale, (0, 0, 0, 0, 0, sequence_padding))

    return AffineChunkInputs(
        read=read.contiguous(),
        diagonal=diagonal.contiguous(),
        left=left.contiguous(),
        right=right.contiguous(),
        additive=additive.contiguous(),
        residual_scale=residual_scale.contiguous(),
        original_length=length,
        chunk_size=chunk_size,
        update_rank=update_rank,
    )


def allocate_affine_chunk_buffers(inputs: AffineChunkInputs, state: torch.Tensor) -> AffineChunkBuffers:
    """Allocate bounded stage outputs independently of any model name."""
    batch, _, heads, key_dimension = inputs.read.shape
    value_dimension = state.shape[-1]
    chunk_count = inputs.chunk_count
    summary_rank = inputs.chunk_size * inputs.update_rank
    if state.shape != (batch, heads, key_dimension, value_dimension) or state.dtype is not torch.float32:
        raise ValueError("state must be FP32 [B,H,K,V]")
    device = state.device
    return AffineChunkBuffers(
        triangular_inverse=torch.empty(
            (batch, chunk_count, heads, summary_rank, summary_rank), dtype=torch.bfloat16, device=device
        ),
        transformed_left=torch.empty(
            (batch, chunk_count, heads, summary_rank, key_dimension), dtype=torch.bfloat16, device=device
        ),
        transformed_right=torch.empty(
            (batch, chunk_count, heads, summary_rank, key_dimension), dtype=torch.bfloat16, device=device
        ),
        final_diagonal=torch.empty((batch, chunk_count, heads, key_dimension), dtype=torch.float32, device=device),
        solved_right=torch.empty(
            (batch, chunk_count, heads, summary_rank, key_dimension), dtype=torch.bfloat16, device=device
        ),
        solved_additive=torch.empty(
            (batch, chunk_count, heads, summary_rank, value_dimension), dtype=torch.bfloat16, device=device
        ),
        chunk_state=torch.empty(
            (batch, chunk_count, heads, key_dimension, value_dimension), dtype=torch.bfloat16, device=device
        ),
        projected_residual=torch.empty(
            (batch, chunk_count, heads, summary_rank, value_dimension), dtype=torch.bfloat16, device=device
        ),
        output=torch.empty((batch, inputs.read.shape[1], heads, value_dimension), dtype=torch.bfloat16, device=device),
    )


@triton.jit
def _affine_triangular_inverse(
    diagonal,
    left,
    right,
    residual_scale,
    inverse,
    transformed_left_output,
    final_diagonal,
    length,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    chunk_size: tl.constexpr,
    update_rank: tl.constexpr,
    summary_rank: tl.constexpr,
):
    program = tl.program_id(0)
    head = program % heads
    chunk_batch = program // heads
    chunk_count = tl.cdiv(length, chunk_size)
    chunk = chunk_batch % chunk_count
    batch = chunk_batch // chunk_count

    rank_offsets = tl.arange(0, summary_rank)
    key_offsets = tl.arange(0, key_dimension)
    positions = rank_offsets // update_rank
    prefix = tl.zeros((summary_rank, key_dimension), dtype=tl.float32)
    running = tl.full((key_dimension,), 1.0, dtype=tl.float32)
    for position in tl.static_range(0, chunk_size):
        token = chunk * chunk_size + position
        diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
        running *= tl.load(diagonal + diagonal_offsets).to(tl.float32)
        prefix += tl.where(positions[:, None] == position, running[None, :], 0.0)

    token_offsets = chunk * chunk_size + positions
    factor_offsets = (
        ((batch * length + token_offsets[:, None]) * heads + head) * update_rank + rank_offsets[:, None] % update_rank
    ) * key_dimension + key_offsets[None, :]
    left_values = tl.load(left + factor_offsets).to(tl.float32) / prefix
    scale_offsets = ((batch * length + token_offsets) * heads + head) * update_rank + rank_offsets % update_rank
    scale = tl.load(residual_scale + scale_offsets).to(tl.float32)
    right_values = tl.load(right + factor_offsets).to(tl.float32) * prefix * scale[:, None]
    interactions = tl.dot(right_values.to(tl.bfloat16), tl.trans(left_values.to(tl.bfloat16)), out_dtype=tl.float32)
    row = rank_offsets[:, None]
    column = rank_offsets[None, :]
    prior_position = positions[:, None] > positions[None, :]
    triangular = tl.where(prior_position, interactions, 0.0)
    inverse_values = tl.where(row == column, 1.0, 0.0)
    for row_index in tl.static_range(1, summary_rank):
        triangular_row = tl.sum(tl.where(row == row_index, triangular, 0.0), axis=0)
        candidate = -tl.sum(triangular_row[:, None] * inverse_values, axis=0)
        candidate = tl.where(rank_offsets < row_index, candidate, tl.where(rank_offsets == row_index, 1.0, 0.0))
        inverse_values = tl.where(row == row_index, candidate[None, :], inverse_values)

    inverse_base = program * summary_rank * summary_rank
    inverse_offsets = inverse_base + row * summary_rank + column
    tl.store(inverse + inverse_offsets, inverse_values.to(tl.bfloat16))
    transformed_left_offsets = (
        program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
    )
    tl.store(transformed_left_output + transformed_left_offsets, left_values.to(tl.bfloat16))
    diagonal_output_offsets = program * key_dimension + key_offsets
    tl.store(final_diagonal + diagonal_output_offsets, running)


@triton.jit
def _invert_unit_lower_16(strict_lower):
    offsets = tl.arange(0, 16)
    row = offsets[:, None]
    inverse = -strict_lower
    for row_index in tl.static_range(2, 16):
        source_row = tl.sum(tl.where(row == row_index, -strict_lower, 0.0), axis=0)
        source_row = tl.where(offsets < row_index, source_row, 0.0)
        solved_row = source_row + tl.sum(source_row[:, None] * inverse, axis=0)
        inverse = tl.where(row == row_index, solved_row[None, :], inverse)
    return inverse + tl.where(row == offsets[None, :], 1.0, 0.0)


@triton.jit
def _affine_triangular_inverse_64_rank_1(
    diagonal,
    left,
    right,
    residual_scale,
    inverse,
    transformed_left_output,
    final_diagonal,
    length,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    block_k: tl.constexpr,
):
    """Four-block inverse template for a generic rank-64 affine summary."""
    program = tl.program_id(0)
    head = program % heads
    chunk_batch = program // heads
    chunk_count = tl.cdiv(length, 64)
    chunk = chunk_batch % chunk_count
    batch = chunk_batch // chunk_count
    block_offsets = tl.arange(0, 16)
    row = block_offsets[:, None]
    column = block_offsets[None, :]
    strict_lower = row > column
    token_0 = chunk * 64 + block_offsets
    token_1 = chunk * 64 + 16 + block_offsets
    token_2 = chunk * 64 + 32 + block_offsets
    token_3 = chunk * 64 + 48 + block_offsets
    scale_0 = tl.load(residual_scale + (batch * length + token_0) * heads + head).to(tl.float32)
    scale_1 = tl.load(residual_scale + (batch * length + token_1) * heads + head).to(tl.float32)
    scale_2 = tl.load(residual_scale + (batch * length + token_2) * heads + head).to(tl.float32)
    scale_3 = tl.load(residual_scale + (batch * length + token_3) * heads + head).to(tl.float32)
    matrix_00 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_11 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_22 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_33 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_10 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_20 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_21 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_30 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_31 = tl.zeros((16, 16), dtype=tl.float32)
    matrix_32 = tl.zeros((16, 16), dtype=tl.float32)
    for key_block in tl.static_range(0, (key_dimension + block_k - 1) // block_k):
        key_offsets = key_block * block_k + tl.arange(0, block_k)
        key_valid = key_offsets < key_dimension
        running = tl.full((block_k,), 1.0, dtype=tl.float32)
        prefix_0 = tl.zeros((16, block_k), dtype=tl.float32)
        prefix_1 = tl.zeros((16, block_k), dtype=tl.float32)
        prefix_2 = tl.zeros((16, block_k), dtype=tl.float32)
        prefix_3 = tl.zeros((16, block_k), dtype=tl.float32)
        for position in tl.static_range(0, 16):
            token = chunk * 64 + position
            diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
            running *= tl.load(diagonal + diagonal_offsets, mask=key_valid, other=1.0).to(tl.float32)
            prefix_0 += tl.where(block_offsets[:, None] == position, running[None, :], 0.0)
        for position in tl.static_range(0, 16):
            token = chunk * 64 + 16 + position
            diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
            running *= tl.load(diagonal + diagonal_offsets, mask=key_valid, other=1.0).to(tl.float32)
            prefix_1 += tl.where(block_offsets[:, None] == position, running[None, :], 0.0)
        for position in tl.static_range(0, 16):
            token = chunk * 64 + 32 + position
            diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
            running *= tl.load(diagonal + diagonal_offsets, mask=key_valid, other=1.0).to(tl.float32)
            prefix_2 += tl.where(block_offsets[:, None] == position, running[None, :], 0.0)
        for position in tl.static_range(0, 16):
            token = chunk * 64 + 48 + position
            diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
            running *= tl.load(diagonal + diagonal_offsets, mask=key_valid, other=1.0).to(tl.float32)
            prefix_3 += tl.where(block_offsets[:, None] == position, running[None, :], 0.0)
        factor_0 = ((batch * length + token_0[:, None]) * heads + head) * key_dimension + key_offsets[None, :]
        factor_1 = ((batch * length + token_1[:, None]) * heads + head) * key_dimension + key_offsets[None, :]
        factor_2 = ((batch * length + token_2[:, None]) * heads + head) * key_dimension + key_offsets[None, :]
        factor_3 = ((batch * length + token_3[:, None]) * heads + head) * key_dimension + key_offsets[None, :]
        left_0 = tl.load(left + factor_0, mask=key_valid[None, :], other=0.0).to(tl.float32) / prefix_0
        left_1 = tl.load(left + factor_1, mask=key_valid[None, :], other=0.0).to(tl.float32) / prefix_1
        left_2 = tl.load(left + factor_2, mask=key_valid[None, :], other=0.0).to(tl.float32) / prefix_2
        left_3 = tl.load(left + factor_3, mask=key_valid[None, :], other=0.0).to(tl.float32) / prefix_3
        right_0 = tl.load(right + factor_0, mask=key_valid[None, :], other=0.0).to(tl.float32)
        right_1 = tl.load(right + factor_1, mask=key_valid[None, :], other=0.0).to(tl.float32)
        right_2 = tl.load(right + factor_2, mask=key_valid[None, :], other=0.0).to(tl.float32)
        right_3 = tl.load(right + factor_3, mask=key_valid[None, :], other=0.0).to(tl.float32)
        right_0 *= prefix_0 * scale_0[:, None]
        right_1 *= prefix_1 * scale_1[:, None]
        right_2 *= prefix_2 * scale_2[:, None]
        right_3 *= prefix_3 * scale_3[:, None]
        left_0_bf16 = left_0.to(tl.bfloat16)
        left_1_bf16 = left_1.to(tl.bfloat16)
        left_2_bf16 = left_2.to(tl.bfloat16)
        left_3_bf16 = left_3.to(tl.bfloat16)
        right_0_bf16 = right_0.to(tl.bfloat16)
        right_1_bf16 = right_1.to(tl.bfloat16)
        right_2_bf16 = right_2.to(tl.bfloat16)
        right_3_bf16 = right_3.to(tl.bfloat16)
        matrix_00 += tl.dot(right_0_bf16, tl.trans(left_0_bf16))
        matrix_11 += tl.dot(right_1_bf16, tl.trans(left_1_bf16))
        matrix_22 += tl.dot(right_2_bf16, tl.trans(left_2_bf16))
        matrix_33 += tl.dot(right_3_bf16, tl.trans(left_3_bf16))
        matrix_10 += tl.dot(right_1_bf16, tl.trans(left_0_bf16))
        matrix_20 += tl.dot(right_2_bf16, tl.trans(left_0_bf16))
        matrix_21 += tl.dot(right_2_bf16, tl.trans(left_1_bf16))
        matrix_30 += tl.dot(right_3_bf16, tl.trans(left_0_bf16))
        matrix_31 += tl.dot(right_3_bf16, tl.trans(left_1_bf16))
        matrix_32 += tl.dot(right_3_bf16, tl.trans(left_2_bf16))
        left_base = program * 64 * key_dimension
        left_offsets_0 = left_base + row * key_dimension + key_offsets[None, :]
        left_offsets_1 = left_base + (16 + row) * key_dimension + key_offsets[None, :]
        left_offsets_2 = left_base + (32 + row) * key_dimension + key_offsets[None, :]
        left_offsets_3 = left_base + (48 + row) * key_dimension + key_offsets[None, :]
        tl.store(transformed_left_output + left_offsets_0, left_0_bf16, mask=key_valid[None, :])
        tl.store(transformed_left_output + left_offsets_1, left_1_bf16, mask=key_valid[None, :])
        tl.store(transformed_left_output + left_offsets_2, left_2_bf16, mask=key_valid[None, :])
        tl.store(transformed_left_output + left_offsets_3, left_3_bf16, mask=key_valid[None, :])
        tl.store(final_diagonal + program * key_dimension + key_offsets, running, mask=key_valid)

    matrix_00 = tl.where(strict_lower, matrix_00, 0.0)
    matrix_11 = tl.where(strict_lower, matrix_11, 0.0)
    matrix_22 = tl.where(strict_lower, matrix_22, 0.0)
    matrix_33 = tl.where(strict_lower, matrix_33, 0.0)
    inverse_00 = _invert_unit_lower_16(matrix_00)
    inverse_11 = _invert_unit_lower_16(matrix_11)
    inverse_22 = _invert_unit_lower_16(matrix_22)
    inverse_33 = _invert_unit_lower_16(matrix_33)
    inverse_10 = -tl.dot(tl.dot(inverse_11, matrix_10, input_precision="tf32"), inverse_00, input_precision="tf32")
    inverse_21 = -tl.dot(tl.dot(inverse_22, matrix_21, input_precision="tf32"), inverse_11, input_precision="tf32")
    inverse_32 = -tl.dot(tl.dot(inverse_33, matrix_32, input_precision="tf32"), inverse_22, input_precision="tf32")
    inverse_20 = -tl.dot(
        inverse_22,
        tl.dot(matrix_20, inverse_00, input_precision="tf32") + tl.dot(matrix_21, inverse_10, input_precision="tf32"),
        input_precision="tf32",
    )
    inverse_31 = -tl.dot(
        inverse_33,
        tl.dot(matrix_31, inverse_11, input_precision="tf32") + tl.dot(matrix_32, inverse_21, input_precision="tf32"),
        input_precision="tf32",
    )
    inverse_30 = -tl.dot(
        inverse_33,
        tl.dot(matrix_30, inverse_00, input_precision="tf32")
        + tl.dot(matrix_31, inverse_10, input_precision="tf32")
        + tl.dot(matrix_32, inverse_20, input_precision="tf32"),
        input_precision="tf32",
    )

    inverse_base = program * 64 * 64
    block_row = block_offsets[:, None]
    block_column = block_offsets[None, :]
    inverse_offsets_00 = inverse_base + block_row * 64 + block_column
    inverse_offsets_10 = inverse_base + (16 + block_row) * 64 + block_column
    inverse_offsets_11 = inverse_base + (16 + block_row) * 64 + 16 + block_column
    inverse_offsets_20 = inverse_base + (32 + block_row) * 64 + block_column
    inverse_offsets_21 = inverse_base + (32 + block_row) * 64 + 16 + block_column
    inverse_offsets_22 = inverse_base + (32 + block_row) * 64 + 32 + block_column
    inverse_offsets_30 = inverse_base + (48 + block_row) * 64 + block_column
    inverse_offsets_31 = inverse_base + (48 + block_row) * 64 + 16 + block_column
    inverse_offsets_32 = inverse_base + (48 + block_row) * 64 + 32 + block_column
    inverse_offsets_33 = inverse_base + (48 + block_row) * 64 + 48 + block_column
    tl.store(inverse + inverse_offsets_00, inverse_00.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_10, inverse_10.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_11, inverse_11.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_20, inverse_20.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_21, inverse_21.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_22, inverse_22.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_30, inverse_30.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_31, inverse_31.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_32, inverse_32.to(tl.bfloat16))
    tl.store(inverse + inverse_offsets_33, inverse_33.to(tl.bfloat16))


@triton.jit
def _affine_transform_factors(
    diagonal,
    left,
    right,
    residual_scale,
    transformed_left,
    transformed_right,
    final_diagonal,
    length,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    chunk_size: tl.constexpr,
    update_rank: tl.constexpr,
    summary_rank: tl.constexpr,
    block_k: tl.constexpr,
):
    key_block = tl.program_id(0)
    program = tl.program_id(1)
    head = program % heads
    chunk_batch = program // heads
    chunk_count = tl.cdiv(length, chunk_size)
    chunk = chunk_batch % chunk_count
    batch = chunk_batch // chunk_count
    rank_offsets = tl.arange(0, summary_rank)
    positions = rank_offsets // update_rank
    key_offsets = key_block * block_k + tl.arange(0, block_k)
    key_valid = key_offsets < key_dimension
    prefix = tl.zeros((summary_rank, block_k), dtype=tl.float32)
    running = tl.full((block_k,), 1.0, dtype=tl.float32)
    for position in tl.static_range(0, chunk_size):
        token = chunk * chunk_size + position
        diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
        running *= tl.load(diagonal + diagonal_offsets, mask=key_valid, other=1.0).to(tl.float32)
        prefix += tl.where(positions[:, None] == position, running[None, :], 0.0)
    token_offsets = chunk * chunk_size + positions
    factor_offsets = (
        ((batch * length + token_offsets[:, None]) * heads + head) * update_rank + rank_offsets[:, None] % update_rank
    ) * key_dimension + key_offsets[None, :]
    scale_offsets = ((batch * length + token_offsets) * heads + head) * update_rank + rank_offsets % update_rank
    scale = tl.load(residual_scale + scale_offsets).to(tl.float32)
    left_values = tl.load(left + factor_offsets, mask=key_valid[None, :], other=0.0).to(tl.float32) / prefix
    right_values = tl.load(right + factor_offsets, mask=key_valid[None, :], other=0.0).to(tl.float32)
    right_values *= prefix * scale[:, None]
    output_offsets = (
        program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
    )
    tl.store(transformed_left + output_offsets, left_values.to(tl.bfloat16), mask=key_valid[None, :])
    tl.store(transformed_right + output_offsets, right_values.to(tl.bfloat16), mask=key_valid[None, :])
    tl.store(final_diagonal + program * key_dimension + key_offsets, running, mask=key_valid)


@triton.jit
def _affine_interaction_matrix(
    transformed_left,
    transformed_right,
    interaction,
    key_dimension: tl.constexpr,
    update_rank: tl.constexpr,
    summary_rank: tl.constexpr,
):
    program = tl.program_id(0)
    rank_offsets = tl.arange(0, summary_rank)
    key_offsets = tl.arange(0, key_dimension)
    factor_offsets = (
        program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
    )
    left_values = tl.load(transformed_left + factor_offsets)
    right_values = tl.load(transformed_right + factor_offsets)
    values = tl.dot(right_values, tl.trans(left_values), out_dtype=tl.float32)
    positions = rank_offsets // update_rank
    values = tl.where(positions[:, None] > positions[None, :], values, 0.0)
    matrix_offsets = program * summary_rank * summary_rank + rank_offsets[:, None] * summary_rank + rank_offsets[None, :]
    tl.store(interaction + matrix_offsets, values.to(tl.bfloat16))


@triton.jit
def _affine_block_inverse_16(interaction):
    program = tl.program_id(0)
    offsets = tl.arange(0, 16)
    matrix_offsets = program * 16 * 16 + offsets[:, None] * 16 + offsets[None, :]
    matrix = tl.load(interaction + matrix_offsets).to(tl.float32)
    inverse = _invert_unit_lower_16(matrix)
    tl.store(interaction + matrix_offsets, inverse.to(tl.bfloat16))


@triton.jit
def _affine_block_inverse_32(interaction):
    program = tl.program_id(0)
    offsets = tl.arange(0, 16)
    row = offsets[:, None]
    column = offsets[None, :]
    base = program * 32 * 32
    offsets_00 = base + row * 32 + column
    offsets_10 = base + (16 + row) * 32 + column
    offsets_11 = base + (16 + row) * 32 + 16 + column
    matrix_00 = tl.load(interaction + offsets_00).to(tl.float32)
    matrix_10 = tl.load(interaction + offsets_10).to(tl.float32)
    matrix_11 = tl.load(interaction + offsets_11).to(tl.float32)
    inverse_00 = _invert_unit_lower_16(matrix_00)
    inverse_11 = _invert_unit_lower_16(matrix_11)
    inverse_10 = -tl.dot(tl.dot(inverse_11, matrix_10, input_precision="tf32"), inverse_00, input_precision="tf32")
    tl.store(interaction + offsets_00, inverse_00.to(tl.bfloat16))
    tl.store(interaction + offsets_10, inverse_10.to(tl.bfloat16))
    tl.store(interaction + offsets_11, inverse_11.to(tl.bfloat16))


@triton.jit
def _affine_block_inverse_64(interaction):
    program = tl.program_id(0)
    offsets = tl.arange(0, 16)
    row = offsets[:, None]
    column = offsets[None, :]
    base = program * 64 * 64
    offsets_00 = base + row * 64 + column
    offsets_10 = base + (16 + row) * 64 + column
    offsets_11 = base + (16 + row) * 64 + 16 + column
    offsets_20 = base + (32 + row) * 64 + column
    offsets_21 = base + (32 + row) * 64 + 16 + column
    offsets_22 = base + (32 + row) * 64 + 32 + column
    offsets_30 = base + (48 + row) * 64 + column
    offsets_31 = base + (48 + row) * 64 + 16 + column
    offsets_32 = base + (48 + row) * 64 + 32 + column
    offsets_33 = base + (48 + row) * 64 + 48 + column
    matrix_00 = tl.load(interaction + offsets_00).to(tl.float32)
    matrix_10 = tl.load(interaction + offsets_10).to(tl.float32)
    matrix_11 = tl.load(interaction + offsets_11).to(tl.float32)
    matrix_20 = tl.load(interaction + offsets_20).to(tl.float32)
    matrix_21 = tl.load(interaction + offsets_21).to(tl.float32)
    matrix_22 = tl.load(interaction + offsets_22).to(tl.float32)
    matrix_30 = tl.load(interaction + offsets_30).to(tl.float32)
    matrix_31 = tl.load(interaction + offsets_31).to(tl.float32)
    matrix_32 = tl.load(interaction + offsets_32).to(tl.float32)
    matrix_33 = tl.load(interaction + offsets_33).to(tl.float32)
    inverse_00 = _invert_unit_lower_16(matrix_00)
    inverse_11 = _invert_unit_lower_16(matrix_11)
    inverse_22 = _invert_unit_lower_16(matrix_22)
    inverse_33 = _invert_unit_lower_16(matrix_33)
    inverse_10 = -tl.dot(tl.dot(inverse_11, matrix_10, input_precision="tf32"), inverse_00, input_precision="tf32")
    inverse_21 = -tl.dot(tl.dot(inverse_22, matrix_21, input_precision="tf32"), inverse_11, input_precision="tf32")
    inverse_32 = -tl.dot(tl.dot(inverse_33, matrix_32, input_precision="tf32"), inverse_22, input_precision="tf32")
    inverse_20 = -tl.dot(
        inverse_22,
        tl.dot(matrix_20, inverse_00, input_precision="tf32") + tl.dot(matrix_21, inverse_10, input_precision="tf32"),
        input_precision="tf32",
    )
    inverse_31 = -tl.dot(
        inverse_33,
        tl.dot(matrix_31, inverse_11, input_precision="tf32") + tl.dot(matrix_32, inverse_21, input_precision="tf32"),
        input_precision="tf32",
    )
    inverse_30 = -tl.dot(
        inverse_33,
        tl.dot(matrix_30, inverse_00, input_precision="tf32")
        + tl.dot(matrix_31, inverse_10, input_precision="tf32")
        + tl.dot(matrix_32, inverse_20, input_precision="tf32"),
        input_precision="tf32",
    )
    tl.store(interaction + offsets_00, inverse_00.to(tl.bfloat16))
    tl.store(interaction + offsets_10, inverse_10.to(tl.bfloat16))
    tl.store(interaction + offsets_11, inverse_11.to(tl.bfloat16))
    tl.store(interaction + offsets_20, inverse_20.to(tl.bfloat16))
    tl.store(interaction + offsets_21, inverse_21.to(tl.bfloat16))
    tl.store(interaction + offsets_22, inverse_22.to(tl.bfloat16))
    tl.store(interaction + offsets_30, inverse_30.to(tl.bfloat16))
    tl.store(interaction + offsets_31, inverse_31.to(tl.bfloat16))
    tl.store(interaction + offsets_32, inverse_32.to(tl.bfloat16))
    tl.store(interaction + offsets_33, inverse_33.to(tl.bfloat16))


@triton.jit
def _affine_apply_inverse_right(
    inverse,
    transformed_right,
    solved_right,
    key_dimension: tl.constexpr,
    summary_rank: tl.constexpr,
):
    program = tl.program_id(0)
    rank_offsets = tl.arange(0, summary_rank)
    key_offsets = tl.arange(0, key_dimension)
    matrix_offsets = program * summary_rank * summary_rank + rank_offsets[:, None] * summary_rank + rank_offsets[None, :]
    factor_offsets = (
        program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
    )
    inverse_values = tl.load(inverse + matrix_offsets)
    right_values = tl.load(transformed_right + factor_offsets)
    solved = -tl.dot(inverse_values, right_values, out_dtype=tl.float32)
    tl.store(solved_right + factor_offsets, solved.to(tl.bfloat16))


@triton.jit
def _affine_solve_right(
    inverse,
    diagonal,
    right,
    residual_scale,
    solved_right,
    length,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    chunk_size: tl.constexpr,
    update_rank: tl.constexpr,
    summary_rank: tl.constexpr,
):
    program = tl.program_id(0)
    head = program % heads
    chunk_batch = program // heads
    chunk_count = tl.cdiv(length, chunk_size)
    chunk = chunk_batch % chunk_count
    batch = chunk_batch // chunk_count
    rank_offsets = tl.arange(0, summary_rank)
    key_offsets = tl.arange(0, key_dimension)
    positions = rank_offsets // update_rank

    prefix = tl.zeros((summary_rank, key_dimension), dtype=tl.float32)
    running = tl.full((key_dimension,), 1.0, dtype=tl.float32)
    for position in tl.static_range(0, chunk_size):
        token = chunk * chunk_size + position
        diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
        running *= tl.load(diagonal + diagonal_offsets).to(tl.float32)
        prefix += tl.where(positions[:, None] == position, running[None, :], 0.0)
    token_offsets = chunk * chunk_size + positions
    factor_offsets = (
        ((batch * length + token_offsets[:, None]) * heads + head) * update_rank + rank_offsets[:, None] % update_rank
    ) * key_dimension + key_offsets[None, :]
    scale_offsets = ((batch * length + token_offsets) * heads + head) * update_rank + rank_offsets % update_rank
    scale = tl.load(residual_scale + scale_offsets).to(tl.float32)
    transformed_right = tl.load(right + factor_offsets).to(tl.float32) * prefix * scale[:, None]
    matrix_offsets = program * summary_rank * summary_rank + rank_offsets[:, None] * summary_rank + rank_offsets[None, :]
    inverse_values = tl.load(inverse + matrix_offsets)
    solved = -tl.dot(inverse_values, transformed_right.to(tl.bfloat16), out_dtype=tl.float32)
    output_offsets = (
        program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
    )
    tl.store(solved_right + output_offsets, solved.to(tl.bfloat16))


@triton.jit
def _affine_solve_additive(
    inverse,
    additive,
    residual_scale,
    solved_additive,
    length,
    heads: tl.constexpr,
    value_dimension: tl.constexpr,
    chunk_size: tl.constexpr,
    update_rank: tl.constexpr,
    summary_rank: tl.constexpr,
    block_v: tl.constexpr,
):
    program_v = tl.program_id(0)
    program = tl.program_id(1)
    head = program % heads
    chunk_batch = program // heads
    chunk_count = tl.cdiv(length, chunk_size)
    chunk = chunk_batch % chunk_count
    batch = chunk_batch // chunk_count
    rank_offsets = tl.arange(0, summary_rank)
    value_offsets = program_v * block_v + tl.arange(0, block_v)
    value_valid = value_offsets < value_dimension
    positions = rank_offsets // update_rank
    token_offsets = chunk * chunk_size + positions
    scale_offsets = ((batch * length + token_offsets) * heads + head) * update_rank + rank_offsets % update_rank
    scale = tl.load(residual_scale + scale_offsets).to(tl.float32)
    additive_offsets = (
        (((batch * length + token_offsets[:, None]) * heads + head) * update_rank + rank_offsets[:, None] % update_rank)
        * value_dimension
    ) + value_offsets[None, :]
    transformed_additive = tl.load(additive + additive_offsets, mask=value_valid[None, :], other=0.0).to(tl.float32)
    transformed_additive *= scale[:, None]
    matrix_offsets = program * summary_rank * summary_rank + rank_offsets[:, None] * summary_rank + rank_offsets[None, :]
    inverse_values = tl.load(inverse + matrix_offsets)
    solved = tl.dot(inverse_values, transformed_additive.to(tl.bfloat16), out_dtype=tl.float32)
    output_offsets = (
        program * summary_rank * value_dimension + rank_offsets[:, None] * value_dimension + value_offsets[None, :]
    )
    tl.store(solved_additive + output_offsets, solved.to(tl.bfloat16), mask=value_valid[None, :])


def affine_intra_chunk_prepare(
    inputs: AffineChunkInputs,
    buffers: AffineChunkBuffers,
    *,
    block_v: int = 32,
) -> None:
    """Derive bounded solved factors from generic affine inputs."""
    batch, length, heads, key_dimension = inputs.read.shape
    value_dimension = inputs.additive.shape[-1]
    programs = batch * inputs.chunk_count * heads
    summary_rank = inputs.chunk_size * inputs.update_rank
    block_k = 32
    _affine_transform_factors[(triton.cdiv(key_dimension, block_k), programs)](
        inputs.diagonal,
        inputs.left,
        inputs.right,
        inputs.residual_scale,
        buffers.transformed_left,
        buffers.transformed_right,
        buffers.final_diagonal,
        length,
        heads,
        key_dimension,
        inputs.chunk_size,
        inputs.update_rank,
        summary_rank,
        block_k,
        num_warps=4,
        num_stages=2,
    )
    _affine_interaction_matrix[(programs,)](
        buffers.transformed_left,
        buffers.transformed_right,
        buffers.triangular_inverse,
        key_dimension,
        inputs.update_rank,
        summary_rank,
        num_warps=4,
        num_stages=2,
    )
    if summary_rank == 16:
        _affine_block_inverse_16[(programs,)](
            buffers.triangular_inverse,
            num_warps=4,
            num_stages=2,
        )
    elif summary_rank == 32:
        _affine_block_inverse_32[(programs,)](
            buffers.triangular_inverse,
            num_warps=4,
            num_stages=2,
        )
    else:
        _affine_block_inverse_64[(programs,)](
            buffers.triangular_inverse,
            num_warps=4,
            num_stages=2,
        )
    _affine_apply_inverse_right[(programs,)](
        buffers.triangular_inverse,
        buffers.transformed_right,
        buffers.solved_right,
        key_dimension,
        summary_rank,
        num_warps=4,
        num_stages=2,
    )
    _affine_solve_additive[(triton.cdiv(value_dimension, block_v), programs)](
        buffers.triangular_inverse,
        inputs.additive,
        inputs.residual_scale,
        buffers.solved_additive,
        length,
        heads,
        value_dimension,
        inputs.chunk_size,
        inputs.update_rank,
        summary_rank,
        block_v,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _affine_state_scan(
    transformed_left,
    final_diagonal,
    solved_right,
    solved_additive,
    state,
    chunk_state,
    projected_residual,
    length,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    value_dimension: tl.constexpr,
    chunk_size: tl.constexpr,
    update_rank: tl.constexpr,
    summary_rank: tl.constexpr,
    block_v: tl.constexpr,
):
    program = tl.program_id(0)
    value_blocks = tl.cdiv(value_dimension, block_v)
    value_block = program % value_blocks
    head_batch = program // value_blocks
    head = head_batch % heads
    batch = head_batch // heads
    chunk_count = tl.cdiv(length, chunk_size)
    rank_offsets = tl.arange(0, summary_rank)
    key_offsets = tl.arange(0, key_dimension)
    value_offsets = value_block * block_v + tl.arange(0, block_v)
    value_valid = value_offsets < value_dimension
    state_offsets = (
        (batch * heads + head) * key_dimension * value_dimension
        + key_offsets[:, None] * value_dimension
        + value_offsets[None, :]
    )
    accumulator = tl.load(state + state_offsets, mask=value_valid[None, :], other=0.0).to(tl.float32)

    for chunk in tl.range(0, chunk_count):
        summary_program = (batch * chunk_count + chunk) * heads + head
        chunk_state_offsets = (
            summary_program * key_dimension * value_dimension
            + key_offsets[:, None] * value_dimension
            + value_offsets[None, :]
        )
        tl.store(chunk_state + chunk_state_offsets, accumulator.to(tl.bfloat16), mask=value_valid[None, :])

        right_offsets = (
            summary_program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
        )
        right_values = tl.load(solved_right + right_offsets)
        projection = tl.dot(right_values, accumulator.to(tl.bfloat16), out_dtype=tl.float32)
        additive_offsets = (
            summary_program * summary_rank * value_dimension
            + rank_offsets[:, None] * value_dimension
            + value_offsets[None, :]
        )
        projection += tl.load(solved_additive + additive_offsets, mask=value_valid[None, :], other=0.0)
        projection_offsets = (
            summary_program * summary_rank * value_dimension
            + rank_offsets[:, None] * value_dimension
            + value_offsets[None, :]
        )
        tl.store(projected_residual + projection_offsets, projection.to(tl.bfloat16), mask=value_valid[None, :])

        left_offsets = (
            summary_program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
        )
        left_values = tl.load(transformed_left + left_offsets)
        diagonal_offsets = summary_program * key_dimension + key_offsets
        diagonal_values = tl.load(final_diagonal + diagonal_offsets).to(tl.float32)
        summary_left = diagonal_values[None, :] * left_values.to(tl.float32)
        correction = tl.dot(tl.trans(summary_left.to(tl.bfloat16)), projection.to(tl.bfloat16), out_dtype=tl.float32)
        accumulator = diagonal_values[:, None] * accumulator + correction

    tl.store(state + state_offsets, accumulator, mask=value_valid[None, :])


def affine_state_scan(
    inputs: AffineChunkInputs,
    buffers: AffineChunkBuffers,
    state: torch.Tensor,
    *,
    block_v: int = 32,
) -> None:
    """Advance persistent state and forward chunk-start state and coefficients."""
    batch, length, heads, key_dimension = inputs.read.shape
    value_dimension = state.shape[-1]
    summary_rank = inputs.chunk_size * inputs.update_rank
    value_blocks = triton.cdiv(value_dimension, block_v)
    _affine_state_scan[(batch * heads * value_blocks,)](
        buffers.transformed_left,
        buffers.final_diagonal,
        buffers.solved_right,
        buffers.solved_additive,
        state,
        buffers.chunk_state,
        buffers.projected_residual,
        length,
        heads,
        key_dimension,
        value_dimension,
        inputs.chunk_size,
        inputs.update_rank,
        summary_rank,
        block_v,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _affine_readout(
    read,
    diagonal,
    transformed_left,
    chunk_state,
    projected_residual,
    output,
    length,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    value_dimension: tl.constexpr,
    chunk_size: tl.constexpr,
    update_rank: tl.constexpr,
    summary_rank: tl.constexpr,
    block_v: tl.constexpr,
):
    value_block = tl.program_id(0)
    summary_program = tl.program_id(1)
    head = summary_program % heads
    chunk_batch = summary_program // heads
    chunk_count = tl.cdiv(length, chunk_size)
    chunk = chunk_batch % chunk_count
    batch = chunk_batch // chunk_count
    token_offsets = tl.arange(0, chunk_size)
    rank_offsets = tl.arange(0, summary_rank)
    key_offsets = tl.arange(0, key_dimension)
    value_offsets = value_block * block_v + tl.arange(0, block_v)
    value_valid = value_offsets < value_dimension
    positions = rank_offsets // update_rank

    prefix_by_token = tl.zeros((chunk_size, key_dimension), dtype=tl.float32)
    running = tl.full((key_dimension,), 1.0, dtype=tl.float32)
    for position in tl.static_range(0, chunk_size):
        token = chunk * chunk_size + position
        diagonal_offsets = ((batch * length + token) * heads + head) * key_dimension + key_offsets
        running *= tl.load(diagonal + diagonal_offsets).to(tl.float32)
        prefix_by_token += tl.where(token_offsets[:, None] == position, running[None, :], 0.0)

    global_tokens = chunk * chunk_size + token_offsets
    read_offsets = ((batch * length + global_tokens[:, None]) * heads + head) * key_dimension + key_offsets[None, :]
    scaled_read = tl.load(read + read_offsets).to(tl.float32) * prefix_by_token
    left_offsets = (
        summary_program * summary_rank * key_dimension + rank_offsets[:, None] * key_dimension + key_offsets[None, :]
    )
    left_values = tl.load(transformed_left + left_offsets)

    state_offsets = (
        summary_program * key_dimension * value_dimension
        + key_offsets[:, None] * value_dimension
        + value_offsets[None, :]
    )
    state_values = tl.load(chunk_state + state_offsets, mask=value_valid[None, :], other=0.0)
    result = tl.dot(scaled_read.to(tl.bfloat16), state_values, out_dtype=tl.float32)
    coefficients = tl.dot(scaled_read.to(tl.bfloat16), tl.trans(left_values), out_dtype=tl.float32)
    coefficients = tl.where(token_offsets[:, None] >= positions[None, :], coefficients, 0.0)
    projection_offsets = (
        summary_program * summary_rank * value_dimension
        + rank_offsets[:, None] * value_dimension
        + value_offsets[None, :]
    )
    projection = tl.load(projected_residual + projection_offsets, mask=value_valid[None, :], other=0.0)
    result += tl.dot(coefficients.to(tl.bfloat16), projection, out_dtype=tl.float32)
    output_offsets = ((batch * length + global_tokens[:, None]) * heads + head) * value_dimension + value_offsets[
        None, :
    ]
    tl.store(output + output_offsets, result.to(tl.bfloat16), mask=value_valid[None, :])


def affine_readout(
    inputs: AffineChunkInputs,
    buffers: AffineChunkBuffers,
    *,
    block_v: int = 32,
) -> torch.Tensor:
    """Reconstruct per-position outputs from forwarded generic affine state."""
    batch, length, heads, key_dimension = inputs.read.shape
    value_dimension = buffers.output.shape[-1]
    programs = batch * inputs.chunk_count * heads
    summary_rank = inputs.chunk_size * inputs.update_rank
    _affine_readout[(triton.cdiv(value_dimension, block_v), programs)](
        inputs.read,
        inputs.diagonal,
        buffers.transformed_left,
        buffers.chunk_state,
        buffers.projected_residual,
        buffers.output,
        length,
        heads,
        key_dimension,
        value_dimension,
        inputs.chunk_size,
        inputs.update_rank,
        summary_rank,
        block_v,
        num_warps=4,
        num_stages=2,
    )
    return buffers.output[:, : inputs.original_length]


def execute_affine_chunk_pipeline(
    inputs: AffineChunkInputs,
    buffers: AffineChunkBuffers,
    state: torch.Tensor,
    *,
    block_v: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run all compiler-owned physical stages in dependency order."""
    affine_intra_chunk_prepare(inputs, buffers, block_v=block_v)
    affine_state_scan(inputs, buffers, state, block_v=block_v)
    return affine_readout(inputs, buffers, block_v=block_v), state
