# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compiler-owned factored-chunk backend for bounded-rank affine scans.

The preparation step derives a bounded affine chunk summary from generic
recovered factors. The execution kernel applies those summaries in source
chunk order while retaining one state value block in registers. No named
recurrent architecture is recognized by this backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl

from tile_lifetime.plan import StateTransitionStructure
from tile_lifetime.stateful_scan_recovery import RecoveredAffineStateUpdate

NUMERICAL_CONTRACT = "bounded_reassociation"


@dataclass(frozen=True)
class TorchFactoredAffineChunks:
    """Physical chunk summaries consumed by the ordered GPU skeleton."""

    diagonal: torch.Tensor
    low_rank_left: torch.Tensor
    low_rank_right: torch.Tensor
    additive_coefficients: torch.Tensor
    transformed_read: torch.Tensor
    local_output: torch.Tensor
    original_length: int
    chunk_size: int
    update_rank: int
    numerical_contract: str = NUMERICAL_CONTRACT

    @property
    def materialized_bytes(self) -> int:
        tensors = (
            self.diagonal,
            self.low_rank_left,
            self.low_rank_right,
            self.additive_coefficients,
            self.transformed_read,
            self.local_output,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def prepare_factored_affine_chunks(
    recovery: RecoveredAffineStateUpdate,
    read: torch.Tensor,
    diagonal: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    additive: torch.Tensor,
    residual_scale: torch.Tensor,
    *,
    chunk_size: int = 64,
    summary_dtype: torch.dtype = torch.bfloat16,
) -> TorchFactoredAffineChunks:
    """Derive contraction-oriented chunk factors from one recovered update."""
    if recovery.transition_structure is not StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK:
        raise ValueError("factored chunk preparation requires diagonal-plus-low-rank state structure")
    tensors = (read, diagonal, left, right, additive, residual_scale)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("factored chunk preparation requires CUDA tensors")
    if summary_dtype is not torch.bfloat16:
        raise ValueError("the first factored chunk skeleton requires BF16 physical factors")
    if chunk_size not in (16, 32, 64):
        raise ValueError("chunk_size must be 16, 32, or 64")
    if read.ndim != 4 or diagonal.shape != read.shape:
        raise ValueError("read and diagonal must have shape [B,T,H,K]")
    batch, length, heads, key_dimension = read.shape
    update_rank = recovery.maximum_low_rank
    expected_vectors = (batch, length, heads, update_rank, key_dimension)
    if left.shape != expected_vectors or right.shape != expected_vectors:
        raise ValueError(f"left and right factors must have shape {expected_vectors}")
    if additive.ndim != 5 or additive.shape[:4] != (batch, length, heads, update_rank):
        raise ValueError("additive factors must have shape [B,T,H,R,V]")
    if residual_scale.shape != (batch, length, heads, update_rank):
        raise ValueError("residual scale must have shape [B,T,H,R]")

    chunk_count = triton.cdiv(length, chunk_size)
    padded_length = chunk_count * chunk_size
    sequence_padding = padded_length - length
    if sequence_padding:
        read = functional.pad(read, (0, 0, 0, 0, 0, sequence_padding))
        diagonal = functional.pad(diagonal, (0, 0, 0, 0, 0, sequence_padding), value=1.0)
        left = functional.pad(left, (0, 0, 0, 0, 0, 0, 0, sequence_padding))
        right = functional.pad(right, (0, 0, 0, 0, 0, 0, 0, sequence_padding))
        additive = functional.pad(additive, (0, 0, 0, 0, 0, 0, 0, sequence_padding))
        residual_scale = functional.pad(residual_scale, (0, 0, 0, 0, 0, sequence_padding))

    value_dimension = additive.shape[-1]
    q = read.float().view(batch, chunk_count, chunk_size, heads, key_dimension).permute(0, 1, 3, 2, 4)
    d = diagonal.float().view(batch, chunk_count, chunk_size, heads, key_dimension).permute(0, 1, 3, 2, 4)
    u = left.float().view(batch, chunk_count, chunk_size, heads, update_rank, key_dimension)
    u = u.permute(0, 1, 3, 2, 4, 5)
    v = right.float().view(batch, chunk_count, chunk_size, heads, update_rank, key_dimension)
    v = v.permute(0, 1, 3, 2, 4, 5)
    c = additive.float().view(batch, chunk_count, chunk_size, heads, update_rank, value_dimension)
    c = c.permute(0, 1, 3, 2, 4, 5)
    scale = residual_scale.float().view(batch, chunk_count, chunk_size, heads, update_rank)
    scale = scale.permute(0, 1, 3, 2, 4)

    prefix = torch.cumprod(d, dim=3)
    transformed_left = u / prefix.unsqueeze(-2)
    transformed_right = v * prefix.unsqueeze(-2) * scale.unsqueeze(-1)
    transformed_additive = c * scale.unsqueeze(-1)
    summary_rank = chunk_size * update_rank
    left_matrix = transformed_left.reshape(batch, chunk_count, heads, summary_rank, key_dimension).transpose(-1, -2)
    right_matrix = transformed_right.reshape(batch, chunk_count, heads, summary_rank, key_dimension).transpose(-1, -2)
    additive_matrix = transformed_additive.reshape(batch, chunk_count, heads, summary_rank, value_dimension)
    physical_left = left_matrix.to(summary_dtype)
    physical_right = right_matrix.to(summary_dtype)

    interactions = (physical_right.transpose(-1, -2) @ physical_left).float()
    positions = torch.arange(summary_rank, device=read.device) // update_rank
    prior_position = positions[:, None] > positions[None, :]
    triangular = interactions * prior_position
    triangular.diagonal(dim1=-2, dim2=-1).add_(1.0)
    solved_additive = torch.linalg.solve_triangular(triangular, additive_matrix, upper=False, unitriangular=True)
    solved_right = -torch.linalg.solve_triangular(
        triangular,
        right_matrix.transpose(-1, -2),
        upper=False,
        unitriangular=True,
    ).transpose(-1, -2)

    final_diagonal = prefix[:, :, :, -1]
    summary_left = final_diagonal.unsqueeze(-1) * physical_left
    scaled_read = q * prefix
    read_left = (scaled_read.to(summary_dtype) @ physical_left).float()
    visible_update = torch.arange(chunk_size, device=read.device)[:, None] >= positions[None, :]
    read_left *= visible_update
    read_solve = torch.linalg.solve_triangular(
        triangular.transpose(-1, -2),
        read_left.transpose(-1, -2),
        upper=True,
        unitriangular=True,
    ).transpose(-1, -2)
    transformed_read = scaled_read - read_solve.to(summary_dtype) @ physical_right.transpose(-1, -2)
    local_output = read_left.to(summary_dtype) @ solved_additive.to(summary_dtype)

    return TorchFactoredAffineChunks(
        diagonal=final_diagonal.contiguous(),
        low_rank_left=summary_left.to(summary_dtype).contiguous(),
        low_rank_right=solved_right.to(summary_dtype).contiguous(),
        additive_coefficients=solved_additive.to(summary_dtype).contiguous(),
        transformed_read=transformed_read.to(summary_dtype).contiguous(),
        local_output=local_output.to(summary_dtype).contiguous(),
        original_length=length,
        chunk_size=chunk_size,
        update_rank=update_rank,
    )


@triton.jit
def _ordered_factored_chunk_scan(
    diagonal,
    left,
    right,
    additive,
    read,
    local_output,
    state,
    output,
    chunk_count,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    value_dimension: tl.constexpr,
    chunk_size: tl.constexpr,
    summary_rank: tl.constexpr,
    block_v: tl.constexpr,
):
    program = tl.program_id(0)
    value_blocks = tl.cdiv(value_dimension, block_v)
    value_block = program % value_blocks
    head_batch = program // value_blocks
    head = head_batch % heads
    batch = head_batch // heads

    key_offsets = tl.arange(0, key_dimension)
    rank_offsets = tl.arange(0, summary_rank)
    chunk_offsets = tl.arange(0, chunk_size)
    value_offsets = value_block * block_v + tl.arange(0, block_v)
    value_valid = value_offsets < value_dimension
    state_base = (batch * heads + head) * key_dimension * value_dimension
    state_offsets = state_base + key_offsets[:, None] * value_dimension + value_offsets[None, :]
    accumulator = tl.load(state + state_offsets, mask=value_valid[None, :], other=0.0).to(tl.float32)

    for chunk in tl.range(0, chunk_count):
        summary_base = (batch * chunk_count + chunk) * heads + head
        read_offsets = (
            summary_base * chunk_size * key_dimension + chunk_offsets[:, None] * key_dimension + key_offsets[None, :]
        )
        read_values = tl.load(read + read_offsets)
        result = tl.dot(read_values, accumulator.to(tl.bfloat16), out_dtype=tl.float32)
        local_offsets = (
            summary_base * chunk_size * value_dimension
            + chunk_offsets[:, None] * value_dimension
            + value_offsets[None, :]
        )
        result += tl.load(local_output + local_offsets, mask=value_valid[None, :], other=0.0)
        output_offsets = (
            ((batch * chunk_count + chunk) * chunk_size + chunk_offsets[:, None]) * heads * value_dimension
            + head * value_dimension
            + value_offsets[None, :]
        )
        tl.store(output + output_offsets, result, mask=value_valid[None, :])

        right_offsets = (
            summary_base * key_dimension * summary_rank + key_offsets[None, :] * summary_rank + rank_offsets[:, None]
        )
        right_values = tl.load(right + right_offsets)
        projection = tl.dot(right_values, accumulator.to(tl.bfloat16), out_dtype=tl.float32)
        additive_offsets = (
            summary_base * summary_rank * value_dimension
            + rank_offsets[:, None] * value_dimension
            + value_offsets[None, :]
        )
        projection += tl.load(additive + additive_offsets, mask=value_valid[None, :], other=0.0)
        left_offsets = (
            summary_base * key_dimension * summary_rank + key_offsets[:, None] * summary_rank + rank_offsets[None, :]
        )
        left_values = tl.load(left + left_offsets)
        correction = tl.dot(left_values, projection.to(tl.bfloat16), out_dtype=tl.float32)
        diagonal_offsets = summary_base * key_dimension + key_offsets
        diagonal_values = tl.load(diagonal + diagonal_offsets).to(tl.float32)
        accumulator = diagonal_values[:, None] * accumulator + correction

    tl.store(state + state_offsets, accumulator, mask=value_valid[None, :])


def execute_ordered_factored_chunks(
    recovery: RecoveredAffineStateUpdate,
    summary: TorchFactoredAffineChunks,
    state: torch.Tensor,
    *,
    block_v: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute prepared chunk summaries in source chunk order."""
    if recovery.transition_structure is not StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK:
        raise ValueError("ordered factored execution requires diagonal-plus-low-rank state structure")
    if summary.numerical_contract != NUMERICAL_CONTRACT:
        raise ValueError(f"unsupported numerical contract: {summary.numerical_contract}")
    if not state.is_cuda or state.dtype != torch.float32 or state.ndim != 4:
        raise ValueError("state must be a CUDA FP32 tensor with shape [B,H,K,V]")
    batch, heads, key_dimension, value_dimension = state.shape
    chunk_count = summary.diagonal.shape[1]
    if summary.diagonal.shape != (batch, chunk_count, heads, key_dimension):
        raise ValueError("summary factors do not match the persistent state shape")
    summary_rank = summary.chunk_size * summary.update_rank
    if summary_rank & (summary_rank - 1):
        raise ValueError("the first ordered chunk kernel requires a power-of-two summary rank")
    if summary.low_rank_left.shape != (batch, chunk_count, heads, key_dimension, summary_rank):
        raise ValueError("invalid left summary factor shape")
    if block_v not in (16, 32, 64):
        raise ValueError("block_v must be 16, 32, or 64")

    state = state.contiguous()
    padded_output = torch.empty(
        (batch, chunk_count, summary.chunk_size, heads, value_dimension),
        dtype=summary.local_output.dtype,
        device=state.device,
    )
    value_blocks = triton.cdiv(value_dimension, block_v)
    _ordered_factored_chunk_scan[(batch * heads * value_blocks,)](
        summary.diagonal,
        summary.low_rank_left,
        summary.low_rank_right,
        summary.additive_coefficients,
        summary.transformed_read,
        summary.local_output,
        state,
        padded_output,
        chunk_count,
        heads,
        key_dimension,
        value_dimension,
        summary.chunk_size,
        summary_rank,
        block_v,
        num_warps=8,
        num_stages=2,
    )
    output = padded_output.view(batch, chunk_count * summary.chunk_size, heads, value_dimension)
    return output[:, : summary.original_length], state
