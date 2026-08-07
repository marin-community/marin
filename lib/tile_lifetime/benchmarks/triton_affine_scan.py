# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated Triton skeleton for bounded-rank affine matrix-state scans.

The backend consumes physical factors recovered from tensor algebra. It does
not recognize or invoke a named recurrent architecture. The first skeleton
supports a diagonal transition followed by a bounded-rank correction:

    decayed = diagonal * state
    residual[r] = residual_scale[r] * (additive[r] - right[r]^T @ decayed)
    next = decayed + sum_r outer(left[r], residual[r])
    output = read^T @ next

All tensors use the expanded state-head domain. Scalar or grouped diagonal
generators are prepared by producer maps before this skeleton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tile_lifetime.plan import StateTransitionStructure
from tile_lifetime.stateful_scan_recovery import RecoveredAffineStateUpdate


@triton.jit
def _recurrent_factored_affine_scan(
    read,
    diagonal,
    left,
    right,
    additive,
    residual_scale,
    state,
    output,
    sequence_length,
    stride_rb: tl.constexpr,
    stride_rt: tl.constexpr,
    stride_rh: tl.constexpr,
    stride_rk: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dt: tl.constexpr,
    stride_dh: tl.constexpr,
    stride_dk: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lt: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lr: tl.constexpr,
    stride_lk: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vr: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_at: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_ar: tl.constexpr,
    stride_av: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ct: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cr: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_sv: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ov: tl.constexpr,
    heads: tl.constexpr,
    key_dimension: tl.constexpr,
    value_dimension: tl.constexpr,
    update_rank: tl.constexpr,
    block_v: tl.constexpr,
):
    program = tl.program_id(0)
    value_block = program % tl.cdiv(value_dimension, block_v)
    head_batch = program // tl.cdiv(value_dimension, block_v)
    head = head_batch % heads
    batch = head_batch // heads

    key_offsets = tl.arange(0, key_dimension)
    value_offsets = value_block * block_v + tl.arange(0, block_v)
    value_valid = value_offsets < value_dimension
    state_offsets = (
        batch * stride_sb + head * stride_sh + key_offsets[:, None] * stride_sk + value_offsets[None, :] * stride_sv
    )
    accumulator = tl.load(state + state_offsets, mask=value_valid[None, :], other=0.0)

    for position in tl.range(0, sequence_length):
        diagonal_values = tl.load(
            diagonal + batch * stride_db + position * stride_dt + head * stride_dh + key_offsets * stride_dk
        ).to(tl.float32)
        accumulator *= diagonal_values[:, None]

        correction = tl.zeros((key_dimension, block_v), dtype=tl.float32)
        for rank_index in tl.static_range(0, update_rank):
            right_values = tl.load(
                right
                + batch * stride_vb
                + position * stride_vt
                + head * stride_vh
                + rank_index * stride_vr
                + key_offsets * stride_vk
            ).to(tl.float32)
            prediction = tl.sum(accumulator * right_values[:, None], axis=0)
            additive_values = tl.load(
                additive
                + batch * stride_ab
                + position * stride_at
                + head * stride_ah
                + rank_index * stride_ar
                + value_offsets * stride_av,
                mask=value_valid,
                other=0.0,
            ).to(tl.float32)
            correction_scale = tl.load(
                residual_scale + batch * stride_cb + position * stride_ct + head * stride_ch + rank_index * stride_cr
            ).to(tl.float32)
            left_values = tl.load(
                left
                + batch * stride_lb
                + position * stride_lt
                + head * stride_lh
                + rank_index * stride_lr
                + key_offsets * stride_lk
            ).to(tl.float32)
            residual = correction_scale * (additive_values - prediction)
            correction += left_values[:, None] * residual[None, :]
        accumulator += correction

        read_values = tl.load(
            read + batch * stride_rb + position * stride_rt + head * stride_rh + key_offsets * stride_rk
        ).to(tl.float32)
        result = tl.sum(accumulator * read_values[:, None], axis=0)
        output_offsets = batch * stride_ob + position * stride_ot + head * stride_oh + value_offsets * stride_ov
        tl.store(output + output_offsets, result, mask=value_valid)

    tl.store(state + state_offsets, accumulator, mask=value_valid[None, :])


def execute_recurrent_affine_scan(
    recovery: RecoveredAffineStateUpdate,
    read: torch.Tensor,
    diagonal: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    additive: torch.Tensor,
    residual_scale: torch.Tensor,
    state: torch.Tensor,
    *,
    block_v: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Instantiate the generic recurrent skeleton from recovered affine factors."""
    if recovery.transition_structure is not StateTransitionStructure.DIAGONAL_PLUS_LOW_RANK:
        raise ValueError("the first generated skeleton requires diagonal-plus-low-rank state structure")
    tensors = (read, diagonal, left, right, additive, residual_scale, state)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("generated Triton scan execution requires CUDA tensors")
    if state.dtype != torch.float32:
        raise ValueError("the first generated skeleton requires FP32 persistent state")
    if read.ndim != 4 or diagonal.shape != read.shape:
        raise ValueError("read and expanded diagonal factors must have shape [B,T,H,K]")
    batch, length, heads, key_dimension = read.shape
    if state.shape[:3] != (batch, heads, key_dimension) or state.ndim != 4:
        raise ValueError("state must have shape [B,H,K,V] matching the read factors")
    value_dimension = state.shape[-1]
    expected_vector = (batch, length, heads, recovery.maximum_low_rank, key_dimension)
    if left.shape != expected_vector or right.shape != expected_vector:
        raise ValueError(f"left/right factors must have shape {expected_vector}")
    if additive.shape != (batch, length, heads, recovery.maximum_low_rank, value_dimension):
        raise ValueError("additive factors must have shape [B,T,H,R,V]")
    if residual_scale.shape != (batch, length, heads, recovery.maximum_low_rank):
        raise ValueError("residual scale must have shape [B,T,H,R]")
    if block_v not in (8, 16, 32):
        raise ValueError("block_v must be 8, 16, or 32")

    read = read.contiguous()
    diagonal = diagonal.contiguous()
    left = left.contiguous()
    right = right.contiguous()
    additive = additive.contiguous()
    residual_scale = residual_scale.contiguous()
    state = state.contiguous()
    output = torch.empty((batch, length, heads, value_dimension), dtype=read.dtype, device=read.device)
    value_blocks = triton.cdiv(value_dimension, block_v)
    _recurrent_factored_affine_scan[(batch * heads * value_blocks,)](
        read,
        diagonal,
        left,
        right,
        additive,
        residual_scale,
        state,
        output,
        length,
        *read.stride(),
        *diagonal.stride(),
        *left.stride(),
        *right.stride(),
        *additive.stride(),
        *residual_scale.stride(),
        *state.stride(),
        *output.stride(),
        heads,
        key_dimension,
        value_dimension,
        recovery.maximum_low_rank,
        block_v,
        num_warps=8,
        num_stages=2,
    )
    return output, state
