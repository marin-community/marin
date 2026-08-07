# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for deterministic KV-major selected-slot waves.

This bounded first candidate uses one launch per selected slot. Each program
owns one relation edge and one query head, so a wave has exactly one writer for
every query-token/head state and needs no atomics or per-edge partial buffer.
Sorting edges by KV block only improves launch/L2 locality: it does not yet
stage a KV block once for all incident query CTAs. The FP32 online state remains
in global memory between waves. The kernel is limited to block-aligned BF16
causal self-attention with equal Q/K/V feature dimensions and D=64 or 128.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl
from kv_major_slot_waves import SlotWaveSchedule


@dataclass(frozen=True)
class DeviceSlotWave:
    """Device-resident edge list for one selected-slot kernel launch."""

    selected_slot: int
    query_blocks: torch.Tensor
    key_value_blocks: torch.Tensor

    @property
    def edge_count(self) -> int:
        """Number of edges in the wave."""
        return int(self.query_blocks.numel())


@triton.jit
def _slot_wave_update(
    query,
    key,
    value,
    query_blocks,
    key_value_blocks,
    row_max,
    row_sum_exp,
    weighted_value,
    sequence_length,
    scale,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_st: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    query_heads: tl.constexpr,
    key_value_heads: tl.constexpr,
    query_block_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dimension: tl.constexpr,
):
    edge = tl.program_id(0)
    query_head = tl.program_id(1)
    query_row_tile = tl.program_id(2)
    query_block = tl.load(query_blocks + edge)
    key_value_block = tl.load(key_value_blocks + edge)
    key_value_head = query_head // (query_heads // key_value_heads)

    rows = query_row_tile * block_m + tl.arange(0, block_m)
    columns = tl.arange(0, block_n)
    features = tl.arange(0, head_dimension)
    query_tokens = query_block * query_block_size + rows
    key_value_tokens = key_value_block * block_n + columns
    query_valid = (rows < query_block_size) & (query_tokens < sequence_length)
    key_value_valid = key_value_tokens < sequence_length

    query_tile = tl.load(
        query + query_tokens[:, None] * stride_qt + query_head * stride_qh + features[None, :] * stride_qd,
        mask=query_valid[:, None],
        other=0.0,
    )
    key_tile = tl.load(
        key + key_value_tokens[None, :] * stride_kt + key_value_head * stride_kh + features[:, None] * stride_kd,
        mask=key_value_valid[None, :],
        other=0.0,
    )
    scores = tl.dot(query_tile, key_tile) * scale
    score_valid = query_valid[:, None] & key_value_valid[None, :] & (query_tokens[:, None] >= key_value_tokens[None, :])
    scores = tl.where(score_valid, scores, -float("inf"))

    state_offsets = query_tokens * stride_st + query_head * stride_sh
    old_max = tl.load(row_max + state_offsets, mask=query_valid, other=-float("inf"))
    old_sum = tl.load(row_sum_exp + state_offsets, mask=query_valid, other=0.0)
    block_max = tl.max(scores, axis=1)
    new_max = tl.maximum(old_max, block_max)
    row_has_scores = query_valid & (tl.sum(score_valid.to(tl.int32), axis=1) > 0)
    safe_new_max = tl.where(row_has_scores, new_max, 0.0)
    old_delta = tl.where((old_sum > 0.0) & row_has_scores, old_max - safe_new_max, -float("inf"))
    old_scale = tl.where(row_has_scores, tl.exp(old_delta), 1.0)
    probabilities = tl.where(score_valid, tl.exp(scores - safe_new_max[:, None]), 0.0)
    new_sum = old_sum * old_scale + tl.sum(probabilities, axis=1)
    new_max = tl.where(row_has_scores, new_max, old_max)

    output_offsets = query_tokens[:, None] * stride_ot + query_head * stride_oh + features[None, :] * stride_od
    old_weighted_value = tl.load(weighted_value + output_offsets, mask=query_valid[:, None], other=0.0)
    value_tile = tl.load(
        value + key_value_tokens[:, None] * stride_vt + key_value_head * stride_vh + features[None, :] * stride_vd,
        mask=key_value_valid[:, None],
        other=0.0,
    )
    new_weighted_value = old_weighted_value * old_scale[:, None] + tl.dot(probabilities.to(tl.bfloat16), value_tile)
    tl.store(row_max + state_offsets, new_max, mask=query_valid)
    tl.store(row_sum_exp + state_offsets, new_sum, mask=query_valid)
    tl.store(weighted_value + output_offsets, new_weighted_value, mask=query_valid[:, None])


@triton.jit
def _finalize_attention(
    row_sum_exp,
    weighted_value,
    output,
    sequence_length,
    stride_st: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    query_heads: tl.constexpr,
    block_m: tl.constexpr,
    head_dimension: tl.constexpr,
):
    token_tile = tl.program_id(0)
    query_head = tl.program_id(1)
    rows = token_tile * block_m + tl.arange(0, block_m)
    features = tl.arange(0, head_dimension)
    valid = rows < sequence_length
    denominator = tl.load(row_sum_exp + rows * stride_st + query_head * stride_sh, mask=valid, other=1.0)
    offsets = rows[:, None] * stride_ot + query_head * stride_oh + features[None, :] * stride_od
    accumulator = tl.load(weighted_value + offsets, mask=valid[:, None], other=0.0)
    tl.store(output + offsets, accumulator / denominator[:, None], mask=valid[:, None])


def device_slot_waves(schedule: SlotWaveSchedule, device: torch.device) -> tuple[DeviceSlotWave, ...]:
    """Copy the preplanned bounded edge arrays to a CUDA device."""
    return tuple(
        DeviceSlotWave(
            selected_slot=wave.selected_slot,
            query_blocks=torch.from_numpy(np.ascontiguousarray(wave.query_blocks)).to(device=device),
            key_value_blocks=torch.from_numpy(np.ascontiguousarray(wave.key_value_blocks)).to(device=device),
        )
        for wave in schedule.waves
    )


def execute_slot_waves(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    waves: tuple[DeviceSlotWave, ...],
    *,
    block_size: int,
    query_tile_size: int,
    scale: float,
) -> torch.Tensor:
    """Run slot-ordered KV-major updates and finalize BF16 attention output."""
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("slot-wave Triton execution requires CUDA tensors")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError("the first slot-wave kernel supports BF16 Q/K/V only")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Q/K/V must have shapes [token, head, feature]")
    if query.shape[-1] != key.shape[-1] or value.shape != key.shape:
        raise ValueError("the first slot-wave kernel requires equal Q/K/V feature dimensions")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query heads must map evenly onto KV heads")
    if query.shape[0] != key.shape[0] or query.shape[0] % block_size:
        raise ValueError("the first slot-wave kernel requires equal block-aligned Q/KV lengths")
    if query.shape[-1] not in (64, 128):
        raise ValueError("the first slot-wave kernel supports head dimensions 64 and 128")
    if query_tile_size not in (16, 32, 64) or query_tile_size > block_size:
        raise ValueError("query tile size must be 16, 32, or 64 and no larger than the logical block")

    num_warps = 8 if query_tile_size == 64 else 4

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    sequence_length, query_heads, head_dimension = query.shape
    row_max = torch.full((sequence_length, query_heads), -torch.inf, dtype=torch.float32, device=query.device)
    row_sum_exp = torch.zeros_like(row_max)
    weighted_value = torch.zeros(
        (sequence_length, query_heads, head_dimension), dtype=torch.float32, device=query.device
    )
    output = torch.empty_like(query)

    for wave in waves:
        if wave.edge_count == 0:
            continue
        _slot_wave_update[(wave.edge_count, query_heads, triton.cdiv(block_size, query_tile_size))](
            query,
            key,
            value,
            wave.query_blocks,
            wave.key_value_blocks,
            row_max,
            row_sum_exp,
            weighted_value,
            sequence_length,
            scale,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *row_max.stride(),
            *weighted_value.stride(),
            query_heads,
            key.shape[1],
            block_size,
            query_tile_size,
            block_size,
            head_dimension,
            num_warps=num_warps,
            num_stages=2,
        )
    _finalize_attention[(triton.cdiv(sequence_length, query_tile_size), query_heads)](
        row_sum_exp,
        weighted_value,
        output,
        sequence_length,
        *row_sum_exp.stride(),
        *output.stride(),
        query_heads,
        query_tile_size,
        head_dimension,
        num_warps=num_warps,
    )
    return output
