# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw Sonic Triton gather/combine backend for local Grug MoE.

The Triton gather kernel is adapted from SonicMoE commit
cfbd65f39b980b85b878b3cccdacb09191e24993:
https://github.com/Dao-AILab/sonic-moe/blob/cfbd65f39b980b85b878b3cccdacb09191e24993/sonicmoe/functional/reduction_over_k_gather.py.
SonicMoE is also Apache-2.0.
"""

from collections.abc import Callable
import math
import os

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_MOE_OUTPUT,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    _zero_dropped_assignments,
    split_moe_w13_output,
)

try:
    import jax_triton as jt
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    jt = None
    triton = None
    tl = None


_DEFAULT_TRITON_CACHE_DIR = "/tmp/marin-triton-cache"


if triton is not None and tl is not None:

    @triton.jit
    def _sonic_token_gather_sum_kernel(
        x_ptr,  # (Mtotal, H)
        w_ptr,  # (Mtotal,)
        m_perm_ptr,  # (Mtotal,) int32
        m_offset_ptr,  # unused for fixed-K, kept to match Sonic metadata
        out_ptr,  # (T, H)
        m: tl.constexpr,
        t: tl.constexpr,
        h: tl.constexpr,
        max_k: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_outt: tl.constexpr,
        stride_outh: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
        w_is_none: tl.constexpr,
        is_varlen_k: tl.constexpr,
        accumulate_in_bf16: tl.constexpr,
    ):
        pid_t = tl.program_id(axis=0)
        t_idx = pid_t.to(tl.int64)

        if is_varlen_k:
            ms = tl.load(m_offset_ptr + t_idx).to(tl.int64)
            me = tl.load(m_offset_ptr + t_idx + 1).to(tl.int64)
            k_this_token = me - ms
        else:
            ms = max_k * t_idx
            k_this_token: tl.constexpr = max_k

        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            acc = tl.zeros([block_h], dtype=tl.float32)

            for k_tile in tl.range(tl.cdiv(k_this_token, block_k)):
                k_offset = k_tile * block_k
                k_idx = (k_offset + tl.arange(0, block_k)).to(tl.int64)
                k_mask = k_idx < k_this_token
                m_abs = ms + k_idx
                perm_idx = tl.load(m_perm_ptr + m_abs, mask=k_mask, other=0).to(tl.int64)
                valid_perm = perm_idx < m

                x_ptrs = x_ptr + perm_idx[:, None] * stride_xm + h_idx[None, :] * stride_xh
                x_mask = k_mask[:, None] & valid_perm[:, None] & h_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                if w_is_none:
                    update = tl.sum(x_vals, axis=0)
                else:
                    w_vals = tl.load(w_ptr + m_abs, mask=k_mask, other=0.0).to(tl.float32)
                    update = tl.sum(x_vals * w_vals[:, None], axis=0)
                if accumulate_in_bf16:
                    acc = (acc + update).to(tl.bfloat16).to(tl.float32)
                else:
                    acc += update

            out_ptrs = out_ptr + t_idx * stride_outt + h_idx * stride_outh
            tl.store(out_ptrs, acc, mask=h_mask)

    @triton.jit
    def _sonic_token_gather_sum_bwd_kernel(
        dout_ptr,  # (T, H)
        x_ptr,  # (Mtotal, H)
        w_ptr,  # (Mtotal,)
        m_perm_ptr,  # (Mtotal,) int32
        m_offset_ptr,  # unused for fixed-K, kept to match Sonic metadata
        dx_ptr,  # (Mtotal, H)
        dw_ptr,  # (Mtotal,)
        m: tl.constexpr,
        t: tl.constexpr,
        h: tl.constexpr,
        max_k: tl.constexpr,
        stride_doutt: tl.constexpr,
        stride_douth: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        block_h: tl.constexpr,
    ):
        assignment = tl.program_id(axis=0)
        token = assignment // max_k

        perm_idx = tl.load(m_perm_ptr + assignment).to(tl.int64)
        valid_perm = perm_idx < m
        weight = tl.load(w_ptr + assignment).to(tl.float32)

        dw = 0.0
        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            dout = tl.load(
                dout_ptr + token * stride_doutt + h_idx * stride_douth,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + perm_idx * stride_xm + h_idx * stride_xh,
                mask=valid_perm & h_mask,
                other=0.0,
            ).to(tl.float32)

            tl.store(
                dx_ptr + perm_idx * stride_xm + h_idx * stride_xh,
                dout * weight,
                mask=valid_perm & h_mask,
            )
            dw += tl.sum(dout * x, axis=0)
        tl.store(dw_ptr + assignment, dw)

    @triton.jit
    def _sonic_token_gather_sum_two_buffers_kernel(
        x0_ptr,  # (M0, H)
        x1_ptr,  # (M1, H)
        w_ptr,  # (T * K,)
        m_perm_ptr,  # (T * K,) int32, indexing the logical concatenation of x0 and x1
        out_ptr,  # (T, H)
        m0: tl.constexpr,
        m1: tl.constexpr,
        h: tl.constexpr,
        max_k: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_outt: tl.constexpr,
        stride_outh: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_t = tl.program_id(axis=0)
        t_idx = pid_t.to(tl.int64)
        ms = max_k * t_idx

        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            acc = tl.zeros([block_h], dtype=tl.float32)

            for k_tile in tl.range(tl.cdiv(max_k, block_k)):
                k_offset = k_tile * block_k
                k_idx = (k_offset + tl.arange(0, block_k)).to(tl.int64)
                k_mask = k_idx < max_k
                assignment = ms + k_idx
                position = tl.load(m_perm_ptr + assignment, mask=k_mask, other=m0 + m1).to(tl.int64)
                in_first = position < m0
                in_second = (position >= m0) & (position < m0 + m1)
                first_position = tl.where(in_first, position, 0)
                second_position = tl.where(in_second, position - m0, 0)

                x0_ptrs = x0_ptr + first_position[:, None] * stride_xm + h_idx[None, :] * stride_xh
                x1_ptrs = x1_ptr + second_position[:, None] * stride_xm + h_idx[None, :] * stride_xh
                x0 = tl.load(
                    x0_ptrs,
                    mask=k_mask[:, None] & in_first[:, None] & h_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                x1 = tl.load(
                    x1_ptrs,
                    mask=k_mask[:, None] & in_second[:, None] & h_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                x = tl.where(in_first[:, None], x0, x1)
                weight = tl.load(w_ptr + assignment, mask=k_mask, other=0.0).to(tl.float32)
                acc += tl.sum(x * weight[:, None], axis=0)

            out_ptrs = out_ptr + t_idx * stride_outt + h_idx * stride_outh
            tl.store(out_ptrs, acc, mask=h_mask)

    @triton.jit
    def _sonic_token_gather_sum_two_buffers_bwd_kernel(
        dout_ptr,  # (T, H)
        x0_ptr,  # (M0, H)
        x1_ptr,  # (M1, H)
        w_ptr,  # (T * K,)
        m_perm_ptr,  # (T * K,) int32, indexing the logical concatenation of x0 and x1
        dx0_ptr,  # (M0, H)
        dx1_ptr,  # (M1, H)
        dw_ptr,  # (T * K,)
        m0: tl.constexpr,
        m1: tl.constexpr,
        h: tl.constexpr,
        max_k: tl.constexpr,
        stride_doutt: tl.constexpr,
        stride_douth: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        block_h: tl.constexpr,
    ):
        assignment = tl.program_id(axis=0)
        token = assignment // max_k
        position = tl.load(m_perm_ptr + assignment).to(tl.int64)
        in_first = position < m0
        in_second = (position >= m0) & (position < m0 + m1)
        first_position = tl.where(in_first, position, 0)
        second_position = tl.where(in_second, position - m0, 0)
        weight = tl.load(w_ptr + assignment).to(tl.float32)

        dw = 0.0
        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            dout = tl.load(
                dout_ptr + token * stride_doutt + h_idx * stride_douth,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                x0_ptr + first_position * stride_xm + h_idx * stride_xh,
                mask=in_first & h_mask,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                x1_ptr + second_position * stride_xm + h_idx * stride_xh,
                mask=in_second & h_mask,
                other=0.0,
            ).to(tl.float32)
            x = tl.where(in_first, x0, x1)

            tl.store(
                dx0_ptr + first_position * stride_xm + h_idx * stride_xh,
                dout * weight,
                mask=in_first & h_mask,
            )
            tl.store(
                dx1_ptr + second_position * stride_xm + h_idx * stride_xh,
                dout * weight,
                mask=in_second & h_mask,
            )
            dw += tl.sum(dout * x, axis=0)
        tl.store(dw_ptr + assignment, dw)

    @triton.jit
    def _sonic_dispatch_gather_kernel(
        x_ptr,  # (T, H)
        token_sources_ptr,  # (M,) int32; T denotes an empty slot
        out_ptr,  # (M, H)
        t: tl.constexpr,
        h: tl.constexpr,
        stride_xt: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_outm: tl.constexpr,
        stride_outh: tl.constexpr,
        block_h: tl.constexpr,
    ):
        output_row = tl.program_id(axis=0).to(tl.int64)
        token = tl.load(token_sources_ptr + output_row).to(tl.int64)
        token_is_valid = token < t

        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            values = tl.load(
                x_ptr + token * stride_xt + h_idx * stride_xh,
                mask=token_is_valid & h_mask,
                other=0.0,
            )
            tl.store(out_ptr + output_row * stride_outm + h_idx * stride_outh, values, mask=h_mask)

    @triton.jit
    def _sonic_unique_row_scatter_kernel(
        rows_ptr,  # (M, H)
        destinations_ptr,  # (M,) int32; output_rows denotes a dropped row
        out_ptr,  # (T, H), zero-initialized
        output_rows: tl.constexpr,
        rows: tl.constexpr,
        h: tl.constexpr,
        stride_rowsm: tl.constexpr,
        stride_rowsh: tl.constexpr,
        stride_outt: tl.constexpr,
        stride_outh: tl.constexpr,
        block_h: tl.constexpr,
    ):
        source_row = tl.program_id(axis=0).to(tl.int64)
        destination = tl.load(destinations_ptr + source_row).to(tl.int64)
        destination_is_valid = destination < output_rows

        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            values = tl.load(
                rows_ptr + source_row * stride_rowsm + h_idx * stride_rowsh,
                mask=h_mask,
                other=0.0,
            )
            tl.store(
                out_ptr + destination * stride_outt + h_idx * stride_outh,
                values,
                mask=destination_is_valid & h_mask,
            )

    @triton.jit
    def _sonic_slot_weighted_grad_kernel(
        dout_ptr,  # (M, H)
        x_ptr,  # (M, H)
        w_ptr,  # (M,)
        dx_ptr,  # (M, H)
        dw_ptr,  # (M,)
        m: tl.constexpr,
        h: tl.constexpr,
        stride_doutm: tl.constexpr,
        stride_douth: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        block_h: tl.constexpr,
    ):
        slot = tl.program_id(axis=0).to(tl.int64)
        weight = tl.load(w_ptr + slot).to(tl.float32)

        dw = 0.0
        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            dout = tl.load(
                dout_ptr + slot * stride_doutm + h_idx * stride_douth,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + slot * stride_xm + h_idx * stride_xh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)

            tl.store(dx_ptr + slot * stride_xm + h_idx * stride_xh, dout * weight, mask=h_mask)
            dw += tl.sum(dout * x, axis=0)
        tl.store(dw_ptr + slot, dw)

    @triton.jit
    def _sonic_clone_weight_reduce_kernel(
        clone_grads_ptr,  # (P, F)
        packed_experts_ptr,  # (P,) int32; local_experts denotes padding
        local_grads_ptr,  # (E, F)
        packed_count: tl.constexpr,
        local_experts: tl.constexpr,
        features: tl.constexpr,
        block_features: tl.constexpr,
        accumulate_in_bf16: tl.constexpr,
    ):
        local_expert = tl.program_id(axis=0)
        feature_offsets = tl.program_id(axis=1) * block_features + tl.arange(0, block_features)
        feature_mask = feature_offsets < features
        accumulator = tl.zeros([block_features], dtype=tl.float32)

        for packed_index in tl.range(0, packed_count):
            packed_expert = tl.load(packed_experts_ptr + packed_index)
            belongs_to_expert = packed_expert == local_expert
            values = tl.load(
                clone_grads_ptr + packed_index * features + feature_offsets,
                mask=belongs_to_expert & feature_mask,
                other=0.0,
            ).to(tl.float32)
            if accumulate_in_bf16:
                accumulator = (accumulator + values).to(tl.bfloat16).to(tl.float32)
            else:
                accumulator += values

        tl.store(
            local_grads_ptr + local_expert * features + feature_offsets,
            accumulator,
            mask=(local_expert < local_experts) & feature_mask,
        )

    @triton.jit
    def _sonic_unpermute_i32_kernel(
        values_ptr,  # (N,) int32
        permutation_ptr,  # (N,) int32, unique destinations
        out_ptr,  # (N,) int32
        n: tl.constexpr,
        block_n: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * block_n + tl.arange(0, block_n)
        mask = offsets < n
        destinations = tl.load(permutation_ptr + offsets, mask=mask)
        values = tl.load(values_ptr + offsets, mask=mask)
        tl.store(out_ptr + destinations, values, mask=mask)

    @triton.jit
    def _sonic_expert_local_rank_kernel(
        experts_ptr,  # (N,) int32
        ranks_ptr,  # (N,) int32
        n: tl.constexpr,
        block_n: tl.constexpr,
    ):
        expert = tl.program_id(axis=0)
        running_count = 0
        for block_start in tl.range(0, n, block_n):
            offsets = block_start + tl.arange(0, block_n)
            mask = offsets < n
            matches = (tl.load(experts_ptr + offsets, mask=mask, other=-1) == expert).to(tl.int32)
            ranks = running_count + tl.cumsum(matches, axis=0) - 1
            tl.store(ranks_ptr + offsets, ranks, mask=mask & (matches != 0))
            running_count += tl.sum(matches, axis=0)

    @triton.jit
    def _sonic_refill_rank_and_occupancy_kernel(
        experts_ptr,  # (N,) int32
        ranks_ptr,  # (N,) int32
        occupied_ptr,  # (E,) int32
        n: tl.constexpr,
        capacity: tl.constexpr,
        block_n: tl.constexpr,
    ):
        expert = tl.program_id(axis=0)
        running_count = 0
        for block_start in tl.range(0, n, block_n):
            offsets = block_start + tl.arange(0, block_n)
            mask = offsets < n
            matches = (tl.load(experts_ptr + offsets, mask=mask, other=-1) == expert).to(tl.int32)
            ranks = running_count + tl.cumsum(matches, axis=0) - 1
            tl.store(ranks_ptr + offsets, ranks, mask=mask & (matches != 0))
            running_count += tl.sum(matches, axis=0)
        tl.store(occupied_ptr + expert, tl.minimum(running_count, capacity))

    @triton.jit
    def _sonic_refill_overflow_rank_kernel(
        ranks_ptr,  # (N,) int32
        overflow_ranks_ptr,  # (N,) int32; -1 for kept assignments
        overflow_count_ptr,  # (1,) int32
        n: tl.constexpr,
        capacity: tl.constexpr,
        block_n: tl.constexpr,
    ):
        running_count = 0
        for block_start in tl.range(0, n, block_n):
            offsets = block_start + tl.arange(0, block_n)
            mask = offsets < n
            ranks = tl.load(ranks_ptr + offsets, mask=mask, other=-1)
            overflow = mask & (ranks >= capacity)
            block_ranks = running_count + tl.cumsum(overflow.to(tl.int32), axis=0) - 1
            tl.store(overflow_ranks_ptr + offsets, tl.where(overflow, block_ranks, -1), mask=mask)
            running_count += tl.sum(overflow.to(tl.int32), axis=0)
        tl.store(overflow_count_ptr, running_count)

    @triton.jit
    def _sonic_refill_vacancies_kernel(
        occupied_ptr,  # (E,) int32
        vacancy_experts_ptr,  # (capacity * E,) int32
        vacancy_slots_ptr,  # (capacity * E,) int32
        num_experts: tl.constexpr,
        block_e: tl.constexpr,
    ):
        slot = tl.program_id(axis=0)
        experts = tl.arange(0, block_e)
        expert_mask = experts < num_experts
        occupied = tl.load(occupied_ptr + experts, mask=expert_mask, other=slot + 1)
        vacant = expert_mask & (slot >= occupied)

        # Vacancy order is slot-major, then expert-major. Count vacancies in all
        # prior slot rows to obtain this row's compact output offset.
        row_offset = tl.sum(tl.maximum(slot - occupied, 0), axis=0)
        row_rank = tl.cumsum(vacant.to(tl.int32), axis=0) - 1
        destinations = row_offset + row_rank
        tl.store(vacancy_experts_ptr + destinations, experts, mask=vacant)
        tl.store(vacancy_slots_ptr + destinations, slot, mask=vacant)

else:
    _sonic_token_gather_sum_kernel = None
    _sonic_token_gather_sum_bwd_kernel = None
    _sonic_token_gather_sum_two_buffers_kernel = None
    _sonic_token_gather_sum_two_buffers_bwd_kernel = None
    _sonic_dispatch_gather_kernel = None
    _sonic_unique_row_scatter_kernel = None
    _sonic_slot_weighted_grad_kernel = None
    _sonic_clone_weight_reduce_kernel = None
    _sonic_unpermute_i32_kernel = None
    _sonic_expert_local_rank_kernel = None
    _sonic_refill_rank_and_occupancy_kernel = None
    _sonic_refill_overflow_rank_kernel = None
    _sonic_refill_vacancies_kernel = None


def _require_sonic_deps() -> None:
    if (
        jt is None
        or _sonic_token_gather_sum_kernel is None
        or _sonic_token_gather_sum_bwd_kernel is None
        or _sonic_token_gather_sum_two_buffers_kernel is None
        or _sonic_token_gather_sum_two_buffers_bwd_kernel is None
        or _sonic_dispatch_gather_kernel is None
        or _sonic_unique_row_scatter_kernel is None
        or _sonic_slot_weighted_grad_kernel is None
        or _sonic_clone_weight_reduce_kernel is None
        or _sonic_unpermute_i32_kernel is None
        or _sonic_expert_local_rank_kernel is None
        or _sonic_refill_rank_and_occupancy_kernel is None
        or _sonic_refill_overflow_rank_kernel is None
        or _sonic_refill_vacancies_kernel is None
    ):
        raise ImportError(
            "implementation='sonic' requires jax-triton and triton; install the gpu extra for marin-levanter "
            "or marin."
        )
    if not os.environ.get("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = _DEFAULT_TRITON_CACHE_DIR


def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def _sonic_kernel_config(hidden_dim: int) -> tuple[int, int, int]:
    block_h = min(max(256, _next_power_of_2(hidden_dim)), 4096)
    block_k = 1
    num_warps = 8 if block_h >= 1024 else 4
    return block_h, block_k, num_warps


def _sonic_fixed_k_offsets(*, tokens: int, topk: int) -> Int[Array, "Tp1"]:
    return jnp.arange(0, tokens * topk + 1, topk, dtype=jnp.int32)


def sonic_dispatch_gather(
    x: Float[Array, "T H"],
    token_sources: Int[Array, "M"],
) -> Float[Array, "M H"]:
    """Materialize fixed-capacity dispatch slots from their source token ids."""
    _require_sonic_deps()
    tokens, hidden_dim = x.shape
    block_h, _block_k, num_warps = _sonic_kernel_config(hidden_dim)
    out_shape = jax.ShapeDtypeStruct((token_sources.shape[0], hidden_dim), x.dtype)
    return jt.triton_call(
        x,
        token_sources,
        kernel=_sonic_dispatch_gather_kernel,
        out_shape=out_shape,
        grid=(token_sources.shape[0],),
        num_warps=num_warps,
        num_stages=4,
        t=tokens,
        h=hidden_dim,
        stride_xt=hidden_dim,
        stride_xh=1,
        stride_outm=hidden_dim,
        stride_outh=1,
        block_h=block_h,
    )


def sonic_unique_row_scatter(
    rows: Float[Array, "M H"],
    destinations: Int[Array, "M"],
    *,
    output_rows: int,
) -> Float[Array, "T H"]:
    """Scatter rows to unique shard-local destinations without atomics.

    Valid destination ids must be unique. ``output_rows`` is also the sentinel
    destination for dropped rows. The output is zero-filled before the kernel
    writes valid rows.
    """
    _require_sonic_deps()
    if destinations.shape != (rows.shape[0],):
        raise ValueError(f"destinations must have shape {(rows.shape[0],)}, got {destinations.shape}")
    if destinations.dtype != jnp.int32:
        raise ValueError(f"destinations must be int32, got {destinations.dtype}")
    if output_rows < 1:
        raise ValueError(f"output_rows must be positive, got {output_rows}")

    source_rows, hidden_dim = rows.shape
    block_h, _block_k, num_warps = _sonic_kernel_config(hidden_dim)
    output_shape = jax.ShapeDtypeStruct((output_rows, hidden_dim), rows.dtype)
    return jt.triton_call(
        rows,
        destinations,
        kernel=_sonic_unique_row_scatter_kernel,
        out_shape=output_shape,
        zeroed_outputs=(0,),
        grid=(source_rows,),
        num_warps=num_warps,
        num_stages=4,
        output_rows=output_rows,
        rows=source_rows,
        h=hidden_dim,
        stride_rowsm=hidden_dim,
        stride_rowsh=1,
        stride_outt=hidden_dim,
        stride_outh=1,
        block_h=block_h,
    )


def sonic_slot_weighted_grad(
    dout: Float[Array, "M H"],
    x: Float[Array, "M H"],
    weights: Float[Array, "M"],
) -> tuple[Float[Array, "M H"], Float[Array, "M"]]:
    """Compute slotwise ``dout * weight`` and ``dot(dout, x)`` on shard-local buffers.

    The caller invokes this inside the fixed-A2A ``shard_map``, so all three
    inputs are already local to one expert shard.
    """
    _require_sonic_deps()
    if dout.shape != x.shape:
        raise ValueError(f"dout and x must have the same shape, got {dout.shape} and {x.shape}")
    if weights.shape != (x.shape[0],):
        raise ValueError(f"weights must have shape {(x.shape[0],)}, got {weights.shape}")

    slots, hidden_dim = x.shape
    block_h, _block_k, num_warps = _sonic_kernel_config(hidden_dim)
    dx_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    dw_shape = jax.ShapeDtypeStruct(weights.shape, jnp.float32)
    return jt.triton_call(
        dout,
        x,
        weights,
        kernel=_sonic_slot_weighted_grad_kernel,
        out_shape=(dx_shape, dw_shape),
        grid=(slots,),
        num_warps=num_warps,
        num_stages=4,
        m=slots,
        h=hidden_dim,
        stride_doutm=hidden_dim,
        stride_douth=1,
        stride_xm=hidden_dim,
        stride_xh=1,
        block_h=block_h,
    )


def sonic_clone_weight_reduce(
    clone_grads: Float[Array, "P *W"],
    packed_experts: Int[Array, "P"],
    *,
    local_experts: int,
    block_features: int = 1024,
) -> Float[Array, "E *W"]:
    """Reduce received clone gradients into the shard-local owned experts.

    The caller invokes this inside the EP ``shard_map`` after the transpose of
    the clone-weight ragged all-to-all, so both inputs are fully shard-local.
    Each output element owns its reduction and reads only the packed rows for
    that expert; no atomics or cross-device communication occur here.
    """
    _require_sonic_deps()
    if clone_grads.shape[0] != packed_experts.shape[0]:
        raise ValueError(
            "clone_grads and packed_experts must have the same leading dimension, "
            f"got {clone_grads.shape[0]} and {packed_experts.shape[0]}"
        )
    if packed_experts.dtype != jnp.int32:
        raise ValueError(f"packed_experts must be int32, got {packed_experts.dtype}")
    if local_experts < 1:
        raise ValueError(f"local_experts must be positive, got {local_experts}")
    if block_features < 1 or block_features & (block_features - 1):
        raise ValueError(f"block_features must be a positive power of two, got {block_features}")

    packed_count = clone_grads.shape[0]
    features = math.prod(clone_grads.shape[1:])
    clone_grads_flat = clone_grads.reshape(packed_count, features)
    output_shape = jax.ShapeDtypeStruct((local_experts, features), clone_grads.dtype)
    local_grads = jt.triton_call(
        clone_grads_flat,
        packed_experts,
        kernel=_sonic_clone_weight_reduce_kernel,
        out_shape=output_shape,
        grid=(local_experts, triton.cdiv(features, block_features)),
        num_warps=8,
        num_stages=2,
        packed_count=packed_count,
        local_experts=local_experts,
        features=features,
        block_features=block_features,
        accumulate_in_bf16=clone_grads.dtype == jnp.bfloat16,
    )
    return local_grads.reshape((local_experts, *clone_grads.shape[1:]))


def sonic_unpermute_i32(
    values: Int[Array, "N"],
    permutation: Int[Array, "N"],
) -> Int[Array, "N"]:
    """Scatter int32 values through a shard-local permutation."""
    _require_sonic_deps()
    if values.shape != permutation.shape:
        raise ValueError(
            f"values and permutation must have the same shape, got {values.shape} and {permutation.shape}"
        )
    if values.dtype != jnp.int32 or permutation.dtype != jnp.int32:
        raise ValueError(f"values and permutation must be int32, got {values.dtype} and {permutation.dtype}")

    block_n = 256
    out_shape = jax.ShapeDtypeStruct(values.shape, values.dtype)
    return jt.triton_call(
        values,
        permutation,
        kernel=_sonic_unpermute_i32_kernel,
        out_shape=out_shape,
        grid=(triton.cdiv(values.size, block_n),),
        num_warps=4,
        num_stages=1,
        n=values.size,
        block_n=block_n,
    )


def sonic_expert_local_rank(
    experts: Int[Array, "N"],
    *,
    num_experts: int,
) -> Int[Array, "N"]:
    """Return each assignment's zero-based rank among earlier assignments to the same expert."""
    _require_sonic_deps()
    if experts.dtype != jnp.int32:
        raise ValueError(f"experts must be int32, got {experts.dtype}")
    if experts.ndim != 1:
        raise ValueError(f"experts must be one-dimensional, got shape {experts.shape}")
    if num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    block_n = 4096
    out_shape = jax.ShapeDtypeStruct(experts.shape, experts.dtype)
    return jt.triton_call(
        experts,
        kernel=_sonic_expert_local_rank_kernel,
        out_shape=out_shape,
        grid=(num_experts,),
        num_warps=8,
        num_stages=1,
        n=experts.size,
        block_n=block_n,
    )


def sonic_capacity_refill(
    experts: Int[Array, "N"],
    *,
    num_experts: int,
    capacity: int,
) -> tuple[Int[Array, "N"], Int[Array, "N"], Int[Array, ""]]:
    """Refill overflowing expert assignments in stable vacancy order."""
    _require_sonic_deps()
    if experts.dtype != jnp.int32:
        raise ValueError(f"experts must be int32, got {experts.dtype}")
    if experts.ndim != 1:
        raise ValueError(f"experts must be one-dimensional, got shape {experts.shape}")
    if num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if capacity < 1:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if capacity * num_experts < experts.size:
        raise ValueError(f"capacity={capacity} across {num_experts} experts cannot hold {experts.size} assignments")

    block_n = 4096
    ranks_shape = jax.ShapeDtypeStruct(experts.shape, experts.dtype)
    occupied_shape = jax.ShapeDtypeStruct((num_experts,), experts.dtype)
    local_ranks, occupied = jt.triton_call(
        experts,
        kernel=_sonic_refill_rank_and_occupancy_kernel,
        out_shape=(ranks_shape, occupied_shape),
        grid=(num_experts,),
        num_warps=8,
        num_stages=1,
        n=experts.size,
        capacity=capacity,
        block_n=block_n,
    )

    overflow_count_shape = jax.ShapeDtypeStruct((1,), experts.dtype)
    overflow_ranks, overflow_count = jt.triton_call(
        local_ranks,
        kernel=_sonic_refill_overflow_rank_kernel,
        out_shape=(ranks_shape, overflow_count_shape),
        grid=(1,),
        num_warps=8,
        num_stages=1,
        n=experts.size,
        capacity=capacity,
        block_n=block_n,
    )

    total_slots = capacity * num_experts
    vacancies_shape = jax.ShapeDtypeStruct((total_slots,), experts.dtype)
    vacancy_experts, vacancy_slots = jt.triton_call(
        occupied,
        kernel=_sonic_refill_vacancies_kernel,
        out_shape=(vacancies_shape, vacancies_shape),
        grid=(capacity,),
        num_warps=8,
        num_stages=1,
        num_experts=num_experts,
        block_e=_next_power_of_2(num_experts),
    )

    keep = local_ranks < capacity
    replacement_indices = jnp.maximum(overflow_ranks, 0)
    refilled_experts = jnp.where(keep, experts, vacancy_experts[replacement_indices])
    refilled_slots = jnp.where(keep, local_ranks, vacancy_slots[replacement_indices])
    return refilled_experts, refilled_slots, overflow_count[0]


def _sonic_gather_sum_impl(
    dispatch_output: Float[Array, "M H"],
    weights_flat: Float[Array, "M"],
    positions_flat: Int[Array, "M"],
    offsets: Int[Array, "Tp1"],
    *,
    tokens: int,
    topk: int,
    accumulate_in_bf16: bool = False,
) -> Float[Array, "T H"]:
    _require_sonic_deps()
    dispatch_rows, hidden_dim = dispatch_output.shape
    block_h, block_k, num_warps = _sonic_kernel_config(hidden_dim)
    out_shape = jax.ShapeDtypeStruct((tokens, hidden_dim), dispatch_output.dtype)
    return jt.triton_call(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        kernel=_sonic_token_gather_sum_kernel,
        out_shape=out_shape,
        grid=(tokens,),
        num_warps=num_warps,
        num_stages=4,
        m=dispatch_rows,
        t=tokens,
        h=hidden_dim,
        max_k=topk,
        stride_xm=hidden_dim,
        stride_xh=1,
        stride_outt=hidden_dim,
        stride_outh=1,
        block_h=block_h,
        block_k=block_k,
        w_is_none=False,
        is_varlen_k=False,
        accumulate_in_bf16=accumulate_in_bf16,
    )


def _sonic_gather_sum_bwd_impl(
    dout: Float[Array, "T H"],
    dispatch_output: Float[Array, "M H"],
    weights_flat: Float[Array, "M"],
    positions_flat: Int[Array, "M"],
    offsets: Int[Array, "Tp1"],
    *,
    tokens: int,
    topk: int,
) -> tuple[Float[Array, "M H"], Float[Array, "M"]]:
    _require_sonic_deps()
    dispatch_rows, hidden_dim = dispatch_output.shape
    block_h, _block_k, num_warps = _sonic_kernel_config(hidden_dim)
    dx_shape = jax.ShapeDtypeStruct(dispatch_output.shape, dispatch_output.dtype)
    dw_shape = jax.ShapeDtypeStruct(weights_flat.shape, jnp.float32)
    return jt.triton_call(
        dout,
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        kernel=_sonic_token_gather_sum_bwd_kernel,
        out_shape=(dx_shape, dw_shape),
        zeroed_outputs=(0,),
        grid=(tokens * topk,),
        num_warps=num_warps,
        num_stages=4,
        m=dispatch_rows,
        t=tokens,
        h=hidden_dim,
        max_k=topk,
        stride_doutt=hidden_dim,
        stride_douth=1,
        stride_xm=hidden_dim,
        stride_xh=1,
        block_h=block_h,
    )


@jax.custom_vjp
def sonic_gather_sum(
    dispatch_output: Float[Array, "M H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> Float[Array, "T H"]:
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    return _sonic_gather_sum_impl(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
        accumulate_in_bf16=False,
    )


def sonic_gather_sum_bf16_accum(
    dispatch_output: Float[Array, "M H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> Float[Array, "T H"]:
    """Gather and sum with BF16 rounding after each fixed-K assignment."""
    if dispatch_output.dtype != jnp.bfloat16:
        raise ValueError(f"BF16 accumulation requires a bfloat16 input, got {dispatch_output.dtype}")
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    return _sonic_gather_sum_impl(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
        accumulate_in_bf16=True,
    )


def _sonic_gather_sum_fwd(
    dispatch_output: Float[Array, "M H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> tuple[Float[Array, "T H"], tuple[Float[Array, "M H"], Int[Array, "T K"], Float[Array, "T K"]]]:
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    out = _sonic_gather_sum_impl(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
        accumulate_in_bf16=False,
    )
    return out, (dispatch_output, dispatch_positions, combine_weights)


def _sonic_gather_sum_bwd(
    residuals: tuple[Float[Array, "M H"], Int[Array, "T K"], Float[Array, "T K"]],
    dout: Float[Array, "T H"],
) -> tuple[Float[Array, "M H"], None, Float[Array, "T K"]]:
    dispatch_output, dispatch_positions, combine_weights = residuals
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    d_dispatch_output, d_weights_flat = _sonic_gather_sum_bwd_impl(
        dout,
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
    )
    d_combine_weights = d_weights_flat.reshape(combine_weights.shape).astype(combine_weights.dtype)
    return d_dispatch_output, None, d_combine_weights


sonic_gather_sum.defvjp(_sonic_gather_sum_fwd, _sonic_gather_sum_bwd)


def _sonic_gather_sum_two_buffers_impl(
    first_dispatch_output: Float[Array, "M0 H"],
    second_dispatch_output: Float[Array, "M1 H"],
    weights_flat: Float[Array, "M"],
    positions_flat: Int[Array, "M"],
    *,
    tokens: int,
    topk: int,
) -> Float[Array, "T H"]:
    _require_sonic_deps()
    first_rows, hidden_dim = first_dispatch_output.shape
    second_rows = second_dispatch_output.shape[0]
    block_h, block_k, num_warps = _sonic_kernel_config(hidden_dim)
    out_shape = jax.ShapeDtypeStruct((tokens, hidden_dim), first_dispatch_output.dtype)
    return jt.triton_call(
        first_dispatch_output,
        second_dispatch_output,
        weights_flat,
        positions_flat,
        kernel=_sonic_token_gather_sum_two_buffers_kernel,
        out_shape=out_shape,
        grid=(tokens,),
        num_warps=num_warps,
        num_stages=4,
        m0=first_rows,
        m1=second_rows,
        h=hidden_dim,
        max_k=topk,
        stride_xm=hidden_dim,
        stride_xh=1,
        stride_outt=hidden_dim,
        stride_outh=1,
        block_h=block_h,
        block_k=block_k,
    )


def _sonic_gather_sum_two_buffers_bwd_impl(
    dout: Float[Array, "T H"],
    first_dispatch_output: Float[Array, "M0 H"],
    second_dispatch_output: Float[Array, "M1 H"],
    weights_flat: Float[Array, "M"],
    positions_flat: Int[Array, "M"],
    *,
    tokens: int,
    topk: int,
) -> tuple[Float[Array, "M0 H"], Float[Array, "M1 H"], Float[Array, "M"]]:
    _require_sonic_deps()
    first_rows, hidden_dim = first_dispatch_output.shape
    second_rows = second_dispatch_output.shape[0]
    block_h, _block_k, num_warps = _sonic_kernel_config(hidden_dim)
    first_grad_shape = jax.ShapeDtypeStruct(first_dispatch_output.shape, first_dispatch_output.dtype)
    second_grad_shape = jax.ShapeDtypeStruct(second_dispatch_output.shape, second_dispatch_output.dtype)
    weights_grad_shape = jax.ShapeDtypeStruct(weights_flat.shape, jnp.float32)
    return jt.triton_call(
        dout,
        first_dispatch_output,
        second_dispatch_output,
        weights_flat,
        positions_flat,
        kernel=_sonic_token_gather_sum_two_buffers_bwd_kernel,
        out_shape=(first_grad_shape, second_grad_shape, weights_grad_shape),
        zeroed_outputs=(0, 1),
        grid=(tokens * topk,),
        num_warps=num_warps,
        num_stages=4,
        m0=first_rows,
        m1=second_rows,
        h=hidden_dim,
        max_k=topk,
        stride_doutt=hidden_dim,
        stride_douth=1,
        stride_xm=hidden_dim,
        stride_xh=1,
        block_h=block_h,
    )


def _validate_two_buffer_gather_sum_inputs(
    first_dispatch_output: Float[Array, "M0 H"],
    second_dispatch_output: Float[Array, "M1 H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> tuple[int, int]:
    if first_dispatch_output.ndim != 2 or second_dispatch_output.ndim != 2:
        raise ValueError("dispatch outputs must be two-dimensional")
    if first_dispatch_output.shape[1] != second_dispatch_output.shape[1]:
        raise ValueError(
            "dispatch outputs must have the same hidden dimension, got "
            f"{first_dispatch_output.shape[1]} and {second_dispatch_output.shape[1]}"
        )
    if first_dispatch_output.dtype != second_dispatch_output.dtype:
        raise ValueError(
            "dispatch outputs must have the same dtype, got "
            f"{first_dispatch_output.dtype} and {second_dispatch_output.dtype}"
        )
    if dispatch_positions.shape != combine_weights.shape:
        raise ValueError(
            f"dispatch_positions shape {dispatch_positions.shape} must match combine_weights shape {combine_weights.shape}"
        )
    if dispatch_positions.ndim != 2:
        raise ValueError(f"dispatch_positions must be two-dimensional, got shape {dispatch_positions.shape}")
    return combine_weights.shape[0], combine_weights.shape[1]


@jax.custom_vjp
def sonic_gather_sum_two_buffers(
    first_dispatch_output: Float[Array, "M0 H"],
    second_dispatch_output: Float[Array, "M1 H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> Float[Array, "T H"]:
    """Gather from two logical-concatenated buffers without materializing the concatenation."""
    tokens, topk = _validate_two_buffer_gather_sum_inputs(
        first_dispatch_output,
        second_dispatch_output,
        dispatch_positions,
        combine_weights,
    )
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    return _sonic_gather_sum_two_buffers_impl(
        first_dispatch_output,
        second_dispatch_output,
        weights_flat,
        positions_flat,
        tokens=tokens,
        topk=topk,
    )


def _sonic_gather_sum_two_buffers_fwd(
    first_dispatch_output: Float[Array, "M0 H"],
    second_dispatch_output: Float[Array, "M1 H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> tuple[
    Float[Array, "T H"],
    tuple[Float[Array, "M0 H"], Float[Array, "M1 H"], Int[Array, "T K"], Float[Array, "T K"]],
]:
    tokens, topk = _validate_two_buffer_gather_sum_inputs(
        first_dispatch_output,
        second_dispatch_output,
        dispatch_positions,
        combine_weights,
    )
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    out = _sonic_gather_sum_two_buffers_impl(
        first_dispatch_output,
        second_dispatch_output,
        weights_flat,
        positions_flat,
        tokens=tokens,
        topk=topk,
    )
    return out, (first_dispatch_output, second_dispatch_output, dispatch_positions, combine_weights)


def _sonic_gather_sum_two_buffers_bwd(
    residuals: tuple[
        Float[Array, "M0 H"],
        Float[Array, "M1 H"],
        Int[Array, "T K"],
        Float[Array, "T K"],
    ],
    dout: Float[Array, "T H"],
) -> tuple[Float[Array, "M0 H"], Float[Array, "M1 H"], None, Float[Array, "T K"]]:
    first_dispatch_output, second_dispatch_output, dispatch_positions, combine_weights = residuals
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    first_grad, second_grad, weights_grad_flat = _sonic_gather_sum_two_buffers_bwd_impl(
        dout,
        first_dispatch_output,
        second_dispatch_output,
        weights_flat,
        positions_flat,
        tokens=tokens,
        topk=topk,
    )
    weights_grad = weights_grad_flat.reshape(combine_weights.shape).astype(combine_weights.dtype)
    return first_grad, second_grad, None, weights_grad


sonic_gather_sum_two_buffers.defvjp(
    _sonic_gather_sum_two_buffers_fwd,
    _sonic_gather_sum_two_buffers_bwd,
)


def _moe_mlp_local_sonic(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    """Local raw-Sonic path: JAX grouped GEMMs plus Sonic Triton gather/combine."""
    token_ids_sort, dispatch_positions, group_sizes, _sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            num_experts=num_experts,
        )
    )
    x_dispatch = tree_checkpoint_name(x[token_ids_sort], _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        hidden = activation_fn(gate) * up
        out_dispatch = ragged_dot(hidden, moe_w2, group_sizes)
        out = tree_checkpoint_name(
            sonic_gather_sum(out_dispatch, dispatch_positions, combine_weights),
            _CHECKPOINT_MOE_OUTPUT,
        )

    return out, _zero_dropped_assignments()
