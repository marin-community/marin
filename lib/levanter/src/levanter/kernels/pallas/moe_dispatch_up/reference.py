# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure JAX oracle for the MoE dispatch-up subkernel."""

import math
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from haliax.nn.ragged_dot import ragged_dot

RaggedDotImplementation = Literal["auto", "megablox", "triton", "xla"]


class MoeDispatchUpLayout(NamedTuple):
    recv_x: jax.Array
    recv_valid: jax.Array
    rows_per_expert: jax.Array
    expert_base: jax.Array
    recv_local_expert: jax.Array
    recv_src_rank: jax.Array
    recv_src_token_idx: jax.Array
    recv_topk_slot: jax.Array
    recv_router_weight: jax.Array
    overflow_count: jax.Array


class MoeDispatchUpPrepackedSend(NamedTuple):
    send_x_by_dst: jax.Array
    send_row_by_dst: jax.Array
    send_local_expert_by_dst: jax.Array
    send_src_token_idx_by_dst: jax.Array
    send_topk_slot_by_dst: jax.Array
    send_router_weight_by_dst: jax.Array
    send_count_by_dst: jax.Array
    rows_per_expert: jax.Array
    expert_base: jax.Array
    send_expert_base_by_dst: jax.Array
    send_expert_count_by_dst: jax.Array
    recv_source_expert_base: jax.Array
    recv_source_expert_count: jax.Array
    overflow_count: jax.Array


def _validate_routing_inputs(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    *,
    num_experts: int,
) -> tuple[int, int, int, int]:
    if x_by_rank.ndim != 3:
        raise ValueError(f"x_by_rank must have shape [EP, T, D], got {x_by_rank.shape}")
    if expert_ids_by_rank.ndim != 3:
        raise ValueError(f"expert_ids_by_rank must have shape [EP, T, K], got {expert_ids_by_rank.shape}")
    if router_weights_by_rank.shape != expert_ids_by_rank.shape:
        raise ValueError(
            "router_weights_by_rank must match expert_ids_by_rank shape; "
            f"got {router_weights_by_rank.shape} vs {expert_ids_by_rank.shape}"
        )
    ep_size, tokens_per_rank, hidden = x_by_rank.shape
    if expert_ids_by_rank.shape[0] != ep_size or expert_ids_by_rank.shape[1] != tokens_per_rank:
        raise ValueError(
            "x_by_rank and expert_ids_by_rank must share EP and token dimensions; "
            f"got {x_by_rank.shape} vs {expert_ids_by_rank.shape}"
        )
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
    top_k = expert_ids_by_rank.shape[2]
    return ep_size, tokens_per_rank, hidden, top_k


def _recv_capacity(
    *,
    ep_size: int,
    tokens_per_rank: int,
    top_k: int,
    capacity_factor: float,
    recv_capacity: int | None,
) -> int:
    if recv_capacity is not None:
        return recv_capacity
    assignments_per_rank = tokens_per_rank * top_k
    return max(1, int(math.ceil(capacity_factor * assignments_per_rank)))


def _send_capacity(*, tokens_per_rank: int, top_k: int, send_capacity: int | None) -> int:
    if send_capacity is not None:
        return send_capacity
    return tokens_per_rank * top_k


def _counts_by_destination_source_expert(
    expert_ids_by_rank: jax.Array,
    *,
    num_experts: int,
) -> jax.Array:
    ep_size = expert_ids_by_rank.shape[0]
    local_experts = num_experts // ep_size
    counts = jnp.zeros((ep_size, local_experts, ep_size), dtype=jnp.int32)
    flat_experts_by_rank = expert_ids_by_rank.reshape(ep_size, -1)
    for dst_rank in range(ep_size):
        expert_start = dst_rank * local_experts
        for local_expert in range(local_experts):
            global_expert = expert_start + local_expert
            for src_rank in range(ep_size):
                count = jnp.sum(flat_experts_by_rank[src_rank] == global_expert, dtype=jnp.int32)
                counts = counts.at[dst_rank, local_expert, src_rank].set(count)
    return counts


def prepack_moe_dispatch_up_reference(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float = 1.25,
    recv_capacity: int | None = None,
    send_capacity: int | None = None,
) -> MoeDispatchUpPrepackedSend:
    """Prepack source-owned sends and final destination row indices.

    Rows are ordered by destination local expert, then source rank, then source
    local assignment position. The final row index is included so the Mosaic
    validation primitive can focus only on remote writes.
    """

    ep_size, tokens_per_rank, hidden, top_k = _validate_routing_inputs(
        x_by_rank,
        expert_ids_by_rank,
        router_weights_by_rank,
        num_experts=num_experts,
    )
    local_experts = num_experts // ep_size
    recv_capacity = _recv_capacity(
        ep_size=ep_size,
        tokens_per_rank=tokens_per_rank,
        top_k=top_k,
        capacity_factor=capacity_factor,
        recv_capacity=recv_capacity,
    )
    send_capacity = _send_capacity(tokens_per_rank=tokens_per_rank, top_k=top_k, send_capacity=send_capacity)

    counts = _counts_by_destination_source_expert(expert_ids_by_rank, num_experts=num_experts)
    rows_per_expert = jnp.sum(counts, axis=2, dtype=jnp.int32)
    expert_base = jnp.cumsum(rows_per_expert, axis=1, dtype=jnp.int32) - rows_per_expert
    send_expert_count_by_dst = jnp.transpose(counts, (2, 0, 1))
    send_expert_base_by_dst = jnp.cumsum(send_expert_count_by_dst, axis=2, dtype=jnp.int32) - send_expert_count_by_dst
    recv_source_expert_count = jnp.transpose(counts, (0, 2, 1))
    recv_source_expert_base = jnp.transpose(
        expert_base[:, :, None] + jnp.cumsum(counts, axis=2, dtype=jnp.int32) - counts,
        (0, 2, 1),
    )
    rows_per_rank = jnp.sum(rows_per_expert, axis=1, dtype=jnp.int32)
    overflow_count = jnp.sum(jnp.maximum(rows_per_rank - recv_capacity, 0), dtype=jnp.int32)

    send_x = jnp.zeros((ep_size, ep_size, send_capacity, hidden), dtype=x_by_rank.dtype)
    send_row = jnp.zeros((ep_size, ep_size, send_capacity), dtype=jnp.int32)
    send_local_expert = jnp.zeros((ep_size, ep_size, send_capacity), dtype=jnp.int32)
    send_src_token_idx = jnp.zeros((ep_size, ep_size, send_capacity), dtype=jnp.int32)
    send_topk_slot = jnp.zeros((ep_size, ep_size, send_capacity), dtype=jnp.int32)
    send_router_weight = jnp.zeros((ep_size, ep_size, send_capacity), dtype=router_weights_by_rank.dtype)
    send_count = jnp.zeros((ep_size, ep_size), dtype=jnp.int32)

    token_ids = jnp.arange(tokens_per_rank, dtype=jnp.int32)[:, None]
    topk_slots = jnp.arange(top_k, dtype=jnp.int32)[None, :]
    flat_token_ids = jnp.broadcast_to(token_ids, (tokens_per_rank, top_k)).reshape(-1)
    flat_topk_slots = jnp.broadcast_to(topk_slots, (tokens_per_rank, top_k)).reshape(-1)

    for src_rank in range(ep_size):
        flat_experts = expert_ids_by_rank[src_rank].reshape(-1)
        flat_weights = router_weights_by_rank[src_rank].reshape(-1)
        flat_x = x_by_rank[src_rank, flat_token_ids]
        for dst_rank in range(ep_size):
            send_base_for_expert = (
                jnp.cumsum(counts[dst_rank, :, src_rank], dtype=jnp.int32) - counts[dst_rank, :, src_rank]
            )
            for local_expert in range(local_experts):
                global_expert = dst_rank * local_experts + local_expert
                mask = flat_experts == global_expert
                local_rank = jnp.cumsum(mask.astype(jnp.int32), dtype=jnp.int32) - 1
                send_slot = send_base_for_expert[local_expert] + local_rank
                dst_src_base = jnp.sum(counts[dst_rank, local_expert, :src_rank], dtype=jnp.int32)
                dst_row = expert_base[dst_rank, local_expert] + dst_src_base + local_rank
                safe_slot = jnp.where(mask, send_slot, 0)

                send_x = send_x.at[src_rank, dst_rank, safe_slot].add(flat_x * mask[:, None])
                send_row = send_row.at[src_rank, dst_rank, safe_slot].add(jnp.where(mask, dst_row, 0))
                send_local_expert = send_local_expert.at[src_rank, dst_rank, safe_slot].add(
                    jnp.where(mask, local_expert, 0)
                )
                send_src_token_idx = send_src_token_idx.at[src_rank, dst_rank, safe_slot].add(
                    jnp.where(mask, flat_token_ids, 0)
                )
                send_topk_slot = send_topk_slot.at[src_rank, dst_rank, safe_slot].add(
                    jnp.where(mask, flat_topk_slots, 0)
                )
                send_router_weight = send_router_weight.at[src_rank, dst_rank, safe_slot].add(
                    flat_weights * mask.astype(flat_weights.dtype)
                )
            send_count = send_count.at[src_rank, dst_rank].set(jnp.sum(counts[dst_rank, :, src_rank], dtype=jnp.int32))

    return MoeDispatchUpPrepackedSend(
        send_x,
        send_row,
        send_local_expert,
        send_src_token_idx,
        send_topk_slot,
        send_router_weight,
        send_count,
        rows_per_expert,
        expert_base,
        send_expert_base_by_dst,
        send_expert_count_by_dst,
        recv_source_expert_base,
        recv_source_expert_count,
        overflow_count,
    )


def dispatch_prepacked_moe_dispatch_up_reference(
    prepacked: MoeDispatchUpPrepackedSend,
    *,
    recv_capacity: int | None = None,
) -> MoeDispatchUpLayout:
    """Destination-owned expert-major dispatch from prepacked source sends."""

    send_x = prepacked.send_x_by_dst
    ep_size, _, send_capacity, hidden = send_x.shape
    if recv_capacity is None:
        recv_capacity = send_capacity

    recv_x = jnp.zeros((ep_size, recv_capacity, hidden), dtype=send_x.dtype)
    recv_valid_count = jnp.zeros((ep_size, recv_capacity), dtype=jnp.int32)
    recv_local_expert = jnp.zeros((ep_size, recv_capacity), dtype=jnp.int32)
    recv_src_rank = jnp.zeros((ep_size, recv_capacity), dtype=jnp.int32)
    recv_src_token_idx = jnp.zeros((ep_size, recv_capacity), dtype=jnp.int32)
    recv_topk_slot = jnp.zeros((ep_size, recv_capacity), dtype=jnp.int32)
    recv_router_weight = jnp.zeros((ep_size, recv_capacity), dtype=prepacked.send_router_weight_by_dst.dtype)

    send_positions = jnp.arange(send_capacity, dtype=jnp.int32)
    for src_rank in range(ep_size):
        for dst_rank in range(ep_size):
            valid_send = send_positions < prepacked.send_count_by_dst[src_rank, dst_rank]
            row = prepacked.send_row_by_dst[src_rank, dst_rank]
            valid_recv = valid_send & (row < recv_capacity)
            safe_row = jnp.where(valid_recv, row, 0)
            recv_x = recv_x.at[dst_rank, safe_row].add(
                prepacked.send_x_by_dst[src_rank, dst_rank] * valid_recv[:, None]
            )
            recv_valid_count = recv_valid_count.at[dst_rank, safe_row].add(valid_recv.astype(jnp.int32))
            recv_local_expert = recv_local_expert.at[dst_rank, safe_row].add(
                prepacked.send_local_expert_by_dst[src_rank, dst_rank] * valid_recv.astype(jnp.int32)
            )
            recv_src_rank = recv_src_rank.at[dst_rank, safe_row].add(src_rank * valid_recv.astype(jnp.int32))
            recv_src_token_idx = recv_src_token_idx.at[dst_rank, safe_row].add(
                prepacked.send_src_token_idx_by_dst[src_rank, dst_rank] * valid_recv.astype(jnp.int32)
            )
            recv_topk_slot = recv_topk_slot.at[dst_rank, safe_row].add(
                prepacked.send_topk_slot_by_dst[src_rank, dst_rank] * valid_recv.astype(jnp.int32)
            )
            recv_router_weight = recv_router_weight.at[dst_rank, safe_row].add(
                prepacked.send_router_weight_by_dst[src_rank, dst_rank]
                * valid_recv.astype(prepacked.send_router_weight_by_dst.dtype)
            )

    return MoeDispatchUpLayout(
        recv_x,
        recv_valid_count > 0,
        prepacked.rows_per_expert,
        prepacked.expert_base,
        recv_local_expert,
        recv_src_rank,
        recv_src_token_idx,
        recv_topk_slot,
        recv_router_weight,
        prepacked.overflow_count,
    )


def moe_dispatch_up_layout_reference(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float = 1.25,
    recv_capacity: int | None = None,
) -> MoeDispatchUpLayout:
    prepacked = prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids_by_rank,
        router_weights_by_rank,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        recv_capacity=recv_capacity,
    )
    return dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=prepacked.send_x_by_dst.shape[2])


def compute_moe_up_from_layout_reference(
    layout: MoeDispatchUpLayout,
    w_gate_up_by_rank: jax.Array,
) -> jax.Array:
    """Compute expert-local W13 followed by `silu(gate) * up` for routed rows."""

    if w_gate_up_by_rank.ndim != 4:
        raise ValueError(f"w_gate_up_by_rank must have shape [EP, EL, D, 2I], got {w_gate_up_by_rank.shape}")
    ep_size, recv_capacity, hidden = layout.recv_x.shape
    if w_gate_up_by_rank.shape[0] != ep_size or w_gate_up_by_rank.shape[2] != hidden:
        raise ValueError(
            "w_gate_up_by_rank must share EP and hidden dimensions with layout.recv_x; "
            f"got {w_gate_up_by_rank.shape} vs {layout.recv_x.shape}"
        )
    intermediate = w_gate_up_by_rank.shape[3] // 2
    h = jnp.zeros((ep_size, recv_capacity, intermediate), dtype=layout.recv_x.dtype)
    for dst_rank in range(ep_size):
        for local_expert in range(w_gate_up_by_rank.shape[1]):
            mask = (layout.recv_local_expert[dst_rank] == local_expert) & layout.recv_valid[dst_rank]
            gate_up = layout.recv_x[dst_rank] @ w_gate_up_by_rank[dst_rank, local_expert]
            gate, up = jnp.split(gate_up, [intermediate], axis=-1)
            expert_h = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up
            h = h.at[dst_rank].add(expert_h * mask[:, None])
    return h


def _effective_rows_per_expert(layout: MoeDispatchUpLayout) -> jax.Array:
    recv_capacity = layout.recv_x.shape[1]
    remaining_capacity = jnp.maximum(recv_capacity - layout.expert_base, 0)
    return jnp.minimum(layout.rows_per_expert, remaining_capacity)


def compute_moe_up_from_layout_ragged_dot(
    layout: MoeDispatchUpLayout,
    w_gate_up_by_rank: jax.Array,
    *,
    implementation: RaggedDotImplementation = "auto",
) -> jax.Array:
    """Compute W13/SiLU from a dispatched layout using Haliax grouped matmul."""

    if w_gate_up_by_rank.ndim != 4:
        raise ValueError(f"w_gate_up_by_rank must have shape [EP, EL, D, 2I], got {w_gate_up_by_rank.shape}")
    ep_size, recv_capacity, hidden = layout.recv_x.shape
    if w_gate_up_by_rank.shape[0] != ep_size or w_gate_up_by_rank.shape[2] != hidden:
        raise ValueError(
            "w_gate_up_by_rank must share EP and hidden dimensions with layout.recv_x; "
            f"got {w_gate_up_by_rank.shape} vs {layout.recv_x.shape}"
        )
    if w_gate_up_by_rank.shape[3] % 2 != 0:
        raise ValueError(f"w_gate_up_by_rank last dimension must be even, got {w_gate_up_by_rank.shape[3]}")

    effective_rows = _effective_rows_per_expert(layout)

    def rank_w13(recv_x: jax.Array, group_sizes: jax.Array, w_gate_up: jax.Array) -> jax.Array:
        w13_out = ragged_dot(recv_x, w_gate_up, group_sizes, implementation=implementation)
        gate, up = jnp.split(w13_out, 2, axis=-1)
        h = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up
        valid_rows = jnp.arange(recv_capacity, dtype=jnp.int32) < jnp.sum(group_sizes, dtype=jnp.int32)
        return jnp.where(valid_rows[:, None], h, jnp.zeros((), dtype=h.dtype))

    return jax.vmap(rank_w13)(layout.recv_x, effective_rows, w_gate_up_by_rank)


def compute_moe_up_from_layout_reference_bwd(
    layout: MoeDispatchUpLayout,
    w_gate_up_by_rank: jax.Array,
    grad_dispatch_up: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Backward oracle for the W13/SiLU part of `moe_dispatch_up`.

    Returns gradients for `layout.recv_x` and `w_gate_up_by_rank`.
    Metadata in `layout` is treated as non-differentiable routing state.
    """

    if grad_dispatch_up.shape[:2] != layout.recv_x.shape[:2]:
        raise ValueError(
            "grad_dispatch_up must share EP and recv row dimensions with layout.recv_x; "
            f"got {grad_dispatch_up.shape} vs {layout.recv_x.shape}"
        )
    if w_gate_up_by_rank.ndim != 4:
        raise ValueError(f"w_gate_up_by_rank must have shape [EP, EL, D, 2I], got {w_gate_up_by_rank.shape}")

    ep_size, recv_capacity, hidden = layout.recv_x.shape
    local_experts = w_gate_up_by_rank.shape[1]
    intermediate = w_gate_up_by_rank.shape[3] // 2
    if grad_dispatch_up.shape != (ep_size, recv_capacity, intermediate):
        raise ValueError(
            "grad_dispatch_up must have shape [EP, R, I] matching W13 output; "
            f"got {grad_dispatch_up.shape}, expected {(ep_size, recv_capacity, intermediate)}"
        )

    grad_recv_x = jnp.zeros_like(layout.recv_x)
    grad_w_gate_up = jnp.zeros_like(w_gate_up_by_rank)
    for dst_rank in range(ep_size):
        for local_expert in range(local_experts):
            mask = (layout.recv_local_expert[dst_rank] == local_expert) & layout.recv_valid[dst_rank]
            x = layout.recv_x[dst_rank]
            weights = w_gate_up_by_rank[dst_rank, local_expert]
            gate_up = x @ weights
            gate, up = jnp.split(gate_up, [intermediate], axis=-1)
            grad_h = grad_dispatch_up[dst_rank] * mask[:, None]

            gate_f32 = gate.astype(jnp.float32)
            sigmoid_gate = jax.nn.sigmoid(gate_f32)
            silu_gate = (gate_f32 * sigmoid_gate).astype(gate.dtype)
            silu_grad = (sigmoid_gate * (1 + gate_f32 * (1 - sigmoid_gate))).astype(gate.dtype)
            grad_gate = grad_h * up * silu_grad
            grad_up = grad_h * silu_gate
            grad_gate_up = jnp.concatenate([grad_gate, grad_up], axis=-1)

            grad_recv_x = grad_recv_x.at[dst_rank].add(grad_gate_up @ weights.T)
            grad_w_gate_up = grad_w_gate_up.at[dst_rank, local_expert].add(x.T @ grad_gate_up)
    return grad_recv_x, grad_w_gate_up


def dispatch_moe_dispatch_up_grad_reference(
    prepacked: MoeDispatchUpPrepackedSend,
    grad_recv_x: jax.Array,
    *,
    tokens_per_rank: int,
    recv_capacity: int | None = None,
) -> jax.Array:
    """Route dispatched-row gradients back to source-rank token gradients."""

    send_x = prepacked.send_x_by_dst
    ep_size, _, send_capacity, hidden = send_x.shape
    if recv_capacity is None:
        recv_capacity = send_capacity
    if grad_recv_x.shape != (ep_size, recv_capacity, hidden):
        raise ValueError(
            "grad_recv_x must have shape [EP, recv_capacity, hidden]; "
            f"got {grad_recv_x.shape}, expected {(ep_size, recv_capacity, hidden)}"
        )

    grad_x_by_rank = jnp.zeros((ep_size, tokens_per_rank, hidden), dtype=grad_recv_x.dtype)
    send_positions = jnp.arange(send_capacity, dtype=jnp.int32)
    for src_rank in range(ep_size):
        for dst_rank in range(ep_size):
            valid_send = send_positions < prepacked.send_count_by_dst[src_rank, dst_rank]
            row = prepacked.send_row_by_dst[src_rank, dst_rank]
            valid_recv = valid_send & (row < recv_capacity)
            safe_row = jnp.where(valid_recv, row, 0)
            token_idx = prepacked.send_src_token_idx_by_dst[src_rank, dst_rank]
            grad_x_by_rank = grad_x_by_rank.at[src_rank, token_idx].add(
                grad_recv_x[dst_rank, safe_row] * valid_recv[:, None]
            )
    return grad_x_by_rank


def moe_dispatch_up_reference_bwd(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    w_gate_up_by_rank: jax.Array,
    grad_dispatch_up: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float = 1.25,
    recv_capacity: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Backward oracle for dispatch + W13/SiLU.

    Returns gradients for `x_by_rank` and `w_gate_up_by_rank`. Expert ids and
    router weights are routing metadata for this subkernel and are not
    differentiated here.
    """

    prepacked = prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids_by_rank,
        router_weights_by_rank,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        recv_capacity=recv_capacity,
    )
    actual_recv_capacity = recv_capacity if recv_capacity is not None else prepacked.send_x_by_dst.shape[2]
    layout = dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=actual_recv_capacity)
    grad_recv_x, grad_w_gate_up = compute_moe_up_from_layout_reference_bwd(layout, w_gate_up_by_rank, grad_dispatch_up)
    grad_x_by_rank = dispatch_moe_dispatch_up_grad_reference(
        prepacked,
        grad_recv_x,
        tokens_per_rank=x_by_rank.shape[1],
        recv_capacity=actual_recv_capacity,
    )
    return grad_x_by_rank, grad_w_gate_up
