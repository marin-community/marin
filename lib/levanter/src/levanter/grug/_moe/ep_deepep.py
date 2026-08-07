# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP intranode expert-parallel Grug MoE backend.

DeepEP source: https://github.com/deepseek-ai/DeepEP
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    split_moe_w13_output,
)
from levanter.grug.sharding import _batch_axes
from levanter.kernels.deepep import deepep_combine_intranode, deepep_dispatch_intranode, deepep_get_dispatch_layout


class DeepEPLocalAssignments(NamedTuple):
    """Local expert assignment batch after DeepEP token dispatch.

    Attributes:
        x_dispatch: Received token activations repeated once per local expert assignment.
        assignment_weights: Combine weights aligned with `x_dispatch`.
        assignment_positions: Original flattened `[received token, route slot]` positions aligned with `x_dispatch`.
        assignment_valid: Whether each expert-sorted assignment belongs to a local expert.
        recv_token_indices: Receive-buffer token row for each local assignment.
        local_group_sizes: Assignment counts per local expert for `ragged_dot`.
        overflow_assignments: Number of valid local assignments that did not fit the static capacity.
    """

    x_dispatch: Float[Array, "TK H"]
    assignment_weights: Float[Array, "TK"]
    assignment_positions: Int[Array, "TK"]
    assignment_valid: Bool[Array, "TK"]
    recv_token_indices: Int[Array, "TK"]
    local_group_sizes: Int[Array, "Elocal"]
    overflow_assignments: Int[Array, ""]


def _pack_deepep_local_assignments(
    recv_x: Float[Array, "Trecv H"],
    recv_topk_idx: Int[Array, "Trecv K"],
    recv_topk_weights: Float[Array, "Trecv K"],
    *,
    local_experts: int,
    num_recv_tokens: Int[Array, ""],
    assignment_capacity: int | None = None,
) -> DeepEPLocalAssignments:
    with jax.named_scope("deepep_pack_local_assignments"):
        max_recv_tokens, topk = recv_topk_idx.shape
        total_assignments = max_recv_tokens * topk
        if assignment_capacity is None:
            assignment_capacity = total_assignments
        if assignment_capacity <= 0 or assignment_capacity > total_assignments:
            raise ValueError(f"assignment_capacity must be in [1, {total_assignments}], got {assignment_capacity}")

        recv_token_indices = jnp.repeat(jnp.arange(max_recv_tokens, dtype=jnp.int32), topk)
        expert_flat = recv_topk_idx.reshape(-1).astype(jnp.int32)
        recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < num_recv_tokens
        local_mask = recv_valid[:, None] & (recv_topk_idx >= 0) & (recv_topk_idx < local_experts)
        local_mask_flat = local_mask.reshape(-1)
        local_bucket = jnp.where(local_mask_flat, expert_flat, local_experts)
        total_valid = jnp.sum(local_mask_flat, dtype=jnp.int32)

        flat_positions = jnp.arange(total_assignments, dtype=jnp.int32)
        order_key = local_bucket * total_assignments + flat_positions
        max_order_key = (local_experts + 1) * total_assignments
        selection_key = jnp.where(local_mask_flat, max_order_key - order_key, -1)
        _, sorted_assignment_indices = jax.lax.top_k(selection_key, assignment_capacity)

        recv_token_indices = jnp.take(recv_token_indices, sorted_assignment_indices, axis=0)
        x_dispatch = jnp.take(recv_x, recv_token_indices, axis=0)
        assignment_weights = jnp.take(recv_topk_weights.reshape(-1), sorted_assignment_indices, axis=0).astype(
            jnp.float32
        )
        valid_sorted = jnp.arange(assignment_capacity, dtype=jnp.int32) < total_valid
        selected_local_bucket = jnp.take(local_bucket, sorted_assignment_indices, axis=0)
        selected_local_bucket = jnp.where(valid_sorted, selected_local_bucket, local_experts)
        local_group_sizes = jnp.bincount(selected_local_bucket, length=local_experts + 1).astype(jnp.int32)[:-1]
        x_dispatch = jnp.where(valid_sorted[:, None], x_dispatch, 0)
        assignment_weights = jnp.where(valid_sorted, assignment_weights, 0)
        return DeepEPLocalAssignments(
            x_dispatch,
            assignment_weights,
            sorted_assignment_indices,
            valid_sorted,
            recv_token_indices,
            local_group_sizes,
            jnp.maximum(total_valid - assignment_capacity, 0),
        )


def _deepep_assignment_capacity(
    *,
    local_tokens: int,
    topk: int,
    capacity_factor: float,
) -> int:
    """Return the local assignment capacity derived from the sender's nominal assignment count."""
    if local_tokens <= 0:
        raise ValueError(f"local_tokens must be positive, got {local_tokens}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if not math.isfinite(capacity_factor) or capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be finite and positive, got {capacity_factor}")

    return int(math.ceil(capacity_factor * local_tokens * topk))


def _collapse_deepep_local_assignments(
    out_dispatch: Float[Array, "TK H"],
    assignment_positions: Int[Array, "TK"],
    assignment_valid: Bool[Array, "TK"],
    recv_topk_weights: Float[Array, "Trecv K"],
    *,
    recv_capacity: int,
    num_recv_tokens: Int[Array, ""],
) -> Float[Array, "Trecv H"]:
    with jax.named_scope("deepep_collapse_local_assignments"):
        if recv_topk_weights.shape[0] != recv_capacity:
            raise ValueError(
                f"recv_topk_weights leading dim={recv_topk_weights.shape[0]} must equal recv_capacity={recv_capacity}"
            )
        topk = recv_topk_weights.shape[1]
        compact_capacity = out_dispatch.shape[0]

        # Restore semantic position order in compact storage. Each queried position is unique, so search+gather needs
        # neither a scatter reduction nor atomics. Accumulating the dense route slots in the loop below fixes the
        # floating-point order independently of expert load and the expert-grouped execution order.
        position_order = jnp.argsort(assignment_positions)
        sorted_positions = jnp.take(assignment_positions, position_order, axis=0)
        sorted_valid = jnp.take(assignment_valid, position_order, axis=0)
        sorted_outputs = jnp.take(out_dispatch, position_order, axis=0)
        recv_out = jnp.zeros((recv_capacity, out_dispatch.shape[1]), dtype=jnp.float32)
        for route_slot in range(topk):
            desired_positions = jnp.arange(recv_capacity, dtype=jnp.int32) * topk + route_slot
            compact_rows = jnp.searchsorted(sorted_positions, desired_positions, side="left")
            safe_rows = jnp.minimum(compact_rows, compact_capacity - 1)
            present = (
                (compact_rows < compact_capacity)
                & jnp.take(sorted_valid, safe_rows, axis=0)
                & (jnp.take(sorted_positions, safe_rows, axis=0) == desired_positions)
            )
            route_outputs = jnp.take(sorted_outputs, safe_rows, axis=0).astype(jnp.float32)
            route_contribution = route_outputs * recv_topk_weights[:, route_slot, None].astype(jnp.float32)
            recv_out = recv_out + jnp.where(present[:, None], route_contribution, 0)
        recv_valid = jnp.arange(recv_capacity, dtype=jnp.int32) < num_recv_tokens
        return jnp.where(recv_valid[:, None], recv_out, 0).astype(out_dispatch.dtype)


def _moe_mlp_ep_deepep_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """DeepEP dispatch/combine path for an intranode expert mesh."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    ep_size = num_experts // local_experts
    max_recv_tokens = x_local.shape[0] * ep_size
    topk = selected_experts_local.shape[1]
    assignment_capacity = _deepep_assignment_capacity(
        local_tokens=x_local.shape[0],
        topk=topk,
        capacity_factor=capacity_factor,
    )
    assignment_capacity = min(assignment_capacity, max_recv_tokens * topk)

    with jax.named_scope("dispatch"):
        with jax.named_scope("deepep_layout"):
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
                selected_experts_local,
                num_ranks=ep_size,
                num_experts=num_experts,
            )
        with jax.named_scope("deepep_dispatch_transport"):
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                _local_expert_counts,
                num_recv_tokens,
            ) = deepep_dispatch_intranode(
                x_local,
                selected_experts_local,
                combine_weights_local,
                num_tokens_per_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
            )
        num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)

        local_assignments = _pack_deepep_local_assignments(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            local_experts=local_experts,
            num_recv_tokens=num_recv_tokens_scalar,
            assignment_capacity=assignment_capacity,
        )
        x_dispatch = tree_checkpoint_name(local_assignments.x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, local_assignments.local_group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, local_assignments.local_group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("combine"):
        recv_out = _collapse_deepep_local_assignments(
            out_dispatch,
            local_assignments.assignment_positions,
            local_assignments.assignment_valid,
            recv_topk_weights,
            recv_capacity=recv_x.shape[0],
            num_recv_tokens=num_recv_tokens_scalar,
        )
        with jax.named_scope("deepep_combine_transport"):
            out_local, _ = deepep_combine_intranode(
                recv_out,
                recv_topk_weights,
                recv_src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                num_recv_tokens,
                is_token_in_rank,
            )
        dropped_total = jax.lax.psum(
            local_assignments.overflow_assignments,
            _batch_axes(jax.sharding.get_abstract_mesh()),
        )
    return out_local.astype(x_local.dtype), dropped_total
