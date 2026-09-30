# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP intranode expert-parallel Grug MoE backend.

DeepEP source: https://github.com/deepseek-ai/DeepEP
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from shape_extensions import IntVar

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _zero_dropped_assignments,
    _zero_inactive_grouped_rows,
    CapacityDrops,
    split_moe_w13_output,
)
from levanter.kernels.deepep import deepep_combine_intranode, deepep_dispatch_intranode, deepep_get_dispatch_layout


class DeepEPLocalAssignments(NamedTuple):
    """Local expert assignment batch after DeepEP token dispatch.

    Attributes:
        x_dispatch: Received token activations repeated once per local expert assignment.
        assignment_weights: Combine weights aligned with `x_dispatch`.
        recv_token_indices: Receive-buffer token row for each local assignment.
        local_group_sizes: Assignment counts per local expert for `ragged_dot`.

    Not generic over the assignment-count axis: it is ``max_recv_tokens * topk``, a
    capacity derived from ``ep_size`` at the call site, so it stays an unknown length.
    """

    x_dispatch: jax.Array[[int, int]]
    assignment_weights: jax.Array[[int]]
    recv_token_indices: jax.Array[[int]]
    local_group_sizes: jax.Array[[int]]


def _pack_deepep_local_assignments[Trecv: IntVar, K: IntVar, H: IntVar](
    recv_x: jax.Array[[Trecv, H]],
    recv_topk_idx: jax.Array[[Trecv, K]],
    recv_topk_weights: jax.Array[[Trecv, K]],
    *,
    local_experts: int,
    num_recv_tokens: jax.Array[[]],
) -> DeepEPLocalAssignments:
    with jax.named_scope("deepep_pack_local_assignments"):
        max_recv_tokens, topk = recv_topk_idx.shape
        total_assignments = max_recv_tokens * topk

        recv_token_indices: jax.Array[[Trecv * K]] = jnp.repeat(jnp.arange(max_recv_tokens, dtype=jnp.int32), topk)
        expert_flat: jax.Array[[Trecv * K]] = recv_topk_idx.reshape(-1).astype(jnp.int32)
        recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < num_recv_tokens
        local_mask = recv_valid[:, None] & (recv_topk_idx >= 0) & (recv_topk_idx < local_experts)
        local_mask_flat: jax.Array[[Trecv * K]] = local_mask.reshape(-1)
        local_bucket = jnp.where(local_mask_flat, expert_flat, local_experts)
        local_group_sizes = jnp.bincount(local_bucket, length=local_experts + 1).astype(jnp.int32)[:-1]
        total_valid = jnp.sum(local_group_sizes, dtype=jnp.int32)

        flat_positions = jnp.arange(total_assignments, dtype=jnp.int32)
        order_key = local_bucket * total_assignments + flat_positions
        max_order_key = (local_experts + 1) * total_assignments
        selection_key: jax.Array[[Trecv * K]] = jnp.where(local_mask_flat, max_order_key - order_key, -1)
        # pyrefly: ignore[bad-assignment]  # top_k with a non-literal `k` falls back to a rank-0 result.
        _, sorted_assignment_indices = jax.lax.top_k(selection_key, total_assignments)
        sorted_assignment_indices: jax.Array[[int]] = sorted_assignment_indices

        recv_token_indices = jnp.take(recv_token_indices, sorted_assignment_indices, axis=0)
        x_dispatch = jnp.take(recv_x, recv_token_indices, axis=0)
        assignment_weights = jnp.take(recv_topk_weights.reshape(-1), sorted_assignment_indices, axis=0).astype(
            recv_x.dtype
        )
        valid_sorted = jnp.arange(total_assignments, dtype=jnp.int32) < total_valid
        x_dispatch = jnp.where(valid_sorted[:, None], x_dispatch, 0)
        assignment_weights = jnp.where(valid_sorted, assignment_weights, 0)
        return DeepEPLocalAssignments(x_dispatch, assignment_weights, recv_token_indices, local_group_sizes)


def _collapse_deepep_local_assignments[H: IntVar](
    out_dispatch: jax.Array[[int, H]],
    assignment_weights: jax.Array[[int]],
    recv_token_indices: jax.Array[[int]],
    *,
    recv_capacity: int,
    num_recv_tokens: jax.Array[[]],
) -> jax.Array[[int, H]]:
    with jax.named_scope("deepep_collapse_local_assignments"):
        # The `jax.ops` stubs are not shape-typed; their params take the nominally distinct `basearray.Array`.
        segment_sum_out = jax.ops.segment_sum(
            # pyrefly: ignore[bad-argument-type]  # jax.ops.segment_sum is outside the shape stubs.
            out_dispatch * assignment_weights[:, None],
            # pyrefly: ignore[bad-argument-type]  # jax.ops.segment_sum is outside the shape stubs.
            recv_token_indices,
            num_segments=recv_capacity,
            indices_are_sorted=False,
        )
        # pyrefly: ignore[bad-assignment]  # jax.ops.segment_sum is outside the shape stubs.
        recv_out: jax.Array[[int, H]] = segment_sum_out
        recv_valid = jnp.arange(recv_capacity, dtype=jnp.int32) < num_recv_tokens
        result: jax.Array[[int, H]] = jnp.where(recv_valid[:, None], recv_out, 0)
        return result


def _moe_mlp_ep_deepep_local[Tlocal: IntVar, K: IntVar, H: IntVar, Elocal: IntVar, I: IntVar, I2: IntVar](
    x_local: jax.Array[[Tlocal, H]],
    selected_experts_local: jax.Array[[Tlocal, K]],
    combine_weights_local: jax.Array[[Tlocal, K]],
    token_valid_local: jax.Array[[Tlocal]],
    moe_w13_local: jax.Array[[Elocal, H, I2]],
    moe_w2_local: jax.Array[[Elocal, I, H]],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
) -> tuple[jax.Array[[Tlocal, H]], CapacityDrops]:
    """DeepEP dispatch/combine path for an intranode expert mesh."""
    del capacity_factor, token_sharding_axes  # DeepEP is dropless, so it reduces nothing over tokens
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    ep_size = num_experts // local_experts
    max_recv_tokens = x_local.shape[0] * ep_size
    selected_experts_local = jnp.where(token_valid_local[:, None], selected_experts_local, -1)
    combine_weights_local = jnp.where(token_valid_local[:, None], combine_weights_local, 0)

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
        )
        x_dispatch = tree_checkpoint_name(local_assignments.x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        cumulative_group_sizes = jnp.cumsum(local_assignments.local_group_sizes).astype(jnp.int32)

    with jax.named_scope("moe_up_down"):
        # Rows past the last group are unspecified kernel output, but every consumer is
        # row-local or group-bounded, so only the combine boundary below needs zeroing.
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, local_assignments.local_group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch = _zero_inactive_grouped_rows(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, local_assignments.local_group_sizes),
            cumulative_group_sizes,
        )
        out_dispatch = tree_checkpoint_name(
            out_dispatch,
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("combine"):
        recv_out = _collapse_deepep_local_assignments(
            out_dispatch,
            local_assignments.assignment_weights,
            local_assignments.recv_token_indices,
            recv_capacity=recv_x.shape[0],
            num_recv_tokens=num_recv_tokens_scalar,
        )
        with jax.named_scope("deepep_combine_transport"):
            combined, _ = deepep_combine_intranode(
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
    no_drops = _zero_dropped_assignments()
    # The FFI combine result is unshaped; pin it so the masked `where` broadcasts to [Tlocal, H].
    out_local: jax.Array[[Tlocal, H]] = combined
    out = jnp.where(token_valid_local[:, None], out_local, 0).astype(x_local.dtype)
    return out, CapacityDrops(
        sender_dropped=no_drops,
        receiver_dropped=no_drops,
    )
