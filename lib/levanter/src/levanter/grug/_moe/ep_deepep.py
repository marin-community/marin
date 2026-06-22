# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP expert-parallel Grug MoE backends.

DeepEP source: https://github.com/deepseek-ai/DeepEP
"""

import math
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DEEPEP_ASSIGNMENT_WEIGHTS,
    _CHECKPOINT_DEEPEP_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_IS_TOKEN_IN_RANK,
    _CHECKPOINT_DEEPEP_LOCAL_GROUP_SIZES,
    _CHECKPOINT_DEEPEP_NUM_RECV_TOKENS,
    _CHECKPOINT_DEEPEP_NUM_RECV_RDMA_TOKENS,
    _CHECKPOINT_DEEPEP_GBL_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RANK_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RDMA_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_GBL_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_GBL_RANK_PREFIX_SUM,
    _CHECKPOINT_DEEPEP_RECV_RDMA_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_RDMA_RANK_PREFIX_SUM,
    _CHECKPOINT_DEEPEP_RECV_SRC_IDX,
    _CHECKPOINT_DEEPEP_RECV_SRC_META,
    _CHECKPOINT_DEEPEP_RECV_TOPK_IDX,
    _CHECKPOINT_DEEPEP_RECV_TOKEN_INDICES,
    _CHECKPOINT_DEEPEP_RECV_TOPK_WEIGHTS,
    _CHECKPOINT_DEEPEP_RECV_X,
    _CHECKPOINT_DEEPEP_SEND_NVL_HEAD,
    _CHECKPOINT_DEEPEP_SEND_RDMA_HEAD,
    _CHECKPOINT_DEEPEP_SEND_HEAD,
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    MOE_REMAT_EXPERT_OFFLOAD_NAMES,
    MOE_REMAT_EXPERT_SAVE_NAMES,
    MOE_REMAT_HIDDEN_OFFLOAD_NAMES,
    MOE_REMAT_HIDDEN_SAVE_NAMES,
    MOE_REMAT_OFFLOAD_NAMES,
    MOE_REMAT_OUTPUT_OFFLOAD_NAMES,
    MOE_REMAT_OUTPUT_SAVE_NAMES,
    MOE_REMAT_SAVE_NAMES,
    MoERematMode,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.kernels.deepep import (
    deepep_collapse_local_assignments,
    deepep_combine_internode,
    deepep_combine_internode_x_only,
    deepep_combine_internode_with_local_collapse,
    deepep_combine_internode_x_only_with_local_collapse,
    deepep_combine_intranode,
    deepep_dispatch_internode,
    deepep_dispatch_intranode,
    deepep_dispatch_intranode_with_assignments,
    deepep_get_dispatch_layout,
    deepep_pack_local_assignments_from_counts,
)

_DEEPEP_RANKS_PER_NODE_ENV = "DEEPEP_RANKS_PER_NODE"
_DEEPEP_INTERNODE_COLLAPSE_MODE_ENV = "LEVANTER_DEEPEP_INTERNODE_COLLAPSE_MODE"
_DEEPEP_INTERNODE_COMBINE_X_ONLY_ENV = "LEVANTER_DEEPEP_INTERNODE_COMBINE_X_ONLY"
_DEEPEP_INTERNODE_RECV_CAPACITY_MODE_ENV = "LEVANTER_DEEPEP_INTERNODE_RECV_CAPACITY_MODE"
_DEEPEP_INTERNODE_COLLAPSE_SCATTER = "scatter"
_DEEPEP_INTERNODE_COLLAPSE_GATHER = "gather"
_DEEPEP_INTERNODE_COLLAPSE_FFI = "ffi"
_DEEPEP_INTERNODE_COLLAPSE_FUSED_COMBINE = "fused_combine"
_DEEPEP_INTERNODE_RECV_CAPACITY_WORST_CASE = "worst_case"
_DEEPEP_INTERNODE_RECV_CAPACITY_LOCAL_ASSIGNMENT = "local_assignment"


def _deepep_num_local_ranks() -> int:
    raw = os.environ.get(_DEEPEP_RANKS_PER_NODE_ENV, "8")
    try:
        num_local_ranks = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_DEEPEP_RANKS_PER_NODE_ENV}={raw!r} must be an integer") from exc
    if num_local_ranks <= 0:
        raise ValueError(f"{_DEEPEP_RANKS_PER_NODE_ENV} must be positive, got {num_local_ranks}")
    return num_local_ranks


def _deepep_internode_collapse_mode() -> str:
    mode = os.environ.get(_DEEPEP_INTERNODE_COLLAPSE_MODE_ENV, _DEEPEP_INTERNODE_COLLAPSE_SCATTER)
    if mode not in {
        _DEEPEP_INTERNODE_COLLAPSE_SCATTER,
        _DEEPEP_INTERNODE_COLLAPSE_GATHER,
        _DEEPEP_INTERNODE_COLLAPSE_FFI,
        _DEEPEP_INTERNODE_COLLAPSE_FUSED_COMBINE,
    }:
        raise ValueError(
            f"{_DEEPEP_INTERNODE_COLLAPSE_MODE_ENV} must be "
            f"{_DEEPEP_INTERNODE_COLLAPSE_SCATTER!r}, {_DEEPEP_INTERNODE_COLLAPSE_GATHER!r}, "
            f"{_DEEPEP_INTERNODE_COLLAPSE_FFI!r}, or {_DEEPEP_INTERNODE_COLLAPSE_FUSED_COMBINE!r}, got {mode!r}"
        )
    return mode


def _deepep_internode_combine_x_only() -> bool:
    return os.environ.get(_DEEPEP_INTERNODE_COMBINE_X_ONLY_ENV, "").lower() in {"1", "true", "yes", "on"}


def _deepep_internode_recv_capacity(
    *,
    local_tokens: int,
    topk: int,
    num_rdma_ranks: int,
    local_assignment_capacity: int,
) -> int:
    mode = os.environ.get(_DEEPEP_INTERNODE_RECV_CAPACITY_MODE_ENV, _DEEPEP_INTERNODE_RECV_CAPACITY_WORST_CASE)
    if mode == _DEEPEP_INTERNODE_RECV_CAPACITY_WORST_CASE:
        return local_tokens * topk * num_rdma_ranks
    if mode == _DEEPEP_INTERNODE_RECV_CAPACITY_LOCAL_ASSIGNMENT:
        return local_assignment_capacity
    raise ValueError(
        f"{_DEEPEP_INTERNODE_RECV_CAPACITY_MODE_ENV} must be "
        f"{_DEEPEP_INTERNODE_RECV_CAPACITY_WORST_CASE!r} or "
        f"{_DEEPEP_INTERNODE_RECV_CAPACITY_LOCAL_ASSIGNMENT!r}, got {mode!r}"
    )


def _deepep_combine_internode_output(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
) -> jax.Array:
    if _deepep_internode_combine_x_only():
        return deepep_combine_internode_x_only(
            recv_x,
            is_token_in_rank,
            recv_src_meta,
            rdma_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            send_rdma_head,
            send_nvl_head,
            num_recv_tokens,
            num_recv_rdma_tokens,
            num_topk=recv_topk_weights.shape[1],
        )
    out_local, _ = deepep_combine_internode(
        recv_x,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        rdma_channel_prefix_matrix,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
    )
    return out_local


def _tokens_per_rdma_rank(num_tokens_per_rank: jax.Array, *, num_local_ranks: int) -> jax.Array:
    """Collapse global-rank token counts to DeepEP RDMA node-rank counts."""
    num_ranks = int(num_tokens_per_rank.shape[0])
    if num_ranks % num_local_ranks != 0:
        raise ValueError(
            f"DeepEP internode requires global rank count divisible by ranks per node, "
            f"got {num_ranks=} {num_local_ranks=}"
        )
    num_rdma_ranks = num_ranks // num_local_ranks
    if num_rdma_ranks <= 1:
        raise ValueError(
            "DeepEP internode requires more than one RDMA node rank; "
            f"got {num_ranks=} {num_local_ranks=} {num_rdma_ranks=}"
        )
    return jnp.sum(jnp.reshape(num_tokens_per_rank, (num_rdma_ranks, num_local_ranks)), axis=1)


def _pack_local_assignments_from_counts_jax(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    raw_local_group_sizes: jax.Array,
    *,
    assignment_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pack DeepEP-received tokens by local expert using JAX-only layout logic."""
    if assignment_capacity <= 0:
        raise ValueError(f"assignment_capacity must be positive, got {assignment_capacity}")

    recv_x = jnp.asarray(recv_x, dtype=jnp.bfloat16)
    recv_topk_idx = jnp.asarray(recv_topk_idx, dtype=jnp.int32)
    recv_topk_weights = jnp.asarray(recv_topk_weights, dtype=jnp.float32)
    raw_local_group_sizes = jnp.asarray(raw_local_group_sizes, dtype=jnp.int32)
    num_recv_tokens = jnp.reshape(jnp.asarray(num_recv_tokens, dtype=jnp.int32), (1,))

    recv_capacity, hidden = recv_x.shape
    topk = recv_topk_idx.shape[1]
    local_experts = raw_local_group_sizes.shape[0]
    max_assignments = recv_capacity * topk

    assignment = jnp.arange(max_assignments, dtype=jnp.int32)
    token_index = assignment // topk
    expert = jnp.reshape(recv_topk_idx, (max_assignments,))
    weights = jnp.reshape(recv_topk_weights, (max_assignments,))
    active = assignment < num_recv_tokens[0] * topk
    valid = active & (expert >= 0) & (expert < local_experts)

    # Count prior valid assignments for the same local expert. This is deliberately
    # simple for the internode smoke path; the optimized path can use the CUDA FFI.
    same_expert = expert[:, None] == expert[None, :]
    prior = assignment[None, :] < assignment[:, None]
    ordinal = jnp.sum(same_expert & prior & valid[None, :], axis=1, dtype=jnp.int32)
    expert_offsets = jnp.cumsum(raw_local_group_sizes, dtype=jnp.int32) - raw_local_group_sizes
    safe_expert = jnp.clip(expert, 0, local_experts - 1)
    destination = expert_offsets[safe_expert] + ordinal
    accepted = valid & (destination < assignment_capacity)
    safe_destination = jnp.where(accepted, destination, 0)

    x_rows = jnp.where(accepted[:, None], recv_x[token_index], jnp.zeros((max_assignments, hidden), recv_x.dtype))
    x_dispatch = jnp.zeros((assignment_capacity, hidden), recv_x.dtype).at[safe_destination].set(x_rows)
    assignment_weights = (
        jnp.zeros((assignment_capacity,), recv_x.dtype).at[safe_destination].set(weights.astype(recv_x.dtype))
    )
    recv_token_indices = jnp.zeros((assignment_capacity,), jnp.int32).at[safe_destination].set(token_index)
    assignment_destinations = (
        jnp.full((max_assignments,), -1, jnp.int32).at[assignment].set(jnp.where(accepted, destination, -1))
    )
    return x_dispatch, assignment_weights, recv_token_indices, assignment_destinations


def _collapse_local_assignments_jax(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    local_group_sizes: jax.Array,
    *,
    recv_capacity: int,
) -> jax.Array:
    """Undo local expert packing after expert MLP evaluation."""
    out_dispatch = jnp.asarray(out_dispatch, dtype=jnp.bfloat16)
    assignment_weights = jnp.asarray(assignment_weights, dtype=jnp.bfloat16)
    recv_token_indices = jnp.asarray(recv_token_indices, dtype=jnp.int32)
    local_group_sizes = jnp.asarray(local_group_sizes, dtype=jnp.int32)

    assignment_capacity = out_dispatch.shape[0]
    valid = jnp.arange(assignment_capacity, dtype=jnp.int32) < jnp.sum(local_group_sizes, dtype=jnp.int32)
    safe_token_indices = jnp.where(valid, recv_token_indices, 0)
    weighted = jnp.where(valid[:, None], out_dispatch * assignment_weights[:, None], 0)
    return jnp.zeros((recv_capacity, out_dispatch.shape[1]), out_dispatch.dtype).at[safe_token_indices].add(weighted)


def _collapse_local_assignments_gather_jax(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    recv_capacity: int,
) -> jax.Array:
    """Undo local assignment packing by gathering packed rows per received token."""
    out_dispatch = jnp.asarray(out_dispatch, dtype=jnp.bfloat16)
    assignment_weights = jnp.asarray(assignment_weights, dtype=jnp.bfloat16)
    assignment_destinations = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    local_group_sizes = jnp.asarray(local_group_sizes, dtype=jnp.int32)
    num_recv_tokens = jnp.reshape(jnp.asarray(num_recv_tokens, dtype=jnp.int32), (1,))

    topk = assignment_destinations.shape[0] // recv_capacity
    destinations = jnp.reshape(assignment_destinations, (recv_capacity, topk))
    active_recv = jnp.arange(recv_capacity, dtype=jnp.int32) < num_recv_tokens[0]
    accepted_total = jnp.sum(local_group_sizes, dtype=jnp.int32)
    valid = active_recv[:, None] & (destinations >= 0) & (destinations < accepted_total)
    safe_destinations = jnp.where(valid, destinations, 0)
    gathered = jnp.take(out_dispatch, safe_destinations, axis=0)
    gathered_weights = jnp.take(assignment_weights, safe_destinations, axis=0).astype(gathered.dtype)
    weighted = gathered * gathered_weights[:, :, None]
    return jnp.sum(jnp.where(valid[:, :, None], weighted, 0), axis=1)


def _assignment_destinations_have_recv_topk_layout(assignment_destinations: jax.Array, *, recv_capacity: int) -> bool:
    """Return whether destination metadata can be viewed as per-received-token top-k rows."""
    return recv_capacity > 0 and assignment_destinations.shape[0] % recv_capacity == 0


def _deepep_moe_up_down(
    x_dispatch: Float[Array, "TK D"],
    local_group_sizes: Int[Array, "EL"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "TK D"]:
    with jax.named_scope("moe_expert_mlp/w13_ragged_dot"):
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, local_group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
    with jax.named_scope("moe_expert_mlp/split_gate_up"):
        moe_dim = moe_w2_local.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
    with jax.named_scope("moe_expert_mlp/activation"):
        hidden = activation_fn(gate) * up
    with jax.named_scope("moe_expert_mlp/w2_ragged_dot"):
        return tree_checkpoint_name(
            ragged_dot(hidden, moe_w2_local, local_group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )


def _deepep_moe_up_down_remat(
    x_dispatch: Float[Array, "TK D"],
    local_group_sizes: Int[Array, "EL"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    remat_mode: MoERematMode,
) -> Float[Array, "TK D"]:
    if remat_mode == "none":
        return _deepep_moe_up_down(
            x_dispatch,
            local_group_sizes,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
        )

    policy = None
    if remat_mode == "save_moe":
        policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
    elif remat_mode == "offload_moe":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=(),
            names_which_can_be_offloaded=MOE_REMAT_OFFLOAD_NAMES,
            offload_src="device",
            offload_dst="pinned_host",
        )
    elif remat_mode == "offload_moe_hidden":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=MOE_REMAT_HIDDEN_SAVE_NAMES,
            names_which_can_be_offloaded=MOE_REMAT_HIDDEN_OFFLOAD_NAMES,
            offload_src="device",
            offload_dst="pinned_host",
        )
    elif remat_mode == "offload_moe_output":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=MOE_REMAT_OUTPUT_SAVE_NAMES,
            names_which_can_be_offloaded=MOE_REMAT_OUTPUT_OFFLOAD_NAMES,
            offload_src="device",
            offload_dst="pinned_host",
        )
    elif remat_mode == "offload_moe_expert":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=MOE_REMAT_EXPERT_SAVE_NAMES,
            names_which_can_be_offloaded=MOE_REMAT_EXPERT_OFFLOAD_NAMES,
            offload_src="device",
            offload_dst="pinned_host",
        )

    def compute(
        remat_x_dispatch: jax.Array,
        remat_local_group_sizes: jax.Array,
        remat_moe_w13_local: jax.Array,
        remat_moe_w2_local: jax.Array,
    ) -> jax.Array:
        return _deepep_moe_up_down(
            remat_x_dispatch,
            remat_local_group_sizes,
            remat_moe_w13_local,
            remat_moe_w2_local,
            activation_fn=activation_fn,
        )

    return jax.checkpoint(compute, policy=policy)(x_dispatch, local_group_sizes, moe_w13_local, moe_w2_local)


def _moe_mlp_ep_deepep_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    remat_mode: MoERematMode = "none",
    composed_dispatch: bool = False,
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    """DeepEP dispatch/combine path for an intranode expert mesh."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    ep_size = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    local_capacity = int(math.ceil(capacity_factor * x_local.shape[0] * topk))
    local_capacity = max(local_experts, local_capacity)
    max_recv_tokens = x_local.shape[0] * ep_size

    with jax.named_scope("moe_ep_deepep/dispatch"):
        with jax.named_scope("deepep_layout"):
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
                selected_experts_local,
                num_ranks=ep_size,
                num_experts=num_experts,
            )
        with jax.named_scope("deepep_dispatch_transport"):
            if composed_dispatch:
                dispatch = deepep_dispatch_intranode(
                    x_local,
                    selected_experts_local,
                    combine_weights_local,
                    num_tokens_per_rank,
                    num_tokens_per_expert,
                    is_token_in_rank,
                    num_experts=num_experts,
                    max_recv_tokens=max_recv_tokens,
                )
            else:
                dispatch = deepep_dispatch_intranode_with_assignments(
                    x_local,
                    selected_experts_local,
                    combine_weights_local,
                    num_tokens_per_rank,
                    num_tokens_per_expert,
                    is_token_in_rank,
                    num_experts=num_experts,
                    max_recv_tokens=max_recv_tokens,
                    assignment_capacity=local_capacity,
                )
        recv_x = tree_checkpoint_name(dispatch.recv_x, _CHECKPOINT_DEEPEP_RECV_X)
        recv_topk_weights = tree_checkpoint_name(dispatch.recv_topk_weights, _CHECKPOINT_DEEPEP_RECV_TOPK_WEIGHTS)
        recv_src_idx = tree_checkpoint_name(dispatch.recv_src_idx, _CHECKPOINT_DEEPEP_RECV_SRC_IDX)
        rank_prefix_matrix = tree_checkpoint_name(dispatch.rank_prefix_matrix, _CHECKPOINT_DEEPEP_RANK_PREFIX_MATRIX)
        channel_prefix_matrix = tree_checkpoint_name(
            dispatch.channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_CHANNEL_PREFIX_MATRIX,
        )
        recv_channel_prefix_matrix = tree_checkpoint_name(
            dispatch.recv_channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_RECV_CHANNEL_PREFIX_MATRIX,
        )
        send_head = tree_checkpoint_name(dispatch.send_head, _CHECKPOINT_DEEPEP_SEND_HEAD)
        num_recv_tokens = tree_checkpoint_name(dispatch.num_recv_tokens, _CHECKPOINT_DEEPEP_NUM_RECV_TOKENS)
        is_token_in_rank = tree_checkpoint_name(is_token_in_rank, _CHECKPOINT_DEEPEP_IS_TOKEN_IN_RANK)
        raw_local_group_sizes = dispatch.local_expert_counts if composed_dispatch else dispatch.local_group_sizes
        local_group_sizes = _prefix_cap_counts(raw_local_group_sizes, capacity=local_capacity)
        accepted_total = jnp.sum(local_group_sizes, dtype=jnp.int32)
        dropped_local = jnp.sum(raw_local_group_sizes, dtype=jnp.int32) - accepted_total
        local_group_sizes = tree_checkpoint_name(local_group_sizes, _CHECKPOINT_DEEPEP_LOCAL_GROUP_SIZES)
        if composed_dispatch:
            recv_topk_idx = tree_checkpoint_name(dispatch.recv_topk_idx, _CHECKPOINT_DEEPEP_RECV_TOPK_IDX)
            with jax.named_scope("deepep_pack_local_assignments"):
                x_dispatch, assignment_weights, recv_token_indices, _, assignment_destinations = (
                    deepep_pack_local_assignments_from_counts(
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        num_recv_tokens,
                        raw_local_group_sizes,
                        assignment_capacity=local_capacity,
                    )
                )
        else:
            x_dispatch = dispatch.x_dispatch
            assignment_weights = dispatch.assignment_weights
            recv_token_indices = dispatch.recv_token_indices
            assignment_destinations = dispatch.assignment_destinations
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        assignment_weights = tree_checkpoint_name(
            assignment_weights,
            _CHECKPOINT_DEEPEP_ASSIGNMENT_WEIGHTS,
        )
        recv_token_indices = tree_checkpoint_name(
            recv_token_indices,
            _CHECKPOINT_DEEPEP_RECV_TOKEN_INDICES,
        )

    out_dispatch = _deepep_moe_up_down_remat(
        x_dispatch,
        local_group_sizes,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        remat_mode=remat_mode,
    )

    with jax.named_scope("moe_ep_deepep/combine"):
        with jax.named_scope("deepep_collapse_local_assignments"):
            recv_out = deepep_collapse_local_assignments(
                out_dispatch,
                assignment_weights,
                recv_token_indices,
                assignment_destinations,
                local_group_sizes,
                num_recv_tokens,
                recv_capacity=recv_x.shape[0],
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
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local.astype(x_local.dtype), dropped_total


def _moe_mlp_ep_deepep_internode_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    remat_mode: MoERematMode = "none",
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    """DeepEP internode dispatch/combine path for a process-per-GPU expert mesh."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    ep_size = num_experts // local_experts
    num_local_ranks = _deepep_num_local_ranks()
    num_rdma_ranks = ep_size // num_local_ranks
    if ep_size % num_local_ranks != 0 or num_rdma_ranks <= 1:
        raise ValueError(
            "DeepEP internode requires expert-parallel rank count to be an integer number of nodes, "
            f"got {ep_size=} {num_local_ranks=} {num_rdma_ranks=}"
        )
    topk = selected_experts_local.shape[1]
    local_capacity = int(math.ceil(capacity_factor * x_local.shape[0] * topk))
    local_capacity = max(local_experts, local_capacity)
    max_recv_tokens = _deepep_internode_recv_capacity(
        local_tokens=x_local.shape[0],
        topk=topk,
        num_rdma_ranks=num_rdma_ranks,
        local_assignment_capacity=local_capacity,
    )

    with jax.named_scope("moe_ep_deepep_internode/dispatch"):
        with jax.named_scope("deepep_layout"):
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
                selected_experts_local,
                num_ranks=ep_size,
                num_experts=num_experts,
            )
            num_tokens_per_rdma_rank = _tokens_per_rdma_rank(
                num_tokens_per_rank,
                num_local_ranks=num_local_ranks,
            )
        with jax.named_scope("deepep_dispatch_internode_transport"):
            dispatch = deepep_dispatch_internode(
                x_local,
                selected_experts_local,
                combine_weights_local,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_rdma_recv_tokens=max_recv_tokens,
                num_local_ranks=num_local_ranks,
                assignment_capacity=local_capacity,
            )
        recv_x = tree_checkpoint_name(dispatch.recv_x, _CHECKPOINT_DEEPEP_RECV_X)
        recv_topk_weights = tree_checkpoint_name(dispatch.recv_topk_weights, _CHECKPOINT_DEEPEP_RECV_TOPK_WEIGHTS)
        is_token_in_rank = tree_checkpoint_name(dispatch.is_token_in_rank, _CHECKPOINT_DEEPEP_IS_TOKEN_IN_RANK)
        recv_src_meta = tree_checkpoint_name(dispatch.recv_src_meta, _CHECKPOINT_DEEPEP_RECV_SRC_META)
        rdma_channel_prefix_matrix = tree_checkpoint_name(
            dispatch.rdma_channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_RDMA_CHANNEL_PREFIX_MATRIX,
        )
        recv_rdma_channel_prefix_matrix = tree_checkpoint_name(
            dispatch.recv_rdma_channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_RECV_RDMA_CHANNEL_PREFIX_MATRIX,
        )
        recv_rdma_rank_prefix_sum = tree_checkpoint_name(
            dispatch.recv_rdma_rank_prefix_sum,
            _CHECKPOINT_DEEPEP_RECV_RDMA_RANK_PREFIX_SUM,
        )
        gbl_channel_prefix_matrix = tree_checkpoint_name(
            dispatch.gbl_channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_GBL_CHANNEL_PREFIX_MATRIX,
        )
        recv_gbl_channel_prefix_matrix = tree_checkpoint_name(
            dispatch.recv_gbl_channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_RECV_GBL_CHANNEL_PREFIX_MATRIX,
        )
        recv_gbl_rank_prefix_sum = tree_checkpoint_name(
            dispatch.recv_gbl_rank_prefix_sum,
            _CHECKPOINT_DEEPEP_RECV_GBL_RANK_PREFIX_SUM,
        )
        send_rdma_head = tree_checkpoint_name(dispatch.send_rdma_head, _CHECKPOINT_DEEPEP_SEND_RDMA_HEAD)
        send_nvl_head = tree_checkpoint_name(dispatch.send_nvl_head, _CHECKPOINT_DEEPEP_SEND_NVL_HEAD)
        num_recv_tokens = tree_checkpoint_name(dispatch.num_recv_tokens, _CHECKPOINT_DEEPEP_NUM_RECV_TOKENS)
        num_recv_rdma_tokens = tree_checkpoint_name(
            dispatch.num_recv_rdma_tokens,
            _CHECKPOINT_DEEPEP_NUM_RECV_RDMA_TOKENS,
        )
        raw_local_group_sizes = dispatch.local_expert_counts
        local_group_sizes = dispatch.local_group_sizes
        accepted_total = jnp.sum(local_group_sizes, dtype=jnp.int32)
        dropped_local = jnp.sum(raw_local_group_sizes, dtype=jnp.int32) - accepted_total
        local_group_sizes = tree_checkpoint_name(local_group_sizes, _CHECKPOINT_DEEPEP_LOCAL_GROUP_SIZES)
        x_dispatch = dispatch.x_dispatch
        assignment_weights = dispatch.assignment_weights
        recv_token_indices = dispatch.recv_token_indices
        assignment_destinations = dispatch.assignment_destinations
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        assignment_weights = tree_checkpoint_name(
            assignment_weights,
            _CHECKPOINT_DEEPEP_ASSIGNMENT_WEIGHTS,
        )
        recv_token_indices = tree_checkpoint_name(
            recv_token_indices,
            _CHECKPOINT_DEEPEP_RECV_TOKEN_INDICES,
        )

    out_dispatch = _deepep_moe_up_down_remat(
        x_dispatch,
        local_group_sizes,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        remat_mode=remat_mode,
    )

    with jax.named_scope("moe_ep_deepep_internode/combine"):
        collapse_mode = _deepep_internode_collapse_mode()
        has_recv_topk_destination_layout = _assignment_destinations_have_recv_topk_layout(
            assignment_destinations, recv_capacity=recv_x.shape[0]
        )
        if collapse_mode == _DEEPEP_INTERNODE_COLLAPSE_FUSED_COMBINE and has_recv_topk_destination_layout:
            if _deepep_internode_combine_x_only():
                with jax.named_scope("deepep_combine_internode_x_only_with_local_collapse"):
                    out_local = deepep_combine_internode_x_only_with_local_collapse(
                        out_dispatch,
                        assignment_weights,
                        recv_token_indices,
                        assignment_destinations,
                        local_group_sizes,
                        is_token_in_rank,
                        recv_src_meta,
                        rdma_channel_prefix_matrix,
                        recv_rdma_channel_prefix_matrix,
                        recv_rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        recv_gbl_channel_prefix_matrix,
                        recv_gbl_rank_prefix_sum,
                        send_rdma_head,
                        send_nvl_head,
                        num_recv_tokens,
                        num_recv_rdma_tokens,
                        num_topk=recv_topk_weights.shape[1],
                    )
            else:
                with jax.named_scope("deepep_combine_internode_with_local_collapse"):
                    out_local, _ = deepep_combine_internode_with_local_collapse(
                        out_dispatch,
                        assignment_weights,
                        recv_token_indices,
                        assignment_destinations,
                        local_group_sizes,
                        recv_topk_weights,
                        is_token_in_rank,
                        recv_src_meta,
                        rdma_channel_prefix_matrix,
                        recv_rdma_channel_prefix_matrix,
                        recv_rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        recv_gbl_channel_prefix_matrix,
                        recv_gbl_rank_prefix_sum,
                        send_rdma_head,
                        send_nvl_head,
                        num_recv_tokens,
                        num_recv_rdma_tokens,
                    )
        elif collapse_mode == _DEEPEP_INTERNODE_COLLAPSE_GATHER and has_recv_topk_destination_layout:
            with jax.named_scope("deepep_collapse_local_assignments_gather_jax"):
                recv_out = _collapse_local_assignments_gather_jax(
                    out_dispatch,
                    assignment_weights,
                    assignment_destinations,
                    local_group_sizes,
                    num_recv_tokens,
                    recv_capacity=recv_x.shape[0],
                )
            with jax.named_scope("deepep_combine_internode_transport"):
                out_local = _deepep_combine_internode_output(
                    recv_out,
                    recv_topk_weights,
                    is_token_in_rank,
                    recv_src_meta,
                    rdma_channel_prefix_matrix,
                    recv_rdma_channel_prefix_matrix,
                    recv_rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    recv_gbl_channel_prefix_matrix,
                    recv_gbl_rank_prefix_sum,
                    send_rdma_head,
                    send_nvl_head,
                    num_recv_tokens,
                    num_recv_rdma_tokens,
                )
        elif collapse_mode == _DEEPEP_INTERNODE_COLLAPSE_FFI and has_recv_topk_destination_layout:
            with jax.named_scope("deepep_collapse_local_assignments_ffi"):
                recv_out = deepep_collapse_local_assignments(
                    out_dispatch,
                    assignment_weights,
                    recv_token_indices,
                    assignment_destinations,
                    local_group_sizes,
                    num_recv_tokens,
                    recv_capacity=recv_x.shape[0],
                    internode=True,
                )
            with jax.named_scope("deepep_combine_internode_transport"):
                out_local = _deepep_combine_internode_output(
                    recv_out,
                    recv_topk_weights,
                    is_token_in_rank,
                    recv_src_meta,
                    rdma_channel_prefix_matrix,
                    recv_rdma_channel_prefix_matrix,
                    recv_rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    recv_gbl_channel_prefix_matrix,
                    recv_gbl_rank_prefix_sum,
                    send_rdma_head,
                    send_nvl_head,
                    num_recv_tokens,
                    num_recv_rdma_tokens,
                )
        else:
            fallback_scope = (
                "deepep_collapse_local_assignments_shape_fallback_jax"
                if collapse_mode
                in {
                    _DEEPEP_INTERNODE_COLLAPSE_FUSED_COMBINE,
                    _DEEPEP_INTERNODE_COLLAPSE_GATHER,
                    _DEEPEP_INTERNODE_COLLAPSE_FFI,
                }
                else "deepep_collapse_local_assignments_jax"
            )
            with jax.named_scope(fallback_scope):
                recv_out = _collapse_local_assignments_jax(
                    out_dispatch,
                    assignment_weights,
                    recv_token_indices,
                    local_group_sizes,
                    recv_capacity=recv_x.shape[0],
                )
            with jax.named_scope("deepep_combine_internode_transport"):
                out_local = _deepep_combine_internode_output(
                    recv_out,
                    recv_topk_weights,
                    is_token_in_rank,
                    recv_src_meta,
                    rdma_channel_prefix_matrix,
                    recv_rdma_channel_prefix_matrix,
                    recv_rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix,
                    recv_gbl_channel_prefix_matrix,
                    recv_gbl_rank_prefix_sum,
                    send_rdma_head,
                    send_nvl_head,
                    num_recv_tokens,
                    num_recv_rdma_tokens,
                )
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local.astype(x_local.dtype), dropped_total


def _moe_mlp_ep_deepep_composed_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    remat_mode: MoERematMode = "none",
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    """DeepEP dispatch/combine with capped assignment packing as a separate FFI."""
    return _moe_mlp_ep_deepep_local(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        remat_mode=remat_mode,
        composed_dispatch=True,
    )
