# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bulk-ring expert parallelism with Triton token-oriented routing."""

import math
import os
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug._moe.sonic import sonic_gather_sum
from levanter.grug.sharding import _batch_axes

try:
    import jax_triton as jt
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    jt = None
    triton = None
    tl = None


RingFusedImplementation = Literal["triton", "reference"]

_DEFAULT_TRITON_CACHE_DIR = "/tmp/marin-triton-cache"


if triton is not None and tl is not None:

    @triton.jit
    def _ring_combine_bwd_kernel(
        dout_global_ptr,  # (T, H)
        out_dispatch_ptr,  # (C, H)
        local_assignment_indices_ptr,  # (C,) int32
        valid_ptr,  # (C,) bool
        assignment_weights_ptr,  # (T * K,)
        dout_dispatch_ptr,  # (C, H)
        compact_weight_grads_ptr,  # (C,)
        hidden_dim: tl.constexpr,
        topk: tl.constexpr,
        block_h: tl.constexpr,
    ):
        compact_row = tl.program_id(axis=0)
        assignment = tl.load(local_assignment_indices_ptr + compact_row).to(tl.int64)
        accepted = tl.load(valid_ptr + compact_row)
        token = assignment // topk
        hidden = tl.arange(0, block_h).to(tl.int64)
        hidden_mask = hidden < hidden_dim
        mask = accepted & hidden_mask

        dout = tl.load(
            dout_global_ptr + token * hidden_dim + hidden,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dispatch_output = tl.load(
            out_dispatch_ptr + compact_row * hidden_dim + hidden,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(assignment_weights_ptr + assignment, mask=accepted, other=0.0).to(tl.float32)

        tl.store(
            dout_dispatch_ptr + compact_row * hidden_dim + hidden,
            dout * weight,
            mask=hidden_mask,
        )
        tl.store(compact_weight_grads_ptr + compact_row, tl.sum(dout * dispatch_output, axis=0))

else:
    _ring_combine_bwd_kernel = None


def _require_ring_fused_triton() -> None:
    if jt is None or _ring_combine_bwd_kernel is None:
        raise ImportError(
            "ring_fused requires jax-triton and triton; install the gpu extra for marin-levanter or marin"
        )
    if not os.environ.get("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = _DEFAULT_TRITON_CACHE_DIR


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _assignment_to_compact_rows(
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    *,
    tokens: int,
    topk: int,
) -> Int[Array, "T K"]:
    """Invert compact assignment ids into a token-major compact-row table."""
    assignments = tokens * topk
    compact_rows_with_sentinel = local_assignment_indices.shape[0]
    sentinel = jnp.array(compact_rows_with_sentinel - 1, dtype=jnp.int32)
    compact_rows = jnp.arange(compact_rows_with_sentinel, dtype=jnp.int32)
    dispatch_rows = jnp.full((assignments,), sentinel, dtype=jnp.int32)
    dispatch_rows = dispatch_rows.at[local_assignment_indices].set(
        jnp.where(valid, compact_rows, sentinel),
        mode="drop",
    )
    return dispatch_rows.reshape(tokens, topk)


def ring_dispatch_reference(
    x_global: Float[Array, "T H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    *,
    topk: int,
) -> Float[Array, "C H"]:
    """Gather compact routed inputs with ordinary JAX autodiff as an oracle."""
    token_indices = jnp.floor_divide(local_assignment_indices, topk)
    return jnp.where(valid[:, None], jnp.take(x_global, token_indices, axis=0), 0)


def _ring_dispatch_bwd_triton(
    dout_dispatch: Float[Array, "C H"],
    dispatch_rows: Int[Array, "T K"],
) -> Float[Array, "T H"]:
    sentinel = dout_dispatch.shape[0] - 1
    accepted = dispatch_rows < sentinel
    return sonic_gather_sum(
        dout_dispatch,
        dispatch_rows,
        accepted.astype(jnp.float32),
    )


@jax.custom_vjp
def ring_dispatch_triton(
    x_global: Float[Array, "T H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    dispatch_rows: Int[Array, "T K"],
) -> Float[Array, "C H"]:
    topk = dispatch_rows.shape[1]
    return ring_dispatch_reference(x_global, local_assignment_indices, valid, topk=topk)


def _ring_dispatch_triton_fwd(
    x_global: Float[Array, "T H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    dispatch_rows: Int[Array, "T K"],
) -> tuple[Float[Array, "C H"], Int[Array, "T K"]]:
    topk = dispatch_rows.shape[1]
    out = ring_dispatch_reference(x_global, local_assignment_indices, valid, topk=topk)
    return out, dispatch_rows


def _ring_dispatch_triton_bwd(
    dispatch_rows: Int[Array, "T K"],
    dout_dispatch: Float[Array, "C H"],
) -> tuple[Float[Array, "T H"], None, None, None]:
    return _ring_dispatch_bwd_triton(dout_dispatch, dispatch_rows), None, None, None


ring_dispatch_triton.defvjp(_ring_dispatch_triton_fwd, _ring_dispatch_triton_bwd)


def ring_combine_reference(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    *,
    tokens: int,
    topk: int,
) -> Float[Array, "T H"]:
    """Combine compact expert outputs with the ring backend's scatter oracle."""
    token_indices = jnp.floor_divide(local_assignment_indices, topk)
    weights = jnp.take(assignment_weights, local_assignment_indices, axis=0).astype(out_dispatch.dtype)
    contributions = jnp.where(valid[:, None], out_dispatch * weights[:, None], 0)
    return (
        jnp.zeros((tokens, out_dispatch.shape[1]), dtype=out_dispatch.dtype)
        .at[token_indices]
        .add(
            contributions,
            mode="drop",
        )
    )


def _ring_combine_fwd_triton(
    out_dispatch: Float[Array, "C H"],
    assignment_weights: Float[Array, "A"],
    dispatch_rows: Int[Array, "T K"],
) -> Float[Array, "T H"]:
    sentinel = out_dispatch.shape[0] - 1
    accepted = dispatch_rows < sentinel
    weights = assignment_weights.reshape(dispatch_rows.shape)
    masked_weights = jnp.where(accepted, weights, 0)
    return sonic_gather_sum(
        out_dispatch,
        dispatch_rows,
        masked_weights,
    )


def _ring_combine_bwd_triton(
    dout_global: Float[Array, "T H"],
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
) -> tuple[Float[Array, "C H"], Float[Array, "C"]]:
    _require_ring_fused_triton()
    capacity, hidden_dim = out_dispatch.shape
    topk = assignment_weights.shape[0] // dout_global.shape[0]
    block_h = _next_power_of_two(hidden_dim)
    num_warps = 8 if block_h >= 1024 else 4
    dout_dispatch_shape = jax.ShapeDtypeStruct(out_dispatch.shape, out_dispatch.dtype)
    compact_weight_grads_shape = jax.ShapeDtypeStruct((capacity,), jnp.float32)
    return jt.triton_call(
        dout_global,
        out_dispatch,
        local_assignment_indices,
        valid,
        assignment_weights,
        kernel=_ring_combine_bwd_kernel,
        out_shape=(dout_dispatch_shape, compact_weight_grads_shape),
        grid=(capacity,),
        num_warps=num_warps,
        num_stages=2,
        hidden_dim=hidden_dim,
        topk=topk,
        block_h=block_h,
    )


@jax.custom_vjp
def _ring_combine_triton(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    dispatch_rows: Int[Array, "T K"],
) -> Float[Array, "T H"]:
    return _ring_combine_fwd_triton(out_dispatch, assignment_weights, dispatch_rows)


def _ring_combine_triton_fwd(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    dispatch_rows: Int[Array, "T K"],
) -> tuple[Float[Array, "T H"], tuple[jax.Array, ...]]:
    out = _ring_combine_fwd_triton(out_dispatch, assignment_weights, dispatch_rows)
    return out, (out_dispatch, local_assignment_indices, valid, assignment_weights)


def _ring_combine_triton_bwd(
    residuals: tuple[jax.Array, ...],
    dout_global: Float[Array, "T H"],
) -> tuple[Float[Array, "C H"], None, None, Float[Array, "A"], None]:
    out_dispatch, local_assignment_indices, valid, assignment_weights = residuals
    dout_dispatch, compact_weight_grads = _ring_combine_bwd_triton(
        dout_global,
        out_dispatch,
        local_assignment_indices,
        valid,
        assignment_weights,
    )
    assignments = assignment_weights.shape[0]
    sentinel = jnp.array(assignments, dtype=jnp.int32)
    safe_assignment_indices = jnp.where(valid, local_assignment_indices, sentinel)
    d_assignment_weights = (
        jnp.zeros((assignments + 1,), dtype=assignment_weights.dtype)
        .at[safe_assignment_indices]
        .set(compact_weight_grads.astype(assignment_weights.dtype), mode="drop")[:-1]
    )
    return dout_dispatch, None, None, d_assignment_weights, None


_ring_combine_triton.defvjp(_ring_combine_triton_fwd, _ring_combine_triton_bwd)


def ring_combine_triton(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    *,
    tokens: int,
    topk: int,
) -> Float[Array, "T H"]:
    """Combine compact outputs with direct-output Triton forward/backward kernels."""
    dispatch_rows = _assignment_to_compact_rows(
        local_assignment_indices,
        valid,
        tokens=tokens,
        topk=topk,
    )
    return _ring_combine_triton(
        out_dispatch,
        local_assignment_indices,
        valid,
        assignment_weights,
        dispatch_rows,
    )


def ring_combine(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    *,
    tokens: int,
    topk: int,
    implementation: RingFusedImplementation,
) -> Float[Array, "T H"]:
    """Combine compact ring outputs using Triton or the JAX oracle."""
    if implementation == "triton":
        return ring_combine_triton(
            out_dispatch,
            local_assignment_indices,
            valid,
            assignment_weights,
            tokens=tokens,
            topk=topk,
        )
    if implementation == "reference":
        return ring_combine_reference(
            out_dispatch,
            local_assignment_indices,
            valid,
            assignment_weights,
            tokens=tokens,
            topk=topk,
        )
    raise ValueError(f"Unknown ring_fused implementation {implementation!r}")


def _moe_mlp_ep_ring_fused_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    routing_implementation: RingFusedImplementation = "triton",
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Run bulk-ring MoE with direct-output Triton routing kernels."""
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

        local_experts = moe_w13_local.shape[0]
        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )

        ep_size = num_experts // local_experts
        local_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments / ep_size)))
        compact_capacity = local_capacity + 1

        expert_axis = jax.lax.axis_index("expert")
        local_expert = expert_flat - expert_axis * local_experts
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)
        local_expert = jnp.where(local_mask, local_expert, 0)

        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        counts = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask[:, None].astype(jnp.int32),
            axis=0,
            dtype=jnp.int32,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
        valid = jnp.arange(compact_capacity, dtype=jnp.int32) < accepted_total

        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
        _, local_idx = jax.lax.top_k(selection_key, compact_capacity)

        dispatch_rows = _assignment_to_compact_rows(
            local_idx,
            valid,
            tokens=tokens,
            topk=topk,
        )
        if routing_implementation == "triton":
            x_dispatch = ring_dispatch_triton(x_global, local_idx, valid, dispatch_rows)
        elif routing_implementation == "reference":
            x_dispatch = ring_dispatch_reference(x_global, local_idx, valid, topk=topk)
        else:
            raise ValueError(f"Unknown ring_fused implementation {routing_implementation!r}")
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    group_sizes = accepted_counts.at[-1].add(compact_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        intermediate_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [intermediate_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        if routing_implementation == "triton":
            out_global = _ring_combine_triton(out_dispatch, local_idx, valid, weight_flat, dispatch_rows)
        else:
            out_global = ring_combine_reference(
                out_dispatch,
                local_idx,
                valid,
                weight_flat,
                tokens=tokens,
                topk=topk,
            )
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
