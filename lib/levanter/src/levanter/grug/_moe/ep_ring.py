# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable
from functools import cache
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import (
    ragged_dot,
    ragged_dot_accumulating_weight_gradient,
    ragged_dot_accumulating_weight_gradient_backward,
)
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    MoeRaggedDotOps,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts, _resolve_ragged_dot_fns
from levanter.grug._moe.sonic_quack import quack_mlp_varlen
from levanter.grug.sharding import _batch_axes


_RING_FP8_WIRE_DTYPE = jnp.float8_e4m3fn


class _RingRouting(NamedTuple):
    assignment_indices: Int[Array, "C"]
    valid: Bool[Array, "C"]
    accepted_counts: Int[Array, "Elocal"]
    local_expert: Int[Array, "A"]
    dropped_local: Int[Array, ""]
    local_capacity: int
    tokens_per_shard: int
    topk: int
    expert_axis_size: int


class _BulkRingDispatchState(NamedTuple):
    """Expert-sharded state between bulk-ring dispatch and expert compute."""

    x_dispatch: Float[Array, "C H"]
    weight_dispatch: Float[Array, "C"]
    token_global: Int[Array, "C"]
    group_sizes: Int[Array, "Elocal"]
    dropped_local: Int[Array, "One"]


class _BulkRingExpertState(NamedTuple):
    """Expert-sharded state between bulk-ring expert compute and combine."""

    out_dispatch: Float[Array, "C H"]


def _validate_fp8_wire_contract(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
) -> None:
    if x_local.dtype != jnp.bfloat16 or moe_w13_local.dtype != jnp.bfloat16 or moe_w2_local.dtype != jnp.bfloat16:
        raise TypeError("approximate FP8 ring wire requires bfloat16 activations and expert weights")
    if combine_weights_local.dtype != jnp.float32:
        raise TypeError("approximate FP8 ring wire requires float32 combine weights")
    if not jnp.issubdtype(selected_experts_local.dtype, jnp.integer):
        raise TypeError("approximate FP8 ring wire requires integer selected experts")
    if x_local.ndim != 2 or selected_experts_local.ndim != 2 or combine_weights_local.ndim != 2:
        raise ValueError("approximate FP8 ring wire inputs must have shapes [T,H], [T,K], and [T,K]")
    if moe_w13_local.ndim != 3 or moe_w2_local.ndim != 3:
        raise ValueError("approximate FP8 ring wire weights must have shapes [Elocal,H,2I] and [Elocal,I,H]")
    if selected_experts_local.shape != combine_weights_local.shape:
        raise ValueError("approximate FP8 ring wire selected experts and combine weights must have the same shape")
    if x_local.shape[0] != selected_experts_local.shape[0]:
        raise ValueError("approximate FP8 ring wire routing must have one row per activation token")
    if x_local.shape[1] == 0 or selected_experts_local.shape[1] == 0:
        raise ValueError("approximate FP8 ring wire hidden and top-k dimensions must be non-empty")
    if moe_w13_local.shape[0] != moe_w2_local.shape[0]:
        raise ValueError("approximate FP8 ring wire W13 and W2 must have the same local expert count")
    if moe_w13_local.shape[1] != x_local.shape[1] or moe_w2_local.shape[2] != x_local.shape[1]:
        raise ValueError("approximate FP8 ring wire W13/W2 hidden dimensions must match the activations")
    if moe_w13_local.shape[2] != 2 * moe_w2_local.shape[1]:
        raise ValueError("approximate FP8 ring wire W13 output must be twice the W2 intermediate dimension")


def _validate_accumulating_weight_gradient_contract(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    w13_accumulator_local: jax.Array,
    w2_accumulator_local: jax.Array,
) -> None:
    if x_local.dtype != jnp.bfloat16 or moe_w13_local.dtype != jnp.bfloat16 or moe_w2_local.dtype != jnp.bfloat16:
        raise TypeError("accumulating Ring requires bfloat16 activations and expert compute weights")
    if w13_accumulator_local.dtype != jnp.float32 or w2_accumulator_local.dtype != jnp.float32:
        raise TypeError("accumulating Ring requires float32 expert-gradient accumulators")
    if not jnp.issubdtype(selected_experts_local.dtype, jnp.integer):
        raise TypeError("accumulating Ring requires integer selected experts")
    if not jnp.issubdtype(combine_weights_local.dtype, jnp.floating):
        raise TypeError("accumulating Ring requires floating-point combine weights")
    if x_local.ndim != 2 or selected_experts_local.ndim != 2 or combine_weights_local.ndim != 2:
        raise ValueError("accumulating Ring inputs must have shapes [T,H], [T,K], and [T,K]")
    if moe_w13_local.ndim != 3 or moe_w2_local.ndim != 3:
        raise ValueError("accumulating Ring weights must have shapes [Elocal,H,2I] and [Elocal,I,H]")
    if selected_experts_local.shape != combine_weights_local.shape:
        raise ValueError("accumulating Ring selected experts and combine weights must have the same shape")
    if x_local.shape[0] != selected_experts_local.shape[0]:
        raise ValueError("accumulating Ring routing must have one row per activation token")
    if moe_w13_local.shape[0] != moe_w2_local.shape[0]:
        raise ValueError("accumulating Ring W13 and W2 must have the same local expert count")
    if moe_w13_local.shape[1] != x_local.shape[1] or moe_w2_local.shape[2] != x_local.shape[1]:
        raise ValueError("accumulating Ring W13/W2 hidden dimensions must match the activations")
    if moe_w13_local.shape[2] != 2 * moe_w2_local.shape[1]:
        raise ValueError("accumulating Ring W13 output must be twice the W2 intermediate dimension")
    if w13_accumulator_local.shape != moe_w13_local.shape:
        raise ValueError(
            "accumulating Ring W13 accumulator must match the compute weight shape; "
            f"got {w13_accumulator_local.shape} and {moe_w13_local.shape}"
        )
    if w2_accumulator_local.shape != moe_w2_local.shape:
        raise ValueError(
            "accumulating Ring W2 accumulator must match the compute weight shape; "
            f"got {w2_accumulator_local.shape} and {moe_w2_local.shape}"
        )


def _quantize_fp8_wire_per_token(
    value: Float[Array, "T H"],
    *,
    reduction_terms: int = 1,
) -> tuple[Array, Float[Array, "T"]]:
    if reduction_terms <= 0:
        raise ValueError(f"reduction_terms must be positive, got {reduction_terms}")
    value_f32 = value.astype(jnp.float32)
    amax = jnp.max(jnp.abs(value_f32), axis=-1)
    fp8_max = jnp.asarray(jnp.finfo(_RING_FP8_WIRE_DTYPE).max, dtype=jnp.float32)
    dequant_scale = jnp.where(amax > 0, amax * reduction_terms / fp8_max, jnp.ones_like(amax))
    dequant_scale = jax.lax.stop_gradient(dequant_scale)
    quantized = (value_f32 / dequant_scale[:, None]).astype(_RING_FP8_WIRE_DTYPE)
    return quantized, dequant_scale


def _dequantize_fp8_wire_per_token(
    quantized: Array,
    dequant_scale: Float[Array, "T"],
) -> Float[Array, "T H"]:
    return (quantized.astype(jnp.float32) * dequant_scale[:, None]).astype(jnp.bfloat16)


def _ring_routing_prepass(
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
) -> _RingRouting:
    """Select the exact source-major assignment prefix used by bulk ring."""
    selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
    tokens = selected_experts_global.shape[0]
    tokens_per_shard, topk = selected_experts_local.shape
    assignments = tokens * topk
    expert_flat = selected_experts_global.reshape(assignments)

    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    ep_size = num_experts // local_experts
    local_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments / ep_size)))
    expert_start = jax.lax.axis_index("expert") * local_experts
    local_expert: jax.Array = expert_flat - expert_start
    local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)
    local_expert = jnp.where(local_mask, local_expert, 0)

    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    local_mask_i32 = local_mask.astype(jnp.int32)
    counts = jnp.sum(
        (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
        axis=0,
        dtype=jnp.int32,
    )
    accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    # Select by (local expert, source-major assignment position), exactly as
    # the original bulk implementation. The resulting accepted prefix remains
    # grouped by expert and source-major within each group.
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, assignment_indices = jax.lax.top_k(selection_key, local_capacity)

    return _RingRouting(
        assignment_indices=assignment_indices,
        valid=valid,
        accepted_counts=accepted_counts,
        local_expert=local_expert,
        dropped_local=jnp.sum(counts, dtype=jnp.int32) - accepted_total,
        local_capacity=local_capacity,
        tokens_per_shard=tokens_per_shard,
        topk=topk,
        expert_axis_size=ep_size,
    )


def _group_sizes_with_padding(accepted_counts: Int[Array, "Elocal"], capacity: int) -> Int[Array, "Elocal"]:
    return accepted_counts.at[-1].add(capacity - jnp.sum(accepted_counts, dtype=jnp.int32))


def _bulk_ring_dispatch_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    routing: _RingRouting,
    *,
    combine_dtype: Literal["bf16", "fp32"] = "bf16",
) -> _BulkRingDispatchState:
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        weight_flat = combine_weights_global.reshape(-1)
        token_global = jnp.floor_divide(routing.assignment_indices, routing.topk)
        if combine_dtype == "bf16":
            weight = jnp.take(weight_flat, routing.assignment_indices, axis=0).astype(x_local.dtype)
        elif combine_dtype == "fp32":
            weight = jnp.take(weight_flat, routing.assignment_indices, axis=0).astype(jnp.float32)
        else:
            raise ValueError(f"unknown bulk-ring combine dtype: {combine_dtype!r}")
        x_take = jnp.take(x_global, token_global, axis=0)
        x_dispatch = jnp.where(routing.valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(routing.valid, weight, jnp.zeros_like(weight))

    group_sizes = _group_sizes_with_padding(routing.accepted_counts, routing.local_capacity)
    return _BulkRingDispatchState(
        x_dispatch=x_dispatch,
        weight_dispatch=weight_dispatch,
        token_global=token_global,
        group_sizes=group_sizes,
        dropped_local=routing.dropped_local[None],
    )


def _bulk_ring_fp8_wire_dispatch_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    routing: _RingRouting,
) -> _BulkRingDispatchState:
    with jax.named_scope("gather"):
        x_quantized_local, x_scale_local = _quantize_fp8_wire_per_token(x_local)
        x_quantized_global = jax.lax.all_gather(x_quantized_local, "expert", tiled=True)
        x_scale_global = jax.lax.all_gather(x_scale_local, "expert", tiled=True)
        x_global = _dequantize_fp8_wire_per_token(x_quantized_global, x_scale_global)

        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        weight_flat = combine_weights_global.reshape(-1)
        token_global = jnp.floor_divide(routing.assignment_indices, routing.topk)
        weight = jnp.take(weight_flat, routing.assignment_indices, axis=0).astype(x_local.dtype)
        x_take = jnp.take(x_global, token_global, axis=0)
        x_dispatch = jnp.where(routing.valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(routing.valid, weight, jnp.zeros_like(weight))

    group_sizes = _group_sizes_with_padding(routing.accepted_counts, routing.local_capacity)
    return _BulkRingDispatchState(
        x_dispatch=x_dispatch,
        weight_dispatch=weight_dispatch,
        token_global=token_global,
        group_sizes=group_sizes,
        dropped_local=routing.dropped_local[None],
    )


def _bulk_ring_expert_compute(
    dispatch: _BulkRingDispatchState,
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    ops: MoeRaggedDotOps | None = None,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> _BulkRingExpertState:
    ragged_w13, ragged_w2 = _resolve_ragged_dot_fns(ops)
    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(
            ragged_w13(dispatch.x_dispatch, moe_w13_local, dispatch.group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_w2(activation_fn(gate) * up, moe_w2_local, dispatch.group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )
    return _BulkRingExpertState(out_dispatch=out_dispatch)


def _bulk_ring_expert_compute_accumulating_weight_gradient(
    dispatch: _BulkRingDispatchState,
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    w13_accumulator_local: Float[Array, "Elocal H I2"],
    w2_accumulator_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> tuple[_BulkRingExpertState, Float[Array, ""]]:
    with jax.named_scope("moe_up_down"):
        w13_out, w13_token = ragged_dot_accumulating_weight_gradient(
            dispatch.x_dispatch,
            moe_w13_local,
            dispatch.group_sizes,
            w13_accumulator_local,
        )
        w13_out = tree_checkpoint_name(w13_out, _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch, w2_token = ragged_dot_accumulating_weight_gradient(
            activation_fn(gate) * up,
            moe_w2_local,
            dispatch.group_sizes,
            w2_accumulator_local,
        )
        out_dispatch = tree_checkpoint_name(out_dispatch, _CHECKPOINT_DISPATCH_OUTPUT)
    return _BulkRingExpertState(out_dispatch=out_dispatch), w13_token + w2_token


def _bulk_ring_combine(
    dispatch: _BulkRingDispatchState,
    expert: _BulkRingExpertState,
    *,
    tokens_per_shard: int,
    expert_axis_size: int,
    combine_dtype: Literal["bf16", "fp32"] = "bf16",
) -> Float[Array, "Tlocal H"]:
    with jax.named_scope("scatter"):
        output_shape = (tokens_per_shard * expert_axis_size, expert.out_dispatch.shape[-1])
        if combine_dtype == "bf16":
            out_global = (
                jnp.zeros(output_shape, dtype=dispatch.x_dispatch.dtype)
                .at[dispatch.token_global]
                .add(expert.out_dispatch * dispatch.weight_dispatch[:, None], mode="drop")
            )
        elif combine_dtype == "fp32":
            weighted = expert.out_dispatch.astype(jnp.float32) * dispatch.weight_dispatch[:, None]
            out_global = (
                jnp.zeros(output_shape, dtype=jnp.float32).at[dispatch.token_global].add(weighted, mode="drop")
            )
        else:
            raise ValueError(f"unknown bulk-ring combine dtype: {combine_dtype!r}")
        return jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True).astype(
            dispatch.x_dispatch.dtype
        )


def _bulk_ring_fp8_wire_combine(
    dispatch: _BulkRingDispatchState,
    expert: _BulkRingExpertState,
    *,
    tokens_per_shard: int,
    expert_axis_size: int,
    topk: int,
) -> Float[Array, "Tlocal H"]:
    with jax.named_scope("scatter"):
        output_shape = (tokens_per_shard * expert_axis_size, expert.out_dispatch.shape[-1])
        out_global = (
            jnp.zeros(output_shape, dtype=dispatch.x_dispatch.dtype)
            .at[dispatch.token_global]
            .add(expert.out_dispatch * dispatch.weight_dispatch[:, None], mode="drop")
        )
        local_amax = jnp.max(jnp.abs(out_global.astype(jnp.float32)), axis=-1)
        local_amax = jax.lax.stop_gradient(local_amax)
        shared_amax = jax.lax.pmax(local_amax, "expert")
        max_contributing_ranks = min(topk, expert_axis_size)
        fp8_max = jnp.asarray(jnp.finfo(_RING_FP8_WIRE_DTYPE).max, dtype=jnp.float32)
        shared_scale = jnp.where(
            shared_amax > 0,
            shared_amax * max_contributing_ranks / fp8_max,
            jnp.ones_like(shared_amax),
        )
        out_quantized = (out_global.astype(jnp.float32) / shared_scale[:, None]).astype(_RING_FP8_WIRE_DTYPE)
        out_quantized_local = jax.lax.psum_scatter(
            out_quantized,
            "expert",
            scatter_dimension=0,
            tiled=True,
        )
        scale_start = jax.lax.axis_index("expert") * tokens_per_shard
        scale_local = jax.lax.dynamic_slice_in_dim(shared_scale, scale_start, tokens_per_shard)
        return _dequantize_fp8_wire_per_token(out_quantized_local, scale_local)


def _bulk_ring_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    ops: MoeRaggedDotOps | None = None,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    combine_dtype: Literal["bf16", "fp32"] = "bf16",
) -> Float[Array, "Tlocal H"]:
    dispatch = _bulk_ring_dispatch_from_routing(
        x_local,
        combine_weights_local,
        routing,
        combine_dtype=combine_dtype,
    )
    expert = _bulk_ring_expert_compute(
        dispatch,
        moe_w13_local,
        moe_w2_local,
        ops,
        activation_fn=activation_fn,
    )
    return _bulk_ring_combine(
        dispatch,
        expert,
        tokens_per_shard=routing.tokens_per_shard,
        expert_axis_size=routing.expert_axis_size,
        combine_dtype=combine_dtype,
    )


def _bulk_ring_from_routing_accumulating_weight_gradient(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    w13_accumulator_local: Float[Array, "Elocal H I2"],
    w2_accumulator_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> tuple[Float[Array, "Tlocal H"], Float[Array, ""]]:
    dispatch = _bulk_ring_dispatch_from_routing(
        x_local,
        combine_weights_local,
        routing,
    )
    expert, accumulation_token = _bulk_ring_expert_compute_accumulating_weight_gradient(
        dispatch,
        moe_w13_local,
        moe_w2_local,
        w13_accumulator_local,
        w2_accumulator_local,
        activation_fn=activation_fn,
    )
    output = _bulk_ring_combine(
        dispatch,
        expert,
        tokens_per_shard=routing.tokens_per_shard,
        expert_axis_size=routing.expert_axis_size,
    )
    return output, accumulation_token


def _accumulating_weight_cotangent(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    accumulator: jax.Array,
    output_cotangent: jax.Array,
    accumulation_scale: jax.Array,
) -> jax.Array:
    """Keep the FP32 weight cotangent custom call as a late-inline compiler unit."""
    _, weight_cotangent = ragged_dot_accumulating_weight_gradient_backward(
        lhs,
        rhs,
        group_sizes,
        accumulator,
        output_cotangent,
        accumulation_scale,
    )
    return weight_cotangent


@cache
def _outlined_accumulating_weight_cotangent_fn() -> Callable[..., jax.Array]:
    # The Iris submission process imports this module before the worker upgrades JAX.
    return jax.jit(_accumulating_weight_cotangent, inline=jax.Inline.XLA_LATE)


def _outlined_accumulating_weight_cotangent(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    accumulator: jax.Array,
    output_cotangent: jax.Array,
    accumulation_scale: jax.Array,
) -> jax.Array:
    return _outlined_accumulating_weight_cotangent_fn()(
        lhs,
        rhs,
        group_sizes,
        accumulator,
        output_cotangent,
        accumulation_scale,
    )


def _bulk_ring_from_routing_accumulating_weight_gradient_backward(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    w13_accumulator_local: Float[Array, "Elocal H I2"],
    w2_accumulator_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    output_cotangent_local: Float[Array, "Tlocal H"],
    accumulation_scale: Float[Array, ""],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> tuple[
    Float[Array, "Tlocal H"],
    Float[Array, "Tlocal K"],
    Float[Array, "Elocal H I2"],
    Float[Array, "Elocal I H"],
]:
    """Apply the explicit local pullback for accumulating bulk Ring."""
    dispatch = _bulk_ring_dispatch_from_routing(
        x_local,
        combine_weights_local,
        routing,
    )

    w13_out, _ = ragged_dot_accumulating_weight_gradient(
        dispatch.x_dispatch,
        moe_w13_local,
        dispatch.group_sizes,
        w13_accumulator_local,
    )
    moe_dim = moe_w2_local.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    activated_gate, activation_pullback = jax.vjp(activation_fn, gate)
    expert_hidden = activated_gate * up
    out_dispatch, _ = ragged_dot_accumulating_weight_gradient(
        expert_hidden,
        moe_w2_local,
        dispatch.group_sizes,
        w2_accumulator_local,
    )

    output_cotangent_global = jax.lax.all_gather(output_cotangent_local, "expert", tiled=True)
    weighted_output_cotangent = jnp.take(output_cotangent_global, dispatch.token_global, axis=0)
    out_dispatch_cotangent = weighted_output_cotangent * dispatch.weight_dispatch[:, None]
    weight_dispatch_cotangent = jnp.sum(weighted_output_cotangent * out_dispatch, axis=-1)

    expert_hidden_cotangent, _ = ragged_dot_accumulating_weight_gradient_backward(
        expert_hidden,
        moe_w2_local,
        dispatch.group_sizes,
        w2_accumulator_local,
        out_dispatch_cotangent,
        accumulation_scale,
    )
    w2_cotangent = _outlined_accumulating_weight_cotangent(
        expert_hidden,
        moe_w2_local,
        dispatch.group_sizes,
        w2_accumulator_local,
        out_dispatch_cotangent,
        accumulation_scale,
    )
    gate_cotangent = activation_pullback(expert_hidden_cotangent * up)[0]
    up_cotangent = expert_hidden_cotangent * activated_gate
    w13_out_cotangent = jnp.concatenate((gate_cotangent, up_cotangent), axis=-1)
    x_dispatch_cotangent, _ = ragged_dot_accumulating_weight_gradient_backward(
        dispatch.x_dispatch,
        moe_w13_local,
        dispatch.group_sizes,
        w13_accumulator_local,
        w13_out_cotangent,
        accumulation_scale,
    )
    w13_cotangent = _outlined_accumulating_weight_cotangent(
        dispatch.x_dispatch,
        moe_w13_local,
        dispatch.group_sizes,
        w13_accumulator_local,
        w13_out_cotangent,
        accumulation_scale,
    )

    valid = routing.valid
    x_take_cotangent = jnp.where(valid[:, None], x_dispatch_cotangent, jnp.zeros_like(x_dispatch_cotangent))
    x_global_shape = (routing.tokens_per_shard * routing.expert_axis_size, x_local.shape[1])
    x_global_cotangent = (
        jnp.zeros(x_global_shape, dtype=x_local.dtype).at[dispatch.token_global].add(x_take_cotangent, mode="drop")
    )
    x_cotangent = jax.lax.psum_scatter(x_global_cotangent, "expert", scatter_dimension=0, tiled=True)

    weight_cotangent = jnp.where(valid, weight_dispatch_cotangent, jnp.zeros_like(weight_dispatch_cotangent))
    combine_weights_global_shape = (
        routing.tokens_per_shard * routing.expert_axis_size,
        routing.topk,
    )
    combine_weights_global_cotangent = (
        jnp.zeros(math.prod(combine_weights_global_shape), dtype=combine_weights_local.dtype)
        .at[routing.assignment_indices]
        .add(weight_cotangent.astype(combine_weights_local.dtype), mode="drop")
        .reshape(combine_weights_global_shape)
    )
    combine_weights_cotangent = jax.lax.psum_scatter(
        combine_weights_global_cotangent,
        "expert",
        scatter_dimension=0,
        tiled=True,
    )
    return x_cotangent, combine_weights_cotangent, w13_cotangent, w2_cotangent


def _bulk_ring_fp8_wire_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "Tlocal H"]:
    dispatch = _bulk_ring_fp8_wire_dispatch_from_routing(x_local, combine_weights_local, routing)
    expert = _bulk_ring_expert_compute(
        dispatch,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
    )
    return _bulk_ring_fp8_wire_combine(
        dispatch,
        expert,
        tokens_per_shard=routing.tokens_per_shard,
        expert_axis_size=routing.expert_axis_size,
        topk=routing.topk,
    )


def _validate_quack_bulk_ring_contract(
    x_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> None:
    if activation_fn is not jax.nn.silu:
        raise ValueError("EP-local QuACK bulk ring only supports SiLU/SwiGLU activation")
    if x_local.dtype != jnp.bfloat16 or moe_w13_local.dtype != jnp.bfloat16 or moe_w2_local.dtype != jnp.bfloat16:
        raise TypeError("EP-local QuACK bulk ring requires bfloat16 activations and weights")
    if x_local.ndim != 2 or moe_w13_local.ndim != 3 or moe_w2_local.ndim != 3:
        raise ValueError("EP-local QuACK inputs must have shapes [C,H], [Elocal,H,2I], and [Elocal,I,H]")
    if moe_w13_local.shape[0] != moe_w2_local.shape[0]:
        raise ValueError("EP-local QuACK W13 and W2 must have the same local expert count")
    if moe_w13_local.shape[1] != x_local.shape[1] or moe_w2_local.shape[2] != x_local.shape[1]:
        raise ValueError("EP-local QuACK W13/W2 hidden dimensions must match the activation hidden dimension")
    if moe_w13_local.shape[2] != 2 * moe_w2_local.shape[1]:
        raise ValueError("EP-local QuACK W13 output must be twice the W2 intermediate dimension")


def _bulk_ring_quack_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "Tlocal H"]:
    _validate_quack_bulk_ring_contract(x_local, moe_w13_local, moe_w2_local, activation_fn=activation_fn)

    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        weight_flat = combine_weights_global.reshape(-1)
        token_global = jnp.floor_divide(routing.assignment_indices, routing.topk)
        weight = jnp.take(weight_flat, routing.assignment_indices, axis=0).astype(x_local.dtype)
        x_take = jnp.take(x_global, token_global, axis=0)
        x_dispatch = jnp.where(routing.valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(routing.valid, weight, jnp.zeros_like(weight))

    group_sizes = _group_sizes_with_padding(routing.accepted_counts, routing.local_capacity)
    with jax.named_scope("moe_up_down"):
        out_dispatch = tree_checkpoint_name(
            quack_mlp_varlen(x_dispatch, moe_w13_local, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        out_global = (
            jnp.zeros_like(x_global).at[token_global].add(out_dispatch * weight_dispatch[:, None], mode="drop")
        )
        return jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)


def _chunk_routing(
    routing: _RingRouting,
    *,
    chunk_start: int,
    chunk_tokens: int,
    chunk_capacity: int,
) -> tuple[Int[Array, "Cchunk"], Bool[Array, "Cchunk"], Int[Array, "Elocal"], Int[Array, "Cchunk"]]:
    token_global = jnp.floor_divide(routing.assignment_indices, routing.topk)
    source_token = token_global % routing.tokens_per_shard
    in_chunk = jnp.logical_and(source_token >= chunk_start, source_token < chunk_start + chunk_tokens)
    keep = jnp.logical_and(routing.valid, in_chunk)

    # Filtering the already expert-grouped accepted list preserves its order
    # without another sort. The fixed size keeps both branch shapes static.
    selected_positions = jnp.nonzero(keep, size=chunk_capacity, fill_value=0)[0]
    assignment_indices = jnp.take(routing.assignment_indices, selected_positions, axis=0)
    selected_expert = jnp.take(routing.local_expert, assignment_indices, axis=0)
    chunk_total = jnp.sum(keep.astype(jnp.int32), dtype=jnp.int32)
    chunk_valid = jnp.arange(chunk_capacity, dtype=jnp.int32) < chunk_total

    expert_ids = jnp.arange(routing.accepted_counts.shape[0], dtype=jnp.int32)
    chunk_counts = jnp.sum(
        (selected_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * chunk_valid[:, None].astype(jnp.int32),
        axis=0,
        dtype=jnp.int32,
    )

    selected_token_global = jnp.floor_divide(assignment_indices, routing.topk)
    selected_source_shard = jnp.floor_divide(selected_token_global, routing.tokens_per_shard)
    selected_source_token = selected_token_global % routing.tokens_per_shard
    chunk_token_indices = selected_source_shard * chunk_tokens + selected_source_token - chunk_start
    chunk_token_indices = jnp.where(chunk_valid, chunk_token_indices, 0)
    return assignment_indices, chunk_valid, chunk_counts, chunk_token_indices


def _two_chunk_fast_path_decision(routing: _RingRouting) -> Bool[Array, ""]:
    first_tokens = (routing.tokens_per_shard + 1) // 2
    first_capacity = (routing.local_capacity + 1) // 2
    first_source_token = jnp.floor_divide(routing.assignment_indices, routing.topk) % routing.tokens_per_shard
    first_count = jnp.sum(
        jnp.logical_and(routing.valid, first_source_token < first_tokens).astype(jnp.int32), dtype=jnp.int32
    )
    second_count = jnp.sum(routing.valid.astype(jnp.int32), dtype=jnp.int32) - first_count
    local_fast = jnp.logical_and(
        first_count <= first_capacity, second_count <= routing.local_capacity - first_capacity
    )
    return jax.lax.pmin(local_fast.astype(jnp.int32), "expert") != 0


def _two_chunk_ring_from_routing(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _RingRouting,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "Tlocal H"]:
    first_tokens = (routing.tokens_per_shard + 1) // 2
    second_tokens = routing.tokens_per_shard - first_tokens
    first_capacity = (routing.local_capacity + 1) // 2
    second_capacity = routing.local_capacity - first_capacity

    first_indices, first_valid, first_counts, first_token_indices = _chunk_routing(
        routing, chunk_start=0, chunk_tokens=first_tokens, chunk_capacity=first_capacity
    )
    second_indices, second_valid, second_counts, second_token_indices = _chunk_routing(
        routing, chunk_start=first_tokens, chunk_tokens=second_tokens, chunk_capacity=second_capacity
    )

    # Keep both gathers ahead of the first expert compute. This gives the
    # latency-hiding scheduler an AG1 -> compute0-independent edge.
    with jax.named_scope("gather_chunk_0"):
        x_global_0 = jax.lax.all_gather(x_local[:first_tokens], "expert", tiled=True)
        weights_global_0 = jax.lax.all_gather(combine_weights_local[:first_tokens], "expert", tiled=True)
        weights_flat_0 = weights_global_0.reshape(-1)
        weight_index_0 = (first_token_indices * routing.topk) + (first_indices % routing.topk)
        x_take_0 = jnp.take(x_global_0, first_token_indices, axis=0)
        x_dispatch_0 = jnp.where(first_valid[:, None], x_take_0, jnp.zeros_like(x_take_0))
        weight_0 = jnp.take(weights_flat_0, weight_index_0, axis=0).astype(x_local.dtype)
        weight_dispatch_0 = jnp.where(first_valid, weight_0, jnp.zeros_like(weight_0))

    with jax.named_scope("gather_chunk_1"):
        x_global_1 = jax.lax.all_gather(x_local[first_tokens:], "expert", tiled=True)
        weights_global_1 = jax.lax.all_gather(combine_weights_local[first_tokens:], "expert", tiled=True)
        weights_flat_1 = weights_global_1.reshape(-1)
        weight_index_1 = (second_token_indices * routing.topk) + (second_indices % routing.topk)
        x_take_1 = jnp.take(x_global_1, second_token_indices, axis=0)
        x_dispatch_1 = jnp.where(
            second_valid[:, None],
            x_take_1,
            jnp.zeros_like(x_take_1),
        )
        weight_1 = jnp.take(weights_flat_1, weight_index_1, axis=0).astype(x_local.dtype)
        weight_dispatch_1 = jnp.where(second_valid, weight_1, jnp.zeros_like(weight_1))

    x_dispatch_0 = tree_checkpoint_name(x_dispatch_0, _CHECKPOINT_DISPATCH_INPUT)
    group_sizes_0 = _group_sizes_with_padding(first_counts, first_capacity)
    with jax.named_scope("moe_up_down_chunk_0"):
        w13_out_0 = tree_checkpoint_name(
            ragged_dot(x_dispatch_0, moe_w13_local, group_sizes_0), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate_0, up_0 = jnp.split(w13_out_0, [moe_dim], axis=-1)
        out_dispatch_0 = tree_checkpoint_name(
            ragged_dot(activation_fn(gate_0) * up_0, moe_w2_local, group_sizes_0),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )
    with jax.named_scope("scatter_chunk_0"):
        out_global_0 = (
            jnp.zeros_like(x_global_0)
            .at[first_token_indices]
            .add(out_dispatch_0 * weight_dispatch_0[:, None], mode="drop")
        )
        out_local_0 = jax.lax.psum_scatter(out_global_0, "expert", scatter_dimension=0, tiled=True)

    x_dispatch_1 = tree_checkpoint_name(x_dispatch_1, _CHECKPOINT_DISPATCH_INPUT)
    group_sizes_1 = _group_sizes_with_padding(second_counts, second_capacity)
    with jax.named_scope("moe_up_down_chunk_1"):
        w13_out_1 = tree_checkpoint_name(
            ragged_dot(x_dispatch_1, moe_w13_local, group_sizes_1), _CHECKPOINT_EXPERT_HIDDEN
        )
        gate_1, up_1 = jnp.split(w13_out_1, [moe_dim], axis=-1)
        out_dispatch_1 = tree_checkpoint_name(
            ragged_dot(activation_fn(gate_1) * up_1, moe_w2_local, group_sizes_1),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )
    with jax.named_scope("scatter_chunk_1"):
        out_global_1 = (
            jnp.zeros_like(x_global_1)
            .at[second_token_indices]
            .add(out_dispatch_1 * weight_dispatch_1[:, None], mode="drop")
        )
        out_local_1 = jax.lax.psum_scatter(out_global_1, "expert", scatter_dimension=0, tiled=True)

    return jnp.concatenate((out_local_0, out_local_1), axis=0)


def _moe_mlp_ep_ring_dispatch_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
    combine_dtype: Literal["bf16", "fp32"] = "bf16",
) -> _BulkRingDispatchState:
    """Run the exact bulk-ring routing and gather phase for overlap benchmarks."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=local_experts,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    return _bulk_ring_dispatch_from_routing(
        x_local,
        combine_weights_local,
        routing,
        combine_dtype=combine_dtype,
    )


def _moe_mlp_ep_ring_expert_local(
    dispatch: _BulkRingDispatchState,
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    ops: MoeRaggedDotOps | None = None,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> _BulkRingExpertState:
    """Run exact bulk-ring expert compute on an expert-sharded dispatch buffer."""
    return _bulk_ring_expert_compute(
        dispatch,
        moe_w13_local,
        moe_w2_local,
        ops,
        activation_fn=activation_fn,
    )


def _moe_mlp_ep_ring_combine_local(
    dispatch: _BulkRingDispatchState,
    expert: _BulkRingExpertState,
    *,
    tokens_per_shard: int,
    expert_axis_size: int,
    combine_dtype: Literal["bf16", "fp32"] = "bf16",
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Run exact bulk-ring scatter and reduce the carried drop count."""
    out_local = _bulk_ring_combine(
        dispatch,
        expert,
        tokens_per_shard=tokens_per_shard,
        expert_axis_size=expert_axis_size,
        combine_dtype=combine_dtype,
    )
    dropped_total = jax.lax.psum(dispatch.dropped_local[0], _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    ops: MoeRaggedDotOps | None = None,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    combine_dtype: Literal["bf16", "fp32"] = "bf16",
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    out_local = _bulk_ring_from_routing(
        x_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        routing,
        ops,
        activation_fn=activation_fn,
        combine_dtype=combine_dtype,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _moe_mlp_ep_ring_local_accumulating_weight_gradient(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    w13_accumulator_local: Float[Array, "Elocal H I2"],
    w2_accumulator_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""], Float[Array, ""]]:
    """Ring path whose expert-weight VJPs include data-local FP32 accumulators.

    The scalar token is zero in the forward pass. Callers must add it to the
    normalized loss with coefficient exactly one so reverse mode includes each
    prior accumulator once.
    """
    _validate_accumulating_weight_gradient_contract(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        w13_accumulator_local,
        w2_accumulator_local,
    )
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    out_local, accumulation_token = _bulk_ring_from_routing_accumulating_weight_gradient(
        x_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        w13_accumulator_local,
        w2_accumulator_local,
        routing,
        activation_fn=activation_fn,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total, accumulation_token


def _moe_mlp_ep_ring_local_accumulating_weight_gradient_backward(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    w13_accumulator_local: Float[Array, "Elocal H I2"],
    w2_accumulator_local: Float[Array, "Elocal I H"],
    output_cotangent_local: Float[Array, "Tlocal H"],
    accumulation_scale: Float[Array, ""],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[
    Float[Array, "Tlocal H"],
    Float[Array, "Tlocal K"],
    Float[Array, "Elocal H I2"],
    Float[Array, "Elocal I H"],
]:
    """Apply accumulating Ring's explicit local pullback."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    return _bulk_ring_from_routing_accumulating_weight_gradient_backward(
        x_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        w13_accumulator_local,
        w2_accumulator_local,
        routing,
        output_cotangent_local,
        accumulation_scale,
        activation_fn=activation_fn,
    )


def _moe_mlp_ep_ring_fp8_wire_approx_local(
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
    """Benchmark-only approximate ring transport with E4M3 collective payloads."""
    _validate_fp8_wire_contract(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
    )
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    out_local = _bulk_ring_fp8_wire_from_routing(
        x_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        routing,
        activation_fn=activation_fn,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _moe_mlp_ep_ring_quack_local(
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
    """Bulk-ring path using QuACK's approximate fused SwiGLU expert MLP."""
    _validate_quack_bulk_ring_contract(x_local, moe_w13_local, moe_w2_local, activation_fn=activation_fn)
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    out_local = _bulk_ring_quack_from_routing(
        x_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        routing,
        activation_fn=activation_fn,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _moe_mlp_ep_ring_two_chunk_local(
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
    """Benchmark-only exact two-chunk bulk-ring prototype."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    use_two_chunk = _two_chunk_fast_path_decision(routing)
    out_local = jax.lax.cond(
        use_two_chunk,
        lambda _: _two_chunk_ring_from_routing(
            x_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            routing,
            activation_fn=activation_fn,
        ),
        lambda _: _bulk_ring_from_routing(
            x_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            routing,
            activation_fn=activation_fn,
        ),
        operand=None,
    )
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _ep_ring_two_chunk_fast_path_local(
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
) -> Bool[Array, ""]:
    """Return the globally consistent two-chunk gate for benchmark reporting."""
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=local_experts,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    return _two_chunk_fast_path_decision(routing)
