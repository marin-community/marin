# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate gate and up expert projections for bounded serving dispatch."""

import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.ep_ring import (
    _gather_ring_inputs,
    _GatheredInputs,
    _ring_physical_capacity,
    _ring_route_order,
    _take_valid_rows,
)

_SMALL_DECODE_ROUTE_CAPACITY = 256
_MIN_CHUNKED_PREFILL_TOKENS = 512


def _gather_serving_inputs(
    activations: jax.Array,
    expert_ids: jax.Array,
    weights: jax.Array,
    token_valid: jax.Array,
    *,
    pack_metadata: bool,
) -> _GatheredInputs:
    if not pack_metadata:
        return _gather_ring_inputs(activations, expert_ids, weights, token_valid)
    # BF16 represents every expert ID exactly for banks up to 256 experts.
    width = activations.shape[1]
    top_k = expert_ids.shape[1]
    payload = jnp.concatenate(
        (activations, expert_ids.astype(activations.dtype), weights, token_valid[:, None]), axis=1
    )
    gathered = jax.lax.all_gather(payload, "expert", tiled=True)
    return _GatheredInputs(
        gathered[:, :width],
        gathered[:, width : width + top_k].astype(expert_ids.dtype),
        gathered[:, width + top_k : width + 2 * top_k],
        gathered[:, -1].astype(jnp.bool_),
    )


def _moe_mlp_serving_ring_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    token_valid_local: Bool[Array, "Tlocal"],
    gate_weight: Float[Array, "Elocal H I"],
    up_weight: Float[Array, "Elocal H I"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    megablox_tiles: tuple[int, int] | None = None,
) -> Float[Array, "Tlocal H"]:
    """Return combined outputs for this shard's tokens with bounded prefill scratch."""
    local_experts = gate_weight.shape[0]
    ep_size, remainder = divmod(num_experts, local_experts)
    if remainder:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    local_tokens = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    tokens = local_tokens * ep_size
    assignments = tokens * topk
    physical_capacity = _ring_physical_capacity(assignments, local_experts, ep_size, capacity_factor)
    pack_metadata = (
        physical_capacity <= _SMALL_DECODE_ROUTE_CAPACITY
        and x_local.dtype == jnp.bfloat16
        and num_experts <= 256
        and combine_weights_local.dtype == x_local.dtype
    )
    with jax.named_scope("gather"):
        gathered = _gather_serving_inputs(
            x_local, selected_experts_local, combine_weights_local, token_valid_local, pack_metadata=pack_metadata
        )
        x_global = gathered.activations
        weight_flat = gathered.weights.reshape(assignments)
        order = _ring_route_order(
            gathered.expert_ids.reshape(assignments),
            gathered.valid,
            topk=topk,
            local_experts=local_experts,
            ep_size=ep_size,
            capacity_factor=capacity_factor,
            physical_capacity=physical_capacity,
        )
        accepted_counts, selection_key = order.accepted_counts, order.selection_key
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)

    dot = ragged_dot if megablox_tiles is None else partial(ragged_dot, megablox_tiles=megablox_tiles)

    def accumulate(local_idx, group_sizes, valid, output):
        token_index = jnp.floor_divide(local_idx, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_dispatch = _take_valid_rows(x_global, token_index, valid)
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
        with jax.named_scope("moe_up_down"):
            # Omit unused rows from the grouped dots and mask their unwritten results.
            active_rows = valid[:, None]
            gate = jnp.where(active_rows, dot(x_dispatch, gate_weight, group_sizes), 0)
            up = jnp.where(active_rows, dot(x_dispatch, up_weight, group_sizes), 0)
            out_dispatch = jnp.where(active_rows, dot(activation(gate) * up, moe_w2_local, group_sizes), 0)

        with jax.named_scope("combine"):
            destinations = jnp.where(valid, token_index, tokens)
            return output.at[destinations].add(out_dispatch * weight_dispatch[:, None], mode="drop")

    def compute(buffer_rows: int) -> jax.Array:
        _, local_idx = jax.lax.top_k(selection_key, buffer_rows)
        valid = jnp.arange(buffer_rows, dtype=jnp.int32) < accepted_total
        return accumulate(local_idx, accepted_counts, valid, jnp.zeros_like(x_global))

    def compute_in_chunks(buffer_rows: int) -> jax.Array:
        _, local_idx = jax.lax.top_k(selection_key, physical_capacity)
        num_chunks = math.ceil(physical_capacity / buffer_rows)
        local_idx = jnp.pad(local_idx, (0, num_chunks * buffer_rows - physical_capacity))
        ends = jnp.cumsum(accepted_counts)
        starts = ends - accepted_counts
        rows = jnp.arange(buffer_rows, dtype=jnp.int32)

        def process_chunk(chunk, output):
            offset = chunk * buffer_rows

            def process():
                indices = jax.lax.dynamic_slice_in_dim(local_idx, offset, buffer_rows)
                group_sizes = jnp.maximum(0, jnp.minimum(ends, offset + buffer_rows) - jnp.maximum(starts, offset))
                return accumulate(indices, group_sizes, offset + rows < accepted_total, output)

            return jax.lax.cond(offset < accepted_total, process, lambda: output)

        return jax.lax.fori_loop(0, num_chunks, process_chunk, jnp.zeros_like(x_global))

    # Chunk large prefills when routes exceed twice the receiving shard's even share.
    balanced_capacity = max(local_experts, 2 * assignments // ep_size)
    if balanced_capacity < physical_capacity and tokens >= _MIN_CHUNKED_PREFILL_TOKENS:
        out_global = jax.lax.cond(
            accepted_total <= balanced_capacity,
            lambda: compute(balanced_capacity),
            lambda: compute_in_chunks(balanced_capacity),
        )
    elif jax.default_backend() == "tpu" and balanced_capacity < physical_capacity:
        out_global = jax.lax.cond(
            accepted_total <= balanced_capacity, lambda: compute(balanced_capacity), lambda: compute(physical_capacity)
        )
    else:
        out_global = compute(physical_capacity)
    with jax.named_scope("scatter"):
        return jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
