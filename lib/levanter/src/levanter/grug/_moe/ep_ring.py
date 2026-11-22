# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot, triton_decode_ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _assignment_validity,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug._moe.selected_gpu import selected_expert_mlp, supports_selected_matvec

_SMALL_DECODE_ROUTE_CAPACITY = 256
_MIN_CHUNKED_PREFILL_TOKENS = 512


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    token_valid_local: Bool[Array, "Tlocal"],
    gate_or_packed: Float[Array, "Elocal H W"],
    moe_w2_local: Float[Array, "Elocal I H"],
    up_weight: Float[Array, "Elocal H I"] | None = None,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
    megablox_token_tile: int | None = None,
    megablox_output_tile: int | None = None,
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
    """Gather routes and scatter expert outputs using packed or separate gate and up weights."""
    local_experts = gate_or_packed.shape[0]
    ep_size, remainder = divmod(num_experts, local_experts)
    if remainder:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    local_tokens, width = x_local.shape
    topk = selected_experts_local.shape[1]
    tokens = local_tokens * ep_size
    assignments = tokens * topk
    physical_capacity = min(assignments, max(local_experts, math.ceil(capacity_factor * assignments / ep_size)))
    small_decode = (
        up_weight is not None and physical_capacity <= _SMALL_DECODE_ROUTE_CAPACITY and x_local.dtype == jnp.bfloat16
    )
    # #2710 ring EP strategy: gather tokens and their selected-expert routing
    # assignments across expert shards, then psum-scatter back to local tokens.
    with jax.named_scope("gather"):
        if small_decode and num_experts <= 256 and combine_weights_local.dtype == x_local.dtype:
            # BF16 represents every expert ID exactly for banks up to 256 experts.
            payload = jnp.concatenate(
                (
                    x_local,
                    selected_experts_local.astype(x_local.dtype),
                    combine_weights_local,
                    token_valid_local[:, None],
                ),
                axis=1,
            )
            gathered = jax.lax.all_gather(payload, "expert", tiled=True)
            x_global = gathered[:, :width]
            selected_experts_global = gathered[:, width : width + topk].astype(selected_experts_local.dtype)
            combine_weights_global = gathered[:, width + topk : width + 2 * topk]
            token_valid_global = gathered[:, -1].astype(jnp.bool_)
        else:
            x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
            selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
            combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
            token_valid_global = jax.lax.all_gather(token_valid_local, "expert", tiled=True)

        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)
        assignment_valid = _assignment_validity(token_valid_global, tokens=tokens, topk=topk)
        valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
        logical_capacity = _scaled_capacity(
            valid_assignments,
            capacity_factor=capacity_factor,
            divisor=ep_size,
            minimum=local_experts,
            maximum=physical_capacity,
        )

        expert_axis = jax.lax.axis_index("expert")
        expert_start = expert_axis * local_experts
        local_expert: jax.Array = expert_flat - expert_start
        local_mask = assignment_valid & jnp.logical_and(local_expert >= 0, local_expert < local_experts)

        # Keep only the assignments this shard will execute, ordered by
        # (local expert id, original flat position). This avoids the global
        # argsort + fused takes over all assignments that dominated high-EP
        # shapes, while preserving the grouped layout expected by ragged_dot.
        local_expert = jnp.where(local_mask, local_expert, 0)
        # TPU lowers this small-expert count reduction better as a dense
        # compare+sum than as `bincount`.
        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        local_mask_i32 = local_mask.astype(jnp.int32)
        counts = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
            axis=0,
            dtype=jnp.int32,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=logical_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total

        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)

    if small_decode and jax.default_backend() == "gpu" and "H100" in jax.devices()[0].device_kind:
        dot = triton_decode_ragged_dot
    else:
        dot = partial(ragged_dot, megablox_token_tile=megablox_token_tile, megablox_output_tile=megablox_output_tile)

    def accumulate(local_idx, group_sizes, valid, output):
        token_local = jnp.floor_divide(local_idx, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_take = jnp.take(x_global, token_local, axis=0)
        x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
        with jax.named_scope("moe_up_down"):
            # Omit unused rows from the grouped dots and mask their unwritten results.
            active_rows = valid[:, None]
            gate_out = jnp.where(active_rows, dot(x_dispatch, gate_or_packed, group_sizes), 0)
            if up_weight is None:
                gate_out = tree_checkpoint_name(gate_out, _CHECKPOINT_EXPERT_HIDDEN)
                gate, up = jnp.split(gate_out, [moe_w2_local.shape[1]], axis=-1)
            else:
                gate = gate_out
                up = jnp.where(active_rows, dot(x_dispatch, up_weight, group_sizes), 0)
            out_dispatch = jnp.where(active_rows, dot(activation_fn(gate) * up, moe_w2_local, group_sizes), 0)
            out_dispatch = tree_checkpoint_name(out_dispatch, _CHECKPOINT_DISPATCH_OUTPUT)

        with jax.named_scope("scatter"):
            destinations = jnp.where(valid, token_local, tokens)
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

    # Bound serving scratch at twice the even share, including on skewed routes.
    balanced_capacity = max(local_experts, 2 * assignments // ep_size)
    if balanced_capacity < physical_capacity and up_weight is not None and tokens >= _MIN_CHUNKED_PREFILL_TOKENS:
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
        # #2710 ring EP strategy: collect only this shard's token slice after
        # reducing contributions from experts across the EP mesh.
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, token_sharding_axes)
    return out_local, CapacityDrops(sender_dropped=jnp.zeros_like(dropped_total), receiver_dropped=dropped_total)


def _moe_mlp_ep_small_local(
    x: jax.Array,
    selected: jax.Array,
    weights: jax.Array,
    token_valid: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    *,
    activation: Callable[[jax.Array], jax.Array],
    max_active_tokens: int | None = None,
    replicated_token: bool = False,
) -> jax.Array:
    """Evaluate active routes in a small decode batch with capacity for every assignment."""
    width, top_k = x.shape[1], selected.shape[1]
    local_valid = token_valid[:, None]
    if not replicated_token:
        # A single collective carries the activation and its routing metadata.
        transport_dtype = jnp.result_type(x.dtype, weights.dtype, jnp.float32)
        payload = jnp.concatenate(
            (
                x.astype(transport_dtype),
                selected.astype(transport_dtype),
                weights.astype(transport_dtype),
                token_valid[:, None],
            ),
            axis=1,
        )
        if max_active_tokens == 1:
            # Only one rank contributes an active row, so no rank metadata is needed.
            payload = jax.lax.psum(jnp.where(local_valid, payload, 0), "expert")
        else:
            payload = jax.lax.all_gather(payload, "expert", tiled=True)
        x = payload[:, :width].astype(x.dtype)
        selected = payload[:, width : width + top_k].astype(selected.dtype)
        weights = payload[:, width + top_k : width + 2 * top_k].astype(weights.dtype)
        token_valid = payload[:, -1].astype(jnp.bool_)
    tokens = x.shape[0]
    shard_start = jax.lax.axis_index("expert") * w_gate.shape[0]
    local_experts = selected - shard_start
    local_mask = token_valid[:, None] & (local_experts >= 0) & (local_experts < w_gate.shape[0])
    if max_active_tokens == 1 and jax.default_backend() == "gpu" and supports_selected_matvec(width, w_gate.shape[-1]):
        result = selected_expert_mlp(
            x[0],
            jnp.where(local_mask[0], local_experts[0], -1),
            weights[0],
            w_gate,
            w_up,
            w_down,
            activation=activation,
        )
        return jnp.where(local_valid, jax.lax.psum(result[None], "expert"), 0)

    assignments = jnp.nonzero(local_mask.reshape(-1), size=tokens * top_k)[0]
    assignment_count = jnp.sum(local_mask, dtype=jnp.int32)

    def apply(i, outputs):
        assignment = assignments[i]
        token, route = assignment // top_k, assignment % top_k
        expert = local_experts[token, route]
        gate = jnp.dot(x[token], w_gate[expert], preferred_element_type=x.dtype)
        up = jnp.dot(x[token], w_up[expert], preferred_element_type=x.dtype)
        down = jnp.dot(activation(gate) * up, w_down[expert], preferred_element_type=x.dtype)
        previous = jax.lax.dynamic_index_in_dim(outputs, token, keepdims=False)
        return jax.lax.dynamic_update_index_in_dim(outputs, previous + down * weights[token, route], token, axis=0)

    local_output = jax.lax.fori_loop(0, assignment_count, apply, jnp.zeros_like(x))
    if max_active_tokens == 1:
        return jnp.where(local_valid, jax.lax.psum(local_output, "expert"), 0)
    return jax.lax.psum_scatter(local_output, "expert", scatter_dimension=0, tiled=True)
