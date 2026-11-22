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

_H100_DECODE_EXPERT_CAPACITY = 256


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
    # #2710 ring EP strategy: gather tokens and their selected-expert routing
    # assignments across expert shards, then psum-scatter back to local tokens.
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        token_valid_global = jax.lax.all_gather(token_valid_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

        local_experts = gate_or_packed.shape[0]
        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )

        ep_size = num_experts // local_experts
        physical_capacity = int(math.ceil(capacity_factor * assignments / ep_size))
        physical_capacity = min(assignments, max(local_experts, physical_capacity))
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

    if (
        up_weight is not None
        and physical_capacity <= _H100_DECODE_EXPERT_CAPACITY
        and x_local.dtype == jnp.bfloat16
        and jax.default_backend() == "gpu"
        and "H100" in jax.devices()[0].device_kind
    ):
        dot = triton_decode_ragged_dot
    else:
        dot = partial(ragged_dot, megablox_token_tile=megablox_token_tile, megablox_output_tile=megablox_output_tile)

    def compute(buffer_rows: int) -> jax.Array:
        valid = jnp.arange(buffer_rows, dtype=jnp.int32) < accepted_total
        _, local_idx = jax.lax.top_k(selection_key, buffer_rows)
        token_local = jnp.floor_divide(local_idx, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_take = jnp.take(x_global, token_local, axis=0)
        x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
        with jax.named_scope("moe_up_down"):
            # Omit unused rows from the grouped dots and mask their unwritten results.
            active_rows = valid[:, None]
            gate_out = jnp.where(active_rows, dot(x_dispatch, gate_or_packed, accepted_counts), 0)
            if up_weight is None:
                gate_out = tree_checkpoint_name(gate_out, _CHECKPOINT_EXPERT_HIDDEN)
                gate, up = jnp.split(gate_out, [moe_w2_local.shape[1]], axis=-1)
            else:
                gate = gate_out
                up = jnp.where(active_rows, dot(x_dispatch, up_weight, accepted_counts), 0)
            out_dispatch = jnp.where(active_rows, dot(activation_fn(gate) * up, moe_w2_local, accepted_counts), 0)
            out_dispatch = tree_checkpoint_name(out_dispatch, _CHECKPOINT_DISPATCH_OUTPUT)

        with jax.named_scope("scatter"):
            return jnp.zeros_like(x_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")

    # A TPU shard normally needs no more than twice its even share of routes.
    # Keep the full capacity for skewed routing. Communication stays outside the branch.
    balanced_capacity = max(local_experts, 2 * assignments // ep_size)
    if jax.default_backend() == "tpu" and balanced_capacity < physical_capacity:
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
