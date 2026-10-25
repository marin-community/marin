# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Origin-side MoE routing gradients adapted from Base16 aef63f29d.

Retaining returned expert outputs removes one scalar all-to-all per layer.
The direct weight derivative avoids the source's BF16 multiply/divide roundoff.
"""

from functools import partial

import jax
import jax.numpy as jnp
from levanter.grug._moe import ep_ragged_all_to_all as ragged


def _forward(sorted_x, weights, up, down, routing, layout):
    if layout.mxfp8_dispatch:
        raise ValueError("The saved-output diagnostic requires the frozen BF16 dispatch path")
    returned = ragged._transport_buffer(
        sorted_x.shape[0],
        down.shape[-1],
        sorted_x.dtype,
        routing.group_sizes,
        site=ragged._TransportBufferSite.RETURN_OUTPUT,
    )
    chunks = []
    for index, plan in enumerate(ragged._chunk_plans(routing, layout)):
        with jax.named_scope(f"moe_chunk_{index}"):
            source = sorted_x
            if chunks:
                source, _ = jax.lax.optimization_barrier((sorted_x, chunks[-1].expert_mlp))
            dispatched = ragged._dispatch_chunk(source, plan, layout)
            experts = slice(index * layout.chunk_experts, (index + 1) * layout.chunk_experts)
            output, residuals = layout.expert_mlp.forward(
                dispatched, up[experts], down[experts], plan.physical_group_sizes, plan.active_group_sizes
            )
            returned = jax.lax.ragged_all_to_all(output, returned, *plan.return_params, axis_name="expert")
            chunks.append(ragged._ChunkResiduals(plan, residuals))
    with jax.named_scope("combine"):
        output = ragged._unpermute_from_global_expert(
            returned,
            routing.sorted_indices,
            weights,
            routing.accepted,
            tokens_per_shard=layout.tokens_per_shard,
            topk=layout.topk,
        ).astype(sorted_x.dtype)
    return output, (weights, routing, tuple(chunks), returned)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def saved_output_routed_experts(sorted_x, weights, up, down, routing, layout):
    """Keep forward returns at their origin to avoid the backward scalar collective."""
    output, _ = _forward(sorted_x, weights, up, down, routing, layout)
    return output


def routing_cotangents(returned, weights, sorted_indices, accepted, cotangent):
    """Differentiate weighted expert returns while retaining the source's zero-weight convention."""
    weights_f32 = weights.astype(jnp.float32)
    sorted_weights = ragged._permute_assignment_weights(weights_f32.reshape(-1), sorted_indices)
    token_cotangent = cotangent[sorted_indices // weights.shape[1]].astype(jnp.float32)
    returned_cotangent = (token_cotangent * sorted_weights[:, None]).astype(cotangent.dtype)
    sorted_weight_cotangent = jnp.sum(returned.astype(jnp.float32) * token_cotangent, axis=-1)
    weight_cotangent = ragged._restore_assignment_scalars(sorted_weight_cotangent, sorted_indices).reshape(weights.shape)
    weight_cotangent = jnp.where(accepted & (weights_f32 != 0), weight_cotangent, 0).astype(weights.dtype)
    return returned_cotangent, weight_cotangent


def _backward(layout, residuals, cotangent):
    weights, routing, chunks, returned = residuals
    assignments = routing.sorted_indices.shape[0]
    with jax.named_scope("combine"):
        returned_cotangent, weight_cotangent = routing_cotangents(
            returned, weights, routing.sorted_indices, routing.accepted, cotangent
        )
    dx = ragged._transport_buffer(
        assignments,
        layout.input_dim,
        cotangent.dtype,
        routing.group_sizes,
        site=ragged._TransportBufferSite.DISPATCH_COTANGENT,
    )
    up_gradients, down_gradients = [], []
    for index in reversed(range(layout.chunks)):
        plan, expert_residuals = chunks[index]
        with jax.named_scope(f"moe_chunk_{index}"):
            initial = ragged._transport_buffer(
                layout.chunk_capacity,
                cotangent.shape[1],
                cotangent.dtype,
                plan.return_params.recv_sizes,
                site=ragged._TransportBufferSite.RETURN_COTANGENT,
            )
            expert_cotangent = jax.lax.ragged_all_to_all(
                returned_cotangent, initial, *plan.dispatch_params, axis_name="expert"
            )
            expert_dx, dup, ddown, _ = layout.expert_mlp.backward(expert_residuals, expert_cotangent)
            dx = jax.lax.ragged_all_to_all(expert_dx, dx, *plan.return_params, axis_name="expert")
            up_gradients.append(dup)
            down_gradients.append(ddown)
    return dx, weight_cotangent, jnp.concatenate(up_gradients[::-1]), jnp.concatenate(down_gradients[::-1]), None


saved_output_routed_experts.defvjp(_forward, _backward)
