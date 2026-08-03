# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-capacity all-to-all expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.jax_utils import tree_checkpoint_name
from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_MOE_OUTPUT
from levanter.grug.sharding import _batch_axes


def _fixed_a2a_capacity(*, assignments_per_shard: int, num_experts: int, capacity_factor: float) -> int:
    return max(int(math.ceil(capacity_factor * assignments_per_shard / num_experts)), 1)


@jax.custom_vjp
def _dispatch_gather(
    x_local: Float[Array, "Tlocal H"],
    token_sources: Int[Array, " send"],
    linear_indices: Int[Array, " assignments"],
    keep: Array,
) -> Float[Array, "send H"]:
    """Gather dispatch rows with a backward gather that avoids scatter-add."""
    hidden_dim = x_local.shape[1]
    padded_x = jnp.concatenate([x_local, jnp.zeros((1, hidden_dim), x_local.dtype)], axis=0)
    return padded_x[token_sources]


def _dispatch_gather_fwd(x_local, token_sources, linear_indices, keep):
    send_x = _dispatch_gather(x_local, token_sources, linear_indices, keep)
    return send_x, (linear_indices, keep, x_local.shape[0])


def _dispatch_gather_bwd(residual, cotangent):
    linear_indices, keep, tokens_per_shard = residual
    send_size, hidden_dim = cotangent.shape
    topk = linear_indices.shape[0] // tokens_per_shard
    grad_rows = cotangent[jnp.minimum(linear_indices, send_size - 1)]
    grad_rows = jnp.where(keep[:, None], grad_rows, 0).astype(jnp.float32)
    grad_rows = grad_rows.reshape(tokens_per_shard, topk, hidden_dim)
    return grad_rows.sum(axis=1).astype(cotangent.dtype), None, None, None


_dispatch_gather.defvjp(_dispatch_gather_fwd, _dispatch_gather_bwd)


@jax.custom_vjp
def _combine_gather(
    send_output: Float[Array, "send H"],
    gather_indices: Int[Array, " assignments"],
    keep: Array,
    assignment_sources: Int[Array, " send"],
) -> Float[Array, "assignments H"]:
    """Restore assignment order with a backward gather from unique send slots."""
    return jnp.where(keep[:, None], send_output[gather_indices], 0)


def _combine_gather_fwd(send_output, gather_indices, keep, assignment_sources):
    gathered = _combine_gather(send_output, gather_indices, keep, assignment_sources)
    return gathered, (assignment_sources,)


def _combine_gather_bwd(residual, cotangent):
    (assignment_sources,) = residual
    assignments_per_shard = cotangent.shape[0]
    valid = assignment_sources < assignments_per_shard
    sources = jnp.minimum(assignment_sources, assignments_per_shard - 1)
    d_send_output = jnp.where(valid[:, None], cotangent[sources], 0).astype(cotangent.dtype)
    return d_send_output, None, None, None


_combine_gather.defvjp(_combine_gather_fwd, _combine_gather_bwd)


def _moe_mlp_ep_fixed_a2a_local(
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
    """Run fixed-capacity all-to-all dispatch, expert MLPs, and combine.

    ``capacity_factor`` scales each fixed (sender shard, global expert) cell as
    ``ceil(factor * local assignments / num_experts)``. An idle cell cannot lend
    rows to a hot cell. Thus, this factor is not directly comparable to the
    receiver-pool factor in the ragged all-to-all backend.
    """
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard = x_local.shape[0]
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    hidden_dim = x_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    capacity = _fixed_a2a_capacity(
        assignments_per_shard=assignments_per_shard,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)

    order = jnp.argsort(flat_experts, stable=True)
    inverse_order = jnp.argsort(order)
    expert_counts = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    segment_start = jnp.cumsum(expert_counts) - expert_counts
    sorted_rank = jnp.arange(assignments_per_shard, dtype=jnp.int32) - segment_start[flat_experts[order]]
    slot = sorted_rank[inverse_order]
    keep = slot < capacity
    local_expert_indices = (flat_experts % local_experts).astype(jnp.int32)
    destination_shards = (flat_experts // local_experts).astype(jnp.int32)
    bucket_size = expert_shards * capacity
    send_size = local_experts * bucket_size
    linear_indices = jnp.where(
        keep,
        local_expert_indices * bucket_size + destination_shards * capacity + slot,
        send_size,
    )

    with jax.named_scope("dispatch"):
        assignment_sources = (
            jnp.full((send_size,), assignments_per_shard, dtype=jnp.int32)
            .at[linear_indices]
            .set(jnp.arange(assignments_per_shard, dtype=jnp.int32), mode="drop")
        )
        token_sources = jnp.where(
            assignment_sources < assignments_per_shard,
            assignment_sources // topk,
            tokens_per_shard,
        )
        send_x = _dispatch_gather(x_local, token_sources, linear_indices, keep)
        send_x = send_x.reshape(local_experts, expert_shards, capacity, hidden_dim)

    moe_dim = moe_w2_local.shape[1]
    output_parts = []
    for local_expert_index in range(local_experts):
        with jax.named_scope("dispatch"):
            received = jax.lax.all_to_all(
                send_x[local_expert_index],
                "expert",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
            received = tree_checkpoint_name(received, _CHECKPOINT_DISPATCH_INPUT)
        with jax.named_scope("moe_up_down"):
            expert_input = received.reshape(bucket_size, hidden_dim)
            hidden = expert_input @ moe_w13_local[local_expert_index]
            gate, up = jnp.split(hidden, [moe_dim], axis=-1)
            expert_output = (activation_fn(gate) * up) @ moe_w2_local[local_expert_index]
        with jax.named_scope("combine"):
            returned = jax.lax.all_to_all(
                expert_output.reshape(expert_shards, capacity, hidden_dim),
                "expert",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
            output_parts.append(returned)

    with jax.named_scope("combine"):
        send_output = jnp.stack(output_parts, axis=0)
        send_output = tree_checkpoint_name(send_output, _CHECKPOINT_MOE_OUTPUT)
        send_output = send_output.reshape(send_size, hidden_dim)
        gather_indices = jnp.minimum(linear_indices, send_size - 1)
        gathered = _combine_gather(send_output, gather_indices, keep, assignment_sources)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        dropped_local = assignments_per_shard - jnp.sum(keep, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
