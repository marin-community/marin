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
    """Run fixed-capacity all-to-all dispatch, expert MLPs, and combine."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard = x_local.shape[0]
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    hidden_dim = x_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    capacity = max(int(math.ceil(capacity_factor * assignments_per_shard / num_experts)), 1)

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
        padded_x = jnp.concatenate([x_local, jnp.zeros((1, hidden_dim), dtype=x_local.dtype)], axis=0)
        send_x = padded_x[token_sources]
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
        gathered = send_output[jnp.minimum(linear_indices, send_size - 1)]
        gathered = jnp.where(keep[:, None], gathered, 0)
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
