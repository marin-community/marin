# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Share unexpanded tokens within expert groups while retaining global capacity."""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.common import MoeDispatchCounts
from levanter.grug.sharding import _reshard_for_shard_map, _token_spec_from_x

from experiments.grug.belay_tpu import local_experts
from experiments.grug.belay_tpu.capacity_layout import interval_route_layout
from experiments.grug.belay_tpu.streaming_combine import combine_streaming, dispatch_streaming


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def gather_group(value, groups):
    return jax.lax.all_gather(value, "expert", axis=0, tiled=True, axis_index_groups=groups)


def _gather_forward(value, groups):
    return gather_group(value, groups), None


def _gather_backward(groups, residuals, gradient):
    del residuals
    reduced = jax.lax.psum_scatter(
        gradient.astype(jnp.float32), "expert", scatter_dimension=0, tiled=True, axis_index_groups=groups
    )
    return (reduced.astype(gradient.dtype),)


gather_group.defvjp(_gather_forward, _gather_backward)


def hybrid_gathered_expert_mlp(
    x, selected, combine, w_up, w_down, *, mesh, capacity_factor, capacity_pools=16, expert_group_size=2
):
    """Compute disjoint expert subsets on each group of contiguous token shards.

    Parameters and optimizer state keep their original global sharding. Global
    capacity clipping precedes grouping; each expert still accepts a prefix in
    global token order. Only unexpanded tokens and partial token outputs move
    within a group. Weight gathering and its transpose use the other mesh groups.
    """
    if any(size != 1 for name, size in mesh.shape.items() if name != "expert"):
        raise ValueError("Hybrid experts require only the expert mesh axis to be nontrivial")
    peers = mesh.shape["expert"]
    experts = w_up.shape[0]
    if peers % expert_group_size or experts % peers or experts % capacity_pools:
        raise ValueError("Expert groups, parameter shards and capacity pools must divide their dimensions")
    per_shard = experts // peers
    local_experts_count = experts // expert_group_size
    token_groups = tuple(tuple(range(start, start + expert_group_size)) for start in range(0, peers, expert_group_size))
    weight_groups = tuple(tuple(range(owner, peers, expert_group_size)) for owner in range(expert_group_size))
    capacity = max(experts // capacity_pools, math.ceil(capacity_factor * selected.size / capacity_pools))

    def local(x_local, ids, weights, up, down):
        rank = jax.lax.axis_index("expert")
        owner = rank % expert_group_size
        counts = local_experts.route_counts(ids.reshape(-1), experts).astype(jnp.int32)
        all_counts = jax.lax.all_gather(counts, "expert")
        accepted = local_experts.clip_capacity_pools(all_counts, pools=capacity_pools, capacity=capacity)
        group_start = rank - owner
        group_counts = jax.lax.dynamic_slice_in_dim(all_counts, group_start, expert_group_size, axis=0).sum(axis=0)
        group_accepted = jax.lax.dynamic_slice_in_dim(accepted, group_start, expert_group_size, axis=0).sum(axis=0)
        local_ids = jnp.arange(local_experts_count, dtype=jnp.int32)
        owned = (local_ids // per_shard) * per_shard * expert_group_size + owner * per_shard + local_ids % per_shard
        owned_counts, owned_accepted = group_counts[owned], group_accepted[owned]

        group_x = gather_group(x_local, token_groups)
        group_ids = jax.lax.all_gather(ids, "expert", axis=0, tiled=True, axis_index_groups=token_groups)
        group_weights = gather_group(weights, token_groups)
        up, down = gather_group(up, weight_groups), gather_group(down, weight_groups)
        flat = group_ids.reshape(-1)
        local_flat = (flat // (per_shard * expert_group_size)) * per_shard + flat % per_shard
        local_flat = jnp.where((flat // per_shard) % expert_group_size == owner, local_flat, local_experts_count)
        order, inverse, _ = interval_route_layout(local_flat, owned_counts, owned_accepted)
        active_rows = owned_accepted.sum()
        active = jnp.arange(flat.size) < active_rows
        rows = jnp.where(active[:, None], dispatch_streaming(group_x, order, inverse, ids.shape[1]), 0)
        # Group sizes exclude the unused suffix. The output mask prevents reads
        # of unwritten Pallas rows from affecting outputs or parameter gradients.
        outputs = local_experts.expert_mlp(rows, up, down, owned_accepted, active_rows)
        outputs = jnp.where(active[:, None], outputs, 0)
        partial_output = combine_streaming(
            outputs.astype(jnp.float32), order, inverse, group_weights, implementation="scan"
        )
        result = jax.lax.psum_scatter(
            partial_output, "expert", scatter_dimension=0, tiled=True, axis_index_groups=token_groups
        ).astype(x_local.dtype)
        dropped = jax.lax.psum(ids.size - accepted[rank].sum(), "expert")
        return result, dropped

    token = P(_token_spec_from_x(x, mesh)[0], None)
    expert = P("expert", None, None)
    specs = (token, token, token, expert, expert)
    inputs = tuple(
        _reshard_for_shard_map(value, mesh, spec)
        for value, spec in zip((x, selected, combine, w_up, w_down), specs, strict=True)
    )
    output, dropped = jax.shard_map(local, mesh=mesh, in_specs=specs, out_specs=(token, P()), check_vma=False)(*inputs)
    zero = jnp.zeros((), jnp.int32)
    return output, MoeDispatchCounts(sender_dropped=zero, receiver_dropped=dropped, padding_skipped=zero)
