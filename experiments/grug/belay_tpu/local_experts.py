# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local-token MoE with sharded expert state and gathered compute weights.

Capacity pools are independent of the device count. The Ckpt 1 configuration
keeps sixteen pools of twenty-four experts, including their acceptance order.
Only weights, weight gradients and small routing counts cross devices.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.common import MoeDispatchCounts
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug.sharding import _reshard_for_shard_map, _token_spec_from_x


def clip_capacity_pools(counts, *, pools, capacity):
    """Accept expert-major, then sender-major prefixes in each global pool."""
    senders, experts = counts.shape
    if experts % pools:
        raise ValueError("Capacity pools must divide the expert bank")
    width = experts // pools
    ordered = counts.reshape(senders, pools, width).transpose(1, 2, 0).reshape(pools, -1)
    accepted = _prefix_cap_counts(ordered, capacity=capacity)
    return accepted.reshape(pools, width, senders).transpose(2, 0, 1).reshape(counts.shape)


@jax.custom_vjp
def gather_expert_weights(weights):
    return jax.lax.all_gather(weights, "expert", axis=0, tiled=True)


def _gather_weights_fwd(weights):
    return gather_expert_weights(weights), None


def reduce_expert_gradients(gradient):
    return jax.lax.psum_scatter(gradient, "expert", scatter_dimension=0, tiled=True)


def _gather_weights_bwd(_, gradient):
    # Accumulate contributions from token shards in FP32 before returning to the
    # compute-weight dtype. Master parameters and optimizer state remain FP32.
    reduced = reduce_expert_gradients(gradient.astype(jnp.float32))
    return (reduced.astype(gradient.dtype),)


gather_expert_weights.defvjp(_gather_weights_fwd, _gather_weights_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def dispatch_local(x, order, inverse, topk):
    return x[order // topk]


def _dispatch_fwd(x, order, inverse, topk):
    return dispatch_local(x, order, inverse, topk), inverse


def _dispatch_bwd(topk, inverse, gradient):
    # Each input occurs K times. Invert the permutation and reduce slots rather
    # than issuing a general scatter-add into token rows.
    restored = gradient[inverse].reshape(-1, topk, gradient.shape[-1])
    return restored.astype(jnp.float32).sum(axis=1).astype(gradient.dtype), None, None


dispatch_local.defvjp(_dispatch_fwd, _dispatch_bwd)


@jax.custom_vjp
def restore_local(rows, order, inverse):
    return rows[inverse]


def _restore_fwd(rows, order, inverse):
    return restore_local(rows, order, inverse), order


def _restore_bwd(order, gradient):
    return gradient[order], None, None


restore_local.defvjp(_restore_fwd, _restore_bwd)


def combine_local(rows, order, inverse, weights):
    slots = restore_local(rows, order, inverse).reshape(*weights.shape, rows.shape[-1])
    return (slots.astype(jnp.float32) * weights[..., None]).sum(axis=1).astype(rows.dtype)


def _tiling(m, k, n):
    return math.gcd(m, 128), math.gcd(k, 256), math.gcd(n, 256)


def grouped_dot(x, weights, groups):
    if jax.default_backend() == "tpu":
        return gmm(x, weights, groups, preferred_element_type=x.dtype, tiling=_tiling)
    # Operator and gradient oracles can run remotely on CPU; no training here.
    return jax.lax.ragged_dot(x, weights, groups, preferred_element_type=x.dtype)


def expert_mlp(rows, w_up, w_down, groups, active_rows):
    del active_rows
    up = grouped_dot(rows, w_up, groups)
    hidden = jnp.square(jax.nn.relu(up))
    return grouped_dot(hidden, w_down, groups)


def route_counts(flat, experts):
    return jnp.bincount(flat, length=experts)


def route_layout(flat, counts, accepted_counts):
    experts = counts.shape[0]
    original_order = jnp.argsort(flat, stable=True)
    original_positions = jnp.argsort(original_order)
    cutoff = jnp.cumsum(counts) - counts + accepted_counts
    keep = original_positions < cutoff[flat]
    order = jnp.argsort(jnp.where(keep, flat, experts), stable=True)
    inverse = jnp.argsort(order)
    return order, inverse, keep


def local_expert_mlp(x, selected, combine, w_up, w_down, *, accepted_counts, expert_combine=None):
    """Compute locally after global capacity allocation, retaining activations."""
    tokens, topk = selected.shape
    experts = w_up.shape[0]
    flat = selected.reshape(-1)
    counts = route_counts(flat, experts)
    order, inverse, keep = route_layout(flat, counts, accepted_counts)
    active = jnp.arange(flat.size) < accepted_counts.sum()
    # Fill the unused suffix with zero rows in the last group. This keeps every
    # output initialized, including empty experts and completely rejected shards.
    physical_counts = accepted_counts.at[-1].add(flat.size - accepted_counts.sum())
    rows = jnp.where(active[:, None], dispatch_local(x, order, inverse, topk), 0)
    if expert_combine is None:
        outputs = expert_mlp(rows, w_up, w_down, physical_counts, accepted_counts.sum())
        outputs = jnp.where(active[:, None], outputs, 0)
        # Match the source combine's FP32 sum; rejected routes have zero outputs.
        result = combine_local(outputs, order, inverse, combine)
    else:
        result = expert_combine(rows, w_up, w_down, physical_counts, accepted_counts.sum(), order, inverse, combine)
    return result, keep.reshape(tokens, topk)


def gathered_expert_mlp(
    x, selected, combine, w_up, w_down, *, mesh, capacity_factor, capacity_pools=16, expert_combine=None
):
    """Run local-token MoE on an expert-state-sharded, model-axis-one mesh."""
    if any(size != 1 for name, size in mesh.shape.items() if name != "expert"):
        raise ValueError("Local-token MoE requires only the expert mesh axis to have size greater than one")
    experts = w_up.shape[0]
    if experts % capacity_pools:
        raise ValueError("Capacity pools must divide the expert bank")
    capacity = max(experts // capacity_pools, math.ceil(capacity_factor * selected.size / capacity_pools))

    def local(x_local, ids, weights, up, down):
        counts = route_counts(ids.reshape(-1), experts).astype(jnp.int32)
        all_counts = jax.lax.all_gather(counts, "expert")
        accepted = clip_capacity_pools(all_counts, pools=capacity_pools, capacity=capacity)
        local_accepted = accepted[jax.lax.axis_index("expert")]
        up, down = gather_expert_weights(up), gather_expert_weights(down)
        output, _ = local_expert_mlp(
            x_local, ids, weights, up, down, accepted_counts=local_accepted, expert_combine=expert_combine
        )
        dropped = jax.lax.psum(ids.size - local_accepted.sum(), "expert")
        return output, dropped

    token = P(_token_spec_from_x(x, mesh)[0], None)
    expert = P("expert", None, None)
    specs = (token, token, token, expert, expert)
    inputs = tuple(
        _reshard_for_shard_map(value, mesh, spec)
        for value, spec in zip((x, selected, combine, w_up, w_down), specs, strict=True)
    )
    output, dropped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=specs,
        out_specs=(token, P()),
        check_vma=False,
    )(*inputs)
    zero = jnp.zeros((), jnp.int32)
    return output, MoeDispatchCounts(sender_dropped=zero, receiver_dropped=dropped, padding_skipped=zero)
