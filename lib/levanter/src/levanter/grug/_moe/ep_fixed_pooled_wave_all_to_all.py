# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed all-to-all with destination-pooled static waves."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from shape_extensions import Int, IntVar

from levanter.grug._moe.common import _assignment_validity, _scaled_capacity, CapacityDrops
from levanter.grug._moe.ep_common import (
    _assignment_sources,
    _ranks_within_groups,
    _token_sources,
)
from levanter.grug._moe.sonic import sonic_gather_sum_available, sonic_gather_sum_masked

# `_PooledDispatch`/`_PooledOutput` are not generic: their send, receive, and assignment axes are
# fixed physical capacities, so the fields use unknown lengths rather than symbolic dimensions.


class _PooledDispatch(NamedTuple):
    compacted_x: jax.Array[[int, int, int]]
    receiver_linear_indices: jax.Array[[int]]
    receiver_keep: jax.Array[[int]]
    sender_linear_indices: jax.Array[[int]]
    sender_keep: jax.Array[[int]]
    assignment_sources: jax.Array[[int]]
    receiver_dropped: jax.Array[[]]


class _PooledOutput(NamedTuple):
    compacted_output: jax.Array[[int, int, int]]
    receiver_linear_indices: jax.Array[[int]]
    receiver_keep: jax.Array[[int]]
    sender_linear_indices: jax.Array[[int]]
    sender_keep: jax.Array[[int]]
    assignment_sources: jax.Array[[int]]
    receiver_dropped: jax.Array[[]]


def _dispatch_gather_input_grad[Tlocal: IntVar, H: IntVar](
    cotangent: jax.Array[[int, H]],
    linear_indices: jax.Array[[int]],
    keep: jax.Array[[int]],
    tokens_per_shard: Int[Tlocal],
) -> jax.Array[[Tlocal, H]]:
    send_size, hidden_dim = cotangent.shape
    topk = linear_indices.shape[0] // tokens_per_shard
    linear_indices_grouped: jax.Array[[Tlocal, int]] = linear_indices.reshape(tokens_per_shard, topk)
    keep_grouped: jax.Array[[Tlocal, int]] = keep.reshape(tokens_per_shard, topk)

    if sonic_gather_sum_available():
        return sonic_gather_sum_masked(cotangent, linear_indices_grouped, keep_grouped.astype(jnp.float32))

    grad_x = jnp.zeros((tokens_per_shard, hidden_dim), dtype=jnp.float32)

    def add_route(route_index, grad_x):
        rows = cotangent[jnp.minimum(linear_indices_grouped[:, route_index], send_size - 1)]
        rows = jnp.where(keep_grouped[:, route_index, None], rows, 0).astype(jnp.float32)
        return grad_x + rows

    grad_x = jax.lax.fori_loop(0, topk, add_route, grad_x)
    return grad_x.astype(cotangent.dtype)


def _combine_gather_sum_impl[Tlocal: IntVar, K: IntVar, H: IntVar](
    send_output: jax.Array[[int, H]],
    gather_indices: jax.Array[[int]],
    keep: jax.Array[[int]],
    combine_weights: jax.Array[[Tlocal, K]],
) -> jax.Array[[Tlocal, H]]:
    tokens_per_shard, topk = combine_weights.shape
    hidden_dim = send_output.shape[1]
    gather_indices_grouped: jax.Array[[Tlocal, K]] = gather_indices.reshape(tokens_per_shard, topk)
    keep_grouped: jax.Array[[Tlocal, K]] = keep.reshape(tokens_per_shard, topk)

    if sonic_gather_sum_available():
        weights = jnp.where(keep_grouped, combine_weights, 0)
        return sonic_gather_sum_masked(send_output, gather_indices_grouped, weights, output_dtype=jnp.float32)

    out = jnp.zeros((tokens_per_shard, hidden_dim), dtype=jnp.float32)

    def add_route(route_index, out):
        rows = send_output[jnp.minimum(gather_indices_grouped[:, route_index], send_output.shape[0] - 1)]
        rows = jnp.where(keep_grouped[:, route_index, None], rows, 0)
        weight = combine_weights[:, route_index, None].astype(jnp.float32)
        return out + rows.astype(jnp.float32) * weight

    return jax.lax.fori_loop(0, topk, add_route, out)


@jax.custom_vjp
def _combine_gather_sum[Tlocal: IntVar, K: IntVar, H: IntVar](
    send_output: jax.Array[[int, H]],
    gather_indices: jax.Array[[int]],
    keep: jax.Array[[int]],
    assignment_sources: jax.Array[[int]],
    combine_weights: jax.Array[[Tlocal, K]],
) -> jax.Array[[Tlocal, H]]:
    return _combine_gather_sum_impl(send_output, gather_indices, keep, combine_weights)


def _combine_gather_sum_fwd[Tlocal: IntVar, K: IntVar, H: IntVar](
    send_output: jax.Array[[int, H]],
    gather_indices: jax.Array[[int]],
    keep: jax.Array[[int]],
    assignment_sources: jax.Array[[int]],
    combine_weights: jax.Array[[Tlocal, K]],
) -> tuple[
    jax.Array[[Tlocal, H]],
    tuple[jax.Array[[int, H]], jax.Array[[int]], jax.Array[[int]], jax.Array[[int]], jax.Array[[Tlocal, K]]],
]:
    out = _combine_gather_sum_impl(send_output, gather_indices, keep, combine_weights)
    return out, (send_output, gather_indices, keep, assignment_sources, combine_weights)


def _combine_gather_sum_bwd[Tlocal: IntVar, K: IntVar, H: IntVar](
    residual: tuple[jax.Array[[int, H]], jax.Array[[int]], jax.Array[[int]], jax.Array[[int]], jax.Array[[Tlocal, K]]],
    cotangent: jax.Array[[Tlocal, H]],
) -> tuple[jax.Array[[int, H]], None, None, None, jax.Array[[Tlocal, K]]]:
    send_output, gather_indices, keep, assignment_sources, combine_weights = residual
    tokens_per_shard, topk = combine_weights.shape
    assignments_per_shard = tokens_per_shard * topk
    valid = assignment_sources < assignments_per_shard
    sources = jnp.minimum(assignment_sources, assignments_per_shard - 1)
    source_tokens = sources // topk
    source_weights = combine_weights.reshape(-1)[sources].astype(jnp.float32)
    gathered: jax.Array[[int, H]] = cotangent[source_tokens]
    d_send_output_raw = gathered.astype(jnp.float32) * source_weights[:, None]
    d_send_output = jnp.where(valid[:, None], d_send_output_raw, 0).astype(send_output.dtype)

    gather_indices_grouped: jax.Array[[Tlocal, K]] = gather_indices.reshape(tokens_per_shard, topk)
    keep_grouped: jax.Array[[Tlocal, K]] = keep.reshape(tokens_per_shard, topk)
    d_combine_weights = jnp.zeros_like(combine_weights)

    def set_route(route_index, d_combine_weights):
        rows = send_output[jnp.minimum(gather_indices_grouped[:, route_index], send_output.shape[0] - 1)]
        d_weight = jnp.sum(cotangent.astype(jnp.float32) * rows.astype(jnp.float32), axis=1)
        d_weight = jnp.where(keep_grouped[:, route_index], d_weight, 0).astype(combine_weights.dtype)
        return jax.lax.dynamic_update_slice_in_dim(
            d_combine_weights,
            d_weight[:, None],
            route_index,
            axis=1,
        )

    d_combine_weights = jax.lax.fori_loop(0, topk, set_route, d_combine_weights)
    return d_send_output, None, None, None, d_combine_weights


_combine_gather_sum.defvjp(_combine_gather_sum_fwd, _combine_gather_sum_bwd)


def _compact_received_impl[H: IntVar](
    received_x: jax.Array[[int, H]],
    receiver_linear_indices: jax.Array[[int]],
    local_experts: int,
    receiver_capacity: int,
) -> jax.Array[[int, int, H]]:
    compact_size = local_experts * receiver_capacity
    compact_sources = _assignment_sources(receiver_linear_indices, send_size=compact_size)
    source_valid = compact_sources < received_x.shape[0]
    source_indices = jnp.minimum(compact_sources, received_x.shape[0] - 1)
    compacted_x = jnp.where(source_valid[:, None], received_x[source_indices], 0)
    return compacted_x.reshape(local_experts, receiver_capacity, received_x.shape[1])


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _compact_received[H: IntVar](
    received_x: jax.Array[[int, H]],
    receiver_linear_indices: jax.Array[[int]],
    receiver_keep: jax.Array[[int]],
    local_experts: int,
    receiver_capacity: int,
) -> jax.Array[[int, int, H]]:
    """Compact received rows with a one-to-one gather backward pass."""
    return _compact_received_impl(
        received_x,
        receiver_linear_indices,
        local_experts,
        receiver_capacity,
    )


def _compact_received_fwd[H: IntVar](
    received_x: jax.Array[[int, H]],
    receiver_linear_indices: jax.Array[[int]],
    receiver_keep: jax.Array[[int]],
    local_experts: int,
    receiver_capacity: int,
) -> tuple[jax.Array[[int, int, H]], tuple[jax.Array[[int]], jax.Array[[int]], tuple[int, ...]]]:
    compacted_x = _compact_received_impl(
        received_x,
        receiver_linear_indices,
        local_experts,
        receiver_capacity,
    )
    return compacted_x, (receiver_linear_indices, receiver_keep, received_x.shape)


def _compact_received_bwd[H: IntVar](
    local_experts: int,
    receiver_capacity: int,
    residual: tuple[jax.Array[[int]], jax.Array[[int]], tuple[int, ...]],
    cotangent: jax.Array[[int, int, H]],
) -> tuple[jax.Array[[int, H]], None, None]:
    receiver_linear_indices, receiver_keep, received_shape = residual
    compact_size = local_experts * receiver_capacity
    flat_cotangent: jax.Array[[int, H]] = cotangent.reshape(compact_size, received_shape[1])
    source_indices = jnp.minimum(receiver_linear_indices, compact_size - 1)
    received_grad: jax.Array[[int, H]] = flat_cotangent[source_indices]
    received_grad_masked = jnp.where(receiver_keep[:, None], received_grad, 0)
    return received_grad_masked, None, None


_compact_received.defvjp(_compact_received_fwd, _compact_received_bwd)


def _pooled_dispatch_payload_impl[Tlocal: IntVar, H: IntVar](
    x_local: jax.Array[[Tlocal, H]],
    header: jax.Array[[int, int, H]],
    token_sources: jax.Array[[int]],
) -> jax.Array[[int, int, H]]:
    expert_shards, metadata_rows, hidden_dim = header.shape
    pool_capacity = token_sources.shape[0] // expert_shards
    tokens_per_shard = x_local.shape[0]
    padding_source = tokens_per_shard + expert_shards * metadata_rows

    source_rows = jnp.concatenate(
        [x_local, header.reshape(-1, hidden_dim), jnp.zeros((1, hidden_dim), dtype=x_local.dtype)],
        axis=0,
    )
    header_sources = jnp.arange(
        tokens_per_shard,
        tokens_per_shard + expert_shards * metadata_rows,
        dtype=jnp.int32,
    ).reshape(expert_shards, metadata_rows)
    activation_sources = jnp.where(token_sources < tokens_per_shard, token_sources, padding_source).reshape(
        expert_shards, pool_capacity
    )
    payload_sources = jnp.concatenate([header_sources, activation_sources], axis=1)
    return source_rows[payload_sources.reshape(-1)].reshape(expert_shards, metadata_rows + pool_capacity, hidden_dim)


@jax.custom_vjp
def _pooled_dispatch_payload[Tlocal: IntVar, H: IntVar](
    x_local: jax.Array[[Tlocal, H]],
    header: jax.Array[[int, int, H]],
    token_sources: jax.Array[[int]],
    linear_indices: jax.Array[[int]],
    keep: jax.Array[[int]],
) -> jax.Array[[int, int, H]]:
    """Build the header and activation payload with one output gather."""
    return _pooled_dispatch_payload_impl(x_local, header, token_sources)


def _pooled_dispatch_payload_fwd[Tlocal: IntVar, H: IntVar](
    x_local: jax.Array[[Tlocal, H]],
    header: jax.Array[[int, int, H]],
    token_sources: jax.Array[[int]],
    linear_indices: jax.Array[[int]],
    keep: jax.Array[[int]],
) -> tuple[jax.Array[[int, int, H]], tuple[jax.Array[[int]], jax.Array[[int]], int, int]]:
    payload = _pooled_dispatch_payload_impl(x_local, header, token_sources)
    return payload, (linear_indices, keep, x_local.shape[0], header.shape[1])


def _pooled_dispatch_payload_bwd[Tlocal: IntVar, H: IntVar](
    residual: tuple[jax.Array[[int]], jax.Array[[int]], int, int],
    cotangent: jax.Array[[int, int, H]],
) -> tuple[jax.Array[[Tlocal, H]], None, None, None, None]:
    linear_indices, keep, tokens_per_shard, metadata_rows = residual
    activation_cotangent: jax.Array[[int, H]] = cotangent[:, metadata_rows:].reshape(-1, cotangent.shape[-1])
    grad_x = _dispatch_gather_input_grad(activation_cotangent, linear_indices, keep, tokens_per_shard)
    return grad_x, None, None, None, None


_pooled_dispatch_payload.defvjp(_pooled_dispatch_payload_fwd, _pooled_dispatch_payload_bwd)


def _expand_compacted_impl[H: IntVar](
    compacted_output: jax.Array[[int, int, H]],
    receiver_linear_indices: jax.Array[[int]],
    receiver_keep: jax.Array[[int]],
) -> jax.Array[[int, H]]:
    flat_output = compacted_output.reshape(-1, compacted_output.shape[-1])
    output_indices = jnp.minimum(receiver_linear_indices, flat_output.shape[0] - 1)
    received_output: jax.Array[[int, H]] = flat_output[output_indices]
    return jnp.where(receiver_keep[:, None], received_output, 0)


@jax.custom_vjp
def _expand_compacted[H: IntVar](
    compacted_output: jax.Array[[int, int, H]],
    receiver_linear_indices: jax.Array[[int]],
    receiver_keep: jax.Array[[int]],
) -> jax.Array[[int, H]]:
    """Expand expert rows with a one-to-one set backward pass."""
    return _expand_compacted_impl(compacted_output, receiver_linear_indices, receiver_keep)


def _expand_compacted_fwd[H: IntVar](
    compacted_output: jax.Array[[int, int, H]],
    receiver_linear_indices: jax.Array[[int]],
    receiver_keep: jax.Array[[int]],
) -> tuple[jax.Array[[int, H]], tuple[jax.Array[[int]], jax.Array[[int]], tuple[int, ...]]]:
    received_output = _expand_compacted_impl(compacted_output, receiver_linear_indices, receiver_keep)
    return received_output, (receiver_linear_indices, receiver_keep, compacted_output.shape)


def _expand_compacted_bwd[H: IntVar](
    residual: tuple[jax.Array[[int]], jax.Array[[int]], tuple[int, ...]],
    cotangent: jax.Array[[int, H]],
) -> tuple[jax.Array[[int, int, H]], None, None]:
    receiver_linear_indices, receiver_keep, compacted_shape = residual
    compact_size = compacted_shape[0] * compacted_shape[1]
    compacted_grad = (
        jnp.zeros((compact_size + 1, compacted_shape[2]), dtype=cotangent.dtype)
        .at[receiver_linear_indices]
        .set(jnp.where(receiver_keep[:, None], cotangent, 0), mode="drop")
    )
    result: jax.Array[[int, int, H]] = compacted_grad[:compact_size].reshape(compacted_shape)
    return result, None, None


_expand_compacted.defvjp(_expand_compacted_fwd, _expand_compacted_bwd)


def _in_band_expert_header(
    encoded_experts: jax.Array[[int]],
    *,
    expert_shards: int,
    pool_capacity: int,
    hidden_dim: int,
    dtype: jnp.dtype,
) -> jax.Array[[int, int, int]]:
    """Pack static expert IDs into full-width rows for the activation collective."""
    metadata_rows = math.ceil(pool_capacity / hidden_dim)
    padded_size = metadata_rows * hidden_dim
    encoded_experts_grouped: jax.Array[[int, int]] = encoded_experts.reshape(expert_shards, pool_capacity)
    padding = jnp.zeros((expert_shards, padded_size - pool_capacity), dtype=encoded_experts.dtype)
    return (
        jnp.concatenate([encoded_experts_grouped, padding], axis=1)
        .reshape(expert_shards, metadata_rows, hidden_dim)
        .astype(dtype)
    )


def _receiver_ranks(
    received_experts: jax.Array[[int]],
    *,
    local_experts: int,
) -> jax.Array[[int]]:
    expert_indicators = jax.nn.one_hot(received_experts, local_experts, dtype=jnp.int32)
    inclusive_counts = jnp.cumsum(expert_indicators, axis=0, dtype=jnp.int32)
    return jnp.sum((inclusive_counts - 1) * expert_indicators, axis=1, dtype=jnp.int32)


def _interleaved_receiver_ranks(
    received_experts: jax.Array[[int]],
    *,
    local_experts: int,
    expert_shards: int,
    pool_capacity: int,
) -> jax.Array[[int]]:
    """Per-expert receiver capacity ranks that fill capacity round-robin over source shards.

    The received buffer is source-shard-major (``[expert_shards, pool_capacity]``), so a plain cumsum
    ranks capacity in source order at every receiver -- the same (last-ordered) sender shard absorbs
    the overflow everywhere, a deterministic data-correlated drop bias. Transposing to pool-position
    order (``[pool_capacity, expert_shards]``) before the cumsum visits each source's slot ``p`` before
    any source's slot ``p+1``, so capacity is allocated round-robin across sources and no sender is
    truncated first. Transposing the ranks back restores the source-major layout the caller expects.

    The permutation is static (no ``axis_index``), the per-expert keep count stays ``min(count,
    capacity)``, and within any source pool position order is preserved -- so a later token never
    displaces an earlier one within a sequence (each expert shard holds whole sequences).
    """
    interleaved = received_experts.reshape(expert_shards, pool_capacity).T.reshape(-1)
    ranks = _receiver_ranks(interleaved, local_experts=local_experts)
    return ranks.reshape(pool_capacity, expert_shards).T.reshape(-1)


def _dispatch_pooled[Tlocal: IntVar, H: IntVar](
    *,
    x_local: jax.Array[[Tlocal, H]],
    local_expert_indices: jax.Array[[int]],
    destination_shards: jax.Array[[int]],
    pool_ranks: jax.Array[[int]],
    sender_keep: jax.Array[[int]],
    local_experts: int,
    expert_shards: int,
    pool_capacity: int,
    receiver_capacity: int,
    receiver_limit: jax.Array[[]],
    assignments_per_shard: int,
    topk: int,
) -> _PooledDispatch:
    tokens_per_shard, hidden_dim = x_local.shape
    send_size = expert_shards * pool_capacity
    sender_linear_indices = jnp.where(
        sender_keep,
        destination_shards * pool_capacity + pool_ranks,
        send_size,
    )
    assignment_sources = _assignment_sources(sender_linear_indices, send_size=send_size)
    token_sources = _token_sources(
        assignment_sources,
        assignments=assignments_per_shard,
        topk=topk,
        tokens=tokens_per_shard,
    )
    source_indices = jnp.minimum(assignment_sources, assignments_per_shard - 1)
    source_valid = assignment_sources < assignments_per_shard
    encoded_experts = jnp.where(source_valid, local_expert_indices[source_indices] + 1, 0).astype(jnp.int32)

    with jax.named_scope("dispatch"):
        header = _in_band_expert_header(
            encoded_experts,
            expert_shards=expert_shards,
            pool_capacity=pool_capacity,
            hidden_dim=hidden_dim,
            dtype=x_local.dtype,
        )
        metadata_rows = header.shape[1]
        payload = _pooled_dispatch_payload(
            x_local,
            header,
            token_sources,
            sender_linear_indices,
            sender_keep,
        )
        # `jax.custom_vjp` erases `_pooled_dispatch_payload`'s signature, so pin the received payload.
        received_payload: jax.Array[[int, int, H]] = jax.lax.all_to_all(
            payload,
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(expert_shards, metadata_rows + pool_capacity, hidden_dim)
        received_header = received_payload[:, :metadata_rows]
        received_x: jax.Array[[int, H]] = received_payload[:, metadata_rows:].reshape(send_size, hidden_dim)
        received_encoded_experts: jax.Array[[int]] = received_header.reshape(expert_shards, -1)[
            :, :pool_capacity
        ].reshape(send_size)
        received_experts = received_encoded_experts.astype(jnp.int32) - 1

        # Allocate receiver capacity round-robin over source shards (see `_interleaved_receiver_ranks`)
        # so no sender is systematically truncated first, removing the fixed-source-order overflow bias.
        receiver_ranks = _interleaved_receiver_ranks(
            received_experts,
            local_experts=local_experts,
            expert_shards=expert_shards,
            pool_capacity=pool_capacity,
        )
        receiver_valid = received_experts >= 0
        receiver_keep = receiver_valid & (receiver_ranks < receiver_limit)
        compact_size = local_experts * receiver_capacity
        receiver_linear_indices: jax.Array[[int]] = jnp.where(
            receiver_keep,
            received_experts * receiver_capacity + receiver_ranks,
            compact_size,
        )
        compacted_x = _compact_received(
            received_x,
            receiver_linear_indices,
            receiver_keep,
            local_experts,
            receiver_capacity,
        )

    receiver_dropped = jnp.sum(receiver_valid & ~receiver_keep, dtype=jnp.int32)
    return _PooledDispatch(
        compacted_x,
        receiver_linear_indices,
        receiver_keep,
        sender_linear_indices,
        sender_keep,
        assignment_sources,
        receiver_dropped,
    )


def _compute_pooled[Elocal: IntVar, H: IntVar, I2: IntVar, I: IntVar](
    dispatch: _PooledDispatch,
    *,
    moe_w13_local: jax.Array[[Elocal, H, I2]],
    moe_w2_local: jax.Array[[Elocal, I, H]],
    activation_fn: Callable[[jax.Array], jax.Array],
) -> _PooledOutput:
    with jax.named_scope("moe_up_down"):
        moe_dim = moe_w2_local.shape[1]
        hidden = jnp.einsum("erh,ehi->eri", dispatch.compacted_x, moe_w13_local)
        gate, up = jnp.split(hidden, [moe_dim], axis=-1)
        compacted_output = jnp.einsum("eri,eih->erh", activation_fn(gate) * up, moe_w2_local)

    return _PooledOutput(
        compacted_output,
        dispatch.receiver_linear_indices,
        dispatch.receiver_keep,
        dispatch.sender_linear_indices,
        dispatch.sender_keep,
        dispatch.assignment_sources,
        dispatch.receiver_dropped,
    )


def _combine_pooled[Tlocal: IntVar, K: IntVar, H: IntVar](
    output: _PooledOutput,
    *,
    combine_weights_local: jax.Array[[Tlocal, K]],
    expert_shards: int,
    pool_capacity: int,
) -> jax.Array[[Tlocal, H]]:
    send_size = expert_shards * pool_capacity
    hidden_dim = output.compacted_output.shape[-1]
    with jax.named_scope("combine"):
        received_output = _expand_compacted(
            output.compacted_output,
            output.receiver_linear_indices,
            output.receiver_keep,
        )
        returned = jax.lax.all_to_all(
            received_output.reshape(expert_shards, pool_capacity, hidden_dim),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(send_size, hidden_dim)
        return _combine_gather_sum(
            returned,
            output.sender_linear_indices,
            output.sender_keep,
            output.assignment_sources,
            combine_weights_local,
        )


def _moe_mlp_ep_fixed_pooled_wave_a2a_local[
    Tlocal: IntVar, K: IntVar, H: IntVar, Elocal: IntVar, I: IntVar, I2: IntVar
](
    x_local: jax.Array[[Tlocal, H]],
    selected_experts_local: jax.Array[[Tlocal, K]],
    combine_weights_local: jax.Array[[Tlocal, K]],
    token_valid_local: jax.Array[[Tlocal]],
    moe_w13_local: jax.Array[[Elocal, H, I2]],
    moe_w2_local: jax.Array[[Elocal, I, H]],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
    transport_capacity_factor: float,
    num_expert_waves: int,
) -> tuple[jax.Array[[Tlocal, H]], CapacityDrops]:
    """Stripe each destination pool over fixed waves and report drops at each transport stage."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")
    if transport_capacity_factor <= 0:
        raise ValueError("transport_capacity_factor must be positive")
    if num_expert_waves <= 0:
        raise ValueError(f"num_expert_waves must be positive, got {num_expert_waves}")
    if local_experts % num_expert_waves != 0:
        raise ValueError(
            f"local expert count={local_experts} must be divisible by num_expert_waves={num_expert_waves}"
        )

    tokens_per_shard, hidden_dim = x_local.shape
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    num_waves = num_expert_waves
    physical_pool_capacity = max(
        math.ceil(transport_capacity_factor * assignments_per_shard / (expert_shards * num_waves)),
        1,
    )
    physical_receiver_capacity = max(
        math.ceil(capacity_factor * assignments_per_shard / (local_experts * num_waves)),
        1,
    )

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    assignment_valid = _assignment_validity(token_valid_local, tokens=tokens_per_shard, topk=topk)
    valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
    safe_experts = jnp.where(assignment_valid, flat_experts, 0)
    local_expert_indices = (safe_experts % local_experts).astype(jnp.int32)
    destination_shards = (safe_experts // local_experts).astype(jnp.int32)
    logical_pool_capacity = _scaled_capacity(
        valid_assignments,
        capacity_factor=transport_capacity_factor,
        divisor=expert_shards * num_waves,
        maximum=physical_pool_capacity,
    )
    # One scalar psum sizes the receiver limit from the EP group's valid demand; it is
    # consumed only after each wave's all-to-all, so it does not gate the first transport.
    global_valid_assignments = jax.lax.psum(valid_assignments, "expert")
    logical_receiver_capacity = _scaled_capacity(
        global_valid_assignments,
        capacity_factor=capacity_factor,
        divisor=num_experts * num_waves,
        maximum=physical_receiver_capacity,
    )
    # The quotient keeps one logical capacity pool per destination. The remainder
    # stripes that pool over equal static waves.
    destination_ranks = _ranks_within_groups(
        destination_shards,
        num_groups=expert_shards,
        valid=assignment_valid,
    )
    assignment_waves = destination_ranks % num_waves
    pool_ranks = destination_ranks // num_waves
    sender_keep = assignment_valid & (pool_ranks < logical_pool_capacity)

    remat = partial(
        jax.checkpoint,
        prevent_cse=False,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    out_local = jnp.zeros((tokens_per_shard, hidden_dim), dtype=jnp.float32)
    receiver_dropped = jnp.array(0, dtype=jnp.int32)
    for wave_index in range(num_waves):
        wave_sender_keep = sender_keep & (assignment_waves == wave_index)
        dispatch = partial(
            _dispatch_pooled,
            x_local=x_local,
            local_expert_indices=local_expert_indices,
            destination_shards=destination_shards,
            pool_ranks=pool_ranks,
            sender_keep=wave_sender_keep,
            local_experts=local_experts,
            expert_shards=expert_shards,
            pool_capacity=physical_pool_capacity,
            receiver_capacity=physical_receiver_capacity,
            receiver_limit=logical_receiver_capacity,
            assignments_per_shard=assignments_per_shard,
            topk=topk,
        )
        compute = partial(
            _compute_pooled,
            moe_w13_local=moe_w13_local,
            moe_w2_local=moe_w2_local,
            activation_fn=activation_fn,
        )
        combine = partial(
            _combine_pooled,
            combine_weights_local=combine_weights_local,
            expert_shards=expert_shards,
            pool_capacity=physical_pool_capacity,
        )
        pooled_dispatch = dispatch()
        pooled_output = remat(compute)(pooled_dispatch)
        out_local = out_local + remat(combine)(pooled_output)
        receiver_dropped = receiver_dropped + pooled_dispatch.receiver_dropped

    sender_dropped = valid_assignments - jnp.sum(sender_keep, dtype=jnp.int32)
    dropped_by_stage_local = jnp.stack((sender_dropped, receiver_dropped))
    dropped_by_stage = jax.lax.psum(dropped_by_stage_local, token_sharding_axes)
    drops = CapacityDrops(sender_dropped=dropped_by_stage[0], receiver_dropped=dropped_by_stage[1])
    return out_local.astype(x_local.dtype), drops
