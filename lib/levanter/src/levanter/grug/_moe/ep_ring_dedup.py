# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-deduplicated expert-parallel Grug MoE backend.

Each shard sends every local token at most once to each expert shard that owns one of its
selected experts, through a fixed-capacity ``all_to_all``. The expert shard runs ring's grouped
``ragged_dot`` path on the received rows, sums each token's local expert outputs, and returns one
row per received token through a second ``all_to_all``. The sender then sums the returned rows.

Compared with ``ring``, which all-gathers every token to every shard and reduce-scatters a
full-size output, this moves only the rows a destination works on: with top-4 of 256 experts over
8 shards, about 42% of ring's bytes at a sender capacity factor of 1.0.

Every hidden-width data movement is a gather whose transpose is another gather, so neither the
forward nor the backward uses a hidden-width scatter-add.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug._moe.ep_ring import _combine_rows, _dispatch_rows


def destination_hit_fraction(*, num_experts: int, local_experts: int, topk: int) -> float:
    """Probability that a token's ``topk`` distinct uniform experts include one on a given shard."""
    return 1.0 - math.comb(num_experts - local_experts, topk) / math.comb(num_experts, topk)


def _weighted_gather_sum_impl(src, weights, inv):
    out = jnp.zeros((inv.shape[0], src.shape[1]), jnp.float32)
    for j in range(inv.shape[1]):
        rows = jnp.take(src, inv[:, j], axis=0, mode="fill", fill_value=0).astype(jnp.float32)
        out = out + rows * weights[:, j, None].astype(jnp.float32)
    return out.astype(src.dtype)


@jax.custom_vjp
def _weighted_gather_sum_rows(
    src: Float[Array, "M H"],
    weights: Float[Array, "N J"],
    inv: Int[Array, "N J"],
    flat_idx: Int[Array, "M"],
) -> Float[Array, "N H"]:
    """``out[n] = sum_j weights[n, j] * src[inv[n, j]]``.

    ``flat_idx[m]`` is the flat ``n * J + j`` entry that reads row ``m`` (or a sentinel), so the
    backward for ``src`` is a gather.
    """
    return _weighted_gather_sum_impl(src, weights, inv)


def _weighted_gather_sum_rows_fwd(src, weights, inv, flat_idx):
    return _weighted_gather_sum_impl(src, weights, inv), (src, weights, inv, flat_idx)


def _weighted_gather_sum_rows_bwd(residual, cotangent):
    src, weights, inv, flat_idx = residual
    num_rows, num_slots = weights.shape
    flat_valid = flat_idx < num_rows * num_slots
    safe_flat = jnp.minimum(flat_idx, num_rows * num_slots - 1)
    row_weight = weights.reshape(-1)[safe_flat].astype(jnp.float32)
    d_src = cotangent[safe_flat // num_slots].astype(jnp.float32) * row_weight[:, None]
    d_src = jnp.where(flat_valid[:, None], d_src, 0).astype(src.dtype)

    ct32 = cotangent.astype(jnp.float32)
    d_weights = jnp.stack(
        [
            jnp.sum(jnp.take(src, inv[:, j], axis=0, mode="fill", fill_value=0).astype(jnp.float32) * ct32, axis=-1)
            for j in range(num_slots)
        ],
        axis=1,
    ).astype(weights.dtype)
    return d_src, d_weights, None, None


_weighted_gather_sum_rows.defvjp(_weighted_gather_sum_rows_fwd, _weighted_gather_sum_rows_bwd)


def _moe_mlp_ep_ring_dedup_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    token_valid_local: Bool[Array, "Tlocal"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    transport_capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
    """Deduplicated all-to-all dispatch, ring's grouped expert MLP, expert-side combine.

    ``transport_capacity_factor`` sizes each (sender, destination) block as a multiple of the
    distinct tokens a shard sends one destination under uniform routing,
    ``destination_hit_fraction * Tlocal``. A token past that limit loses all its assignments to
    the destination (sender drops). ``capacity_factor`` keeps ring's meaning: each expert shard
    runs at most ``capacity_factor * Tlocal * K`` assignments, allocated in local-expert order
    (receiver drops).
    """
    tokens, topk = selected_experts_local.shape
    hidden_dim = x_local.shape[1]
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")
    if transport_capacity_factor <= 0:
        raise ValueError("transport_capacity_factor must be positive")
    ep_size = num_experts // local_experts

    hit_fraction = destination_hit_fraction(num_experts=num_experts, local_experts=local_experts, topk=topk)
    pair_capacity = min(tokens, max(1, math.ceil(transport_capacity_factor * hit_fraction * tokens)))
    send_rows = ep_size * pair_capacity
    received_assignments = send_rows * topk
    expert_capacity = min(received_assignments, max(local_experts, math.ceil(capacity_factor * tokens * topk)))

    with jax.named_scope("dispatch"):
        selected = selected_experts_local.astype(jnp.int32)
        destination = selected // local_experts
        shard_ids = jnp.arange(ep_size, dtype=jnp.int32)
        hit = jnp.any(destination[:, :, None] == shard_ids, axis=1) & token_valid_local[:, None]
        rank = jnp.cumsum(hit, axis=0, dtype=jnp.int32) - 1
        valid_tokens = jnp.sum(token_valid_local, dtype=jnp.int32)
        logical_pair_capacity = _scaled_capacity(
            valid_tokens,
            capacity_factor=transport_capacity_factor * hit_fraction,
            maximum=pair_capacity,
        )
        keep = hit & (rank < logical_pair_capacity)
        # slot[t, d]: the token's row in the flat [ep, C] send buffer, or the `send_rows` sentinel.
        slot = jnp.where(keep, shard_ids * pair_capacity + rank, send_rows)
        token_ids = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32)[:, None], slot.shape)
        source = jnp.full((send_rows,), tokens, jnp.int32).at[slot.reshape(-1)].set(token_ids.reshape(-1), mode="drop")

        assignment_valid = jnp.broadcast_to(token_valid_local[:, None], selected.shape)
        valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
        sent_assignments = jnp.sum(jnp.take_along_axis(keep, destination, axis=1) & assignment_valid, dtype=jnp.int32)
        sender_dropped = valid_assignments - sent_assignments

        # Per send row: the token's combine weights and local expert ids on that destination.
        slot_valid = source < tokens
        safe_source = jnp.minimum(source, tokens - 1)
        slot_destination = jnp.arange(send_rows, dtype=jnp.int32) // pair_capacity
        slot_selected = selected[safe_source]
        on_destination = slot_valid[:, None] & (slot_selected // local_experts == slot_destination[:, None])
        slot_local_ids = jnp.where(on_destination, slot_selected - slot_destination[:, None] * local_experts, -1)
        slot_weights = jnp.where(on_destination, combine_weights_local[safe_source].astype(jnp.float32), 0.0)
        metadata = jnp.concatenate([slot_weights, slot_local_ids.astype(jnp.float32)], axis=1)
        metadata = metadata.reshape(ep_size, pair_capacity, 2 * topk)
        # One trailer row per destination carries this shard's valid demand, which the receiver
        # sums into ring's logical capacity without a separate psum.
        trailer = jnp.zeros((ep_size, 1, 2 * topk), jnp.float32).at[:, 0, 0].set(valid_assignments.astype(jnp.float32))
        metadata = jnp.concatenate([metadata, trailer], axis=1)

        received_metadata = jax.lax.all_to_all(metadata, "expert", split_axis=0, concat_axis=0, tiled=True)
        received_metadata = tree_checkpoint_name(received_metadata, _CHECKPOINT_DISPATCH_INPUT)

        send_x = _dispatch_rows(x_local, safe_source, slot_valid, slot)
        received_x = jax.lax.all_to_all(
            send_x.reshape(ep_size, pair_capacity, hidden_dim), "expert", split_axis=0, concat_axis=0, tiled=True
        ).reshape(send_rows, hidden_dim)

    with jax.named_scope("expert_dispatch"):
        group_valid_assignments = jnp.sum(received_metadata[:, pair_capacity, 0]).astype(jnp.int32)
        received_weights = received_metadata[:, :pair_capacity, :topk].reshape(send_rows, topk)
        received_ids = jax.lax.stop_gradient(received_metadata[:, :pair_capacity, topk:])
        received_ids = jnp.round(received_ids).astype(jnp.int32).reshape(received_assignments)
        assignment_mask = received_ids >= 0
        local_expert = jnp.where(assignment_mask, received_ids, 0)

        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        counts = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]) & assignment_mask[:, None],
            axis=0,
            dtype=jnp.int32,
        )
        logical_capacity = _scaled_capacity(
            group_valid_assignments,
            capacity_factor=capacity_factor,
            divisor=ep_size,
            minimum=local_experts,
            maximum=expert_capacity,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=logical_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        receiver_dropped = jnp.sum(counts, dtype=jnp.int32) - accepted_total

        # Same selection as ring: (local expert, arrival position) order, top `expert_capacity`.
        flat_position = jnp.arange(received_assignments, dtype=jnp.int32)
        order_key = local_expert * received_assignments + flat_position
        selection_key = jnp.where(assignment_mask, local_experts * received_assignments - order_key, -1)
        _, picked = jax.lax.top_k(selection_key, expert_capacity)
        dispatch_valid = jnp.arange(expert_capacity, dtype=jnp.int32) < accepted_total
        picked = jnp.where(dispatch_valid, picked, received_assignments)
        dispatch_row = jnp.where(dispatch_valid, picked // topk, send_rows)
        # position[r, j]: the dispatch row of received row r's j-th assignment, or the sentinel.
        position = (
            jnp.full((received_assignments,), expert_capacity, jnp.int32)
            .at[picked]
            .set(jnp.arange(expert_capacity, dtype=jnp.int32), mode="drop")
            .reshape(send_rows, topk)
        )

        x_dispatch = _dispatch_rows(received_x, jnp.minimum(dispatch_row, send_rows - 1), dispatch_valid, position)
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        # Padding rows go to the last expert group, as in ring.
        group_sizes = accepted_counts.at[-1].add(expert_capacity - accepted_total)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("combine"):
        expert_combined = _weighted_gather_sum_rows(out_dispatch, received_weights, position, picked)
        returned = jax.lax.all_to_all(
            expert_combined.reshape(ep_size, pair_capacity, hidden_dim),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(send_rows, hidden_dim)
        out_local = _combine_rows(returned, safe_source, slot_valid, slot).astype(x_local.dtype)
        drops = jax.lax.psum(jnp.stack([sender_dropped, receiver_dropped]), token_sharding_axes)
    return out_local, CapacityDrops(sender_dropped=drops[0], receiver_dropped=drops[1])
