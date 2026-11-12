# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Destination-filtered expert-parallel Grug MoE backend.

Each shard sends every local token only to the expert shards that own one of its selected experts,
once per shard, through a fixed-capacity ``all_to_all``. The expert shard runs ring's grouped
``ragged_dot`` path on the received rows, sums each token's local expert outputs, and returns one
row per received token through a second ``all_to_all``. The sender then sums the returned rows.

Compared with ``ring``, which all-gathers every token to every shard and reduce-scatters a
full-size output, this moves only the rows a destination works on: with top-4 of 256 experts over
8 shards, about 42% of ring's bytes at a sender capacity factor of 1.0.

Every hidden-width data movement is a gather whose transpose is another gather, so neither the
forward nor the backward uses a hidden-width scatter-add. The routing-weight gradient comes from
the down projection's backward (``_down_project_combine``), so the backward never needs the expert
outputs: remat neither saves them nor recomputes the down projection.
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
    _CHECKPOINT_EXPERT_HIDDEN,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.ep_common import _assignment_sources, _combine_gather_sum_impl
from levanter.grug._moe.ep_ring import _combine_rows, _dispatch_rows, _select_local_assignments
from levanter.grug._moe.pallas_routing_metadata import MoeRoutingMetadata, expert_rows, send_slots


def destination_hit_fraction(*, num_experts: int, local_experts: int, topk: int) -> float:
    """Probability that a token's ``topk`` distinct uniform experts include one on a given shard."""
    return 1.0 - math.comb(num_experts - local_experts, topk) / math.comb(num_experts, topk)


@jax.custom_vjp
def _slot_weights(
    combine_weights: Float[Array, "T K"],
    safe_source: Int[Array, "S"],
    on_destination: Bool[Array, "S K"],
    assignment_slot: Int[Array, "T K"],
) -> Float[Array, "S K"]:
    """``where(on_destination, combine_weights[safe_source], 0)`` in float32.

    ``assignment_slot[t, k]`` is the send slot that carries token ``t``'s assignment ``k`` (or an
    out-of-range sentinel). Each kept assignment has exactly one such slot, so the backward is a
    gather through that map instead of autodiff's scatter-add over ``safe_source``, which is slow
    in bfloat16 on AMD GPUs.
    """
    return jnp.where(on_destination, combine_weights[safe_source].astype(jnp.float32), 0.0)


def _slot_weights_fwd(combine_weights, safe_source, on_destination, assignment_slot):
    out = _slot_weights(combine_weights, safe_source, on_destination, assignment_slot)
    # The zero-size array carries the input dtype the cotangent must have.
    return out, (assignment_slot, jnp.zeros((0,), combine_weights.dtype))


def _slot_weights_bwd(residual, cotangent):
    assignment_slot, dtype_carrier = residual
    topk = assignment_slot.shape[1]
    flat = assignment_slot * topk + jnp.arange(topk, dtype=jnp.int32)
    d_weights = jnp.take(cotangent.reshape(-1), flat, mode="fill", fill_value=0)
    return d_weights.astype(dtype_carrier.dtype), None, None, None


_slot_weights.defvjp(_slot_weights_fwd, _slot_weights_bwd)


@jax.custom_vjp
def _down_project_combine(
    hidden: Float[Array, "M I"],
    moe_w2: Float[Array, "Elocal I H"],
    group_sizes: Int[Array, "Elocal"],
    weights: Float[Array, "N J"],
    inv: Int[Array, "N J"],
    flat_idx: Int[Array, "M"],
) -> Float[Array, "N H"]:
    """``out[n] = sum_j weights[n, j] * y[inv[n, j]]`` with ``y = ragged_dot(hidden, moe_w2, group_sizes)``.

    ``flat_idx[m]`` is the flat ``n * J + j`` entry that reads row ``m`` (or a sentinel), so the
    backward for ``y`` is a gather.

    The backward takes the routing-weight gradient from the down projection's own backward
    instead of from ``y``, so ``y`` is never saved or recomputed. For ``m = inv[n, j]``,
    ``dy[m] = weights[n, j] * cotangent[n]`` and ``dhidden = dy @ moe_w2[e].T``, hence
    ``<hidden[m], dhidden[m]> = weights[n, j] * <y[m], cotangent[n]> = weights[n, j] * dweights[n, j]``.
    The gradient is that row sum, in float32, divided by the weight; a zero weight gets zero.
    """
    y = ragged_dot(hidden, moe_w2, group_sizes)
    return _combine_gather_sum_impl(y, inv.reshape(-1), weights, y.dtype)


def _down_project_combine_fwd(hidden, moe_w2, group_sizes, weights, inv, flat_idx):
    y = ragged_dot(hidden, moe_w2, group_sizes)
    out = _combine_gather_sum_impl(y, inv.reshape(-1), weights, y.dtype)
    return out, (hidden, moe_w2, group_sizes, weights, inv, flat_idx)


def _down_project_combine_bwd(residual, cotangent):
    hidden, moe_w2, group_sizes, weights, inv, flat_idx = residual
    num_rows, num_slots = weights.shape
    flat_valid = flat_idx < num_rows * num_slots
    safe_flat = jnp.minimum(flat_idx, num_rows * num_slots - 1)
    row_weight = jnp.where(flat_valid, weights.reshape(-1)[safe_flat].astype(jnp.float32), 0.0)
    d_y = (cotangent[safe_flat // num_slots].astype(jnp.float32) * row_weight[:, None]).astype(hidden.dtype)

    _, down_vjp = jax.vjp(lambda h, w: ragged_dot(h, w, group_sizes), hidden, moe_w2)
    d_hidden, d_w2 = down_vjp(d_y)

    weighted_score = jnp.sum(hidden.astype(jnp.float32) * d_hidden.astype(jnp.float32), axis=-1)
    nonzero = row_weight != 0
    row_score = jnp.where(nonzero, weighted_score / jnp.where(nonzero, row_weight, 1.0), 0.0)
    d_weights = jnp.take(row_score, inv, mode="fill", fill_value=0).astype(weights.dtype)
    return d_hidden, d_w2, None, d_weights, None, None


_down_project_combine.defvjp(_down_project_combine_fwd, _down_project_combine_bwd)


def _moe_mlp_ep_ring_filtered_local(
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
    routing_metadata: MoeRoutingMetadata = "xla",
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
    """Filtered all-to-all dispatch, ring's grouped expert MLP, expert-side combine.

    ``transport_capacity_factor`` limits each (sender, destination) pair to a multiple of the
    distinct tokens the sender would send that destination under uniform routing,
    ``destination_hit_fraction`` times its valid tokens; the send buffer holds the limit for
    ``Tlocal`` valid tokens. A token past the limit loses all its assignments to the destination
    (sender drops). The limit is an expectation for uniform routing, with no worst-case
    guarantee: when routing concentrates on a destination, its tokens past the limit drop even if
    the destination has expert capacity to spare, so a factor that drops nothing on one workload
    can drop on another. ``capacity_factor`` keeps ring's meaning: each expert shard
    runs at most ``capacity_factor * Tlocal * K`` assignments, allocated in local-expert order
    (receiver drops).

    ``routing_metadata`` picks how the integer routing maps (send slots, expert rows, group sizes, drop counts) are
    built from the top-k expert ids: XLA fusions ("xla") or the Pallas-Triton counting sorts of
    ``pallas_routing_metadata`` ("pallas"), with the same values.
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
        slot, safe_source, slot_valid, metadata, sender_dropped = _send_routing(
            selected_experts_local,
            token_valid_local,
            combine_weights_local,
            local_experts=local_experts,
            destinations=ep_size,
            pair_capacity=pair_capacity,
            pair_capacity_factor=transport_capacity_factor * hit_fraction,
            routing_metadata=routing_metadata,
        )
        received_metadata = jax.lax.all_to_all(metadata, "expert", split_axis=0, concat_axis=0, tiled=True)
        received_metadata = tree_checkpoint_name(received_metadata, _CHECKPOINT_DISPATCH_INPUT)

        send_x = _dispatch_rows(x_local, safe_source, slot_valid, slot)
        received_x = jax.lax.all_to_all(
            send_x.reshape(ep_size, pair_capacity, hidden_dim), "expert", split_axis=0, concat_axis=0, tiled=True
        ).reshape(send_rows, hidden_dim)

    with jax.named_scope("expert_dispatch"):
        received_weights, group_sizes, receiver_dropped, picked, dispatch_valid, dispatch_row, position = (
            _expert_routing(
                received_metadata,
                local_experts=local_experts,
                capacity_factor=capacity_factor,
                expert_capacity=expert_capacity,
                routing_metadata=routing_metadata,
            )
        )
        x_dispatch = _dispatch_rows(received_x, jnp.minimum(dispatch_row, send_rows - 1), dispatch_valid, position)
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        expert_combined = _down_project_combine(
            activation_fn(gate) * up, moe_w2_local, group_sizes, received_weights, position, picked
        )

    with jax.named_scope("combine"):
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


def _send_routing(
    selected_experts_local: Int[Array, "T K"],
    token_valid_local: Bool[Array, "T"],
    combine_weights_local: Float[Array, "T K"],
    *,
    local_experts: int,
    destinations: int,
    pair_capacity: int,
    pair_capacity_factor: float,
    routing_metadata: MoeRoutingMetadata,
) -> tuple[Int[Array, "T D"], Int[Array, "R"], Bool[Array, "R"], Float[Array, "D C1 K2"], Int[Array, ""]]:
    """The sender's routing maps and the metadata it sends.

    Returns each token's send row per destination (``slot``), each send row's token (clamped) and validity, the f32
    [destinations, C + 1, 2K] metadata buffer for the all-to-all (combine weights, local expert ids or -1, and a
    trailer row carrying this shard's valid demand) and the sender drops.
    """
    tokens, topk = selected_experts_local.shape
    send_rows = destinations * pair_capacity
    selected = selected_experts_local.astype(jnp.int32)
    destination = selected // local_experts
    valid_tokens = jnp.sum(token_valid_local, dtype=jnp.int32)
    logical_pair_capacity = _scaled_capacity(valid_tokens, capacity_factor=pair_capacity_factor, maximum=pair_capacity)
    assignment_valid = jnp.broadcast_to(token_valid_local[:, None], selected.shape)
    valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
    if routing_metadata == "pallas":
        send_maps = send_slots(
            selected,
            token_valid_local,
            logical_pair_capacity,
            local_experts=local_experts,
            destinations=destinations,
            pair_capacity=pair_capacity,
        )
        slot, source = send_maps.slot, send_maps.source
        sender_dropped = valid_assignments - send_maps.kept_assignments
    else:
        shard_ids = jnp.arange(destinations, dtype=jnp.int32)
        hit = jnp.any(destination[:, :, None] == shard_ids, axis=1) & token_valid_local[:, None]
        rank = jnp.cumsum(hit, axis=0, dtype=jnp.int32) - 1
        keep = hit & (rank < logical_pair_capacity)
        # slot[t, d]: the token's row in the flat [ep, C] send buffer, or the `send_rows` sentinel.
        slot = jnp.where(keep, shard_ids * pair_capacity + rank, send_rows)
        token_ids = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32)[:, None], slot.shape)
        source = jnp.full((send_rows,), tokens, jnp.int32).at[slot.reshape(-1)].set(token_ids.reshape(-1), mode="drop")
        sent_assignments = jnp.sum(jnp.take_along_axis(keep, destination, axis=1) & assignment_valid, dtype=jnp.int32)
        sender_dropped = valid_assignments - sent_assignments

    slot_valid = source < tokens
    safe_source = jnp.minimum(source, tokens - 1)
    # Per send row: the token's combine weights and local expert ids on that destination.
    slot_destination = jnp.arange(send_rows, dtype=jnp.int32) // pair_capacity
    slot_selected = selected[safe_source]
    on_destination = slot_valid[:, None] & (slot_selected // local_experts == slot_destination[:, None])
    slot_local_ids = jnp.where(on_destination, slot_selected - slot_destination[:, None] * local_experts, -1)
    # assignment_slot[t, k]: the send slot carrying assignment (t, k), or the `send_rows` sentinel.
    assignment_slot = jnp.take_along_axis(slot, destination, axis=1)
    slot_weights = _slot_weights(combine_weights_local, safe_source, on_destination, assignment_slot)
    metadata = jnp.concatenate([slot_weights, slot_local_ids.astype(jnp.float32)], axis=1)
    metadata = metadata.reshape(destinations, pair_capacity, 2 * topk)
    # One trailer row per destination carries this shard's valid demand, which the receiver
    # sums into ring's logical capacity without a separate psum.
    trailer = (
        jnp.zeros((destinations, 1, 2 * topk), jnp.float32).at[:, 0, 0].set(valid_assignments.astype(jnp.float32))
    )
    metadata = jnp.concatenate([metadata, trailer], axis=1)
    return slot, safe_source, slot_valid, metadata, sender_dropped


def _expert_routing(
    received_metadata: Float[Array, "D C1 K2"],
    *,
    local_experts: int,
    capacity_factor: float,
    expert_capacity: int,
    routing_metadata: MoeRoutingMetadata,
) -> tuple[
    Float[Array, "R K"],
    Int[Array, "Elocal"],
    Int[Array, ""],
    Int[Array, "P"],
    Bool[Array, "P"],
    Int[Array, "P"],
    Int[Array, "R K"],
]:
    """The receiver's routing maps from the received metadata.

    Returns the received combine weights, the ``ragged_dot`` group sizes (as ``_select_local_assignments`` gives them),
    the receiver drops, the received assignment in each expert row (``picked``, the assignment count for padding
    rows), which expert rows are valid, each expert row's received row (``dispatch_row``, the received row count for
    padding rows) and each received assignment's expert row (``position``, ``expert_capacity`` for none).
    """
    ep_size, pair_capacity = received_metadata.shape[0], received_metadata.shape[1] - 1
    topk = received_metadata.shape[2] // 2
    send_rows = ep_size * pair_capacity
    received_assignments = send_rows * topk
    group_valid_assignments = jnp.sum(received_metadata[:, pair_capacity, 0]).astype(jnp.int32)
    received_weights = received_metadata[:, :pair_capacity, :topk].reshape(send_rows, topk)
    logical_capacity = _scaled_capacity(
        group_valid_assignments,
        capacity_factor=capacity_factor,
        divisor=ep_size,
        minimum=local_experts,
        maximum=expert_capacity,
    )
    if routing_metadata == "pallas":
        rows = expert_rows(
            received_metadata, logical_capacity, local_experts=local_experts, physical_capacity=expert_capacity
        )
        accepted_total = jnp.sum(rows.group_sizes, dtype=jnp.int32)
        dispatch_valid = jnp.arange(expert_capacity, dtype=jnp.int32) < accepted_total
        # Like _select_local_assignments: the padding rows past the accepted ones count toward the last expert.
        group_sizes = rows.group_sizes.at[-1].add(expert_capacity - accepted_total)
        receiver_dropped, picked, position = rows.dropped, rows.picked, rows.position
    else:
        received_ids = jax.lax.stop_gradient(received_metadata[:, :pair_capacity, topk:])
        received_ids = jnp.round(received_ids).astype(jnp.int32).reshape(received_assignments)
        assignment_mask = received_ids >= 0
        local_expert = jnp.where(assignment_mask, received_ids, 0)
        group_sizes, receiver_dropped, picked, dispatch_valid = _select_local_assignments(
            local_expert,
            assignment_mask,
            local_experts=local_experts,
            logical_capacity=logical_capacity,
            physical_capacity=expert_capacity,
        )
        picked = jnp.where(dispatch_valid, picked, received_assignments)
        position = _assignment_sources(picked, send_size=received_assignments)
    dispatch_row = jnp.where(dispatch_valid, picked // topk, send_rows)
    # position[r, j]: the dispatch row of received row r's j-th assignment, or the sentinel.
    position = position.reshape(send_rows, topk)
    return received_weights, group_sizes, receiver_dropped, picked, dispatch_valid, dispatch_row, position
