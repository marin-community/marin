# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical compact Grug MoE kernels.

Implementation overview:
- Routing keeps the argsort-grouped dispatch path that emerged as the stable
  default from https://github.com/marin-community/marin/issues/2704 and commit
  89318a910 (and its parent).
- Expert parallelism keeps the ring-style strategy from
  https://github.com/marin-community/marin/issues/2710: token-sharded
  `all_gather` for dispatch, then `psum_scatter` for collection.
- This module intentionally provides functional kernels only; model/module
  wiring lives in the Grug model files.
"""

import math

from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias, cast, get_args

import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call, tree_checkpoint_name
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh, get_mesh, reshard
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.25
# #2710 used 1.25 as the practical EP ring default to avoid over/under-packing.

MoeActivation: TypeAlias = ActivationFunctionEnum | Callable[[jax.Array], jax.Array]
MoeImplementation: TypeAlias = Literal["ring", "ragged_all_to_all"]
_VALID_MOE_IMPLEMENTATIONS = get_args(MoeImplementation)


def _current_mesh() -> Mesh | jax.sharding.AbstractMesh:
    try:
        mesh = get_mesh()
    except ValueError:
        mesh = None
    if mesh is not None and not mesh.empty:
        return mesh
    return get_abstract_mesh()


def _mesh_has_axis(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> bool:
    if mesh is None or mesh.empty:
        return False
    return axis_name in mesh.shape


def _mesh_axis_size(mesh: Mesh | jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        return 1
    return int(mesh.shape.get(axis_name, 1))


def _batch_spec(mesh: Mesh | jax.sharding.AbstractMesh | None) -> P:
    if _mesh_has_axis(mesh, "expert"):
        return P(("data", "expert"))
    return P(("data",))


def resolve_moe_implementation(
    implementation: MoeImplementation | str | None,
    mesh: jax.sharding.AbstractMesh | None,
) -> MoeImplementation:
    if implementation is not None:
        if implementation not in _VALID_MOE_IMPLEMENTATIONS:
            valid = ", ".join(repr(choice) for choice in _VALID_MOE_IMPLEMENTATIONS)
            raise ValueError(f"implementation must be one of {valid} or None, got {implementation!r}")
        return cast(MoeImplementation, implementation)

    return "ring"


@named_call
def _prepare_moe_dispatch(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    num_experts: int,
) -> tuple[
    Float[Array, "TK D"],
    Float[Array, "TK"],
    Int[Array, "TK"],
    Int[Array, "E"],
]:
    """Flatten + argsort by expert into grouped layout for GMM."""
    # #2704: keep argsort-grouped dispatch as the canonical compact routing
    # strategy, matching the behavior carried forward from 89318a910.
    tokens, topk = selected_experts.shape
    expert_ids = selected_experts.reshape(tokens * topk)
    dispatch_weights = combine_weights.reshape(tokens * topk)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    token_ids = jnp.arange(tokens * topk, dtype=jnp.int32) // topk
    token_ids_sort = token_ids[sort_idx]
    x_sort = x[token_ids_sort]
    w_sort = dispatch_weights[sort_idx].astype(x.dtype)
    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return x_sort, w_sort, token_ids_sort, group_sizes


def _moe_mlp_local(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Per-shard non-EP MoE FFN path with argsort routing + grouped matmul."""
    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        num_experts=num_experts,
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, "grug_moe_dispatch_input")

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), "grug_moe_expert_hidden")
        moe_dim = moe_w2.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes),
            "grug_moe_dispatch_output",
        )

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, jnp.array(0, dtype=jnp.int32)


def _batch_spec_from_x(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return P(spec[0])
    return _batch_spec(mesh)


def _is_replicated_spec(spec: P) -> bool:
    return all(axis is None for axis in spec)


def _value_spec_or_default(x: jax.Array, default: P, *, replace_replicated: bool = False) -> P:
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and not (replace_replicated and _is_replicated_spec(spec)):
        return spec
    return default


def _reshard_for_shard_map(x: jax.Array, mesh: Mesh | jax.sharding.AbstractMesh | None, spec: P) -> jax.Array:
    if mesh is not None and not mesh.empty:
        return reshard(x, NamedSharding(mesh, spec))
    return x


def _sort_activations(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    if inputs.shape[0] != sort_indices.shape[0]:
        raise ValueError(f"Expected matching leading dims, got {inputs.shape[0]} and {sort_indices.shape[0]}")
    return _sort_activations_custom(inputs, sort_indices)


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
    sort_indices = residuals
    return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


def _prefix_cap_counts(counts: Int[Array, "E"], *, capacity: int) -> Int[Array, "E"]:
    accepted = []
    remaining = jnp.array(capacity, dtype=jnp.int32)
    for expert in range(int(counts.shape[0])):
        take = jnp.minimum(counts[expert], remaining)
        accepted.append(take)
        remaining = jnp.maximum(remaining - take, 0)
    return jnp.stack(accepted, axis=0)


def _permute_by_global_expert(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    *,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    topk = selected_experts_local.shape[1]
    flat_selected = selected_experts_local.reshape(-1)
    sorted_indices = jnp.argsort(flat_selected)
    repeated_x = jnp.repeat(x_local, topk, axis=0)
    sorted_x = _sort_activations(repeated_x, sorted_indices)
    group_sizes = jnp.bincount(flat_selected, length=num_experts).astype(jnp.int32)
    return sorted_x, sorted_indices, group_sizes


def _unpermute_from_global_expert(
    intermediate: jax.Array,
    sorted_indices: jax.Array,
    combine_weights_local: jax.Array,
    *,
    tokens_per_shard: int,
    topk: int,
) -> jax.Array:
    unsorted = _sort_activations(intermediate, jnp.argsort(sorted_indices))
    reshaped = unsorted.reshape(tokens_per_shard, topk, -1)
    return jnp.einsum(
        "tkd,tk->td", reshaped, combine_weights_local.astype(reshaped.dtype), preferred_element_type=jnp.float32
    )


def _shard_a2a_params(
    shard_counts: jax.Array,
    shard_id: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    row = shard_counts[shard_id]
    input_offsets = jnp.cumsum(jnp.concatenate((jnp.array([0], dtype=row.dtype), row[:-1])))
    send_sizes = row

    recv_sizes = shard_counts[:, shard_id]
    # `ragged_all_to_all` expects sender-side output offsets: for each
    # destination shard, where this sender's slice should land in the remote
    # receiver buffer. JAX computes the local receive offsets by transposing
    # these offsets with an internal all_to_all.
    sender_output_offsets = jnp.cumsum(shard_counts, axis=0, dtype=shard_counts.dtype) - shard_counts
    output_offsets = sender_output_offsets[shard_id]
    return input_offsets, send_sizes, output_offsets, recv_sizes


def _local_permute_from_counts(
    inputs: jax.Array,
    global_group_sizes: jax.Array,
    *,
    local_expert_size: int,
    shard_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        start_index=shard_index * local_expert_size,
        slice_size=local_expert_size,
        axis=1,
    )
    local_group_sizes = jnp.sum(all_shard_local_sizes, axis=0)
    local_sizes = all_shard_local_sizes.reshape(-1)
    total_valid = jnp.sum(local_sizes, dtype=jnp.int32)
    segment_ends = jnp.cumsum(local_sizes, dtype=jnp.int32)
    positions = jnp.arange(inputs.shape[0], dtype=jnp.int32)
    segment_index = jnp.searchsorted(segment_ends, positions, side="right")
    local_expert_ids = jnp.where(positions < total_valid, segment_index % local_expert_size, local_expert_size)
    sorted_indices = jnp.argsort(local_expert_ids)
    sorted_inputs = _sort_activations(inputs, sorted_indices)
    sorted_inputs = jnp.where((positions < total_valid)[:, None], sorted_inputs, 0)
    group_sizes = local_group_sizes.at[-1].add(inputs.shape[0] - total_valid)
    return sorted_inputs, sorted_indices, group_sizes


def _clip_receiver_group_sizes(
    global_group_sizes: Int[Array, "S E"],
    *,
    local_expert_size: int,
    receiver_capacity: int,
) -> Int[Array, "S E"]:
    """Clip sender->expert group sizes so each receiver shard stays within capacity."""
    num_senders = int(global_group_sizes.shape[0])
    num_experts = int(global_group_sizes.shape[1])
    if num_experts % local_expert_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local_expert_size={local_expert_size}")
    num_receivers = num_experts // local_expert_size
    if num_receivers != num_senders:
        raise ValueError(f"sender/receiver shard mismatch: num_senders={num_senders}, num_receivers={num_receivers}")

    clipped_by_receiver: list[jax.Array] = []
    for receiver_index in range(num_receivers):
        start = receiver_index * local_expert_size
        stop = start + local_expert_size
        receiver_counts = global_group_sizes[:, start:stop]
        receiver_totals = jnp.sum(receiver_counts, axis=0, dtype=jnp.int32)
        accepted_totals = _prefix_cap_counts(receiver_totals, capacity=receiver_capacity)
        remaining = accepted_totals
        accepted_rows: list[jax.Array] = []
        for sender_index in range(num_senders):
            # Greedy first-sender-wins: earlier shards get priority when capacity is scarce.
            accepted = jnp.minimum(receiver_counts[sender_index], remaining)
            accepted_rows.append(accepted)
            remaining = remaining - accepted
        clipped_by_receiver.append(jnp.stack(accepted_rows, axis=0))

    return jnp.concatenate(clipped_by_receiver, axis=1)


def _expert_prefix_keep_mask(
    group_sizes: Int[Array, "E"],
    accepted_group_sizes: Int[Array, "E"],
    *,
    total_size: int,
) -> Bool[Array, "T"]:
    segment_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    segment_starts = jnp.concatenate((jnp.array([0], dtype=segment_ends.dtype), segment_ends[:-1]))
    positions = jnp.arange(total_size, dtype=jnp.int32)
    expert_index = jnp.searchsorted(segment_ends, positions, side="right")
    # Explicitly clip overflow positions to the last segment rather than
    # depending on implicit out-of-bounds `jnp.take` behavior. Those clipped
    # positions will have local_rank >= accepted, so they are masked out.
    expert_index = jnp.minimum(expert_index, group_sizes.shape[0] - 1)
    local_rank = positions - segment_starts[expert_index]
    accepted = accepted_group_sizes[expert_index]
    return local_rank < accepted


def _compact_by_keep_mask(inputs: jax.Array, keep_mask: Bool[Array, "T"]) -> jax.Array:
    total_size = inputs.shape[0]
    positions = jnp.arange(total_size, dtype=jnp.int32)
    sort_key = jnp.where(keep_mask, positions, positions + total_size)
    compacted = _sort_activations(inputs, jnp.argsort(sort_key))
    valid = positions < jnp.sum(keep_mask.astype(jnp.int32), dtype=jnp.int32)
    return jnp.where(valid[:, None], compacted, 0)


def _expand_from_keep_mask(compacted: jax.Array, keep_mask: Bool[Array, "T"]) -> jax.Array:
    keep_i32 = keep_mask.astype(jnp.int32)
    compact_index = jnp.cumsum(keep_i32, dtype=jnp.int32) - 1
    gathered = jnp.take(compacted, jnp.maximum(compact_index, 0), axis=0)
    return jnp.where(keep_mask[:, None], gathered, 0)


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    # #2710 ring EP strategy: gather tokens and their selected-expert routing
    # assignments across expert shards, then psum-scatter back to local tokens.
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

        local_experts = moe_w13_local.shape[0]
        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )

        ep_size = num_experts // local_experts
        local_capacity = int(math.ceil(capacity_factor * assignments / ep_size))
        local_capacity = max(local_experts, local_capacity)

        expert_axis = jax.lax.axis_index("expert")
        expert_start = expert_axis * local_experts
        local_expert = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)

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
        accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
        valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
        _, local_idx = jax.lax.top_k(selection_key, local_capacity)

        token_local = jnp.floor_divide(local_idx, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_take = jnp.take(x_global, token_local, axis=0)
        x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, "grug_moe_dispatch_input")
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    group_sizes = accepted_counts
    # `local_idx` pads by appending invalid rows at the end; keep GMM segment
    # boundaries aligned by attributing padding to the final expert segment.
    group_sizes = group_sizes.at[-1].add(local_capacity - jnp.sum(group_sizes, dtype=jnp.int32))

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), "grug_moe_expert_hidden")
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            "grug_moe_dispatch_output",
        )

    with jax.named_scope("scatter"):
        out_global = jnp.zeros_like(x_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")
        # #2710 ring EP strategy: collect only this shard's token slice after
        # reducing contributions from experts across the EP mesh.
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local, dropped_total


def _moe_mlp_ep_ragged_a2a_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[jax.Array, jax.Array]:
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    local_capacity = int(math.ceil(capacity_factor * assignments_per_shard))
    local_capacity = max(local_experts, local_capacity)
    recv_capacity = local_capacity

    with jax.named_scope("dispatch"):
        sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
        )
        all_group_sizes = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")
        clipped_group_sizes = _clip_receiver_group_sizes(
            all_group_sizes,
            local_expert_size=local_experts,
            receiver_capacity=local_capacity,
        )
        sender_group_sizes = clipped_group_sizes[shard_id]
        keep_mask = _expert_prefix_keep_mask(
            group_sizes.astype(jnp.int32),
            sender_group_sizes,
            total_size=assignments_per_shard,
        )
        sorted_x = _compact_by_keep_mask(sorted_x, keep_mask)

        all_shard_counts = jnp.sum(clipped_group_sizes.reshape(ep_size, ep_size, local_experts), axis=2)
        input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
        dispatch_out_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=x_local.dtype)
        x_dispatched = jax.lax.ragged_all_to_all(
            sorted_x,
            dispatch_out_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
        x_dispatch, local_sorted_indices, local_group_sizes = _local_permute_from_counts(
            x_dispatched,
            clipped_group_sizes,
            local_expert_size=local_experts,
            shard_index=shard_id,
        )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)

    with jax.named_scope("combine"):
        local_output = _sort_activations(out_dispatch, jnp.argsort(local_sorted_indices))
        return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=local_output.dtype)
        return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
            all_shard_counts.T, shard_id
        )
        returned = jax.lax.ragged_all_to_all(
            local_output,
            return_out_shape,
            return_input_offsets,
            return_send_sizes,
            return_output_offsets,
            return_recv_sizes,
            axis_name="expert",
        )
        returned = _expand_from_keep_mask(returned, keep_mask)
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            combine_weights_local,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(sender_group_sizes, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local, dropped_total


@named_call
def moe_mlp(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: MoeActivation = ActivationFunctionEnum.silu,
    implementation: MoeImplementation | str | None = None,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
    report_capacity_overflow: bool = False,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Functional routed MoE MLP core used by Grug modules and benchmarks.

    This helper handles dispatch/permute/unpermute (+EP collectives) from
    precomputed token-to-expert assignments. Routing logits/top-k selection
    stays in the caller (e.g. model MLP block).

    Set `report_capacity_overflow=True` to also return a scalar count of
    dropped expert assignments from EP capacity clipping.
    """
    if mesh is None:
        mesh = _current_mesh()

    if isinstance(activation, ActivationFunctionEnum):
        activation_fn = activation.to_jax_fn()
    else:
        activation_fn = activation

    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
    if selected_experts.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts.shape}")
    if selected_experts.shape != combine_weights.shape:
        raise ValueError(
            "selected_experts and combine_weights must have identical [T, K] shapes; "
            f"got {selected_experts.shape} vs {combine_weights.shape}"
        )
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError(
            f"selected_experts/combine_weights token dim ({selected_experts.shape[0]}) must match x token "
            f"dim ({x.shape[0]})"
        )

    num_experts = int(w_up_gate.shape[0])
    if w_down.shape[0] != num_experts:
        raise ValueError(
            f"w_down expert dimension ({w_down.shape[0]}) must match w_up_gate expert dimension ({num_experts})"
        )

    has_expert_axis = _mesh_has_axis(mesh, "expert")
    expert_axis_size = _mesh_axis_size(mesh, "expert")
    resolved_implementation = resolve_moe_implementation(implementation, mesh)

    if mesh is None or mesh.empty:
        out, dropped = _moe_mlp_local(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
        )
        if report_capacity_overflow:
            return out, dropped
        return out

    batch_spec = _batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size}")

        if resolved_implementation == "ring":
            shard_local_fn = _moe_mlp_ep_ring_local
        elif resolved_implementation == "ragged_all_to_all":
            shard_local_fn = _moe_mlp_ep_ragged_a2a_local
        else:
            raise AssertionError(f"Unhandled MoE implementation {resolved_implementation!r}")

        w_up_gate_spec = P("expert", None, None)
        w_down_spec = P("expert", None, None)

        x = _reshard_for_shard_map(x, mesh, batch_spec)
        selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
        combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
        w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
        w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

        shard_fn = shard_map(
            partial(
                shard_local_fn,
                activation_fn=activation_fn,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            ),
            mesh=mesh,
            in_specs=(
                batch_spec,
                batch_spec,
                batch_spec,
                w_up_gate_spec,
                w_down_spec,
            ),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )
        out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
        if report_capacity_overflow:
            return out, dropped
        return out

    # Fallback path for no expert axis (or expert axis size 1) keeps routing
    # semantics without EP collectives. JAX 0.9 requires shard_map in_specs to
    # match the actual input sharding, so reshard ordinary inputs to the mesh
    # specs that preserve data-axis parallelism.
    x_spec = _value_spec_or_default(x, batch_spec, replace_replicated=True)
    selected_experts_spec = _value_spec_or_default(selected_experts, batch_spec, replace_replicated=True)
    combine_weights_spec = _value_spec_or_default(combine_weights, batch_spec, replace_replicated=True)
    w_up_gate_spec = _value_spec_or_default(w_up_gate, P(*(None for _ in range(w_up_gate.ndim))))
    w_down_spec = _value_spec_or_default(w_down, P(*(None for _ in range(w_down.ndim))))

    x = _reshard_for_shard_map(x, mesh, x_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, selected_experts_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, combine_weights_spec)
    w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

    shard_fn = shard_map(
        partial(
            _moe_mlp_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
        ),
        mesh=mesh,
        in_specs=(
            x_spec,
            selected_experts_spec,
            combine_weights_spec,
            w_up_gate_spec,
            w_down_spec,
        ),
        out_specs=(x_spec, P()),
        check_vma=False,
    )
    out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
    if report_capacity_overflow:
        return out, dropped
    return out


__all__ = [
    "MoeActivation",
    "MoeImplementation",
    "moe_mlp",
    "resolve_moe_implementation",
]
