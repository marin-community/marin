# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged all-to-all expert-parallel Grug MoE backend."""

import logging
import math
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_MOE_OUTPUT
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
    _sort_activations,
    _unpermute_from_global_expert,
)
from levanter.grug.sharding import _batch_axes

logger = logging.getLogger(__name__)

# Sentinel proving the ep25-d1 custom-adjoint build shipped in the iris bundle. Logged once at
# import; grep the task logs for this line to confirm the modified module is the one that ran.
logger.info("ep_ragged_all_to_all loaded: gather-dispatch structured custom-adjoint build (ep25-d1)")


def _gather_dispatch_enabled() -> bool:
    """Gather-dispatch forward: int32 assignment scatter + activation gather (issue 7201 treatment)."""
    return os.environ.get("SCALE_A2A_GATHER_DISPATCH") == "1"


def _custom_adjoint_enabled() -> bool:
    """Structured custom_vjp for the dispatch and combine gathers instead of XLA's generic scatter-add."""
    return os.environ.get("SCALE_A2A_CUSTOM_ADJOINT") == "1"


def _batch_experts_enabled() -> bool:
    """Batch the per-local-expert loop into one dispatch a2a, one grouped GEMM, one combine a2a."""
    return os.environ.get("SCALE_A2A_BATCH_EXPERTS") == "1"


def _mxfp8_enabled() -> bool:
    """MXFP8 (block-scaled fp8) grouped expert GEMMs instead of the bf16 per-expert matmuls."""
    return os.environ.get("SCALE_MOE_MXFP8") == "1"


def _spill_attempts() -> int:
    """Same-step spill attempts per dropped assignment. 0 disables spill (index composition unchanged)."""
    return int(os.environ.get("SCALE_A2A_SPILL", "0"))


def _segment_rank(bucket: Int[Array, " n"], num_buckets: int) -> Int[Array, " n"]:
    """Rank of each element within its bucket, in index order (stable)."""
    total = bucket.shape[0]
    order = jnp.argsort(bucket, stable=True)
    counts = jnp.bincount(bucket, length=num_buckets).astype(jnp.int32)
    starts = jnp.cumsum(counts) - counts
    ranks_sorted = jnp.arange(total, dtype=jnp.int32) - starts[bucket[order]]
    return ranks_sorted[jnp.argsort(order)]


def _assign_with_spill(
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    num_experts: int,
    capacity: int,
    attempts: int,
) -> tuple[Int[Array, " A"], Int[Array, " A"], Array]:
    """Place each (token, k) assignment into a per-(sender, expert) capacity bucket.

    Round 0 is the baseline policy: assignments to expert e take slots in e's bucket in index
    order, and those past ``capacity`` are dropped. Each spill attempt then re-offers the still
    unplaced assignments to the next-ranked expert the same token selected, filling only that
    bucket's remaining headroom. A spilled assignment is computed by an expert the token actually
    chose and keeps its own combine weight, so it contributes real signal instead of nothing.

    Returns the target expert, slot within that expert's bucket, and the placed mask.
    """
    topk = selected_experts_local.shape[1]
    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)
    total = flat_experts.shape[0]

    slot = _segment_rank(flat_experts, num_experts)
    placed = slot < capacity
    if attempts <= 0:
        return flat_experts, slot, placed

    target = flat_experts
    occupancy = jnp.minimum(jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32), capacity)
    # Sentinel bucket num_experts collects already-placed assignments so they do not consume ranks.
    for attempt in range(1, attempts + 1):
        candidate = jnp.roll(selected_experts_local, -attempt, axis=1).reshape(-1).astype(jnp.int32)
        offered = jnp.where(placed, num_experts, candidate)
        rank = _segment_rank(offered, num_experts + 1)
        candidate_slot = occupancy[candidate] + rank
        spilled = jnp.logical_and(~placed, candidate_slot < capacity)
        target = jnp.where(spilled, candidate, target)
        slot = jnp.where(spilled, candidate_slot, slot)
        placed = jnp.logical_or(placed, spilled)
        filled = jnp.bincount(jnp.where(spilled, candidate, num_experts), length=num_experts + 1)[:num_experts]
        occupancy = jnp.minimum(occupancy + filled.astype(jnp.int32), capacity)
    assert target.shape == (total,) and topk > 0
    return target, slot, placed


def _batch_expert_group_size(local_experts: int) -> int:
    """Experts per batched a2a/GEMM group. Defaults to all local experts (one group = full batch).

    Smaller groups trade the batching's launch-overhead win for a lower memory peak and a smaller
    compiled graph: only ``group_size`` experts' received activations and grouped-GEMM intermediates
    are live at once instead of all ``local_experts``.
    """
    group_size = int(os.environ.get("SCALE_A2A_BATCH_GROUP", local_experts))
    if group_size <= 0 or local_experts % group_size != 0:
        raise ValueError(
            f"SCALE_A2A_BATCH_GROUP={group_size} must be a positive divisor of local_experts={local_experts}"
        )
    return group_size


@jax.custom_vjp
def _dispatch_gather(
    x_local: Float[Array, "Tlocal H"],
    token_sources: Int[Array, " send"],
    linear_indices: Int[Array, " assignments"],
    keep: Array,
) -> Float[Array, "send H"]:
    """Gather token rows into the fixed-capacity send buffer.

    Forward is the plain gather ``padded_x[token_sources]``. The custom VJP replaces XLA's
    generic scatter-add transpose with a structured gather-then-segment-sum over the top-k
    assignments, reusing the forward's ``linear_indices`` composition. ``token_sources``,
    ``linear_indices`` and ``keep`` are integer/bool and carry no cotangent.
    """
    hidden_dim = x_local.shape[1]
    padded_x = jnp.concatenate([x_local, jnp.zeros((1, hidden_dim), x_local.dtype)], axis=0)
    return padded_x[token_sources]


def _dispatch_gather_fwd(x_local, token_sources, linear_indices, keep):
    send_x = _dispatch_gather(x_local, token_sources, linear_indices, keep)
    # send_x shares x_local's dtype, so the incoming cotangent carries it too; only static shape
    # (tokens_per_shard) and the integer index arrays need to survive into the backward.
    return send_x, (linear_indices, keep, x_local.shape[0])


def _dispatch_gather_bwd(residual, cotangent):
    linear_indices, keep, tokens_per_shard = residual
    send_size = cotangent.shape[0]
    hidden_dim = cotangent.shape[1]
    topk = linear_indices.shape[0] // tokens_per_shard
    # d_x_local[t] = sum over token t's kept assignments of cotangent at their send slot.
    grad_rows = cotangent[jnp.minimum(linear_indices, send_size - 1)]
    grad_rows = jnp.where(keep[:, None], grad_rows, 0).astype(jnp.float32)
    grad_rows = grad_rows.reshape(tokens_per_shard, topk, hidden_dim)
    d_x_local = grad_rows.sum(axis=1).astype(cotangent.dtype)
    return d_x_local, None, None, None


_dispatch_gather.defvjp(_dispatch_gather_fwd, _dispatch_gather_bwd)


@jax.custom_vjp
def _combine_gather(
    send_output: Float[Array, "send H"],
    gather_indices: Int[Array, " assignments"],
    keep: Array,
    assignment_sources: Int[Array, " send"],
) -> Float[Array, "assignments H"]:
    """Gather expert outputs from the send buffer back into assignment order.

    Forward is ``send_output[gather_indices]`` with dropped assignments zeroed. The index map is
    injective on kept assignments, so the true transpose is a gather along the slot->assignment
    inverse (``assignment_sources``) rather than XLA's scatter-add. ``gather_indices``, ``keep``
    and ``assignment_sources`` are integer/bool and carry no cotangent.
    """
    gathered = send_output[gather_indices]
    return jnp.where(keep[:, None], gathered, 0)


def _combine_gather_fwd(send_output, gather_indices, keep, assignment_sources):
    gathered = _combine_gather(send_output, gather_indices, keep, assignment_sources)
    return gathered, (assignment_sources,)


def _combine_gather_bwd(residual, cotangent):
    (assignment_sources,) = residual
    assignments_per_shard = cotangent.shape[0]
    # slot j was read by assignment assignment_sources[j] (or is unfilled -> assignments_per_shard).
    valid = assignment_sources < assignments_per_shard
    src = jnp.minimum(assignment_sources, assignments_per_shard - 1)
    d_send_output = jnp.where(valid[:, None], cotangent[src], 0).astype(cotangent.dtype)
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
    """Expert-parallel MoE with fixed-capacity all-to-all dispatch and combine."""
    chunks = max(int(os.environ.get("SCALE_A2A_CHUNKS", "1")), 1)
    tokens_per_shard = x_local.shape[0]
    if tokens_per_shard % chunks != 0:
        raise ValueError(f"tokens_per_shard={tokens_per_shard} must be divisible by SCALE_A2A_CHUNKS={chunks}")
    if chunks == 1:
        return _fixed_a2a_core(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    tokens_per_chunk = tokens_per_shard // chunks
    chunk_outputs = []
    chunk_dropped = []
    for chunk_index in range(chunks):
        chunk_start = chunk_index * tokens_per_chunk
        chunk_end = (chunk_index + 1) * tokens_per_chunk
        output, dropped = _fixed_a2a_core(
            x_local[chunk_start:chunk_end],
            selected_experts_local[chunk_start:chunk_end],
            combine_weights_local[chunk_start:chunk_end],
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        chunk_outputs.append(output)
        chunk_dropped.append(dropped)
    return jnp.concatenate(chunk_outputs, axis=0), jnp.sum(jnp.stack(chunk_dropped))


def _fixed_a2a_core(
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
    """Run one fixed-capacity all-to-all dispatch, expert MLP, and combine."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard = x_local.shape[0]
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    hidden_dim = x_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    capacity = max(int(math.ceil(capacity_factor * assignments_per_shard / num_experts)), 1)

    # Keep XLA's auto-rematerializer from materializing flash-attention scores when
    # estimating memory pressure for the full attention-plus-MoE scan body.
    use_barrier = os.environ.get("SCALE_A2A_NO_BARRIER") != "1"
    if use_barrier:
        x_local, combine_weights_local = jax.lax.optimization_barrier((x_local, combine_weights_local))

    gather_dispatch = _gather_dispatch_enabled()
    custom_adjoint = _custom_adjoint_enabled()
    if custom_adjoint and not gather_dispatch:
        raise ValueError("SCALE_A2A_CUSTOM_ADJOINT=1 requires SCALE_A2A_GATHER_DISPATCH=1")
    if _mxfp8_enabled() and _batch_experts_enabled():
        raise ValueError("SCALE_MOE_MXFP8=1 and SCALE_A2A_BATCH_EXPERTS=1 are mutually exclusive expert-GEMM arms")

    # Spill only rewrites which bucket slot an assignment occupies, so everything downstream --
    # the dispatch gather, the collectives, and both custom-adjoint VJPs -- is generic in
    # (linear_indices, keep) and needs no change.
    target_experts, slot, keep = _assign_with_spill(
        selected_experts_local,
        num_experts=num_experts,
        capacity=capacity,
        attempts=_spill_attempts(),
    )
    local_expert_indices = (target_experts % local_experts).astype(jnp.int32)
    destination_shards = (target_experts // local_experts).astype(jnp.int32)
    bucket_size = expert_shards * capacity
    send_size = local_experts * bucket_size
    linear_indices = jnp.where(
        keep,
        local_expert_indices * bucket_size + destination_shards * capacity + slot,
        send_size,
    )

    # slot -> assignment inverse of linear_indices (assignments_per_shard marks an unfilled slot).
    # Shared by the gather-dispatch forward and the combine gather's structured backward.
    if gather_dispatch:
        assignment_sources = (
            jnp.full((send_size,), assignments_per_shard, dtype=jnp.int32)
            .at[linear_indices]
            .set(jnp.arange(assignments_per_shard, dtype=jnp.int32), mode="drop")
        )

    moe_dim = moe_w2_local.shape[1]
    with jax.named_scope("dispatch"):
        if gather_dispatch:
            token_sources = jnp.where(
                assignment_sources < assignments_per_shard,
                assignment_sources // topk,
                tokens_per_shard,
            )
            if custom_adjoint:
                send_x = _dispatch_gather(x_local, token_sources, linear_indices, keep)
            else:
                padded_x = jnp.concatenate([x_local, jnp.zeros((1, hidden_dim), x_local.dtype)], axis=0)
                send_x = padded_x[token_sources]
        else:
            repeated_x = jnp.repeat(x_local, topk, axis=0)
            send_x = jnp.zeros((send_size, hidden_dim), x_local.dtype).at[linear_indices].set(repeated_x, mode="drop")
        send_x = send_x.reshape(local_experts, expert_shards, capacity, hidden_dim)

    if _mxfp8_enabled():
        # All local experts dispatch first, then one MXFP8 grouped GEMM covers every local expert's
        # bucket, then each expert's slice combines. Ported verbatim from agent/ep25-d2-bakeoff, where
        # it measured -2.83pp at d5120/i1280; the reopening condition d2 named was fatter expert GEMMs.
        from levanter.grug._moe.mxfp8 import mxfp8_expert_mlp  # noqa: PLC0415

        expert_inputs = []
        for local_expert_index in range(local_experts):
            with jax.named_scope("dispatch"):
                received = jax.lax.all_to_all(
                    send_x[local_expert_index], "expert", split_axis=0, concat_axis=0, tiled=True
                )
                received = tree_checkpoint_name(received, _CHECKPOINT_DISPATCH_INPUT)
                expert_inputs.append(received.reshape(bucket_size, hidden_dim))
        with jax.named_scope("moe_up_down"):
            grouped_input = jnp.concatenate(expert_inputs, axis=0)
            group_sizes = jnp.full((local_experts,), bucket_size, dtype=jnp.int32)
            grouped_output = mxfp8_expert_mlp(
                grouped_input, moe_w13_local, moe_w2_local, group_sizes
            ).reshape(local_experts, bucket_size, hidden_dim)
        output_parts = []
        for local_expert_index in range(local_experts):
            with jax.named_scope("combine"):
                returned = jax.lax.all_to_all(
                    grouped_output[local_expert_index].reshape(expert_shards, capacity, hidden_dim),
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
    elif _batch_experts_enabled():
        # Process local experts in groups of `group_size`: one dispatch a2a, one grouped up/down GEMM,
        # and one combine a2a per group. The grouped-expert axis is a batch dim the a2a passes through
        # (split/concat move the expert-shard axis 1), so results match the per-expert loop. group_size
        # = local_experts is the full batch (fewest collectives); a smaller group lowers the memory peak.
        group_size = _batch_expert_group_size(local_experts)
        send_x_grouped = send_x.reshape(local_experts // group_size, group_size, expert_shards, capacity, hidden_dim)
        output_parts = []
        for group_index in range(local_experts // group_size):
            expert_slice = slice(group_index * group_size, (group_index + 1) * group_size)
            with jax.named_scope("dispatch"):
                received = jax.lax.all_to_all(
                    send_x_grouped[group_index], "expert", split_axis=1, concat_axis=1, tiled=True
                )
                received = tree_checkpoint_name(received, _CHECKPOINT_DISPATCH_INPUT)
            with jax.named_scope("moe_up_down"):
                expert_input = received.reshape(group_size, bucket_size, hidden_dim)
                hidden = jnp.einsum("nbh,nhi->nbi", expert_input, moe_w13_local[expert_slice])
                gate, up = jnp.split(hidden, [moe_dim], axis=-1)
                expert_output = jnp.einsum("nbi,nih->nbh", activation_fn(gate) * up, moe_w2_local[expert_slice])
            with jax.named_scope("combine"):
                returned = jax.lax.all_to_all(
                    expert_output.reshape(group_size, expert_shards, capacity, hidden_dim),
                    "expert",
                    split_axis=1,
                    concat_axis=1,
                    tiled=True,
                )
                output_parts.append(returned)
        with jax.named_scope("combine"):
            send_output = jnp.concatenate(output_parts, axis=0) if len(output_parts) > 1 else output_parts[0]
            send_output = tree_checkpoint_name(send_output, _CHECKPOINT_MOE_OUTPUT)
            send_output = send_output.reshape(send_size, hidden_dim)
    else:
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

    with jax.named_scope("combine"):
        gather_indices = jnp.minimum(linear_indices, send_size - 1)
        if custom_adjoint:
            gathered = _combine_gather(send_output, gather_indices, keep, assignment_sources)
        else:
            gathered = jnp.where(keep[:, None], send_output[gather_indices], 0)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        dropped_local = assignments_per_shard - jnp.sum(keep, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    if use_barrier:
        out_local = jax.lax.optimization_barrier(out_local)
    return out_local, dropped_total


def _moe_mlp_ep_ragged_a2a_local(
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
    if os.environ.get("SCALE_A2A_FIXED") == "1":
        return _moe_mlp_ep_fixed_a2a_local(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

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
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
