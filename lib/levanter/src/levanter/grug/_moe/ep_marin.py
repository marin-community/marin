# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin EP Grug MoE backend: group-pooled waterfilling drops + fused transport.

Implements the Marin EP spec (`experiments/marin_ep/SPEC.md`) — group-pooled
waterfilling drop rule with the (expert, source, rank) receive layout —
inside `jax.shard_map` over the `"expert"` axis. Production transport is the
fused Mosaic-GPU put kernel (`marin_ep_transport`), which writes each
owner's expert-major pool directly, eliminating both local permutes;
`ragged_all_to_all` and an `all_gather` emulation remain selectable for
comparison and XLA:CPU conformance runs. Expert GEMMs are `ragged_dot`.

Structurally this mirrors `levanter.grug._moe.ep_ragged_all_to_all`; the
difference is the accept rule: instead of a per-receiver pool clipped
first-sender-wins, acceptance is `clip(kept_g - base[d, g], 0, n_d[g])`
with `kept_g` from the group-pooled waterfilling over global expert
totals, which is deterministic, EP-degree invariant, and gives ~10x fewer
drops at hero shape (see `experiments/marin_ep/bench/drop_rate_study.py`).
Layer-level A/B on one GB200 tray (EP4, hero widths, 32k tokens/device):
6.4x fwd / 5.1x fwd+bwd over the ragged transport, bit-identical outputs.
"""

import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int
from levanter.grug._moe.ep_common import (
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
    _sort_activations,
    _unpermute_from_global_expert,
)
from levanter.grug._moe.marin_ep_transport import combine_segments, dispatch_segments, put_with_transpose
from levanter.grug.sharding import _batch_axes


def kept_rows_waterfill(total_counts: Int[Array, " E"], *, capacity: int, group_size: int) -> Int[Array, " E"]:
    """Traceable twin of `oracle.kept_rows_per_expert` (SPEC S2 waterfilling)."""
    num_experts = total_counts.shape[0]
    if num_experts % group_size:
        raise ValueError(f"group_size={group_size} must divide num_experts={num_experts}")
    counts = total_counts.reshape(-1, group_size).astype(jnp.int32)
    floor = jnp.minimum(counts, capacity)
    overflow = counts - floor
    slack = jnp.maximum(capacity - counts, 0).sum(axis=1, keepdims=True)
    granted_cum = jnp.minimum(jnp.cumsum(overflow, axis=1), slack)
    grants = jnp.diff(granted_cum, axis=1, prepend=0)
    return (floor + grants).reshape(num_experts)


def _static_capacity(assignments_global: int, num_experts: int, capacity_factor: float) -> int:
    raw = max(1, math.ceil(capacity_factor * assignments_global / num_experts))
    return min(raw, assignments_global)


def _ragged_a2a_gathered(
    operand: Float[Array, "N H"],
    *,
    out_rows: int,
    shard_counts: Int[Array, "S S"],
    shard_id: Int[Array, ""],
    axis_name: str,
) -> Float[Array, "outrows H"]:
    """Emulate `ragged_all_to_all` with `all_gather` + local assembly.

    Same receive layout as `jax.lax.ragged_all_to_all` driven by
    `_shard_a2a_params(shard_counts, ...)`: receiver `r` concatenates, in
    sender order, sender `s`'s slice `operand[sum_{r'<r} counts[s, r'] :][:counts[s, r]]`.
    Bandwidth-wasteful (full gather) but supported on XLA:CPU, so spec
    conformance runs end-to-end on host devices; also differentiable.
    """
    gathered = jax.lax.all_gather(operand, axis_name)  # [S, N, H]
    recv_counts = shard_counts[:, shard_id]  # rows from each sender
    recv_ends = jnp.cumsum(recv_counts)
    positions = jnp.arange(out_rows, dtype=jnp.int32)
    sender = jnp.searchsorted(recv_ends, positions, side="right")
    sender_clipped = jnp.minimum(sender, shard_counts.shape[0] - 1)
    within = positions - (recv_ends[sender_clipped] - recv_counts[sender_clipped])
    sender_input_offsets = jnp.cumsum(shard_counts, axis=1) - shard_counts  # [S, S]
    src_row = sender_input_offsets[sender_clipped, shard_id] + within
    valid = positions < recv_ends[-1]
    rows = gathered[sender_clipped, jnp.clip(src_row, 0, operand.shape[0] - 1)]
    return jnp.where(valid[:, None], rows, 0)


def marin_ep_moe_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    pool_group_size: int,
    transport: Literal["ragged", "gathered", "mgpu"] = "ragged",
    splits_per_peer: int = 1,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Shard-local Marin EP MoE body (runs inside shard_map over "expert").

    `transport="ragged"` uses `jax.lax.ragged_all_to_all` (production);
    `"gathered"` uses the all_gather emulation (XLA:CPU conformance runs);
    `"mgpu"` uses the fused Mosaic-GPU put kernel, which writes the
    expert-major pool layout directly — no local permute on either side.
    """
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")
    if local_experts % pool_group_size:
        raise ValueError(f"pool_group_size={pool_group_size} must divide local expert count={local_experts}")

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    capacity = _static_capacity(assignments_per_shard * ep_size, num_experts, capacity_factor)
    pool_rows = local_experts * capacity

    with jax.named_scope("dispatch"):
        sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
        )
        all_counts = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")  # [S, E]
        kept = kept_rows_waterfill(all_counts.sum(axis=0), capacity=capacity, group_size=pool_group_size)
        base = jnp.cumsum(all_counts, axis=0) - all_counts
        accepted = jnp.clip(kept[None, :] - base, 0, all_counts)  # [S, E]
        my_accepted = accepted[shard_id]
        keep_mask = _expert_prefix_keep_mask(
            group_sizes.astype(jnp.int32),
            my_accepted,
            total_size=assignments_per_shard,
        )
        sorted_x = _compact_by_keep_mask(sorted_x, keep_mask)

        if transport == "mgpu":
            # Region = compacted per-owner prefix of kept rows: the pool has
            # no capacity gaps, groups are contiguous, and `ragged_dot`
            # consumes it directly (kept[g] == sum_d accepted[d, g]).
            kept_by_owner = kept.reshape(ep_size, local_experts)
            region = (jnp.cumsum(kept_by_owner, axis=1) - kept_by_owner).reshape(num_experts).astype(jnp.int32)
            dispatch_plan = dispatch_segments(accepted, region, shard_id, local_experts=local_experts)
            combine_plan = combine_segments(accepted, region, shard_id, local_experts=local_experts)
            send_rows = jnp.sum(my_accepted, dtype=jnp.int32)
            pool_recv_rows = jnp.sum(kept_by_owner[shard_id], dtype=jnp.int32)
            x_pool = put_with_transpose(
                sorted_x,
                dispatch_plan,
                combine_plan,
                pool_recv_rows,
                send_rows,
                pool_rows,
                assignments_per_shard,
                "expert",
                ep_size,
            )
            local_group_sizes = kept_by_owner[shard_id].astype(jnp.int32)
            local_sorted_indices = shard_counts = None
        else:
            shard_counts = jnp.sum(accepted.reshape(ep_size, ep_size, local_experts), axis=2)
            if transport == "ragged":
                input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(
                    shard_counts, shard_id, splits_per_peer
                )
                dispatch_out_shape = jnp.zeros((pool_rows, x_local.shape[1]), dtype=x_local.dtype)
                x_dispatched = jax.lax.ragged_all_to_all(
                    sorted_x,
                    dispatch_out_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name="expert",
                )
            else:
                x_dispatched = _ragged_a2a_gathered(
                    sorted_x, out_rows=pool_rows, shard_counts=shard_counts, shard_id=shard_id, axis_name="expert"
                )
            x_pool, local_sorted_indices, local_group_sizes = _local_permute_from_counts(
                x_dispatched,
                accepted,
                local_expert_size=local_experts,
                shard_index=shard_id,
            )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_pool, moe_w13_local, local_group_sizes)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_pool = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)

    with jax.named_scope("combine"):
        if transport == "mgpu":
            # `out_pool` is already in the (expert, source, rank) layout the
            # combine plan reads from — no unsort needed.
            returned = put_with_transpose(
                out_pool,
                combine_plan,
                dispatch_plan,
                send_rows,
                pool_recv_rows,
                assignments_per_shard,
                pool_rows,
                "expert",
                ep_size,
            )
        elif transport == "ragged":
            local_output = _sort_activations(out_pool, jnp.argsort(local_sorted_indices))
            return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=local_output.dtype)
            return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
                shard_counts.T, shard_id, splits_per_peer
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
        else:
            local_output = _sort_activations(out_pool, jnp.argsort(local_sorted_indices))
            returned = _ragged_a2a_gathered(
                local_output,
                out_rows=assignments_per_shard,
                shard_counts=shard_counts.T,
                shard_id=shard_id,
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
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(my_accepted, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total


def _moe_mlp_ep_marin_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    splits_per_peer: int = 1,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """`moe_mlp` entry point: per-owner capacity pooling, best available transport.

    Pooling over each owner's co-located experts is free (one receive pool
    per device) and cuts drops ~10x vs per-expert capacity at hero shape,
    so it is not a tunable here.

    The fused Mosaic-GPU transport needs symmetric memory across the expert
    axis, which current jax provides only within a single process (upstream
    removed the multi-process NVSHMEM path from Mosaic custom calls in
    2026-07; the enable flag is `reserved` in xla.proto). Multi-controller
    runs fall back to `ragged_all_to_all` — same semantics, same drop rule.
    """
    transport = "mgpu" if jax.process_count() == 1 else "ragged"
    return marin_ep_moe_local(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        pool_group_size=moe_w13_local.shape[0],
        transport=transport,
        splits_per_peer=splits_per_peer,
    )
