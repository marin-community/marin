# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged all-to-all expert-parallel Grug MoE backend."""

import math
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot

try:
    # QuACK SM100 cutlass grouped SwiGLU GEMM (custom_vjp, shard_map-safe) — much faster than
    # the Pallas/Triton ragged_dot on Blackwell. Optional: needs quack-kernels + cutlass-dsl.
    from levanter.grug._moe.sonic_cute import _expert_mlp as _quack_expert_mlp
    from levanter.grug._moe.sonic_cute import _interleave_gate_up as _quack_interleave_gate_up
except ImportError:
    _quack_expert_mlp = None
    _quack_interleave_gate_up = None
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
    # fp8 token payload halves the all-to-all bytes (dispatch+combine SendRecv is the top step cost
    # at 512E). Tokens go e4m3 over the wire; QuACK consumes fp8 directly (weights follow via astype).
    _a2a_dt = jnp.float8_e4m3fn if os.environ.get("SCALE_RAGGED_FP8") == "1" else x_local.dtype

    with jax.named_scope("dispatch"):
        # Move tokens in the a2a dtype (fp8 when enabled) so the permute / compact / local-permute
        # sorts (memory-bound gathers on [tokens, hidden]) move half the bytes. Cast up to the
        # compute dtype only at the QuACK GEMM input.
        x_move = x_local.astype(_a2a_dt)
        sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
            x_move,
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
        dispatch_out_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=_a2a_dt)
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
        # Cast the fp8 wire payload up to the compute dtype for the GEMM (QuACK's backward mixes the
        # saved activation with bf16 weights via ragged_dot, which rejects implicit fp8 promotion).
        x_dispatch = x_dispatch.astype(x_local.dtype)
        moe_dim = moe_w2_local.shape[1]
        if os.environ.get("SCALE_RAGGED_QUACK") == "1" and _quack_expert_mlp is not None:
            # QuACK grouped SwiGLU over the packed dispatch buffer. `local_group_sizes` already
            # absorbs the trailing padding rows into the last expert (ep_common line 121), so its
            # prefix sum tiles the full buffer; padding rows are zeroed and dropped in combine.
            w13_il = _quack_interleave_gate_up(moe_w13_local, moe_dim).astype(x_dispatch.dtype)
            cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(local_group_sizes.astype(jnp.int32))])
            out_dispatch = _quack_expert_mlp(
                x_dispatch,
                w13_il,
                moe_w2_local.astype(x_dispatch.dtype),
                local_group_sizes.astype(jnp.int32),
                cu,
            )
        else:
            w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
            gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
            out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)

    with jax.named_scope("combine"):
        local_output = _sort_activations(out_dispatch.astype(_a2a_dt), jnp.argsort(local_sorted_indices))
        return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=_a2a_dt)
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
        returned = returned.astype(x_local.dtype)
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
