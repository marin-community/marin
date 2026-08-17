# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shape and dtype wrappers for the local MoK-like JAX FFI calls."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.mixture_of_kittens.config import MokLikeConfig, MokLikeRuntime


FAILURE_FENCE_TARGET = "levanter_mok_failure_fence"


def forward_target(num_devices: int) -> str:
    """Native forward handler symbol for one expert-group size.

    The adapter is compiled per rank count and exports a rank-suffixed symbol,
    so an object built for a different group size fails to resolve rather than
    being loaded silently.
    """

    return f"levanter_mok_forward_bf16_{num_devices}"


def backward_target(num_devices: int) -> str:
    """Native backward handler symbol for one expert-group size."""

    return f"levanter_mok_backward_bf16_{num_devices}"


_TILE_ROWS = 256
_SWIGLU_TILE_COLUMNS = 128
_CLUSTER_SIZE = 2


class MokLikeForwardContext(NamedTuple):
    """Activations saved by the BF16 forward for the fused backward."""

    x_routed: jax.Array
    gate_shared: jax.Array
    gate_routed: jax.Array
    up_shared: jax.Array
    up_routed: jax.Array
    hidden_shared: jax.Array
    hidden_routed: jax.Array
    stamp_slot: jax.Array
    stamp_generation_high: jax.Array
    stamp_generation_low: jax.Array
    stamp_runtime_epoch: jax.Array


def forward_bf16_local(
    x: jax.Array,
    routed_x: jax.Array,
    router_weights: jax.Array,
    shared_gate: jax.Array,
    routed_gate: jax.Array,
    shared_up: jax.Array,
    routed_up: jax.Array,
    shared_down: jax.Array,
    routed_down: jax.Array,
    schedule_peer_rank: jax.Array,
    schedule_peer_token_idx: jax.Array,
    num_scheduled_tokens: jax.Array,
    tokens_per_expert: jax.Array,
    *,
    runtime: MokLikeRuntime,
    config: MokLikeConfig,
    collective_id: int,
) -> tuple[jax.Array, jax.Array, MokLikeForwardContext, jax.Array]:
    """Run fused dispatch, shared and routed SwiGLU, combine, and epilogue.

    The routed combine and the shared output are returned separately, at the routed and shared
    widths respectively, so a caller with a latent projection can expand the first before adding
    the second.
    """
    if x.ndim != 2 or routed_x.ndim != 2 or router_weights.ndim != 2:
        raise ValueError("x, routed_x, and router_weights must be rank two")
    num_tokens, hidden_dim = x.shape
    routed_dim = routed_x.shape[1]
    top_k = router_weights.shape[1]
    if routed_x.shape[0] != num_tokens:
        raise ValueError("routed_x must have the same token count as x")
    if router_weights.shape[0] != num_tokens:
        raise ValueError("router_weights must have the same token count as x")
    if schedule_peer_rank.shape != schedule_peer_token_idx.shape:
        raise ValueError("schedule arrays must have the same shape")
    capacity = schedule_peer_rank.shape[0]
    if capacity % config.minibatch_size != 0:
        raise ValueError("schedule capacity must be divisible by minibatch_size")
    if num_tokens % _TILE_ROWS != 0 or capacity % _TILE_ROWS != 0:
        raise ValueError("token count and schedule capacity must be divisible by 256")

    # Only routed activations cross the wire, so the symmetric arena is sized from the routed width.
    runtime.require_compatible(
        num_tokens=num_tokens,
        hidden_dim=routed_dim,
        top_k=top_k,
        workspace_slots=config.workspace_slots,
    )
    intermediate_dim = shared_gate.shape[0]
    # The routed experts may be wider than the shared one, so every routed tile count is its own.
    routed_intermediate_dim = routed_gate.shape[1]
    routed_rows = config.macrobatch_size
    global_minibatches = capacity // config.minibatch_size
    global_row_blocks = capacity // (_TILE_ROWS // _CLUSTER_SIZE)
    shared_row_blocks = num_tokens // _TILE_ROWS
    routed_row_blocks = capacity // _TILE_ROWS
    # The routed tiles are indexed after the shared ones, so this array is the sum of two products
    # rather than one product over their combined row blocks.
    gate_ready = shared_row_blocks * (intermediate_dim // _TILE_ROWS) + routed_row_blocks * (
        routed_intermediate_dim // _TILE_ROWS
    )
    hidden_ready = shared_row_blocks + routed_row_blocks

    result_shapes = (
        jax.ShapeDtypeStruct((num_tokens, routed_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((gate_ready,), jnp.int32),
        jax.ShapeDtypeStruct((hidden_ready,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((global_row_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        forward_target(config.num_devices),
        result_shapes,
        # Scratch-buffer mutation is not observable through the array result. This
        # lets the saved custom-VJP context eliminate rematerialized communication.
        has_side_effect=False,
        vmap_method="broadcast_all",
    )(
        jnp.asarray(x, dtype=jnp.bfloat16),
        jnp.asarray(routed_x, dtype=jnp.bfloat16),
        jnp.asarray(router_weights, dtype=jnp.float32),
        jnp.asarray(shared_gate, dtype=jnp.bfloat16),
        jnp.asarray(routed_gate, dtype=jnp.bfloat16),
        jnp.asarray(shared_up, dtype=jnp.bfloat16),
        jnp.asarray(routed_up, dtype=jnp.bfloat16),
        jnp.asarray(shared_down, dtype=jnp.bfloat16),
        jnp.asarray(routed_down, dtype=jnp.bfloat16),
        jnp.asarray(schedule_peer_rank, dtype=jnp.int32),
        jnp.asarray(schedule_peer_token_idx, dtype=jnp.int32),
        jnp.reshape(jnp.asarray(num_scheduled_tokens, dtype=jnp.int32), (1,)),
        jnp.asarray(tokens_per_expert, dtype=jnp.int32),
        top_k=np.int32(top_k),
        num_comm_sms=np.int32(config.num_comm_sms),
        macrobatch_size=np.int32(config.macrobatch_size),
        minibatch_size=np.int32(config.minibatch_size),
        forward_x_storage=np.int32(config.forward_x_storage.native_ffi_code),
        collective_id=np.int64(collective_id),
    )
    context = MokLikeForwardContext(
        x_routed=results[1],
        gate_shared=results[2],
        gate_routed=results[3],
        up_shared=results[4],
        up_routed=results[5],
        hidden_shared=results[6],
        hidden_routed=results[7],
        stamp_slot=results[15],
        stamp_generation_high=results[16],
        stamp_generation_low=results[17],
        stamp_runtime_epoch=results[18],
    )
    return results[0], results[8], context, results[-1][0]


def backward_bf16_local(
    grad_output: jax.Array,
    grad_routed_output: jax.Array,
    x: jax.Array,
    routed_x: jax.Array,
    router_weights: jax.Array,
    shared_gate: jax.Array,
    routed_gate: jax.Array,
    shared_up: jax.Array,
    routed_up: jax.Array,
    shared_down: jax.Array,
    routed_down: jax.Array,
    context: MokLikeForwardContext,
    schedule_peer_rank: jax.Array,
    schedule_peer_token_idx: jax.Array,
    num_scheduled_tokens: jax.Array,
    tokens_per_expert: jax.Array,
    *,
    runtime: MokLikeRuntime,
    config: MokLikeConfig,
    collective_id: int,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Run the fused BF16 backward, accumulating routed weight gradients in FP32."""
    if x.ndim != 2 or router_weights.ndim != 2 or grad_output.shape != x.shape:
        raise ValueError("x, grad_output, and router_weights must have matching rank-two token shapes")
    num_tokens, hidden_dim = x.shape
    routed_dim = routed_x.shape[1]
    top_k = router_weights.shape[1]
    if routed_x.ndim != 2 or grad_routed_output.shape != routed_x.shape:
        raise ValueError("routed_x and grad_routed_output must have matching rank-two token shapes")
    if routed_x.shape[0] != num_tokens:
        raise ValueError("routed_x must have the same token count as x")
    if router_weights.shape[0] != num_tokens:
        raise ValueError("router_weights must have the same token count as x")
    if schedule_peer_rank.shape != schedule_peer_token_idx.shape:
        raise ValueError("schedule arrays must have the same shape")
    capacity = schedule_peer_rank.shape[0]
    if capacity % config.minibatch_size != 0:
        raise ValueError("schedule capacity must be divisible by minibatch_size")
    if num_tokens % _TILE_ROWS != 0 or capacity % _TILE_ROWS != 0:
        raise ValueError("token count and schedule capacity must be divisible by 256")

    runtime.require_compatible(
        num_tokens=num_tokens,
        hidden_dim=routed_dim,
        top_k=top_k,
        workspace_slots=config.workspace_slots,
    )
    intermediate_dim = shared_gate.shape[0]
    routed_intermediate_dim = routed_gate.shape[1]
    local_experts = routed_gate.shape[0]
    routed_rows = config.macrobatch_size
    global_minibatches = capacity // config.minibatch_size
    num_macrobatches = math.ceil(capacity / config.macrobatch_size)
    shared_row_blocks = num_tokens // _TILE_ROWS
    routed_row_blocks = capacity // _TILE_ROWS
    intermediate_col_blocks = intermediate_dim // _TILE_ROWS
    routed_intermediate_col_blocks = routed_intermediate_dim // _TILE_ROWS

    result_shapes = (
        jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, routed_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, top_k), jnp.float32),
        jax.ShapeDtypeStruct((local_experts, routed_intermediate_dim, routed_dim), jnp.float32),
        jax.ShapeDtypeStruct((local_experts, routed_intermediate_dim, routed_dim), jnp.float32),
        jax.ShapeDtypeStruct((local_experts, routed_dim, routed_intermediate_dim), jnp.float32),
        jax.ShapeDtypeStruct((intermediate_dim, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((intermediate_dim, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((hidden_dim, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows,), jnp.float32),
        jax.ShapeDtypeStruct((routed_rows, routed_intermediate_dim // _SWIGLU_TILE_COLUMNS), jnp.float32),
        jax.ShapeDtypeStruct((routed_rows, routed_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, routed_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_macrobatches,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct(
            (shared_row_blocks * intermediate_col_blocks + routed_row_blocks * routed_intermediate_col_blocks,),
            jnp.int32,
        ),
        jax.ShapeDtypeStruct((shared_row_blocks + routed_row_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((routed_row_blocks * routed_intermediate_col_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((routed_row_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((num_macrobatches,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        backward_target(config.num_devices),
        result_shapes,
        # Runtime workspaces, counters, and leases are private protocol state.
        # Gradients and failure status fully describe the observable result, so
        # the call does not need an ordered FFI-effect token.
        has_side_effect=False,
        vmap_method="broadcast_all",
    )(
        jnp.asarray(grad_output, dtype=jnp.bfloat16),
        jnp.asarray(grad_routed_output, dtype=jnp.bfloat16),
        jnp.asarray(x, dtype=jnp.bfloat16),
        jnp.asarray(routed_x, dtype=jnp.bfloat16),
        jnp.asarray(router_weights, dtype=jnp.float32),
        jnp.asarray(shared_gate, dtype=jnp.bfloat16),
        jnp.asarray(routed_gate, dtype=jnp.bfloat16),
        jnp.asarray(shared_up, dtype=jnp.bfloat16),
        jnp.asarray(routed_up, dtype=jnp.bfloat16),
        jnp.asarray(shared_down, dtype=jnp.bfloat16),
        jnp.asarray(routed_down, dtype=jnp.bfloat16),
        jnp.asarray(context.x_routed, dtype=jnp.bfloat16),
        jnp.asarray(context.gate_shared, dtype=jnp.bfloat16),
        jnp.asarray(context.gate_routed, dtype=jnp.bfloat16),
        jnp.asarray(context.up_shared, dtype=jnp.bfloat16),
        jnp.asarray(context.up_routed, dtype=jnp.bfloat16),
        jnp.asarray(context.hidden_shared, dtype=jnp.bfloat16),
        jnp.asarray(context.hidden_routed, dtype=jnp.bfloat16),
        jnp.asarray(context.stamp_slot, dtype=jnp.int32),
        jnp.asarray(context.stamp_generation_high, dtype=jnp.int32),
        jnp.asarray(context.stamp_generation_low, dtype=jnp.int32),
        jnp.asarray(context.stamp_runtime_epoch, dtype=jnp.int32),
        jnp.asarray(schedule_peer_rank, dtype=jnp.int32),
        jnp.asarray(schedule_peer_token_idx, dtype=jnp.int32),
        jnp.reshape(jnp.asarray(num_scheduled_tokens, dtype=jnp.int32), (1,)),
        jnp.asarray(tokens_per_expert, dtype=jnp.int32),
        top_k=np.int32(top_k),
        num_comm_sms=np.int32(config.bwd_num_comm_sms),
        macrobatch_size=np.int32(config.macrobatch_size),
        minibatch_size=np.int32(config.minibatch_size),
        backward_peer_storage=np.int32(config.backward_peer_storage.native_ffi_code),
        collective_id=np.int64(collective_id),
    )
    active_experts = tokens_per_expert > 0
    routed_weight_gradients = tuple(
        jnp.where(active_experts[:, None, None], gradient, 0).astype(jnp.bfloat16) for gradient in results[3:6]
    )
    gradients = tuple(results[:3]) + routed_weight_gradients + tuple(results[6:9])
    return gradients, results[-1][0]


def fence_mok_like_failure(marker: jax.Array) -> jax.Array:
    """Raise a typed-FFI error whose data result belongs to the awaited execution."""
    marker = jnp.asarray(marker, dtype=jnp.int32)
    if marker.shape != ():
        raise ValueError(f"failure marker must be scalar, got shape {marker.shape}")
    return jax.ffi.ffi_call(
        FAILURE_FENCE_TARGET,
        jax.ShapeDtypeStruct((), jnp.int32),
        has_side_effect=False,
        vmap_method="broadcast_all",
    )(marker)
