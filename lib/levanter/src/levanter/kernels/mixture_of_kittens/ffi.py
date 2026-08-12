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


FORWARD_TARGET = "levanter_mok_forward_bf16_4"
BACKWARD_TARGET = "levanter_mok_backward_bf16_4"
FAILURE_FENCE_TARGET = "levanter_mok_failure_fence"
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
) -> tuple[jax.Array, MokLikeForwardContext, jax.Array]:
    """Run fused dispatch, shared and routed SwiGLU, combine, and epilogue."""
    if x.ndim != 2 or router_weights.ndim != 2:
        raise ValueError("x and router_weights must be rank two")
    num_tokens, hidden_dim = x.shape
    top_k = router_weights.shape[1]
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
        hidden_dim=hidden_dim,
        top_k=top_k,
        workspace_slots=config.workspace_slots,
    )
    intermediate_dim = shared_gate.shape[0]
    routed_rows = config.macrobatch_size
    global_minibatches = capacity // config.minibatch_size
    global_row_blocks = capacity // (_TILE_ROWS // _CLUSTER_SIZE)
    shared_row_blocks = num_tokens // _TILE_ROWS
    routed_row_blocks = capacity // _TILE_ROWS
    gate_ready = (shared_row_blocks + routed_row_blocks) * (intermediate_dim // _TILE_ROWS)
    hidden_ready = shared_row_blocks + routed_row_blocks

    result_shapes = (
        jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, hidden_dim), jnp.bfloat16),
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
        FORWARD_TARGET,
        result_shapes,
        # Scratch-buffer mutation is not observable through the array result. This
        # lets the saved custom-VJP context eliminate rematerialized communication.
        has_side_effect=False,
        vmap_method="broadcast_all",
    )(
        jnp.asarray(x, dtype=jnp.bfloat16),
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
    return results[0], context, results[-1][0]


def backward_bf16_local(
    grad_output: jax.Array,
    x: jax.Array,
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
    top_k = router_weights.shape[1]
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
        hidden_dim=hidden_dim,
        top_k=top_k,
        workspace_slots=config.workspace_slots,
    )
    intermediate_dim = shared_gate.shape[0]
    local_experts = routed_gate.shape[0]
    routed_rows = config.macrobatch_size
    global_minibatches = capacity // config.minibatch_size
    num_macrobatches = math.ceil(capacity / config.macrobatch_size)
    shared_row_blocks = num_tokens // _TILE_ROWS
    routed_row_blocks = capacity // _TILE_ROWS
    intermediate_col_blocks = intermediate_dim // _TILE_ROWS

    result_shapes = (
        jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, top_k), jnp.float32),
        jax.ShapeDtypeStruct((local_experts, intermediate_dim, hidden_dim), jnp.float32),
        jax.ShapeDtypeStruct((local_experts, intermediate_dim, hidden_dim), jnp.float32),
        jax.ShapeDtypeStruct((local_experts, hidden_dim, intermediate_dim), jnp.float32),
        jax.ShapeDtypeStruct((intermediate_dim, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((intermediate_dim, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((hidden_dim, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows,), jnp.float32),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim // _SWIGLU_TILE_COLUMNS), jnp.float32),
        jax.ShapeDtypeStruct((routed_rows, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_macrobatches,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct(((shared_row_blocks + routed_row_blocks) * intermediate_col_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((shared_row_blocks + routed_row_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((routed_row_blocks * intermediate_col_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((routed_row_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((num_macrobatches,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        BACKWARD_TARGET,
        result_shapes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        jnp.asarray(grad_output, dtype=jnp.bfloat16),
        jnp.asarray(x, dtype=jnp.bfloat16),
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
        jnp.where(active_experts[:, None, None], gradient, 0).astype(jnp.bfloat16) for gradient in results[2:5]
    )
    gradients = tuple(results[:2]) + routed_weight_gradients + tuple(results[5:8])
    return gradients, results[-1][0]


def raise_mok_like_failure() -> None:
    """Raise a typed-FFI error on the already-uniform mesh failure branch."""
    jax.ffi.ffi_call(
        FAILURE_FENCE_TARGET,
        (),
        has_side_effect=True,
        vmap_method="broadcast_all",
    )()
