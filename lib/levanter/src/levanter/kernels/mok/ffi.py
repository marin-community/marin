# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX shard boundary and custom VJP for the MoK BF16 megakernel."""

from __future__ import annotations

import importlib
import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

from levanter.kernels.mok.runtime import MokBf16Config


_FORWARD_TARGET = "levanter_mok_bf16_forward"
_BACKWARD_TARGET = "levanter_mok_bf16_backward"
_BATCH_AXIS_CANDIDATES = ("replica_dcn", "data", "expert")
_TARGETS_REGISTERED = False


def _native_extension() -> Any:
    try:
        native = importlib.import_module("mok._C")
    except ImportError as exc:
        raise RuntimeError(
            "MoK's native extension is unavailable; install mixture-of-kittens with the "
            "torch 2.11 cu130 and CUDA 13.0 toolchain"
        ) from exc
    return native


def register_ffi_targets() -> None:
    """Register the version-1 MoK XLA FFI handlers for CUDA."""

    global _TARGETS_REGISTERED
    if _TARGETS_REGISTERED:
        return
    native = _native_extension()
    missing = tuple(name for name in (_FORWARD_TARGET, _BACKWARD_TARGET) if not hasattr(native, name))
    if missing:
        raise RuntimeError(
            "MoK was built without the Levanter XLA FFI handlers: "
            + ", ".join(missing)
            + ". Rebuild mixture-of-kittens from the adapter branch."
        )
    for target in (_FORWARD_TARGET, _BACKWARD_TARGET):
        capsule = getattr(native, target)()
        jax.ffi.register_ffi_target(target, capsule, platform="CUDA", api_version=1)
        jax.ffi.register_ffi_target_as_batch_partitionable(target)
    _TARGETS_REGISTERED = True


def _row_major_layout(shape: tuple[int, ...]) -> tuple[int, ...]:
    # ffi_call uses major-to-minor order (unlike raw XLA's minor-to-major
    # layout), so a C-contiguous rank-3 buffer is explicitly (0, 1, 2).
    return tuple(range(len(shape)))


def _schedule_capacity(tokens: int, topk: int, ep_size: int, multiplier: float) -> int:
    factor = max(2, math.ceil(ep_size * multiplier))
    unaligned = tokens * topk * factor
    return ((unaligned + 255) // 256) * 256


def _scratch_bytes(
    name: str,
    *,
    tokens: int,
    hidden_size: int,
    topk: int,
    num_local_experts: int,
    routed_intermediate_size: int,
    shared_intermediate_size: int,
    ep_size: int,
    schedule_capacity: int,
    config: MokBf16Config,
) -> int:
    native = _native_extension()
    query = getattr(native, name, None)
    if query is None:
        raise RuntimeError(f"MoK's native extension is missing required scratch query {name}")
    size = int(
        query(
            tokens,
            hidden_size,
            topk,
            num_local_experts,
            routed_intermediate_size,
            shared_intermediate_size,
            ep_size,
            config.macrobatch_size,
            config.minibatch_size,
            schedule_capacity,
        )
    )
    if size < 0:
        raise RuntimeError(f"{name} returned a negative scratch size: {size}")
    return size


def _pack_weights(
    shared0_gate: jax.Array,
    shared0_up: jax.Array,
    shared0_down: jax.Array,
    shared1_gate: jax.Array,
    shared1_up: jax.Array,
    shared1_down: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    shared_gate = jnp.concatenate((shared0_gate.T, shared1_gate.T), axis=0)
    shared_up = jnp.concatenate((shared0_up.T, shared1_up.T), axis=0)
    shared_down = jnp.concatenate((shared0_down.T, shared1_down.T), axis=1)
    return (
        shared_gate,
        shared_up,
        shared_down,
        jnp.swapaxes(routed_gate, -1, -2),
        jnp.swapaxes(routed_up, -1, -2),
        jnp.swapaxes(routed_down, -1, -2),
    )


def _unpack_weight_grads(
    d_shared_gate: jax.Array,
    d_shared_up: jax.Array,
    d_shared_down: jax.Array,
    d_routed_gate: jax.Array,
    d_routed_up: jax.Array,
    d_routed_down: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    shared_intermediate_size = d_shared_gate.shape[0] // 2
    d_shared0_gate, d_shared1_gate = jnp.split(d_shared_gate, 2, axis=0)
    d_shared0_up, d_shared1_up = jnp.split(d_shared_up, 2, axis=0)
    d_shared0_down, d_shared1_down = jnp.split(d_shared_down, 2, axis=1)
    assert d_shared0_gate.shape[0] == shared_intermediate_size
    return (
        d_shared0_gate.T,
        d_shared0_up.T,
        d_shared0_down.T,
        d_shared1_gate.T,
        d_shared1_up.T,
        d_shared1_down.T,
        jnp.swapaxes(d_routed_gate, -1, -2),
        jnp.swapaxes(d_routed_up, -1, -2),
        jnp.swapaxes(d_routed_down, -1, -2),
    )


def _ffi_attributes(config: MokBf16Config) -> dict[str, np.generic]:
    return {
        "workspace_id": np.int64(config.workspace_id),
        "fwd_num_comm_sms": np.int64(config.fwd_num_comm_sms),
        "bwd_num_comm_sms": np.int64(config.bwd_num_comm_sms),
        "minibatch_size": np.int64(config.minibatch_size),
        "macrobatch_size": np.int64(config.macrobatch_size),
        "schedule_capacity_multiplier": np.float32(config.schedule_capacity_multiplier),
        "all_gather_top_experts_chunk_bytes": np.int64(config.all_gather_top_experts_chunk_bytes),
    }


def _forward_ffi(
    x: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    shared_gate: jax.Array,
    shared_up: jax.Array,
    shared_down: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
    *,
    ep_size: int,
    config: MokBf16Config,
) -> tuple[jax.Array, ...]:
    register_ffi_targets()
    tokens, hidden_size = x.shape
    topk = selected_experts.shape[1]
    num_local_experts, routed_intermediate_size, _ = routed_gate.shape
    shared_intermediate_size = shared_gate.shape[0]
    schedule_capacity = _schedule_capacity(tokens, topk, ep_size, config.schedule_capacity_multiplier)
    scratch_bytes = _scratch_bytes(
        "levanter_mok_bf16_forward_scratch_bytes_v1",
        tokens=tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_local_experts=num_local_experts,
        routed_intermediate_size=routed_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
        ep_size=ep_size,
        schedule_capacity=schedule_capacity,
        config=config,
    )
    output_metadata = (
        jax.ShapeDtypeStruct((tokens, hidden_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((config.macrobatch_size, hidden_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, shared_intermediate_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((config.macrobatch_size, routed_intermediate_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, shared_intermediate_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((config.macrobatch_size, routed_intermediate_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, shared_intermediate_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((config.macrobatch_size, routed_intermediate_size), jnp.bfloat16),
        jax.ShapeDtypeStruct((schedule_capacity,), jnp.int32),
        jax.ShapeDtypeStruct((schedule_capacity,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((num_local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((scratch_bytes,), jnp.uint8),
    )
    inputs = (
        x,
        selected_experts,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
    )
    return tuple(
        jax.ffi.ffi_call(
            _FORWARD_TARGET,
            output_metadata,
            # JAX 0.11 does not permit FfiEffect under checkpoint/remat. Every
            # native write is represented by a returned buffer consumed by the
            # custom VJP, so ordinary data dependencies provide sequencing.
            has_side_effect=False,
            vmap_method="broadcast_all",
            input_layouts=tuple(_row_major_layout(value.shape) for value in inputs),
            output_layouts=tuple(_row_major_layout(value.shape) for value in output_metadata),
        )(*inputs, **_ffi_attributes(config))
    )


def _backward_ffi(
    grad_y: jax.Array,
    primals: tuple[jax.Array, ...],
    forward_context: tuple[jax.Array, ...],
    *,
    ep_size: int,
    config: MokBf16Config,
) -> tuple[jax.Array, ...]:
    register_ffi_targets()
    x, selected_experts, router_weights, shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down = (
        primals
    )
    tokens, hidden_size = x.shape
    topk = selected_experts.shape[1]
    num_local_experts, routed_intermediate_size, _ = routed_gate.shape
    shared_intermediate_size = shared_gate.shape[0]
    schedule_capacity = _schedule_capacity(tokens, topk, ep_size, config.schedule_capacity_multiplier)
    scratch_bytes = _scratch_bytes(
        "levanter_mok_bf16_backward_scratch_bytes_v1",
        tokens=tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_local_experts=num_local_experts,
        routed_intermediate_size=routed_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
        ep_size=ep_size,
        schedule_capacity=schedule_capacity,
        config=config,
    )
    output_metadata = (
        jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(router_weights.shape, jnp.float32),
        jax.ShapeDtypeStruct(shared_gate.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(shared_up.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(shared_down.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(routed_gate.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(routed_up.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(routed_down.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct((scratch_bytes,), jnp.uint8),
    )
    inputs = _backward_ffi_inputs(grad_y, primals, forward_context)
    return tuple(
        jax.ffi.ffi_call(
            _BACKWARD_TARGET,
            output_metadata,
            has_side_effect=False,
            vmap_method="broadcast_all",
            input_layouts=tuple(_row_major_layout(value.shape) for value in inputs),
            output_layouts=tuple(_row_major_layout(value.shape) for value in output_metadata),
        )(*inputs, **_ffi_attributes(config))
    )


def _backward_ffi_inputs(
    grad_y: jax.Array,
    primals: tuple[jax.Array, ...],
    forward_context: tuple[jax.Array, ...],
) -> tuple[jax.Array, ...]:
    x, _selected_experts, router_weights, shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down = (
        primals
    )
    # The schedule residuals fully encode routing choices, so the native
    # backward ABI does not consume selected_experts again.
    return (
        grad_y,
        x,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
        *forward_context,
    )


@partial(jax.custom_vjp, nondiff_argnums=(12, 13))
def _mok_bf16_local(
    x: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    shared0_gate: jax.Array,
    shared0_up: jax.Array,
    shared0_down: jax.Array,
    shared1_gate: jax.Array,
    shared1_up: jax.Array,
    shared1_down: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
    ep_size: int,
    config: MokBf16Config,
) -> jax.Array:
    packed = _pack_weights(
        shared0_gate,
        shared0_up,
        shared0_down,
        shared1_gate,
        shared1_up,
        shared1_down,
        routed_gate,
        routed_up,
        routed_down,
    )
    return _forward_ffi(x, selected_experts, router_weights, *packed, ep_size=ep_size, config=config)[0]


def _mok_bf16_local_fwd(
    x: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    shared0_gate: jax.Array,
    shared0_up: jax.Array,
    shared0_down: jax.Array,
    shared1_gate: jax.Array,
    shared1_up: jax.Array,
    shared1_down: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
    ep_size: int,
    config: MokBf16Config,
) -> tuple[jax.Array, tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]]:
    packed = _pack_weights(
        shared0_gate,
        shared0_up,
        shared0_down,
        shared1_gate,
        shared1_up,
        shared1_down,
        routed_gate,
        routed_up,
        routed_down,
    )
    outputs = _forward_ffi(x, selected_experts, router_weights, *packed, ep_size=ep_size, config=config)
    primals = (x, selected_experts, router_weights, *packed)
    forward_context = outputs[1:12]
    return outputs[0], (primals, forward_context)


def _mok_bf16_local_bwd(
    ep_size: int,
    config: MokBf16Config,
    residual: tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]],
    grad_y: jax.Array,
) -> tuple[jax.Array | None, ...]:
    primals, forward_context = residual
    native_grads = _backward_ffi(grad_y, primals, forward_context, ep_size=ep_size, config=config)
    dx, d_router = native_grads[:2]
    canonical_weight_grads = _unpack_weight_grads(*native_grads[2:8])
    return dx, None, d_router, *canonical_weight_grads


_mok_bf16_local.defvjp(_mok_bf16_local_fwd, _mok_bf16_local_bwd)


def _validate_shapes(
    x: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    shared_weights: tuple[jax.Array, ...],
    routed_weights: tuple[jax.Array, ...],
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape [tokens, hidden], got {x.shape}")
    if selected_experts.ndim != 2 or selected_experts.shape[0] != x.shape[0]:
        raise ValueError("selected_experts must have shape [tokens, topk]")
    if router_weights.shape != selected_experts.shape:
        raise ValueError("router_weights must have the same [tokens, topk] shape as selected_experts")
    hidden_size = x.shape[1]
    shared0_gate, shared0_up, shared0_down, shared1_gate, shared1_up, shared1_down = shared_weights
    shared_intermediate_size = shared0_gate.shape[1]
    expected_shared = (
        (hidden_size, shared_intermediate_size),
        (hidden_size, shared_intermediate_size),
        (shared_intermediate_size, hidden_size),
    )
    for expert, weights in enumerate(
        ((shared0_gate, shared0_up, shared0_down), (shared1_gate, shared1_up, shared1_down))
    ):
        if tuple(weight.shape for weight in weights) != expected_shared:
            raise ValueError(
                f"shared expert {expert} must have canonical gate/up/down shapes {expected_shared}, "
                f"got {tuple(weight.shape for weight in weights)}"
            )
    routed_gate, routed_up, routed_down = routed_weights
    if routed_gate.ndim != 3:
        raise ValueError("routed_gate must have shape [experts, hidden, intermediate]")
    num_experts, routed_hidden_size, routed_intermediate_size = routed_gate.shape
    expected_routed = (
        (num_experts, hidden_size, routed_intermediate_size),
        (num_experts, hidden_size, routed_intermediate_size),
        (num_experts, routed_intermediate_size, hidden_size),
    )
    if routed_hidden_size != hidden_size or tuple(weight.shape for weight in routed_weights) != expected_routed:
        raise ValueError(
            f"routed weights must have canonical gate/up/down shapes {expected_routed}, "
            f"got {tuple(weight.shape for weight in routed_weights)}"
        )


def _require_bf16_inputs(x: jax.Array, weights: tuple[jax.Array, ...]) -> None:
    if jnp.dtype(x.dtype) != jnp.dtype(jnp.bfloat16):
        raise TypeError(f"MoK BF16 requires x dtype bfloat16, got {x.dtype}")
    non_bf16 = tuple(str(weight.dtype) for weight in weights if jnp.dtype(weight.dtype) != jnp.dtype(jnp.bfloat16))
    if non_bf16:
        raise TypeError(f"MoK BF16 requires BF16 compute weights, found non-BF16 dtypes {non_bf16}")


def mok_bf16(
    x: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    shared0_gate: jax.Array,
    shared0_up: jax.Array,
    shared0_down: jax.Array,
    shared1_gate: jax.Array,
    shared1_up: jax.Array,
    shared1_down: jax.Array,
    routed_gate: jax.Array,
    routed_up: jax.Array,
    routed_down: jax.Array,
    *,
    config: MokBf16Config,
) -> jax.Array:
    """Run dropless MoK EP while preserving Marin's canonical parameter leaves.

    The shard boundary all-gathers the hidden/intermediate shards needed by the
    native kernel, gives each device only its expert slice, and packs the two
    shared experts only as BF16 compute values. Gradients are unpacked back into
    the nine original leaves before they reach the optimizer.
    """

    shared_weights = (
        shared0_gate,
        shared0_up,
        shared0_down,
        shared1_gate,
        shared1_up,
        shared1_down,
    )
    routed_weights = (routed_gate, routed_up, routed_down)
    _validate_shapes(x, selected_experts, router_weights, shared_weights, routed_weights)
    _require_bf16_inputs(x, (*shared_weights, *routed_weights))
    if jnp.dtype(selected_experts.dtype) != jnp.dtype(jnp.int32):
        raise TypeError(f"selected_experts must have dtype int32, got {selected_experts.dtype}")
    if jnp.dtype(router_weights.dtype) != jnp.dtype(jnp.float32):
        raise TypeError(f"router_weights must have dtype float32, got {router_weights.dtype}")

    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty or "expert" not in mesh.shape:
        raise ValueError("MoK requires a non-empty JAX mesh with an 'expert' axis")
    ep_size = int(mesh.shape["expert"])
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError(f"MoK expert-axis size must be one of 4, 8, 16, 32, 64, got {ep_size}")
    batch_axes = tuple(axis for axis in _BATCH_AXIS_CANDIDATES if axis in mesh.shape)
    duplicate_axes = tuple(axis for axis, size in mesh.shape.items() if axis != "expert" and int(size) != 1)
    if duplicate_axes:
        raise ValueError(
            "MoK currently requires the expert axis to cover the entire JAX world; nontrivial other axes: "
            + ", ".join(duplicate_axes)
        )
    batch_spec = P(batch_axes, None)
    shared_spec = P(None, None)
    routed_spec = P("expert", None, None)

    x = reshard(x, batch_spec)
    selected_experts = reshard(selected_experts, batch_spec)
    router_weights = reshard(router_weights, batch_spec)
    shared_weights = tuple(reshard(weight, shared_spec) for weight in shared_weights)
    routed_weights = tuple(reshard(weight, routed_spec) for weight in routed_weights)

    local_call = partial(_mok_bf16_local, ep_size=ep_size, config=config)
    return shard_map(
        local_call,
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
            batch_spec,
            *(shared_spec for _ in shared_weights),
            *(routed_spec for _ in routed_weights),
        ),
        out_specs=batch_spec,
        check_vma=False,
    )(x, selected_experts, router_weights, *shared_weights, *routed_weights)
