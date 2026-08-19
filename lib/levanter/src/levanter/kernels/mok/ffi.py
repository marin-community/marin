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
# The kernel runs the shared path at the token width it is handed and the routed path at the
# latent width, so LatentMoE's two projections and its pre-dispatch RMSNorm live inside the call.
# ``latent_size == 0`` disables them: the routed width collapses onto the shared width and the
# three latent operands are passed with a zero-length latent axis.
_NUM_FORWARD_RESULTS = 16
_NUM_FORWARD_RESIDUALS = 14


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
    """Register the version-2 MoK XLA FFI handlers for CUDA."""

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
    shared_dim: int,
    latent_size: int,
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
    # ``latent_size`` is the third positional argument, which is why these queries are ``_v2``:
    # an extension built for the single-width ABI would silently read ``topk`` here.
    size = int(
        query(
            tokens,
            shared_dim,
            latent_size,
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
    latent_down: jax.Array,
    latent_norm_weight: jax.Array,
    latent_up: jax.Array,
) -> tuple[jax.Array, ...]:
    """Map Marin's canonical leaves onto the native operand layout.

    The two latent projections are transposed in opposite directions: Marin builds
    ``w_latent_down`` as ``(hidden, latent)`` and ``w_latent_up`` as ``(latent, hidden)``, while
    the kernel wants ``(latent, hidden)`` and ``(hidden, latent)`` respectively. The RMSNorm gain
    is float32 on both sides and is passed through untouched.
    """

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
        latent_down.T,
        latent_norm_weight,
        latent_up.T,
    )


def _unpack_weight_grads(
    d_shared_gate: jax.Array,
    d_shared_up: jax.Array,
    d_shared_down: jax.Array,
    d_routed_gate: jax.Array,
    d_routed_up: jax.Array,
    d_routed_down: jax.Array,
    d_latent_down: jax.Array,
    d_latent_norm_weight: jax.Array,
    d_latent_up: jax.Array,
) -> tuple[jax.Array, ...]:
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
        # The inverse of _pack_weights' two opposite transposes; getting either backwards would
        # corrupt optimizer state rather than raise, whenever hidden == latent.
        d_latent_down.T,
        d_latent_norm_weight,
        d_latent_up.T,
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
        "latent_size": np.int64(config.latent_size),
        "latent_norm_eps": np.float32(config.latent_norm_eps),
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
    latent_down: jax.Array,
    latent_norm_gain: jax.Array,
    latent_up: jax.Array,
    *,
    ep_size: int,
    config: MokBf16Config,
) -> tuple[jax.Array, ...]:
    register_ffi_targets()
    # ``x`` is the block's full-width token; the routed traffic the kernel dispatches is the
    # latent-width row. Never derive one from the other.
    tokens, shared_dim = x.shape
    routed_dim = config.latent_size or shared_dim
    latent_cols = config.latent_size
    latent_rows = tokens if config.latent_size else 0
    topk = selected_experts.shape[1]
    num_local_experts, routed_intermediate_size, _ = routed_gate.shape
    shared_intermediate_size = shared_gate.shape[0]
    schedule_capacity = _schedule_capacity(tokens, topk, ep_size, config.schedule_capacity_multiplier)
    scratch_bytes = _scratch_bytes(
        "levanter_mok_bf16_forward_scratch_bytes_v2",
        tokens=tokens,
        shared_dim=shared_dim,
        latent_size=config.latent_size,
        topk=topk,
        num_local_experts=num_local_experts,
        routed_intermediate_size=routed_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
        ep_size=ep_size,
        schedule_capacity=schedule_capacity,
        config=config,
    )
    output_metadata = (
        jax.ShapeDtypeStruct((tokens, shared_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((config.macrobatch_size, routed_dim), jnp.bfloat16),
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
        jax.ShapeDtypeStruct((tokens, latent_cols), jnp.bfloat16),
        jax.ShapeDtypeStruct((latent_rows,), jnp.float32),
        jax.ShapeDtypeStruct((tokens, latent_cols), jnp.bfloat16),
        jax.ShapeDtypeStruct((scratch_bytes,), jnp.uint8),
    )
    assert len(output_metadata) == _NUM_FORWARD_RESULTS
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
        latent_down,
        latent_norm_gain,
        latent_up,
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
    (
        x,
        selected_experts,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
        latent_down,
        latent_norm_gain,
        latent_up,
    ) = primals
    tokens, shared_dim = x.shape
    topk = selected_experts.shape[1]
    num_local_experts, routed_intermediate_size, _ = routed_gate.shape
    shared_intermediate_size = shared_gate.shape[0]
    schedule_capacity = _schedule_capacity(tokens, topk, ep_size, config.schedule_capacity_multiplier)
    scratch_bytes = _scratch_bytes(
        "levanter_mok_bf16_backward_scratch_bytes_v2",
        tokens=tokens,
        shared_dim=shared_dim,
        latent_size=config.latent_size,
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
        # The two latent projection gradients stay BF16 because the native accumulation writes
        # BF16; only the norm gain comes back in float32, matching its own dtype.
        jax.ShapeDtypeStruct(latent_down.shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(latent_norm_gain.shape, jnp.float32),
        jax.ShapeDtypeStruct(latent_up.shape, jnp.bfloat16),
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
    (
        x,
        _selected_experts,
        router_weights,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate,
        routed_up,
        routed_down,
        latent_down,
        latent_norm_gain,
        latent_up,
    ) = primals
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
        latent_down,
        latent_norm_gain,
        latent_up,
        *forward_context,
    )


@partial(jax.custom_vjp, nondiff_argnums=(15, 16))
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
    latent_down: jax.Array,
    latent_norm_weight: jax.Array,
    latent_up: jax.Array,
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
        latent_down,
        latent_norm_weight,
        latent_up,
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
    latent_down: jax.Array,
    latent_norm_weight: jax.Array,
    latent_up: jax.Array,
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
        latent_down,
        latent_norm_weight,
        latent_up,
    )
    outputs = _forward_ffi(x, selected_experts, router_weights, *packed, ep_size=ep_size, config=config)
    primals = (x, selected_experts, router_weights, *packed)
    # Results 1..14 are the backward's forward context; the trailing scratch buffer is dropped.
    forward_context = outputs[1 : 1 + _NUM_FORWARD_RESIDUALS]
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
    canonical_weight_grads = _unpack_weight_grads(*native_grads[2:11])
    return dx, None, d_router, *canonical_weight_grads


_mok_bf16_local.defvjp(_mok_bf16_local_fwd, _mok_bf16_local_bwd)


def disabled_latent_weights(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Canonical latent leaves with a zero-length latent axis, for ``latent_size == 0``.

    The latent axis is the zeroed one, which lands in a different position for each leaf:
    ``w_latent_down`` is ``(hidden, 0)``, the norm gain is ``(0,)`` and ``w_latent_up`` is
    ``(0, hidden)``. These cost no bytes and no FLOPs, so the control arm pays nothing for the
    operand slots it does not use.
    """

    shared_dim = x.shape[1]
    return (
        jnp.zeros((shared_dim, 0), dtype=jnp.bfloat16),
        jnp.zeros((0,), dtype=jnp.float32),
        jnp.zeros((0, shared_dim), dtype=jnp.bfloat16),
    )


def _validate_shapes(
    x: jax.Array,
    selected_experts: jax.Array,
    router_weights: jax.Array,
    shared_weights: tuple[jax.Array, ...],
    routed_weights: tuple[jax.Array, ...],
    latent_weights: tuple[jax.Array, ...],
    *,
    latent_size: int,
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape [tokens, hidden], got {x.shape}")
    if selected_experts.ndim != 2 or selected_experts.shape[0] != x.shape[0]:
        raise ValueError("selected_experts must have shape [tokens, topk]")
    if router_weights.shape != selected_experts.shape:
        raise ValueError("router_weights must have the same [tokens, topk] shape as selected_experts")
    # Two independent token widths: the shared experts read `shared_dim`, the routed experts and
    # every dispatched row read `routed_dim`. They coincide only when latent is disabled.
    shared_dim = x.shape[1]
    routed_dim = latent_size or shared_dim
    if shared_dim % 256:
        raise ValueError(f"MoK requires the hidden width to be divisible by 256, got {shared_dim}")
    if routed_dim % 256:
        raise ValueError(f"MoK requires the routed width to be divisible by 256, got {routed_dim}")
    shared0_gate, shared0_up, shared0_down, shared1_gate, shared1_up, shared1_down = shared_weights
    shared_intermediate_size = shared0_gate.shape[1]
    expected_shared = (
        (shared_dim, shared_intermediate_size),
        (shared_dim, shared_intermediate_size),
        (shared_intermediate_size, shared_dim),
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
        raise ValueError("routed_gate must have shape [experts, routed, intermediate]")
    num_experts, routed_gate_width, routed_intermediate_size = routed_gate.shape
    expected_routed = (
        (num_experts, routed_dim, routed_intermediate_size),
        (num_experts, routed_dim, routed_intermediate_size),
        (num_experts, routed_intermediate_size, routed_dim),
    )
    if routed_gate_width != routed_dim or tuple(weight.shape for weight in routed_weights) != expected_routed:
        raise ValueError(
            f"routed weights must have canonical gate/up/down shapes {expected_routed}, "
            f"got {tuple(weight.shape for weight in routed_weights)}"
        )
    latent_down, latent_norm_weight, latent_up = latent_weights
    # Canonical Marin orientation: down is (hidden, latent), up is (latent, hidden). The
    # zero-length axis for a disabled latent therefore lands in a different position in each.
    expected_latent = (
        (shared_dim, latent_size),
        (latent_size,),
        (latent_size, shared_dim),
    )
    if tuple(weight.shape for weight in latent_weights) != expected_latent:
        raise ValueError(
            f"latent weights must have canonical down/norm/up shapes {expected_latent} for "
            f"latent_size={latent_size}, got {tuple(weight.shape for weight in latent_weights)}"
        )
    if jnp.dtype(latent_norm_weight.dtype) != jnp.dtype(jnp.float32):
        raise TypeError(f"the latent RMSNorm gain must have dtype float32, got {latent_norm_weight.dtype}")
    del latent_down, latent_up


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
    latent_down: jax.Array | None = None,
    latent_norm_weight: jax.Array | None = None,
    latent_up: jax.Array | None = None,
    *,
    config: MokBf16Config,
) -> jax.Array:
    """Run dropless MoK EP while preserving Marin's canonical parameter leaves.

    The shard boundary all-gathers the hidden/intermediate shards needed by the native kernel,
    gives each device only its expert slice, and packs the two shared experts and the two latent
    projections only as BF16 compute values. Gradients are unpacked back into the twelve original
    leaves before they reach the optimizer.

    ``x`` is always the block's full-width token. When ``config.latent_size`` is non-zero the call
    owns LatentMoE end to end: it down-projects, RMSNorms, dispatches the latent-width row, runs
    the two shared experts at the full width, and up-projects the combined routed result back.
    When it is zero the latent leaves may be omitted, and zero-length stand-ins are substituted.
    """

    shared_weights = (
        shared0_gate,
        shared0_up,
        shared0_down,
        shared1_gate,
        shared1_up,
        shared1_down,
    )
    if any(weight is None for weight in shared_weights):
        raise ValueError("mok_bf16 requires all six shared expert weights; the fused slot is not optional")
    routed_weights = (routed_gate, routed_up, routed_down)
    supplied_latent = (latent_down, latent_norm_weight, latent_up)
    if any(weight is None for weight in supplied_latent):
        if any(weight is not None for weight in supplied_latent):
            raise ValueError("latent weights must be passed together or omitted together")
        if config.latent_size:
            raise ValueError(f"config.latent_size={config.latent_size} requires the three latent weights")
        latent_weights: tuple[jax.Array, ...] = disabled_latent_weights(x)
    else:
        latent_weights = tuple(weight for weight in supplied_latent if weight is not None)
    _validate_shapes(
        x,
        selected_experts,
        router_weights,
        shared_weights,
        routed_weights,
        latent_weights,
        latent_size=config.latent_size,
    )
    # The RMSNorm gain is float32 and is checked by _validate_shapes; everything else is BF16.
    _require_bf16_inputs(x, (*shared_weights, *routed_weights, latent_weights[0], latent_weights[2]))
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
    norm_spec = P(None)
    # The latent projections and the norm gain are replicated exactly like the shared experts;
    # their per-rank cotangent partials are psummed by shard_map's transpose rule.
    latent_specs = (shared_spec, norm_spec, shared_spec)

    x = reshard(x, batch_spec)
    selected_experts = reshard(selected_experts, batch_spec)
    router_weights = reshard(router_weights, batch_spec)
    shared_weights = tuple(reshard(weight, shared_spec) for weight in shared_weights)
    routed_weights = tuple(reshard(weight, routed_spec) for weight in routed_weights)
    latent_weights = tuple(reshard(weight, spec) for weight, spec in zip(latent_weights, latent_specs, strict=True))

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
            *latent_specs,
        ),
        out_specs=batch_spec,
        check_vma=False,
    )(x, selected_experts, router_weights, *shared_weights, *routed_weights, *latent_weights)
