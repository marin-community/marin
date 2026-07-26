# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX FFI bindings for HybridEP's GB200 MNNVL transport."""

from __future__ import annotations

import atexit
import ctypes
import importlib
import os
import sys
import types
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import torch


_DISPATCH_TARGET = "levanter_hybridep_dispatch"
_COMBINE_TARGET = "levanter_hybridep_combine"
_COMBINE_WITH_PROBABILITIES_TARGET = "levanter_hybridep_combine_with_probabilities"
_LIBRARY_DLOPEN_MODE = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)

# Every dispatch produces an explicit scalar handle consumed by its matching
# combine, and successive layers are data-dependent through their hidden state.
# Keep the FFI calls effect-free from JAX's perspective so an enclosing layer
# checkpoint can rematerialize them; FfiEffect is unsupported by jax.remat.
_FFI_HAS_SIDE_EFFECT = False

_library: ctypes.CDLL | None = None
_runtime_signature: tuple[int, int, int, int, int] | None = None


def _load_hybrid_module(source_root: Path):
    package = sys.modules.get("deep_ep")
    if package is None:
        package = types.ModuleType("deep_ep")
        package.__path__ = [str(source_root / "deep_ep")]
        sys.modules["deep_ep"] = package
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return importlib.import_module("hybrid_ep_cpp")


def _register_targets(library: ctypes.CDLL) -> None:
    for target in (_DISPATCH_TARGET, _COMBINE_TARGET, _COMBINE_WITH_PROBABILITIES_TARGET):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(target)


def _last_error(library: ctypes.CDLL) -> str:
    function = library.levanter_hybridep_last_error
    function.argtypes = []
    function.restype = ctypes.c_char_p
    message = function()
    return message.decode() if message else "unknown error"


def ensure_hybridep_runtime(
    process_group,
    *,
    rank: int,
    world_size: int,
    device_index: int,
    source_root: Path,
    hidden_dim: int,
    tokens_per_rank: int,
    local_experts: int,
    dispatch_sms: int = 32,
    combine_sms: int = 32,
) -> None:
    """Initialize HybridEP and register its typed JAX FFI handlers."""
    global _library, _runtime_signature

    signature = (world_size, rank, hidden_dim, tokens_per_rank, local_experts)
    if _runtime_signature == signature:
        return
    if _runtime_signature is not None:
        shutdown_hybridep_runtime()

    source_root = source_root.resolve()
    module = _load_hybrid_module(source_root)
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        raise RuntimeError("HybridEP extension module has no library path")
    library = ctypes.CDLL(module_path, mode=_LIBRARY_DLOPEN_MODE)
    _register_targets(library)

    torch.cuda.set_device(device_index)
    initialize = library.levanter_hybridep_init
    initialize.argtypes = [
        ctypes.py_object,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    initialize.restype = ctypes.c_int
    status = initialize(
        process_group,
        rank,
        world_size,
        hidden_dim,
        tokens_per_rank,
        local_experts,
        dispatch_sms,
        combine_sms,
        os.fsencode(source_root / "deep_ep"),
    )
    if status != 0:
        raise RuntimeError(f"Failed to initialize HybridEP: {_last_error(library)}")
    _library = library
    _runtime_signature = signature


def shutdown_hybridep_runtime() -> None:
    """Release the process-local HybridEP runtime."""
    global _runtime_signature

    if _library is None or _runtime_signature is None:
        return
    function = _library.levanter_hybridep_shutdown
    function.argtypes = []
    function.restype = None
    function()
    _runtime_signature = None


atexit.register(shutdown_hybridep_runtime)


def _raw_dispatch(
    hidden: jax.Array,
    routing_map: jax.Array,
    probabilities: jax.Array,
    output_rows: int,
    local_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    output_shapes = (
        jax.ShapeDtypeStruct((output_rows, hidden.shape[1]), jnp.bfloat16),
        jax.ShapeDtypeStruct((output_rows,), jnp.float32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.float32),
    )
    dispatched_hidden, dispatched_probabilities, tokens_per_expert, handle_token = jax.ffi.ffi_call(
        _DISPATCH_TARGET,
        output_shapes,
        has_side_effect=_FFI_HAS_SIDE_EFFECT,
        vmap_method="broadcast_all",
    )(hidden, routing_map, probabilities)
    return dispatched_hidden, dispatched_probabilities, tokens_per_expert, handle_token


def _raw_combine(expert_hidden: jax.Array, handle_token: jax.Array, output_rows: int) -> jax.Array:
    output_shape = jax.ShapeDtypeStruct((output_rows, expert_hidden.shape[1]), jnp.bfloat16)
    return jax.ffi.ffi_call(
        _COMBINE_TARGET,
        output_shape,
        has_side_effect=_FFI_HAS_SIDE_EFFECT,
        vmap_method="broadcast_all",
    )(expert_hidden, handle_token)


def _raw_combine_with_probabilities(
    expert_hidden: jax.Array,
    expert_probabilities: jax.Array,
    handle_token: jax.Array,
    rematerialized_handle_token: jax.Array,
    output_rows: int,
    num_experts: int,
) -> tuple[jax.Array, jax.Array]:
    output_shapes = (
        jax.ShapeDtypeStruct((output_rows, expert_hidden.shape[1]), jnp.bfloat16),
        jax.ShapeDtypeStruct((output_rows, num_experts), jnp.float32),
    )
    combined_hidden, combined_probabilities = jax.ffi.ffi_call(
        _COMBINE_WITH_PROBABILITIES_TARGET,
        output_shapes,
        has_side_effect=_FFI_HAS_SIDE_EFFECT,
        vmap_method="broadcast_all",
    )(expert_hidden, expert_probabilities, handle_token, rematerialized_handle_token)
    return combined_hidden, combined_probabilities


def _cotangent_array(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
    *,
    shape: tuple[int, ...],
    dtype,
) -> jax.Array:
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(shape, dtype=dtype)
    return jnp.asarray(cotangent, dtype=dtype)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def hybridep_dispatch(
    hidden: jax.Array,
    routing_map: jax.Array,
    probabilities: jax.Array,
    output_rows: int,
    local_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Permute token rows to synthetic experts using a static output envelope."""
    return _raw_dispatch(hidden, routing_map, probabilities, output_rows, local_experts)


def _hybridep_dispatch_fwd(
    hidden: jax.Array,
    routing_map: jax.Array,
    probabilities: jax.Array,
    output_rows: int,
    local_experts: int,
):
    outputs = _raw_dispatch(hidden, routing_map, probabilities, output_rows, local_experts)
    return outputs, (hidden.shape, routing_map.shape[1], outputs[3])


def _hybridep_dispatch_bwd(
    output_rows: int,
    local_experts: int,
    residuals,
    cotangents,
):
    del local_experts
    hidden_shape, num_experts, rematerialized_handle = residuals
    hidden_cotangent = _cotangent_array(
        cotangents[0],
        shape=(output_rows, hidden_shape[1]),
        dtype=jnp.bfloat16,
    )
    probability_cotangent = _cotangent_array(
        cotangents[1],
        shape=(output_rows,),
        dtype=jnp.float32,
    )
    backward_handle = _cotangent_array(
        cotangents[3],
        shape=(),
        dtype=jnp.float32,
    )
    combined_hidden, combined_probabilities = _raw_combine_with_probabilities(
        hidden_cotangent,
        probability_cotangent,
        backward_handle,
        rematerialized_handle,
        hidden_shape[0],
        num_experts,
    )
    return combined_hidden, None, combined_probabilities


hybridep_dispatch.defvjp(_hybridep_dispatch_fwd, _hybridep_dispatch_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def hybridep_combine(
    expert_hidden: jax.Array,
    routing_map: jax.Array,
    probabilities: jax.Array,
    handle_token: jax.Array,
    local_experts: int,
) -> jax.Array:
    """Undo the matching dispatch; routing inputs define its backward permutation."""
    del probabilities
    return _raw_combine(expert_hidden, handle_token, routing_map.shape[0])


def _hybridep_combine_fwd(
    expert_hidden: jax.Array,
    routing_map: jax.Array,
    probabilities: jax.Array,
    handle_token: jax.Array,
    local_experts: int,
):
    output = _raw_combine(expert_hidden, handle_token, routing_map.shape[0])
    return output, (routing_map, probabilities, expert_hidden.shape)


def _hybridep_combine_bwd(
    local_experts: int,
    residuals,
    output_cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
):
    routing_map, probabilities, expert_shape = residuals
    cotangent = _cotangent_array(
        output_cotangent,
        shape=(routing_map.shape[0], expert_shape[1]),
        dtype=jnp.bfloat16,
    )
    dispatched_cotangent, _, _, backward_handle = _raw_dispatch(
        cotangent,
        routing_map,
        probabilities,
        expert_shape[0],
        local_experts,
    )
    return dispatched_cotangent, None, None, backward_handle


hybridep_combine.defvjp(_hybridep_combine_fwd, _hybridep_combine_bwd)
