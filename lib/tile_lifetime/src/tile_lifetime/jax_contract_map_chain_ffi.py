# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Torch-free JAX typed-FFI boundary for generated Contract/Map chains."""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.cuda_contract_map_chain_codegen import GeneratedCudaContractMapChainFfi


def register_cuda_contract_map_chain_ffi(
    generated: GeneratedCudaContractMapChainFfi,
    library: ctypes.CDLL,
) -> None:
    """Register both generated handlers with JAX's CUDA backend."""
    for target, symbol in (
        (generated.forward_target, generated.forward_handler_symbol),
        (generated.reverse_target, generated.reverse_handler_symbol),
    ):
        handler = getattr(library, symbol)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )


def call_cuda_contract_map_chain_forward_ffi(
    generated: GeneratedCudaContractMapChainFfi,
    activation: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Execute the generated forward and return its explicit BF16 save set."""
    rows = generated.rows
    input_features = generated.input_features
    rank = generated.rank
    _require_bf16("activation", activation, (rows, input_features))
    _require_bf16("first_weight", first_weight, (input_features, rank))
    _require_bf16("second_weight", second_weight, (rank, input_features))
    result_shapes = (
        jax.ShapeDtypeStruct((rows, input_features), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, rank), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, rank), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, input_features), jnp.bfloat16),
    )
    results = jax.ffi.ffi_call(
        generated.forward_target,
        result_shapes,
        vmap_method="broadcast_all",
    )(activation, first_weight, second_weight)
    return results[0], results[1], results[2], results[3]


def call_cuda_contract_map_chain_reverse_ffi(
    generated: GeneratedCudaContractMapChainFfi,
    activation: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
    first_contract_output: jax.Array,
    hidden: jax.Array,
    second_contract_output: jax.Array,
    output_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute JAX-owned reverse Maps and generated generic adjoint Contracts."""
    rows = generated.rows
    input_features = generated.input_features
    rank = generated.rank
    _require_bf16("activation", activation, (rows, input_features))
    _require_bf16("first_weight", first_weight, (input_features, rank))
    _require_bf16("second_weight", second_weight, (rank, input_features))
    _require_bf16("first_contract_output", first_contract_output, (rows, rank))
    _require_bf16("hidden", hidden, (rows, rank))
    _require_bf16("second_contract_output", second_contract_output, (rows, input_features))
    _require_bf16("output_cotangent", output_cotangent, (rows, input_features))
    result_shapes = (
        jax.ShapeDtypeStruct((rows, input_features), jnp.bfloat16),
        jax.ShapeDtypeStruct((input_features, rank), jnp.bfloat16),
        jax.ShapeDtypeStruct((rank, input_features), jnp.bfloat16),
    )
    results = jax.ffi.ffi_call(
        generated.reverse_target,
        result_shapes,
        vmap_method="broadcast_all",
    )(
        activation,
        first_weight,
        second_weight,
        first_contract_output,
        hidden,
        second_contract_output,
        output_cotangent,
    )
    return results[0], results[1], results[2]


def _require_bf16(name: str, value: jax.Array, shape: tuple[int, int]) -> None:
    if value.shape != shape:
        raise ValueError(f"Contract/Map FFI input {name!r} must have shape {shape}, found {value.shape}")
    if np.dtype(value.dtype) != np.dtype(jnp.bfloat16):
        raise ValueError(f"Contract/Map FFI input {name!r} must have dtype bfloat16, found {value.dtype}")
