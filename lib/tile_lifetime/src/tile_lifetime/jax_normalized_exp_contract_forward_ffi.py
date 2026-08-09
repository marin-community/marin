# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register and invoke generated normalized-exp Contract forward code in JAX."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.cuda_normalized_exp_contract_forward_codegen import GeneratedCudaNormalizedExpContractForwardFfi


def register_cuda_normalized_exp_contract_forward_ffi(
    generated: GeneratedCudaNormalizedExpContractForwardFfi,
    library: ctypes.CDLL,
) -> None:
    """Register one compiled generated handler with JAX's CUDA backend."""
    handler = getattr(library, generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        generated.target,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_cuda_normalized_exp_contract_forward_ffi(
    generated: GeneratedCudaNormalizedExpContractForwardFfi,
    inputs: Mapping[str, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Invoke the generated two-output compact forward through typed FFI."""
    specifications = (
        ("lhs", (generated.rows, generated.reduction), jnp.bfloat16),
        ("rhs", (generated.reduction, generated.fold_extent), jnp.bfloat16),
        ("fold_validity", (generated.fold_extent,), jnp.bool_),
        ("selected_indices", (generated.rows,), jnp.int32),
    )
    expected_names = tuple(name for name, _, _ in specifications)
    if set(inputs) != set(expected_names):
        raise ValueError(f"normalized-exp forward FFI inputs must be {expected_names}, found {tuple(inputs)}")
    arguments: list[jax.Array] = []
    for name, shape, dtype in specifications:
        argument = inputs[name]
        if argument.shape != shape:
            raise ValueError(
                f"normalized-exp forward FFI input {name!r} must have shape {shape}, found {argument.shape}"
            )
        if np.dtype(argument.dtype) != np.dtype(dtype):
            raise ValueError(
                f"normalized-exp forward FFI input {name!r} must have dtype {np.dtype(dtype)}, "
                f"found {argument.dtype}"
            )
        arguments.append(argument)
    result_shapes = (
        jax.ShapeDtypeStruct((generated.rows,), jnp.float32),
        jax.ShapeDtypeStruct((generated.rows,), jnp.float32),
    )
    results = jax.ffi.ffi_call(
        generated.target,
        result_shapes,
        vmap_method="broadcast_all",
    )(*arguments)
    return results[0], results[1]
