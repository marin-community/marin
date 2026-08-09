# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register and invoke generated normalized-exp Contract reverse code in JAX."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.cuda_normalized_exp_contract_reverse_codegen import (
    GeneratedCudaNormalizedExpContractReverseFfi,
)

_INPUTS = (
    ("lhs", jnp.bfloat16),
    ("rhs", jnp.bfloat16),
    ("saved_state", jnp.float32),
    ("fold_validity", jnp.bool_),
    ("row_cotangent", jnp.float32),
    ("selected_indices", jnp.int32),
    ("row_validity", jnp.bool_),
)


def register_cuda_normalized_exp_contract_reverse_ffi(
    generated: GeneratedCudaNormalizedExpContractReverseFfi,
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


def call_cuda_normalized_exp_contract_reverse_ffi(
    generated: GeneratedCudaNormalizedExpContractReverseFfi,
    inputs: Mapping[str, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Invoke the generated two-output reverse family through typed FFI."""
    expected_names = tuple(name for name, _ in _INPUTS)
    if set(inputs) != set(expected_names):
        raise ValueError(f"normalized-exp reverse FFI inputs must be {expected_names}, found {tuple(inputs)}")
    expected_shapes = {
        "lhs": (generated.rows, generated.reduction),
        "rhs": (generated.reduction, generated.fold_extent),
        "saved_state": (generated.rows,),
        "fold_validity": (generated.fold_extent,),
        "row_cotangent": (generated.rows,),
        "selected_indices": (generated.rows,),
        "row_validity": (generated.rows,),
    }
    arguments: list[jax.Array] = []
    for name, dtype in _INPUTS:
        argument = inputs[name]
        if argument.shape != expected_shapes[name]:
            raise ValueError(
                f"normalized-exp reverse FFI input {name!r} must have shape {expected_shapes[name]}, "
                f"found {argument.shape}"
            )
        if np.dtype(argument.dtype) != np.dtype(dtype):
            raise ValueError(
                f"normalized-exp reverse FFI input {name!r} must have dtype {np.dtype(dtype)}, "
                f"found {argument.dtype}"
            )
        arguments.append(argument)
    result_shapes = (
        jax.ShapeDtypeStruct((generated.rows, generated.reduction), jnp.bfloat16),
        jax.ShapeDtypeStruct((generated.reduction, generated.fold_extent), jnp.bfloat16),
    )
    results = jax.ffi.ffi_call(
        generated.target,
        result_shapes,
        vmap_method="broadcast_all",
    )(*arguments)
    return results[0], results[1]
