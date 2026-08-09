# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register and invoke generated axis Folds through JAX typed FFI."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.cuda_axis_fold_codegen import GeneratedCudaAxisFoldFfi
from tile_lifetime.ir import DType


def register_cuda_axis_fold_ffi(generated: GeneratedCudaAxisFoldFfi, library: ctypes.CDLL) -> None:
    """Register one compiled generated handler with JAX's CUDA backend."""
    handler = getattr(library, generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        generated.target_name,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_cuda_axis_fold_ffi(
    generated: GeneratedCudaAxisFoldFfi,
    inputs: Mapping[str, jax.Array],
) -> tuple[jax.Array, ...]:
    """Invoke generated Fold kernels in their declared input/output order."""
    expected_names = tuple(value.name for value in generated.inputs)
    if set(inputs) != set(expected_names):
        raise ValueError(f"axis-Fold FFI inputs must be {expected_names}, found {tuple(inputs)}")
    arguments: list[jax.Array] = []
    for value in generated.inputs:
        argument = inputs[value.name]
        if argument.shape != value.shape:
            raise ValueError(f"axis-Fold FFI input {value.name!r} must have shape {value.shape}, found {argument.shape}")
        expected_dtype = _jax_dtype(value.dtype)
        if np.dtype(argument.dtype) != np.dtype(expected_dtype):
            raise ValueError(
                f"axis-Fold FFI input {value.name!r} must have dtype {np.dtype(expected_dtype)}, "
                f"found {argument.dtype}"
            )
        arguments.append(argument)
    result_shapes = tuple(jax.ShapeDtypeStruct(value.shape, _jax_dtype(value.dtype)) for value in generated.outputs)
    results = jax.ffi.ffi_call(
        generated.target_name,
        result_shapes,
        vmap_method="broadcast_all",
    )(*arguments)
    return tuple(results)


def _jax_dtype(dtype: DType) -> np.dtype:
    return np.dtype({DType.BF16: jnp.bfloat16, DType.FP32: jnp.float32}[dtype])
