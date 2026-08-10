# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX typed-FFI calls for generated anonymous Contract/Map backends."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.cuda_contract_map_backend_codegen import (
    ContractMapBackendBuffer,
    GeneratedCudaContractMapBackendFfi,
)


@dataclass(frozen=True)
class ContractMapForwardFfiResult:
    """Logical forward output and values saved for the generic reverse."""

    output: jax.Array
    preactivation: jax.Array
    hidden: jax.Array


@dataclass(frozen=True)
class ContractMapReverseFfiResult:
    """Logical reverse outputs plus the explicit XLA-owned scratch result."""

    input_adjoint: jax.Array
    first_weight_adjoint: jax.Array
    second_weight_adjoint: jax.Array
    preactivation_adjoint_scratch: jax.Array


def register_cuda_contract_map_backend_ffi(
    generated: GeneratedCudaContractMapBackendFfi,
    library: ctypes.CDLL,
) -> None:
    """Register both generated policy-specific targets with JAX CUDA."""
    for target, symbol in (
        (generated.forward_target, generated.forward_handler_symbol),
        (generated.reverse_target, generated.reverse_handler_symbol),
    ):
        handler = getattr(library, symbol)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(handler), platform="CUDA", api_version=1)


def call_cuda_contract_map_backend_forward_ffi(
    generated: GeneratedCudaContractMapBackendFfi,
    activation: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
) -> ContractMapForwardFfiResult:
    """Invoke the generated forward without defining a custom AD rule."""
    inputs = (activation, first_weight, second_weight)
    for expected, actual in zip(generated.physical_abi.forward_inputs, inputs, strict=True):
        _require_buffer(expected.role, actual, expected.shape)
    shapes = tuple(jax.ShapeDtypeStruct(buffer.shape, jnp.bfloat16) for buffer in generated.physical_abi.forward_outputs)
    input_layouts = tuple(_jax_layout(buffer) for buffer in generated.physical_abi.forward_inputs)
    output_layouts = tuple(_jax_layout(buffer) for buffer in generated.physical_abi.forward_outputs)
    output, preactivation, hidden = jax.ffi.ffi_call(
        generated.forward_target,
        shapes,
        vmap_method="broadcast_all",
        input_layouts=input_layouts,
        output_layouts=output_layouts,
    )(*inputs)
    return ContractMapForwardFfiResult(output=output, preactivation=preactivation, hidden=hidden)


def call_cuda_contract_map_backend_reverse_ffi(
    generated: GeneratedCudaContractMapBackendFfi,
    activation: jax.Array,
    first_weight: jax.Array,
    second_weight: jax.Array,
    preactivation: jax.Array,
    hidden: jax.Array,
    output_cotangent: jax.Array,
) -> ContractMapReverseFfiResult:
    """Invoke the reverse mechanically derived from the same tensor program."""
    inputs = (activation, first_weight, second_weight, preactivation, hidden, output_cotangent)
    for expected, actual in zip(generated.physical_abi.reverse_inputs, inputs, strict=True):
        _require_buffer(expected.role, actual, expected.shape)
    buffers = (*generated.physical_abi.reverse_outputs, *generated.physical_abi.reverse_scratch_outputs)
    shapes = tuple(jax.ShapeDtypeStruct(buffer.shape, jnp.bfloat16) for buffer in buffers)
    input_layouts = tuple(_jax_layout(buffer) for buffer in generated.physical_abi.reverse_inputs)
    output_layouts = tuple(_jax_layout(buffer) for buffer in buffers)
    input_adjoint, first_weight_adjoint, second_weight_adjoint, scratch = jax.ffi.ffi_call(
        generated.reverse_target,
        shapes,
        vmap_method="broadcast_all",
        input_layouts=input_layouts,
        output_layouts=output_layouts,
    )(*inputs)
    return ContractMapReverseFfiResult(
        input_adjoint=input_adjoint,
        first_weight_adjoint=first_weight_adjoint,
        second_weight_adjoint=second_weight_adjoint,
        preactivation_adjoint_scratch=scratch,
    )


def _require_buffer(name: str, value: jax.Array, shape: tuple[int, int]) -> None:
    if value.shape != shape:
        raise ValueError(f"Contract/Map FFI input {name!r} must have shape {shape}, found {value.shape}")
    if np.dtype(value.dtype) != np.dtype(jnp.bfloat16):
        raise ValueError(f"Contract/Map FFI input {name!r} must have dtype bfloat16, found {value.dtype}")


def _jax_layout(buffer: ContractMapBackendBuffer) -> tuple[int, ...]:
    return tuple(reversed(buffer.minor_to_major))
