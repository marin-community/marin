# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX typed-FFI calls for generic routed reverse physical families."""

import re

import jax
import jax.numpy as jnp

from tile_lifetime.xla_relation_program_recovery import (
    RoutedInputAdjointTypedFfiCodegenPlan,
    RoutedWeightGradientTypedFfiCodegenPlan,
)
from tile_lifetime.xla_routed_input_adjoint_ffi import GeneratedRoutedInputAdjointFfi
from tile_lifetime.xla_routed_weight_gradient_ffi import GeneratedGroupBatchedContractFfi
from tile_lifetime.xla_source_indexed_fold_ffi import GeneratedSourceIndexedFoldFfi, SourceIndexedFoldTypedFfiPlan

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")
_JAX_DTYPES = {
    "bf16": jnp.bfloat16,
    "f32": jnp.float32,
    "pred": jnp.bool_,
    "s32": jnp.int32,
}


def call_cuda_routed_input_adjoint_ffi(
    generated: GeneratedRoutedInputAdjointFfi,
    plan: RoutedInputAdjointTypedFfiCodegenPlan,
    operands: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array]:
    """Emit the generated Contract/Map/Contract/identity-Fold call."""
    if len(operands) != len(plan.operands):
        raise ValueError("routed input-adjoint operand count disagrees with its generated plan")
    for binding, operand in zip(plan.operands, operands, strict=True):
        _validate_array(operand, binding.value.shape, role=binding.role.value)
    index_map = plan.segmented_layout.index_map
    auxiliary = jax.ShapeDtypeStruct(
        (index_map.segment_count, index_map.padded_row_extent, index_map.logical_feature_extent),
        jnp.bfloat16,
    )
    fold_shape, fold_dtype = _shape_dtype(plan.fold_stage.output_shape)
    result = jax.ffi.ffi_call(
        generated.target,
        (auxiliary, jax.ShapeDtypeStruct(fold_shape, fold_dtype)),
        vmap_method="sequential",
    )(*operands)
    assert isinstance(result, tuple)
    if len(result) != 2:
        raise ValueError("routed input-adjoint FFI returned an unexpected result arity")
    return result[0], result[1]


def call_cuda_group_batched_contract_ffi(
    generated: GeneratedGroupBatchedContractFfi,
    plan: RoutedWeightGradientTypedFfiCodegenPlan,
    lhs: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Emit one generated group-batched Contract call."""
    operands = (lhs, rhs)
    for binding, operand in zip(plan.operands, operands, strict=True):
        _validate_array(operand, binding.value.shape, role=binding.role.value)
    shape, dtype = _shape_dtype(plan.contract.output_shape)
    result = jax.ffi.ffi_call(
        generated.target,
        jax.ShapeDtypeStruct(shape, dtype),
        vmap_method="sequential",
    )(lhs, rhs)
    assert isinstance(result, jax.Array)
    return result


def call_cuda_source_indexed_fold_ffi(
    generated: GeneratedSourceIndexedFoldFfi,
    plan: SourceIndexedFoldTypedFfiPlan,
    initial: jax.Array,
    source_indices: jax.Array,
    contributions: jax.Array,
) -> jax.Array:
    """Emit one deterministic source-indexed Fold call."""
    for value, expected, role in (
        (initial, plan.initial.shape, "initial"),
        (source_indices, plan.source_indices.shape, "source_indices"),
        (contributions, plan.contributions.shape, "contributions"),
    ):
        _validate_array(value, expected, role=role)
    shape, dtype = _shape_dtype(plan.output_shape)
    result = jax.ffi.ffi_call(
        generated.target,
        jax.ShapeDtypeStruct(shape, dtype),
        vmap_method="sequential",
    )(initial, source_indices, contributions)
    assert isinstance(result, jax.Array)
    return result


def _validate_array(value: jax.Array, expected: str, *, role: str) -> None:
    shape, dtype = _shape_dtype(expected)
    if value.shape != shape:
        raise ValueError(f"{role} shape {value.shape} != generated ABI {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{role} dtype {value.dtype} != generated ABI {dtype}")


def _shape_dtype(value: str) -> tuple[tuple[int, ...], jnp.dtype]:
    match = _ARRAY_SHAPE.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported physical array shape {value!r}")
    dtype_name = match.group("dtype")
    if dtype_name not in _JAX_DTYPES:
        raise ValueError(f"unsupported JAX FFI dtype {dtype_name!r}")
    return (
        tuple(int(dimension) for dimension in match.group("dims").split(",") if dimension),
        jnp.dtype(_JAX_DTYPES[dtype_name]),
    )
