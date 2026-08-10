# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX typed-FFI calls for generic routed reverse physical families."""

import ctypes
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib

from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.xla_relation_program_recovery import (
    RoutedInputAdjointTypedFfiCodegenPlan,
    RoutedWeightGradientTypedFfiCodegenPlan,
)
from tile_lifetime.xla_routed_input_adjoint_ffi import GeneratedRoutedInputAdjointFfi
from tile_lifetime.xla_routed_weight_gradient_ffi import GeneratedGroupBatchedContractFfi
from tile_lifetime.xla_segmented_input_adjoint_ffi import (
    GeneratedSegmentedInputAdjointFfi,
    SegmentedInputAdjointFfiPlan,
)
from tile_lifetime.xla_source_indexed_fold_ffi import GeneratedSourceIndexedFoldFfi, SourceIndexedFoldTypedFfiPlan

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")
_JAX_DTYPES = {
    "bf16": jnp.bfloat16,
    "f32": jnp.float32,
    "pred": jnp.bool_,
    "s32": jnp.int32,
}


@dataclass(frozen=True)
class SegmentedInputAdjointCudaCompilePlan:
    """Exact standalone NVCC invocation for one generated segmented handler."""

    source_path: Path
    library_path: Path
    argv: tuple[str, ...]


def register_cuda_segmented_input_adjoint_ffi(
    generated: GeneratedSegmentedInputAdjointFfi,
    library: ctypes.CDLL,
) -> None:
    """Register one compiled segmented input-adjoint handler with JAX CUDA."""
    handler = getattr(library, generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        generated.target,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def segmented_input_adjoint_cuda_compile_plan(
    generated: GeneratedSegmentedInputAdjointFfi,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
    jaxlib_include: Path | None = None,
) -> SegmentedInputAdjointCudaCompilePlan:
    """Build a Torch-free compile plan using CUDA, cuBLAS, and XLA FFI."""
    if not nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {nvcc}")
    include_directory = jaxlib_include or Path(jaxlib.__file__).resolve().parent / "include"
    if not include_directory.is_dir():
        raise ValueError(f"jaxlib include directory does not exist: {include_directory}")
    source_path = directory / f"{generated.handler_symbol}.cu"
    library_path = directory / f"{generated.handler_symbol}.so"
    argv = (
        str(nvcc),
        "-std=c++17",
        "-O3",
        f"-arch={architecture}",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-I",
        str(include_directory),
        str(source_path),
        "-o",
        str(library_path),
        "-cudart=none",
        *cuda_toolkit_link_flags(nvcc, runtime_search_path=True),
        *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart", "cublas")),
    )
    return SegmentedInputAdjointCudaCompilePlan(source_path, library_path, argv)


def compile_cuda_segmented_input_adjoint_ffi(
    generated: GeneratedSegmentedInputAdjointFfi,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    """Compile and load one generated segmented input-adjoint handler."""
    plan = segmented_input_adjoint_cuda_compile_plan(
        generated,
        directory=directory,
        nvcc=nvcc,
        architecture=architecture,
    )
    directory.mkdir(parents=True, exist_ok=True)
    plan.source_path.write_text(generated.source + "\n")
    subprocess.run(plan.argv, check=True)
    return ctypes.CDLL(str(plan.library_path))


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


def call_cuda_segmented_input_adjoint_ffi(
    generated: GeneratedSegmentedInputAdjointFfi,
    plan: SegmentedInputAdjointFfiPlan,
    padded_cotangent: jax.Array,
    saved_pair: jax.Array,
    validity: jax.Array,
    down_input_adjoint_weight: jax.Array,
    gate_up_input_adjoint_weight: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Emit one fixed-capacity segmented Contract/Map/Contract call."""
    groups = plan.segment_count
    capacity = plan.capacity
    hidden = plan.input_features
    intermediate = plan.intermediate_features
    for value, shape, dtype, role in (
        (padded_cotangent, (groups, capacity, hidden), jnp.bfloat16, "padded_cotangent"),
        (saved_pair, (groups, capacity, 2 * intermediate), jnp.bfloat16, "saved_pair"),
        (validity, (groups, capacity), jnp.bool_, "validity"),
        (
            down_input_adjoint_weight,
            (groups, hidden, intermediate),
            jnp.bfloat16,
            "down_input_adjoint_weight",
        ),
        (
            gate_up_input_adjoint_weight,
            (groups, 2 * intermediate, hidden),
            jnp.bfloat16,
            "gate_up_input_adjoint_weight",
        ),
    ):
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{role} {value.shape}/{value.dtype} != segmented ABI {shape}/{dtype}")
    outputs = (
        jax.ShapeDtypeStruct((groups, capacity, 2 * intermediate), jnp.bfloat16),
        jax.ShapeDtypeStruct((groups, capacity, hidden), jnp.bfloat16),
    )
    result = jax.ffi.ffi_call(generated.target, outputs, vmap_method="sequential")(
        padded_cotangent,
        saved_pair,
        validity,
        down_input_adjoint_weight,
        gate_up_input_adjoint_weight,
    )
    assert isinstance(result, tuple)
    return result[0], result[1]


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
