# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Torch-free JAX typed-FFI boundary for generated partitioned Contracts."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cast_scalar_program import (
    CastScalarDType,
    CastScalarExpression,
    CastScalarKind,
    ScalarIndexRelation,
)
from tile_lifetime.cuda_partitioned_gemm_codegen import (
    GeneratedCudaPartitionedGemmFfi,
    PartitionedGemmFfiBuffer,
    partitioned_gemm_ffi_abi,
)
from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.partitioned_gemm_program import PartitionedGemmProgram


@dataclass(frozen=True)
class PartitionedGemmJaxFfiSpec:
    """JAX shapes and major-to-minor layouts for one generated call."""

    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    input_layouts: tuple[tuple[int, ...], ...]
    output_layouts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PartitionedGemmCudaCompilePlan:
    """Exact standalone NVCC invocation for one generated handler."""

    source_path: Path
    library_path: Path
    argv: tuple[str, ...]


def partitioned_gemm_jax_ffi_spec(generated: GeneratedCudaPartitionedGemmFfi) -> PartitionedGemmJaxFfiSpec:
    """Convert XLA minor-to-major ABI layouts to JAX FFI layout order."""
    return PartitionedGemmJaxFfiSpec(
        input_shapes=tuple(buffer.shape for buffer in generated.abi.inputs),
        output_shapes=tuple(buffer.shape for buffer in generated.abi.outputs),
        input_layouts=tuple(_to_jax_layout(buffer) for buffer in generated.abi.inputs),
        output_layouts=tuple(_to_jax_layout(buffer) for buffer in generated.abi.outputs),
    )


def register_cuda_partitioned_gemm_ffi(
    generated: GeneratedCudaPartitionedGemmFfi,
    library: ctypes.CDLL,
) -> None:
    """Register one generated partitioned Contract handler with JAX CUDA."""
    handler = getattr(library, generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        generated.target,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_cuda_partitioned_gemm_ffi(
    generated: GeneratedCudaPartitionedGemmFfi,
    operands: tuple[jax.Array, ...],
) -> tuple[jax.Array, ...]:
    """Invoke generated partitioned Contract code with its exact physical ABI."""
    spec = partitioned_gemm_jax_ffi_spec(generated)
    if len(operands) != len(generated.abi.inputs):
        raise ValueError(f"partitioned Contract expected {len(generated.abi.inputs)} operands, found {len(operands)}")
    for operand, buffer in zip(operands, generated.abi.inputs, strict=True):
        _validate_operand(operand, buffer)
    result_shapes = tuple(jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in spec.output_shapes)
    result = jax.ffi.ffi_call(
        generated.target,
        result_shapes,
        vmap_method="broadcast_all",
        input_layouts=spec.input_layouts,
        output_layouts=spec.output_layouts,
    )(*operands)
    return tuple(result)


def evaluate_partitioned_gemm_jax(
    program: PartitionedGemmProgram,
    operands: tuple[jax.Array, ...],
) -> tuple[jax.Array, ...]:
    """Evaluate the decomposed generic Contract/Map program with ordinary JAX."""
    abi = partitioned_gemm_ffi_abi(program)
    if len(operands) != len(abi.inputs):
        raise ValueError(f"partitioned Contract expected {len(abi.inputs)} operands, found {len(operands)}")
    for operand, buffer in zip(operands, abi.inputs, strict=True):
        _validate_operand(operand, buffer)
    m, _, k = program.shape
    lhs = operands[0].reshape(m, k)
    boundaries = tuple(
        jax.lax.dot_general(
            lhs,
            rhs,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        for rhs in operands[1:]
    )
    outputs: list[jax.Array] = []
    for finalization in program.scalar_finalizations:
        values: dict[str, jax.Array] = {}
        for partition_index, scalar_input in zip(
            finalization.source_partitions, finalization.program.inputs, strict=True
        ):
            assert scalar_input.input_name is not None and scalar_input.input_index is not None
            if scalar_input.input_index != ScalarIndexRelation(0, 0):
                raise ValueError("ordinary JAX partitioned Contract reference requires pointwise scalar Maps")
            values[scalar_input.input_name] = boundaries[partition_index]
        outputs.append(
            _evaluate_jax_expression(finalization.program.expression, values).reshape(abi.outputs[len(outputs)].shape)
        )
    for finalization in program.passthrough_finalizations:
        outputs.append(boundaries[finalization.source_partition].reshape(abi.outputs[len(outputs)].shape))
    return tuple(outputs)


def partitioned_gemm_cuda_compile_plan(
    generated: GeneratedCudaPartitionedGemmFfi,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
    jaxlib_include: Path | None = None,
) -> PartitionedGemmCudaCompilePlan:
    """Build a Torch-free compile plan using only CUDA runtime and XLA FFI."""
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
        *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart",)),
    )
    return PartitionedGemmCudaCompilePlan(source_path, library_path, argv)


def compile_cuda_partitioned_gemm_ffi(
    generated: GeneratedCudaPartitionedGemmFfi,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    """Compile and load one generated partitioned Contract handler."""
    plan = partitioned_gemm_cuda_compile_plan(
        generated,
        directory=directory,
        nvcc=nvcc,
        architecture=architecture,
    )
    directory.mkdir(parents=True, exist_ok=True)
    plan.source_path.write_text(generated.source + "\n")
    subprocess.run(plan.argv, check=True)
    return ctypes.CDLL(str(plan.library_path))


def _to_jax_layout(buffer: PartitionedGemmFfiBuffer) -> tuple[int, ...]:
    if sorted(buffer.minor_to_major) != list(range(len(buffer.shape))):
        raise ValueError(f"physical layout must permute all axes, found {buffer.minor_to_major}")
    return tuple(reversed(buffer.minor_to_major))


def _validate_operand(operand: jax.Array, buffer: PartitionedGemmFfiBuffer) -> None:
    if operand.shape != buffer.shape:
        raise ValueError(
            f"partitioned Contract input {buffer.name!r} must have shape {buffer.shape}, found {operand.shape}"
        )
    if np.dtype(operand.dtype) != np.dtype(jnp.bfloat16):
        raise ValueError(f"partitioned Contract input {buffer.name!r} must have dtype bfloat16, found {operand.dtype}")


def _evaluate_jax_expression(
    expression: CastScalarExpression,
    inputs: dict[str, jax.Array],
) -> jax.Array:
    if expression.kind is CastScalarKind.INPUT:
        assert expression.input_name is not None
        return inputs[expression.input_name]
    if expression.kind is CastScalarKind.CONSTANT:
        assert expression.constant is not None
        value = jnp.asarray(expression.constant)
    else:
        operands = tuple(_evaluate_jax_expression(operand, inputs) for operand in expression.operands)
        if expression.kind is CastScalarKind.CONVERT:
            value = operands[0]
        elif expression.kind is CastScalarKind.NEGATE:
            value = -operands[0]
        elif expression.kind is CastScalarKind.ADD:
            value = operands[0] + operands[1]
        elif expression.kind is CastScalarKind.SUBTRACT:
            value = operands[0] - operands[1]
        elif expression.kind is CastScalarKind.MULTIPLY:
            value = operands[0] * operands[1]
        elif expression.kind is CastScalarKind.DIVIDE:
            value = operands[0] / operands[1]
        elif expression.kind is CastScalarKind.EXP:
            value = jnp.exp(operands[0])
        elif expression.kind is CastScalarKind.TANH:
            value = jnp.tanh(operands[0])
        else:
            assert expression.kind is CastScalarKind.SELECT
            value = jnp.where(operands[0], operands[1], operands[2])
    dtype = {
        CastScalarDType.BF16: jnp.bfloat16,
        CastScalarDType.F32: jnp.float32,
        CastScalarDType.PRED: jnp.bool_,
    }[expression.dtype]
    return value.astype(dtype)
