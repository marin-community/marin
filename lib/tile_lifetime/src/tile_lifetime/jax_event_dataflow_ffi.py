# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile, register, and invoke generated Event Tensor CUDA through JAX FFI."""

from __future__ import annotations

import ctypes
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_dynamic_event_dataflow_codegen import (
    CudaEventFfiBuffer,
    CudaEventFfiKind,
    CudaEventFfiLowering,
)
from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.event_dataflow import EventTensorPlan, event_tensor_runtime_inputs
from tile_lifetime.ir import DType


@dataclass(frozen=True)
class RuntimeEventFfiArguments:
    """JAX-owned payload and runtime relation tables for counted readiness."""

    input: jax.Array
    event_counts: jax.Array
    event_source_offsets: jax.Array
    event_sources: jax.Array
    maximum_count: int


@dataclass(frozen=True)
class CudaEventFfiCompilePlan:
    """One explicit generated-source compilation without framework headers."""

    source_path: Path
    library_path: Path
    argv: tuple[str, ...]


def runtime_event_ffi_arguments(
    plan: EventTensorPlan,
    payload: jax.Array,
) -> RuntimeEventFfiArguments:
    """Materialize one EventTensorPlan's runtime tables as JAX arrays."""
    runtime = event_tensor_runtime_inputs(plan)
    source_count = len(runtime.event_sources)
    if payload.shape != (source_count,):
        raise ValueError(f"runtime event input must have shape {(source_count,)}, found {payload.shape}")
    if np.dtype(payload.dtype) != np.dtype(jnp.float32):
        raise ValueError(f"runtime event input must have dtype float32, found {payload.dtype}")
    return RuntimeEventFfiArguments(
        input=payload,
        event_counts=jnp.asarray(runtime.event_initial_counts, dtype=jnp.int32),
        event_source_offsets=jnp.asarray(runtime.event_source_offsets, dtype=jnp.int32),
        event_sources=jnp.asarray(runtime.event_sources, dtype=jnp.int32),
        maximum_count=max(runtime.event_initial_counts),
    )


def register_cuda_event_ffi(generated: CudaEventFfiLowering, library: ctypes.CDLL) -> None:
    """Register one compiled generated handler with JAX's CUDA backend."""
    handler = getattr(library, generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        generated.target_name,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_cuda_runtime_event_ffi(
    generated: CudaEventFfiLowering,
    arguments: RuntimeEventFfiArguments,
) -> tuple[jax.Array, jax.Array]:
    """Launch generated runtime readiness with JAX-owned buffers and stream."""
    if generated.kind is not CudaEventFfiKind.RUNTIME_RELATION:
        raise ValueError(f"expected a runtime-relation FFI lowering, found {generated.kind.value}")
    arrays = {
        "input": arguments.input,
        "event_counts": arguments.event_counts,
        "event_source_offsets": arguments.event_source_offsets,
        "event_sources": arguments.event_sources,
    }
    ordered = _validated_arguments(generated.inputs, arrays)
    result_shapes = tuple(_result_shape(value) for value in generated.outputs)
    results = jax.ffi.ffi_call(
        generated.target_name,
        result_shapes,
        vmap_method="broadcast_all",
    )(
        *ordered,
        maximum_count=np.int64(arguments.maximum_count),
        event_count=np.int64(generated.outputs[1].shape[0]),
    )
    partials, output = tuple(results)
    return partials, output


def call_cuda_phased_pipeline_ffi(
    generated: CudaEventFfiLowering,
    *,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
) -> jax.Array:
    """Launch a generated phased Contract/Fold pipeline on JAX's stream."""
    if generated.kind is not CudaEventFfiKind.PHASED_PIPELINE:
        raise ValueError(f"expected a phased-pipeline FFI lowering, found {generated.kind.value}")
    arrays = {"query": query, "key": key, "value": value}
    ordered = _validated_arguments(generated.inputs, arrays)
    generations, depth, dimension = generated.inputs[1].shape
    return jax.ffi.ffi_call(
        generated.target_name,
        _result_shape(generated.outputs[0]),
        vmap_method="broadcast_all",
    )(
        *ordered,
        generation_count=np.int64(generations),
        pipeline_depth=np.int64(depth),
        dimension=np.int64(dimension),
    )


def call_cuda_segmented_contract_ffi(
    generated: CudaEventFfiLowering,
    *,
    source: jax.Array,
    weight: jax.Array,
    event_counts: jax.Array,
    event_offsets: jax.Array,
    edge_sources: jax.Array,
) -> jax.Array:
    """Execute a generated runtime RelationPlan/SegmentedContract body."""
    if generated.kind is not CudaEventFfiKind.SEGMENTED_CONTRACT:
        raise ValueError(f"expected a segmented-Contract FFI lowering, found {generated.kind.value}")
    arrays = {
        "source": source,
        "weight": weight,
        "event_counts": event_counts,
        "event_offsets": event_offsets,
        "edge_sources": edge_sources,
    }
    ordered = _validated_arguments(generated.inputs, arrays)
    return jax.ffi.ffi_call(
        generated.target_name,
        _result_shape(generated.outputs[0]),
        vmap_method="broadcast_all",
    )(*ordered)


def call_cuda_streaming_contract_fold_ffi(
    generated: CudaEventFfiLowering,
    *,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    domain_valid: jax.Array,
) -> jax.Array:
    """Execute generated QK/normalized-exp Fold/PV with staged reuse."""
    if generated.kind is not CudaEventFfiKind.STREAMING_CONTRACT_FOLD:
        raise ValueError(f"expected a streaming Contract/Fold FFI lowering, found {generated.kind.value}")
    arrays = {
        "query": query,
        "key": key,
        "value": value,
        "domain_valid": domain_valid,
    }
    ordered = _validated_arguments(generated.inputs, arrays)
    return jax.ffi.ffi_call(
        generated.target_name,
        _result_shape(generated.outputs[0]),
        vmap_method="broadcast_all",
    )(*ordered)


def cuda_event_ffi_compile_plan(
    generated: CudaEventFfiLowering,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
    jaxlib_include: Path | None = None,
) -> CudaEventFfiCompilePlan:
    """Build the exact NVCC invocation for one generated typed-FFI library."""
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
    return CudaEventFfiCompilePlan(source_path, library_path, argv)


def compile_cuda_event_ffi(
    generated: CudaEventFfiLowering,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    """Compile and load one Torch-free generated Event Tensor handler."""
    plan = cuda_event_ffi_compile_plan(
        generated,
        directory=directory,
        nvcc=nvcc,
        architecture=architecture,
    )
    directory.mkdir(parents=True, exist_ok=True)
    source = generated.source + "\n"
    if not plan.source_path.exists() or plan.source_path.read_text() != source:
        plan.source_path.write_text(source)
    subprocess.run(plan.argv, check=True)
    return ctypes.CDLL(str(plan.library_path))


def _validated_arguments(
    specifications: tuple[CudaEventFfiBuffer, ...],
    arrays: Mapping[str, jax.Array],
) -> tuple[jax.Array, ...]:
    expected_names = tuple(value.name for value in specifications)
    if set(arrays) != set(expected_names):
        raise ValueError(f"Event Tensor FFI inputs must be {expected_names}, found {tuple(arrays)}")
    ordered: list[jax.Array] = []
    for specification in specifications:
        array = arrays[specification.name]
        if array.shape != specification.shape:
            raise ValueError(
                f"Event Tensor FFI input {specification.name!r} must have shape "
                f"{specification.shape}, found {array.shape}"
            )
        expected_dtype = _jax_dtype(specification.dtype)
        if np.dtype(array.dtype) != expected_dtype:
            raise ValueError(
                f"Event Tensor FFI input {specification.name!r} must have dtype "
                f"{expected_dtype}, found {array.dtype}"
            )
        ordered.append(array)
    return tuple(ordered)


def _result_shape(buffer: CudaEventFfiBuffer) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(buffer.shape, _jax_dtype(buffer.dtype))


def _jax_dtype(dtype: DType) -> np.dtype:
    return np.dtype({DType.FP32: jnp.float32, DType.INT32: jnp.int32}[dtype])
