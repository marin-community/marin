# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose a generated streaming Contract/Fold forward as JAX typed FFI.

JAX remains responsible for differentiation.  This module owns only the early
physical realization of the recovered QK Contract, score Map,
DomainRestriction, normalized-exponential Fold, and PV Contract.  It returns
the natural BF16 output together with the minimal FP32 log-normalizer state
consumed by a separately scheduled generated reverse.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.ir import DType
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    FORWARD_KERNEL_SOURCE,
    LINKER_DIRECTIVE,
    StreamingAttentionBackwardFfiBuffer,
    StreamingAttentionBackwardFfiBufferLayout,
    StreamingAttentionBackwardStatePolicy,
    TritonAotKernelPlan,
    _aot_launcher_declarations,
    _forward_aot_plan,
    _run_triton_aot_compile,
    _score_map_parameters,
    _validate_program,
    _validated_output_layouts,
)
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardProgram,
    StreamingAttentionBackwardTileSchedule,
)
from tile_lifetime.tensor_program import serialize_scalar_expression

FORWARD_POINTER_COUNT = 9


@dataclass(frozen=True)
class GeneratedStreamingAttentionForwardFfi:
    """One early generated forward target and its explicit saved-state ABI."""

    target_name: str
    handler_symbol: str
    inputs: tuple[StreamingAttentionBackwardFfiBuffer, ...]
    outputs: tuple[StreamingAttentionBackwardFfiBuffer, ...]
    aot_kernel: TritonAotKernelPlan
    handler_template: str
    semantic_fingerprint: str
    reverse_state_policy: StreamingAttentionBackwardStatePolicy


@dataclass(frozen=True)
class CompiledStreamingAttentionForwardFfi:
    """A self-contained typed-FFI forward library and its generated sources."""

    generated: GeneratedStreamingAttentionForwardFfi
    library: ctypes.CDLL
    source_path: Path
    library_path: Path
    aot_source: Path
    compile_argv: tuple[str, ...]


def generate_streaming_attention_forward_ffi(
    program: StreamingAttentionBackwardProgram,
    schedule: StreamingAttentionBackwardTileSchedule,
    *,
    target_name: str,
    output_layouts: tuple[StreamingAttentionBackwardFfiBufferLayout, ...] = (),
    num_warps: int = 8,
    num_stages: int = 3,
) -> GeneratedStreamingAttentionForwardFfi:
    """Generate the early O-plus-log-normalizer family from generic semantics."""
    _validate_program(program, schedule)
    if not target_name or not target_name.replace(".", "_").isidentifier():
        raise ValueError(f"FFI target does not map to a C identifier: {target_name!r}")
    if num_warps not in (4, 8) or num_stages not in (2, 3, 4):
        raise ValueError("streaming forward AOT schedule requires 4/8 warps and 2-4 stages")

    query, key = program.forward.qk.inputs
    value = program.forward.pv.inputs[1]
    output_shape = program.forward.finalize.output.shape
    log_sum_exp_shape = (query.shape[0], query.shape[2], query.shape[1])
    inputs = (
        StreamingAttentionBackwardFfiBuffer("query", DType.BF16, query.shape),
        StreamingAttentionBackwardFfiBuffer("key", DType.BF16, key.shape),
        StreamingAttentionBackwardFfiBuffer("value", DType.BF16, value.shape),
    )
    output_shapes = {"output": output_shape, "log_sum_exp": log_sum_exp_shape}
    layout_by_name = _validated_output_layouts(output_shapes, output_layouts)
    outputs = (
        StreamingAttentionBackwardFfiBuffer("output", DType.BF16, output_shape, layout_by_name["output"]),
        StreamingAttentionBackwardFfiBuffer(
            "log_sum_exp",
            DType.FP32,
            log_sum_exp_shape,
            layout_by_name["log_sum_exp"],
        ),
    )
    parameters = _score_map_parameters(program.forward.score_map.expression, program.forward.qk.output.name)
    kernel = _forward_aot_plan(
        query.shape,
        key.shape,
        outputs[0].strides,
        parameters,
        schedule,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    semantic_payload = {
        "query": query.shape,
        "key": key.shape,
        "value": value.shape,
        "score_map": json.loads(serialize_scalar_expression(program.forward.score_map.expression)),
        "fold_state": ("maximum", "sum_exp", "weighted_value"),
        "saved_state": "log_sum_exp",
        "output_layouts": {output.name: output.layout for output in outputs},
        "schedule": {
            "query_tile": schedule.query_tile_size,
            "key_value_tile": schedule.key_value_tile_size,
            "head_group": schedule.query_heads_per_key_value_tile,
            "domain": schedule.domain_traversal.value,
            "fold_order": schedule.key_value_fold_order.value,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    handler_symbol = target_name.replace(".", "_")
    return GeneratedStreamingAttentionForwardFfi(
        target_name=target_name,
        handler_symbol=handler_symbol,
        inputs=inputs,
        outputs=outputs,
        aot_kernel=kernel,
        handler_template=_forward_handler_template(
            handler_symbol=handler_symbol,
            inputs=inputs,
            outputs=outputs,
        ),
        semantic_fingerprint=fingerprint,
        reverse_state_policy=StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP,
    )


def register_streaming_attention_forward_ffi(compiled: CompiledStreamingAttentionForwardFfi) -> None:
    """Register the self-contained forward handler with JAX's CUDA backend."""
    handler = getattr(compiled.library, compiled.generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        compiled.generated.target_name,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_streaming_attention_forward_ffi(
    generated: GeneratedStreamingAttentionForwardFfi,
    *,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Invoke the early forward and return O plus its generic saved Fold state."""
    arrays = _validated_arrays(generated.inputs, {"query": query, "key": key, "value": value})
    result_shapes = tuple(
        jax.ShapeDtypeStruct(specification.shape, _jax_dtype(specification.dtype)) for specification in generated.outputs
    )
    results = jax.ffi.ffi_call(
        generated.target_name,
        result_shapes,
        vmap_method="broadcast_all",
        input_layouts=tuple(specification.jax_layout for specification in generated.inputs),
        output_layouts=tuple(specification.jax_layout for specification in generated.outputs),
    )(*arrays)
    output, log_sum_exp = results
    return output, log_sum_exp


def compile_streaming_attention_forward_ffi(
    generated: GeneratedStreamingAttentionForwardFfi,
    *,
    repository: Path,
    directory: Path,
    nvcc: Path,
    architecture: str,
    triton_target: str | None,
    python: Path | None = None,
) -> CompiledStreamingAttentionForwardFfi:
    """AOT-compile one generated forward and build a Torch-free typed-FFI DSO."""
    if triton_target is not None:
        raise ValueError("Triton AOT cross-target compilation is unsupported; use the current-device target")
    if not nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {nvcc}")
    interpreter = python or Path(sys.executable)
    directory.mkdir(parents=True, exist_ok=True)
    _write_forward_aot_input_source(repository=repository, directory=directory)
    command = generated.aot_kernel.compile_argv(
        repository=directory,
        output_directory=directory,
        target=triton_target,
        python=interpreter,
    )
    cache_directory = directory / ".triton-cache"
    _run_triton_aot_compile(command, repository=repository, cache_directory=cache_directory)
    sources = sorted(directory.glob(f"{generated.aot_kernel.output_name}.*.c"))
    headers = sorted(directory.glob(f"{generated.aot_kernel.output_name}.*.h"))
    if len(sources) != 1 or len(headers) != 1:
        raise RuntimeError("Triton AOT did not emit one forward C/header pair")
    match = LINKER_DIRECTIVE.search(headers[0].read_text())
    if match is None:
        raise RuntimeError(f"Triton AOT header has no linker directive: {headers[0]}")
    launcher = match.group(1)
    handler_source = generated.handler_template.replace(
        "{aot_declaration}",
        _aot_launcher_declarations(launcher, FORWARD_POINTER_COUNT),
    ).replace("{forward_launcher}", launcher)
    source_path = directory / f"{generated.handler_symbol}.cu"
    library_path = directory / f"{generated.handler_symbol}.so"
    source_path.write_text(handler_source + "\n")
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    cuda_include_directory = nvcc.resolve().parent.parent / "include"
    compile_argv = (
        str(nvcc),
        "-std=c++17",
        "-O3",
        f"-arch={architecture}",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-I",
        str(include_directory),
        "-I",
        str(cuda_include_directory),
        str(source_path),
        str(sources[0]),
        "-o",
        str(library_path),
        "-cudart=none",
        *cuda_toolkit_link_flags(nvcc, runtime_search_path=True),
        *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart",)),
        "-lcuda",
    )
    subprocess.run(compile_argv, check=True)
    return CompiledStreamingAttentionForwardFfi(
        generated=generated,
        library=ctypes.CDLL(str(library_path)),
        source_path=source_path,
        library_path=library_path,
        aot_source=sources[0],
        compile_argv=compile_argv,
    )


def _write_forward_aot_input_source(*, repository: Path, directory: Path) -> None:
    source = (repository / FORWARD_KERNEL_SOURCE).read_text()
    sanitized = source.replace("from __future__ import annotations\n\n", "")
    destination = directory / FORWARD_KERNEL_SOURCE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(sanitized)


def _forward_handler_template(
    *,
    handler_symbol: str,
    inputs: tuple[StreamingAttentionBackwardFfiBuffer, ...],
    outputs: tuple[StreamingAttentionBackwardFfiBuffer, ...],
) -> str:
    input_arguments = ",\n    ".join(
        f"ffi::Buffer<{_ffi_dtype(value.dtype)}, {len(value.shape)}> {value.name}_buffer" for value in inputs
    )
    output_arguments = ",\n    ".join(
        f"ffi::Result<ffi::Buffer<{_ffi_dtype(value.dtype)}, {len(value.shape)}>> {value.name}" for value in outputs
    )
    input_pointers = "\n".join(
        f"  auto* {value.name} = reinterpret_cast<{_cpp_dtype(value.dtype)}*>({value.name}_buffer.typed_data());"
        for value in inputs
    )
    output_pointers = "\n".join(
        f"  auto* {value.name}_pointer = reinterpret_cast<{_cpp_dtype(value.dtype)}*>({value.name}->typed_data());"
        for value in outputs
    )
    input_bindings = "\n".join(
        f"      .Arg<ffi::Buffer<{_ffi_dtype(value.dtype)}, {len(value.shape)}>>()" for value in inputs
    )
    output_bindings = "\n".join(
        f"      .Ret<ffi::Buffer<{_ffi_dtype(value.dtype)}, {len(value.shape)}>>()" for value in outputs
    )
    sequence = inputs[0].shape[1]
    return f"""// Generated from generic Contract/Fold/DomainRestriction forward semantics; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{{aot_declaration}}

namespace ffi = xla::ffi;

namespace {{
std::atomic<int> call_count{{0}};

__global__ void ShuttleIotaKernel(int32_t* output, int extent) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < extent) output[index] = index;
}}

ffi::Error DriverError(const char* stage, CUresult result) {{
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  return ffi::Error::Internal(std::string(stage) + ": " + (name == nullptr ? "unknown CUDA driver error" : name));
}}

ffi::Error ShuttleStreamingAttentionForward(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {input_arguments},
    {output_arguments}) {{
{input_pointers}
{output_pointers}
  auto position_storage = scratch.Allocate(sizeof(int32_t) * {sequence}, alignof(int32_t));
  if (!position_storage) return ffi::Error::Internal("failed to allocate streaming forward position state");
  auto* positions = static_cast<int32_t*>(*position_storage);
  constexpr int kPositionThreads = 256;
  ShuttleIotaKernel<<<({sequence} + kPositionThreads - 1) / kPositionThreads, kPositionThreads, 0, stream>>>(
      positions, {sequence});
  cudaError_t runtime_status = cudaGetLastError();
  if (runtime_status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttleIotaKernel: ") + cudaGetErrorString(runtime_status));
  }}
  CUresult driver_status = {{forward_launcher}}(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(key),
      reinterpret_cast<CUdeviceptr>(value),
      reinterpret_cast<CUdeviceptr>(output_pointer),
      reinterpret_cast<CUdeviceptr>(log_sum_exp_pointer),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(positions),
      reinterpret_cast<CUdeviceptr>(positions));
  if (driver_status != CUDA_SUCCESS) return DriverError("streaming forward", driver_status);
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleStreamingAttentionForwardBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{input_bindings}
{output_bindings};
}}
}}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {handler_symbol},
    ShuttleStreamingAttentionForward,
    ShuttleStreamingAttentionForwardBinding());

extern "C" int shuttle_streaming_attention_forward_ffi_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""


def _validated_arrays(
    specifications: tuple[StreamingAttentionBackwardFfiBuffer, ...],
    arrays: Mapping[str, jax.Array],
) -> tuple[jax.Array, ...]:
    expected_names = tuple(value.name for value in specifications)
    if set(arrays) != set(expected_names):
        raise ValueError(f"streaming forward FFI inputs must be {expected_names}, found {tuple(arrays)}")
    ordered: list[jax.Array] = []
    for specification in specifications:
        array = arrays[specification.name]
        if array.shape != specification.shape:
            raise ValueError(
                f"streaming forward FFI input {specification.name!r} must have shape "
                f"{specification.shape}, found {array.shape}"
            )
        expected_dtype = _jax_dtype(specification.dtype)
        if np.dtype(array.dtype) != expected_dtype:
            raise ValueError(
                f"streaming forward FFI input {specification.name!r} must have dtype "
                f"{expected_dtype}, found {array.dtype}"
            )
        ordered.append(array)
    return tuple(ordered)


def _ffi_dtype(dtype: DType) -> str:
    if dtype is DType.BF16:
        return "ffi::DataType::BF16"
    if dtype is DType.FP32:
        return "ffi::DataType::F32"
    raise ValueError(f"unsupported streaming forward FFI dtype {dtype.value}")


def _cpp_dtype(dtype: DType) -> str:
    if dtype is DType.BF16:
        return "__nv_bfloat16"
    if dtype is DType.FP32:
        return "float"
    raise ValueError(f"unsupported streaming forward C++ dtype {dtype.value}")


def _jax_dtype(dtype: DType) -> np.dtype:
    if dtype is DType.BF16:
        return np.dtype(jnp.bfloat16)
    if dtype is DType.FP32:
        return np.dtype(jnp.float32)
    raise ValueError(f"unsupported streaming forward JAX dtype {dtype.value}")
