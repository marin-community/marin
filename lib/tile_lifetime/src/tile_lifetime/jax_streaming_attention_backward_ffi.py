# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower generated streaming-attention reverse programs to JAX typed FFI.

The semantic input to this module is a reverse program recovered from ordinary
JAX VJP StableHLO.  Triton is used only as an ahead-of-time compiler for the
bounded physical skeleton.  The registered runtime library embeds the emitted
CUBINs and depends on neither Torch nor Triton.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.ir import DType
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    StreamingAttentionBackwardMaximumVJP,
    StreamingAttentionBackwardProgram,
    StreamingAttentionBackwardProvenance,
    StreamingAttentionBackwardTileSchedule,
    verify_streaming_attention_backward_score_map_vjp,
)
from tile_lifetime.tensor_program import ScalarExpression, ScalarExpressionKind, serialize_scalar_expression

LOG2_E = 1.4426950408889634
TRITON_COMPILE_MODULE = "triton.tools.compile"
FORWARD_KERNEL_SOURCE = Path("lib/tile_lifetime/benchmarks/h100_generated_streaming_attention.py")
REVERSE_KERNEL_SOURCE = Path("lib/tile_lifetime/benchmarks/h100_generated_streaming_attention_backward.py")
LINKER_DIRECTIVE = re.compile(r"^//\s*tt-linker:\s*([A-Za-z_][A-Za-z0-9_]*):", re.MULTILINE)


class StreamingAttentionBackwardStatePolicy(StrEnum):
    """Whether the reverse consumes saved state or recomputes it on stream."""

    RECOMPUTE = "recompute"
    SAVED_OUTPUT_AND_LOG_SUM_EXP = "saved_output_and_log_sum_exp"


@dataclass(frozen=True)
class StreamingAttentionBackwardFfiBuffer:
    """One statically shaped buffer in the generated typed-FFI boundary."""

    name: str
    dtype: DType
    shape: tuple[int, ...]


@dataclass(frozen=True)
class TritonAotKernelPlan:
    """One explicit Triton AOT specialization of a generic physical skeleton."""

    source: Path
    kernel_name: str
    output_name: str
    signature: tuple[str, ...]
    grid: tuple[int, int, int]
    num_warps: int
    num_stages: int

    def compile_argv(
        self,
        *,
        repository: Path,
        output_directory: Path,
        target: str | None,
        python: Path,
    ) -> tuple[str, ...]:
        """Return the self-contained Triton AOT compile command."""
        arguments = (
            str(python),
            "-m",
            TRITON_COMPILE_MODULE,
            str(repository / self.source),
            "--kernel-name",
            self.kernel_name,
            "--out-name",
            self.output_name,
            "--out-path",
            str(output_directory / self.output_name),
            "--signature",
            ",".join(self.signature),
            "--grid",
            ",".join(str(value) for value in self.grid),
            "--num-warps",
            str(self.num_warps),
            "--num-stages",
            str(self.num_stages),
        )
        if target is None:
            return arguments
        return (*arguments, "--target", target)


@dataclass(frozen=True)
class GeneratedStreamingAttentionBackwardFfi:
    """A Torch-free runtime target plus its build-time AOT kernel plans."""

    target_name: str
    handler_symbol: str
    state_policy: StreamingAttentionBackwardStatePolicy
    inputs: tuple[StreamingAttentionBackwardFfiBuffer, ...]
    outputs: tuple[StreamingAttentionBackwardFfiBuffer, ...]
    aot_kernels: tuple[TritonAotKernelPlan, ...]
    handler_template: str
    semantic_fingerprint: str


@dataclass(frozen=True)
class CompiledStreamingAttentionBackwardFfi:
    """A compiled typed-FFI library and the generated sources that produced it."""

    generated: GeneratedStreamingAttentionBackwardFfi
    library: ctypes.CDLL
    source_path: Path
    library_path: Path
    aot_sources: tuple[Path, ...]
    compile_argv: tuple[str, ...]


@dataclass(frozen=True)
class _ScoreMapParameters:
    scale: float
    softcap: float | None
    causal: bool


def generate_streaming_attention_backward_ffi(
    program: StreamingAttentionBackwardProgram,
    schedule: StreamingAttentionBackwardTileSchedule,
    *,
    target_name: str,
    state_policy: StreamingAttentionBackwardStatePolicy = StreamingAttentionBackwardStatePolicy.RECOMPUTE,
    num_warps: int = 8,
    num_stages: int = 3,
) -> GeneratedStreamingAttentionBackwardFfi:
    """Generate an AOT/FFI plan from recovered generic reverse semantics."""
    _validate_program(program, schedule)
    if not target_name or not target_name.replace(".", "_").isidentifier():
        raise ValueError(f"FFI target does not map to a C identifier: {target_name!r}")
    if num_warps not in (4, 8) or num_stages not in (2, 3, 4):
        raise ValueError("streaming reverse AOT schedule requires 4/8 warps and 2-4 stages")

    query, key = program.forward.qk.inputs
    value = program.forward.pv.inputs[1]
    query_shape = query.shape
    key_shape = key.shape
    value_shape = value.shape
    output_shape = program.forward.finalize.output.shape
    parameters = _score_map_parameters(program.forward.score_map.expression, program.forward.qk.output.name)
    handler_symbol = target_name.replace(".", "_")
    inputs = [
        StreamingAttentionBackwardFfiBuffer("query", DType.BF16, query_shape),
        StreamingAttentionBackwardFfiBuffer("key", DType.BF16, key_shape),
        StreamingAttentionBackwardFfiBuffer("value", DType.BF16, value_shape),
    ]
    if state_policy is StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP:
        inputs.extend(
            (
                StreamingAttentionBackwardFfiBuffer("output", DType.BF16, output_shape),
                StreamingAttentionBackwardFfiBuffer(
                    "log_sum_exp",
                    DType.FP32,
                    (query_shape[0], query_shape[2], query_shape[1]),
                ),
            )
        )
    inputs.append(StreamingAttentionBackwardFfiBuffer("output_cotangent", DType.BF16, output_shape))
    outputs = (
        StreamingAttentionBackwardFfiBuffer("query_cotangent", DType.BF16, query_shape),
        StreamingAttentionBackwardFfiBuffer("key_cotangent", DType.BF16, key_shape),
        StreamingAttentionBackwardFfiBuffer("value_cotangent", DType.BF16, value_shape),
    )
    kernels: list[TritonAotKernelPlan] = []
    if state_policy is StreamingAttentionBackwardStatePolicy.RECOMPUTE:
        kernels.append(
            _forward_aot_plan(
                query_shape,
                key_shape,
                parameters,
                schedule,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    kernels.extend(
        _reverse_aot_plans(
            query_shape,
            key_shape,
            parameters,
            program.output_scale,
            schedule,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )
    semantic_fingerprint = _semantic_fingerprint(program, schedule, state_policy)
    handler_template = _handler_template(
        handler_symbol=handler_symbol,
        state_policy=state_policy,
        inputs=tuple(inputs),
        outputs=outputs,
        query_shape=query_shape,
    )
    return GeneratedStreamingAttentionBackwardFfi(
        target_name=target_name,
        handler_symbol=handler_symbol,
        state_policy=state_policy,
        inputs=tuple(inputs),
        outputs=outputs,
        aot_kernels=tuple(kernels),
        handler_template=handler_template,
        semantic_fingerprint=semantic_fingerprint,
    )


def register_streaming_attention_backward_ffi(
    compiled: CompiledStreamingAttentionBackwardFfi,
) -> None:
    """Register the self-contained AOT handler with JAX's CUDA backend."""
    handler = getattr(compiled.library, compiled.generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        compiled.generated.target_name,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def call_streaming_attention_backward_ffi(
    generated: GeneratedStreamingAttentionBackwardFfi,
    *,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    output_cotangent: jax.Array,
    output: jax.Array | None = None,
    log_sum_exp: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Invoke the generated reverse on JAX-owned buffers and execution stream."""
    arrays: dict[str, jax.Array] = {
        "query": query,
        "key": key,
        "value": value,
        "output_cotangent": output_cotangent,
    }
    if generated.state_policy is StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP:
        if output is None or log_sum_exp is None:
            raise ValueError("saved-state reverse requires output and log_sum_exp")
        arrays["output"] = output
        arrays["log_sum_exp"] = log_sum_exp
    elif output is not None or log_sum_exp is not None:
        raise ValueError("recompute reverse does not accept saved output or log_sum_exp")
    ordered = _validated_arrays(generated.inputs, arrays)
    result_shapes = tuple(
        jax.ShapeDtypeStruct(specification.shape, _jax_dtype(specification.dtype)) for specification in generated.outputs
    )
    results = jax.ffi.ffi_call(
        generated.target_name,
        result_shapes,
        vmap_method="broadcast_all",
    )(*ordered)
    query_cotangent, key_cotangent, value_cotangent = tuple(results)
    return query_cotangent, key_cotangent, value_cotangent


def compile_streaming_attention_backward_ffi(
    generated: GeneratedStreamingAttentionBackwardFfi,
    *,
    repository: Path,
    directory: Path,
    nvcc: Path,
    architecture: str,
    triton_target: str | None,
    python: Path | None = None,
) -> CompiledStreamingAttentionBackwardFfi:
    """AOT-compile the skeletons and build one self-contained typed-FFI DSO."""
    if not nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {nvcc}")
    interpreter = python or Path(sys.executable)
    directory.mkdir(parents=True, exist_ok=True)
    _write_aot_input_sources(generated, repository=repository, directory=directory)
    aot_sources: list[Path] = []
    launchers: dict[str, str] = {}
    for kernel in generated.aot_kernels:
        command = kernel.compile_argv(
            repository=directory,
            output_directory=directory,
            target=triton_target,
            python=interpreter,
        )
        subprocess.run(command, check=True, cwd=repository)
        candidates = sorted(directory.glob(f"{kernel.output_name}.*.c"))
        headers = sorted(directory.glob(f"{kernel.output_name}.*.h"))
        if len(candidates) != 1 or len(headers) != 1:
            raise RuntimeError(f"Triton AOT did not emit one C/header pair for {kernel.output_name}")
        aot_sources.append(candidates[0])
        match = LINKER_DIRECTIVE.search(headers[0].read_text())
        if match is None:
            raise RuntimeError(f"Triton AOT header has no linker directive: {headers[0]}")
        launchers[kernel.output_name] = match.group(1)

    pointer_counts = {
        "shuttle_streaming_forward": 9,
        "shuttle_streaming_dq": 8,
        "shuttle_streaming_dkdv": 8,
    }
    aot_declarations = "\n".join(
        _aot_launcher_declarations(launcher, pointer_counts[output_name]) for output_name, launcher in launchers.items()
    )
    handler_source = generated.handler_template
    substitutions = {
        "{aot_declarations}": aot_declarations,
        "{forward_launcher}": launchers.get("shuttle_streaming_forward", ""),
        "{dq_launcher}": launchers["shuttle_streaming_dq"],
        "{dkdv_launcher}": launchers["shuttle_streaming_dkdv"],
    }
    for marker, replacement in substitutions.items():
        handler_source = handler_source.replace(marker, replacement)
    source_path = directory / f"{generated.handler_symbol}.cu"
    library_path = directory / f"{generated.handler_symbol}.so"
    source_path.write_text(handler_source + "\n")
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    cuda_include_directory = nvcc.resolve().parent.parent / "include"
    command = (
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
        *(str(path) for path in aot_sources),
        "-o",
        str(library_path),
        "-cudart=none",
        *cuda_toolkit_link_flags(nvcc, runtime_search_path=True),
        *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart",)),
        "-lcuda",
    )
    subprocess.run(command, check=True)
    return CompiledStreamingAttentionBackwardFfi(
        generated=generated,
        library=ctypes.CDLL(str(library_path)),
        source_path=source_path,
        library_path=library_path,
        aot_sources=tuple(aot_sources),
        compile_argv=command,
    )


def _aot_launcher_declarations(launcher: str, pointer_count: int) -> str:
    arguments = ", ".join(f"CUdeviceptr pointer_{index}" for index in range(pointer_count))
    return f'extern "C" CUresult {launcher}(CUstream stream, {arguments});'


def _write_aot_input_sources(
    generated: GeneratedStreamingAttentionBackwardFfi,
    *,
    repository: Path,
    directory: Path,
) -> None:
    """Write import-compatible AOT inputs around a Triton 3.6 loader bug.

    ``triton.tools.compile`` executes source modules without first inserting
    them into ``sys.modules``.  Python dataclasses inspect that table when
    postponed annotations are enabled.  Removing only the future-annotations
    directive leaves kernel semantics unchanged and makes the expert-derived
    benchmark modules valid AOT inputs.  Both modules are copied so the reverse
    module's sibling import remains explicit and reproducible.
    """
    source_paths = {kernel.source for kernel in generated.aot_kernels}
    source_paths.add(FORWARD_KERNEL_SOURCE)
    for relative_path in source_paths:
        source = (repository / relative_path).read_text()
        sanitized = source.replace("from __future__ import annotations\n\n", "")
        destination = directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(sanitized)


def _validate_program(
    program: StreamingAttentionBackwardProgram,
    schedule: StreamingAttentionBackwardTileSchedule,
) -> None:
    if program.provenance is not StreamingAttentionBackwardProvenance.JAX_VJP_HLO_RECOVERY:
        raise ValueError("accepted JAX FFI lowering requires reverse semantics recovered from JAX VJP HLO")
    if program.maximum_vjp is not StreamingAttentionBackwardMaximumVJP.NORMALIZED_EXP_INVARIANT:
        raise ValueError("AOT reverse requires the explicit normalized-exp maximum-VJP rewrite")
    verify_streaming_attention_backward_score_map_vjp(program)
    query, key = program.forward.qk.inputs
    value = program.forward.pv.inputs[1]
    if any(value.dtype is not DType.BF16 for value in (query, key, value)):
        raise ValueError("AOT streaming reverse requires BF16 Q/K/V")
    if query.shape[0] != key.shape[0] or key.shape != value.shape:
        raise ValueError("AOT streaming reverse requires equal K/V and matched batch dimensions")
    if query.shape[1] != key.shape[1]:
        raise ValueError("the first AOT streaming reverse requires equal query/key lengths")
    if query.shape[-1] not in (64, 128) or key.shape[-1] != query.shape[-1]:
        raise ValueError("AOT streaming reverse supports equal head dimensions 64 and 128")
    if query.shape[2] % key.shape[2]:
        raise ValueError("query heads must be divisible by key/value heads")
    if schedule.query_heads_per_key_value_tile != query.shape[2] // key.shape[2]:
        raise ValueError("physical head packing disagrees with the recovered Contract index relation")
    if query.shape[1] % schedule.query_tile_size or key.shape[1] % schedule.key_value_tile_size:
        raise ValueError("AOT streaming reverse requires tile-aligned sequence lengths")
    parameters = _score_map_parameters(program.forward.score_map.expression, program.forward.qk.output.name)
    expected_domain = (
        StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR
        if parameters.causal
        else StreamingAttentionBackwardDomainTraversal.FULL
    )
    if schedule.domain_traversal is not expected_domain:
        raise ValueError("physical traversal disagrees with the recovered DomainRestriction")
    if parameters.causal and schedule.query_tile_size % schedule.key_value_tile_size:
        raise ValueError("causal AOT schedule requires query tiles divisible by key/value tiles")


def _forward_aot_plan(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    parameters: _ScoreMapParameters,
    schedule: StreamingAttentionBackwardTileSchedule,
    *,
    num_warps: int,
    num_stages: int,
) -> TritonAotKernelPlan:
    batch, sequence, query_heads, dimension = query_shape
    key_value_heads = key_shape[2]
    heads_per_program = schedule.query_heads_per_key_value_tile
    pointer_types = (
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*fp32:16",
        "*fp32:16",
        "*i1:16",
        "*i32:16",
        "*i32:16",
    )
    q_strides = _contiguous_strides(query_shape)
    k_strides = _contiguous_strides(key_shape)
    signature = (
        *pointer_types,
        str(sequence),
        str(query_heads),
        str(key_value_heads),
        _float_token(parameters.scale * LOG2_E),
        _float_token(parameters.softcap or 1.0),
        *(str(value) for value in q_strides),
        *(str(value) for value in k_strides),
        *(str(value) for value in k_strides),
        *(str(value) for value in q_strides),
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        str(schedule.query_tile_size),
        str(schedule.key_value_tile_size),
        str(dimension),
        str(heads_per_program),
        str(int(parameters.causal)),
        "0",
        "0",
        str(int(parameters.softcap is not None)),
    )
    return TritonAotKernelPlan(
        source=FORWARD_KERNEL_SOURCE,
        kernel_name="_streaming_grouped_query_forward",
        output_name="shuttle_streaming_forward",
        signature=signature,
        grid=(sequence // schedule.query_tile_size, batch * query_heads // heads_per_program, 1),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _reverse_aot_plans(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    parameters: _ScoreMapParameters,
    output_scale: float,
    schedule: StreamingAttentionBackwardTileSchedule,
    *,
    num_warps: int,
    num_stages: int,
) -> tuple[TritonAotKernelPlan, TritonAotKernelPlan]:
    batch, sequence, query_heads, dimension = query_shape
    key_value_heads = key_shape[2]
    q_strides = _contiguous_strides(query_shape)
    k_strides = _contiguous_strides(key_shape)
    scalar = (
        str(sequence),
        str(query_heads),
        str(key_value_heads),
        _float_token(parameters.scale),
        _float_token(parameters.scale * LOG2_E),
        _float_token(parameters.softcap or 1.0),
        _float_token(output_scale),
    )
    dq_signature = (
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*fp32:16",
        "*fp32:16",
        "*bf16:16",
        *scalar,
        *(str(value) for value in q_strides),
        *(str(value) for value in k_strides),
        *(str(value) for value in k_strides),
        *(str(value) for value in q_strides),
        *(str(value) for value in q_strides),
        *(str(value) for value in q_strides),
        str(schedule.query_tile_size),
        str(schedule.key_value_tile_size),
        str(dimension),
        str(schedule.query_heads_per_key_value_tile),
        str(int(parameters.causal)),
        str(int(parameters.softcap is not None)),
    )
    dkdv_signature = (
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*fp32:16",
        "*fp32:16",
        "*bf16:16",
        "*bf16:16",
        *scalar,
        *(str(value) for value in q_strides),
        *(str(value) for value in k_strides),
        *(str(value) for value in k_strides),
        *(str(value) for value in q_strides),
        *(str(value) for value in k_strides),
        *(str(value) for value in k_strides),
        str(schedule.query_tile_size),
        str(schedule.key_value_tile_size),
        str(dimension),
        str(schedule.query_heads_per_key_value_tile),
        str(int(parameters.causal)),
        str(int(parameters.softcap is not None)),
    )
    return (
        TritonAotKernelPlan(
            source=REVERSE_KERNEL_SOURCE,
            kernel_name="_streaming_dq_kernel",
            output_name="shuttle_streaming_dq",
            signature=dq_signature,
            grid=(sequence // schedule.query_tile_size, batch * key_value_heads, 1),
            num_warps=num_warps,
            num_stages=num_stages,
        ),
        TritonAotKernelPlan(
            source=REVERSE_KERNEL_SOURCE,
            kernel_name="_streaming_dkdv_kernel",
            output_name="shuttle_streaming_dkdv",
            signature=dkdv_signature,
            grid=(sequence // schedule.key_value_tile_size, batch * key_value_heads, 1),
            num_warps=num_warps,
            num_stages=num_stages,
        ),
    )


def _handler_template(
    *,
    handler_symbol: str,
    state_policy: StreamingAttentionBackwardStatePolicy,
    inputs: tuple[StreamingAttentionBackwardFfiBuffer, ...],
    outputs: tuple[StreamingAttentionBackwardFfiBuffer, ...],
    query_shape: tuple[int, ...],
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
    batch, sequence, query_heads, dimension = query_shape
    output_elements = batch * sequence * query_heads * dimension
    state_elements = batch * query_heads * sequence
    if state_policy is StreamingAttentionBackwardStatePolicy.RECOMPUTE:
        state_setup = f"""
  auto output_storage = scratch.Allocate(sizeof(__nv_bfloat16) * {output_elements}, alignof(__nv_bfloat16));
  auto lse_storage = scratch.Allocate(sizeof(float) * {state_elements}, alignof(float));
  auto position_storage = scratch.Allocate(sizeof(int32_t) * {sequence}, alignof(int32_t));
  if (!output_storage || !lse_storage || !position_storage) {{
    return ffi::Error::Internal("failed to allocate streaming reverse recompute state");
  }}
  auto* output = static_cast<__nv_bfloat16*>(*output_storage);
  auto* log_sum_exp = static_cast<float*>(*lse_storage);
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
      reinterpret_cast<CUdeviceptr>(output),
      reinterpret_cast<CUdeviceptr>(log_sum_exp),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(positions),
      reinterpret_cast<CUdeviceptr>(positions));
  if (driver_status != CUDA_SUCCESS) return DriverError("streaming forward", driver_status);
"""
    else:
        state_setup = "  CUresult driver_status = CUDA_SUCCESS;\n"
    return f"""// Generated from generic Contract/Fold/DomainRestriction reverse semantics; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{{aot_declarations}}

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

ffi::Error ShuttleStreamingAttentionBackward(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {input_arguments},
    {output_arguments}) {{
{input_pointers}
{output_pointers}
  auto dot_storage = scratch.Allocate(sizeof(float) * {state_elements}, alignof(float));
  if (!dot_storage) return ffi::Error::Internal("failed to allocate streaming reverse output-dot state");
  auto* output_dot = static_cast<float*>(*dot_storage);
{state_setup}
  driver_status = {{dq_launcher}}(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(key),
      reinterpret_cast<CUdeviceptr>(value),
      reinterpret_cast<CUdeviceptr>(output),
      reinterpret_cast<CUdeviceptr>(output_cotangent),
      reinterpret_cast<CUdeviceptr>(log_sum_exp),
      reinterpret_cast<CUdeviceptr>(output_dot),
      reinterpret_cast<CUdeviceptr>(query_cotangent_pointer));
  if (driver_status != CUDA_SUCCESS) return DriverError("query cotangent", driver_status);
  driver_status = {{dkdv_launcher}}(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(key),
      reinterpret_cast<CUdeviceptr>(value),
      reinterpret_cast<CUdeviceptr>(output_cotangent),
      reinterpret_cast<CUdeviceptr>(log_sum_exp),
      reinterpret_cast<CUdeviceptr>(output_dot),
      reinterpret_cast<CUdeviceptr>(key_cotangent_pointer),
      reinterpret_cast<CUdeviceptr>(value_cotangent_pointer));
  if (driver_status != CUDA_SUCCESS) return DriverError("key/value cotangents", driver_status);
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleStreamingAttentionBackwardBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{input_bindings}
{output_bindings};
}}
}}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {handler_symbol},
    ShuttleStreamingAttentionBackward,
    ShuttleStreamingAttentionBackwardBinding());

extern "C" int shuttle_streaming_attention_backward_ffi_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""


def _validated_arrays(
    specifications: tuple[StreamingAttentionBackwardFfiBuffer, ...],
    arrays: Mapping[str, jax.Array],
) -> tuple[jax.Array, ...]:
    expected_names = tuple(value.name for value in specifications)
    if set(arrays) != set(expected_names):
        raise ValueError(f"streaming reverse FFI inputs must be {expected_names}, found {tuple(arrays)}")
    ordered: list[jax.Array] = []
    for specification in specifications:
        array = arrays[specification.name]
        if array.shape != specification.shape:
            raise ValueError(
                f"streaming reverse FFI input {specification.name!r} must have shape "
                f"{specification.shape}, found {array.shape}"
            )
        expected_dtype = _jax_dtype(specification.dtype)
        if np.dtype(array.dtype) != expected_dtype:
            raise ValueError(
                f"streaming reverse FFI input {specification.name!r} must have dtype "
                f"{expected_dtype}, found {array.dtype}"
            )
        ordered.append(array)
    return tuple(ordered)


def _score_map_parameters(expression: ScalarExpression, raw_score_name: str) -> _ScoreMapParameters:
    causal = False
    softcap: float | None = None

    def literal(candidate: ScalarExpression) -> float | bool | None:
        return candidate.constant if candidate.kind is ScalarExpressionKind.CONSTANT else None

    def visit(candidate: ScalarExpression) -> float:
        nonlocal causal, softcap
        if candidate.kind is ScalarExpressionKind.SELECT:
            predicate, selected, rejected = candidate.operands
            if literal(rejected) != float("-inf"):
                raise ValueError("AOT score Map only supports DomainRestriction select-to-negative-infinity")
            if predicate.kind is not ScalarExpressionKind.LESS_EQUAL:
                raise ValueError("AOT score Map only supports an affine less-equal DomainRestriction")
            causal = True
            return visit(selected)
        if candidate.kind is ScalarExpressionKind.MULTIPLY:
            left, right = candidate.operands
            left_literal = literal(left)
            right_literal = literal(right)
            tanh_expression = right if left_literal is not None else left
            cap_literal = left_literal if left_literal is not None else right_literal
            if tanh_expression.kind is ScalarExpressionKind.TANH and cap_literal is not None:
                divided = tanh_expression.operands[0]
                if divided.kind is not ScalarExpressionKind.DIVIDE or literal(divided.operands[1]) != cap_literal:
                    raise ValueError("softcap Map must have form cap * tanh(score / cap)")
                softcap = float(cap_literal)
                return visit(divided.operands[0])
            raw = left if left.kind is ScalarExpressionKind.INPUT else right
            scale_literal = right_literal if raw is left else left_literal
            if raw.input_name != raw_score_name or scale_literal is None:
                raise ValueError("AOT score Map requires raw Contract output times one scalar")
            return float(scale_literal)
        raise ValueError(f"unsupported AOT score Map expression {candidate.kind.value}")

    scale = visit(expression)
    if not np.isfinite(scale):
        raise ValueError("score scale must be finite")
    return _ScoreMapParameters(scale=scale, softcap=softcap, causal=causal)


def _semantic_fingerprint(
    program: StreamingAttentionBackwardProgram,
    schedule: StreamingAttentionBackwardTileSchedule,
    state_policy: StreamingAttentionBackwardStatePolicy,
) -> str:
    payload = {
        "query": program.forward.qk.inputs[0].shape,
        "key": program.forward.qk.inputs[1].shape,
        "value": program.forward.pv.inputs[1].shape,
        "score_map": json.loads(serialize_scalar_expression(program.forward.score_map.expression)),
        "score_map_vjp": json.loads(serialize_scalar_expression(program.score_map_vjp.expression)),
        "output_scale": program.output_scale,
        "maximum_vjp": program.maximum_vjp.value,
        "reassociation": program.reassociation.value,
        "state_policy": state_policy.value,
        "schedule": {
            "query_tile": schedule.query_tile_size,
            "key_value_tile": schedule.key_value_tile_size,
            "head_group": schedule.query_heads_per_key_value_tile,
            "domain": schedule.domain_traversal.value,
            "fold_order": schedule.key_value_fold_order.value,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def _float_token(value: float) -> str:
    return repr(float(value))


def _ffi_dtype(dtype: DType) -> str:
    return {DType.BF16: "ffi::BF16", DType.FP32: "ffi::F32"}[dtype]


def _cpp_dtype(dtype: DType) -> str:
    return {DType.BF16: "__nv_bfloat16", DType.FP32: "float"}[dtype]


def _jax_dtype(dtype: DType) -> np.dtype:
    return np.dtype({DType.BF16: jnp.bfloat16, DType.FP32: jnp.float32}[dtype])
