# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a bounded CUDA skeleton for a partitioned generic Contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from tile_lifetime.ffi_command_buffer import (
    audit_ffi_command_buffer_eligibility,
    finalize_ffi_handler_source,
)
from tile_lifetime.partitioned_gemm_program import (
    GeneratedPartitionedGemmFinalization,
    PartitionedGemmProgram,
    generate_partitioned_gemm_finalization,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")
_MAX_SHARED_BYTES = 48 * 1024


@dataclass(frozen=True)
class PartitionedGemmFfiBuffer:
    """One physical typed-FFI buffer and its XLA layout."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    minor_to_major: tuple[int, ...]


@dataclass(frozen=True)
class PartitionedGemmFfiAbi:
    """Complete physical ABI for a generated partitioned Contract."""

    inputs: tuple[PartitionedGemmFfiBuffer, ...]
    outputs: tuple[PartitionedGemmFfiBuffer, ...]


@dataclass(frozen=True)
class GeneratedCudaPartitionedGemmFfi:
    """Generated typed-FFI CUDA source and its provenance."""

    target: str
    handler_symbol: str
    source: str
    semantic_digest: str
    source_digest: str
    threads: int
    shared_bytes: int
    abi: PartitionedGemmFfiAbi


@dataclass(frozen=True)
class PartitionedGemmSourceAudit:
    """Static ownership and physical-contract evidence for generated CUDA."""

    kernel_count: int
    segmented_rhs_count: int
    direct_output_count: int
    has_ordered_fp32_mainloop: bool
    has_bf16_rne_partition_boundary: bool
    has_command_buffer_trait: bool
    command_buffer_eligible: bool
    forbidden_command_buffer_operations: tuple[str, ...]
    has_atomics: bool
    opaque_semantic_dependencies: tuple[str, ...]


def partitioned_gemm_ffi_abi(program: PartitionedGemmProgram) -> PartitionedGemmFfiAbi:
    """Validate and return the exact physical typed-FFI ABI."""
    if program.partitioned_operand != 1:
        raise ValueError("the bounded CUDA skeleton currently supports static RHS partitions")
    if len(program.operand_shapes) != len(program.partitions) + 1:
        raise ValueError("partitioned Contract requires one lhs and one rhs buffer per accumulator partition")
    m, _, k = program.shape
    inputs = tuple(_parse_buffer(f"operand{index}", shape) for index, shape in enumerate(program.operand_shapes))
    lhs = inputs[0]
    if lhs.dtype != "bf16" or not lhs.shape or lhs.shape[-1] != k or math.prod(lhs.shape[:-1]) != m:
        raise ValueError("partitioned Contract lhs must flatten to [M,K] BF16")
    for partition, rhs in zip(program.partitions, inputs[1:], strict=True):
        if rhs.dtype != "bf16" or rhs.shape != (partition.extent, k):
            raise ValueError("each static RHS segment must have physical shape [partition_N,K]")

    outputs = tuple(_parse_buffer(f"output{index}", shape) for index, shape in enumerate(program.output_shapes))
    scalar_outputs = outputs[: len(program.scalar_finalizations)]
    passthrough_outputs = outputs[len(program.scalar_finalizations) :]
    for output, finalization in zip(scalar_outputs, program.scalar_finalizations, strict=True):
        extent = program.partitions[finalization.source_partitions[0]].extent
        if output.dtype != "bf16" or not output.shape or output.shape[-1] != extent:
            raise ValueError("partitioned Contract output feature extent disagrees with its source partition")
        if math.prod(output.shape[:-1]) != m:
            raise ValueError("partitioned Contract output row domain disagrees with M")
    for output, finalization in zip(passthrough_outputs, program.passthrough_finalizations, strict=True):
        extent = program.partitions[finalization.source_partition].extent
        if output.dtype != "bf16" or not output.shape or output.shape[-1] != extent:
            raise ValueError("partitioned Contract output feature extent disagrees with its source partition")
        if math.prod(output.shape[:-1]) != m:
            raise ValueError("partitioned Contract output row domain disagrees with M")
    for finalization in program.scalar_finalizations:
        if any(
            value.input_index is None or value.input_index.row_offset != 0 or value.input_index.feature_offset != 0
            for value in finalization.program.inputs
        ):
            raise ValueError("the bounded scalar skeleton requires pointwise partition Maps")
    shared_bytes = m * program.shape[1] * 2
    if shared_bytes > _MAX_SHARED_BYTES:
        raise ValueError(
            f"partition boundary needs {shared_bytes} shared bytes; bounded skeleton limit is {_MAX_SHARED_BYTES}"
        )
    return PartitionedGemmFfiAbi(inputs=inputs, outputs=outputs)


def generate_cuda_partitioned_gemm_ffi(
    program: PartitionedGemmProgram,
    *,
    target: str,
    threads: int = 256,
) -> GeneratedCudaPartitionedGemmFfi:
    """Generate one ordered scalar Contract mainloop and generic partition stores."""
    if threads <= 0 or threads > 1024:
        raise ValueError("partitioned Contract threads must be in [1,1024]")
    abi = partitioned_gemm_ffi_abi(program)
    m, n, k = program.shape
    generated_finalization = generate_partitioned_gemm_finalization(program)
    scalar_sources = "\n".join(body.source for body in generated_finalization.scalar_bodies)
    input_arguments = tuple(f"ffi::Buffer<ffi::BF16, {len(buffer.shape)}> {buffer.name}_buffer" for buffer in abi.inputs)
    result_arguments = tuple(
        f"ffi::Result<ffi::Buffer<ffi::BF16, {len(buffer.shape)}>> {buffer.name}_buffer" for buffer in abi.outputs
    )
    input_bindings = tuple(f"      .Arg<ffi::Buffer<ffi::BF16, {len(buffer.shape)}>>()" for buffer in abi.inputs)
    result_bindings = tuple(f"      .Ret<ffi::Buffer<ffi::BF16, {len(buffer.shape)}>>()" for buffer in abi.outputs)
    kernel_arguments = tuple(f"const __nv_bfloat16* __restrict__ {buffer.name}" for buffer in abi.inputs) + tuple(
        f"__nv_bfloat16* __restrict__ {buffer.name}" for buffer in abi.outputs
    )
    launch_arguments = tuple(
        f"reinterpret_cast<const __nv_bfloat16*>({buffer.name}_buffer.typed_data())" for buffer in abi.inputs
    ) + tuple(f"reinterpret_cast<__nv_bfloat16*>({buffer.name}_buffer->typed_data())" for buffer in abi.outputs)

    lhs_offset = _physical_offset(abi.inputs[0], row="row", feature="reduction")
    rhs_selection = _rhs_selection(program, abi)
    output_loops = _output_loops(program, abi, generated_finalization)
    handler_symbol = _target_symbol(target)
    semantic_record = {
        "program": program.semantic_digest,
        "physical_family": "bounded_one_cta_partitioned_contract",
        "threads": threads,
        "abi": {
            "inputs": [_buffer_record(buffer) for buffer in abi.inputs],
            "outputs": [_buffer_record(buffer) for buffer in abi.outputs],
        },
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_template = f"""// Generated from a generic partitioned Contract and scalar ASTs; do not edit.
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{scalar_sources}

namespace ffi = xla::ffi;

namespace {{
constexpr int kRows = {m};
constexpr int kFeatures = {n};
constexpr int kReduction = {k};
constexpr int kThreads = {threads};

__global__ void ShuttlePartitionedGemmKernel(
    {",\n    ".join(kernel_arguments)}) {{
  __shared__ __nv_bfloat16 accumulator_boundary[kRows * kFeatures];

  for (int linear = threadIdx.x; linear < kRows * kFeatures; linear += blockDim.x) {{
    const int row = linear / kFeatures;
    const int feature = linear - row * kFeatures;
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kReduction; ++reduction) {{
      const float lhs_value = __bfloat162float(operand0[{lhs_offset}]);
      float rhs_value = 0.0f;
{rhs_selection}
      accumulator = __fadd_rn(accumulator, __fmul_rn(lhs_value, rhs_value));
    }}
    accumulator_boundary[linear] = __float2bfloat16_rn(accumulator);
  }}
  __syncthreads();

{output_loops}
}}

ffi::Error ShuttlePartitionedGemm(
    cudaStream_t stream,
    {",\n    ".join((*input_arguments, *result_arguments))}) {{
  ShuttlePartitionedGemmKernel<<<1, kThreads, 0, stream>>>(
      {",\n      ".join(launch_arguments)});
  return ffi::Error::Success();
}}

auto ShuttlePartitionedGemmBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
{chr(10).join(input_bindings)}
{chr(10).join(result_bindings)};
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {handler_symbol},
    ShuttlePartitionedGemm,
    ShuttlePartitionedGemmBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);
"""
    source = finalize_ffi_handler_source(source_template, command_buffer_compatible=True)
    return GeneratedCudaPartitionedGemmFfi(
        target=target,
        handler_symbol=handler_symbol,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
        threads=threads,
        shared_bytes=m * n * 2,
        abi=abi,
    )


def audit_cuda_partitioned_gemm_source(
    generated: GeneratedCudaPartitionedGemmFfi,
) -> PartitionedGemmSourceAudit:
    """Audit that the generated executor owns generic arithmetic only."""
    lowered = generated.source.lower()
    command_buffer = audit_ffi_command_buffer_eligibility(generated.source)
    opaque_tokens = (
        "flash_attention",
        "mok_forward",
        "gdn_chunk",
        "sparse_attention_forward",
        "deep_ep",
        "swiglu",
        "router",
        "cublas",
        "quack",
    )
    return PartitionedGemmSourceAudit(
        kernel_count=generated.source.count("__global__ void "),
        segmented_rhs_count=len(generated.abi.inputs) - 1,
        direct_output_count=len(generated.abi.outputs),
        has_ordered_fp32_mainloop="__fadd_rn(accumulator, __fmul_rn(lhs_value, rhs_value))" in generated.source,
        has_bf16_rne_partition_boundary=(
            "accumulator_boundary[linear] = __float2bfloat16_rn(accumulator);" in generated.source
        ),
        has_command_buffer_trait="{ffi::Traits::kCmdBufferCompatible}" in generated.source,
        command_buffer_eligible=command_buffer.eligible,
        forbidden_command_buffer_operations=command_buffer.forbidden_operations,
        has_atomics="atomic" in lowered,
        opaque_semantic_dependencies=tuple(token for token in opaque_tokens if token in lowered),
    )


def _rhs_selection(program: PartitionedGemmProgram, abi: PartitionedGemmFfiAbi) -> str:
    branches: list[str] = []
    for index, (partition, rhs) in enumerate(zip(program.partitions, abi.inputs[1:], strict=True)):
        condition = "if" if index == 0 else "else if"
        local_feature = f"feature - {partition.start}"
        offset = _physical_offset(rhs, row=local_feature, feature="reduction")
        branches.append(
            f"      {condition} (feature < {partition.limit}) {{\n"
            f"        rhs_value = __bfloat162float(operand{index + 1}[{offset}]);\n"
            "      }"
        )
    return "\n".join(branches)


def _output_loops(
    program: PartitionedGemmProgram,
    abi: PartitionedGemmFfiAbi,
    generated_finalization: GeneratedPartitionedGemmFinalization,
) -> str:
    loops: list[str] = []
    output_index = 0
    for scalar_index, finalization in enumerate(program.scalar_finalizations):
        extent = program.partitions[finalization.source_partitions[0]].extent
        arguments: list[str] = []
        for partition_index, scalar_input in zip(
            finalization.source_partitions, finalization.program.inputs, strict=True
        ):
            assert scalar_input.input_index is not None
            partition = program.partitions[partition_index]
            arguments.append(
                "__bfloat162float(accumulator_boundary["
                f"(row + {scalar_input.input_index.row_offset}) * kFeatures + "
                f"{partition.start} + feature + {scalar_input.input_index.feature_offset}])"
            )
        output = abi.outputs[output_index]
        output_offset = _physical_offset(output, row="row", feature="feature")
        body = generated_finalization.scalar_bodies[scalar_index]
        loops.append(
            f"  for (int linear = threadIdx.x; linear < kRows * {extent}; linear += blockDim.x) {{\n"
            f"    const int row = linear / {extent};\n"
            f"    const int feature = linear - row * {extent};\n"
            f"    output{output_index}[{output_offset}] = "
            f"__float2bfloat16_rn({body.symbol}({', '.join(arguments)}));\n"
            "  }"
        )
        output_index += 1
    for finalization in program.passthrough_finalizations:
        partition = program.partitions[finalization.source_partition]
        output = abi.outputs[output_index]
        output_offset = _physical_offset(output, row="row", feature="feature")
        loops.append(
            f"  for (int linear = threadIdx.x; linear < kRows * {partition.extent}; linear += blockDim.x) {{\n"
            f"    const int row = linear / {partition.extent};\n"
            f"    const int feature = linear - row * {partition.extent};\n"
            f"    output{output_index}[{output_offset}] = "
            f"accumulator_boundary[row * kFeatures + {partition.start} + feature];\n"
            "  }"
        )
        output_index += 1
    return "\n\n".join(loops)


def _parse_buffer(name: str, shape: str) -> PartitionedGemmFfiBuffer:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"partitioned Contract requires an explicit physical array shape, found {shape!r}")
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    layout = tuple(int(value) for value in match.group("layout").split(","))
    if sorted(layout) != list(range(len(dimensions))):
        raise ValueError(f"physical layout must permute all axes, found {layout}")
    return PartitionedGemmFfiBuffer(name, match.group("dtype"), dimensions, layout)


def _physical_offset(buffer: PartitionedGemmFfiBuffer, *, row: str, feature: str) -> str:
    if not buffer.shape:
        raise ValueError("partitioned Contract does not support scalar buffers")
    strides = _physical_strides(buffer.shape, buffer.minor_to_major)
    leading = buffer.shape[:-1]
    terms: list[str] = []
    for axis, extent in enumerate(leading):
        suffix = math.prod(leading[axis + 1 :])
        coordinate = f"(({row}) / {suffix}) % {extent}" if suffix != 1 else f"({row}) % {extent}"
        terms.append(f"({coordinate}) * {strides[axis]}")
    terms.append(f"({feature}) * {strides[-1]}")
    return " + ".join(terms)


def _physical_strides(shape: tuple[int, ...], minor_to_major: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = 1
    for axis in minor_to_major:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def _buffer_record(buffer: PartitionedGemmFfiBuffer) -> dict[str, object]:
    return {
        "name": buffer.name,
        "dtype": buffer.dtype,
        "shape": buffer.shape,
        "minor_to_major": buffer.minor_to_major,
    }


def _target_symbol(target: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", target)
    if not symbol or symbol[0].isdigit():
        raise ValueError(f"typed-FFI target cannot form a C++ symbol: {target!r}")
    return symbol
