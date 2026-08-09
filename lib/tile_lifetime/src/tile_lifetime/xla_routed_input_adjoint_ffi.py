# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a CUDA typed-FFI executor for a routed input adjoint."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from tile_lifetime.cast_scalar_program import evaluate_cast_scalar_program
from tile_lifetime.xla_relation_program_recovery import (
    RoutedInputAdjointFfiOperandRole,
    RoutedInputAdjointTypedFfiCodegenPlan,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")


@dataclass(frozen=True)
class GeneratedRoutedInputAdjointFfi:
    """CUDA source and semantic identity for one generated input adjoint."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str


def generate_cuda_routed_input_adjoint_ffi(
    plan: RoutedInputAdjointTypedFfiCodegenPlan,
    *,
    target: str,
) -> GeneratedRoutedInputAdjointFfi:
    """Generate a deterministic BF16 Contract/Map/Contract/Fold executor."""
    dimensions = _validated_dimensions(plan)
    roles = {operand.role: index for index, operand in enumerate(plan.operands)}
    if len(roles) != len(RoutedInputAdjointFfiOperandRole):
        raise ValueError("input-adjoint plan does not bind every operand role exactly once")
    map_calls: list[str] = []
    scalar_sources: list[str] = []
    for output in plan.map_stage.scalar_outputs:
        arguments: list[str] = []
        for value in output.scalar_program.inputs:
            if value.input_name is None or value.input_index is None or value.input_index.row_offset != 0:
                raise ValueError("reverse Map supports only named same-row scalar inputs")
            feature = f"feature - {output.feature_offset} + {value.input_index.feature_offset}"
            if value.input_name.startswith("input0"):
                arguments.append("shuttle_bf16_to_f32(projection[row * kFirstOutputFeatures + " + feature + "])")
            elif value.input_name.startswith("input1"):
                arguments.append("shuttle_bf16_to_f32(auxiliary[row * kLogicalFeatures + " + feature + "])")
            else:
                raise ValueError(f"unsupported reverse Map input {value.input_name!r}")
        upper = output.feature_offset + output.feature_extent
        map_calls.append(f"feature < {upper} ? {output.generated_cuda.symbol}({', '.join(arguments)})")
        scalar_sources.append(output.generated_cuda.source)
    if not map_calls or plan.map_stage.scalar_outputs[0].feature_offset != 0:
        raise ValueError("reverse Map outputs must start at feature zero")
    expected_offset = 0
    for output in plan.map_stage.scalar_outputs:
        if output.feature_offset != expected_offset:
            raise ValueError("reverse Map outputs must cover one contiguous feature range")
        expected_offset += output.feature_extent
    map_expression = " : ".join((*map_calls, "0.0f"))

    contribution = plan.fold_stage.contribution_program
    if len(contribution.inputs) != 1:
        raise ValueError("input-adjoint source Fold requires one generated contribution input")
    contribution_input = contribution.inputs[0]
    if (
        contribution_input.input_index is None
        or contribution_input.input_index.row_offset != 0
        or contribution_input.input_index.feature_offset != 0
    ):
        raise ValueError("input-adjoint Fold contribution must be a same-element scalar")
    scalar_sources.extend(
        (
            plan.fold_stage.generated_contribution_cuda.source,
            plan.fold_stage.generated_reducer_cuda.source,
        )
    )

    ffi_types = {"bf16": "ffi::BF16", "pred": "ffi::PRED", "s32": "ffi::S32"}
    ffi_arguments: list[str] = []
    ffi_bindings: list[str] = []
    data_bindings: list[str] = []
    for index, operand in enumerate(plan.operands):
        dtype, shape, _ = _parse_shape(operand.value.shape)
        if dtype not in ffi_types:
            raise ValueError(f"unsupported input-adjoint FFI dtype {dtype!r}")
        ffi_arguments.append(f"ffi::Buffer<{ffi_types[dtype]}, {len(shape)}> input{index}_buffer")
        ffi_bindings.append(f"      .Arg<ffi::Buffer<{ffi_types[dtype]}, {len(shape)}>>()")
        if dtype == "bf16":
            data_bindings.append(
                f"  const auto* input{index} = reinterpret_cast<const std::uint16_t*>(input{index}_buffer.typed_data());"
            )
        else:
            data_bindings.append(f"  const auto* input{index} = input{index}_buffer.typed_data();")

    second_rhs = roles[RoutedInputAdjointFfiOperandRole.SECOND_CONTRACT_RHS]
    initial = roles[RoutedInputAdjointFfiOperandRole.FOLD_INITIAL]
    indices = roles[RoutedInputAdjointFfiOperandRole.FOLD_INDICES]
    validity = roles[RoutedInputAdjointFfiOperandRole.SEGMENT_VALIDITY]
    first_lhs = roles[RoutedInputAdjointFfiOperandRole.FIRST_CONTRACT_LHS]
    first_rhs = roles[RoutedInputAdjointFfiOperandRole.FIRST_CONTRACT_RHS]
    auxiliary = roles[RoutedInputAdjointFfiOperandRole.MAP_AUXILIARY]
    target_symbol = target.replace(".", "_")
    semantic_record = {
        "contracts": [contract.output_shape for contract in plan.contracts],
        "map": [output.scalar_program.digest for output in plan.map_stage.scalar_outputs],
        "contribution": plan.fold_stage.contribution_program.digest,
        "reducer": plan.fold_stage.reducer_program.digest,
        "layout": {
            "logical_edges": dimensions["logical_edges"],
            "logical_features": dimensions["logical_features"],
            "segments": dimensions["segments"],
            "feature_stride": dimensions["feature_stride"],
            "segment_stride": dimensions["segment_stride"],
        },
        "operand_roles": [(operand.role.value, operand.value.shape) for operand in plan.operands],
        "numerical_policy": plan.region.numerical_policy.value,
    }
    semantic_digest = hashlib.sha256(json.dumps(semantic_record, sort_keys=True).encode()).hexdigest()
    source = f"""// Generated by tile_lifetime.xla_routed_input_adjoint_ffi; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{chr(10).join(scalar_sources)}

namespace ffi = xla::ffi;

namespace {{
constexpr int kPaddedRows = {dimensions["padded_rows"]};
constexpr int kFirstReduction = {dimensions["first_reduction"]};
constexpr int kFirstOutputFeatures = {dimensions["first_output_features"]};
constexpr int kLogicalEdges = {dimensions["logical_edges"]};
constexpr int kLogicalFeatures = {dimensions["logical_features"]};
constexpr int kSegments = {dimensions["segments"]};
constexpr int kPhysicalMapFeatures = {dimensions["physical_map_features"]};
constexpr int kFeatureStride = {dimensions["feature_stride"]};
constexpr int kSegmentStride = {dimensions["segment_stride"]};
constexpr int kOutputRows = {dimensions["output_rows"]};
constexpr int kOutputFeatures = {dimensions["output_features"]};
std::atomic<int> call_count{{0}};
thread_local cublasHandle_t contract_handle = nullptr;

__device__ __forceinline__ float shuttle_bf16_to_f32(std::uint16_t value) {{
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16);
}}

__device__ __forceinline__ std::uint16_t shuttle_f32_to_bf16(float value) {{
  union ShuttleBf16Bits {{
    __nv_bfloat16 value;
    std::uint16_t bits;
  }} converted;
  converted.value = __float2bfloat16_rn(value);
  return converted.bits;
}}

ffi::Error Contract(
    cudaStream_t stream,
    const std::uint16_t* lhs,
    const std::uint16_t* rhs,
    std::uint16_t* output,
    int rows,
    int reduction,
    int features) {{
  if (contract_handle == nullptr) {{
    const cublasStatus_t create_status = cublasCreate(&contract_handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {{
      return ffi::Error::Internal(
          "cublasCreate failed with status " + std::to_string(static_cast<int>(create_status)));
    }}
  }}
  cublasStatus_t status = cublasSetStream(contract_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasSetStream failed with status " + std::to_string(static_cast<int>(status)));
  }}
  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasGemmEx(
      contract_handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      features,
      rows,
      reduction,
      &alpha,
      rhs,
      CUDA_R_16BF,
      features,
      lhs,
      CUDA_R_16BF,
      reduction,
      &beta,
      output,
      CUDA_R_16BF,
      features,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmEx failed with status " + std::to_string(static_cast<int>(status)));
  }}
  return ffi::Error::Success();
}}

__global__ void ShuttleReverseMapKernel(
    const std::uint16_t* projection,
    const std::uint16_t* auxiliary,
    const bool* validity,
    std::uint16_t* auxiliary_output,
    std::uint16_t* mapped) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kPaddedRows * kLogicalFeatures * kSegments) {{
    return;
  }}
  const int feature = index % kLogicalFeatures;
  const int row = (index / kLogicalFeatures) % kPaddedRows;
  const int segment = index / (kLogicalFeatures * kPaddedRows);
  const int validity_index = (segment * kPaddedRows + row) * kLogicalFeatures + feature;
  const bool valid = row < kLogicalEdges && validity[validity_index];
  const float value = valid ? {map_expression} : {plan.segmented_layout.fill_value}f;
  const std::uint16_t stored = shuttle_f32_to_bf16(value);
  auxiliary_output[index] = stored;
  const int physical_feature = feature * kFeatureStride + segment * kSegmentStride;
  mapped[row * kPhysicalMapFeatures + physical_feature] = stored;
}}

__global__ void ShuttleSourceFoldKernel(
    const std::uint16_t* initial,
    const std::int32_t* source_indices,
    const std::uint16_t* routed_output,
    std::uint16_t* output) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kOutputRows * kOutputFeatures) {{
    return;
  }}
  const int source = index / kOutputFeatures;
  const int feature = index - source * kOutputFeatures;
  float accumulator = shuttle_bf16_to_f32(initial[index]);
  for (int edge = 0; edge < kLogicalEdges; ++edge) {{
    if (source_indices[edge] != source) {{
      continue;
    }}
    const float contribution = {plan.fold_stage.generated_contribution_cuda.symbol}(
        shuttle_bf16_to_f32(routed_output[edge * kOutputFeatures + feature]));
    accumulator = {plan.fold_stage.generated_reducer_cuda.symbol}(accumulator, contribution);
  }}
  output[index] = shuttle_f32_to_bf16(accumulator);
}}

ffi::Error ShuttleRoutedInputAdjoint(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {",\n    ".join(ffi_arguments)},
    ffi::Result<ffi::Buffer<ffi::BF16, 3>> auxiliary_output_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output_buffer) {{
{chr(10).join(data_bindings)}
  auto* auxiliary_output = reinterpret_cast<std::uint16_t*>(auxiliary_output_buffer->typed_data());
  auto* output = reinterpret_cast<std::uint16_t*>(output_buffer->typed_data());
  auto projection_storage = scratch.Allocate(
      sizeof(std::uint16_t) * kPaddedRows * kFirstOutputFeatures, alignof(std::uint16_t));
  auto mapped_storage = scratch.Allocate(
      sizeof(std::uint16_t) * kPaddedRows * kPhysicalMapFeatures, alignof(std::uint16_t));
  auto routed_output_storage = scratch.Allocate(
      sizeof(std::uint16_t) * kPaddedRows * kOutputFeatures, alignof(std::uint16_t));
  if (!projection_storage || !mapped_storage || !routed_output_storage) {{
    return ffi::Error::Internal("failed to allocate routed input-adjoint scratch storage");
  }}
  auto* projection = static_cast<std::uint16_t*>(*projection_storage);
  auto* mapped = static_cast<std::uint16_t*>(*mapped_storage);
  auto* routed_output = static_cast<std::uint16_t*>(*routed_output_storage);

  ffi::Error first_contract = Contract(
      stream, input{first_lhs}, input{first_rhs}, projection,
      kPaddedRows, kFirstReduction, kFirstOutputFeatures);
  if (first_contract.failure()) {{
    return first_contract;
  }}
  constexpr int kThreads = 256;
  constexpr int kMapItems = kPaddedRows * kLogicalFeatures * kSegments;
  constexpr int kMapBlocks = (kMapItems + kThreads - 1) / kThreads;
  ShuttleReverseMapKernel<<<kMapBlocks, kThreads, 0, stream>>>(
      projection, input{auxiliary}, input{validity}, auxiliary_output, mapped);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttleReverseMapKernel: ") + cudaGetErrorString(status));
  }}

  ffi::Error second_contract = Contract(
      stream, mapped, input{second_rhs}, routed_output,
      kPaddedRows, kPhysicalMapFeatures, kOutputFeatures);
  if (second_contract.failure()) {{
    return second_contract;
  }}
  constexpr int kFoldItems = kOutputRows * kOutputFeatures;
  constexpr int kFoldBlocks = (kFoldItems + kThreads - 1) / kThreads;
  ShuttleSourceFoldKernel<<<kFoldBlocks, kThreads, 0, stream>>>(
      input{initial}, input{indices}, routed_output, output);
  status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttleSourceFoldKernel: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleRoutedInputAdjointBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{chr(10).join(ffi_bindings)}
      .Ret<ffi::Buffer<ffi::BF16, 3>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleRoutedInputAdjoint,
    ShuttleRoutedInputAdjointBinding());

extern "C" int shuttle_routed_input_adjoint_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    return GeneratedRoutedInputAdjointFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def replace_routed_input_adjoint_region_with_custom_call(
    hlo_text: str,
    plan: RoutedInputAdjointTypedFfiCodegenPlan,
    *,
    target: str,
) -> str:
    """Replace the physical input-adjoint interval with one two-result custom call."""
    outputs = plan.region.boundary.outputs
    if len(outputs) != 2:
        raise ValueError("routed input-adjoint FFI requires Map auxiliary and Fold outputs")
    map_output = next(output for output in outputs if output.instruction == _map_auxiliary_output(plan))
    fold_output = next(output for output in outputs if output.instruction == plan.region.insertion_instruction)
    operands = ", ".join(f"%{operand.value.instruction}" for operand in plan.operands)
    constraints = ", ".join(operand.value.shape for operand in plan.operands)
    call_name = "shuttle_generated_routed_input_adjoint_region"
    rewritten = hlo_text
    for instruction in plan.region.boundary.internal_instructions:
        if instruction == plan.region.insertion_instruction:
            continue
        pattern = re.compile(rf"^\s*%{re.escape(instruction)} = .*?\n", re.MULTILINE)
        matches = tuple(pattern.finditer(rewritten))
        if len(matches) != 1:
            raise ValueError(f"expected one physical definition for internal instruction %{instruction}")
        rewritten = rewritten[: matches[0].start()] + rewritten[matches[0].end() :]
    insertion_pattern = re.compile(
        rf"^(?P<indent>\s*)%{re.escape(plan.region.insertion_instruction)} = .*?$",
        re.MULTILINE,
    )
    insertion_matches = tuple(insertion_pattern.finditer(rewritten))
    if len(insertion_matches) != 1:
        raise ValueError("expected one input-adjoint insertion instruction")
    insertion = insertion_matches[0]
    call = (
        f"{insertion.group('indent')}%{call_name} = ({map_output.shape}, {fold_output.shape}) "
        f'custom-call({operands}), custom_call_target="{target}", '
        f"operand_layout_constraints={{{constraints}}}, api_version=API_VERSION_TYPED_FFI, backend_config={{}}\n"
        f"{insertion.group('indent')}%{map_output.instruction} = {map_output.shape} "
        f"get-tuple-element(%{call_name}), index=0\n"
        f"{insertion.group('indent')}%{fold_output.instruction} = {fold_output.shape} "
        f"get-tuple-element(%{call_name}), index=1"
    )
    return rewritten[: insertion.start()] + call + rewritten[insertion.end() :]


def evaluate_routed_input_adjoint_plan(
    plan: RoutedInputAdjointTypedFfiCodegenPlan,
    operands: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the generated region in deterministic reference order on CPU."""
    dimensions = _validated_dimensions(plan)
    if len(operands) != len(plan.operands):
        raise ValueError("runtime operand count does not match the input-adjoint plan")
    by_role = {binding.role: np.asarray(operands[index]) for index, binding in enumerate(plan.operands)}
    projection = _round_bf16_array(
        np.matmul(
            by_role[RoutedInputAdjointFfiOperandRole.FIRST_CONTRACT_LHS].astype(np.float32),
            by_role[RoutedInputAdjointFfiOperandRole.FIRST_CONTRACT_RHS].astype(np.float32),
        )
    )
    auxiliary = by_role[RoutedInputAdjointFfiOperandRole.MAP_AUXILIARY].astype(np.float32)
    validity = by_role[RoutedInputAdjointFfiOperandRole.SEGMENT_VALIDITY]
    auxiliary_output = np.full(
        (dimensions["segments"], dimensions["padded_rows"], dimensions["logical_features"]),
        plan.segmented_layout.fill_value,
        dtype=np.float32,
    )
    mapped = np.full(
        (dimensions["padded_rows"], dimensions["physical_map_features"]),
        plan.segmented_layout.fill_value,
        dtype=np.float32,
    )
    for output in plan.map_stage.scalar_outputs:
        program = output.scalar_program
        for row in range(dimensions["logical_edges"]):
            for local_feature in range(output.feature_extent):
                feature = output.feature_offset + local_feature
                scalar_inputs: dict[str, float] = {}
                for value in program.inputs:
                    if value.input_name is None or value.input_index is None:
                        raise ValueError("reverse Map input lacks a concrete scalar index")
                    if value.input_name.startswith("input0"):
                        scalar_inputs[value.input_name] = projection[
                            row, local_feature + value.input_index.feature_offset
                        ]
                    elif value.input_name.startswith("input1"):
                        scalar_inputs[value.input_name] = auxiliary[
                            row, local_feature + value.input_index.feature_offset
                        ]
                    else:
                        raise ValueError(f"unsupported reverse Map input {value.input_name!r}")
                result = evaluate_cast_scalar_program(program, scalar_inputs)
                for segment in range(dimensions["segments"]):
                    if validity[segment, row, feature]:
                        auxiliary_output[segment, row, feature] = result
                        physical_feature = (
                            feature * dimensions["feature_stride"] + segment * dimensions["segment_stride"]
                        )
                        mapped[row, physical_feature] = result
    routed_output = _round_bf16_array(
        np.matmul(
            mapped,
            by_role[RoutedInputAdjointFfiOperandRole.SECOND_CONTRACT_RHS].astype(np.float32),
        )
    )
    output = by_role[RoutedInputAdjointFfiOperandRole.FOLD_INITIAL].astype(np.float32).copy()
    indices = by_role[RoutedInputAdjointFfiOperandRole.FOLD_INDICES].reshape(-1)
    for edge in range(dimensions["logical_edges"]):
        source = int(indices[edge])
        for feature in range(dimensions["output_features"]):
            contribution = evaluate_cast_scalar_program(
                plan.fold_stage.contribution_program,
                {"input_r0_f0": routed_output[edge, feature]},
            )
            output[source, feature] = evaluate_cast_scalar_program(
                plan.fold_stage.reducer_program,
                {"input0": output[source, feature], "input1": contribution},
            )
    return auxiliary_output, output


def _map_auxiliary_output(plan: RoutedInputAdjointTypedFfiCodegenPlan) -> str:
    path_instructions = {node.split("/", 1)[-1] for node in plan.map_stage.layout_path}
    candidates = tuple(
        output.instruction
        for output in plan.region.boundary.outputs
        if output.instruction != plan.region.insertion_instruction and output.instruction in path_instructions
    )
    if len(candidates) != 1:
        raise ValueError("input-adjoint physical Map path has no unique live auxiliary output")
    return candidates[0]


def _validated_dimensions(plan: RoutedInputAdjointTypedFfiCodegenPlan) -> dict[str, int]:
    if len(plan.contracts) != 2:
        raise ValueError("input-adjoint executor requires exactly two Contracts")
    first_dtype, first_output, first_layout = _parse_shape(plan.contracts[0].output_shape)
    second_dtype, second_output, second_layout = _parse_shape(plan.contracts[1].output_shape)
    physical_dtype, physical_map, physical_layout = _parse_shape(plan.map_stage.physical_output_shape)
    output_dtype, output, output_layout = _parse_shape(plan.fold_stage.output_shape)
    if {first_dtype, second_dtype, physical_dtype, output_dtype} != {"bf16"}:
        raise ValueError("input-adjoint executor currently requires BF16 storage")
    if any(
        layout != tuple(reversed(range(len(shape))))
        for shape, layout in (
            (first_output, first_layout),
            (second_output, second_layout),
            (physical_map, physical_layout),
            (output, output_layout),
        )
    ):
        raise ValueError("input-adjoint executor requires row-major physical arrays")
    first_lhs = _shape_for_role(plan, RoutedInputAdjointFfiOperandRole.FIRST_CONTRACT_LHS)
    first_rhs = _shape_for_role(plan, RoutedInputAdjointFfiOperandRole.FIRST_CONTRACT_RHS)
    second_rhs = _shape_for_role(plan, RoutedInputAdjointFfiOperandRole.SECOND_CONTRACT_RHS)
    auxiliary = _shape_for_role(plan, RoutedInputAdjointFfiOperandRole.MAP_AUXILIARY)
    if any(value[0] != "bf16" for value in (first_lhs, first_rhs, second_rhs, auxiliary)):
        raise ValueError("input-adjoint Contract and Map operands must be BF16")
    if first_lhs[1][0] != first_output[0] or first_rhs[1] != (first_lhs[1][1], first_output[1]):
        raise ValueError("first input-adjoint Contract dimensions disagree")
    index_map = plan.segmented_layout.index_map
    if auxiliary[1] != (index_map.padded_row_extent, index_map.logical_feature_extent):
        raise ValueError("reverse Map auxiliary shape disagrees with the logical Map domain")
    if physical_map != (index_map.padded_row_extent, index_map.logical_feature_extent * index_map.segment_count):
        raise ValueError("segmented index relation does not cover the physical Map output")
    if second_rhs[1] != (physical_map[1], second_output[1]) or second_output[1] != output[1]:
        raise ValueError("second input-adjoint Contract dimensions disagree")
    return {
        "padded_rows": index_map.padded_row_extent,
        "first_reduction": first_lhs[1][1],
        "first_output_features": first_output[1],
        "logical_edges": index_map.logical_edge_count,
        "logical_features": index_map.logical_feature_extent,
        "segments": index_map.segment_count,
        "physical_map_features": physical_map[1],
        "feature_stride": index_map.feature_stride,
        "segment_stride": index_map.segment_stride,
        "output_rows": output[0],
        "output_features": output[1],
    }


def _shape_for_role(
    plan: RoutedInputAdjointTypedFfiCodegenPlan,
    role: RoutedInputAdjointFfiOperandRole,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    operands = tuple(operand for operand in plan.operands if operand.role is role)
    if len(operands) != 1:
        raise ValueError(f"input-adjoint plan has {len(operands)} operands for role {role.value}")
    return _parse_shape(operands[0].value.shape)


def _parse_shape(shape: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"unsupported physical array shape {shape!r}")
    return (
        match.group("dtype"),
        tuple(int(value) for value in match.group("dims").split(",") if value),
        tuple(int(value) for value in match.group("layout").split(",")),
    )


def _round_bf16_array(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)
