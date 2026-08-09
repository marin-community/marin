# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a CUDA typed-FFI routed Contract/Map/Contract/Fold executor."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from tile_lifetime.cast_scalar_program import evaluate_cast_scalar_program
from tile_lifetime.xla_relation_program_recovery import (
    RoutedForwardCodegenDisposition,
    RoutedForwardFfiOperandRole,
    RoutedForwardTypedFfiCodegenPlan,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")


@dataclass(frozen=True)
class GeneratedRoutedForwardFfi:
    """CUDA source and semantic identity for one recovered routed region."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str


def generate_cuda_routed_forward_ffi(
    plan: RoutedForwardTypedFfiCodegenPlan,
    *,
    target: str,
) -> GeneratedRoutedForwardFfi:
    """Generate one deterministic CUDA executor from a READY generic plan."""
    dimensions = _validated_dimensions(plan)
    roles = {operand.role: index for index, operand in enumerate(plan.operands)}
    if len(roles) != len(RoutedForwardFfiOperandRole):
        raise ValueError("routed FFI plan does not bind every generic operand role exactly once")

    map_output = plan.map_stage.scalar_outputs
    if len(map_output) != 1 or map_output[0].feature_offset != 0:
        raise ValueError("routed FFI prototype requires one zero-based Map output")
    map_program = map_output[0].scalar_program
    map_arguments = []
    for value in map_program.inputs:
        if value.input_index is None or value.input_index.row_offset != 0:
            raise ValueError("routed Map supports only same-row scalar inputs")
        map_arguments.append(f"projection[row * kFirstOutputFeatures + feature + {value.input_index.feature_offset}]")
    contribution_arguments = []
    for value in plan.fold_stage.contribution_program.inputs:
        if value.input_index is None or value.input_index.row_offset != 0 or value.input_index.feature_offset != 0:
            raise ValueError("routed Fold contribution supports one same-element value per input")
        if value.input_name is not None and value.input_name.startswith("input0"):
            contribution_arguments.append("routed_output[edge * kOutputFeatures + feature]")
        elif value.input_name is not None and value.input_name.startswith("input1"):
            contribution_arguments.append("shuttle_bf16_to_f32(route_weight[edge])")
        else:
            raise ValueError(f"unsupported Fold contribution input {value.input_name!r}")

    ffi_types = {
        "bf16": "ffi::BF16",
        "f32": "ffi::F32",
        "pred": "ffi::PRED",
        "s32": "ffi::S32",
    }
    cpp_types = {
        "bf16": "std::uint16_t",
        "f32": "float",
        "pred": "bool",
        "s32": "std::int32_t",
    }
    ffi_arguments = []
    ffi_bindings = []
    data_bindings = []
    for index, operand in enumerate(plan.operands):
        dtype, shape, _ = _parse_shape(operand.value.shape)
        ffi_arguments.append(f"ffi::Buffer<{ffi_types[dtype]}, {len(shape)}> input{index}_buffer")
        ffi_bindings.append(f"      .Arg<ffi::Buffer<{ffi_types[dtype]}, {len(shape)}>>()")
        pointer_type = cpp_types[dtype]
        if dtype == "bf16":
            declaration = f"reinterpret_cast<const {pointer_type}*>(input{index}_buffer.typed_data())"
            data_bindings.append(f"  const auto* input{index} = {declaration};")
        else:
            data_bindings.append(f"  const auto* input{index} = input{index}_buffer.typed_data();")

    initial = roles[RoutedForwardFfiOperandRole.FOLD_INITIAL]
    indices = roles[RoutedForwardFfiOperandRole.FOLD_INDICES]
    route_weight = roles[RoutedForwardFfiOperandRole.FOLD_CONTRIBUTION_INPUT]
    validity = roles[RoutedForwardFfiOperandRole.SEGMENT_VALIDITY]
    first_lhs = roles[RoutedForwardFfiOperandRole.FIRST_CONTRACT_LHS]
    first_rhs = roles[RoutedForwardFfiOperandRole.FIRST_CONTRACT_RHS]
    second_rhs = roles[RoutedForwardFfiOperandRole.SECOND_CONTRACT_RHS]
    target_symbol = target.replace(".", "_")
    scalar_sources = "\n".join(
        (
            map_output[0].generated_cuda.source,
            plan.fold_stage.generated_contribution_cuda.source,
            plan.fold_stage.generated_reducer_cuda.source,
        )
    )
    semantic_record = {
        "contracts": [contract.output_shape for contract in plan.contracts],
        "map": map_program.digest,
        "contribution": plan.fold_stage.contribution_program.digest,
        "reducer": plan.fold_stage.reducer_program.digest,
        "layout": {
            "logical_edges": dimensions["logical_edges"],
            "logical_features": dimensions["logical_features"],
            "segments": dimensions["segments"],
            "padded_rows": dimensions["padded_rows"],
            "feature_stride": dimensions["feature_stride"],
            "segment_stride": dimensions["segment_stride"],
        },
        "operand_roles": [(operand.role.value, operand.value.shape) for operand in plan.operands],
        "numerical_policy": plan.region.numerical_policy.value,
    }
    semantic_digest = hashlib.sha256(json.dumps(semantic_record, sort_keys=True).encode()).hexdigest()
    fill_value = plan.segmented_layout.fill_value
    source = f"""// Generated by tile_lifetime.xla_routed_forward_ffi; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{scalar_sources}

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

ffi::Error Contract(
    cudaStream_t stream,
    const float* lhs,
    const float* rhs,
    float* output,
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
      CUDA_R_32F,
      features,
      lhs,
      CUDA_R_32F,
      reduction,
      &beta,
      output,
      CUDA_R_32F,
      features,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmEx failed with status " + std::to_string(static_cast<int>(status)));
  }}
  return ffi::Error::Success();
}}

__global__ void ShuttleSegmentedMapKernel(
    const float* projection,
    const bool* validity,
    float* mapped) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kPaddedRows * kLogicalFeatures * kSegments) {{
    return;
  }}
  const int segment = index % kSegments;
  const int feature = (index / kSegments) % kLogicalFeatures;
  const int row = index / (kSegments * kLogicalFeatures);
  const int physical_feature = feature * kFeatureStride + segment * kSegmentStride;
  const int validity_index = (segment * kPaddedRows + row) * kLogicalFeatures + feature;
  const bool valid = row < kLogicalEdges && validity[validity_index];
  mapped[row * kPhysicalMapFeatures + physical_feature] =
      valid ? {map_output[0].generated_cuda.symbol}({", ".join(map_arguments)}) : {fill_value}f;
}}

__global__ void ShuttleSourceFoldKernel(
    const float* initial,
    const std::int32_t* source_indices,
    const std::uint16_t* route_weight,
    const float* routed_output,
    float* output) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kOutputRows * kOutputFeatures) {{
    return;
  }}
  const int source = index / kOutputFeatures;
  const int feature = index - source * kOutputFeatures;
  float accumulator = initial[index];
  for (int edge = 0; edge < kLogicalEdges; ++edge) {{
    if (source_indices[edge] != source) {{
      continue;
    }}
    const float contribution = {plan.fold_stage.generated_contribution_cuda.symbol}({", ".join(contribution_arguments)});
    accumulator = {plan.fold_stage.generated_reducer_cuda.symbol}(accumulator, contribution);
  }}
  output[index] = accumulator;
}}

ffi::Error ShuttleRoutedForward(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {",\n    ".join(ffi_arguments)},
    ffi::Result<ffi::Buffer<ffi::F32, 2>> output_buffer) {{
{chr(10).join(data_bindings)}
  auto* output = output_buffer->typed_data();
  auto projection_storage = scratch.Allocate(
      sizeof(float) * kPaddedRows * kFirstOutputFeatures, alignof(float));
  auto mapped_storage = scratch.Allocate(
      sizeof(float) * kPaddedRows * kPhysicalMapFeatures, alignof(float));
  auto routed_output_storage = scratch.Allocate(
      sizeof(float) * kPaddedRows * kOutputFeatures, alignof(float));
  if (!projection_storage || !mapped_storage || !routed_output_storage) {{
    return ffi::Error::Internal("failed to allocate routed-region scratch storage");
  }}
  auto* projection = static_cast<float*>(*projection_storage);
  auto* mapped = static_cast<float*>(*mapped_storage);
  auto* routed_output = static_cast<float*>(*routed_output_storage);

  ffi::Error first_contract = Contract(
      stream, input{first_lhs}, input{first_rhs}, projection,
      kPaddedRows, kFirstReduction, kFirstOutputFeatures);
  if (first_contract.failure()) {{
    return first_contract;
  }}
  constexpr int kThreads = 256;
  constexpr int kMapItems = kPaddedRows * kLogicalFeatures * kSegments;
  constexpr int kMapBlocks = (kMapItems + kThreads - 1) / kThreads;
  ShuttleSegmentedMapKernel<<<kMapBlocks, kThreads, 0, stream>>>(
      projection, input{validity}, mapped);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttleSegmentedMapKernel: ") + cudaGetErrorString(status));
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
      input{initial}, input{indices}, input{route_weight}, routed_output, output);
  status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttleSourceFoldKernel: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleRoutedForwardBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{chr(10).join(ffi_bindings)}
      .Ret<ffi::Buffer<ffi::F32, 2>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleRoutedForward,
    ShuttleRoutedForwardBinding());

extern "C" int shuttle_routed_forward_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    return GeneratedRoutedForwardFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def replace_routed_forward_region_with_custom_call(
    hlo_text: str,
    plan: RoutedForwardTypedFfiCodegenPlan,
    *,
    target: str,
) -> str:
    """Replace one recovered routed region with a typed-FFI tuple call."""
    outputs = plan.region.boundary.outputs
    if len(outputs) != 1:
        raise ValueError("routed FFI prototype requires exactly one live region output")
    output = outputs[0]
    pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(output.instruction)} = .*$", re.MULTILINE)
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one textual definition for routed output %{output.instruction}")
    match = matches[0]
    operands = ", ".join(f"%{operand.value.instruction}" for operand in plan.operands)
    constraints = ", ".join(operand.value.shape for operand in plan.operands)
    call_name = "shuttle_generated_routed_forward_region"
    call = (
        f"{match.group('indent')}%{call_name} = ({output.shape}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}\n"
    )
    replacement = (
        f"{match.group('indent')}%{output.instruction} = {output.shape} get-tuple-element(%{call_name}), index=0"
    )
    return hlo_text[: match.start()] + call + replacement + hlo_text[match.end() :]


def evaluate_routed_forward_plan(
    plan: RoutedForwardTypedFfiCodegenPlan,
    operands: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Execute the recovered generic program as a deterministic CPU reference."""
    dimensions = _validated_dimensions(plan)
    if len(operands) != len(plan.operands):
        raise ValueError("runtime operand count does not match the routed FFI plan")
    by_role = {binding.role: np.asarray(operands[index]) for index, binding in enumerate(plan.operands)}
    projection = np.matmul(
        by_role[RoutedForwardFfiOperandRole.FIRST_CONTRACT_LHS].astype(np.float32),
        by_role[RoutedForwardFfiOperandRole.FIRST_CONTRACT_RHS].astype(np.float32),
    ).astype(np.float32)
    validity = by_role[RoutedForwardFfiOperandRole.SEGMENT_VALIDITY]
    mapped = np.full(
        (dimensions["padded_rows"], dimensions["physical_map_features"]),
        plan.segmented_layout.fill_value if plan.segmented_layout is not None else 0.0,
        dtype=np.float32,
    )
    map_program = plan.map_stage.scalar_outputs[0].scalar_program
    for row in range(dimensions["logical_edges"]):
        for feature in range(dimensions["logical_features"]):
            scalar_inputs = {
                value.input_name: projection[row, feature + value.input_index.feature_offset]
                for value in map_program.inputs
                if value.input_name is not None and value.input_index is not None
            }
            value = evaluate_cast_scalar_program(map_program, scalar_inputs)
            for segment in range(dimensions["segments"]):
                if validity[segment, row, feature]:
                    physical_feature = feature * dimensions["feature_stride"] + segment * dimensions["segment_stride"]
                    mapped[row, physical_feature] = value
    routed_output = np.matmul(
        mapped,
        by_role[RoutedForwardFfiOperandRole.SECOND_CONTRACT_RHS].astype(np.float32),
    ).astype(np.float32)
    output = by_role[RoutedForwardFfiOperandRole.FOLD_INITIAL].astype(np.float32).copy()
    source_indices = by_role[RoutedForwardFfiOperandRole.FOLD_INDICES].reshape(-1)
    route_weight = by_role[RoutedForwardFfiOperandRole.FOLD_CONTRIBUTION_INPUT].reshape(-1).astype(np.float32)
    for edge in range(dimensions["logical_edges"]):
        source = int(source_indices[edge])
        for feature in range(dimensions["output_features"]):
            contribution = evaluate_cast_scalar_program(
                plan.fold_stage.contribution_program,
                {
                    "input0_r0_f0": routed_output[edge, feature],
                    "input1_r0_f0": route_weight[edge],
                },
            )
            output[source, feature] = evaluate_cast_scalar_program(
                plan.fold_stage.reducer_program,
                {"input0": output[source, feature], "input1": contribution},
            )
    return output


def _validated_dimensions(plan: RoutedForwardTypedFfiCodegenPlan) -> dict[str, int]:
    if plan.disposition is not RoutedForwardCodegenDisposition.READY or plan.segmented_layout is None:
        raise ValueError("routed CUDA generation requires a READY segmented-layout plan")
    if len(plan.contracts) != 2:
        raise ValueError("routed CUDA prototype requires exactly two Contracts")
    first_output_dtype, first_output, first_layout = _parse_shape(plan.contracts[0].output_shape)
    second_output_dtype, second_output, second_layout = _parse_shape(plan.contracts[1].output_shape)
    physical_dtype, physical_map, physical_layout = _parse_shape(plan.map_stage.physical_output_shape)
    output_dtype, output, output_layout = _parse_shape(plan.fold_stage.output_shape)
    if any(dtype != "f32" for dtype in (first_output_dtype, second_output_dtype, physical_dtype, output_dtype)):
        raise ValueError("routed CUDA prototype requires FP32 Contract, Map, and Fold storage")
    if any(
        layout != tuple(reversed(range(len(shape))))
        for shape, layout in (
            (first_output, first_layout),
            (second_output, second_layout),
            (physical_map, physical_layout),
            (output, output_layout),
        )
    ):
        raise ValueError("routed CUDA prototype requires row-major physical arrays")
    first_lhs = _shape_for_role(plan, RoutedForwardFfiOperandRole.FIRST_CONTRACT_LHS)
    first_rhs = _shape_for_role(plan, RoutedForwardFfiOperandRole.FIRST_CONTRACT_RHS)
    second_rhs = _shape_for_role(plan, RoutedForwardFfiOperandRole.SECOND_CONTRACT_RHS)
    if first_lhs[0] != "f32" or first_rhs[0] != "f32" or second_rhs[0] != "f32":
        raise ValueError("generic cuBLAS prototype requires FP32 Contract operands")
    if first_lhs[1][0] != first_output[0] or first_rhs[1] != (first_lhs[1][1], first_output[1]):
        raise ValueError("first Contract dimensions do not match physical operands")
    if physical_map[0] != second_output[0] or second_rhs[1] != (physical_map[1], second_output[1]):
        raise ValueError("second Contract dimensions do not match physical operands")
    index_map = plan.segmented_layout.index_map
    if index_map.row_stride != 1 or index_map.row_offset != 0:
        raise ValueError("routed CUDA prototype requires compact edge rows at physical row zero")
    if physical_map != (index_map.padded_row_extent, index_map.logical_feature_extent * index_map.segment_count):
        raise ValueError("segmented index relation does not cover the physical Map output")
    if second_output[1] != output[1]:
        raise ValueError("Fold feature extent does not match the second Contract")
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
    plan: RoutedForwardTypedFfiCodegenPlan,
    role: RoutedForwardFfiOperandRole,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    operands = tuple(operand for operand in plan.operands if operand.role is role)
    if len(operands) != 1:
        raise ValueError(f"routed FFI plan has {len(operands)} operands for role {role.value}")
    return _parse_shape(operands[0].value.shape)


def _parse_shape(shape: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"unsupported physical array shape {shape!r}")
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    layout = tuple(int(value) for value in match.group("layout").split(","))
    return match.group("dtype"), dimensions, layout
