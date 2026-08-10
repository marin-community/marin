# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a fixed-capacity segmented Contract/Map/Contract input adjoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from tile_lifetime.cast_scalar_program import (
    CastScalarExpression,
    CastScalarKind,
    CastScalarNumericalPolicy,
    CastScalarProgram,
    ScalarIndexRelation,
    evaluate_cast_scalar_program,
    generate_cuda_scalar_body,
)
from tile_lifetime.xla_relation_program_recovery import (
    RoutedInputAdjointTypedFfiCodegenPlan,
    ScalarMapOutputRecord,
)


@dataclass(frozen=True)
class SegmentedInputAdjointFfiPlan:
    """Generic per-segment reverse Contracts surrounding one generated Map."""

    segment_count: int
    capacity: int
    input_features: int
    intermediate_features: int
    scalar_outputs: tuple[ScalarMapOutputRecord, ...]
    numerical_policy: CastScalarNumericalPolicy

    @property
    def pair_features(self) -> int:
        return 2 * self.intermediate_features

    @property
    def fusion_seams(self) -> SegmentedInputAdjointFusionSeams:
        """Describe tile-level producer and consumer seams for fused candidates."""
        return SegmentedInputAdjointFusionSeams(
            consumed_edge_tile=SegmentedEdgeTileDomain(
                segment_count=self.segment_count,
                capacity=self.capacity,
                feature_extent=self.input_features,
            ),
            pair_state_tile=SegmentedEdgeTileDomain(
                segment_count=self.segment_count,
                capacity=self.capacity,
                feature_extent=self.pair_features,
            ),
            produced_edge_tile=SegmentedEdgeTileDomain(
                segment_count=self.segment_count,
                capacity=self.capacity,
                feature_extent=self.input_features,
            ),
            readiness=SegmentedTileReadiness(
                requires_consumed_edge_tile=True,
                requires_saved_pair_tile=True,
                validity_predicated=True,
                produced_edge_ready_after="second_contract",
            ),
            buffer_elision=SegmentedBufferElisionLegality(
                consumed_edge_payload=True,
                pair_state=True,
                produced_edge_payload=True,
                exact_edge_identity_required=True,
                all_pair_consumers_must_share_tile_lifetime=True,
            ),
            maximum_logical_edges=self.segment_count * self.capacity,
            standalone_ffi_materializes_full_buffers=True,
            fused_candidate_requires_full_materialization=False,
        )


@dataclass(frozen=True)
class SegmentedEdgeTileDomain:
    """Bounded segment/row/feature domain shared by payload and Contract tiles."""

    segment_count: int
    capacity: int
    feature_extent: int
    axes: tuple[str, str, str] = ("segment", "row_within_segment", "feature")


@dataclass(frozen=True)
class SegmentedTileReadiness:
    """Logical readiness conditions for an adjacent fused schedule."""

    requires_consumed_edge_tile: bool
    requires_saved_pair_tile: bool
    validity_predicated: bool
    produced_edge_ready_after: str


@dataclass(frozen=True)
class SegmentedBufferElisionLegality:
    """Conditions under which standalone ABI buffers may become tile-local."""

    consumed_edge_payload: bool
    pair_state: bool
    produced_edge_payload: bool
    exact_edge_identity_required: bool
    all_pair_consumers_must_share_tile_lifetime: bool


@dataclass(frozen=True)
class SegmentedInputAdjointFusionSeams:
    """Physical metadata for fusing adjacent transport and Contract schedules."""

    consumed_edge_tile: SegmentedEdgeTileDomain
    pair_state_tile: SegmentedEdgeTileDomain
    produced_edge_tile: SegmentedEdgeTileDomain
    readiness: SegmentedTileReadiness
    buffer_elision: SegmentedBufferElisionLegality
    maximum_logical_edges: int
    standalone_ffi_materializes_full_buffers: bool
    fused_candidate_requires_full_materialization: bool


@dataclass(frozen=True)
class GeneratedSegmentedInputAdjointFfi:
    """CUDA source and semantic identity for one segmented input adjoint."""

    target: str
    handler_symbol: str
    source: str
    semantic_digest: str
    source_digest: str


@dataclass(frozen=True)
class SegmentedInputAdjointResourceAudit:
    """Non-allocating physical resource and Contract-work accounting."""

    segment_count: int
    capacity: int
    input_features: int
    intermediate_features: int
    projection_scratch_bytes: int
    pair_output_bytes: int
    input_adjoint_output_bytes: int
    total_generated_bytes: int
    map_items: int
    contract_flops: int
    rejected_dense_first_lhs_bytes: int
    rejected_dense_validity_bytes: int
    rejected_dense_auxiliary_output_bytes: int
    rejected_dense_mapped_scratch_bytes: int
    rejected_dense_total_intermediate_bytes: int
    rejected_dense_map_items: int
    rejected_dense_contract_flops: int

    @property
    def dense_contract_work_ratio(self) -> int:
        return self.rejected_dense_contract_flops // self.contract_flops


def plan_segmented_input_adjoint_ffi(
    template: RoutedInputAdjointTypedFfiCodegenPlan,
    *,
    segment_count: int,
    capacity: int,
    input_features: int,
    intermediate_features: int,
) -> SegmentedInputAdjointFfiPlan:
    """Retarget a recovered pair-Map VJP to explicit fixed-capacity segments."""
    dimensions = (segment_count, capacity, input_features, intermediate_features)
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"segmented input-adjoint dimensions must be positive, got {dimensions}")
    template_outputs = template.map_stage.scalar_outputs
    if len(template_outputs) != 2:
        raise ValueError("recovered pair-Map VJP must have exactly two contiguous outputs")
    template_intermediate = template_outputs[0].feature_extent
    if tuple((output.feature_offset, output.feature_extent) for output in template_outputs) != (
        (0, template_intermediate),
        (template_intermediate, template_intermediate),
    ):
        raise ValueError("recovered pair-Map VJP does not have two equal contiguous feature panels")
    scalar_outputs = tuple(
        _retarget_scalar_output(
            output,
            feature_offset=index * intermediate_features,
            feature_extent=intermediate_features,
            old_panel_extent=template_intermediate,
            new_panel_extent=intermediate_features,
        )
        for index, output in enumerate(template_outputs)
    )
    for output in scalar_outputs:
        for value in output.scalar_program.inputs:
            if value.input_name is None or value.input_index is None or value.input_index.row_offset != 0:
                raise ValueError("segmented pair-Map supports named same-row scalar inputs only")
            if not value.input_name.startswith(("input0", "input1")):
                raise ValueError(f"unsupported segmented pair-Map input {value.input_name!r}")
    return SegmentedInputAdjointFfiPlan(
        segment_count=segment_count,
        capacity=capacity,
        input_features=input_features,
        intermediate_features=intermediate_features,
        scalar_outputs=scalar_outputs,
        numerical_policy=template.region.numerical_policy,
    )


def _retarget_scalar_output(
    output: ScalarMapOutputRecord,
    *,
    feature_offset: int,
    feature_extent: int,
    old_panel_extent: int,
    new_panel_extent: int,
) -> ScalarMapOutputRecord:
    def retarget(expression: CastScalarExpression) -> CastScalarExpression:
        if expression.kind is CastScalarKind.INPUT:
            assert expression.input_index is not None
            offset = expression.input_index.feature_offset
            if offset not in (0, old_panel_extent):
                raise ValueError(f"pair-Map input feature offset {offset} is not panel-relative")
            return replace(
                expression,
                input_index=ScalarIndexRelation(
                    row_offset=expression.input_index.row_offset,
                    feature_offset=new_panel_extent if offset == old_panel_extent else 0,
                ),
            )
        return replace(expression, operands=tuple(retarget(operand) for operand in expression.operands))

    program = CastScalarProgram(
        expression=retarget(output.scalar_program.expression),
        numerical_policy=output.scalar_program.numerical_policy,
    )
    return ScalarMapOutputRecord(
        feature_offset=feature_offset,
        feature_extent=feature_extent,
        scalar_program=program,
        generated_cuda=generate_cuda_scalar_body(program, symbol=output.generated_cuda.symbol),
    )


def audit_segmented_input_adjoint_resources(plan: SegmentedInputAdjointFfiPlan) -> SegmentedInputAdjointResourceAudit:
    """Account for the segmented family and the rejected dense expert expansion."""
    groups = plan.segment_count
    rows = groups * plan.capacity
    hidden = plan.input_features
    intermediate = plan.intermediate_features
    pair_features = plan.pair_features
    contract_flops = 2 * rows * hidden * intermediate + 2 * rows * pair_features * hidden
    projection_bytes = 2 * rows * intermediate
    pair_bytes = 2 * rows * pair_features
    input_adjoint_bytes = 2 * rows * hidden
    dense_first_lhs_bytes = 2 * rows * groups * hidden
    dense_validity_bytes = rows * groups * pair_features
    dense_auxiliary_bytes = 2 * rows * groups * pair_features
    dense_mapped_bytes = dense_auxiliary_bytes
    dense_total_bytes = (
        dense_first_lhs_bytes
        + dense_validity_bytes
        + dense_auxiliary_bytes
        + dense_mapped_bytes
        + projection_bytes
        + input_adjoint_bytes
    )
    return SegmentedInputAdjointResourceAudit(
        segment_count=groups,
        capacity=plan.capacity,
        input_features=hidden,
        intermediate_features=intermediate,
        projection_scratch_bytes=projection_bytes,
        pair_output_bytes=pair_bytes,
        input_adjoint_output_bytes=input_adjoint_bytes,
        total_generated_bytes=projection_bytes + pair_bytes + input_adjoint_bytes,
        map_items=rows * pair_features,
        contract_flops=contract_flops,
        rejected_dense_first_lhs_bytes=dense_first_lhs_bytes,
        rejected_dense_validity_bytes=dense_validity_bytes,
        rejected_dense_auxiliary_output_bytes=dense_auxiliary_bytes,
        rejected_dense_mapped_scratch_bytes=dense_mapped_bytes,
        rejected_dense_total_intermediate_bytes=dense_total_bytes,
        rejected_dense_map_items=rows * groups * pair_features,
        rejected_dense_contract_flops=groups * contract_flops,
    )


def generate_cuda_segmented_input_adjoint_ffi(
    plan: SegmentedInputAdjointFfiPlan,
    *,
    target: str,
) -> GeneratedSegmentedInputAdjointFfi:
    """Generate two group-batched Contracts and the recovered pair-Map VJP."""
    _validate_cublas_dimensions(plan)
    map_calls: list[str] = []
    scalar_sources: list[str] = []
    for output in plan.scalar_outputs:
        arguments: list[str] = []
        for value in output.scalar_program.inputs:
            assert value.input_name is not None
            assert value.input_index is not None
            local_feature = f"feature - {output.feature_offset} + {value.input_index.feature_offset}"
            if value.input_name.startswith("input0"):
                arguments.append(
                    "shuttle_bf16_to_f32(projection[(segment * kCapacity + row) * kIntermediate + "
                    + local_feature
                    + "])"
                )
            else:
                arguments.append(
                    "shuttle_bf16_to_f32(saved_pair[(segment * kCapacity + row) * kPairFeatures + "
                    + local_feature
                    + "])"
                )
        upper = output.feature_offset + output.feature_extent
        map_calls.append(f"feature < {upper} ? {output.generated_cuda.symbol}({', '.join(arguments)})")
        scalar_sources.append(output.generated_cuda.source)
    map_expression = " : ".join((*map_calls, "0.0f"))
    semantic_record = {
        "segments": plan.segment_count,
        "capacity": plan.capacity,
        "input_features": plan.input_features,
        "intermediate_features": plan.intermediate_features,
        "map": [output.scalar_program.digest for output in plan.scalar_outputs],
        "numerical_policy": plan.numerical_policy.value,
        "contracts": (
            "segment[C,H] @ segment[H,I] -> segment[C,I]",
            "segment[C,2I] @ segment[2I,H] -> segment[C,H]",
        ),
    }
    semantic_digest = hashlib.sha256(json.dumps(semantic_record, sort_keys=True).encode()).hexdigest()
    target_symbol = target.replace(".", "_")
    source = f"""// Generated generic fixed-capacity segmented input adjoint; do not edit.
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{chr(10).join(scalar_sources)}

namespace ffi = xla::ffi;

namespace {{
constexpr int kSegments = {plan.segment_count};
constexpr int kCapacity = {plan.capacity};
constexpr int kInputFeatures = {plan.input_features};
constexpr int kIntermediate = {plan.intermediate_features};
constexpr int kPairFeatures = {plan.pair_features};
constexpr std::uint64_t kMapItems =
    static_cast<std::uint64_t>(kSegments) * kCapacity * kPairFeatures;
constexpr std::uint64_t kProjectionItems =
    static_cast<std::uint64_t>(kSegments) * kCapacity * kIntermediate;
constexpr int kThreads = 256;
constexpr std::uint64_t kMapBlocks = (kMapItems + kThreads - 1) / kThreads;
static_assert(kMapBlocks <= std::numeric_limits<unsigned int>::max());
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

ffi::Error ContractStridedBatched(
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
  const long long lhs_stride = static_cast<long long>(rows) * reduction;
  const long long rhs_stride = static_cast<long long>(reduction) * features;
  const long long output_stride = static_cast<long long>(rows) * features;
  status = cublasGemmStridedBatchedEx(
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
      rhs_stride,
      lhs,
      CUDA_R_16BF,
      reduction,
      lhs_stride,
      &beta,
      output,
      CUDA_R_16BF,
      features,
      output_stride,
      kSegments,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmStridedBatchedEx failed with status " +
        std::to_string(static_cast<int>(status)));
  }}
  return ffi::Error::Success();
}}

__global__ void ShuttlePairMapKernel(
    const std::uint16_t* projection,
    const std::uint16_t* saved_pair,
    const bool* validity,
    std::uint16_t* pair_cotangent) {{
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kMapItems) {{
    return;
  }}
  const std::uint64_t feature = index % kPairFeatures;
  const std::uint64_t row = (index / kPairFeatures) % kCapacity;
  const std::uint64_t segment = index / (static_cast<std::uint64_t>(kPairFeatures) * kCapacity);
  const bool valid = validity[segment * kCapacity + row];
  const float value = valid ? {map_expression} : 0.0f;
  pair_cotangent[index] = shuttle_f32_to_bf16(value);
}}

ffi::Error ShuttleSegmentedInputAdjoint(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::BF16, 3> padded_cotangent_buffer,
    ffi::Buffer<ffi::BF16, 3> saved_pair_buffer,
    ffi::Buffer<ffi::PRED, 2> validity_buffer,
    ffi::Buffer<ffi::BF16, 3> down_input_adjoint_weight_buffer,
    ffi::Buffer<ffi::BF16, 3> gate_up_input_adjoint_weight_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 3>> pair_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 3>> input_adjoint_buffer) {{
  const auto* padded_cotangent =
      reinterpret_cast<const std::uint16_t*>(padded_cotangent_buffer.typed_data());
  const auto* saved_pair =
      reinterpret_cast<const std::uint16_t*>(saved_pair_buffer.typed_data());
  const auto* validity = validity_buffer.typed_data();
  const auto* down_input_adjoint_weight =
      reinterpret_cast<const std::uint16_t*>(down_input_adjoint_weight_buffer.typed_data());
  const auto* gate_up_input_adjoint_weight =
      reinterpret_cast<const std::uint16_t*>(gate_up_input_adjoint_weight_buffer.typed_data());
  auto* pair_cotangent =
      reinterpret_cast<std::uint16_t*>(pair_cotangent_buffer->typed_data());
  auto* input_adjoint =
      reinterpret_cast<std::uint16_t*>(input_adjoint_buffer->typed_data());
  auto projection_storage = scratch.Allocate(
      sizeof(std::uint16_t) * kProjectionItems, alignof(std::uint16_t));
  if (!projection_storage) {{
    return ffi::Error::Internal("failed to allocate segmented input-adjoint projection scratch");
  }}
  auto* projection = static_cast<std::uint16_t*>(*projection_storage);
  ffi::Error first_contract = ContractStridedBatched(
      stream,
      padded_cotangent,
      down_input_adjoint_weight,
      projection,
      kCapacity,
      kInputFeatures,
      kIntermediate);
  if (first_contract.failure()) {{
    return first_contract;
  }}
  ShuttlePairMapKernel<<<static_cast<unsigned int>(kMapBlocks), kThreads, 0, stream>>>(
      projection, saved_pair, validity, pair_cotangent);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(std::string("ShuttlePairMapKernel: ") + cudaGetErrorString(status));
  }}
  ffi::Error second_contract = ContractStridedBatched(
      stream,
      pair_cotangent,
      gate_up_input_adjoint_weight,
      input_adjoint,
      kCapacity,
      kPairFeatures,
      kInputFeatures);
  if (second_contract.failure()) {{
    return second_contract;
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleSegmentedInputAdjointBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::PRED, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Ret<ffi::Buffer<ffi::BF16, 3>>()
      .Ret<ffi::Buffer<ffi::BF16, 3>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleSegmentedInputAdjoint,
    ShuttleSegmentedInputAdjointBinding());

extern "C" int shuttle_segmented_input_adjoint_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    return GeneratedSegmentedInputAdjointFfi(
        target=target,
        handler_symbol=target_symbol,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def evaluate_segmented_input_adjoint_plan(
    plan: SegmentedInputAdjointFfiPlan,
    padded_cotangent: np.ndarray,
    saved_pair: np.ndarray,
    validity: np.ndarray,
    down_input_adjoint_weight: np.ndarray,
    gate_up_input_adjoint_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the segmented family in deterministic reference order on CPU."""
    expected = _operand_shapes(plan)
    values = (
        np.asarray(padded_cotangent),
        np.asarray(saved_pair),
        np.asarray(validity),
        np.asarray(down_input_adjoint_weight),
        np.asarray(gate_up_input_adjoint_weight),
    )
    for value, shape in zip(values, expected, strict=True):
        if value.shape != shape:
            raise ValueError(f"segmented input-adjoint operand shape {value.shape} != {shape}")
    projection = _round_bf16_array(np.matmul(values[0].astype(np.float32), values[3].astype(np.float32)))
    pair_cotangent = np.zeros(expected[1], dtype=np.float32)
    for segment in range(plan.segment_count):
        for row in range(plan.capacity):
            if not values[2][segment, row]:
                continue
            for output in plan.scalar_outputs:
                for local_feature in range(output.feature_extent):
                    feature = output.feature_offset + local_feature
                    scalar_inputs: dict[str, float] = {}
                    for scalar_input in output.scalar_program.inputs:
                        assert scalar_input.input_name is not None
                        assert scalar_input.input_index is not None
                        input_feature = local_feature + scalar_input.input_index.feature_offset
                        if scalar_input.input_name.startswith("input0"):
                            scalar_inputs[scalar_input.input_name] = projection[segment, row, input_feature]
                        else:
                            scalar_inputs[scalar_input.input_name] = values[1][segment, row, input_feature]
                    pair_cotangent[segment, row, feature] = evaluate_cast_scalar_program(
                        output.scalar_program,
                        scalar_inputs,
                    )
    pair_cotangent = _round_bf16_array(pair_cotangent)
    input_adjoint = _round_bf16_array(np.matmul(pair_cotangent.astype(np.float32), values[4].astype(np.float32)))
    return pair_cotangent, input_adjoint


def _operand_shapes(plan: SegmentedInputAdjointFfiPlan) -> tuple[tuple[int, ...], ...]:
    return (
        (plan.segment_count, plan.capacity, plan.input_features),
        (plan.segment_count, plan.capacity, plan.pair_features),
        (plan.segment_count, plan.capacity),
        (plan.segment_count, plan.input_features, plan.intermediate_features),
        (plan.segment_count, plan.pair_features, plan.input_features),
    )


def _validate_cublas_dimensions(plan: SegmentedInputAdjointFfiPlan) -> None:
    int32_max = np.iinfo(np.int32).max
    dimensions = (
        plan.segment_count,
        plan.capacity,
        plan.input_features,
        plan.intermediate_features,
        plan.pair_features,
    )
    if any(value > int32_max for value in dimensions):
        raise ValueError("individual cuBLAS Contract dimensions must fit signed 32-bit integers")
    audit = audit_segmented_input_adjoint_resources(plan)
    if audit.map_items > np.iinfo(np.uint32).max * 256:
        raise ValueError("segmented Map launch exceeds the CUDA one-dimensional grid")


def _round_bf16_array(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)
