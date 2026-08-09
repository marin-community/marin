# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a CUDA typed-FFI executor for one Contract with several scalar Maps."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from tile_lifetime.cast_scalar_program import CastScalarExpression, generate_cuda_scalar_body
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_relation_program_recovery import (
    SharedContractMapDependence,
    SharedContractMapOutput,
    SharedContractMultiMapOperandRole,
    SharedContractMultiMapRegionRecord,
    form_shared_contract_multi_map_region,
)
from tile_lifetime.xla_shared_contract_multimap import (
    SharedContractMultiMapReplacementAudit,
    audit_shared_contract_multi_map_replacement,
    replace_shared_contract_multi_map_region_with_custom_call,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")


@dataclass(frozen=True)
class GeneratedSharedContractMultiMapFfi:
    """CUDA source and semantic identity for a generated Contract/multi-Map."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str
    output_count: int
    scalar_semantic_digests: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SharedContractMultiMapFfiCompilation:
    """Recovered plan, generated executor, and audited physical HLO replacement."""

    plan: SharedContractMultiMapRegionRecord
    generated: GeneratedSharedContractMultiMapFfi
    transformed_hlo: str
    replacement_audit: SharedContractMultiMapReplacementAudit


@dataclass(frozen=True)
class _OutputLowering:
    output: SharedContractMapOutput
    shape: tuple[int, int, int]
    output_strides: tuple[int, int, int]
    validity_operand: int
    validity_strides: tuple[int, int, int]
    scalar_sources: tuple[str, ...]
    scalar_expression: str


def compile_shared_contract_multi_map_ffi(
    hlo_text: str,
    *,
    target: str,
    numerical_policy: NumericalPolicy = NumericalPolicy.ALLOW_ROUNDING_REORDER,
) -> SharedContractMultiMapFfiCompilation:
    """Recover, generate, replace, and audit one shared Contract/multi-Map region."""
    plan = form_shared_contract_multi_map_region(hlo_text, numerical_policy=numerical_policy)
    generated = generate_cuda_shared_contract_multi_map_ffi(plan, target=target)
    transformed_hlo = replace_shared_contract_multi_map_region_with_custom_call(hlo_text, plan, target=target)
    audit = audit_shared_contract_multi_map_replacement(
        hlo_text,
        transformed_hlo,
        plan,
        target=target,
    )
    return SharedContractMultiMapFfiCompilation(
        plan=plan,
        generated=generated,
        transformed_hlo=transformed_hlo,
        replacement_audit=audit,
    )


def generate_cuda_shared_contract_multi_map_ffi(
    plan: SharedContractMultiMapRegionRecord,
    *,
    target: str,
) -> GeneratedSharedContractMultiMapFfi:
    """Generate one rank-two BF16 Contract and all plan-provided scalar Maps."""
    dimensions = _validated_contract_dimensions(plan)
    roles = {operand.role: index for index, operand in enumerate(plan.operands)}
    if len(roles) != len(plan.operands):
        raise ValueError("shared Contract/multi-Map operand roles must be unique")
    required_roles = {
        SharedContractMultiMapOperandRole.CONTRACT_LHS,
        SharedContractMultiMapOperandRole.CONTRACT_RHS,
        SharedContractMultiMapOperandRole.MAP_AUXILIARY,
    }
    if not required_roles.issubset(roles):
        raise ValueError("shared Contract/multi-Map plan is missing a data operand")
    if not plan.outputs:
        raise ValueError("shared Contract/multi-Map executor requires at least one output")

    output_lowerings = tuple(
        _lower_output(plan, output, index=index, operand_roles=roles) for index, output in enumerate(plan.outputs)
    )
    ffi_types = {"bf16": "ffi::BF16", "pred": "ffi::PRED"}
    ffi_arguments: list[str] = []
    ffi_bindings: list[str] = []
    data_bindings: list[str] = []
    for index, operand in enumerate(plan.operands):
        dtype, shape, _ = _parse_shape(operand.value.shape)
        if dtype not in ffi_types:
            raise ValueError(f"unsupported shared Contract/multi-Map FFI dtype {dtype!r}")
        ffi_arguments.append(f"ffi::Buffer<{ffi_types[dtype]}, {len(shape)}> input{index}_buffer")
        ffi_bindings.append(f"      .Arg<ffi::Buffer<{ffi_types[dtype]}, {len(shape)}>>()")
        if dtype == "bf16":
            data_bindings.append(
                f"  const auto* input{index} = reinterpret_cast<const std::uint16_t*>(input{index}_buffer.typed_data());"
            )
        else:
            data_bindings.append(f"  const auto* input{index} = input{index}_buffer.typed_data();")
    result_arguments = tuple(
        f"ffi::Result<ffi::Buffer<ffi::BF16, {len(lowering.shape)}>> output{index}_buffer"
        for index, lowering in enumerate(output_lowerings)
    )
    result_bindings = tuple(
        f"      .Ret<ffi::Buffer<ffi::BF16, {len(lowering.shape)}>>()" for lowering in output_lowerings
    )
    result_data = tuple(
        f"  auto* output{index} = reinterpret_cast<std::uint16_t*>(output{index}_buffer->typed_data());"
        for index in range(len(output_lowerings))
    )
    kernel_sources = tuple(
        _render_output_kernel(lowering, output_index=index) for index, lowering in enumerate(output_lowerings)
    )
    auxiliary_operand = roles[SharedContractMultiMapOperandRole.MAP_AUXILIARY]
    launches = tuple(
        _render_output_launch(lowering, output_index=index, auxiliary_operand=auxiliary_operand)
        for index, lowering in enumerate(output_lowerings)
    )
    semantic_record = _semantic_record(plan, dimensions)
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target_symbol = re.sub(r"[^A-Za-z0-9_]", "_", target)
    if not target_symbol or target_symbol[0].isdigit():
        raise ValueError(f"typed-FFI target cannot form a C++ symbol: {target!r}")
    lhs = roles[SharedContractMultiMapOperandRole.CONTRACT_LHS]
    rhs = roles[SharedContractMultiMapOperandRole.CONTRACT_RHS]
    scalar_sources = tuple(source for lowering in output_lowerings for source in lowering.scalar_sources)
    source = f"""// Generated by tile_lifetime.xla_shared_contract_multimap_ffi; do not edit.
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
constexpr int kContractRows = {dimensions["rows"]};
constexpr int kContractReduction = {dimensions["reduction"]};
constexpr int kContractFeatures = {dimensions["features"]};
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
    std::uint16_t* output) {{
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
      kContractFeatures,
      kContractRows,
      kContractReduction,
      &alpha,
      rhs,
      CUDA_R_16BF,
      kContractFeatures,
      lhs,
      CUDA_R_16BF,
      kContractReduction,
      &beta,
      output,
      CUDA_R_16BF,
      kContractFeatures,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmEx failed with status " + std::to_string(static_cast<int>(status)));
  }}
  return ffi::Error::Success();
}}

{chr(10).join(kernel_sources)}

ffi::Error ShuttleSharedContractMultiMap(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {",\n    ".join((*ffi_arguments, *result_arguments))}) {{
{chr(10).join(data_bindings)}
{chr(10).join(result_data)}
  auto projection_storage = scratch.Allocate(
      sizeof(std::uint16_t) * kContractRows * kContractFeatures, alignof(std::uint16_t));
  if (!projection_storage) {{
    return ffi::Error::Internal("failed to allocate shared Contract output storage");
  }}
  auto* projection = static_cast<std::uint16_t*>(*projection_storage);
  ffi::Error contract = Contract(stream, input{lhs}, input{rhs}, projection);
  if (contract.failure()) {{
    return contract;
  }}
  constexpr int kThreads = 256;
{chr(10).join(launches)}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleSharedContractMultiMapBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{chr(10).join(ffi_bindings)}
{chr(10).join(result_bindings)};
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleSharedContractMultiMap,
    ShuttleSharedContractMultiMapBinding());

extern "C" int shuttle_shared_contract_multi_map_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    return GeneratedSharedContractMultiMapFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
        output_count=len(output_lowerings),
        scalar_semantic_digests=tuple(
            tuple(scalar.scalar_program.digest for scalar in output.scalar_outputs) for output in plan.outputs
        ),
    )


def _lower_output(
    plan: SharedContractMultiMapRegionRecord,
    output: SharedContractMapOutput,
    *,
    index: int,
    operand_roles: dict[SharedContractMultiMapOperandRole, int],
) -> _OutputLowering:
    dtype, physical_shape, output_layout = _parse_shape(output.value.shape)
    if dtype != "bf16" or len(physical_shape) != 3:
        raise ValueError("shared Contract/multi-Map outputs must be rank-three BF16 arrays")
    if physical_shape[1] < output.logical_row_extent or physical_shape[2] != output.logical_feature_extent:
        raise ValueError("shared Contract/multi-Map output does not cover its logical row/feature domain")
    if output.validity_role not in operand_roles:
        raise ValueError(f"output {index} has no bound validity operand")
    validity_operand = operand_roles[output.validity_role]
    validity_binding = plan.operands[validity_operand]
    validity_dtype, validity_shape, validity_layout = _parse_shape(validity_binding.value.shape)
    if validity_dtype != "pred" or validity_shape != physical_shape:
        raise ValueError("shared Contract/multi-Map validity must match its physical output shape")
    scalar_sources: list[str] = []
    scalar_calls: list[str] = []
    expected_offset = 0
    for scalar_index, scalar in enumerate(output.scalar_outputs):
        if scalar.feature_offset != expected_offset or scalar.feature_extent <= 0:
            raise ValueError("scalar Map outputs must cover one contiguous positive feature domain")
        expected_offset += scalar.feature_extent
        symbol = f"shuttle_scalar_map_{index}_{scalar_index}"
        generated = generate_cuda_scalar_body(scalar.scalar_program, symbol=symbol)
        scalar_sources.append(generated.source)
        arguments = tuple(
            _scalar_argument(
                output,
                scalar_feature_offset=scalar.feature_offset,
                scalar_feature_extent=scalar.feature_extent,
                scalar_input=scalar_input,
                plan=plan,
            )
            for scalar_input in scalar.scalar_program.inputs
        )
        upper = scalar.feature_offset + scalar.feature_extent
        scalar_calls.append(f"feature < {upper} ? {symbol}({', '.join(arguments)})")
    if expected_offset != output.logical_feature_extent:
        raise ValueError("scalar Map outputs do not cover the declared logical feature domain")
    return _OutputLowering(
        output=output,
        shape=(physical_shape[0], physical_shape[1], physical_shape[2]),
        output_strides=_physical_strides(physical_shape, output_layout),
        validity_operand=validity_operand,
        validity_strides=_physical_strides(validity_shape, validity_layout),
        scalar_sources=tuple(scalar_sources),
        scalar_expression=" : ".join((*scalar_calls, "0.0f")),
    )


def _scalar_argument(
    output: SharedContractMapOutput,
    *,
    scalar_feature_offset: int,
    scalar_feature_extent: int,
    scalar_input: CastScalarExpression,
    plan: SharedContractMultiMapRegionRecord,
) -> str:
    input_name = scalar_input.input_name
    input_index = scalar_input.input_index
    if input_name is None or input_index is None:
        raise ValueError("generated scalar Map input lacks a concrete index relation")
    source_row = f"row + ({input_index.row_offset})"
    source_feature = f"feature - {scalar_feature_offset} + ({input_index.feature_offset})"
    if output.dependence is SharedContractMapDependence.CONTRACT_ONLY:
        _, source_shape, _ = _parse_shape(plan.contract.output_shape)
        source = "projection"
        source_strides = (source_shape[1], 1)
    elif input_name.startswith("input0"):
        _, source_shape, auxiliary_layout = _shape_for_role(plan, SharedContractMultiMapOperandRole.MAP_AUXILIARY)
        source = "auxiliary"
        source_strides = _physical_strides(source_shape, auxiliary_layout)
    elif input_name.startswith("input1"):
        _, source_shape, _ = _parse_shape(plan.contract.output_shape)
        source = "projection"
        source_strides = (source_shape[1], 1)
    else:
        raise ValueError(f"unsupported shared scalar Map input {input_name!r}")
    if not (
        0 <= input_index.row_offset
        and output.logical_row_extent - 1 + input_index.row_offset < source_shape[0]
        and 0 <= input_index.feature_offset
        and scalar_feature_extent - 1 + input_index.feature_offset < source_shape[1]
    ):
        raise ValueError("generated scalar Map index relation escapes its source array")
    return (
        f"shuttle_bf16_to_f32({source}[({source_row}) * {source_strides[0]} + ({source_feature}) * {source_strides[1]}])"
    )


def _render_output_kernel(lowering: _OutputLowering, *, output_index: int) -> str:
    segments, padded_rows, features = lowering.shape
    output_strides = lowering.output_strides
    validity_strides = lowering.validity_strides
    return f"""__global__ void ShuttleMapOutput{output_index}Kernel(
    const std::uint16_t* projection,
    const std::uint16_t* auxiliary,
    const bool* validity,
    std::uint16_t* output) {{
  constexpr int kSegments = {segments};
  constexpr int kPaddedRows = {padded_rows};
  constexpr int kFeatures = {features};
  constexpr int kLogicalRows = {lowering.output.logical_row_extent};
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kSegments * kPaddedRows * kFeatures) {{
    return;
  }}
  const int feature = index % kFeatures;
  const int row = (index / kFeatures) % kPaddedRows;
  const int segment = index / (kFeatures * kPaddedRows);
  const int validity_index =
      segment * {validity_strides[0]} + row * {validity_strides[1]} + feature * {validity_strides[2]};
  const int output_index =
      segment * {output_strides[0]} + row * {output_strides[1]} + feature * {output_strides[2]};
  const bool valid = row < kLogicalRows && validity[validity_index];
  const float value = valid ? {lowering.scalar_expression} : 0.0f;
  output[output_index] = shuttle_f32_to_bf16(value);
}}"""


def _render_output_launch(
    lowering: _OutputLowering,
    *,
    output_index: int,
    auxiliary_operand: int,
) -> str:
    items = lowering.shape[0] * lowering.shape[1] * lowering.shape[2]
    return f"""  constexpr int kOutput{output_index}Items = {items};
  constexpr int kOutput{output_index}Blocks = (kOutput{output_index}Items + kThreads - 1) / kThreads;
  ShuttleMapOutput{output_index}Kernel<<<kOutput{output_index}Blocks, kThreads, 0, stream>>>(
      projection, input{auxiliary_operand}, input{lowering.validity_operand}, output{output_index});
  cudaError_t output{output_index}_status = cudaGetLastError();
  if (output{output_index}_status != cudaSuccess) {{
    return ffi::Error::Internal(
        std::string("ShuttleMapOutput{output_index}Kernel: ") + cudaGetErrorString(output{output_index}_status));
  }}"""


def _validated_contract_dimensions(plan: SharedContractMultiMapRegionRecord) -> dict[str, int]:
    numerical = plan.numerical_contract
    if (
        numerical.input_dtype != "bf16"
        or numerical.accumulation_dtype != "f32"
        or numerical.contract_output_dtype != "bf16"
        or numerical.contract_output_rounding != "round_to_nearest_even"
    ):
        raise ValueError("shared Contract/multi-Map executor requires explicit BF16/FP32/BF16 semantics")
    lhs_dtype, lhs, lhs_layout = _shape_for_role(plan, SharedContractMultiMapOperandRole.CONTRACT_LHS)
    rhs_dtype, rhs, rhs_layout = _shape_for_role(plan, SharedContractMultiMapOperandRole.CONTRACT_RHS)
    output_dtype, output, output_layout = _parse_shape(plan.contract.output_shape)
    if {lhs_dtype, rhs_dtype, output_dtype} != {"bf16"}:
        raise ValueError("shared Contract/multi-Map Contract requires BF16 physical arrays")
    if lhs_layout != (1, 0) or rhs_layout != (1, 0) or output_layout != (1, 0):
        raise ValueError("shared Contract/multi-Map Contract requires row-major rank-two arrays")
    dimensions = plan.contract.dimensions
    if (
        dimensions.lhs_contracting != (1,)
        or dimensions.rhs_contracting != (0,)
        or dimensions.lhs_batch
        or dimensions.rhs_batch
        or dimensions.lhs_output != (0,)
        or dimensions.rhs_output != (1,)
    ):
        raise ValueError("unsupported shared rank-two Contract dimension relation")
    if len(lhs) != 2 or len(rhs) != 2 or output != (lhs[0], rhs[1]) or lhs[1] != rhs[0]:
        raise ValueError("shared Contract/multi-Map Contract shapes disagree")
    auxiliary_dtype, auxiliary, _ = _shape_for_role(plan, SharedContractMultiMapOperandRole.MAP_AUXILIARY)
    if auxiliary_dtype != "bf16" or len(auxiliary) != 2:
        raise ValueError("shared Contract/multi-Map auxiliary must be a rank-two BF16 array")
    return {"rows": lhs[0], "reduction": lhs[1], "features": rhs[1]}


def _semantic_record(
    plan: SharedContractMultiMapRegionRecord,
    dimensions: dict[str, int],
) -> dict[str, object]:
    numerical = plan.numerical_contract
    return {
        "contract": {
            "dimensions": dimensions,
            "dimension_relation": {
                "lhs_contracting": plan.contract.dimensions.lhs_contracting,
                "rhs_contracting": plan.contract.dimensions.rhs_contracting,
                "lhs_output": plan.contract.dimensions.lhs_output,
                "rhs_output": plan.contract.dimensions.rhs_output,
            },
        },
        "operands": [(operand.role.value, operand.value.shape) for operand in plan.operands],
        "outputs": [
            {
                "dependence": output.dependence.value,
                "shape": output.value.shape,
                "logical_rows": output.logical_row_extent,
                "logical_features": output.logical_feature_extent,
                "validity_role": output.validity_role.value,
                "scalar_outputs": [
                    {
                        "feature_offset": scalar.feature_offset,
                        "feature_extent": scalar.feature_extent,
                        "semantic_digest": scalar.scalar_program.digest,
                    }
                    for scalar in output.scalar_outputs
                ],
            }
            for output in plan.outputs
        ],
        "numerical_contract": {
            "input_dtype": numerical.input_dtype,
            "accumulation_dtype": numerical.accumulation_dtype,
            "contract_output_dtype": numerical.contract_output_dtype,
            "contract_output_rounding": numerical.contract_output_rounding,
            "scalar_policy": numerical.scalar_policy.value,
            "numerical_policy": numerical.numerical_policy.value,
        },
    }


def _shape_for_role(
    plan: SharedContractMultiMapRegionRecord,
    role: SharedContractMultiMapOperandRole,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    operands = tuple(operand for operand in plan.operands if operand.role is role)
    if len(operands) != 1:
        raise ValueError(f"shared Contract/multi-Map plan has {len(operands)} operands for role {role.value}")
    return _parse_shape(operands[0].value.shape)


def _physical_strides(shape: tuple[int, ...], layout: tuple[int, ...]) -> tuple[int, ...]:
    if sorted(layout) != list(range(len(shape))):
        raise ValueError(f"invalid physical layout {layout} for shape {shape}")
    strides = [0] * len(shape)
    current = 1
    for axis in layout:
        strides[axis] = current
        current *= shape[axis]
    return tuple(strides)


def _parse_shape(shape: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"unsupported physical array shape {shape!r}")
    return (
        match.group("dtype"),
        tuple(int(value) for value in match.group("dims").split(",") if value),
        tuple(int(value) for value in match.group("layout").split(",")),
    )
