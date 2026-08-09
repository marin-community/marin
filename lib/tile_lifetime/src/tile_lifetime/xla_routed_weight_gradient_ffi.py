# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate typed-FFI group-batched Contracts for routed weight adjoints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from tile_lifetime.xla_relation_program_recovery import (
    RoutedWeightGradientFfiOperandRole,
    RoutedWeightGradientTypedFfiCodegenPlan,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")


@dataclass(frozen=True)
class GeneratedGroupBatchedContractFfi:
    """CUDA source and semantic identity for one generated Contract."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str


def generate_cuda_group_batched_contract_ffi(
    plan: RoutedWeightGradientTypedFfiCodegenPlan,
    *,
    target: str,
) -> GeneratedGroupBatchedContractFfi:
    """Generate one deterministic group-batched BF16 Contract executor."""
    dimensions = _validated_dimensions(plan)
    target_symbol = target.replace(".", "_")
    semantic_record = {
        "contract": {
            "node": plan.contract.node,
            "dimensions": {
                "lhs_contracting": plan.contract.dimensions.lhs_contracting,
                "rhs_contracting": plan.contract.dimensions.rhs_contracting,
                "lhs_batch": plan.contract.dimensions.lhs_batch,
                "rhs_batch": plan.contract.dimensions.rhs_batch,
                "lhs_output": plan.contract.dimensions.lhs_output,
                "rhs_output": plan.contract.dimensions.rhs_output,
            },
            "output_shape": plan.contract.output_shape,
        },
        "operands": [(operand.role.value, operand.value.instruction, operand.value.shape) for operand in plan.operands],
        "numerical_contract": {
            "input_dtype": plan.numerical_contract.input_dtype,
            "accumulation_dtype": plan.numerical_contract.accumulation_dtype,
            "output_dtype": plan.numerical_contract.output_dtype,
            "output_rounding": plan.numerical_contract.output_rounding,
            "numerical_policy": plan.numerical_contract.numerical_policy.value,
            "deterministic_accumulation": plan.numerical_contract.deterministic_accumulation,
        },
        "output_alias_operand": plan.output_alias_operand,
    }
    semantic_digest = hashlib.sha256(json.dumps(semantic_record, sort_keys=True).encode()).hexdigest()
    source = f"""// Generated generic group-batched Contract; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {{
constexpr int kGroups = {dimensions["groups"]};
constexpr int kReduction = {dimensions["reduction"]};
constexpr int kLhsFeatures = {dimensions["lhs_features"]};
constexpr int kRhsFeatures = {dimensions["rhs_features"]};
std::atomic<int> call_count{{0}};
thread_local cublasHandle_t contract_handle = nullptr;

ffi::Error ShuttleGroupBatchedContract(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 3> lhs_buffer,
    ffi::Buffer<ffi::BF16, 3> rhs_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 3>> output_buffer) {{
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
  const auto* lhs = reinterpret_cast<const std::uint16_t*>(lhs_buffer.typed_data());
  const auto* rhs = reinterpret_cast<const std::uint16_t*>(rhs_buffer.typed_data());
  auto* output = reinterpret_cast<std::uint16_t*>(output_buffer->typed_data());
  status = cublasGemmStridedBatchedEx(
      contract_handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      kRhsFeatures,
      kLhsFeatures,
      kReduction,
      &alpha,
      rhs,
      CUDA_R_16BF,
      kRhsFeatures,
      static_cast<long long>(kReduction) * kRhsFeatures,
      lhs,
      CUDA_R_16BF,
      kLhsFeatures,
      static_cast<long long>(kReduction) * kLhsFeatures,
      &beta,
      output,
      CUDA_R_16BF,
      kRhsFeatures,
      static_cast<long long>(kLhsFeatures) * kRhsFeatures,
      kGroups,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmStridedBatchedEx failed with status " +
        std::to_string(static_cast<int>(status)));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleGroupBatchedContractBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Ret<ffi::Buffer<ffi::BF16, 3>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleGroupBatchedContract,
    ShuttleGroupBatchedContractBinding());

extern "C" int shuttle_routed_weight_gradient_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    return GeneratedGroupBatchedContractFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def replace_group_batched_contract_with_custom_call(
    hlo_text: str,
    plan: RoutedWeightGradientTypedFfiCodegenPlan,
    *,
    target: str,
) -> str:
    """Replace one recovered Contract while preserving its external collective."""
    instruction = plan.region.insertion_instruction
    operands = ", ".join(f"%{operand.value.instruction}" for operand in plan.operands)
    constraints = ", ".join(operand.value.shape for operand in plan.operands)
    pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(instruction)} = .*?$", re.MULTILINE)
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one physical definition for weight-gradient Contract %{instruction}")
    match = matches[0]
    replacement = (
        f"{match.group('indent')}%{instruction} = {plan.contract.output_shape} custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    return hlo_text[: match.start()] + replacement + hlo_text[match.end() :]


def evaluate_group_batched_contract_plan(
    plan: RoutedWeightGradientTypedFfiCodegenPlan,
    operands: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Execute the generated group-batched Contract semantics on CPU."""
    dimensions = _validated_dimensions(plan)
    if len(operands) != len(plan.operands):
        raise ValueError("runtime operand count does not match the weight-gradient plan")
    by_role = {binding.role: np.asarray(operands[index]) for index, binding in enumerate(plan.operands)}
    lhs = by_role[RoutedWeightGradientFfiOperandRole.LHS]
    rhs = by_role[RoutedWeightGradientFfiOperandRole.RHS]
    if lhs.shape != (
        dimensions["groups"],
        dimensions["reduction"],
        dimensions["lhs_features"],
    ):
        raise ValueError("runtime lhs shape does not match the generated Contract")
    if rhs.shape != (
        dimensions["groups"],
        dimensions["reduction"],
        dimensions["rhs_features"],
    ):
        raise ValueError("runtime rhs shape does not match the generated Contract")
    accumulated = np.einsum("gkm,gkn->gmn", lhs.astype(np.float32), rhs.astype(np.float32))
    return _round_bf16_array(accumulated)


def _validated_dimensions(plan: RoutedWeightGradientTypedFfiCodegenPlan) -> dict[str, int]:
    roles = tuple(operand.role for operand in plan.operands)
    if roles != tuple(RoutedWeightGradientFfiOperandRole):
        raise ValueError("weight-gradient operands must be bound in Contract lhs/rhs order")
    lhs_dtype, lhs, lhs_layout = _parse_shape(plan.operands[0].value.shape)
    rhs_dtype, rhs, rhs_layout = _parse_shape(plan.operands[1].value.shape)
    output_dtype, output, output_layout = _parse_shape(plan.contract.output_shape)
    if {lhs_dtype, rhs_dtype, output_dtype} != {"bf16"}:
        raise ValueError("weight-gradient executor requires BF16 physical arrays")
    if any(
        layout != tuple(reversed(range(len(shape))))
        for shape, layout in ((lhs, lhs_layout), (rhs, rhs_layout), (output, output_layout))
    ):
        raise ValueError("weight-gradient executor requires row-major physical arrays")
    dimensions = plan.contract.dimensions
    if (
        dimensions.lhs_batch != (0,)
        or dimensions.rhs_batch != (0,)
        or dimensions.lhs_contracting != (1,)
        or dimensions.rhs_contracting != (1,)
        or dimensions.lhs_output != (2,)
        or dimensions.rhs_output != (2,)
    ):
        raise ValueError("unsupported group-batched Contract dimension relation")
    if lhs[0] != rhs[0] or lhs[1] != rhs[1] or output != (lhs[0], lhs[2], rhs[2]):
        raise ValueError("weight-gradient Contract shapes disagree with its dimension relation")
    if plan.output_alias_operand is not None:
        raise ValueError("weight-gradient output must be freshly allocated before its placement collective")
    return {
        "groups": lhs[0],
        "reduction": lhs[1],
        "lhs_features": lhs[2],
        "rhs_features": rhs[2],
    }


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
