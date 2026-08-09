# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a typed-FFI executor for a generic rank-two BF16 Contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import (
    EntryRegionValue,
    HloComputation,
    HloInstruction,
    parse_hlo_module_text,
)
from tile_lifetime.xla_relation_program_recovery import ContractDimensionMap, RoutedForwardContractStage

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")


@dataclass(frozen=True)
class RankTwoContractNumericalContract:
    """Finite-precision behavior of one generated Contract."""

    input_dtype: str
    accumulation_dtype: str
    output_dtype: str
    output_rounding: str
    numerical_policy: NumericalPolicy
    deterministic_accumulation: bool


@dataclass(frozen=True)
class RankTwoBf16ContractTypedFfiPlan:
    """One entry-local rank-two Contract with explicit physical operands."""

    instruction: str
    lhs: EntryRegionValue
    rhs: EntryRegionValue
    output_shape: str
    dimensions: ContractDimensionMap
    external_users: tuple[str, ...]
    api_version: int
    numerical_contract: RankTwoContractNumericalContract


@dataclass(frozen=True)
class GeneratedRankTwoContractFfi:
    """CUDA source and semantic identity for one generated Contract."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str


@dataclass(frozen=True)
class RankTwoContractReplacementAudit:
    """Post-replacement evidence for a generated rank-two Contract."""

    call_instruction: str
    operands: tuple[str, str]
    output_shape: str
    external_users: tuple[str, ...]
    api_version: int


def plan_rank_two_bf16_contract_typed_ffi(
    hlo_text: str,
    contract: RoutedForwardContractStage,
    *,
    numerical_policy: NumericalPolicy = NumericalPolicy.ALLOW_ROUNDING_REORDER,
) -> RankTwoBf16ContractTypedFfiPlan:
    """Bind one recovered generic Contract to its physical entry values."""
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("a generated Contract cannot preserve an unspecified bitwise dot reduction tree")
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    instruction_name = _entry_instruction_name(module.entry, contract.node)
    instruction = instructions[instruction_name]
    if instruction.opcode != "dot" or len(instruction.operands) != 2:
        raise ValueError(f"%{instruction_name} is not a two-input Contract")
    lhs_name = _entry_instruction_name(module.entry, contract.lhs)
    rhs_name = _entry_instruction_name(module.entry, contract.rhs)
    if instruction.operands != (lhs_name, rhs_name):
        raise ValueError(
            f"Contract %{instruction_name} operands disagree with the recovered stage: "
            f"{instruction.operands} != {(lhs_name, rhs_name)}"
        )
    if instruction.shape != contract.output_shape:
        raise ValueError("Contract result shape changed after structural recovery")
    users = _entry_users(entry)
    plan = RankTwoBf16ContractTypedFfiPlan(
        instruction=instruction_name,
        lhs=EntryRegionValue(lhs_name, instructions[lhs_name].shape),
        rhs=EntryRegionValue(rhs_name, instructions[rhs_name].shape),
        output_shape=instruction.shape,
        dimensions=contract.dimensions,
        external_users=users[instruction_name],
        api_version=1,
        numerical_contract=RankTwoContractNumericalContract(
            input_dtype="bf16",
            accumulation_dtype="f32",
            output_dtype="bf16",
            output_rounding="round_to_nearest_even",
            numerical_policy=numerical_policy,
            deterministic_accumulation=True,
        ),
    )
    _validated_dimensions(plan)
    return plan


def generate_cuda_rank_two_contract_ffi(
    plan: RankTwoBf16ContractTypedFfiPlan,
    *,
    target: str,
) -> GeneratedRankTwoContractFfi:
    """Generate a row-major BF16 Contract with FP32 accumulation."""
    dimensions = _validated_dimensions(plan)
    semantic_record = {
        "lhs_shape": plan.lhs.shape,
        "rhs_shape": plan.rhs.shape,
        "output_shape": plan.output_shape,
        "dimensions": {
            "lhs_contracting": plan.dimensions.lhs_contracting,
            "rhs_contracting": plan.dimensions.rhs_contracting,
            "lhs_batch": plan.dimensions.lhs_batch,
            "rhs_batch": plan.dimensions.rhs_batch,
            "lhs_output": plan.dimensions.lhs_output,
            "rhs_output": plan.dimensions.rhs_output,
        },
        "numerical_contract": {
            "input_dtype": plan.numerical_contract.input_dtype,
            "accumulation_dtype": plan.numerical_contract.accumulation_dtype,
            "output_dtype": plan.numerical_contract.output_dtype,
            "output_rounding": plan.numerical_contract.output_rounding,
            "numerical_policy": plan.numerical_contract.numerical_policy.value,
            "deterministic_accumulation": plan.numerical_contract.deterministic_accumulation,
        },
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target_symbol = _target_symbol(target)
    source = f"""// Generated from a generic rank-two Contract; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {{
constexpr int kRows = {dimensions["rows"]};
constexpr int kReduction = {dimensions["reduction"]};
constexpr int kFeatures = {dimensions["features"]};
std::atomic<int> call_count{{0}};
thread_local cublasHandle_t contract_handle = nullptr;

ffi::Error ShuttleRankTwoContract(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> lhs_buffer,
    ffi::Buffer<ffi::BF16, 2> rhs_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output_buffer) {{
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
  status = cublasGemmEx(
      contract_handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      kFeatures,
      kRows,
      kReduction,
      &alpha,
      rhs,
      CUDA_R_16BF,
      kFeatures,
      lhs,
      CUDA_R_16BF,
      kReduction,
      &beta,
      output,
      CUDA_R_16BF,
      kFeatures,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {{
    return ffi::Error::Internal(
        "cublasGemmEx failed with status " + std::to_string(static_cast<int>(status)));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleRankTwoContractBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleRankTwoContract,
    ShuttleRankTwoContractBinding());

extern "C" int shuttle_rank_two_contract_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    return GeneratedRankTwoContractFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def replace_rank_two_contract_with_custom_call(
    hlo_text: str,
    plan: RankTwoBf16ContractTypedFfiPlan,
    *,
    target: str,
) -> str:
    """Replace one Contract while preserving its physical result name."""
    pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(plan.instruction)} = .*?$", re.MULTILINE)
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one physical definition for Contract %{plan.instruction}")
    match = matches[0]
    replacement = (
        f"{match.group('indent')}%{plan.instruction} = {plan.output_shape} "
        f"custom-call(%{plan.lhs.instruction}, %{plan.rhs.instruction}), "
        f'custom_call_target="{target}", '
        f"operand_layout_constraints={{{plan.lhs.shape}, {plan.rhs.shape}}}, "
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    return hlo_text[: match.start()] + replacement + hlo_text[match.end() :]


def audit_rank_two_contract_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: RankTwoBf16ContractTypedFfiPlan,
    *,
    target: str,
) -> RankTwoContractReplacementAudit:
    """Verify the generated Contract's ABI and preserved consumer boundary."""
    original_module = parse_hlo_module_text(original_hlo)
    original_entry = original_module.computation(original_module.entry)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    original_users = _entry_users(original_entry)
    transformed_users = _entry_users(transformed_entry)
    call = _unique_target_instruction(transformed_entry, target)
    if call.name != plan.instruction:
        raise ValueError(f"generated Contract moved from %{plan.instruction} to %{call.name}")
    if call.operands != (plan.lhs.instruction, plan.rhs.instruction):
        raise ValueError(f"generated Contract operands changed: {call.operands}")
    if call.shape != plan.output_shape:
        raise ValueError(f"generated Contract output changed: {call.shape}")
    if "api_version=API_VERSION_TYPED_FFI" not in call.attributes:
        raise ValueError("generated Contract does not use typed FFI API version 1")
    expected_constraints = f"operand_layout_constraints={{{plan.lhs.shape}, {plan.rhs.shape}}}"
    if expected_constraints not in call.attributes:
        raise ValueError("generated Contract operand layout constraints changed")
    expected_users = original_users[plan.instruction]
    if expected_users != plan.external_users or transformed_users[call.name] != expected_users:
        raise ValueError("generated Contract consumer boundary changed")
    return RankTwoContractReplacementAudit(
        call_instruction=call.name,
        operands=(plan.lhs.instruction, plan.rhs.instruction),
        output_shape=call.shape,
        external_users=expected_users,
        api_version=plan.api_version,
    )


def evaluate_rank_two_contract_plan(
    plan: RankTwoBf16ContractTypedFfiPlan,
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Evaluate the generated Contract semantics on CPU."""
    dimensions = _validated_dimensions(plan)
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != (dimensions["rows"], dimensions["reduction"]):
        raise ValueError("runtime lhs shape does not match the Contract plan")
    if rhs.shape != (dimensions["reduction"], dimensions["features"]):
        raise ValueError("runtime rhs shape does not match the Contract plan")
    return _round_bf16_array(lhs.astype(np.float32) @ rhs.astype(np.float32))


def _validated_dimensions(plan: RankTwoBf16ContractTypedFfiPlan) -> dict[str, int]:
    lhs_dtype, lhs, lhs_layout = _parse_shape(plan.lhs.shape)
    rhs_dtype, rhs, rhs_layout = _parse_shape(plan.rhs.shape)
    output_dtype, output, output_layout = _parse_shape(plan.output_shape)
    if {lhs_dtype, rhs_dtype, output_dtype} != {"bf16"}:
        raise ValueError("rank-two Contract requires BF16 storage")
    if any(
        layout != tuple(reversed(range(len(shape))))
        for shape, layout in ((lhs, lhs_layout), (rhs, rhs_layout), (output, output_layout))
    ):
        raise ValueError("rank-two Contract requires row-major physical arrays")
    if any(len(shape) != 2 for shape in (lhs, rhs, output)):
        raise ValueError("rank-two Contract requires rank-two physical arrays")
    dimensions = plan.dimensions
    if (
        dimensions.lhs_contracting != (1,)
        or dimensions.rhs_contracting != (0,)
        or dimensions.lhs_batch
        or dimensions.rhs_batch
        or dimensions.lhs_output != (0,)
        or dimensions.rhs_output != (1,)
    ):
        raise ValueError("unsupported rank-two Contract dimension relation")
    if lhs[1] != rhs[0] or output != (lhs[0], rhs[1]):
        raise ValueError("rank-two Contract shapes disagree with its dimension relation")
    return {"rows": lhs[0], "reduction": lhs[1], "features": rhs[1]}


def _entry_instruction_name(entry: str, node: str) -> str:
    prefix = f"{entry}/"
    if not node.startswith(prefix):
        raise ValueError(f"recovered node {node!r} is outside entry computation {entry!r}")
    return node.removeprefix(prefix).split("/", 1)[0]


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    users: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)
    return {instruction: tuple(values) for instruction, values in users.items()}


def _unique_target_instruction(entry: HloComputation, target: str) -> HloInstruction:
    attribute = f'custom_call_target="{target}"'
    matches = tuple(
        instruction
        for instruction in entry.instructions
        if instruction.opcode == "custom-call" and attribute in instruction.attributes
    )
    if len(matches) != 1:
        raise ValueError(f"expected one generated Contract target {target!r}, found {len(matches)}")
    return matches[0]


def _target_symbol(target: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", target)
    if not symbol or symbol[0].isdigit():
        raise ValueError(f"typed-FFI target cannot form a C++ symbol: {target!r}")
    return symbol


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
