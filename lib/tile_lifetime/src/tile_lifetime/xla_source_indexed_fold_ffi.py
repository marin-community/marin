# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a deterministic typed-FFI source-indexed scalar Fold."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np

from tile_lifetime.cast_scalar_program import (
    CastScalarNumericalPolicy,
    CastScalarProgram,
    GeneratedCudaScalarBody,
    evaluate_cast_scalar_program,
)
from tile_lifetime.xla_hlo_recovery import EntryRegionValue, HloComputation, HloInstruction, parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import (
    RoutedForwardContractStage,
    RoutedInputAdjointFfiOperandRole,
    RoutedInputAdjointTypedFfiCodegenPlan,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")
_WRAPPER_OPCODES = frozenset({"bitcast", "copy", "reshape", "slice"})


@dataclass(frozen=True)
class SourceIndexedFoldNumericalContract:
    """Finite-precision and ordering promise of a source-indexed Fold."""

    input_dtype: str
    state_dtype: str
    output_dtype: str
    numerical_policy: CastScalarNumericalPolicy
    deterministic: bool
    atomic_accumulation: bool


@dataclass(frozen=True)
class SourceIndexedFoldTypedFfiPlan:
    """One source-indexed Fold with generated scalar contribution and reducer."""

    instruction: str
    initial: EntryRegionValue
    source_indices: EntryRegionValue
    contributions: EntryRegionValue
    contribution_wrappers: tuple[str, ...]
    output_shape: str
    contribution_program: CastScalarProgram
    generated_contribution_cuda: GeneratedCudaScalarBody
    reducer_program: CastScalarProgram
    generated_reducer_cuda: GeneratedCudaScalarBody
    contribution_input_name: str
    reducer_accumulator_input_name: str
    reducer_contribution_input_name: str
    external_users: tuple[str, ...]
    api_version: int
    numerical_contract: SourceIndexedFoldNumericalContract


@dataclass(frozen=True)
class GeneratedSourceIndexedFoldFfi:
    """CUDA source and semantic identity for a generated source-indexed Fold."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str


@dataclass(frozen=True)
class SourceIndexedFoldReplacementAudit:
    """Post-replacement ABI and dataflow evidence for one generated Fold."""

    call_instruction: str
    operands: tuple[str, str, str]
    output_shape: str
    external_users: tuple[str, ...]
    contribution_wrappers: tuple[str, ...]
    api_version: int


def plan_source_indexed_fold_typed_ffi(
    hlo_text: str,
    input_adjoint: RoutedInputAdjointTypedFfiCodegenPlan,
    source_contract: RoutedForwardContractStage,
) -> SourceIndexedFoldTypedFfiPlan:
    """Recover a generic source-indexed Fold from an existing scalar Fold plan."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    fold_stage = input_adjoint.fold_stage
    instruction_name = _entry_instruction_name(module.entry, fold_stage.fold)
    instruction = instructions[instruction_name]
    if instruction.opcode != "scatter" or len(instruction.operands) != 3:
        raise ValueError(f"%{instruction_name} is not a three-input source-indexed Fold")
    by_role = {operand.role: operand.value for operand in input_adjoint.operands}
    initial = by_role[RoutedInputAdjointFfiOperandRole.FOLD_INITIAL]
    source_indices = by_role[RoutedInputAdjointFfiOperandRole.FOLD_INDICES]
    if instruction.operands[:2] != (initial.instruction, source_indices.instruction):
        raise ValueError("source-indexed Fold initial state or index operand changed")
    contributions = EntryRegionValue(instruction.operands[2], instructions[instruction.operands[2]].shape)
    source_instruction = _entry_instruction_name(module.entry, source_contract.node)
    contribution_wrappers = _wrapper_path(instructions, source_instruction, contributions.instruction)
    _validate_source_indexed_scatter_attributes(instruction)
    contribution_input_name = _single_input_name(fold_stage.contribution_program, "contribution")
    reducer_inputs = tuple(value.input_name for value in fold_stage.reducer_program.inputs)
    if reducer_inputs != ("input0", "input1"):
        raise ValueError(f"source-indexed Fold reducer inputs must preserve parameter order, found {reducer_inputs}")
    if fold_stage.contribution_program.numerical_policy is not CastScalarNumericalPolicy.SOURCE_ORDERED:
        raise ValueError("source-indexed Fold contribution must preserve source ordering")
    if fold_stage.reducer_program.numerical_policy is not CastScalarNumericalPolicy.SOURCE_ORDERED:
        raise ValueError("source-indexed Fold reducer must preserve source ordering")
    users = _entry_users(entry)
    plan = SourceIndexedFoldTypedFfiPlan(
        instruction=instruction_name,
        initial=initial,
        source_indices=source_indices,
        contributions=contributions,
        contribution_wrappers=contribution_wrappers,
        output_shape=instruction.shape,
        contribution_program=fold_stage.contribution_program,
        generated_contribution_cuda=fold_stage.generated_contribution_cuda,
        reducer_program=fold_stage.reducer_program,
        generated_reducer_cuda=fold_stage.generated_reducer_cuda,
        contribution_input_name=contribution_input_name,
        reducer_accumulator_input_name="input0",
        reducer_contribution_input_name="input1",
        external_users=users[instruction_name],
        api_version=1,
        numerical_contract=SourceIndexedFoldNumericalContract(
            input_dtype="bf16",
            state_dtype="bf16",
            output_dtype="bf16",
            numerical_policy=CastScalarNumericalPolicy.SOURCE_ORDERED,
            deterministic=True,
            atomic_accumulation=False,
        ),
    )
    _validated_dimensions(plan)
    return plan


def generate_cuda_source_indexed_fold_ffi(
    plan: SourceIndexedFoldTypedFfiPlan,
    *,
    target: str,
) -> GeneratedSourceIndexedFoldFfi:
    """Generate one-thread-per-output source-ordered Fold without atomics."""
    dimensions = _validated_dimensions(plan)
    semantic_record = {
        "initial_shape": plan.initial.shape,
        "source_indices_shape": plan.source_indices.shape,
        "contributions_shape": plan.contributions.shape,
        "output_shape": plan.output_shape,
        "contribution": plan.contribution_program.digest,
        "reducer": plan.reducer_program.digest,
        "numerical_contract": {
            "input_dtype": plan.numerical_contract.input_dtype,
            "state_dtype": plan.numerical_contract.state_dtype,
            "output_dtype": plan.numerical_contract.output_dtype,
            "numerical_policy": plan.numerical_contract.numerical_policy.value,
            "deterministic": plan.numerical_contract.deterministic,
            "atomic_accumulation": plan.numerical_contract.atomic_accumulation,
        },
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target_symbol = _target_symbol(target)
    source = f"""// Generated from a generic source-indexed Fold; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{plan.generated_contribution_cuda.source}
{plan.generated_reducer_cuda.source}

namespace ffi = xla::ffi;

namespace {{
constexpr int kSources = {dimensions["sources"]};
constexpr int kEdges = {dimensions["edges"]};
constexpr int kFeatures = {dimensions["features"]};
std::atomic<int> call_count{{0}};

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

__global__ void ShuttleSourceIndexedFoldKernel(
    const std::uint16_t* initial,
    const std::int32_t* source_indices,
    const std::uint16_t* contributions,
    std::uint16_t* output) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kSources * kFeatures) {{
    return;
  }}
  const int source = index / kFeatures;
  const int feature = index - source * kFeatures;
  float accumulator = shuttle_bf16_to_f32(initial[index]);
  for (int edge = 0; edge < kEdges; ++edge) {{
    if (source_indices[edge] != source) {{
      continue;
    }}
    const float contribution = {plan.generated_contribution_cuda.symbol}(
        shuttle_bf16_to_f32(contributions[edge * kFeatures + feature]));
    accumulator = {plan.generated_reducer_cuda.symbol}(accumulator, contribution);
  }}
  output[index] = shuttle_f32_to_bf16(accumulator);
}}

ffi::Error ShuttleSourceIndexedFold(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> initial_buffer,
    ffi::Buffer<ffi::S32, 2> source_indices_buffer,
    ffi::Buffer<ffi::BF16, {dimensions["contribution_rank"]}> contributions_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output_buffer) {{
  const auto* initial = reinterpret_cast<const std::uint16_t*>(initial_buffer.typed_data());
  const auto* source_indices = source_indices_buffer.typed_data();
  const auto* contributions =
      reinterpret_cast<const std::uint16_t*>(contributions_buffer.typed_data());
  auto* output = reinterpret_cast<std::uint16_t*>(output_buffer->typed_data());
  constexpr int kThreads = 256;
  constexpr int kItems = kSources * kFeatures;
  constexpr int kBlocks = (kItems + kThreads - 1) / kThreads;
  ShuttleSourceIndexedFoldKernel<<<kBlocks, kThreads, 0, stream>>>(
      initial, source_indices, contributions, output);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(
        std::string("ShuttleSourceIndexedFoldKernel: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleSourceIndexedFoldBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, {dimensions["contribution_rank"]}>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleSourceIndexedFold,
    ShuttleSourceIndexedFoldBinding());

extern "C" int shuttle_source_indexed_fold_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    if "atomicAdd(" in source:
        raise ValueError("source-indexed Fold generation introduced atomic accumulation")
    return GeneratedSourceIndexedFoldFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def replace_source_indexed_fold_with_custom_call(
    hlo_text: str,
    plan: SourceIndexedFoldTypedFfiPlan,
    *,
    target: str,
) -> str:
    """Replace one source-indexed Fold while preserving its result name."""
    pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(plan.instruction)} = .*?$", re.MULTILINE)
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one physical definition for source-indexed Fold %{plan.instruction}")
    match = matches[0]
    operands = (plan.initial, plan.source_indices, plan.contributions)
    rendered_operands = ", ".join(f"%{operand.instruction}" for operand in operands)
    constraints = ", ".join(operand.shape for operand in operands)
    replacement = (
        f"{match.group('indent')}%{plan.instruction} = {plan.output_shape} custom-call({rendered_operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    return hlo_text[: match.start()] + replacement + hlo_text[match.end() :]


def audit_source_indexed_fold_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: SourceIndexedFoldTypedFfiPlan,
    *,
    target: str,
) -> SourceIndexedFoldReplacementAudit:
    """Verify generated Fold ABI, ordering inputs, and consumer boundary."""
    original_module = parse_hlo_module_text(original_hlo)
    original_entry = original_module.computation(original_module.entry)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    original_users = _entry_users(original_entry)
    transformed_users = _entry_users(transformed_entry)
    call = _unique_target_instruction(transformed_entry, target)
    operands = (plan.initial.instruction, plan.source_indices.instruction, plan.contributions.instruction)
    if call.name != plan.instruction:
        raise ValueError(f"generated Fold moved from %{plan.instruction} to %{call.name}")
    if call.operands != operands:
        raise ValueError(f"generated Fold operands changed: {call.operands}")
    if call.shape != plan.output_shape:
        raise ValueError(f"generated Fold output changed: {call.shape}")
    if "api_version=API_VERSION_TYPED_FFI" not in call.attributes:
        raise ValueError("generated Fold does not use typed FFI API version 1")
    expected_constraints = (
        f"operand_layout_constraints={{{plan.initial.shape}, {plan.source_indices.shape}, "
        f"{plan.contributions.shape}}}"
    )
    if expected_constraints not in call.attributes:
        raise ValueError("generated Fold operand layout constraints changed")
    expected_users = original_users[plan.instruction]
    if expected_users != plan.external_users or transformed_users[call.name] != expected_users:
        raise ValueError("generated Fold consumer boundary changed")
    return SourceIndexedFoldReplacementAudit(
        call_instruction=call.name,
        operands=operands,
        output_shape=call.shape,
        external_users=expected_users,
        contribution_wrappers=plan.contribution_wrappers,
        api_version=plan.api_version,
    )


def evaluate_source_indexed_fold_plan(
    plan: SourceIndexedFoldTypedFfiPlan,
    initial: np.ndarray,
    source_indices: np.ndarray,
    contributions: np.ndarray,
) -> np.ndarray:
    """Evaluate the source-ordered Fold using an independent CPU schedule."""
    dimensions = _validated_dimensions(plan)
    initial = np.asarray(initial)
    source_indices = np.asarray(source_indices)
    contributions = np.asarray(contributions)
    _, planned_indices, _ = _parse_shape(plan.source_indices.shape)
    _, planned_contributions, _ = _parse_shape(plan.contributions.shape)
    if initial.shape != (dimensions["sources"], dimensions["features"]):
        raise ValueError("runtime initial state does not match the Fold plan")
    if source_indices.shape != planned_indices:
        raise ValueError("runtime source indices do not match the Fold plan")
    if contributions.shape != planned_contributions:
        raise ValueError("runtime contributions do not match the Fold plan")
    source_indices = source_indices.reshape(-1)
    contributions = contributions.reshape(dimensions["edges"], dimensions["features"])
    output = initial.astype(np.float32).copy()
    for source in range(dimensions["sources"]):
        for feature in range(dimensions["features"]):
            accumulator = float(output[source, feature])
            for edge in range(dimensions["edges"]):
                if int(source_indices[edge]) != source:
                    continue
                contribution = evaluate_cast_scalar_program(
                    plan.contribution_program,
                    {plan.contribution_input_name: float(contributions[edge, feature])},
                )
                accumulator = float(
                    evaluate_cast_scalar_program(
                        plan.reducer_program,
                        {
                            plan.reducer_accumulator_input_name: accumulator,
                            plan.reducer_contribution_input_name: float(contribution),
                        },
                    )
                )
            output[source, feature] = accumulator
    return output


def _validated_dimensions(plan: SourceIndexedFoldTypedFfiPlan) -> dict[str, int]:
    initial_dtype, initial, initial_layout = _parse_shape(plan.initial.shape)
    index_dtype, indices, index_layout = _parse_shape(plan.source_indices.shape)
    contribution_dtype, contributions, contribution_layout = _parse_shape(plan.contributions.shape)
    output_dtype, output, output_layout = _parse_shape(plan.output_shape)
    if initial_dtype != "bf16" or contribution_dtype != "bf16" or output_dtype != "bf16":
        raise ValueError("source-indexed Fold requires BF16 state and contributions")
    if index_dtype != "s32":
        raise ValueError("source-indexed Fold requires S32 source indices")
    if initial != output or len(output) != 2:
        raise ValueError("source-indexed Fold output must match its rank-two initial state")
    if len(indices) != 2 or indices[1] != 1:
        raise ValueError("source-indexed Fold requires one scalar index per edge")
    if not contributions or contributions[0] != indices[0]:
        raise ValueError("source-indexed Fold contribution and index edge counts disagree")
    if math.prod(contributions[1:]) != output[1]:
        raise ValueError("source-indexed Fold contribution payload does not match output features")
    for shape, layout in (
        (initial, initial_layout),
        (indices, index_layout),
        (contributions, contribution_layout),
        (output, output_layout),
    ):
        if layout != tuple(reversed(range(len(shape)))):
            raise ValueError("source-indexed Fold requires row-major physical arrays")
    if plan.numerical_contract.numerical_policy is not CastScalarNumericalPolicy.SOURCE_ORDERED:
        raise ValueError("source-indexed Fold must preserve source ordering")
    if not plan.numerical_contract.deterministic or plan.numerical_contract.atomic_accumulation:
        raise ValueError("source-indexed Fold must have one deterministic owner per output element")
    return {
        "sources": output[0],
        "edges": indices[0],
        "features": output[1],
        "contribution_rank": len(contributions),
    }


def _validate_source_indexed_scatter_attributes(instruction: HloInstruction) -> None:
    required = (
        "update_window_dims={1,2}",
        "inserted_window_dims={}",
        "scatter_dims_to_operand_dims={0}",
        "index_vector_dim=1",
    )
    missing = tuple(attribute for attribute in required if attribute not in instruction.attributes)
    if missing:
        raise ValueError(f"unsupported source-indexed Fold relation; missing {missing}")


def _wrapper_path(
    instructions: dict[str, HloInstruction],
    source: str,
    result: str,
) -> tuple[str, ...]:
    path: list[str] = []
    current = result
    while current != source:
        instruction = instructions[current]
        if instruction.opcode not in _WRAPPER_OPCODES or len(instruction.operands) != 1:
            raise ValueError(f"source-indexed Fold contribution %{result} is not a wrapper-only view of %{source}")
        path.append(current)
        current = instruction.operands[0]
    return tuple(reversed(path))


def _single_input_name(program: CastScalarProgram, role: str) -> str:
    if len(program.inputs) != 1 or program.inputs[0].input_name is None:
        raise ValueError(f"source-indexed Fold {role} requires one named scalar input")
    return program.inputs[0].input_name


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
        raise ValueError(f"expected one generated source-indexed Fold target {target!r}, found {len(matches)}")
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
