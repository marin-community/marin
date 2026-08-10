# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a bounded rank-two Contract followed by nested relation Folds."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from tile_lifetime.cast_scalar_program import CastScalarNumericalPolicy
from tile_lifetime.ffi_command_buffer import (
    DirectLaunchFfiPhysicalCandidate,
    direct_launch_status_check,
    finalize_ffi_handler_source,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import HloComputation, HloInstruction, parse_hlo_module_text
from tile_lifetime.xla_rank_two_contract_ffi import (
    RankTwoBf16ContractTypedFfiPlan,
    evaluate_rank_two_contract_plan,
)
from tile_lifetime.xla_weighted_relation_reverse_ffi import (
    RelationEdgeFoldTypedFfiPlan,
    evaluate_relation_edge_fold_plan,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")
_MAX_THREADS_PER_BLOCK = 1024
_MAX_STATIC_SHARED_BYTES = 48 * 1024


@dataclass(frozen=True)
class ContractRelationFoldPhysicalCost:
    """Static work and traffic counts for one bounded physical candidate."""

    contract_fma_count: int
    payload_elements: int
    payload_global_bytes: int
    kernel_launches: int
    threads_per_block: int
    shared_bytes: int


@dataclass(frozen=True)
class GeneratedContractRelationFoldFfi:
    """One generated Contract/Map/Fold CUDA kernel and its semantic identity."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str
    cost: ContractRelationFoldPhysicalCost
    physical_candidate: DirectLaunchFfiPhysicalCandidate

    @property
    def command_buffer_compatible(self) -> bool:
        """Whether the generated direct launch can be replayed."""
        return self.physical_candidate.command_buffer_compatible


@dataclass(frozen=True)
class ContractRelationFoldReplacementAudit:
    """Exact fused-call ownership and dead-intermediate evidence."""

    call_instruction: str
    operands: tuple[str, ...]
    output_shape: str
    dead_instructions: tuple[str, ...]
    external_users: tuple[str, ...]
    placement_wrappers: tuple[str, ...]
    placement_collective: str
    api_version: int


def generate_cuda_contract_relation_fold_ffi(
    contract: RankTwoBf16ContractTypedFfiPlan,
    fold: RelationEdgeFoldTypedFfiPlan,
    *,
    target: str,
    physical_candidate: DirectLaunchFfiPhysicalCandidate = DirectLaunchFfiPhysicalCandidate.LAUNCH_CHECKED,
) -> GeneratedContractRelationFoldFfi:
    """Generate one bounded CTA for Contract -> scalar Map -> two ordered Folds.

    This candidate retains the Contract's BF16 result boundary in shared memory.
    Each Contract output element has one thread-local FP32 reduction and one
    explicit round-to-nearest-even BF16 store before the generated scalar Map.
    The hidden-feature and source-edge Folds retain their recovered serial order.
    """
    dimensions = _validated_dimensions(contract, fold)
    target_symbol = _target_symbol(target)
    contribution_arguments = (
        (
            "shuttle_bf16_to_f32(payload[edge * kFeatures + feature])",
            "shuttle_bf16_to_f32(edge_cotangent[edge * kFeatures + feature])",
        )
        if fold.contribution_program.inputs[0].input_name == fold.payload_input_name
        else (
            "shuttle_bf16_to_f32(edge_cotangent[edge * kFeatures + feature])",
            "shuttle_bf16_to_f32(payload[edge * kFeatures + feature])",
        )
    )
    cost = ContractRelationFoldPhysicalCost(
        contract_fma_count=dimensions["edges"] * dimensions["features"] * dimensions["reduction"],
        payload_elements=dimensions["edges"] * dimensions["features"],
        payload_global_bytes=0,
        kernel_launches=1,
        threads_per_block=dimensions["threads"],
        shared_bytes=dimensions["shared_bytes"],
    )
    semantic_record = {
        "contract": {
            "lhs_shape": contract.lhs.shape,
            "rhs_shape": contract.rhs.shape,
            "output_shape": contract.output_shape,
            "lhs_row_start": contract.lhs_row_start,
            "numerical_contract": {
                "input_dtype": contract.numerical_contract.input_dtype,
                "accumulation_dtype": contract.numerical_contract.accumulation_dtype,
                "output_dtype": contract.numerical_contract.output_dtype,
                "output_rounding": contract.numerical_contract.output_rounding,
                "numerical_policy": contract.numerical_contract.numerical_policy.value,
            },
        },
        "fold": {
            "initial_shape": fold.initial.shape,
            "source_indices_shape": fold.source_indices.shape,
            "edge_cotangent_shape": fold.edge_cotangent.shape,
            "output_shape": fold.output_shape,
            "contribution": fold.contribution_program.digest,
            "inner_reducer": fold.inner_reducer_program.digest,
            "outer_reducer": fold.outer_reducer_program.digest,
            "numerical_policy": fold.numerical_contract.numerical_policy.value,
        },
        "physical": {
            "kind": "single_cta_contract_relation_fold",
            "threads": cost.threads_per_block,
            "shared_bytes": cost.shared_bytes,
        },
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    launch_status_check = direct_launch_status_check(
        physical_candidate,
        operation="Contract/relation Fold",
    )
    source_template = f"""// Generated from generic Contract/Map/Fold structure; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{fold.generated_contribution_cuda.source}
{fold.generated_inner_reducer_cuda.source}
{fold.generated_outer_reducer_cuda.source}

namespace ffi = xla::ffi;

namespace {{
constexpr int kRowStart = {contract.lhs_row_start};
constexpr int kSources = {dimensions["sources"]};
constexpr int kEdges = {dimensions["edges"]};
constexpr int kReduction = {dimensions["reduction"]};
constexpr int kFeatures = {dimensions["features"]};
constexpr int kThreads = {dimensions["threads"]};
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

__global__ void ShuttleContractRelationFoldKernel(
    const std::uint16_t* lhs,
    const std::uint16_t* rhs,
    const std::uint16_t* initial,
    const std::int32_t* source_indices,
    const std::uint16_t* edge_cotangent,
    std::uint16_t* output) {{
  __shared__ std::uint16_t payload[kEdges * kFeatures];
  __shared__ float edge_state[kEdges];

  const int item = threadIdx.x;
  if (item < kEdges * kFeatures) {{
    const int edge = item / kFeatures;
    const int feature = item - edge * kFeatures;
    float accumulator = 0.0f;
    const int lhs_row = kRowStart + edge;
    for (int reduction = 0; reduction < kReduction; ++reduction) {{
      accumulator = __fmaf_rn(
          shuttle_bf16_to_f32(lhs[lhs_row * kReduction + reduction]),
          shuttle_bf16_to_f32(rhs[reduction * kFeatures + feature]),
          accumulator);
    }}
    payload[item] = shuttle_f32_to_bf16(accumulator);
  }}
  __syncthreads();

  if (item < kEdges) {{
    const int edge = item;
    float inner = 0.0f;
    for (int feature = 0; feature < kFeatures; ++feature) {{
      const float contribution = {fold.generated_contribution_cuda.symbol}(
          {contribution_arguments[0]},
          {contribution_arguments[1]});
      inner = {fold.generated_inner_reducer_cuda.symbol}(inner, contribution);
    }}
    edge_state[edge] = inner;
  }}
  __syncthreads();

  for (int source = item; source < kSources; source += blockDim.x) {{
    float outer = shuttle_bf16_to_f32(initial[source]);
    for (int edge = 0; edge < kEdges; ++edge) {{
      if (source_indices[edge] == source) {{
        outer = {fold.generated_outer_reducer_cuda.symbol}(outer, edge_state[edge]);
      }}
    }}
    output[source] = shuttle_f32_to_bf16(outer);
  }}
}}

ffi::Error ShuttleContractRelationFold(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> lhs_buffer,
    ffi::Buffer<ffi::BF16, 2> rhs_buffer,
    ffi::Buffer<ffi::BF16, 1> initial_buffer,
    ffi::Buffer<ffi::S32, 2> source_indices_buffer,
    ffi::Buffer<ffi::BF16, 2> edge_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 1>> output_buffer) {{
  const auto* lhs = reinterpret_cast<const std::uint16_t*>(lhs_buffer.typed_data());
  const auto* rhs = reinterpret_cast<const std::uint16_t*>(rhs_buffer.typed_data());
  const auto* initial = reinterpret_cast<const std::uint16_t*>(initial_buffer.typed_data());
  const auto* source_indices = source_indices_buffer.typed_data();
  const auto* edge_cotangent =
      reinterpret_cast<const std::uint16_t*>(edge_cotangent_buffer.typed_data());
  auto* output = reinterpret_cast<std::uint16_t*>(output_buffer->typed_data());
  ShuttleContractRelationFoldKernel<<<1, kThreads, 0, stream>>>(
      lhs, rhs, initial, source_indices, edge_cotangent, output);
{launch_status_check}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleContractRelationFoldBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 1>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleContractRelationFold,
    ShuttleContractRelationFoldBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);

extern "C" int shuttle_contract_relation_fold_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    source = finalize_ffi_handler_source(
        source_template,
        command_buffer_compatible=physical_candidate.command_buffer_compatible,
    )
    if "atomicAdd(" in source:
        raise ValueError("Contract/relation-Fold generation introduced atomic accumulation")
    return GeneratedContractRelationFoldFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
        cost=cost,
        physical_candidate=physical_candidate,
    )


def evaluate_contract_relation_fold_plan(
    contract: RankTwoBf16ContractTypedFfiPlan,
    fold: RelationEdgeFoldTypedFfiPlan,
    lhs: np.ndarray,
    rhs: np.ndarray,
    initial: np.ndarray,
    source_indices: np.ndarray,
    edge_cotangent: np.ndarray,
) -> np.ndarray:
    """Evaluate the fused boundary through the independent generic stage evaluators."""
    _validated_dimensions(contract, fold)
    payload = evaluate_rank_two_contract_plan(contract, lhs, rhs)
    return evaluate_relation_edge_fold_plan(fold, initial, source_indices, payload, edge_cotangent)


def replace_contract_relation_fold_with_custom_call(
    hlo_text: str,
    contract: RankTwoBf16ContractTypedFfiPlan,
    fold: RelationEdgeFoldTypedFfiPlan,
    *,
    target: str,
) -> str:
    """Replace a generic Contract/Map/Fold region with one generated call."""
    _validated_dimensions(contract, fold)
    pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(fold.instruction)} = .*?$", re.MULTILINE)
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one physical definition for relation Fold %{fold.instruction}")
    match = matches[0]
    operands = (contract.lhs, contract.rhs, fold.initial, fold.source_indices, fold.edge_cotangent)
    rendered_operands = ", ".join(f"%{operand.instruction}" for operand in operands)
    constraints = ", ".join(operand.shape for operand in operands)
    replacement = (
        f"{match.group('indent')}%{fold.instruction} = {fold.output_shape} custom-call({rendered_operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    rewritten = hlo_text[: match.start()] + replacement + hlo_text[match.end() :]
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_contract_relation_fold_replacement(
    original_hlo: str,
    transformed_hlo: str,
    contract: RankTwoBf16ContractTypedFfiPlan,
    fold: RelationEdgeFoldTypedFfiPlan,
    *,
    target: str,
) -> ContractRelationFoldReplacementAudit:
    """Verify one generated call owns the live Contract/Map/Fold region."""
    _validated_dimensions(contract, fold)
    original_module = parse_hlo_module_text(original_hlo)
    original_entry = original_module.computation(original_module.entry)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    users = _entry_users(transformed_entry)
    call = _unique_target_instruction(transformed_entry, target)
    operands = (
        contract.lhs.instruction,
        contract.rhs.instruction,
        fold.initial.instruction,
        fold.source_indices.instruction,
        fold.edge_cotangent.instruction,
    )
    if call.name != fold.instruction or call.operands != operands:
        raise ValueError("generated Contract/relation-Fold call changed its physical binding")
    if call.shape != fold.output_shape or "api_version=API_VERSION_TYPED_FFI" not in call.attributes:
        raise ValueError("generated Contract/relation-Fold call changed its result ABI")
    dead = tuple(dict.fromkeys((contract.instruction, contract.source_instruction, *fold.internal_instructions)))
    live = _live_entry_instructions(transformed_entry)
    still_live = tuple(instruction for instruction in dead if instruction in live)
    if still_live:
        raise ValueError(f"fused Contract/relation-Fold intermediates remain live: {still_live}")
    original_users = _entry_users(original_entry)
    if users[call.name] != fold.external_users or original_users[fold.instruction] != fold.external_users:
        raise ValueError("generated Contract/relation-Fold consumer boundary changed")
    instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    placement_wrappers, placement_collective = _placement_collective_path(instructions, users, call.name)
    return ContractRelationFoldReplacementAudit(
        call_instruction=call.name,
        operands=operands,
        output_shape=call.shape,
        dead_instructions=dead,
        external_users=users[call.name],
        placement_wrappers=placement_wrappers,
        placement_collective=placement_collective,
        api_version=1,
    )


def _validated_dimensions(
    contract: RankTwoBf16ContractTypedFfiPlan,
    fold: RelationEdgeFoldTypedFfiPlan,
) -> dict[str, int]:
    lhs = _required_shape(contract.lhs.shape, dtype="bf16", rank=2)
    rhs = _required_shape(contract.rhs.shape, dtype="bf16", rank=2)
    payload = _required_shape(contract.output_shape, dtype="bf16", rank=2)
    initial = _required_shape(fold.initial.shape, dtype="bf16", rank=1)
    indices = _required_shape(fold.source_indices.shape, dtype="s32", rank=2)
    cotangent = _required_shape(fold.edge_cotangent.shape, dtype="bf16", rank=2)
    output = _required_shape(fold.output_shape, dtype="bf16", rank=1)
    if contract.instruction != fold.payload.instruction or contract.output_shape != fold.payload.shape:
        raise ValueError("Contract output is not the physical payload consumed by the relation Fold")
    if contract.dimensions.lhs_contracting != (1,) or contract.dimensions.rhs_contracting != (0,):
        raise ValueError("bounded Contract/relation-Fold requires a rank-two K contraction")
    if contract.dimensions.lhs_batch or contract.dimensions.rhs_batch:
        raise ValueError("bounded Contract/relation-Fold does not support batch dimensions")
    if lhs[1] != rhs[0] or payload[1] != rhs[1]:
        raise ValueError("Contract shapes disagree with its K relation")
    if contract.lhs_row_start < 0 or contract.lhs_row_start + payload[0] > lhs[0]:
        raise ValueError("Contract row domain is outside its LHS")
    if payload != cotangent or indices != (payload[0], 1):
        raise ValueError("one Contract row, cotangent row, and source index are required per relation edge")
    if initial != output:
        raise ValueError("outer Fold initial and output domains disagree")
    if fold.payload_logical_shape != contract.output_shape:
        raise ValueError("bounded fusion requires an exact physical/logical payload row domain")
    if contract.numerical_contract.accumulation_dtype != "f32":
        raise ValueError("bounded Contract/relation-Fold requires FP32 Contract accumulation")
    if contract.numerical_contract.output_dtype != "bf16" or contract.numerical_contract.output_rounding != (
        "round_to_nearest_even"
    ):
        raise ValueError("bounded fusion requires an explicit BF16 Contract result boundary")
    if contract.numerical_contract.numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("bounded fusion cannot preserve an unspecified Contract reduction tree")
    if not contract.numerical_contract.deterministic_accumulation:
        raise ValueError("bounded fusion requires deterministic per-output Contract ownership")
    if not fold.numerical_contract.deterministic or fold.numerical_contract.atomic_accumulation:
        raise ValueError("bounded fusion requires deterministic atomic-free Fold ownership")
    for program in (fold.contribution_program, fold.inner_reducer_program, fold.outer_reducer_program):
        if program.numerical_policy is not CastScalarNumericalPolicy.SOURCE_ORDERED:
            raise ValueError("bounded fusion requires source-ordered scalar Map/Fold bodies")
    contribution_inputs = tuple(value.input_name for value in fold.contribution_program.inputs)
    if len(contribution_inputs) != 2 or set(contribution_inputs) != {
        fold.payload_input_name,
        fold.edge_cotangent_input_name,
    }:
        raise ValueError("bounded fusion requires one same-element payload and cotangent Map input")
    if any(
        value.input_index is None or value.input_index.row_offset != 0 or value.input_index.feature_offset != 0
        for value in fold.contribution_program.inputs
    ):
        raise ValueError("bounded fusion requires same-element scalar Map index relations")
    if tuple(value.input_name for value in fold.inner_reducer_program.inputs) != (
        fold.inner_accumulator_input_name,
        fold.inner_contribution_input_name,
    ):
        raise ValueError("bounded fusion requires an ordered binary hidden Fold")
    if tuple(value.input_name for value in fold.outer_reducer_program.inputs) != (
        fold.outer_accumulator_input_name,
        fold.outer_contribution_input_name,
    ):
        raise ValueError("bounded fusion requires an ordered binary source Fold")
    work_items = payload[0] * payload[1]
    if work_items == 0 or output[0] == 0:
        raise ValueError("bounded Contract/relation-Fold requires nonempty static domains")
    threads = ((max(work_items, payload[0], output[0]) + 31) // 32) * 32
    shared_bytes = work_items * 2 + payload[0] * 4
    if threads > _MAX_THREADS_PER_BLOCK:
        raise ValueError(
            f"bounded Contract/relation-Fold requires {threads} threads, exceeding {_MAX_THREADS_PER_BLOCK}"
        )
    if shared_bytes > _MAX_STATIC_SHARED_BYTES:
        raise ValueError(
            f"bounded Contract/relation-Fold requires {shared_bytes} shared bytes, exceeding "
            f"{_MAX_STATIC_SHARED_BYTES}"
        )
    return {
        "input_rows": lhs[0],
        "sources": output[0],
        "edges": payload[0],
        "reduction": lhs[1],
        "features": payload[1],
        "threads": threads,
        "shared_bytes": shared_bytes,
    }


def _required_shape(shape: str, *, dtype: str, rank: int) -> tuple[int, ...]:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None or match.group("dtype") != dtype:
        raise ValueError(f"expected {dtype} rank-{rank} row-major array, found {shape}")
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    layout = tuple(int(value) for value in match.group("layout").split(",") if value)
    if len(dimensions) != rank or layout != tuple(reversed(range(rank))):
        raise ValueError(f"expected {dtype} rank-{rank} row-major array, found {shape}")
    return dimensions


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    users: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)
    return {instruction: tuple(values) for instruction, values in users.items()}


def _live_entry_instructions(entry: HloComputation) -> frozenset[str]:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    pending = [entry.root.name]
    live: set[str] = set()
    while pending:
        current = pending.pop()
        if current in live:
            continue
        live.add(current)
        pending.extend(instructions[current].operands)
    return frozenset(live)


def _unique_target_instruction(entry: HloComputation, target: str) -> HloInstruction:
    attribute = f'custom_call_target="{target}"'
    matches = tuple(
        instruction
        for instruction in entry.instructions
        if instruction.opcode == "custom-call" and attribute in instruction.attributes
    )
    if len(matches) != 1:
        raise ValueError(f"expected one generated target {target!r}, found {len(matches)}")
    return matches[0]


def _placement_collective_path(
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    start: str,
) -> tuple[tuple[str, ...], str]:
    wrappers: list[str] = []
    current = start
    while True:
        direct_users = users[current]
        if len(direct_users) != 1:
            raise ValueError(f"relation Fold output %{current} has {len(direct_users)} users")
        user = instructions[direct_users[0]]
        if user.opcode == "all-reduce":
            return tuple(wrappers), user.name
        if user.opcode not in {"bitcast", "copy", "reshape", "slice", "transpose"} or len(user.operands) != 1:
            raise ValueError(f"relation Fold output does not reach a placement collective through views: %{user.name}")
        wrappers.append(user.name)
        current = user.name


def _target_symbol(target: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", target)
    if not symbol or symbol[0].isdigit():
        symbol = f"shuttle_{symbol}"
    return symbol
