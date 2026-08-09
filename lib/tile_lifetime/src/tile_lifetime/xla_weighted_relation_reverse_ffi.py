# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a weighted RelationProgram reverse from generic Contract/Fold structure."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.cast_scalar_program import (
    CastScalarNumericalPolicy,
    CastScalarProgram,
    GeneratedCudaScalarBody,
    evaluate_cast_scalar_program,
    generate_cuda_scalar_body,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import (
    EntryRegionValue,
    HloComputation,
    HloInstruction,
    HloModuleGraph,
    InlinedHloGraph,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)
from tile_lifetime.xla_rank_two_contract_ffi import (
    RankTwoBf16ContractTypedFfiPlan,
    evaluate_rank_two_contract_plan,
    narrow_rank_two_contract_to_consumer_row_domain,
    plan_rank_two_bf16_contract_typed_ffi,
    replace_rank_two_contract_with_custom_call,
)
from tile_lifetime.xla_relation_program_recovery import (
    ContractDimensionMap,
    RelationPlanRecord,
    RoutedForwardContractStage,
    recover_relation_programs,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_computation, import_hlo_scalar_map

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")
_CALLED_COMPUTATION = re.compile(r"to_apply=%(?P<name>[A-Za-z0-9_.-]+)")
_DIMENSIONS = re.compile(r"dimensions=\{(?P<axes>[0-9,]*)\}")
_DOT_DIMENSIONS = re.compile(
    r"(?P<name>lhs_contracting_dims|rhs_contracting_dims|lhs_batch_dims|rhs_batch_dims)=\{(?P<dims>[0-9,]*)\}"
)
_SCALAR_MAP_OPCODES = frozenset({"add", "convert", "divide", "multiply", "negate", "subtract"})
_PAYLOAD_WRAPPER_OPCODES = frozenset({"bitcast", "copy", "reshape", "slice"})
_VIEW_OPCODES = frozenset({"bitcast", "copy", "reshape", "slice", "transpose"})


class RelationPayloadPolicy(StrEnum):
    """How the value paired with an edge cotangent becomes available."""

    RECOMPUTE_CONTRACT = "recompute_contract"
    SAVED_VALUE = "saved_value"


@dataclass(frozen=True)
class RelationEdgeFoldNumericalContract:
    """Finite-precision promises for the nested edge and source Folds."""

    input_dtype: str
    map_dtype: str
    inner_state_dtype: str
    outer_state_dtype: str
    output_dtype: str
    numerical_policy: NumericalPolicy
    deterministic: bool
    atomic_accumulation: bool


@dataclass(frozen=True)
class RelationEdgeFoldTypedFfiPlan:
    """A scalar Map, hidden Fold, and deterministic source-indexed Fold."""

    instruction: str
    initial: EntryRegionValue
    source_indices: EntryRegionValue
    payload: EntryRegionValue
    edge_cotangent: EntryRegionValue
    payload_logical_shape: str
    payload_wrappers: tuple[str, ...]
    internal_instructions: tuple[str, ...]
    output_shape: str
    contribution_program: CastScalarProgram
    generated_contribution_cuda: GeneratedCudaScalarBody
    inner_reducer_program: CastScalarProgram
    generated_inner_reducer_cuda: GeneratedCudaScalarBody
    outer_reducer_program: CastScalarProgram
    generated_outer_reducer_cuda: GeneratedCudaScalarBody
    payload_input_name: str
    edge_cotangent_input_name: str
    inner_accumulator_input_name: str
    inner_contribution_input_name: str
    outer_accumulator_input_name: str
    outer_contribution_input_name: str
    external_users: tuple[str, ...]
    api_version: int
    numerical_contract: RelationEdgeFoldNumericalContract


@dataclass(frozen=True)
class WeightedRelationReverseTypedFfiPlan:
    """One generic weighted-relation reverse through a recomputed payload."""

    relation_plan: RelationPlanRecord
    payload_policy: RelationPayloadPolicy
    legal_payload_policies: tuple[RelationPayloadPolicy, ...]
    payload_contract: RankTwoBf16ContractTypedFfiPlan
    edge_fold: RelationEdgeFoldTypedFfiPlan


@dataclass(frozen=True)
class GeneratedRelationEdgeFoldFfi:
    """CUDA source and semantic identity for a generated nested Fold."""

    target: str
    source: str
    semantic_digest: str
    source_digest: str


@dataclass(frozen=True)
class WeightedRelationReverseReplacementAudit:
    """Exact post-replacement ownership and remaining placement boundary."""

    contract_instruction: str
    fold_instruction: str
    contract_operands: tuple[str, str]
    fold_operands: tuple[str, str, str, str]
    dead_replaced_instructions: tuple[str, ...]
    output_users: tuple[str, ...]
    placement_wrappers: tuple[str, ...]
    placement_collective: str
    api_version: int


def plan_weighted_relation_reverse_typed_ffi(
    hlo_text: str,
    *,
    numerical_policy: NumericalPolicy = NumericalPolicy.ALLOW_ROUNDING_REORDER,
) -> WeightedRelationReverseTypedFfiPlan:
    """Recover Contract -> scalar Map -> hidden Fold -> source Fold by structure."""
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("weighted relation reverse cannot preserve unspecified dot and Fold reduction trees")
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry)
    graph = inline_elementwise_fusions(module)
    relation_plans = recover_relation_programs(hlo_text).relation_plans
    candidates: list[WeightedRelationReverseTypedFfiPlan] = []
    for scatter in entry.instructions:
        try:
            recovered = _recover_candidate(
                hlo_text,
                module.entry,
                module,
                instructions,
                users,
                graph,
                relation_plans,
                scatter,
                numerical_policy=numerical_policy,
            )
        except _NotWeightedRelationReverse:
            continue
        candidates.append(recovered)
    if len(candidates) != 1:
        raise ValueError(f"expected one weighted relation reverse, found {len(candidates)}")
    return candidates[0]


def generate_cuda_relation_edge_fold_ffi(
    plan: RelationEdgeFoldTypedFfiPlan,
    *,
    target: str,
) -> GeneratedRelationEdgeFoldFfi:
    """Generate deterministic nested Folds with one thread per source slot."""
    dimensions = _validated_fold_dimensions(plan)
    semantic_record = {
        "initial_shape": plan.initial.shape,
        "source_indices_shape": plan.source_indices.shape,
        "payload_shape": plan.payload.shape,
        "payload_logical_shape": plan.payload_logical_shape,
        "edge_cotangent_shape": plan.edge_cotangent.shape,
        "output_shape": plan.output_shape,
        "contribution": plan.contribution_program.digest,
        "inner_reducer": plan.inner_reducer_program.digest,
        "outer_reducer": plan.outer_reducer_program.digest,
        "numerical_contract": {
            "input_dtype": plan.numerical_contract.input_dtype,
            "map_dtype": plan.numerical_contract.map_dtype,
            "inner_state_dtype": plan.numerical_contract.inner_state_dtype,
            "outer_state_dtype": plan.numerical_contract.outer_state_dtype,
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
    contribution_arguments = (
        (
            "shuttle_bf16_to_f32(payload[edge * kFeatures + feature])",
            "shuttle_bf16_to_f32(edge_cotangent[edge * kFeatures + feature])",
        )
        if plan.contribution_program.inputs[0].input_name == plan.payload_input_name
        else (
            "shuttle_bf16_to_f32(edge_cotangent[edge * kFeatures + feature])",
            "shuttle_bf16_to_f32(payload[edge * kFeatures + feature])",
        )
    )
    source = f"""// Generated from generic nested relation Folds; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

{plan.generated_contribution_cuda.source}
{plan.generated_inner_reducer_cuda.source}
{plan.generated_outer_reducer_cuda.source}

namespace ffi = xla::ffi;

namespace {{
constexpr int kSources = {dimensions["sources"]};
constexpr int kEdges = {dimensions["edges"]};
constexpr int kPhysicalRows = {dimensions["physical_rows"]};
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

__global__ void ShuttleRelationEdgeFoldKernel(
    const std::uint16_t* initial,
    const std::int32_t* source_indices,
    const std::uint16_t* payload,
    const std::uint16_t* edge_cotangent,
    std::uint16_t* output) {{
  const int source = blockIdx.x * blockDim.x + threadIdx.x;
  if (source >= kSources) {{
    return;
  }}
  float outer = shuttle_bf16_to_f32(initial[source]);
  for (int edge = 0; edge < kEdges; ++edge) {{
    if (source_indices[edge] != source) {{
      continue;
    }}
    float inner = 0.0f;
    for (int feature = 0; feature < kFeatures; ++feature) {{
      const float contribution = {plan.generated_contribution_cuda.symbol}(
          {contribution_arguments[0]},
          {contribution_arguments[1]});
      inner = {plan.generated_inner_reducer_cuda.symbol}(inner, contribution);
    }}
    outer = {plan.generated_outer_reducer_cuda.symbol}(outer, inner);
  }}
  output[source] = shuttle_f32_to_bf16(outer);
}}

ffi::Error ShuttleRelationEdgeFold(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 1> initial_buffer,
    ffi::Buffer<ffi::S32, 2> source_indices_buffer,
    ffi::Buffer<ffi::BF16, 2> payload_buffer,
    ffi::Buffer<ffi::BF16, 2> edge_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 1>> output_buffer) {{
  const auto* initial = reinterpret_cast<const std::uint16_t*>(initial_buffer.typed_data());
  const auto* source_indices = source_indices_buffer.typed_data();
  const auto* payload = reinterpret_cast<const std::uint16_t*>(payload_buffer.typed_data());
  const auto* edge_cotangent =
      reinterpret_cast<const std::uint16_t*>(edge_cotangent_buffer.typed_data());
  auto* output = reinterpret_cast<std::uint16_t*>(output_buffer->typed_data());
  constexpr int kThreads = 256;
  constexpr int kBlocks = (kSources + kThreads - 1) / kThreads;
  ShuttleRelationEdgeFoldKernel<<<kBlocks, kThreads, 0, stream>>>(
      initial, source_indices, payload, edge_cotangent, output);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal(
        std::string("ShuttleRelationEdgeFoldKernel: ") + cudaGetErrorString(status));
  }}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleRelationEdgeFoldBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 1>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {target_symbol},
    ShuttleRelationEdgeFold,
    ShuttleRelationEdgeFoldBinding());

extern "C" int shuttle_relation_edge_fold_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
"""
    if "atomicAdd(" in source:
        raise ValueError("relation edge Fold generation introduced atomic accumulation")
    return GeneratedRelationEdgeFoldFfi(
        target=target,
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
    )


def replace_weighted_relation_reverse_with_custom_calls(
    hlo_text: str,
    plan: WeightedRelationReverseTypedFfiPlan,
    *,
    contract_target: str,
    fold_target: str,
) -> str:
    """Replace the recompute Contract and nested Folds with generic calls."""
    rewritten = replace_rank_two_contract_with_custom_call(
        hlo_text,
        plan.payload_contract,
        target=contract_target,
    )
    fold = plan.edge_fold
    pattern = re.compile(rf"^(?P<indent>\s*)%{re.escape(fold.instruction)} = .*?$", re.MULTILINE)
    matches = tuple(pattern.finditer(rewritten))
    if len(matches) != 1:
        raise ValueError(f"expected one physical definition for relation Fold %{fold.instruction}")
    match = matches[0]
    operands = (fold.initial, fold.source_indices, fold.payload, fold.edge_cotangent)
    rendered_operands = ", ".join(f"%{operand.instruction}" for operand in operands)
    constraints = ", ".join(operand.shape for operand in operands)
    replacement = (
        f"{match.group('indent')}%{fold.instruction} = {fold.output_shape} custom-call({rendered_operands}), "
        f'custom_call_target="{fold_target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    rewritten = rewritten[: match.start()] + replacement + rewritten[match.end() :]
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_weighted_relation_reverse_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: WeightedRelationReverseTypedFfiPlan,
    *,
    contract_target: str,
    fold_target: str,
) -> WeightedRelationReverseReplacementAudit:
    """Verify exact generated ownership and the retained collective boundary."""
    transformed_module = parse_hlo_module_text(transformed_hlo)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    users = _entry_users(transformed_entry)
    contract_call = _unique_target_instruction(transformed_entry, contract_target)
    fold_call = _unique_target_instruction(transformed_entry, fold_target)
    contract = plan.payload_contract
    fold = plan.edge_fold
    if contract_call.name != contract.instruction or contract_call.operands != (
        contract.lhs.instruction,
        contract.rhs.instruction,
    ):
        raise ValueError("generated payload Contract changed its physical binding")
    expected_fold_operands = (
        fold.initial.instruction,
        fold.source_indices.instruction,
        fold.payload.instruction,
        fold.edge_cotangent.instruction,
    )
    if fold_call.name != fold.instruction or fold_call.operands != expected_fold_operands:
        raise ValueError("generated nested Fold changed its physical binding")
    if "api_version=API_VERSION_TYPED_FFI" not in contract_call.attributes or (
        "api_version=API_VERSION_TYPED_FFI" not in fold_call.attributes
    ):
        raise ValueError("weighted relation reverse does not use typed FFI API version 1")
    live = _live_entry_instructions(transformed_entry)
    still_live = tuple(instruction for instruction in fold.internal_instructions if instruction in live)
    if still_live:
        raise ValueError(f"replaced relation reverse arithmetic remains live: {still_live}")
    output_users = users[fold_call.name]
    if output_users != fold.external_users:
        raise ValueError("generated nested Fold consumer boundary changed")
    placement_wrappers, placement_collective = _placement_collective_path(
        instructions,
        users,
        fold_call.name,
    )
    return WeightedRelationReverseReplacementAudit(
        contract_instruction=contract_call.name,
        fold_instruction=fold_call.name,
        contract_operands=contract_call.operands,
        fold_operands=fold_call.operands,
        dead_replaced_instructions=fold.internal_instructions,
        output_users=output_users,
        placement_wrappers=placement_wrappers,
        placement_collective=placement_collective,
        api_version=fold.api_version,
    )


def evaluate_weighted_relation_reverse_plan(
    plan: WeightedRelationReverseTypedFfiPlan,
    lhs: np.ndarray,
    rhs: np.ndarray,
    initial: np.ndarray,
    source_indices: np.ndarray,
    edge_cotangent: np.ndarray,
) -> np.ndarray:
    """Evaluate the full recompute and nested Fold boundary on CPU."""
    payload = evaluate_rank_two_contract_plan(plan.payload_contract, lhs, rhs)
    return evaluate_relation_edge_fold_plan(
        plan.edge_fold,
        initial,
        source_indices,
        payload,
        edge_cotangent,
    )


def evaluate_relation_edge_fold_plan(
    plan: RelationEdgeFoldTypedFfiPlan,
    initial: np.ndarray,
    source_indices: np.ndarray,
    payload: np.ndarray,
    edge_cotangent: np.ndarray,
) -> np.ndarray:
    """Evaluate nested source-ordered Folds using the generated scalar ASTs."""
    dimensions = _validated_fold_dimensions(plan)
    initial = np.asarray(initial)
    source_indices = np.asarray(source_indices)
    payload = np.asarray(payload)
    edge_cotangent = np.asarray(edge_cotangent)
    if initial.shape != (dimensions["sources"],):
        raise ValueError("runtime initial state does not match the nested Fold plan")
    if source_indices.shape != (dimensions["edges"], 1):
        raise ValueError("runtime source indices do not match the nested Fold plan")
    if payload.shape != (dimensions["physical_rows"], dimensions["features"]):
        raise ValueError("runtime payload does not match the nested Fold plan")
    if edge_cotangent.shape != (dimensions["edges"], dimensions["features"]):
        raise ValueError("runtime edge cotangent does not match the nested Fold plan")
    output = initial.astype(np.float32).copy()
    source_indices = source_indices.reshape(-1)
    for source in range(dimensions["sources"]):
        outer = float(output[source])
        for edge in range(dimensions["edges"]):
            if int(source_indices[edge]) != source:
                continue
            inner = 0.0
            for feature in range(dimensions["features"]):
                contribution = evaluate_cast_scalar_program(
                    plan.contribution_program,
                    {
                        plan.payload_input_name: float(payload[edge, feature]),
                        plan.edge_cotangent_input_name: float(edge_cotangent[edge, feature]),
                    },
                )
                inner = float(
                    evaluate_cast_scalar_program(
                        plan.inner_reducer_program,
                        {
                            plan.inner_accumulator_input_name: inner,
                            plan.inner_contribution_input_name: float(contribution),
                        },
                    )
                )
            outer = float(
                evaluate_cast_scalar_program(
                    plan.outer_reducer_program,
                    {
                        plan.outer_accumulator_input_name: outer,
                        plan.outer_contribution_input_name: inner,
                    },
                )
            )
        output[source] = outer
    return output


class _NotWeightedRelationReverse(ValueError):
    pass


def _recover_candidate(
    hlo_text: str,
    entry_name: str,
    module: HloModuleGraph,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    graph: InlinedHloGraph,
    relation_plans: tuple[RelationPlanRecord, ...],
    scatter: HloInstruction,
    *,
    numerical_policy: NumericalPolicy,
) -> WeightedRelationReverseTypedFfiPlan:
    if scatter.opcode != "scatter" or len(scatter.operands) != 3:
        raise _NotWeightedRelationReverse
    output = _parse_shape(scatter.shape)
    if output is None or output[0] != "bf16" or len(output[1]) != 1:
        raise _NotWeightedRelationReverse
    required_scatter_attributes = (
        "update_window_dims={1}",
        "inserted_window_dims={}",
        "scatter_dims_to_operand_dims={0}",
        "index_vector_dim=1",
    )
    if any(attribute not in scatter.attributes for attribute in required_scatter_attributes):
        raise _NotWeightedRelationReverse
    update_wrapper = instructions[scatter.operands[2]]
    if update_wrapper.opcode not in _PAYLOAD_WRAPPER_OPCODES or len(update_wrapper.operands) != 1:
        raise _NotWeightedRelationReverse
    reduction = instructions[update_wrapper.operands[0]]
    if reduction.opcode != "reduce" or len(reduction.operands) != 2 or _reduction_axes(reduction) != (1,):
        raise _NotWeightedRelationReverse
    reduction_input = instructions[reduction.operands[0]]
    reduction_input_shape = _parse_shape(reduction_input.shape)
    if (
        reduction_input.opcode not in _SCALAR_MAP_OPCODES
        or reduction_input_shape is None
        or reduction_input_shape[0] != "bf16"
        or len(reduction_input_shape[1]) != 2
    ):
        raise _NotWeightedRelationReverse
    map_sources = _scalar_map_rank_two_sources(instructions, reduction_input)
    payload_candidates: list[tuple[str, str, tuple[str, ...]]] = []
    for source in map_sources:
        contract_path = _payload_contract_path(instructions, source)
        if contract_path is not None:
            payload_candidates.append((source, contract_path[0], contract_path[1]))
    if len(payload_candidates) != 1:
        raise _NotWeightedRelationReverse
    payload_source, payload_contract_instruction, payload_wrappers = payload_candidates[0]
    edge_cotangent_name = next((source for source in map_sources if source != payload_source), None)
    if edge_cotangent_name is None:
        raise _NotWeightedRelationReverse
    edge_cotangent = instructions[edge_cotangent_name]
    cotangent_shape = _parse_shape(edge_cotangent.shape)
    payload_logical_shape = _parse_shape(instructions[payload_source].shape)
    if cotangent_shape != payload_logical_shape or cotangent_shape != reduction_input_shape:
        raise _NotWeightedRelationReverse
    initial = instructions[scatter.operands[0]]
    source_indices = instructions[scatter.operands[1]]
    initial_shape = _parse_shape(initial.shape)
    index_shape = _parse_shape(source_indices.shape)
    if initial_shape != output or index_shape is None or index_shape[:1] != ("s32",):
        raise _NotWeightedRelationReverse
    if index_shape[1] != (reduction_input_shape[1][0], 1):
        raise _NotWeightedRelationReverse
    source_index_node = graph.entry_value(source_indices.name)
    relation_distances = tuple(
        (distance, relation)
        for relation in relation_plans
        if relation.edge_count == index_shape[1][0]
        and (distance := _graph_ancestor_distance(graph, relation.stable_permutation, source_index_node)) is not None
    )
    if not relation_distances:
        raise _NotWeightedRelationReverse
    minimum_distance = min(distance for distance, _ in relation_distances)
    matching_relations = tuple(relation for distance, relation in relation_distances if distance == minimum_distance)
    if len(matching_relations) != 1:
        raise _NotWeightedRelationReverse
    if not _is_zero_scalar(instructions[reduction.operands[1]]):
        raise _NotWeightedRelationReverse
    inner_reducer = _reducer_program(module, reduction)
    outer_reducer = _reducer_program(module, scatter)
    if tuple(value.input_name for value in inner_reducer.inputs) != ("input0", "input1"):
        raise _NotWeightedRelationReverse
    if tuple(value.input_name for value in outer_reducer.inputs) != ("input0", "input1"):
        raise _NotWeightedRelationReverse
    contribution = import_hlo_scalar_map(
        graph,
        source_nodes=tuple(graph.entry_value(source) for source in map_sources),
        target_node=graph.entry_value(reduction_input.name),
    )
    contribution_inputs = tuple(value.input_name for value in contribution.inputs)
    if len(contribution_inputs) != 2 or any(name is None for name in contribution_inputs):
        raise _NotWeightedRelationReverse
    payload_index = map_sources.index(payload_source)
    contract = instructions[payload_contract_instruction]
    contract_stage = RoutedForwardContractStage(
        node=f"{entry_name}/{contract.name}",
        lhs=f"{entry_name}/{contract.operands[0]}",
        rhs=f"{entry_name}/{contract.operands[1]}",
        output_shape=contract.shape,
        dimensions=_contract_dimensions(contract),
    )
    full_contract_plan = plan_rank_two_bf16_contract_typed_ffi(
        hlo_text,
        contract_stage,
        numerical_policy=numerical_policy,
    )
    contract_plan = narrow_rank_two_contract_to_consumer_row_domain(
        hlo_text,
        full_contract_plan,
        consumer_value=payload_source,
    )
    dead_payload_instructions = (
        (full_contract_plan.instruction, *payload_wrappers[:-1])
        if contract_plan.instruction != full_contract_plan.instruction
        else payload_wrappers
    )
    fold = RelationEdgeFoldTypedFfiPlan(
        instruction=scatter.name,
        initial=EntryRegionValue(initial.name, initial.shape),
        source_indices=EntryRegionValue(source_indices.name, source_indices.shape),
        payload=EntryRegionValue(contract_plan.instruction, contract_plan.output_shape),
        edge_cotangent=EntryRegionValue(edge_cotangent.name, edge_cotangent.shape),
        payload_logical_shape=instructions[payload_source].shape,
        payload_wrappers=payload_wrappers,
        internal_instructions=(
            *dead_payload_instructions,
            reduction_input.name,
            reduction.name,
            update_wrapper.name,
        ),
        output_shape=scatter.shape,
        contribution_program=contribution,
        generated_contribution_cuda=generate_cuda_scalar_body(
            contribution,
            symbol="generated_edge_contribution",
        ),
        inner_reducer_program=inner_reducer,
        generated_inner_reducer_cuda=generate_cuda_scalar_body(
            inner_reducer,
            symbol="generated_inner_fold_update",
        ),
        outer_reducer_program=outer_reducer,
        generated_outer_reducer_cuda=generate_cuda_scalar_body(
            outer_reducer,
            symbol="generated_outer_fold_update",
        ),
        payload_input_name=contribution_inputs[payload_index],
        edge_cotangent_input_name=contribution_inputs[1 - payload_index],
        inner_accumulator_input_name="input0",
        inner_contribution_input_name="input1",
        outer_accumulator_input_name="input0",
        outer_contribution_input_name="input1",
        external_users=users[scatter.name],
        api_version=1,
        numerical_contract=RelationEdgeFoldNumericalContract(
            input_dtype="bf16",
            map_dtype="bf16",
            inner_state_dtype="bf16",
            outer_state_dtype="bf16",
            output_dtype="bf16",
            numerical_policy=numerical_policy,
            deterministic=True,
            atomic_accumulation=False,
        ),
    )
    _validated_fold_dimensions(fold)
    return WeightedRelationReverseTypedFfiPlan(
        relation_plan=matching_relations[0],
        payload_policy=RelationPayloadPolicy.RECOMPUTE_CONTRACT,
        legal_payload_policies=(RelationPayloadPolicy.RECOMPUTE_CONTRACT,),
        payload_contract=contract_plan,
        edge_fold=fold,
    )


def _scalar_map_rank_two_sources(
    instructions: dict[str, HloInstruction],
    root: HloInstruction,
) -> tuple[str, str]:
    if len(root.operands) != 2:
        raise _NotWeightedRelationReverse
    sources = tuple(root.operands)
    shapes = tuple(_parse_shape(instructions[source].shape) for source in sources)
    root_shape = _parse_shape(root.shape)
    if any(shape != root_shape for shape in shapes):
        raise _NotWeightedRelationReverse
    return sources[0], sources[1]


def _payload_contract_path(
    instructions: dict[str, HloInstruction],
    source: str,
) -> tuple[str, tuple[str, ...]] | None:
    wrappers: list[str] = []
    current = source
    while instructions[current].opcode in _PAYLOAD_WRAPPER_OPCODES and len(instructions[current].operands) == 1:
        wrappers.append(current)
        current = instructions[current].operands[0]
    if instructions[current].opcode != "dot" or len(instructions[current].operands) != 2:
        return None
    return current, tuple(reversed(wrappers))


def _contract_dimensions(instruction: HloInstruction) -> ContractDimensionMap:
    parsed = {
        match.group("name"): tuple(int(value) for value in match.group("dims").split(",") if value)
        for match in _DOT_DIMENSIONS.finditer(instruction.attributes)
    }
    lhs_shape = _parse_shape(instruction.shape)
    if lhs_shape is None:
        raise _NotWeightedRelationReverse
    lhs_contracting = parsed.get("lhs_contracting_dims", ())
    rhs_contracting = parsed.get("rhs_contracting_dims", ())
    lhs_batch = parsed.get("lhs_batch_dims", ())
    rhs_batch = parsed.get("rhs_batch_dims", ())
    return ContractDimensionMap(
        lhs_contracting=lhs_contracting,
        rhs_contracting=rhs_contracting,
        lhs_batch=lhs_batch,
        rhs_batch=rhs_batch,
        lhs_output=tuple(axis for axis in range(2) if axis not in lhs_contracting and axis not in lhs_batch),
        rhs_output=tuple(axis for axis in range(2) if axis not in rhs_contracting and axis not in rhs_batch),
    )


def _reducer_program(module: HloModuleGraph, instruction: HloInstruction) -> CastScalarProgram:
    match = _CALLED_COMPUTATION.search(instruction.attributes)
    if match is None:
        raise _NotWeightedRelationReverse
    return import_hlo_scalar_computation(module.computation(match.group("name")))


def _validated_fold_dimensions(plan: RelationEdgeFoldTypedFfiPlan) -> dict[str, int]:
    initial = _required_shape(plan.initial.shape, dtype="bf16", rank=1)
    indices = _required_shape(plan.source_indices.shape, dtype="s32", rank=2)
    payload = _required_shape(plan.payload.shape, dtype="bf16", rank=2)
    logical_payload = _required_shape(plan.payload_logical_shape, dtype="bf16", rank=2)
    cotangent = _required_shape(plan.edge_cotangent.shape, dtype="bf16", rank=2)
    output = _required_shape(plan.output_shape, dtype="bf16", rank=1)
    if initial != output:
        raise ValueError("nested Fold output must match its initial state")
    if indices != (logical_payload[0], 1):
        raise ValueError("nested Fold requires one source index per logical edge")
    if logical_payload != cotangent:
        raise ValueError("payload and edge cotangent logical shapes disagree")
    if payload[0] < logical_payload[0] or payload[1] != logical_payload[1]:
        raise ValueError("physical payload does not contain every logical edge row")
    if plan.numerical_contract.numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("nested Fold cannot promise the original unspecified reduction tree")
    if not plan.numerical_contract.deterministic or plan.numerical_contract.atomic_accumulation:
        raise ValueError("nested Fold requires one deterministic owner per source slot")
    for program in (
        plan.contribution_program,
        plan.inner_reducer_program,
        plan.outer_reducer_program,
    ):
        if program.numerical_policy is not CastScalarNumericalPolicy.SOURCE_ORDERED:
            raise ValueError("nested Fold scalar bodies must preserve explicit casts")
    return {
        "sources": output[0],
        "edges": logical_payload[0],
        "physical_rows": payload[0],
        "features": payload[1],
    }


def _required_shape(shape: str, *, dtype: str, rank: int) -> tuple[int, ...]:
    parsed = _parse_shape(shape)
    if parsed is None or parsed[0] != dtype or len(parsed[1]) != rank:
        raise ValueError(f"expected {dtype} rank-{rank} row-major array, found {shape}")
    if parsed[2] != tuple(reversed(range(rank))):
        raise ValueError(f"expected row-major array, found {shape}")
    return parsed[1]


def _parse_shape(shape: str) -> tuple[str, tuple[int, ...], tuple[int, ...]] | None:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        return None
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    layout = tuple(int(value) for value in match.group("layout").split(",") if value)
    return match.group("dtype"), dimensions, layout


def _reduction_axes(instruction: HloInstruction) -> tuple[int, ...]:
    match = _DIMENSIONS.search(instruction.attributes)
    if match is None:
        return ()
    return tuple(int(value) for value in match.group("axes").split(",") if value)


def _is_zero_scalar(instruction: HloInstruction) -> bool:
    return instruction.shape.startswith("bf16[]") and bool(re.search(r"constant\((?:0|0\.0+)\)", instruction.attributes))


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    users: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)
    return {instruction: tuple(values) for instruction, values in users.items()}


def _graph_ancestor_distance(graph: InlinedHloGraph, ancestor: str, descendant: str) -> int | None:
    nodes = {node.id: node for node in graph.nodes}
    pending = [(descendant, 0)]
    seen: set[str] = set()
    while pending:
        current, distance = pending.pop(0)
        if current == ancestor:
            return distance
        if current in seen:
            continue
        seen.add(current)
        pending.extend((operand, distance + 1) for operand in nodes[current].operands)
    return None


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
            raise ValueError(f"relation reverse output %{current} has {len(direct_users)} users")
        user = instructions[direct_users[0]]
        if user.opcode == "all-reduce":
            return tuple(wrappers), user.name
        if user.opcode not in _VIEW_OPCODES or len(user.operands) != 1:
            raise ValueError(
                f"relation reverse output does not reach a placement collective through views: %{user.name}"
            )
        wrappers.append(user.name)
        current = user.name


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


def _target_symbol(target: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", target)
    if not symbol or symbol[0].isdigit():
        symbol = f"shuttle_{symbol}"
    return symbol
