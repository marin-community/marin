# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generic routed relation programs from physical XLA HLO.

The recovery deliberately ignores frontend metadata, instruction spelling, and
model names.  It reconstructs the pieces Shuttle needs to own after XLA has
already exposed physical padding and fusion choices:

* selection followed by a stable destination-major ``RelationPlan``;
* grouped/segmented Contract chains with an explicit pointwise Map between
  them;
* source-keyed scatter Folds;
* Contract adjoints for routed activations and grouped weights; and
* collectives that remain explicit external boundaries.

This is an inspection and region-formation pass.  It does not yet replace the
identified regions with a GPU custom call.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.cast_scalar_program import (
    CastScalarNumericalPolicy,
    CastScalarProgram,
    GeneratedCudaScalarBody,
    generate_cuda_scalar_body,
)
from tile_lifetime.xla_hlo_recovery import (
    EntryRegionValue,
    HloComputation,
    HloModuleGraph,
    InlinedHloGraph,
    InlinedHloNode,
    RecoveredEntryRegionBoundary,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_computation, import_hlo_scalar_map

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_CALLED_COMPUTATION = re.compile(r"to_apply=%([A-Za-z0-9_.-]+)")
_DOT_DIMENSIONS = re.compile(
    r"(?P<name>lhs_contracting_dims|rhs_contracting_dims|lhs_batch_dims|rhs_batch_dims)=\{(?P<dims>[0-9,]*)\}"
)
_DATAFLOW_OPCODES = frozenset(
    {
        "add",
        "bitcast",
        "broadcast",
        "concatenate",
        "convert",
        "copy",
        "divide",
        "exponential",
        "multiply",
        "negate",
        "pad",
        "reshape",
        "select",
        "slice",
        "subtract",
        "tanh",
        "transpose",
    }
)


class ContractChainRole(StrEnum):
    """Structural role of a pair of routed Contracts."""

    FORWARD = "forward"
    FORWARD_RECOMPUTE = "forward_recompute"
    INPUT_GRADIENT = "input_gradient"


@dataclass(frozen=True)
class RelationPlanRecord:
    """One selected edge set indexed in stable destination-major order."""

    selection: str
    selected_indices: str
    selected_shape: str
    token_count: int
    slots_per_token: int
    edge_count: int
    destination_sort: str
    stable_permutation: str
    destination_counts: str
    destination_offsets: str
    destination_count: int


@dataclass(frozen=True)
class SegmentedContractRecord:
    """One physical Contract whose row domain is relation-segmented."""

    node: str
    input_shapes: tuple[str, ...]
    output_shape: str
    relation_plans: tuple[str, ...]


@dataclass(frozen=True)
class ScalarMapOutputRecord:
    """One generated scalar result placed in a pointwise output feature range."""

    feature_offset: int
    feature_extent: int
    scalar_program: CastScalarProgram
    generated_cuda: GeneratedCudaScalarBody


@dataclass(frozen=True)
class PointwiseMapRecord:
    """Pointwise/shape program between two segmented Contracts."""

    opcodes: tuple[str, ...]
    cast_shapes: tuple[tuple[str, str], ...]
    scalar_outputs: tuple[ScalarMapOutputRecord, ...]


@dataclass(frozen=True)
class SegmentedContractChainRecord:
    """Two segmented Contracts connected by one recovered Map program."""

    role: ContractChainRole
    first: SegmentedContractRecord
    map: PointwiseMapRecord
    second: SegmentedContractRecord


@dataclass(frozen=True)
class FoldRecord:
    """A source-keyed scatter Fold derived from a routed Contract output."""

    source_contract: str
    contribution_inputs: tuple[str, ...]
    contribution_scalar_program: CastScalarProgram
    generated_contribution_cuda: GeneratedCudaScalarBody
    contribution_opcodes: tuple[str, ...]
    fold: str
    output_shape: str
    reducer: str
    reducer_opcodes: tuple[str, ...]
    reducer_scalar_program: CastScalarProgram
    generated_reducer_cuda: GeneratedCudaScalarBody


@dataclass(frozen=True)
class RoutedWeightGradientRecord:
    """A group-batched Contract producing one gradient matrix per segment."""

    contract: SegmentedContractRecord
    external_collectives: tuple[str, ...]


@dataclass(frozen=True)
class ContractDimensionMap:
    """Dimension roles needed by a generic contraction implementation."""

    lhs_contracting: tuple[int, ...]
    rhs_contracting: tuple[int, ...]
    lhs_batch: tuple[int, ...]
    rhs_batch: tuple[int, ...]
    lhs_output: tuple[int, ...]
    rhs_output: tuple[int, ...]


@dataclass(frozen=True)
class RoutedForwardContractStage:
    """One generic cuBLAS-compatible Contract in a routed forward region."""

    node: str
    lhs: str
    rhs: str
    output_shape: str
    dimensions: ContractDimensionMap
    backend: str = "cublas"


@dataclass(frozen=True)
class RoutedForwardMapStage:
    """Recovered scalar Map plus the physical value required by its consumer."""

    source_contract: str
    consumer_contract: str
    logical_row_extent: int
    logical_feature_extent: int
    physical_output: str
    physical_output_shape: str
    layout_path: tuple[str, ...]
    layout_opcodes: tuple[str, ...]
    scalar_outputs: tuple[ScalarMapOutputRecord, ...]
    has_segmented_layout: bool


@dataclass(frozen=True)
class RoutedForwardFoldStage:
    """Generated contribution and reducer programs for a source-keyed Fold."""

    contribution_inputs: tuple[str, ...]
    contribution_program: CastScalarProgram
    generated_contribution_cuda: GeneratedCudaScalarBody
    fold: str
    output_shape: str
    reducer_program: CastScalarProgram
    generated_reducer_cuda: GeneratedCudaScalarBody


@dataclass(frozen=True)
class RoutedForwardRegionRecord:
    """Convex physical region recovered from one Contract/Map/Contract/Fold chain."""

    chain: SegmentedContractChainRecord
    fold: FoldRecord
    boundary: RecoveredEntryRegionBoundary
    contracts: tuple[RoutedForwardContractStage, ...]
    map_stage: RoutedForwardMapStage
    fold_stage: RoutedForwardFoldStage
    numerical_policy: CastScalarNumericalPolicy
    convex: bool
    topologically_insertable: bool
    insertion_instruction: str
    requires_auxiliary_output_split: bool


class RoutedForwardCodegenDisposition(StrEnum):
    """Whether the recovered region has enough information for typed FFI."""

    READY = "ready"
    MISSING_SEGMENTED_LAYOUT = "missing_segmented_layout"


class SegmentedLayoutRelation(StrEnum):
    """Index relations required to lower a logical routed Map physically."""

    EDGE_ROW_TO_PADDED_ROW = "edge_row_to_padded_row"
    SEGMENT_TO_FEATURE_PANEL = "segment_to_feature_panel"
    VALIDITY_AND_FILL = "validity_and_fill"
    SOURCE_FOLD_INVERSE = "source_fold_inverse"


@dataclass(frozen=True)
class SegmentedLayoutRequirement:
    """Missing physical index semantics blocking the routed typed-FFI plan."""

    logical_shape: tuple[int, int]
    physical_shape: str
    observed_path_opcodes: tuple[str, ...]
    required_relations: tuple[SegmentedLayoutRelation, ...]


@dataclass(frozen=True)
class RoutedForwardTypedFfiCodegenPlan:
    """A generic typed-FFI plan or a structured missing-information result."""

    region: RoutedForwardRegionRecord
    api_version: int
    contracts: tuple[RoutedForwardContractStage, ...]
    map_stage: RoutedForwardMapStage
    fold_stage: RoutedForwardFoldStage
    disposition: RoutedForwardCodegenDisposition
    missing_segmented_layout: SegmentedLayoutRequirement | None


@dataclass(frozen=True)
class RelationProgramRecoveryReport:
    """Inspectable generic ownership boundary for routed forward/backward work."""

    relation_plans: tuple[RelationPlanRecord, ...]
    contract_chains: tuple[SegmentedContractChainRecord, ...]
    folds: tuple[FoldRecord, ...]
    weight_gradients: tuple[RoutedWeightGradientRecord, ...]
    external_collectives: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Encode the report using only JSON-compatible values."""

        def contract(value: SegmentedContractRecord) -> dict[str, object]:
            return {
                "node": value.node,
                "input_shapes": list(value.input_shapes),
                "output_shape": value.output_shape,
                "relation_plans": list(value.relation_plans),
            }

        return {
            "relation_plans": [
                {
                    "selection": plan.selection,
                    "selected_indices": plan.selected_indices,
                    "selected_shape": plan.selected_shape,
                    "token_count": plan.token_count,
                    "slots_per_token": plan.slots_per_token,
                    "edge_count": plan.edge_count,
                    "destination_sort": plan.destination_sort,
                    "stable_permutation": plan.stable_permutation,
                    "destination_counts": plan.destination_counts,
                    "destination_offsets": plan.destination_offsets,
                    "destination_count": plan.destination_count,
                }
                for plan in self.relation_plans
            ],
            "contract_chains": [
                {
                    "role": chain.role.value,
                    "first": contract(chain.first),
                    "map": {
                        "opcodes": list(chain.map.opcodes),
                        "cast_shapes": [list(shapes) for shapes in chain.map.cast_shapes],
                        "scalar_outputs": [
                            {
                                "feature_offset": output.feature_offset,
                                "feature_extent": output.feature_extent,
                                "scalar_program": output.scalar_program.to_dict(),
                                "generated_cuda": output.generated_cuda.to_dict(),
                            }
                            for output in chain.map.scalar_outputs
                        ],
                    },
                    "second": contract(chain.second),
                }
                for chain in self.contract_chains
            ],
            "folds": [
                {
                    "source_contract": fold.source_contract,
                    "contribution_inputs": list(fold.contribution_inputs),
                    "contribution_scalar_program": fold.contribution_scalar_program.to_dict(),
                    "generated_contribution_cuda": fold.generated_contribution_cuda.to_dict(),
                    "contribution_opcodes": list(fold.contribution_opcodes),
                    "fold": fold.fold,
                    "output_shape": fold.output_shape,
                    "reducer": fold.reducer,
                    "reducer_opcodes": list(fold.reducer_opcodes),
                    "reducer_scalar_program": fold.reducer_scalar_program.to_dict(),
                    "generated_reducer_cuda": fold.generated_reducer_cuda.to_dict(),
                }
                for fold in self.folds
            ],
            "weight_gradients": [
                {
                    "contract": contract(gradient.contract),
                    "external_collectives": list(gradient.external_collectives),
                }
                for gradient in self.weight_gradients
            ],
            "external_collectives": list(self.external_collectives),
            "limitations": list(self.limitations),
        }


def recover_relation_programs(hlo_text: str) -> RelationProgramRecoveryReport:
    """Recover routed forward/backward structure using only HLO dataflow."""
    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    analysis = _GraphAnalysis(module, graph)
    relation_plans = _recover_relation_plans(analysis)
    relation_nodes = tuple(plan.destination_sort for plan in relation_plans)
    chains = _recover_segmented_contract_chains(analysis, relation_plans)
    folds = _recover_scatter_folds(analysis, chains)
    weight_gradients = _recover_weight_gradients(analysis, relation_nodes, relation_plans)
    collectives = tuple(
        node.id
        for node in graph.nodes
        if node.opcode == "all-reduce"
        and any(
            analysis.is_ancestor(owned, node.id)
            for owned in (
                *(chain.second.node for chain in chains),
                *(gradient.contract.node for gradient in weight_gradients),
            )
        )
    )
    return RelationProgramRecoveryReport(
        relation_plans=relation_plans,
        contract_chains=chains,
        folds=folds,
        weight_gradients=weight_gradients,
        external_collectives=collectives,
        limitations=(
            "The pass forms generic ownership boundaries but does not yet replace them with GPU execution.",
            "Physical HLO has already padded the runtime segments; dynamic counts and offsets remain explicit inputs.",
            "Collectives are intentionally reported as external placement boundaries rather than absorbed.",
            "Forward and concatenated input-adjoint Maps lower to generic cast-aware scalar AST outputs and "
            "generated CUDA device functions; execution inside a grouped Contract remains unimplemented.",
            "Scalar HLO import currently accepts rank-two sources, unit-stride slices, scalar broadcasts, and "
            "feature-axis concatenation; broader affine index maps remain unsupported.",
        ),
    )


def form_routed_forward_region(hlo_text: str) -> RoutedForwardRegionRecord:
    """Form one maximal routed Contract/Map/Contract/Fold entry region.

    Region membership follows physical entry-computation dataflow. Instruction
    spelling and frontend metadata do not participate. The region begins at the
    first Contract in the uniquely recovered forward chain and ends at its
    source-keyed Fold.
    """
    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    report = recover_relation_programs(hlo_text)
    candidates = tuple(chain for chain in report.contract_chains if chain.role is ContractChainRole.FORWARD)
    if len(candidates) != 1:
        raise ValueError(f"expected one structurally recovered forward Contract chain, found {len(candidates)}")
    chain = candidates[0]
    folds = tuple(fold for fold in report.folds if fold.source_contract == chain.second.node)
    if len(folds) != 1:
        raise ValueError(f"forward Contract chain has {len(folds)} source-keyed Folds")
    fold = folds[0]
    entry = module.computation(module.entry)
    first_instruction = _entry_instruction_for_node(module, chain.first.node)
    second_instruction = _entry_instruction_for_node(module, chain.second.node)
    fold_instruction = _entry_instruction_for_node(module, fold.fold)
    internal = _entry_dataflow_interval(entry, first_instruction, fold_instruction)
    if second_instruction not in internal:
        raise ValueError("forward dataflow interval does not contain the second Contract")
    boundary = _entry_region_boundary(entry, internal)
    convex = _entry_region_is_convex(entry, internal)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    insertion_instruction = min(internal, key=source_order.__getitem__)
    topologically_insertable = all(
        source_order[value.instruction] < source_order[insertion_instruction] for value in boundary.inputs
    )
    internal_set = set(internal)
    requires_auxiliary_output_split = any(
        output.instruction != fold_instruction
        and any(user not in internal_set for user in dict(boundary.external_users)[output.instruction])
        for output in boundary.outputs
    )

    contracts = tuple(_contract_codegen_stage(graph, contract) for contract in (chain.first, chain.second))
    relation_plans = tuple(plan for plan in report.relation_plans if plan.destination_sort in chain.first.relation_plans)
    if len(relation_plans) != 1:
        raise ValueError(f"forward Contract chain has {len(relation_plans)} physical relation plans")
    relation_plan = relation_plans[0]
    first_node = graph.node(chain.first.node)
    second_node = graph.node(chain.second.node)
    second_data_operands = tuple(
        operand for operand in second_node.operands if _is_node_descendant(graph, first_node.id, operand)
    )
    if len(second_data_operands) != 1:
        raise ValueError(f"second Contract has {len(second_data_operands)} operands derived from the first Contract")
    physical_map_output = second_data_operands[0]
    layout_path = _inlined_path(graph, first_node.id, physical_map_output)
    logical_feature_extent = sum(output.feature_extent for output in chain.map.scalar_outputs)
    physical_shape = graph.node(physical_map_output).shape
    parsed_physical_shape = _parse_array_shape(physical_shape)
    has_segmented_layout = parsed_physical_shape is not None and parsed_physical_shape[1] == (
        relation_plan.edge_count,
        logical_feature_extent,
    )
    map_stage = RoutedForwardMapStage(
        source_contract=chain.first.node,
        consumer_contract=chain.second.node,
        logical_row_extent=relation_plan.edge_count,
        logical_feature_extent=logical_feature_extent,
        physical_output=physical_map_output,
        physical_output_shape=physical_shape,
        layout_path=layout_path,
        layout_opcodes=tuple(graph.node(node_id).opcode for node_id in layout_path[1:]),
        scalar_outputs=chain.map.scalar_outputs,
        has_segmented_layout=has_segmented_layout,
    )
    fold_stage = RoutedForwardFoldStage(
        contribution_inputs=fold.contribution_inputs,
        contribution_program=fold.contribution_scalar_program,
        generated_contribution_cuda=fold.generated_contribution_cuda,
        fold=fold.fold,
        output_shape=fold.output_shape,
        reducer_program=fold.reducer_scalar_program,
        generated_reducer_cuda=fold.generated_reducer_cuda,
    )
    numerical_policies = {
        *(output.scalar_program.numerical_policy for output in chain.map.scalar_outputs),
        fold.contribution_scalar_program.numerical_policy,
        fold.reducer_scalar_program.numerical_policy,
    }
    if len(numerical_policies) != 1:
        raise ValueError(f"forward region contains {len(numerical_policies)} numerical policies")
    return RoutedForwardRegionRecord(
        chain=chain,
        fold=fold,
        boundary=boundary,
        contracts=contracts,
        map_stage=map_stage,
        fold_stage=fold_stage,
        numerical_policy=next(iter(numerical_policies)),
        convex=convex,
        topologically_insertable=topologically_insertable,
        insertion_instruction=insertion_instruction,
        requires_auxiliary_output_split=requires_auxiliary_output_split,
    )


def plan_routed_forward_typed_ffi(hlo_text: str) -> RoutedForwardTypedFfiCodegenPlan:
    """Plan a generic typed-FFI implementation without inventing layout semantics."""
    region = form_routed_forward_region(hlo_text)
    if not region.convex or not region.topologically_insertable:
        raise ValueError("routed forward region is not convex and topologically insertable")
    if region.requires_auxiliary_output_split:
        raise ValueError("routed forward region requires a generic auxiliary-output split")
    missing_segmented_layout = None
    disposition = RoutedForwardCodegenDisposition.READY
    if not region.map_stage.has_segmented_layout:
        missing_segmented_layout = SegmentedLayoutRequirement(
            logical_shape=(region.map_stage.logical_row_extent, region.map_stage.logical_feature_extent),
            physical_shape=region.map_stage.physical_output_shape,
            observed_path_opcodes=region.map_stage.layout_opcodes,
            required_relations=(
                SegmentedLayoutRelation.EDGE_ROW_TO_PADDED_ROW,
                SegmentedLayoutRelation.SEGMENT_TO_FEATURE_PANEL,
                SegmentedLayoutRelation.VALIDITY_AND_FILL,
                SegmentedLayoutRelation.SOURCE_FOLD_INVERSE,
            ),
        )
        disposition = RoutedForwardCodegenDisposition.MISSING_SEGMENTED_LAYOUT
    return RoutedForwardTypedFfiCodegenPlan(
        region=region,
        api_version=1,
        contracts=region.contracts,
        map_stage=region.map_stage,
        fold_stage=region.fold_stage,
        disposition=disposition,
        missing_segmented_layout=missing_segmented_layout,
    )


def _entry_instruction_for_node(module: HloModuleGraph, node_id: str) -> str:
    prefix = f"{module.entry}/"
    if not node_id.startswith(prefix):
        raise ValueError(f"node {node_id!r} is outside the entry computation")
    instruction = node_id.removeprefix(prefix).split("/", 1)[0]
    entry_names = {value.name for value in module.computation(module.entry).instructions}
    if instruction not in entry_names:
        raise ValueError(f"node {node_id!r} has no physical entry instruction")
    return instruction


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            mutable.setdefault(operand, []).append(instruction.name)
    return {name: tuple(values) for name, values in mutable.items()}


def _entry_dataflow_interval(entry: HloComputation, start: str, finish: str) -> tuple[str, ...]:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry)
    descendants = {start}
    pending = [start]
    while pending:
        current = pending.pop()
        for user in users.get(current, ()):
            if user not in descendants:
                descendants.add(user)
                pending.append(user)
    ancestors = {finish}
    pending = [finish]
    while pending:
        current = pending.pop()
        for operand in instructions[current].operands:
            if operand not in ancestors:
                ancestors.add(operand)
                pending.append(operand)
    interval = descendants & ancestors
    order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    return tuple(sorted(interval, key=order.__getitem__))


def _entry_region_boundary(
    entry: HloComputation,
    internal: tuple[str, ...],
) -> RecoveredEntryRegionBoundary:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    users = _entry_users(entry)
    internal_set = set(internal)
    input_names = {
        operand for name in internal for operand in instructions[name].operands if operand not in internal_set
    }
    output_names = {name for name in internal if any(user not in internal_set for user in users.get(name, ()))}
    ordered_inputs = tuple(sorted(input_names, key=order.__getitem__))
    ordered_outputs = tuple(sorted(output_names, key=order.__getitem__))
    return RecoveredEntryRegionBoundary(
        internal_instructions=internal,
        inputs=tuple(EntryRegionValue(name, instructions[name].shape) for name in ordered_inputs),
        outputs=tuple(EntryRegionValue(name, instructions[name].shape) for name in ordered_outputs),
        external_users=tuple(
            (name, tuple(user for user in users.get(name, ()) if user not in internal_set)) for name in ordered_outputs
        ),
        has_explicit_sharding=any("sharding=" in instructions[name].attributes for name in internal),
        has_side_effect=any("custom_call_has_side_effect=true" in instructions[name].attributes for name in internal),
    )


def _entry_region_is_convex(entry: HloComputation, internal: tuple[str, ...]) -> bool:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry)
    internal_set = set(internal)
    descendants: set[str] = set(internal)
    pending = list(internal)
    while pending:
        current = pending.pop()
        for user in users.get(current, ()):
            if user not in descendants:
                descendants.add(user)
                pending.append(user)
    ancestors: set[str] = set(internal)
    pending = list(internal)
    while pending:
        current = pending.pop()
        for operand in instructions[current].operands:
            if operand not in ancestors:
                ancestors.add(operand)
                pending.append(operand)
    return not ((descendants & ancestors) - internal_set)


def _contract_codegen_stage(
    graph: InlinedHloGraph,
    contract: SegmentedContractRecord,
) -> RoutedForwardContractStage:
    node = graph.node(contract.node)
    if node.opcode != "dot" or len(node.operands) != 2:
        raise ValueError(f"Contract {contract.node!r} is not a binary dot")
    parsed_dimensions = {
        match.group("name"): tuple(int(value) for value in match.group("dims").split(",") if value)
        for match in _DOT_DIMENSIONS.finditer(node.attributes)
    }
    lhs_shape = _parse_array_shape(graph.node(node.operands[0]).shape)
    rhs_shape = _parse_array_shape(graph.node(node.operands[1]).shape)
    if lhs_shape is None or rhs_shape is None:
        raise ValueError(f"Contract {contract.node!r} has a non-array operand")
    lhs_contracting = parsed_dimensions.get("lhs_contracting_dims", ())
    rhs_contracting = parsed_dimensions.get("rhs_contracting_dims", ())
    lhs_batch = parsed_dimensions.get("lhs_batch_dims", ())
    rhs_batch = parsed_dimensions.get("rhs_batch_dims", ())
    lhs_output = tuple(axis for axis in range(len(lhs_shape[1])) if axis not in lhs_contracting + lhs_batch)
    rhs_output = tuple(axis for axis in range(len(rhs_shape[1])) if axis not in rhs_contracting + rhs_batch)
    return RoutedForwardContractStage(
        node=node.id,
        lhs=node.operands[0],
        rhs=node.operands[1],
        output_shape=node.shape,
        dimensions=ContractDimensionMap(
            lhs_contracting=lhs_contracting,
            rhs_contracting=rhs_contracting,
            lhs_batch=lhs_batch,
            rhs_batch=rhs_batch,
            lhs_output=lhs_output,
            rhs_output=rhs_output,
        ),
    )


def _inlined_users(graph: InlinedHloGraph) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for node in graph.nodes:
        for operand in node.operands:
            mutable.setdefault(operand, []).append(node.id)
    return {name: tuple(values) for name, values in mutable.items()}


def _is_node_descendant(graph: InlinedHloGraph, ancestor: str, candidate: str) -> bool:
    pending = [candidate]
    seen = {candidate}
    nodes = {node.id: node for node in graph.nodes}
    while pending:
        current = pending.pop()
        if current == ancestor:
            return True
        for operand in nodes[current].operands:
            if operand not in seen:
                seen.add(operand)
                pending.append(operand)
    return False


def _inlined_path(graph: InlinedHloGraph, start: str, finish: str) -> tuple[str, ...]:
    users = _inlined_users(graph)
    pending = deque(((start, (start,)),))
    seen = {start}
    while pending:
        current, path = pending.popleft()
        if current == finish:
            return path
        for user in users.get(current, ()):
            if user not in seen:
                seen.add(user)
                pending.append((user, (*path, user)))
    raise ValueError(f"no inlined dataflow path from {start!r} to {finish!r}")


class _GraphAnalysis:
    def __init__(self, module: HloModuleGraph, graph: InlinedHloGraph):
        self.module = module
        self.graph = graph
        self.nodes = {node.id: node for node in graph.nodes}
        self.users: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
        for node in graph.nodes:
            for operand in node.operands:
                self.users.setdefault(operand, []).append(node.id)
        self.order = {node.id: index for index, node in enumerate(graph.nodes)}
        self._ancestor_cache: dict[str, frozenset[str]] = {}

    def ancestors(self, node_id: str) -> frozenset[str]:
        if node_id in self._ancestor_cache:
            return self._ancestor_cache[node_id]
        result = {node_id}
        pending = list(self.nodes[node_id].operands)
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(self.nodes[current].operands)
        frozen = frozenset(result)
        self._ancestor_cache[node_id] = frozen
        return frozen

    def is_ancestor(self, possible_ancestor: str, node_id: str) -> bool:
        return possible_ancestor in self.ancestors(node_id)

    def nearest_path(
        self,
        start: str,
        predicate: Callable[[InlinedHloNode], bool],
        *,
        maximum_depth: int = 64,
    ) -> tuple[str, ...] | None:
        pending = deque([(start, (start,))])
        seen = {start}
        while pending:
            current, path = pending.popleft()
            if len(path) - 1 >= maximum_depth:
                continue
            for user in self.users.get(current, ()):
                if user in seen:
                    continue
                next_path = (*path, user)
                if predicate(self.nodes[user]):
                    return next_path
                seen.add(user)
                pending.append((user, next_path))
        return None


def _recover_relation_plans(analysis: _GraphAnalysis) -> tuple[RelationPlanRecord, ...]:
    records: list[RelationPlanRecord] = []
    for node in analysis.graph.nodes:
        if not _is_destination_sort(node, analysis):
            continue
        selected = analysis.graph.strip_wrappers(node.operands[0]).base
        selected_shape = analysis.nodes[selected].shape
        parsed = _parse_array_shape(selected_shape)
        if parsed is None or parsed[0] not in {"s32", "s64"} or len(parsed[1]) != 2:
            continue
        token_count, slots_per_token = parsed[1]
        selection = _nearest_selection_ancestor(analysis, selected)
        if selection is None:
            continue
        permutation = _integer_tuple_projection(analysis, node.id, rank=1)
        counts = _destination_counts(analysis, selection, token_count * slots_per_token)
        offsets = analysis.nearest_path(
            counts,
            lambda candidate: candidate.opcode == "reduce-window" and candidate.dtype in {"s32", "s64"},
            maximum_depth=8,
        )
        if permutation is None or offsets is None:
            continue
        counts_shape = _parse_array_shape(analysis.nodes[counts].shape)
        assert counts_shape is not None and len(counts_shape[1]) == 1
        records.append(
            RelationPlanRecord(
                selection=selection,
                selected_indices=selected,
                selected_shape=selected_shape,
                token_count=token_count,
                slots_per_token=slots_per_token,
                edge_count=token_count * slots_per_token,
                destination_sort=node.id,
                stable_permutation=permutation,
                destination_counts=counts,
                destination_offsets=offsets[-1],
                destination_count=counts_shape[1][0],
            )
        )
    return tuple(records)


def _recover_segmented_contract_chains(
    analysis: _GraphAnalysis,
    relation_plans: tuple[RelationPlanRecord, ...],
) -> tuple[SegmentedContractChainRecord, ...]:
    relation_nodes = tuple(plan.destination_sort for plan in relation_plans)
    contracts = tuple(
        node
        for node in analysis.graph.nodes
        if node.opcode == "dot"
        and (shape := _parse_array_shape(node.shape)) is not None
        and len(shape[1]) == 2
        and any(analysis.is_ancestor(relation, node.id) for relation in relation_nodes)
    )
    records: list[SegmentedContractChainRecord] = []
    for first in contracts:
        first_shape = _parse_array_shape(first.shape)
        assert first_shape is not None
        path = analysis.nearest_path(
            first.id,
            lambda candidate, first_id=first.id, first_extent=first_shape[1][0]: (
                candidate.opcode == "dot"
                and candidate.id != first_id
                and (candidate_shape := _parse_array_shape(candidate.shape)) is not None
                and len(candidate_shape[1]) == 2
                and candidate_shape[1][0] == first_extent
            ),
            maximum_depth=64,
        )
        if path is None:
            continue
        second = analysis.nodes[path[-1]]
        if second not in contracts:
            continue
        intermediate = tuple(analysis.nodes[node_id] for node_id in path[1:-1])
        if any(node.opcode not in _DATAFLOW_OPCODES for node in intermediate):
            continue
        opcodes = tuple(node.opcode for node in intermediate if node.opcode in _DATAFLOW_OPCODES)
        if "slice" in opcodes and "concatenate" not in opcodes:
            role = ContractChainRole.FORWARD
            if _nearest_float_scatter(analysis, second.id) is None:
                role = ContractChainRole.FORWARD_RECOMPUTE
        elif "concatenate" in opcodes:
            role = ContractChainRole.INPUT_GRADIENT
        else:
            continue
        chain_relation_nodes = tuple(relation for relation in relation_nodes if analysis.is_ancestor(relation, first.id))
        scalar_outputs: tuple[ScalarMapOutputRecord, ...] = ()
        if role in {ContractChainRole.FORWARD, ContractChainRole.FORWARD_RECOMPUTE}:
            edge_counts = {
                plan.edge_count for plan in relation_plans if analysis.is_ancestor(plan.destination_sort, first.id)
            }
            if len(edge_counts) != 1:
                raise ValueError(f"Contract {first.id!r} has {len(edge_counts)} routed edge counts")
            scalar_program, feature_extent = _import_forward_scalar_map(
                analysis,
                first=first,
                second=second,
                edge_count=next(iter(edge_counts)),
            )
            scalar_outputs = (
                ScalarMapOutputRecord(
                    feature_offset=0,
                    feature_extent=feature_extent,
                    scalar_program=scalar_program,
                    generated_cuda=generate_cuda_scalar_body(scalar_program),
                ),
            )
        elif role is ContractChainRole.INPUT_GRADIENT:
            edge_counts = {
                plan.edge_count for plan in relation_plans if analysis.is_ancestor(plan.destination_sort, first.id)
            }
            if len(edge_counts) != 1:
                raise ValueError(f"Contract {first.id!r} has {len(edge_counts)} routed edge counts")
            scalar_outputs = _import_concatenated_scalar_map(
                analysis,
                first=first,
                second=second,
                edge_count=next(iter(edge_counts)),
            )
        records.append(
            SegmentedContractChainRecord(
                role=role,
                first=_contract_record(first, chain_relation_nodes, analysis),
                map=PointwiseMapRecord(
                    opcodes=opcodes,
                    cast_shapes=tuple(
                        (analysis.nodes[node.operands[0]].shape, node.shape)
                        for node in intermediate
                        if node.opcode == "convert" and len(node.operands) == 1
                    ),
                    scalar_outputs=scalar_outputs,
                ),
                second=_contract_record(second, chain_relation_nodes, analysis),
            )
        )
    return tuple(records)


def _import_forward_scalar_map(
    analysis: _GraphAnalysis,
    *,
    first: InlinedHloNode,
    second: InlinedHloNode,
    edge_count: int,
) -> tuple[CastScalarProgram, int]:
    second_operands = tuple(operand for operand in second.operands if analysis.is_ancestor(first.id, operand))
    if len(second_operands) != 1:
        raise ValueError(f"Contract pair {first.id!r} -> {second.id!r} has {len(second_operands)} data operands")
    destination = second_operands[0]
    destination_ancestors = analysis.ancestors(destination)
    candidates = {
        node.id
        for node in analysis.graph.nodes
        if node.id != first.id
        and node.id in destination_ancestors
        and analysis.is_ancestor(first.id, node.id)
        and (shape := _parse_array_shape(node.shape)) is not None
        and len(shape[1]) == 2
        and shape[1][0] == edge_count
        and node.opcode in _DATAFLOW_OPCODES
    }
    terminals = tuple(
        node_id for node_id in candidates if not any(user in candidates for user in analysis.users.get(node_id, ()))
    )
    if len(terminals) != 1:
        raise ValueError(f"Contract pair {first.id!r} -> {second.id!r} has {len(terminals)} scalar Map frontiers")
    target = analysis.nodes[terminals[0]]
    target_shape = _parse_array_shape(target.shape)
    assert target_shape is not None and len(target_shape[1]) == 2
    return (
        import_hlo_scalar_map(
            analysis.graph,
            source_nodes=(first.id,),
            target_node=target.id,
        ),
        target_shape[1][1],
    )


def _import_concatenated_scalar_map(
    analysis: _GraphAnalysis,
    *,
    first: InlinedHloNode,
    second: InlinedHloNode,
    edge_count: int,
) -> tuple[ScalarMapOutputRecord, ...]:
    second_operands = tuple(operand for operand in second.operands if analysis.is_ancestor(first.id, operand))
    if len(second_operands) != 1:
        raise ValueError(f"Contract pair {first.id!r} -> {second.id!r} has {len(second_operands)} data operands")
    destination = second_operands[0]
    destination_ancestors = analysis.ancestors(destination)
    candidates = {
        node.id
        for node in analysis.graph.nodes
        if node.id != first.id
        and node.id in destination_ancestors
        and analysis.is_ancestor(first.id, node.id)
        and (shape := _parse_array_shape(node.shape)) is not None
        and len(shape[1]) == 2
        and shape[1][0] == edge_count
        and node.opcode in _DATAFLOW_OPCODES
    }
    terminals = tuple(
        node_id for node_id in candidates if not any(user in candidates for user in analysis.users.get(node_id, ()))
    )
    concatenations = (
        tuple(
            node_id
            for node_id in candidates
            if analysis.nodes[node_id].opcode == "concatenate" and node_id in analysis.ancestors(terminals[0])
        )
        if len(terminals) == 1
        else ()
    )
    if len(terminals) != 1 or len(concatenations) != 1:
        raise ValueError(
            f"Contract pair {first.id!r} -> {second.id!r} has {len(terminals)} Map frontiers and "
            f"{len(concatenations)} concatenations"
        )
    target = terminals[0]
    concatenate = analysis.nodes[concatenations[0]]
    if "dimensions={1}" not in concatenate.attributes:
        raise ValueError(f"scalar output concatenation {concatenate.id!r} must join the feature axis")
    leaf_sources = _dataflow_leaf_sources(analysis, target)
    if first.id not in leaf_sources:
        raise ValueError(f"input-adjoint Map {target!r} does not consume Contract {first.id!r}")
    source_nodes = (first.id, *sorted(leaf_sources - {first.id}, key=analysis.order.__getitem__))
    outputs: list[ScalarMapOutputRecord] = []
    feature_offset = 0
    for operand_index, operand in enumerate(concatenate.operands):
        shape = _parse_array_shape(analysis.nodes[operand].shape)
        if shape is None or len(shape[1]) != 2 or shape[1][0] != edge_count:
            raise ValueError(f"concatenated scalar output {operand!r} must be a routed rank-two array")
        scalar_program = import_hlo_scalar_map(
            analysis.graph,
            source_nodes=source_nodes,
            target_node=target,
            concatenate_choices={concatenate.id: operand_index},
        )
        outputs.append(
            ScalarMapOutputRecord(
                feature_offset=feature_offset,
                feature_extent=shape[1][1],
                scalar_program=scalar_program,
                generated_cuda=generate_cuda_scalar_body(
                    scalar_program,
                    symbol=f"generated_scalar_map_{operand_index}",
                ),
            )
        )
        feature_offset += shape[1][1]
    return tuple(outputs)


def _dataflow_leaf_sources(analysis: _GraphAnalysis, target: str) -> set[str]:
    leaves: set[str] = set()
    pending = [target]
    seen = {target}
    while pending:
        current = pending.pop()
        node = analysis.nodes[current]
        if node.opcode == "constant":
            continue
        if current != target and node.opcode not in _DATAFLOW_OPCODES:
            leaves.add(current)
            continue
        for operand in node.operands:
            if operand not in seen:
                seen.add(operand)
                pending.append(operand)
    return leaves


def _recover_scatter_folds(
    analysis: _GraphAnalysis,
    chains: tuple[SegmentedContractChainRecord, ...],
) -> tuple[FoldRecord, ...]:
    records: list[FoldRecord] = []
    for chain in chains:
        if chain.role not in {ContractChainRole.FORWARD, ContractChainRole.INPUT_GRADIENT}:
            continue
        path = _nearest_float_scatter(analysis, chain.second.node)
        if path is None:
            continue
        fold = analysis.nodes[path[-1]]
        reducer, reducer_opcodes, reducer_program = _scatter_reducer_program(analysis.module, fold)
        if reducer != "add":
            continue
        contribution_inputs, contribution_program = _import_fold_contribution_scalar_map(analysis, path)
        records.append(
            FoldRecord(
                source_contract=chain.second.node,
                contribution_inputs=contribution_inputs,
                contribution_scalar_program=contribution_program,
                generated_contribution_cuda=generate_cuda_scalar_body(
                    contribution_program,
                    symbol="generated_fold_contribution",
                ),
                contribution_opcodes=tuple(
                    analysis.nodes[node_id].opcode
                    for node_id in path[1:-1]
                    if analysis.nodes[node_id].opcode in _DATAFLOW_OPCODES
                ),
                fold=fold.id,
                output_shape=fold.shape,
                reducer=reducer,
                reducer_opcodes=reducer_opcodes,
                reducer_scalar_program=reducer_program,
                generated_reducer_cuda=generate_cuda_scalar_body(
                    reducer_program,
                    symbol="generated_fold_update",
                ),
            )
        )
    return tuple(records)


def _import_fold_contribution_scalar_map(
    analysis: _GraphAnalysis,
    path: tuple[str, ...],
) -> tuple[tuple[str, ...], CastScalarProgram]:
    """Import the scalar contribution before a source-keyed Fold.

    The physical scatter may add singleton update dimensions. Import the last
    rank-two value before those wrappers and expose any rank-two off-path value
    as an ordinary Map input. This covers weighted and unweighted routed
    contributions without assigning workload-specific roles.
    """
    path_without_fold = path[:-1]
    rank_two_nodes = tuple(
        node_id
        for node_id in path_without_fold
        if (shape := _parse_array_shape(analysis.nodes[node_id].shape)) is not None and len(shape[1]) == 2
    )
    if not rank_two_nodes:
        raise ValueError(f"Fold path {path[-1]!r} has no rank-two contribution value")
    target = rank_two_nodes[-1]
    target_shape = _parse_array_shape(analysis.nodes[target].shape)
    assert target_shape is not None
    path_nodes = set(path_without_fold)
    side_inputs: list[str] = []
    for node_id in path_without_fold:
        for operand in analysis.nodes[node_id].operands:
            if operand in path_nodes or analysis.nodes[operand].opcode == "constant" or operand in side_inputs:
                continue
            shape = _parse_array_shape(analysis.nodes[operand].shape)
            if shape is not None and len(shape[1]) == 2 and shape[1] == target_shape[1]:
                side_inputs.append(operand)
    source_nodes = (path_without_fold[0], *side_inputs)
    return source_nodes, import_hlo_scalar_map(
        analysis.graph,
        source_nodes=source_nodes,
        target_node=target,
    )


def _recover_weight_gradients(
    analysis: _GraphAnalysis,
    relation_nodes: tuple[str, ...],
    relation_plans: tuple[RelationPlanRecord, ...],
) -> tuple[RoutedWeightGradientRecord, ...]:
    destination_counts = {plan.destination_count for plan in relation_plans}
    records: list[RoutedWeightGradientRecord] = []
    for node in analysis.graph.nodes:
        shape = _parse_array_shape(node.shape)
        if node.opcode != "dot" or shape is None or len(shape[1]) != 3 or shape[1][0] not in destination_counts:
            continue
        operand_shapes = tuple(_parse_array_shape(analysis.nodes[operand].shape) for operand in node.operands)
        if len(operand_shapes) != 2 or any(value is None or len(value[1]) != 3 for value in operand_shapes):
            continue
        if not any(analysis.is_ancestor(relation, node.id) for relation in relation_nodes):
            continue
        collectives = tuple(user for user in analysis.users[node.id] if analysis.nodes[user].opcode == "all-reduce")
        records.append(
            RoutedWeightGradientRecord(
                contract=_contract_record(
                    node,
                    tuple(relation for relation in relation_nodes if analysis.is_ancestor(relation, node.id)),
                    analysis,
                ),
                external_collectives=collectives,
            )
        )
    return tuple(records)


def _is_destination_sort(node: InlinedHloNode, analysis: _GraphAnalysis) -> bool:
    if node.opcode != "sort" or "is_stable=true" not in node.attributes or len(node.operands) != 2:
        return False
    first = _parse_array_shape(analysis.nodes[node.operands[0]].shape)
    second = _parse_array_shape(analysis.nodes[node.operands[1]].shape)
    return (
        first is not None
        and second is not None
        and first == second
        and first[0] in {"s32", "s64"}
        and len(first[1]) == 1
        and analysis.nodes[node.operands[1]].opcode == "iota"
    )


def _nearest_selection_ancestor(analysis: _GraphAnalysis, node_id: str) -> str | None:
    pending = deque([(node_id, 0)])
    seen = {node_id}
    candidates: list[tuple[int, int, str]] = []
    while pending:
        current, distance = pending.popleft()
        node = analysis.nodes[current]
        if node.opcode in {"sort", "custom-call"} and node.id != node_id and _shape_contains_integer_array(node.shape):
            candidates.append((distance, analysis.order[node.id], node.id))
            continue
        for operand in node.operands:
            if operand not in seen:
                seen.add(operand)
                pending.append((operand, distance + 1))
    return min(candidates)[2] if candidates else None


def _integer_tuple_projection(
    analysis: _GraphAnalysis,
    node_id: str,
    *,
    rank: int,
) -> str | None:
    candidates = []
    for user in analysis.users[node_id]:
        node = analysis.nodes[user]
        shape = _parse_array_shape(node.shape)
        if node.opcode == "get-tuple-element" and shape is not None and shape[0] in {"s32", "s64"}:
            if len(shape[1]) == rank:
                candidates.append(node.id)
    return min(candidates, key=analysis.order.__getitem__) if candidates else None


def _destination_counts(analysis: _GraphAnalysis, selection: str, edge_count: int) -> str:
    candidates = []
    for node in analysis.graph.nodes:
        shape = _parse_array_shape(node.shape)
        if node.opcode != "scatter" or node.dtype not in {"s32", "s64"} or shape is None or len(shape[1]) != 1:
            continue
        if selection not in analysis.ancestors(node.id):
            continue
        if not any(
            (operand_shape := _parse_array_shape(analysis.nodes[operand].shape)) is not None
            and edge_count in operand_shape[1]
            for operand in node.operands
        ):
            continue
        candidates.append(node.id)
    if len(candidates) != 1:
        raise ValueError(
            f"selection {selection!r} has {len(candidates)} structurally compatible destination-count Folds"
        )
    return candidates[0]


def _contract_record(
    node: InlinedHloNode,
    relation_plans: tuple[str, ...],
    analysis: _GraphAnalysis,
) -> SegmentedContractRecord:
    return SegmentedContractRecord(
        node=node.id,
        input_shapes=tuple(analysis.nodes[operand].shape for operand in node.operands),
        output_shape=node.shape,
        relation_plans=relation_plans,
    )


def _nearest_float_scatter(analysis: _GraphAnalysis, node_id: str) -> tuple[str, ...] | None:
    pending = deque([(node_id, (node_id,))])
    seen = {node_id}
    while pending:
        current, path = pending.popleft()
        if len(path) > 48:
            continue
        for user in analysis.users.get(current, ()):
            if user in seen:
                continue
            seen.add(user)
            node = analysis.nodes[user]
            shape = _parse_array_shape(node.shape)
            next_path = (*path, user)
            if (
                node.opcode == "scatter"
                and shape is not None
                and shape[0] in {"bf16", "f16", "f32"}
                and len(shape[1]) == 2
            ):
                return next_path
            if node.opcode in _DATAFLOW_OPCODES:
                pending.append((user, next_path))
    return None


def _scatter_reducer_program(
    module: HloModuleGraph,
    node: InlinedHloNode,
) -> tuple[str, tuple[str, ...], CastScalarProgram]:
    match = _CALLED_COMPUTATION.search(node.attributes)
    if match is None:
        raise ValueError(f"scatter Fold {node.id!r} has no reducer computation")
    computation = module.computation(match.group(1))
    instructions = {instruction.name: instruction for instruction in computation.instructions}
    current = computation.root
    while current.opcode in {"bitcast", "convert", "copy", "reshape"} and len(current.operands) == 1:
        current = instructions[current.operands[0]]
    opcodes = tuple(instruction.opcode for instruction in computation.instructions if instruction.opcode != "parameter")
    return current.opcode, opcodes, import_hlo_scalar_computation(computation)


def _parse_array_shape(shape: str) -> tuple[str, tuple[int, ...]] | None:
    match = _ARRAY_SHAPE.match(shape.lstrip("("))
    if match is None:
        return None
    dims = tuple(int(value) for value in match.group("dims").split(",") if value)
    return match.group("dtype"), dims


def _shape_contains_integer_array(shape: str) -> bool:
    return "s32[" in shape or "s64[" in shape
