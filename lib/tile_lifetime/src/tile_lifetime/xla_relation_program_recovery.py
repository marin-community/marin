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

import numpy as np

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
_LAYOUT = re.compile(r"\{(?P<axes>[0-9,]+)\}$")
_DIMENSIONS = re.compile(r"dimensions=\{(?P<axes>[0-9,]*)\}")
_PADDING = re.compile(r"padding=(?P<dimensions>[0-9_x]+)")
_IOTA_DIMENSION = re.compile(r"iota_dimension=(?P<axis>[0-9]+)")
_COMPARE_DIRECTION = re.compile(r"direction=(?P<direction>[A-Z]+)")
_SCALAR_CONSTANT = re.compile(r"constant\((?P<value>-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\)")
_SINGLETON_CONSTANT = re.compile(r"constant\(\{(?P<value>-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\}\)")
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
    segmented_layout: SegmentedLayoutRecord | None
    segmented_layout_requirement: SegmentedLayoutRequirement | None


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
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SegmentedPhysicalIndex:
    """One logical routed scalar mapped onto a physical Contract operand."""

    physical_row: int
    physical_k: int
    valid: bool


@dataclass(frozen=True)
class SegmentedLayoutIndexMap:
    """Affine/runtime map from edge, feature, and segment to Contract coordinates."""

    logical_edge_count: int
    logical_feature_extent: int
    segment_count: int
    padded_row_extent: int
    row_stride: int
    row_offset: int
    feature_stride: int
    segment_stride: int

    def physical_index(
        self,
        *,
        edge_row: int,
        feature: int,
        segment: int,
        segment_ends: tuple[int, ...],
    ) -> SegmentedPhysicalIndex:
        """Evaluate the recovered finite index relation."""
        if edge_row < 0 or edge_row >= self.logical_edge_count:
            raise ValueError("edge row lies outside the compact relation")
        if feature < 0 or feature >= self.logical_feature_extent:
            raise ValueError("feature lies outside the logical Map output")
        if segment < 0 or segment >= self.segment_count:
            raise ValueError("segment lies outside the relation destination domain")
        if len(segment_ends) != self.segment_count:
            raise ValueError("segment ends do not match the destination domain")
        starts = (0, *segment_ends[:-1])
        return SegmentedPhysicalIndex(
            physical_row=edge_row * self.row_stride + self.row_offset,
            physical_k=feature * self.feature_stride + segment * self.segment_stride,
            valid=starts[segment] <= edge_row < segment_ends[segment],
        )


@dataclass(frozen=True)
class SourceFoldInverseIndexMap:
    """Map a destination-major edge row back to source and route-slot order."""

    stable_permutation: str
    fold_indices: str
    source_item_divisor: int
    route_slot_modulus: int

    def source_coordinate(
        self,
        destination_edge_row: int,
        permutation: tuple[int, ...],
    ) -> tuple[int, int]:
        """Evaluate the recovered inverse relation for one compact edge row."""
        if destination_edge_row < 0 or destination_edge_row >= len(permutation):
            raise ValueError("destination edge row lies outside the stable permutation")
        route = permutation[destination_edge_row]
        return route // self.source_item_divisor, route % self.route_slot_modulus


@dataclass(frozen=True)
class SegmentedLayoutProof:
    """HLO dataflow evidence for one recovered index relation."""

    relation: SegmentedLayoutRelation
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class SegmentedLayoutRecord:
    """Verified relation-driven layout between logical and physical Contracts."""

    relation_plan: str
    index_map: SegmentedLayoutIndexMap
    segment_ends: str
    fill_value: float
    physical_shape: str
    weight_shape: str
    inverse: SourceFoldInverseIndexMap
    proofs: tuple[SegmentedLayoutProof, ...]

    @property
    def verified_relations(self) -> tuple[SegmentedLayoutRelation, ...]:
        """Return the relations whose HLO derivations were verified."""
        return tuple(proof.relation for proof in self.proofs)

    @property
    def runtime_index_inputs(self) -> tuple[str, str]:
        """Return relation metadata that generated execution must receive at runtime."""
        return self.segment_ends, self.inverse.stable_permutation


@dataclass(frozen=True)
class RoutedForwardTypedFfiCodegenPlan:
    """A generic typed-FFI plan or a structured missing-information result."""

    region: RoutedForwardRegionRecord
    api_version: int
    contracts: tuple[RoutedForwardContractStage, ...]
    map_stage: RoutedForwardMapStage
    fold_stage: RoutedForwardFoldStage
    segmented_layout: SegmentedLayoutRecord | None
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
    segmented_layout, segmented_layout_requirement = _recover_segmented_layout(
        module=module,
        graph=graph,
        relation_plan=relation_plan,
        layout_path=layout_path,
        first_contract_value=first_node.operands[0],
        physical_value=physical_map_output,
        weight_value=second_node.operands[1],
        fold_value=fold.fold,
        logical_feature_extent=logical_feature_extent,
        observed_path_opcodes=tuple(graph.node(node_id).opcode for node_id in layout_path[1:]),
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
        has_segmented_layout=segmented_layout is not None,
        segmented_layout=segmented_layout,
        segmented_layout_requirement=segmented_layout_requirement,
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
    segmented_layout = region.map_stage.segmented_layout
    all_relations = tuple(SegmentedLayoutRelation)
    ready = segmented_layout is not None and segmented_layout.verified_relations == all_relations
    disposition = (
        RoutedForwardCodegenDisposition.READY if ready else RoutedForwardCodegenDisposition.MISSING_SEGMENTED_LAYOUT
    )
    missing_segmented_layout = None if ready else region.map_stage.segmented_layout_requirement
    if not ready and missing_segmented_layout is None:
        raise ValueError("segmented layout recovery failed without a structured rejection")
    return RoutedForwardTypedFfiCodegenPlan(
        region=region,
        api_version=1,
        contracts=region.contracts,
        map_stage=region.map_stage,
        fold_stage=region.fold_stage,
        segmented_layout=segmented_layout,
        disposition=disposition,
        missing_segmented_layout=missing_segmented_layout,
    )


@dataclass(frozen=True)
class _PhysicalArrayShape:
    dtype: str
    dimensions: tuple[int, ...]
    minor_to_major: tuple[int, ...]


@dataclass(frozen=True)
class _SegmentedMapLayoutTrace:
    logical_value: str
    pad: str
    broadcast: str
    select: str
    transpose: str
    copy: str
    bitcast: str
    predicate: str
    logical_edge_count: int
    logical_feature_extent: int
    segment_count: int
    padded_row_extent: int
    flattened_pair_order: tuple[str, str]


class _SegmentedLayoutRecoveryError(ValueError):
    def __init__(self, relation: SegmentedLayoutRelation, reason: str):
        self.relation = relation
        self.reason = reason
        super().__init__(reason)


def _recover_segmented_layout(
    *,
    module: HloModuleGraph,
    graph: InlinedHloGraph,
    relation_plan: RelationPlanRecord,
    layout_path: tuple[str, ...],
    first_contract_value: str,
    physical_value: str,
    weight_value: str,
    fold_value: str,
    logical_feature_extent: int,
    observed_path_opcodes: tuple[str, ...],
) -> tuple[SegmentedLayoutRecord | None, SegmentedLayoutRequirement | None]:
    """Recover the physical relation from HLO index and layout operations."""
    physical_shape = graph.node(physical_value).shape
    missing: list[tuple[SegmentedLayoutRelation, str]] = []
    trace: _SegmentedMapLayoutTrace | None = None
    try:
        trace = _segmented_map_layout_trace(
            graph,
            relation_plan=relation_plan,
            layout_path=layout_path,
            physical_value=physical_value,
            logical_feature_extent=logical_feature_extent,
        )
    except _SegmentedLayoutRecoveryError as error:
        missing.extend(
            (relation, error.reason)
            for relation in (
                SegmentedLayoutRelation.EDGE_ROW_TO_PADDED_ROW,
                SegmentedLayoutRelation.SEGMENT_TO_FEATURE_PANEL,
                SegmentedLayoutRelation.VALIDITY_AND_FILL,
            )
        )

    feature_proof: SegmentedLayoutProof | None = None
    edge_order_proof: SegmentedLayoutProof | None = None
    validity_proof: SegmentedLayoutProof | None = None
    fill_value = 0.0
    if trace is not None:
        try:
            edge_order_proof = _verify_first_contract_edge_order(
                graph,
                relation_plan=relation_plan,
                first_contract_value=first_contract_value,
            )
        except _SegmentedLayoutRecoveryError as error:
            missing.append((error.relation, error.reason))
        try:
            feature_proof = _verify_segmented_weight_flattening(
                graph,
                trace=trace,
                weight_value=weight_value,
            )
        except _SegmentedLayoutRecoveryError as error:
            missing.append((error.relation, error.reason))
        try:
            validity_proof, fill_value = _verify_segmented_validity(
                module,
                graph,
                relation_plan=relation_plan,
                trace=trace,
            )
        except _SegmentedLayoutRecoveryError as error:
            missing.append((error.relation, error.reason))

    inverse: SourceFoldInverseIndexMap | None = None
    inverse_proof: SegmentedLayoutProof | None = None
    try:
        inverse, inverse_proof = _recover_source_fold_inverse(
            graph,
            relation_plan=relation_plan,
            fold_value=fold_value,
        )
    except _SegmentedLayoutRecoveryError as error:
        missing.append((error.relation, error.reason))

    if (
        missing
        or trace is None
        or edge_order_proof is None
        or feature_proof is None
        or validity_proof is None
        or inverse is None
    ):
        unique_missing: dict[SegmentedLayoutRelation, str] = {}
        for relation, reason in missing:
            unique_missing.setdefault(relation, reason)
        return None, SegmentedLayoutRequirement(
            logical_shape=(relation_plan.edge_count, logical_feature_extent),
            physical_shape=physical_shape,
            observed_path_opcodes=observed_path_opcodes,
            required_relations=tuple(relation for relation in SegmentedLayoutRelation if relation in unique_missing),
            reasons=tuple(
                unique_missing[relation] for relation in SegmentedLayoutRelation if relation in unique_missing
            ),
        )

    assert inverse_proof is not None
    pair_strides = {
        ("feature", "segment"): (trace.segment_count, 1),
        ("segment", "feature"): (1, trace.logical_feature_extent),
    }
    feature_stride, segment_stride = pair_strides[trace.flattened_pair_order]
    edge_proof = SegmentedLayoutProof(
        relation=SegmentedLayoutRelation.EDGE_ROW_TO_PADDED_ROW,
        nodes=(
            *edge_order_proof.nodes,
            trace.logical_value,
            trace.pad,
            trace.broadcast,
            trace.transpose,
            trace.copy,
            trace.bitcast,
        ),
    )
    return (
        SegmentedLayoutRecord(
            relation_plan=relation_plan.destination_sort,
            index_map=SegmentedLayoutIndexMap(
                logical_edge_count=trace.logical_edge_count,
                logical_feature_extent=trace.logical_feature_extent,
                segment_count=trace.segment_count,
                padded_row_extent=trace.padded_row_extent,
                row_stride=1,
                row_offset=0,
                feature_stride=feature_stride,
                segment_stride=segment_stride,
            ),
            segment_ends=relation_plan.destination_offsets,
            fill_value=fill_value,
            physical_shape=physical_shape,
            weight_shape=graph.node(weight_value).shape,
            inverse=inverse,
            proofs=(edge_proof, feature_proof, validity_proof, inverse_proof),
        ),
        None,
    )


def _segmented_map_layout_trace(
    graph: InlinedHloGraph,
    *,
    relation_plan: RelationPlanRecord,
    layout_path: tuple[str, ...],
    physical_value: str,
    logical_feature_extent: int,
) -> _SegmentedMapLayoutTrace:
    relation = SegmentedLayoutRelation.EDGE_ROW_TO_PADDED_ROW
    path_nodes = tuple(graph.node(node_id) for node_id in layout_path)
    pads = tuple(node for node in path_nodes if node.opcode == "pad")
    if len(pads) != 1 or len(pads[0].operands) != 2:
        raise _SegmentedLayoutRecoveryError(relation, f"layout path has {len(pads)} candidate pads")
    pad = pads[0]
    logical = _physical_array_shape(graph.node(pad.operands[0]).shape)
    padded = _physical_array_shape(pad.shape)
    expected_logical = (relation_plan.edge_count, logical_feature_extent)
    if logical is None or logical.dimensions != expected_logical:
        raise _SegmentedLayoutRecoveryError(relation, "pad input is not the logical edge-by-feature Map result")
    if padded is None or len(padded.dimensions) != 2 or padded.dimensions[1] != logical_feature_extent:
        raise _SegmentedLayoutRecoveryError(relation, "pad output does not preserve the feature axis")
    padding = _padding_pairs(pad.attributes)
    expected_padding = ((0, padded.dimensions[0] - relation_plan.edge_count), (0, 0))
    if padding != expected_padding:
        raise _SegmentedLayoutRecoveryError(relation, "pad does not embed compact edge rows at the physical row origin")
    pad_fill = _scalar_constant(graph.node(pad.operands[1]))
    if pad_fill != 0.0:
        raise _SegmentedLayoutRecoveryError(relation, "compact edge-row padding is not zero-filled")

    broadcasts = tuple(
        node
        for node in path_nodes
        if node.opcode == "broadcast"
        and (shape := _physical_array_shape(node.shape)) is not None
        and len(shape.dimensions) == 3
        and shape.dimensions[1:] == padded.dimensions
        and _attribute_axes(node.attributes) == (1, 2)
    )
    if len(broadcasts) != 1:
        raise _SegmentedLayoutRecoveryError(relation, f"layout path has {len(broadcasts)} row-preserving broadcasts")
    broadcast = broadcasts[0]
    broadcast_shape = _physical_array_shape(broadcast.shape)
    assert broadcast_shape is not None
    segment_count = broadcast_shape.dimensions[0]
    if segment_count != relation_plan.destination_count:
        raise _SegmentedLayoutRecoveryError(relation, "broadcast segment axis differs from the RelationPlan domain")

    selects = tuple(
        node
        for node in path_nodes
        if node.opcode == "select" and len(node.operands) == 3 and node.operands[1] == broadcast.id
    )
    if len(selects) != 1:
        raise _SegmentedLayoutRecoveryError(relation, f"layout path has {len(selects)} segment-validity selects")
    select = selects[0]
    transposes = tuple(
        node for node in path_nodes if node.opcode == "transpose" and select.id in _node_ancestors(graph, node.id)
    )
    if len(transposes) != 1:
        raise _SegmentedLayoutRecoveryError(relation, f"layout path has {len(transposes)} flattening transposes")
    transpose = transposes[0]
    transpose_axes = _attribute_axes(transpose.attributes)
    pair_orders = {
        (1, 2, 0): ("feature", "segment"),
        (1, 0, 2): ("segment", "feature"),
    }
    if transpose_axes not in pair_orders:
        raise _SegmentedLayoutRecoveryError(relation, f"unsupported row-preserving transpose {transpose_axes}")
    transpose_shape = _physical_array_shape(transpose.shape)
    if transpose_shape is None or transpose_shape.dimensions[0] != padded.dimensions[0]:
        raise _SegmentedLayoutRecoveryError(relation, "transpose does not retain padded rows as its leading axis")

    transpose_position = layout_path.index(transpose.id)
    suffix = tuple(graph.node(node_id) for node_id in layout_path[transpose_position + 1 :])
    if len(suffix) != 2 or suffix[0].opcode != "copy" or suffix[1].opcode != "bitcast":
        raise _SegmentedLayoutRecoveryError(relation, "flattening is not an explicit transpose-copy-bitcast chain")
    copy, bitcast = suffix
    copy_shape = _physical_array_shape(copy.shape)
    bitcast_shape = _physical_array_shape(bitcast.shape)
    if copy_shape is None or copy_shape.minor_to_major != (2, 1, 0):
        raise _SegmentedLayoutRecoveryError(relation, "flattening copy does not make the trailing pair contiguous")
    expected_physical = (padded.dimensions[0], logical_feature_extent * segment_count)
    if (
        bitcast.id != physical_value
        or bitcast_shape is None
        or bitcast_shape.dimensions != expected_physical
        or bitcast_shape.minor_to_major != (1, 0)
    ):
        raise _SegmentedLayoutRecoveryError(
            relation, "bitcast does not flatten the segment/feature pair into Contract K"
        )
    return _SegmentedMapLayoutTrace(
        logical_value=pad.operands[0],
        pad=pad.id,
        broadcast=broadcast.id,
        select=select.id,
        transpose=transpose.id,
        copy=copy.id,
        bitcast=bitcast.id,
        predicate=select.operands[0],
        logical_edge_count=relation_plan.edge_count,
        logical_feature_extent=logical_feature_extent,
        segment_count=segment_count,
        padded_row_extent=padded.dimensions[0],
        flattened_pair_order=pair_orders[transpose_axes],
    )


def _verify_segmented_weight_flattening(
    graph: InlinedHloGraph,
    *,
    trace: _SegmentedMapLayoutTrace,
    weight_value: str,
) -> SegmentedLayoutProof:
    relation = SegmentedLayoutRelation.SEGMENT_TO_FEATURE_PANEL
    bitcast = graph.node(weight_value)
    if bitcast.opcode != "bitcast" or len(bitcast.operands) != 1:
        raise _SegmentedLayoutRecoveryError(relation, "Contract weight is not an explicit flattened bitcast")
    copy = graph.node(bitcast.operands[0])
    transpose = graph.node(copy.operands[0]) if copy.opcode == "copy" and len(copy.operands) == 1 else None
    if transpose is None or transpose.opcode != "transpose" or len(transpose.operands) != 1:
        raise _SegmentedLayoutRecoveryError(relation, "Contract weight is not a transpose-copy-bitcast layout")
    source = graph.node(transpose.operands[0])
    source_shape = _physical_array_shape(source.shape)
    transpose_shape = _physical_array_shape(transpose.shape)
    copy_shape = _physical_array_shape(copy.shape)
    bitcast_shape = _physical_array_shape(bitcast.shape)
    if source_shape is None or len(source_shape.dimensions) != 3:
        raise _SegmentedLayoutRecoveryError(relation, "weight source is not segment-by-feature-by-output")
    segment_count, feature_extent, output_extent = source_shape.dimensions
    if (segment_count, feature_extent) != (trace.segment_count, trace.logical_feature_extent):
        raise _SegmentedLayoutRecoveryError(relation, "weight segment/feature axes disagree with the Map layout")
    weight_axes = _attribute_axes(transpose.attributes)
    pair_orders = {
        (1, 0, 2): ("feature", "segment"),
        (0, 1, 2): ("segment", "feature"),
    }
    if weight_axes not in pair_orders or pair_orders[weight_axes] != trace.flattened_pair_order:
        raise _SegmentedLayoutRecoveryError(relation, "Map and weight flatten segment/feature in different orders")
    expected_transpose = tuple(source_shape.dimensions[axis] for axis in weight_axes)
    expected_bitcast = (trace.logical_feature_extent * trace.segment_count, output_extent)
    if (
        transpose_shape is None
        or transpose_shape.dimensions != expected_transpose
        or copy_shape is None
        or copy_shape.dimensions != expected_transpose
        or copy_shape.minor_to_major != (2, 1, 0)
        or bitcast_shape is None
        or bitcast_shape.dimensions != expected_bitcast
        or bitcast_shape.minor_to_major != (1, 0)
    ):
        raise _SegmentedLayoutRecoveryError(
            relation, "weight physical layouts do not legalize the recovered K index map"
        )
    return SegmentedLayoutProof(relation=relation, nodes=(transpose.id, copy.id, bitcast.id))


def _verify_first_contract_edge_order(
    graph: InlinedHloGraph,
    *,
    relation_plan: RelationPlanRecord,
    first_contract_value: str,
) -> SegmentedLayoutProof:
    relation = SegmentedLayoutRelation.EDGE_ROW_TO_PADDED_ROW
    path = _inlined_path(graph, relation_plan.stable_permutation, first_contract_value)
    gathers = tuple(
        graph.node(node_id)
        for node_id in path
        if graph.node(node_id).opcode == "gather" and graph.node(node_id).dtype in {"bf16", "f16", "f32", "f64"}
    )
    if len(gathers) != 1:
        raise _SegmentedLayoutRecoveryError(
            relation,
            f"first Contract relation path has {len(gathers)} payload gathers",
        )
    gather = gathers[0]
    index_operands = tuple(
        operand
        for operand in gather.operands
        if (shape := _parse_array_shape(graph.node(operand).shape)) is not None and shape[0] in {"s32", "s64"}
    )
    if len(index_operands) != 1:
        raise _SegmentedLayoutRecoveryError(relation, "first Contract payload gather has ambiguous indices")
    edge_count = relation_plan.edge_count
    permutations = (
        np.arange(edge_count, dtype=np.int32),
        np.arange(edge_count - 1, -1, -1, dtype=np.int32),
        np.roll(np.arange(edge_count, dtype=np.int32), 3),
    )
    expected_shape = (edge_count, 1)
    for permutation in permutations:
        try:
            observed = _evaluate_integer_node(
                graph,
                index_operands[0],
                {relation_plan.stable_permutation: permutation},
            )
        except ValueError as error:
            raise _SegmentedLayoutRecoveryError(
                relation,
                f"could not evaluate first Contract edge order: {error}",
            ) from error
        expected = (permutation // relation_plan.slots_per_token).reshape(expected_shape)
        if observed.shape != expected_shape or not np.array_equal(observed, expected):
            raise _SegmentedLayoutRecoveryError(
                relation,
                "first Contract row r does not gather source stable_permutation[r] // route_slots",
            )
    return SegmentedLayoutProof(relation=relation, nodes=path)


def _verify_segmented_validity(
    module: HloModuleGraph,
    graph: InlinedHloGraph,
    *,
    relation_plan: RelationPlanRecord,
    trace: _SegmentedMapLayoutTrace,
) -> tuple[SegmentedLayoutProof, float]:
    relation = SegmentedLayoutRelation.VALIDITY_AND_FILL
    predicate = graph.node(trace.predicate)
    if predicate.opcode != "and" or len(predicate.operands) != 2:
        raise _SegmentedLayoutRecoveryError(relation, "segment validity is not an intersection of two bounds")
    comparisons = tuple(graph.node(operand) for operand in predicate.operands)
    lower = next((node for node in comparisons if _compare_direction(node) == "LE"), None)
    upper = next((node for node in comparisons if _compare_direction(node) == "LT"), None)
    if lower is None or upper is None or len(lower.operands) != 2 or len(upper.operands) != 2:
        raise _SegmentedLayoutRecoveryError(relation, "segment validity does not contain lower-LE and upper-LT bounds")
    iota_candidates = {lower.operands[1], upper.operands[0]}
    if len(iota_candidates) != 1:
        raise _SegmentedLayoutRecoveryError(relation, "segment bounds do not compare the same padded-row index")
    iota = graph.node(next(iter(iota_candidates)))
    iota_axis = _IOTA_DIMENSION.search(iota.attributes)
    if iota.opcode != "iota" or iota_axis is None or int(iota_axis.group("axis")) != 1:
        raise _SegmentedLayoutRecoveryError(relation, "segment predicate is not indexed by the padded-row axis")
    if not _is_segment_upper_bound(graph, upper.operands[1], relation_plan.destination_offsets):
        raise _SegmentedLayoutRecoveryError(relation, "upper segment bound is not the recovered inclusive prefix end")
    if not _is_segment_lower_bound(
        graph,
        lower.operands[0],
        relation_plan.destination_offsets,
        trace.segment_count,
    ):
        raise _SegmentedLayoutRecoveryError(relation, "lower segment bound is not zero plus prior prefix ends")
    if not _is_inclusive_prefix_sum(module, graph, relation_plan):
        raise _SegmentedLayoutRecoveryError(relation, "RelationPlan segment ends are not an inclusive count prefix")
    select = graph.node(trace.select)
    fill = _broadcast_scalar_constant(graph, select.operands[2])
    if fill is None or fill != 0.0:
        raise _SegmentedLayoutRecoveryError(relation, "invalid segment/row coordinates are not zero-filled")
    return (
        SegmentedLayoutProof(
            relation=relation,
            nodes=(relation_plan.destination_counts, relation_plan.destination_offsets, predicate.id, select.id),
        ),
        fill,
    )


def _recover_source_fold_inverse(
    graph: InlinedHloGraph,
    *,
    relation_plan: RelationPlanRecord,
    fold_value: str,
) -> tuple[SourceFoldInverseIndexMap, SegmentedLayoutProof]:
    relation = SegmentedLayoutRelation.SOURCE_FOLD_INVERSE
    fold = graph.node(fold_value)
    if fold.opcode != "scatter" or len(fold.operands) < 2:
        raise _SegmentedLayoutRecoveryError(relation, "source Fold is not an indexed scatter")
    fold_indices = fold.operands[1]
    edge_count = relation_plan.edge_count
    permutations = (
        np.arange(edge_count, dtype=np.int32),
        np.arange(edge_count - 1, -1, -1, dtype=np.int32),
        np.roll(np.arange(edge_count, dtype=np.int32), 3),
    )
    expected_shape = (edge_count, 1)
    for permutation in permutations:
        try:
            observed = _evaluate_integer_node(
                graph,
                fold_indices,
                {relation_plan.stable_permutation: permutation},
            )
        except ValueError as error:
            raise _SegmentedLayoutRecoveryError(
                relation, f"could not evaluate source Fold index path: {error}"
            ) from error
        expected = (permutation // relation_plan.slots_per_token).reshape(expected_shape)
        if observed.shape != expected_shape or not np.array_equal(observed, expected):
            raise _SegmentedLayoutRecoveryError(
                relation,
                "source Fold indices do not equal stable_permutation[row] // route_slots",
            )
    path = _inlined_path(graph, relation_plan.stable_permutation, fold_indices)
    inverse = SourceFoldInverseIndexMap(
        stable_permutation=relation_plan.stable_permutation,
        fold_indices=fold_indices,
        source_item_divisor=relation_plan.slots_per_token,
        route_slot_modulus=relation_plan.slots_per_token,
    )
    return inverse, SegmentedLayoutProof(relation=relation, nodes=path)


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


def _physical_array_shape(shape: str) -> _PhysicalArrayShape | None:
    parsed = _parse_array_shape(shape)
    layout_match = _LAYOUT.search(shape)
    if parsed is None or layout_match is None:
        return None
    return _PhysicalArrayShape(
        dtype=parsed[0],
        dimensions=parsed[1],
        minor_to_major=tuple(int(axis) for axis in layout_match.group("axes").split(",")),
    )


def _attribute_axes(attributes: str) -> tuple[int, ...]:
    match = _DIMENSIONS.search(attributes)
    if match is None:
        return ()
    return tuple(int(axis) for axis in match.group("axes").split(",") if axis)


def _padding_pairs(attributes: str) -> tuple[tuple[int, int], ...]:
    match = _PADDING.search(attributes)
    if match is None:
        return ()
    pairs = []
    for dimension in match.group("dimensions").split("x"):
        values = tuple(int(value) for value in dimension.split("_") if value)
        if len(values) != 2:
            return ()
        pairs.append(values)
    return tuple(pairs)


def _scalar_constant(node: InlinedHloNode) -> float | None:
    if node.opcode != "constant":
        return None
    match = _SCALAR_CONSTANT.search(node.attributes) or _SINGLETON_CONSTANT.search(node.attributes)
    return float(match.group("value")) if match is not None else None


def _broadcast_scalar_constant(graph: InlinedHloGraph, node_id: str) -> float | None:
    current = graph.node(node_id)
    while current.opcode in {"broadcast", "bitcast", "convert", "copy", "reshape"} and len(current.operands) == 1:
        current = graph.node(current.operands[0])
    return _scalar_constant(current)


def _compare_direction(node: InlinedHloNode) -> str | None:
    if node.opcode != "compare":
        return None
    match = _COMPARE_DIRECTION.search(node.attributes)
    return match.group("direction") if match is not None else None


def _node_ancestors(graph: InlinedHloGraph, node_id: str) -> frozenset[str]:
    nodes = {node.id: node for node in graph.nodes}
    ancestors = {node_id}
    pending = list(nodes[node_id].operands)
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(nodes[current].operands)
    return frozenset(ancestors)


def _strip_unary_wrappers(graph: InlinedHloGraph, node_id: str) -> str:
    current = graph.node(node_id)
    while current.opcode in {"bitcast", "convert", "copy", "reshape"} and len(current.operands) == 1:
        current = graph.node(current.operands[0])
    return current.id


def _is_segment_upper_bound(graph: InlinedHloGraph, node_id: str, segment_ends: str) -> bool:
    node = graph.node(node_id)
    if node.opcode != "broadcast" or _attribute_axes(node.attributes) != (0,) or len(node.operands) != 1:
        return False
    return _strip_unary_wrappers(graph, node.operands[0]) == _strip_unary_wrappers(graph, segment_ends)


def _is_segment_lower_bound(
    graph: InlinedHloGraph,
    node_id: str,
    segment_ends: str,
    segment_count: int,
) -> bool:
    node = graph.node(node_id)
    if node.opcode != "broadcast" or _attribute_axes(node.attributes) != (0,) or len(node.operands) != 1:
        return False
    concatenate = graph.node(_strip_unary_wrappers(graph, node.operands[0]))
    if concatenate.opcode != "concatenate" or _attribute_axes(concatenate.attributes) != (0,):
        return False
    if len(concatenate.operands) != 2:
        return False
    zero = _broadcast_scalar_constant(graph, concatenate.operands[0])
    sliced = graph.node(_strip_unary_wrappers(graph, concatenate.operands[1]))
    if zero != 0.0 or sliced.opcode != "slice" or len(sliced.operands) != 1:
        return False
    sliced_shape = _parse_array_shape(sliced.shape)
    if sliced_shape is None or sliced_shape[1] not in {(segment_count - 1,), (segment_count - 1, 1)}:
        return False
    return _strip_unary_wrappers(graph, sliced.operands[0]) == _strip_unary_wrappers(graph, segment_ends)


def _is_inclusive_prefix_sum(
    module: HloModuleGraph,
    graph: InlinedHloGraph,
    relation_plan: RelationPlanRecord,
) -> bool:
    offsets = graph.node(relation_plan.destination_offsets)
    if offsets.opcode != "reduce-window" or len(offsets.operands) < 1:
        return False
    if relation_plan.destination_counts not in _node_ancestors(graph, offsets.id):
        return False
    window = re.search(r"window=\{size=(?P<size>[0-9]+)x1 pad=(?P<low>[0-9]+)_0x0_0\}", offsets.attributes)
    if window is None:
        return False
    if int(window.group("size")) != relation_plan.destination_count:
        return False
    if int(window.group("low")) != relation_plan.destination_count - 1:
        return False
    reducer = _CALLED_COMPUTATION.search(offsets.attributes)
    if reducer is None:
        return False
    computation = module.computation(reducer.group(1))
    instructions = {instruction.name: instruction for instruction in computation.instructions}
    root = computation.root
    while root.opcode in {"bitcast", "convert", "copy", "reshape"} and len(root.operands) == 1:
        root = instructions[root.operands[0]]
    return root.opcode == "add"


def _evaluate_integer_node(
    graph: InlinedHloGraph,
    node_id: str,
    bindings: dict[str, np.ndarray],
) -> np.ndarray:
    nodes = {node.id: node for node in graph.nodes}
    cache: dict[str, np.ndarray] = {name: np.asarray(value) for name, value in bindings.items()}

    def evaluate(current_id: str) -> np.ndarray:
        if current_id in cache:
            return cache[current_id]
        node = nodes[current_id]
        shape = _parse_array_shape(node.shape)
        if shape is None:
            raise ValueError(f"integer expression {current_id!r} has non-array shape {node.shape!r}")
        dimensions = shape[1]
        operands = tuple(evaluate(operand) for operand in node.operands)
        if node.opcode == "constant":
            value = _scalar_constant(node)
            if value is None:
                raise ValueError(f"unsupported integer constant {node.attributes!r}")
            result = np.asarray(int(value), dtype=np.int32)
        elif node.opcode == "iota":
            match = _IOTA_DIMENSION.search(node.attributes)
            if match is None:
                raise ValueError(f"iota {current_id!r} has no dimension")
            axis = int(match.group("axis"))
            base = np.arange(dimensions[axis], dtype=np.int32)
            reshape = [1] * len(dimensions)
            reshape[axis] = dimensions[axis]
            result = np.broadcast_to(base.reshape(reshape), dimensions)
        elif node.opcode == "broadcast":
            axes = _attribute_axes(node.attributes)
            if len(axes) != operands[0].ndim:
                raise ValueError(f"broadcast {current_id!r} has incompatible dimensions")
            reshape = [1] * len(dimensions)
            for operand_axis, result_axis in enumerate(axes):
                reshape[result_axis] = operands[0].shape[operand_axis]
            result = np.broadcast_to(operands[0].reshape(reshape), dimensions)
        elif node.opcode in {"bitcast", "copy", "reshape", "convert"}:
            result = operands[0].reshape(dimensions)
        elif node.opcode == "sign":
            result = np.sign(operands[0])
        elif node.opcode == "and":
            result = np.bitwise_and(operands[0], operands[1])
        elif node.opcode == "shift-right-logical":
            result = np.right_shift(operands[0].astype(np.uint32), operands[1].astype(np.uint32)).astype(np.int32)
        elif node.opcode == "add":
            result = operands[0] + operands[1]
        elif node.opcode == "subtract":
            result = operands[0] - operands[1]
        elif node.opcode == "compare":
            directions = {
                "LT": np.less,
                "LE": np.less_equal,
                "GT": np.greater,
                "GE": np.greater_equal,
                "EQ": np.equal,
                "NE": np.not_equal,
            }
            direction = _compare_direction(node)
            if direction not in directions:
                raise ValueError(f"unsupported comparison {direction!r}")
            result = directions[direction](operands[0], operands[1])
        elif node.opcode == "select":
            result = np.where(operands[0], operands[1], operands[2])
        elif node.opcode == "gather":
            if len(operands) != 2 or operands[1].ndim != 2 or operands[1].shape[1] != 1:
                raise ValueError(f"unsupported gather shape at {current_id!r}")
            result = np.take(operands[0], operands[1][:, 0], axis=0)
            if result.ndim == 1 and dimensions == (result.shape[0], 1):
                result = result[:, None]
        else:
            raise ValueError(f"unsupported integer opcode {node.opcode!r} at {current_id!r}")
        result = np.asarray(result)
        if result.shape != dimensions:
            result = result.reshape(dimensions)
        cache[current_id] = result
        return result

    return evaluate(node_id)


def _parse_array_shape(shape: str) -> tuple[str, tuple[int, ...]] | None:
    match = _ARRAY_SHAPE.match(shape.lstrip("("))
    if match is None:
        return None
    dims = tuple(int(value) for value in match.group("dims").split(",") if value)
    return match.group("dtype"), dims


def _shape_contains_integer_array(shape: str) -> bool:
    return "s32[" in shape or "s64[" in shape
