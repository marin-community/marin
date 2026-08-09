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

from tile_lifetime.cast_scalar_program import CastScalarProgram, GeneratedCudaScalarBody, generate_cuda_scalar_body
from tile_lifetime.xla_hlo_recovery import (
    HloModuleGraph,
    InlinedHloGraph,
    InlinedHloNode,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_map

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_CALLED_COMPUTATION = re.compile(r"to_apply=%([A-Za-z0-9_.-]+)")
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
class PointwiseMapRecord:
    """Pointwise/shape program between two segmented Contracts."""

    opcodes: tuple[str, ...]
    cast_shapes: tuple[tuple[str, str], ...]
    scalar_program: CastScalarProgram | None
    generated_cuda: GeneratedCudaScalarBody | None


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
    contribution_opcodes: tuple[str, ...]
    fold: str
    output_shape: str
    reducer: str
    reducer_opcodes: tuple[str, ...]


@dataclass(frozen=True)
class RoutedWeightGradientRecord:
    """A group-batched Contract producing one gradient matrix per segment."""

    contract: SegmentedContractRecord
    external_collectives: tuple[str, ...]


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
                        "scalar_program": (
                            chain.map.scalar_program.to_dict() if chain.map.scalar_program is not None else None
                        ),
                        "generated_cuda": (
                            chain.map.generated_cuda.to_dict() if chain.map.generated_cuda is not None else None
                        ),
                    },
                    "second": contract(chain.second),
                }
                for chain in self.contract_chains
            ],
            "folds": [
                {
                    "source_contract": fold.source_contract,
                    "contribution_opcodes": list(fold.contribution_opcodes),
                    "fold": fold.fold,
                    "output_shape": fold.output_shape,
                    "reducer": fold.reducer,
                    "reducer_opcodes": list(fold.reducer_opcodes),
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
            "Forward pair Maps lower to a generic cast-aware scalar AST and generated CUDA device function; "
            "the concatenated input-adjoint Map still has only structural opcode/cast recovery.",
        ),
    )


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
        scalar_program = None
        generated_cuda = None
        if role in {ContractChainRole.FORWARD, ContractChainRole.FORWARD_RECOMPUTE}:
            edge_counts = {
                plan.edge_count for plan in relation_plans if analysis.is_ancestor(plan.destination_sort, first.id)
            }
            if len(edge_counts) != 1:
                raise ValueError(f"Contract {first.id!r} has {len(edge_counts)} routed edge counts")
            scalar_program = _import_forward_scalar_map(
                analysis,
                first=first,
                second=second,
                edge_count=next(iter(edge_counts)),
            )
            generated_cuda = generate_cuda_scalar_body(scalar_program)
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
                    scalar_program=scalar_program,
                    generated_cuda=generated_cuda,
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
) -> CastScalarProgram:
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
    return import_hlo_scalar_map(
        analysis.graph,
        source_node=first.id,
        target_node=terminals[0],
    )


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
        reducer, reducer_opcodes = _scatter_reducer_program(analysis.module, fold)
        if reducer != "add":
            continue
        records.append(
            FoldRecord(
                source_contract=chain.second.node,
                contribution_opcodes=tuple(
                    analysis.nodes[node_id].opcode
                    for node_id in path[1:-1]
                    if analysis.nodes[node_id].opcode in _DATAFLOW_OPCODES
                ),
                fold=fold.id,
                output_shape=fold.shape,
                reducer=reducer,
                reducer_opcodes=reducer_opcodes,
            )
        )
    return tuple(records)


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


def _scatter_reducer_program(module: HloModuleGraph, node: InlinedHloNode) -> tuple[str, tuple[str, ...]]:
    match = _CALLED_COMPUTATION.search(node.attributes)
    if match is None:
        return "unknown", ()
    computation = module.computation(match.group(1))
    instructions = {instruction.name: instruction for instruction in computation.instructions}
    current = computation.root
    while current.opcode in {"bitcast", "convert", "copy", "reshape"} and len(current.operands) == 1:
        current = instructions[current.operands[0]]
    opcodes = tuple(instruction.opcode for instruction in computation.instructions if instruction.opcode != "parameter")
    return current.opcode, opcodes


def _parse_array_shape(shape: str) -> tuple[str, tuple[int, ...]] | None:
    match = _ARRAY_SHAPE.match(shape.lstrip("("))
    if match is None:
        return None
    dims = tuple(int(value) for value in match.group("dims").split(",") if value)
    return match.group("dtype"), dims


def _shape_contains_integer_array(shape: str) -> bool:
    return "s32[" in shape or "s64[" in shape
