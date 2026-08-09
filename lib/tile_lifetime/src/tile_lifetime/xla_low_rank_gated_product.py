# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover low-rank gated-product training chains from physical XLA HLO."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from tile_lifetime.cast_scalar_program import CastScalarProgram
from tile_lifetime.xla_hlo_recovery import (
    EntryRegionValue,
    HloComputation,
    InlinedHloGraph,
    InlinedHloNode,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_map

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_DOT_DIMENSION = re.compile(
    r"(?P<name>lhs_contracting_dims|rhs_contracting_dims|lhs_batch_dims|rhs_batch_dims)=" r"\{(?P<dims>[0-9,]*)\}"
)
_TUPLE_INDEX = re.compile(r"index=(?P<index>[0-9]+)")
_SCALAR_DATAFLOW = frozenset(
    {
        "add",
        "bitcast",
        "broadcast",
        "convert",
        "copy",
        "divide",
        "exponential",
        "multiply",
        "negate",
        "reshape",
        "select",
        "subtract",
        "tanh",
    }
)
_LAYOUT_WRAPPERS = frozenset({"bitcast", "copy", "reshape", "transpose"})


@dataclass(frozen=True)
class RankTwoContractPlan:
    """One physical rank-two Contract with explicit finite-precision types."""

    instruction: str
    lhs: EntryRegionValue
    rhs: EntryRegionValue
    output: EntryRegionValue
    lhs_contracting_dimension: int
    rhs_contracting_dimension: int
    flops: int


@dataclass(frozen=True)
class LowRankGatedProductForwardPlan:
    """Two Contracts and generated scalar Maps forming one gated product."""

    input: EntryRegionValue
    down_contract: RankTwoContractPlan
    hidden_map: CastScalarProgram
    hidden: EntryRegionValue
    up_contract: RankTwoContractPlan
    output_map: CastScalarProgram
    output: EntryRegionValue
    parameter_origins: tuple[str, str]


@dataclass(frozen=True)
class LowRankGatedProductReversePlan:
    """JAX-owned VJP structure for one low-rank gated-product family."""

    primal: LowRankGatedProductForwardPlan
    up_input_map: CastScalarProgram
    up_input_sources: tuple[EntryRegionValue, ...]
    upstream_collectives: tuple[EntryRegionValue, ...]
    up_input_adjoint: RankTwoContractPlan
    hidden_vjp_map: CastScalarProgram
    down_input_adjoint: RankTwoContractPlan
    residual_vjp_map: CastScalarProgram
    input_adjoint: EntryRegionValue
    down_weight_adjoint: RankTwoContractPlan
    up_weight_adjoint: RankTwoContractPlan


@dataclass(frozen=True)
class LowRankGatedProductTrainingReport:
    """All structurally recovered repeated training families and live work."""

    forward_realizations: tuple[LowRankGatedProductForwardPlan, ...]
    reverse_families: tuple[LowRankGatedProductReversePlan, ...]
    live_contract_count: int
    live_contract_flops: int
    owned_contract_count: int
    owned_contract_flops: int


def recover_low_rank_gated_product_training(
    hlo_text: str,
) -> LowRankGatedProductTrainingReport:
    """Recover repeated generic chains without instruction names or metadata."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    graph = inline_elementwise_fusions(module)
    nodes = {node.id: node for node in graph.nodes}
    reachable_entry = _reachable_entry_instructions(entry)
    reachable_nodes = {
        graph.entry_value(instruction.name) for instruction in entry.instructions if instruction.name in reachable_entry
    }
    users = _users(graph)
    forward = _recover_forward_realizations(graph, nodes, reachable_nodes)
    reverse = _recover_reverse_families(graph, nodes, users, reachable_nodes, forward)
    live_contracts = tuple(
        node for node in graph.nodes if node.id in reachable_nodes and node.opcode == "dot" and len(node.operands) == 2
    )
    owned = [contract for plan in forward for contract in (plan.down_contract, plan.up_contract)]
    owned.extend(
        contract
        for plan in reverse
        for contract in (
            plan.up_input_adjoint,
            plan.down_input_adjoint,
            plan.down_weight_adjoint,
            plan.up_weight_adjoint,
        )
    )
    return LowRankGatedProductTrainingReport(
        forward_realizations=forward,
        reverse_families=reverse,
        live_contract_count=len(live_contracts),
        live_contract_flops=sum(_dot_flops(node, nodes) for node in live_contracts),
        owned_contract_count=len(owned),
        owned_contract_flops=sum(contract.flops for contract in owned),
    )


def _recover_forward_realizations(
    graph: InlinedHloGraph,
    nodes: dict[str, InlinedHloNode],
    reachable: set[str],
) -> tuple[LowRankGatedProductForwardPlan, ...]:
    plans: list[LowRankGatedProductForwardPlan] = []
    for first in graph.nodes:
        if first.id not in reachable or not _is_forward_rank_two_contract(first):
            continue
        first_shape = _array_shape(first.shape)
        assert first_shape is not None
        source = _strip_layout_wrappers(first.operands[0], nodes)
        candidates: list[tuple[InlinedHloNode, CastScalarProgram]] = []
        for second in graph.nodes:
            if second.id not in reachable or not _is_forward_rank_two_contract(second):
                continue
            second_shape = _array_shape(second.shape)
            if second_shape is None or first_shape[1][0] != second_shape[1][0]:
                continue
            try:
                hidden_map = import_hlo_scalar_map(
                    graph,
                    source_nodes=(first.id,),
                    target_node=second.operands[0],
                )
            except (KeyError, ValueError):
                continue
            candidates.append((second, hidden_map))
        for second, hidden_map in candidates:
            output = _find_binary_merge(
                nodes,
                reachable,
                left=source,
                right=second.id,
                opcode="multiply",
            )
            if output is None:
                continue
            try:
                output_map = import_hlo_scalar_map(
                    graph,
                    source_nodes=(source, second.id),
                    target_node=output.id,
                )
            except ValueError:
                continue
            plans.append(
                LowRankGatedProductForwardPlan(
                    input=_entry_value(nodes[source]),
                    down_contract=_rank_two_contract(first, nodes),
                    hidden_map=hidden_map,
                    hidden=_entry_value(nodes[second.operands[0]]),
                    up_contract=_rank_two_contract(second, nodes),
                    output_map=output_map,
                    output=_entry_value(output),
                    parameter_origins=(
                        _semantic_origin(first.operands[1], nodes),
                        _semantic_origin(second.operands[1], nodes),
                    ),
                )
            )
    unique = {plan.down_contract.instruction: plan for plan in plans}
    return tuple(unique[name] for name in sorted(unique, key=lambda value: _entry_order(graph, value)))


def _recover_reverse_families(
    graph: InlinedHloGraph,
    nodes: dict[str, InlinedHloNode],
    users: dict[str, tuple[str, ...]],
    reachable: set[str],
    forward: tuple[LowRankGatedProductForwardPlan, ...],
) -> tuple[LowRankGatedProductReversePlan, ...]:
    groups: dict[tuple[str, str], list[LowRankGatedProductForwardPlan]] = {}
    for plan in forward:
        groups.setdefault(plan.parameter_origins, []).append(plan)
    reverse_contracts = tuple(
        node for node in graph.nodes if node.id in reachable and _rank_two_contract_dimensions(node) == ((1,), (1,))
    )
    weight_contracts = tuple(
        node for node in graph.nodes if node.id in reachable and _rank_two_contract_dimensions(node) == ((0,), (1,))
    )
    plans: list[LowRankGatedProductReversePlan] = []
    for origins, primal_candidates in groups.items():
        first_reverse = tuple(
            node for node in reverse_contracts if _semantic_origin(node.operands[1], nodes) == origins[1]
        )
        second_reverse = tuple(
            node for node in reverse_contracts if _semantic_origin(node.operands[1], nodes) == origins[0]
        )
        if len(first_reverse) != 1 or len(second_reverse) != 1:
            continue
        up_input_adjoint = first_reverse[0]
        down_input_adjoint = second_reverse[0]
        primal = _primal_for_weight_adjoint(
            primal_candidates,
            up_input_adjoint,
            down_input_adjoint,
            weight_contracts,
            nodes,
        )
        if primal is None:
            continue
        hidden_vjp_sources = (up_input_adjoint.id, _node_id(primal.down_contract.instruction, nodes))
        try:
            hidden_vjp = import_hlo_scalar_map(
                graph,
                source_nodes=hidden_vjp_sources,
                target_node=down_input_adjoint.operands[0],
            )
        except ValueError:
            continue
        up_map_sources = _scalar_boundary_sources(
            up_input_adjoint.operands[0],
            nodes,
            protected=(primal.input.instruction, primal.up_contract.instruction),
        )
        up_source_ids = tuple(_node_id(name, nodes) for name in up_map_sources)
        try:
            up_input_map = import_hlo_scalar_map(
                graph,
                source_nodes=up_source_ids,
                target_node=up_input_adjoint.operands[0],
            )
        except ValueError:
            continue
        input_adjoint = _find_residual_add(down_input_adjoint.id, nodes, users)
        if input_adjoint is None:
            continue
        residual_sources = _scalar_boundary_sources(
            input_adjoint.id,
            nodes,
            protected=(down_input_adjoint.id, primal.up_contract.instruction),
        )
        residual_source_ids = tuple(_node_id(name, nodes) for name in residual_sources)
        try:
            residual_map = import_hlo_scalar_map(
                graph,
                source_nodes=residual_source_ids,
                target_node=input_adjoint.id,
            )
        except ValueError:
            continue
        weight_pair = _weight_adjoint_pair(
            primal,
            up_input_adjoint,
            down_input_adjoint,
            weight_contracts,
            nodes,
        )
        if weight_pair is None:
            continue
        plans.append(
            LowRankGatedProductReversePlan(
                primal=primal,
                up_input_map=up_input_map,
                up_input_sources=tuple(_entry_value(nodes[node_id]) for node_id in up_source_ids),
                upstream_collectives=_nearest_upstream_collectives(up_source_ids, nodes),
                up_input_adjoint=_rank_two_contract(up_input_adjoint, nodes),
                hidden_vjp_map=hidden_vjp,
                down_input_adjoint=_rank_two_contract(down_input_adjoint, nodes),
                residual_vjp_map=residual_map,
                input_adjoint=_entry_value(input_adjoint),
                down_weight_adjoint=_rank_two_contract(weight_pair[0], nodes),
                up_weight_adjoint=_rank_two_contract(weight_pair[1], nodes),
            )
        )
    return tuple(sorted(plans, key=lambda plan: plan.primal.down_contract.instruction))


def _primal_for_weight_adjoint(
    candidates: list[LowRankGatedProductForwardPlan],
    up_input_adjoint: InlinedHloNode,
    down_input_adjoint: InlinedHloNode,
    weight_contracts: tuple[InlinedHloNode, ...],
    nodes: dict[str, InlinedHloNode],
) -> LowRankGatedProductForwardPlan | None:
    for primal in candidates:
        pair = _weight_adjoint_pair(primal, up_input_adjoint, down_input_adjoint, weight_contracts, nodes)
        if pair is not None:
            return primal
    return None


def _weight_adjoint_pair(
    primal: LowRankGatedProductForwardPlan,
    up_input_adjoint: InlinedHloNode,
    down_input_adjoint: InlinedHloNode,
    weight_contracts: tuple[InlinedHloNode, ...],
    nodes: dict[str, InlinedHloNode],
) -> tuple[InlinedHloNode, InlinedHloNode] | None:
    input_base = _strip_layout_wrappers(_node_id(primal.input.instruction, nodes), nodes)
    hidden_base = _strip_layout_wrappers(_node_id(primal.hidden.instruction, nodes), nodes)
    down_map_base = _strip_layout_wrappers(down_input_adjoint.operands[0], nodes)
    up_map_base = _strip_layout_wrappers(up_input_adjoint.operands[0], nodes)
    down = tuple(
        node
        for node in weight_contracts
        if _strip_layout_wrappers(node.operands[0], nodes) == input_base
        and _strip_layout_wrappers(node.operands[1], nodes) == down_map_base
    )
    up = tuple(
        node
        for node in weight_contracts
        if _strip_layout_wrappers(node.operands[0], nodes) == hidden_base
        and _strip_layout_wrappers(node.operands[1], nodes) == up_map_base
    )
    if len(down) != 1 or len(up) != 1:
        return None
    return down[0], up[0]


def _find_binary_merge(
    nodes: dict[str, InlinedHloNode],
    reachable: set[str],
    *,
    left: str,
    right: str,
    opcode: str,
) -> InlinedHloNode | None:
    candidates: list[InlinedHloNode] = []
    for node in nodes.values():
        if node.id not in reachable or node.opcode != opcode or len(node.operands) != 2:
            continue
        dependencies = tuple(_dataflow_sources(operand, nodes, {left, right}) for operand in node.operands)
        if set(dependencies) == {frozenset({left}), frozenset({right})}:
            candidates.append(node)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _find_residual_add(
    source: str,
    nodes: dict[str, InlinedHloNode],
    users: dict[str, tuple[str, ...]],
) -> InlinedHloNode | None:
    frontier = [source]
    visited = {source}
    candidates: list[InlinedHloNode] = []
    while frontier:
        current = frontier.pop()
        for user_id in users.get(current, ()):
            if user_id in visited:
                continue
            visited.add(user_id)
            user = nodes[user_id]
            if user.opcode == "add" and len(user.operands) == 2:
                candidates.append(user)
                continue
            if user.opcode in _LAYOUT_WRAPPERS:
                frontier.append(user_id)
    return candidates[0] if len(candidates) == 1 else None


def _scalar_boundary_sources(
    target: str,
    nodes: dict[str, InlinedHloNode],
    *,
    protected: tuple[str, ...],
) -> tuple[str, ...]:
    protected_order = tuple(_node_id(name, nodes) for name in protected)
    protected_ids = set(protected_order)
    leaves: set[str] = set()
    visited: set[str] = set()
    dependence_memo: dict[str, bool] = {}

    def depends_on_protected(node_id: str) -> bool:
        if node_id in dependence_memo:
            return dependence_memo[node_id]
        if node_id in protected_ids:
            dependence_memo[node_id] = True
            return True
        node = nodes[node_id]
        result = node.opcode in _SCALAR_DATAFLOW and any(depends_on_protected(operand) for operand in node.operands)
        dependence_memo[node_id] = result
        return result

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        if node_id in protected_ids:
            leaves.add(node_id)
            return
        node = nodes[node_id]
        if node.opcode == "constant":
            return
        if node.opcode == "broadcast" and len(node.operands) == 1:
            broadcast_source = _array_shape(nodes[node.operands[0]].shape)
            if broadcast_source is not None and not broadcast_source[1]:
                visit(node.operands[0])
                return
        if node.opcode in _SCALAR_DATAFLOW:
            for operand in node.operands:
                if depends_on_protected(operand):
                    visit(operand)
                    continue
                operand_node = nodes[operand]
                if operand_node.opcode == "broadcast" and len(operand_node.operands) == 1:
                    broadcast_source = _array_shape(nodes[operand_node.operands[0]].shape)
                    if broadcast_source is not None and not broadcast_source[1]:
                        visit(operand)
                        continue
                operand_shape = _array_shape(nodes[operand].shape)
                if operand_shape is not None and operand_shape[1]:
                    leaves.add(operand)
                else:
                    visit(operand)
            return
        leaves.add(node_id)

    visit(target)
    ordered_ids = tuple(node_id for node_id in protected_order if node_id in leaves) + tuple(
        sorted(leaves - protected_ids, key=lambda node_id: _entry_instruction(nodes[node_id]))
    )
    return tuple(_entry_instruction(nodes[node_id]) for node_id in ordered_ids)


def _dataflow_sources(
    node_id: str,
    nodes: dict[str, InlinedHloNode],
    protected: set[str],
) -> frozenset[str]:
    if node_id in protected:
        return frozenset({node_id})
    node = nodes[node_id]
    if node.opcode == "constant":
        return frozenset()
    if node.opcode == "broadcast" and len(node.operands) == 1:
        source_shape = _array_shape(nodes[node.operands[0]].shape)
        if source_shape is not None and not source_shape[1]:
            return _dataflow_sources(node.operands[0], nodes, protected)
    if node.opcode not in _SCALAR_DATAFLOW:
        return frozenset({node_id})
    return frozenset().union(*(_dataflow_sources(operand, nodes, protected) for operand in node.operands))


def _semantic_origin(node_id: str, nodes: dict[str, InlinedHloNode]) -> str:
    node = nodes[node_id]
    if node.opcode in _LAYOUT_WRAPPERS | {"convert", "opt-barrier"} and len(node.operands) == 1:
        return _semantic_origin(node.operands[0], nodes)
    if node.opcode == "get-tuple-element" and len(node.operands) == 1:
        match = _TUPLE_INDEX.search(node.attributes)
        if match is None:
            return node.id
        tuple_node = nodes[node.operands[0]]
        while tuple_node.opcode == "opt-barrier" and len(tuple_node.operands) == 1:
            tuple_node = nodes[tuple_node.operands[0]]
        if tuple_node.opcode != "tuple":
            return node.id
        return _semantic_origin(tuple_node.operands[int(match.group("index"))], nodes)
    return node.id


def _nearest_upstream_collectives(
    source_ids: tuple[str, ...],
    nodes: dict[str, InlinedHloNode],
) -> tuple[EntryRegionValue, ...]:
    found: dict[str, EntryRegionValue] = {}
    visited: set[str] = set()
    pending = list(source_ids)
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = nodes[node_id]
        if node.opcode == "all-reduce":
            found[node.source_instruction] = _entry_value(node)
            continue
        pending.extend(node.operands)
    return tuple(found[name] for name in sorted(found))


def _strip_layout_wrappers(node_id: str, nodes: dict[str, InlinedHloNode]) -> str:
    current = nodes[node_id]
    while current.opcode in _LAYOUT_WRAPPERS and len(current.operands) == 1:
        current = nodes[current.operands[0]]
    return current.id


def _rank_two_contract(node: InlinedHloNode, nodes: dict[str, InlinedHloNode]) -> RankTwoContractPlan:
    dimensions = _rank_two_contract_dimensions(node)
    if dimensions is None or len(dimensions[0]) != 1 or len(dimensions[1]) != 1:
        raise ValueError(f"%{_entry_instruction(node)} is not a rank-two Contract")
    return RankTwoContractPlan(
        instruction=_entry_instruction(node),
        lhs=_entry_value(nodes[node.operands[0]]),
        rhs=_entry_value(nodes[node.operands[1]]),
        output=_entry_value(node),
        lhs_contracting_dimension=dimensions[0][0],
        rhs_contracting_dimension=dimensions[1][0],
        flops=_dot_flops(node, nodes),
    )


def _is_forward_rank_two_contract(node: InlinedHloNode) -> bool:
    return _rank_two_contract_dimensions(node) == ((1,), (0,))


def _rank_two_contract_dimensions(node: InlinedHloNode) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    shape = _array_shape(node.shape)
    if node.opcode != "dot" or len(node.operands) != 2 or shape is None or len(shape[1]) != 2:
        return None
    dimensions = _dot_dimensions(node.attributes)
    if dimensions.get("lhs_batch_dims") or dimensions.get("rhs_batch_dims"):
        return None
    return dimensions.get("lhs_contracting_dims", ()), dimensions.get("rhs_contracting_dims", ())


def _dot_flops(node: InlinedHloNode, nodes: dict[str, InlinedHloNode]) -> int:
    output = _array_shape(node.shape)
    lhs = _array_shape(nodes[node.operands[0]].shape)
    if output is None or lhs is None:
        raise ValueError(f"unsupported Contract shape at %{_entry_instruction(node)}")
    dimensions = _dot_dimensions(node.attributes).get("lhs_contracting_dims", ())
    fold_extent = math.prod(lhs[1][axis] for axis in dimensions)
    return 2 * math.prod(output[1]) * fold_extent


def _dot_dimensions(attributes: str) -> dict[str, tuple[int, ...]]:
    return {
        match.group("name"): tuple(int(value) for value in match.group("dims").split(",") if value)
        for match in _DOT_DIMENSION.finditer(attributes)
    }


def _array_shape(shape: str) -> tuple[str, tuple[int, ...]] | None:
    match = _ARRAY_SHAPE.match(shape.lstrip("("))
    if match is None:
        return None
    return match.group("dtype"), tuple(int(value) for value in match.group("dims").split(",") if value)


def _entry_value(node: InlinedHloNode) -> EntryRegionValue:
    return EntryRegionValue(_entry_instruction(node), node.shape)


def _entry_instruction(node: InlinedHloNode) -> str:
    return node.source_instruction


def _node_id(instruction: str, nodes: dict[str, InlinedHloNode]) -> str:
    if instruction in nodes:
        return instruction
    matches = tuple(node.id for node in nodes.values() if node.source_instruction == instruction)
    if len(matches) != 1:
        raise ValueError(f"entry instruction %{instruction} maps to {len(matches)} physical nodes")
    return matches[0]


def _users(graph: InlinedHloGraph) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for node in graph.nodes:
        for operand in node.operands:
            mutable.setdefault(operand, []).append(node.id)
    return {name: tuple(values) for name, values in mutable.items()}


def _reachable_entry_instructions(entry: HloComputation) -> set[str]:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    reachable: set[str] = set()
    pending = [entry.root.name]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(instructions[name].operands)
    return reachable


def _entry_order(graph: InlinedHloGraph, instruction: str) -> int:
    order = {name: index for index, (name, _) in enumerate(graph.entry_values)}
    return order[instruction]
