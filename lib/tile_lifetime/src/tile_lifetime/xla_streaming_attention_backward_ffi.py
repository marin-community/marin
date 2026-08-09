# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace a recovered JAX reverse region with generated typed FFI.

This pass is intentionally limited to a whole-entry proof.  It derives operand
roles from Contract, Fold, and DomainRestriction dataflow and rejects modules
whose physical graph cannot establish that provenance.  No frontend or model
names participate in matching.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.jax_streaming_attention_backward_ffi import (
    GeneratedStreamingAttentionBackwardFfi,
    StreamingAttentionBackwardFfiBuffer,
    StreamingAttentionBackwardStatePolicy,
)
from tile_lifetime.streaming_attention_backward import StreamingAttentionBackwardProgram
from tile_lifetime.tensor_program import ScalarExpression, ScalarExpressionKind
from tile_lifetime.xla_hlo_recovery import (
    HloInstruction,
    HloModuleGraph,
    InlinedHloGraph,
    InlinedHloNode,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?:\{(?P<layout>[0-9,]+)\})?")
_CALLED_COMPUTATION = re.compile(r"to_apply=%?(?P<name>[A-Za-z0-9_.-]+)")
_CONSTANT = re.compile(r"constant\((?P<value>-?inf|[-+0-9.eE]+)\)")


class StreamingReverseHloRole(StrEnum):
    """Natural JAX reverse buffer roles at the physical entry boundary."""

    QUERY = "query"
    KEY = "key"
    VALUE = "value"
    OUTPUT_COTANGENT = "output_cotangent"
    QUERY_COTANGENT = "query_cotangent"
    KEY_COTANGENT = "key_cotangent"
    VALUE_COTANGENT = "value_cotangent"


@dataclass(frozen=True)
class StreamingReverseHloValue:
    """One recovered physical value with its generic semantic role."""

    role: StreamingReverseHloRole
    instruction: str
    physical_shape: str
    canonical_shape: str


@dataclass(frozen=True)
class StreamingReverseHloProvenance:
    """Generic algebra that justifies replacing the materialized reverse."""

    score_contract: str
    reverse_contracts: tuple[str, ...]
    maximum_fold: str
    additive_folds: tuple[str, ...]
    domain_restriction: str | None
    score_scale: float


@dataclass(frozen=True)
class StreamingReverseHloReplacementPlan:
    """A whole-entry typed-FFI replacement proven from physical HLO dataflow."""

    inputs: tuple[StreamingReverseHloValue, ...]
    outputs: tuple[StreamingReverseHloValue, ...]
    root_instruction: str
    root_shape: str
    provenance: StreamingReverseHloProvenance
    state_policy: StreamingAttentionBackwardStatePolicy
    semantic_fingerprint: str
    maximum_vjp: str
    reassociation: str


def plan_streaming_attention_backward_hlo_replacement(
    hlo_text: str,
    program: StreamingAttentionBackwardProgram,
    generated: GeneratedStreamingAttentionBackwardFfi,
) -> StreamingReverseHloReplacementPlan:
    """Prove a whole-entry generic reverse boundary and bind its FFI buffers.

    ``program`` must be recovered from the same natural JAX lowering as
    ``hlo_text``.  The pass independently checks physical shapes, score scale,
    restriction structure, normalized-exponential Folds, and reverse Contracts.
    An unsupported or ambiguous post-SPMD graph is rejected rather than
    approximately rewritten.
    """
    if generated.state_policy is not StreamingAttentionBackwardStatePolicy.RECOMPUTE:
        raise ValueError("natural four-input JAX VJP replacement requires explicit recompute state policy")
    expected_input_names = ("query", "key", "value", "output_cotangent")
    expected_output_names = ("query_cotangent", "key_cotangent", "value_cotangent")
    if tuple(value.name for value in generated.inputs) != expected_input_names:
        raise ValueError("whole-entry reverse replacement requires the natural four-buffer input signature")
    if tuple(value.name for value in generated.outputs) != expected_output_names:
        raise ValueError("whole-entry reverse replacement requires three input-cotangent outputs")

    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    graph = inline_elementwise_fusions(module)
    nodes = {node.id: node for node in graph.nodes}
    entry_instructions = {instruction.name: instruction for instruction in entry.instructions}
    root = entry.root
    if root.opcode != "tuple" or len(root.operands) != 3:
        raise ValueError("streaming reverse replacement requires a three-result entry tuple")
    if any(
        instruction.opcode in {"infeed", "outfeed", "recv", "send"}
        or "custom_call_has_side_effect=true" in instruction.attributes
        for instruction in entry.instructions
    ):
        raise ValueError("whole-entry reverse replacement cannot subsume side effects")

    parameter_instructions = tuple(
        instruction for instruction in entry.instructions if instruction.opcode == "parameter"
    )
    if len(parameter_instructions) != 4:
        raise ValueError("streaming reverse replacement requires exactly four entry parameters")
    parameter_nodes = {graph.entry_value(instruction.name): instruction for instruction in parameter_instructions}
    parameter_ancestors = _parameter_ancestor_function(nodes, frozenset(parameter_nodes))
    contract_ancestors = _opcode_ancestor_function(nodes, "dot")

    maximum_folds = tuple(
        node for node in graph.nodes if node.opcode == "reduce" and _fold_reducer(module, node.attributes) == "maximum"
    )
    maximum_candidates = tuple(
        node for node in maximum_folds if len(node.operands) == 2 and len(contract_ancestors(node.operands[0])) == 1
    )
    if len(maximum_candidates) != 1:
        raise ValueError(f"expected one normalized-exponential maximum Fold, found {len(maximum_candidates)}")
    maximum_fold = maximum_candidates[0]
    score_contract_id = next(iter(contract_ancestors(maximum_fold.operands[0])))
    score_contract = nodes[score_contract_id]
    if len(score_contract.operands) != 2:
        raise ValueError("score Contract must have two operands")

    specifications = {value.name: value for value in (*generated.inputs, *generated.outputs)}
    query_node, key_node = _bind_score_contract_parameters(
        score_contract,
        parameter_ancestors,
        parameter_nodes,
        specifications,
    )
    remaining_parameters = frozenset(parameter_nodes) - {query_node, key_node}
    value_node = _unique_parameter_for_shape(
        remaining_parameters,
        parameter_nodes,
        specifications["value"],
        role="value",
    )
    output_cotangent_node = _unique_parameter_for_shape(
        remaining_parameters - {value_node},
        parameter_nodes,
        specifications["output_cotangent"],
        role="output cotangent",
    )
    if {query_node, key_node, value_node, output_cotangent_node} != set(parameter_nodes):
        raise ValueError("generic reverse roles do not cover the exact entry parameter boundary")

    score_map_nodes = _ancestor_slice(nodes, maximum_fold.operands[0], stop=frozenset({score_contract_id}))
    restriction_nodes = tuple(node for node in score_map_nodes if node.opcode == "select")
    comparison_nodes = tuple(node for node in score_map_nodes if node.opcode == "compare")
    expected_scale, expected_restriction = _program_score_policy(program)
    if expected_restriction:
        if len(restriction_nodes) != 1 or len(comparison_nodes) != 1:
            raise ValueError("causal DomainRestriction must lower to one select and one comparison")
        if "direction=LE" not in comparison_nodes[0].attributes:
            raise ValueError("only less-equal DomainRestriction is supported by the generated skeleton")
    elif restriction_nodes or comparison_nodes:
        raise ValueError("physical HLO has a DomainRestriction absent from the recovered program")
    physical_scale = _score_scale(nodes, score_map_nodes, score_contract_id)
    if not math.isclose(physical_scale, expected_scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"physical score scale {physical_scale} does not match recovered scale {expected_scale}")

    additive_folds = tuple(
        node.id for node in graph.nodes if node.opcode == "reduce" and _fold_reducer(module, node.attributes) == "add"
    )
    contracts = tuple(node.id for node in graph.nodes if node.opcode == "dot")
    reverse_contracts = tuple(node_id for node_id in contracts if node_id != score_contract_id)
    if len(reverse_contracts) < 4 or len(additive_folds) < 2:
        raise ValueError("physical reverse lacks the required Contract/Fold decomposition")

    role_nodes = {
        StreamingReverseHloRole.QUERY: query_node,
        StreamingReverseHloRole.KEY: key_node,
        StreamingReverseHloRole.VALUE: value_node,
        StreamingReverseHloRole.OUTPUT_COTANGENT: output_cotangent_node,
    }
    inputs = tuple(
        _input_value(role, role_nodes[role], parameter_nodes, specifications[role.value])
        for role in (
            StreamingReverseHloRole.QUERY,
            StreamingReverseHloRole.KEY,
            StreamingReverseHloRole.VALUE,
            StreamingReverseHloRole.OUTPUT_COTANGENT,
        )
    )
    outputs = _bind_outputs(
        root,
        graph,
        entry_instructions,
        parameter_ancestors,
        value_node,
        specifications,
    )
    return StreamingReverseHloReplacementPlan(
        inputs=inputs,
        outputs=outputs,
        root_instruction=root.name,
        root_shape=root.shape,
        provenance=StreamingReverseHloProvenance(
            score_contract=score_contract_id,
            reverse_contracts=reverse_contracts,
            maximum_fold=maximum_fold.id,
            additive_folds=additive_folds,
            domain_restriction=restriction_nodes[0].id if restriction_nodes else None,
            score_scale=physical_scale,
        ),
        state_policy=generated.state_policy,
        semantic_fingerprint=generated.semantic_fingerprint,
        maximum_vjp=program.maximum_vjp.value,
        reassociation=program.reassociation.value,
    )


def replace_streaming_attention_backward_entry_with_custom_call(
    hlo_text: str,
    plan: StreamingReverseHloReplacementPlan,
    *,
    target: str,
) -> str:
    """Replace the proven entry root while preserving physical boundary layouts."""
    pattern = re.compile(
        rf"^(?P<indent>\s*)ROOT\s+%?{re.escape(plan.root_instruction)} = .*?$",
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one entry root definition for {plan.root_instruction!r}")
    match = matches[0]
    indent = match.group("indent")
    names: dict[StreamingReverseHloRole, str] = {}
    lines: list[str] = []
    for value in plan.inputs:
        name = value.instruction
        if value.physical_shape != value.canonical_shape:
            name = f"shuttle.{value.role.value}.canonical"
            lines.append(f"{indent}%{name} = {value.canonical_shape} copy(%{value.instruction})")
        names[value.role] = name
    output_shapes = ", ".join(value.canonical_shape for value in plan.outputs)
    operands = ", ".join(
        f"%{names[role]}"
        for role in (
            StreamingReverseHloRole.QUERY,
            StreamingReverseHloRole.KEY,
            StreamingReverseHloRole.VALUE,
            StreamingReverseHloRole.OUTPUT_COTANGENT,
        )
    )
    constraints = ", ".join(value.canonical_shape for value in plan.inputs)
    call_name = "shuttle.generated.streaming_reverse"
    lines.append(
        f"{indent}%{call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    output_names: dict[StreamingReverseHloRole, str] = {}
    for index, value in enumerate(plan.outputs):
        canonical_name = f"shuttle.{value.role.value}.canonical"
        get_tuple_element = f"get-tuple-element(%{call_name}), index={index}"
        lines.append(f"{indent}%{canonical_name} = {value.canonical_shape} {get_tuple_element}")
        output_name = canonical_name
        if value.physical_shape != value.canonical_shape:
            output_name = f"shuttle.{value.role.value}.physical"
            lines.append(f"{indent}%{output_name} = {value.physical_shape} copy(%{canonical_name})")
        output_names[value.role] = output_name
    root_operands = ", ".join(f"%{output_names[value.role]}" for value in plan.outputs)
    lines.append(f"{indent}ROOT %{plan.root_instruction} = {plan.root_shape} tuple({root_operands})")
    replacement = "\n".join(lines)
    return hlo_text[: match.start()] + replacement + hlo_text[match.end() :]


def _bind_score_contract_parameters(
    score_contract: InlinedHloNode,
    parameter_ancestors: Callable[[str], frozenset[str]],
    parameter_nodes: dict[str, HloInstruction],
    specifications: dict[str, StreamingAttentionBackwardFfiBuffer],
) -> tuple[str, str]:
    operand_parameters = tuple(parameter_ancestors(operand) for operand in score_contract.operands)
    if any(len(values) != 1 for values in operand_parameters):
        raise ValueError("score Contract operands must each descend from one entry parameter")
    candidates = tuple(next(iter(values)) for values in operand_parameters)
    query = _unique_parameter_for_shape(candidates, parameter_nodes, specifications["query"], role="query")
    key = _unique_parameter_for_shape(candidates, parameter_nodes, specifications["key"], role="key")
    if query == key:
        raise ValueError("query and key roles must bind distinct parameters")
    return query, key


def _bind_outputs(
    root: HloInstruction,
    graph: InlinedHloGraph,
    entry_instructions: dict[str, HloInstruction],
    parameter_ancestors: Callable[[str], frozenset[str]],
    value_node: str,
    specifications: dict[str, StreamingAttentionBackwardFfiBuffer],
) -> tuple[StreamingReverseHloValue, ...]:
    candidates = tuple(
        (instruction, graph.entry_value(instruction), parameter_ancestors(graph.entry_value(instruction)))
        for instruction in root.operands
    )
    query_candidates = tuple(
        candidate
        for candidate in candidates
        if _shape_signature(entry_instructions[candidate[0]].shape)
        == _buffer_signature(specifications["query_cotangent"])
    )
    if len(query_candidates) != 1:
        raise ValueError("could not uniquely bind the query-cotangent result")
    query = query_candidates[0]
    key_value_candidates = tuple(candidate for candidate in candidates if candidate != query)
    value_candidates = tuple(candidate for candidate in key_value_candidates if value_node not in candidate[2])
    key_candidates = tuple(candidate for candidate in key_value_candidates if value_node in candidate[2])
    if len(key_candidates) != 1 or len(value_candidates) != 1:
        raise ValueError("could not distinguish key and value cotangents from physical dataflow")
    by_role = {
        StreamingReverseHloRole.QUERY_COTANGENT: query,
        StreamingReverseHloRole.KEY_COTANGENT: key_candidates[0],
        StreamingReverseHloRole.VALUE_COTANGENT: value_candidates[0],
    }
    if tuple(by_role[role][0] for role in by_role) != root.operands:
        raise ValueError("entry result order does not match the natural query/key/value cotangent signature")
    return tuple(
        StreamingReverseHloValue(
            role=role,
            instruction=by_role[role][0],
            physical_shape=entry_instructions[by_role[role][0]].shape,
            canonical_shape=_canonical_shape(specifications[role.value]),
        )
        for role in (
            StreamingReverseHloRole.QUERY_COTANGENT,
            StreamingReverseHloRole.KEY_COTANGENT,
            StreamingReverseHloRole.VALUE_COTANGENT,
        )
    )


def _input_value(
    role: StreamingReverseHloRole,
    node_id: str,
    parameter_nodes: dict[str, HloInstruction],
    specification: StreamingAttentionBackwardFfiBuffer,
) -> StreamingReverseHloValue:
    instruction = parameter_nodes[node_id]
    return StreamingReverseHloValue(
        role=role,
        instruction=instruction.name,
        physical_shape=instruction.shape,
        canonical_shape=_canonical_shape(specification),
    )


def _parameter_ancestor_function(
    nodes: dict[str, InlinedHloNode],
    parameter_nodes: frozenset[str],
) -> Callable[[str], frozenset[str]]:
    memo: dict[str, frozenset[str]] = {}

    def ancestors(node_id: str) -> frozenset[str]:
        if node_id in memo:
            return memo[node_id]
        if node_id in parameter_nodes:
            result = frozenset({node_id})
        else:
            result = frozenset().union(*(ancestors(operand) for operand in nodes[node_id].operands))
        memo[node_id] = result
        return result

    return ancestors


def _opcode_ancestor_function(
    nodes: dict[str, InlinedHloNode],
    opcode: str,
) -> Callable[[str], frozenset[str]]:
    memo: dict[str, frozenset[str]] = {}

    def ancestors(node_id: str) -> frozenset[str]:
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        nested = frozenset().union(*(ancestors(operand) for operand in node.operands))
        result = nested | ({node_id} if node.opcode == opcode else set())
        memo[node_id] = frozenset(result)
        return memo[node_id]

    return ancestors


def _ancestor_slice(
    nodes: dict[str, InlinedHloNode],
    root: str,
    *,
    stop: frozenset[str],
) -> tuple[InlinedHloNode, ...]:
    ordered: list[InlinedHloNode] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen or node_id in stop:
            return
        seen.add(node_id)
        node = nodes[node_id]
        for operand in node.operands:
            visit(operand)
        ordered.append(node)

    visit(root)
    return tuple(ordered)


def _fold_reducer(module: HloModuleGraph, attributes: str) -> str | None:
    match = _CALLED_COMPUTATION.search(attributes)
    if match is None:
        return None
    return module.computation(match.group("name")).root.opcode


def _unique_parameter_for_shape(
    candidates: Iterable[str],
    parameter_nodes: dict[str, HloInstruction],
    specification: StreamingAttentionBackwardFfiBuffer,
    *,
    role: str,
) -> str:
    matches = tuple(
        candidate
        for candidate in candidates
        if _shape_signature(parameter_nodes[candidate].shape) == _buffer_signature(specification)
    )
    if len(matches) != 1:
        raise ValueError(f"could not uniquely bind {role} parameter by physical shape")
    return matches[0]


def _score_scale(
    nodes: dict[str, InlinedHloNode],
    score_map_nodes: tuple[InlinedHloNode, ...],
    score_contract_id: str,
) -> float:
    contract_ancestors = _opcode_ancestor_function(nodes, "dot")
    candidates: list[float] = []
    for node in score_map_nodes:
        if node.opcode != "multiply" or len(node.operands) != 2:
            continue
        for value, scalar in ((node.operands[0], node.operands[1]), (node.operands[1], node.operands[0])):
            if score_contract_id not in contract_ancestors(value):
                continue
            constant = _broadcast_scalar_constant(nodes, scalar)
            if constant is not None and math.isfinite(constant):
                candidates.append(constant)
    if len(candidates) != 1:
        raise ValueError(f"expected one affine score scale in physical HLO, found {candidates}")
    return candidates[0]


def _broadcast_scalar_constant(nodes: dict[str, InlinedHloNode], node_id: str) -> float | None:
    node = nodes[node_id]
    if node.opcode == "constant":
        match = _CONSTANT.search(node.attributes)
        return float(match.group("value")) if match is not None else None
    if node.opcode in {"bitcast", "broadcast", "copy", "reshape"} and len(node.operands) == 1:
        return _broadcast_scalar_constant(nodes, node.operands[0])
    return None


def _program_score_policy(program: StreamingAttentionBackwardProgram) -> tuple[float, bool]:
    raw_score_name = program.forward.qk.output.name
    causal = False

    def literal(expression: ScalarExpression) -> float | bool | None:
        return expression.constant if expression.kind is ScalarExpressionKind.CONSTANT else None

    def visit(expression: ScalarExpression) -> float:
        nonlocal causal
        if expression.kind is ScalarExpressionKind.SELECT:
            predicate, selected, rejected = expression.operands
            if literal(rejected) != float("-inf") or predicate.kind is not ScalarExpressionKind.LESS_EQUAL:
                raise ValueError("whole-entry replacement supports only less-equal select-to-negative-infinity")
            causal = True
            return visit(selected)
        if expression.kind is ScalarExpressionKind.MULTIPLY:
            left, right = expression.operands
            if left.kind is ScalarExpressionKind.INPUT and left.input_name == raw_score_name:
                value = literal(right)
                if value is not None:
                    return float(value)
            if right.kind is ScalarExpressionKind.INPUT and right.input_name == raw_score_name:
                value = literal(left)
                if value is not None:
                    return float(value)
        raise ValueError("whole-entry replacement currently supports an affine scaled score Map")

    return visit(program.forward.score_map.expression), causal


def _canonical_shape(specification: StreamingAttentionBackwardFfiBuffer) -> str:
    dimensions = ",".join(str(value) for value in specification.shape)
    layout = ",".join(str(value) for value in reversed(range(len(specification.shape))))
    return f"{specification.dtype.value}[{dimensions}]{{{layout}}}"


def _shape_signature(shape: str) -> tuple[str, tuple[int, ...]]:
    match = _ARRAY_SHAPE.fullmatch(shape.strip())
    if match is None:
        raise ValueError(f"unsupported physical HLO array shape: {shape!r}")
    return match.group("dtype"), tuple(int(value) for value in match.group("dims").split(",") if value)


def _buffer_signature(specification: StreamingAttentionBackwardFfiBuffer) -> tuple[str, tuple[int, ...]]:
    return specification.dtype.value, specification.shape
