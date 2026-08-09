# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace a recovered JAX reverse region with generated typed FFI.

Both the whole-entry and region-local passes derive operand roles from
Contract, Fold, and DomainRestriction dataflow.  They reject modules whose
physical graph cannot establish that provenance.  No frontend or model names
participate in matching.
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
    StreamingAttentionBackwardFfiBufferLayout,
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
_CONTROL_PREDECESSORS = re.compile(r"control-predecessors=\{(?P<values>[^}]*)\}")

_REGION_MAP_OPCODES = frozenset(
    {
        "add",
        "broadcast",
        "compare",
        "constant",
        "convert",
        "copy",
        "divide",
        "exponential",
        "multiply",
        "negate",
        "reshape",
        "select",
        "subtract",
        "transpose",
    }
)
_REGION_REVERSE_OPCODES = _REGION_MAP_OPCODES | {"dot", "reduce"}


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
    ffi_shape: str


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


@dataclass(frozen=True)
class StreamingReverseHloRegionReplacementPlan:
    """A reverse-only entry region proven safe for local typed-FFI replacement."""

    inputs: tuple[StreamingReverseHloValue, ...]
    outputs: tuple[StreamingReverseHloValue, ...]
    insertion_instruction: str
    internal_instructions: tuple[str, ...]
    preserved_shared_inputs: tuple[str, ...]
    external_users: tuple[tuple[str, tuple[str, ...]], ...]
    provenance: StreamingReverseHloProvenance
    state_policy: StreamingAttentionBackwardStatePolicy
    semantic_fingerprint: str
    maximum_vjp: str
    reassociation: str


@dataclass(frozen=True)
class StreamingReverseHloRegionReplacementAudit:
    """Post-roundtrip liveness evidence for one local reverse replacement."""

    call_instruction: str
    dead_reverse_closure: tuple[str, ...]
    preserved_shared_users: tuple[tuple[str, tuple[str, ...]], ...]


def plan_streaming_attention_backward_hlo_region_replacement(
    hlo_text: str,
    program: StreamingAttentionBackwardProgram,
    generated: GeneratedStreamingAttentionBackwardFfi,
) -> StreamingReverseHloRegionReplacementPlan:
    """Prove a reverse-only attention region inside a larger entry graph.

    The proof starts at normalized-exponential maximum Folds, follows nearest
    Contracts, and binds the four generated inputs and three cotangent outputs
    from physical dataflow. Shared forward values are deliberately left live;
    the generated reverse recomputes them from Q/K/V after dO becomes ready.
    """
    _validate_generated_reverse_signature(generated, boundary="region-local")
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry.instructions)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    specifications = {value.name: value for value in (*generated.inputs, *generated.outputs)}
    expected_scale, expected_restriction = _program_score_policy(program)

    candidates: list[StreamingReverseHloRegionReplacementPlan] = []
    mismatch_reasons: list[str] = []
    maximum_folds = tuple(
        instruction
        for instruction in entry.instructions
        if instruction.opcode == "reduce" and _entry_fold_reducer(module, instruction) == "maximum"
    )
    for maximum_fold in maximum_folds:
        try:
            candidates.append(
                _plan_reverse_region_from_maximum(
                    entry.instructions,
                    instructions,
                    users,
                    source_order,
                    module,
                    maximum_fold,
                    specifications,
                    expected_scale=expected_scale,
                    expected_restriction=expected_restriction,
                    generated=generated,
                    program=program,
                )
            )
        except ValueError as error:
            mismatch_reasons.append(f"%{maximum_fold.name}: {error}")
    if len(candidates) != 1:
        detail = "; ".join(mismatch_reasons)
        raise ValueError(
            f"expected one region-local streaming reverse candidate, found {len(candidates)}"
            + (f" ({detail})" if detail else "")
        )
    return candidates[0]


def replace_streaming_attention_backward_region_with_custom_call(
    hlo_text: str,
    plan: StreamingReverseHloRegionReplacementPlan,
    *,
    target: str,
) -> str:
    """Insert one local reverse call and redirect only proven external users."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    reserved_names = set(instructions)
    generated_names = {
        "shuttle.generated.streaming_reverse.region",
        *(f"shuttle.region.{value.role.value}.canonical" for value in (*plan.inputs, *plan.outputs)),
        *(f"shuttle.region.{value.role.value}.physical" for value in plan.outputs),
    }
    collision = reserved_names & generated_names
    if collision:
        raise ValueError(f"region replacement names already exist: {sorted(collision)}")

    insertion_pattern = re.compile(
        rf"^(?P<indent>\s*)(?:ROOT\s+)?%?{re.escape(plan.insertion_instruction)} = .*?$",
        re.MULTILINE,
    )
    insertion_matches = tuple(insertion_pattern.finditer(hlo_text))
    if len(insertion_matches) != 1:
        raise ValueError(f"expected one insertion definition for {plan.insertion_instruction!r}")
    insertion_match = insertion_matches[0]
    indent = insertion_match.group("indent")
    input_names: dict[StreamingReverseHloRole, str] = {}
    lines: list[str] = []
    for value in plan.inputs:
        adapted = _emit_boundary_adapter(
            lines,
            indent=indent,
            source=value.instruction,
            source_shape=value.physical_shape,
            target_shape=value.ffi_shape,
            name=f"shuttle.region.{value.role.value}.canonical",
        )
        input_names[value.role] = adapted
    output_shapes = ", ".join(value.ffi_shape for value in plan.outputs)
    operands = ", ".join(
        f"%{input_names[role]}"
        for role in (
            StreamingReverseHloRole.QUERY,
            StreamingReverseHloRole.KEY,
            StreamingReverseHloRole.VALUE,
            StreamingReverseHloRole.OUTPUT_COTANGENT,
        )
    )
    constraints = ", ".join(value.ffi_shape for value in plan.inputs)
    call_name = "shuttle.generated.streaming_reverse.region"
    lines.append(
        f"{indent}%{call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    replacements: dict[str, str] = {}
    for index, value in enumerate(plan.outputs):
        canonical_name = f"shuttle.region.{value.role.value}.canonical"
        lines.append(f"{indent}%{canonical_name} = {value.ffi_shape} " f"get-tuple-element(%{call_name}), index={index}")
        physical_name = _emit_boundary_adapter(
            lines,
            indent=indent,
            source=canonical_name,
            source_shape=value.ffi_shape,
            target_shape=value.physical_shape,
            name=f"shuttle.region.{value.role.value}.physical",
        )
        replacements[value.instruction] = physical_name
    insertion = "\n" + "\n".join(lines)
    rewritten = hlo_text[: insertion_match.end()] + insertion + hlo_text[insertion_match.end() :]

    for old_name, external_users in plan.external_users:
        new_name = replacements[old_name]
        for user in external_users:
            rewritten = _replace_entry_operand(rewritten, user=user, old=old_name, new=new_name)
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_streaming_attention_backward_region_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: StreamingReverseHloRegionReplacementPlan,
    *,
    target: str,
) -> StreamingReverseHloRegionReplacementAudit:
    """Prove the old reverse closure is dead and shared forward users survive."""
    original_module = parse_hlo_module_text(original_hlo)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    original_entry = original_module.computation(original_module.entry)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    original_users = _entry_users(original_entry.instructions)
    transformed_users = _entry_users(transformed_entry.instructions)
    transformed_instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    target_attribute = f'custom_call_target="{target}"'
    calls = tuple(
        instruction
        for instruction in transformed_entry.instructions
        if instruction.opcode == "custom-call" and target_attribute in instruction.attributes
    )
    if len(calls) != 1:
        raise ValueError(f"expected one post-roundtrip reverse call for {target!r}, found {len(calls)}")

    old_closure = set(plan.internal_instructions) | {value.instruction for value in plan.outputs}
    for name in old_closure:
        crossing = tuple(user for user in transformed_users[name] if user not in old_closure)
        if crossing:
            raise ValueError(f"old reverse value %{name} remains externally live through {crossing}")

    preserved_users: list[tuple[str, tuple[str, ...]]] = []
    for name in plan.preserved_shared_inputs:
        before = tuple(user for user in original_users[name] if user not in old_closure)
        after = tuple(user for user in transformed_users[name] if user not in old_closure)
        if before != after:
            raise ValueError(f"shared forward value %{name} users changed: {before} -> {after}")
        preserved_users.append((name, after))
    for old_output, external_users in plan.external_users:
        for user in external_users:
            if old_output in transformed_instructions[user].operands:
                raise ValueError(f"external user %{user} still consumes old reverse output %{old_output}")
    return StreamingReverseHloRegionReplacementAudit(
        call_instruction=calls[0].name,
        dead_reverse_closure=tuple(
            name for name in (*plan.internal_instructions, *(value.instruction for value in plan.outputs))
        ),
        preserved_shared_users=tuple(preserved_users),
    )


def derive_streaming_attention_backward_ffi_output_layouts(
    plan: StreamingReverseHloReplacementPlan | StreamingReverseHloRegionReplacementPlan,
) -> tuple[StreamingAttentionBackwardFfiBufferLayout, ...]:
    """Derive generic FFI output layouts from the proven physical boundary."""
    return tuple(
        StreamingAttentionBackwardFfiBufferLayout(
            buffer_name=value.role.value,
            minor_to_major=_shape_layout(value.physical_shape),
        )
        for value in plan.outputs
    )


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
        if value.physical_shape != value.ffi_shape:
            name = f"shuttle.{value.role.value}.canonical"
            lines.append(f"{indent}%{name} = {value.ffi_shape} copy(%{value.instruction})")
        names[value.role] = name
    output_shapes = ", ".join(value.ffi_shape for value in plan.outputs)
    operands = ", ".join(
        f"%{names[role]}"
        for role in (
            StreamingReverseHloRole.QUERY,
            StreamingReverseHloRole.KEY,
            StreamingReverseHloRole.VALUE,
            StreamingReverseHloRole.OUTPUT_COTANGENT,
        )
    )
    constraints = ", ".join(value.ffi_shape for value in plan.inputs)
    call_name = "shuttle.generated.streaming_reverse"
    lines.append(
        f"{indent}%{call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    output_names: dict[StreamingReverseHloRole, str] = {}
    for index, value in enumerate(plan.outputs):
        canonical_name = f"shuttle.{value.role.value}.ffi"
        get_tuple_element = f"get-tuple-element(%{call_name}), index={index}"
        lines.append(f"{indent}%{canonical_name} = {value.ffi_shape} {get_tuple_element}")
        output_name = canonical_name
        if value.physical_shape != value.ffi_shape:
            output_name = f"shuttle.{value.role.value}.physical"
            lines.append(f"{indent}%{output_name} = {value.physical_shape} copy(%{canonical_name})")
        output_names[value.role] = output_name
    root_operands = ", ".join(f"%{output_names[value.role]}" for value in plan.outputs)
    lines.append(f"{indent}ROOT %{plan.root_instruction} = {plan.root_shape} tuple({root_operands})")
    replacement = "\n".join(lines)
    return hlo_text[: match.start()] + replacement + hlo_text[match.end() :]


def _plan_reverse_region_from_maximum(
    ordered_instructions: tuple[HloInstruction, ...],
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
    module: HloModuleGraph,
    maximum_fold: HloInstruction,
    specifications: dict[str, StreamingAttentionBackwardFfiBuffer],
    *,
    expected_scale: float,
    expected_restriction: bool,
    generated: GeneratedStreamingAttentionBackwardFfi,
    program: StreamingAttentionBackwardProgram,
) -> StreamingReverseHloRegionReplacementPlan:
    nearest_ancestors = _nearest_entry_opcode_ancestor_function(instructions)
    score_contracts = nearest_ancestors(maximum_fold.operands[0], "dot")
    if len(score_contracts) != 1:
        raise ValueError(f"maximum Fold has {len(score_contracts)} nearest Contracts")
    score_contract = instructions[next(iter(score_contracts))]
    if len(score_contract.operands) != 2:
        raise ValueError("score Contract must have two operands")
    query_name, key_name, physical_scale = _bind_region_score_inputs(
        score_contract,
        instructions,
        specifications["query"],
        specifications["key"],
    )
    # XLA's HLO printer may abbreviate BF16 constants (for example 0.32421875
    # as 0.3242), so compare at the precision retained by the physical dtype.
    if not math.isclose(physical_scale, expected_scale, rel_tol=5e-4, abs_tol=1e-12):
        raise ValueError(f"physical score scale {physical_scale} does not match recovered scale {expected_scale}")

    score_slice = _entry_ancestor_slice(maximum_fold.operands[0], instructions, stop={score_contract.name})
    restrictions = tuple(instructions[name] for name in score_slice if instructions[name].opcode == "select")
    comparisons = tuple(instructions[name] for name in score_slice if instructions[name].opcode == "compare")
    if expected_restriction:
        if len(restrictions) != 1 or len(comparisons) != 1:
            raise ValueError("causal DomainRestriction must lower to one select and one comparison")
        if "direction=LE" not in comparisons[0].attributes:
            raise ValueError("only less-equal DomainRestriction is supported by the generated skeleton")
    elif restrictions or comparisons:
        raise ValueError("physical HLO has a DomainRestriction absent from the recovered program")

    exponentials = tuple(
        instructions[name]
        for name in _first_entry_opcode_users(
            maximum_fold.name,
            instructions,
            users,
            opcode="exponential",
        )
        if score_contract.name in _entry_ancestors(name, instructions)
    )
    if len(exponentials) != 1:
        raise ValueError(f"expected one normalized-exponential Map, found {len(exponentials)}")
    exponential = exponentials[0]
    probability_candidates = tuple(
        instruction
        for instruction in ordered_instructions
        if instruction.opcode == "divide"
        and _shape_signature(instruction.shape)[1] == _shape_signature(score_contract.shape)[1]
        and exponential.name in _entry_ancestors(instruction.name, instructions)
        and nearest_ancestors(instruction.name, "dot") == frozenset({score_contract.name})
    )
    if len(probability_candidates) != 1:
        raise ValueError(f"expected one normalized probability value, found {len(probability_candidates)}")
    probability = probability_candidates[0]
    normalized_sum_folds = tuple(
        instruction
        for instruction in ordered_instructions
        if instruction.opcode == "reduce"
        and _entry_fold_reducer(module, instruction) == "add"
        and exponential.name in _entry_ancestors(instruction.name, instructions)
        and instruction.name in _entry_ancestors(probability.name, instructions)
    )
    if len(normalized_sum_folds) != 1:
        raise ValueError(f"expected one normalized-exponential sum Fold, found {len(normalized_sum_folds)}")

    score_shaped_contracts = tuple(
        instruction
        for instruction in ordered_instructions
        if instruction.opcode == "dot"
        and instruction.name != score_contract.name
        and _shape_signature(instruction.shape) == _shape_signature(score_contract.shape)
        and source_order[instruction.name] > source_order[probability.name]
    )
    reverse_score_candidates: list[tuple[HloInstruction, str, str]] = []
    for contract in score_shaped_contracts:
        if len(contract.operands) != 2:
            continue
        for output_operand, value_operand in (contract.operands, tuple(reversed(contract.operands))):
            output_boundary = _closest_compatible_ancestor(
                output_operand,
                instructions,
                specifications["output_cotangent"],
                allow_singleton_elision=False,
            )
            value_boundary = _closest_compatible_ancestor(
                value_operand,
                instructions,
                specifications["value"],
                allow_singleton_elision=True,
            )
            if output_boundary is not None and value_boundary is not None:
                reverse_score_candidates.append((contract, output_boundary, value_boundary))
    if len(reverse_score_candidates) != 1:
        raise ValueError(f"expected one output-cotangent/value Contract, found {len(reverse_score_candidates)}")
    reverse_score_contract, output_cotangent_name, value_name = reverse_score_candidates[0]

    first_reverse_contracts = _first_entry_opcode_users(
        reverse_score_contract.name,
        instructions,
        users,
        opcode="dot",
    )
    if len(first_reverse_contracts) != 2:
        raise ValueError(f"score Map reverse must feed two nearest Contracts, found {len(first_reverse_contracts)}")
    contract_external_operands: dict[str, str] = {}
    for name in first_reverse_contracts:
        contract = instructions[name]
        score_paths = tuple(
            reverse_score_contract.name in _entry_ancestors(operand, instructions) for operand in contract.operands
        )
        if score_paths.count(True) != 1:
            raise ValueError(f"terminal Contract %{name} does not have one score-cotangent operand")
        contract_external_operands[name] = contract.operands[score_paths.index(False)]
    query_ancestor_contracts = tuple(
        name
        for name, operand in contract_external_operands.items()
        if query_name in _entry_ancestors(operand, instructions)
    )
    key_ancestor_contracts = tuple(
        name
        for name, operand in contract_external_operands.items()
        if key_name in _entry_ancestors(operand, instructions)
    )
    if len(query_ancestor_contracts) != 1 or len(key_ancestor_contracts) != 1:
        raise ValueError("could not distinguish query- and key-dependent reverse Contracts")
    key_contract = query_ancestor_contracts[0]
    query_contract = key_ancestor_contracts[0]
    if key_contract == query_contract:
        raise ValueError("query and key cotangents require distinct terminal Contracts")

    probability_contracts = _first_entry_opcode_users(
        probability.name,
        instructions,
        users,
        opcode="dot",
    )
    value_contracts = tuple(
        name for name in probability_contracts if output_cotangent_name in _entry_ancestors(name, instructions)
    )
    if len(value_contracts) != 1:
        raise ValueError(f"expected one probability/output-cotangent Contract, found {len(value_contracts)}")
    value_contract = value_contracts[0]

    output_bindings = (
        (
            StreamingReverseHloRole.QUERY_COTANGENT,
            _unique_compatible_descendant(
                query_contract,
                instructions,
                users,
                specifications["query_cotangent"],
                allow_singleton_elision=False,
            ),
        ),
        (
            StreamingReverseHloRole.KEY_COTANGENT,
            _unique_compatible_descendant(
                key_contract,
                instructions,
                users,
                specifications["key_cotangent"],
                allow_singleton_elision=False,
            ),
        ),
        (
            StreamingReverseHloRole.VALUE_COTANGENT,
            _unique_compatible_descendant(
                value_contract,
                instructions,
                users,
                specifications["value_cotangent"],
                allow_singleton_elision=True,
            ),
        ),
    )
    output_names = {name for _, name in output_bindings}
    if len(output_names) != 3:
        raise ValueError("reverse cotangent boundaries are not distinct")

    output_ancestors = _entry_ancestors_many(output_names, instructions)
    cotangent_descendants = _entry_descendants(output_cotangent_name, users)
    internal = (output_ancestors & cotangent_descendants) - output_names - {output_cotangent_name}
    required_contracts = {reverse_score_contract.name, query_contract, key_contract, value_contract}
    if not required_contracts <= internal:
        raise ValueError("reverse-only closure omits a required Contract")
    unsupported = tuple(name for name in internal if instructions[name].opcode not in _REGION_REVERSE_OPCODES)
    if unsupported:
        raise ValueError(f"reverse-only closure contains unsupported operations: {sorted(unsupported)}")
    _validate_region_effect_and_control_safety(internal | output_names, instructions)

    external_users: list[tuple[str, tuple[str, ...]]] = []
    for name in sorted(internal, key=source_order.__getitem__):
        crossing = tuple(user for user in users[name] if user not in internal and user not in output_names)
        if crossing:
            raise ValueError(f"reverse internal value %{name} has external users {crossing}")
    for _, name in output_bindings:
        crossing = tuple(user for user in users[name] if user not in internal and user not in output_names)
        if not crossing:
            raise ValueError(f"reverse output %{name} has no external user")
        external_users.append((name, crossing))

    role_names = {
        StreamingReverseHloRole.QUERY: query_name,
        StreamingReverseHloRole.KEY: key_name,
        StreamingReverseHloRole.VALUE: value_name,
        StreamingReverseHloRole.OUTPUT_COTANGENT: output_cotangent_name,
    }
    inputs = tuple(
        StreamingReverseHloValue(
            role=role,
            instruction=role_names[role],
            physical_shape=instructions[role_names[role]].shape,
            ffi_shape=_ffi_shape(specifications[role.value]),
        )
        for role in (
            StreamingReverseHloRole.QUERY,
            StreamingReverseHloRole.KEY,
            StreamingReverseHloRole.VALUE,
            StreamingReverseHloRole.OUTPUT_COTANGENT,
        )
    )
    outputs = tuple(
        StreamingReverseHloValue(
            role=role,
            instruction=name,
            physical_shape=instructions[name].shape,
            ffi_shape=_ffi_shape(specifications[role.value]),
        )
        for role, name in output_bindings
    )
    for value in (*inputs, *outputs):
        _boundary_adapter_opcode(value.physical_shape, value.ffi_shape)

    insertion_instruction = max((value.instruction for value in inputs), key=source_order.__getitem__)
    first_internal = min(internal | output_names, key=source_order.__getitem__)
    if source_order[insertion_instruction] >= source_order[first_internal]:
        raise ValueError("all generated inputs must dominate the reverse-only closure")
    shared_inputs = {
        operand
        for name in internal | output_names
        for operand in instructions[name].operands
        if operand not in internal
        and operand not in output_names
        and operand != output_cotangent_name
        and instructions[operand].opcode != "constant"
        and any(user not in internal and user not in output_names for user in users[operand])
    }
    additive_folds = tuple(
        name
        for name in sorted(internal, key=source_order.__getitem__)
        if instructions[name].opcode == "reduce" and _entry_fold_reducer(module, instructions[name]) == "add"
    )
    if len(additive_folds) < 2:
        raise ValueError("reverse-only closure lacks the required additive Folds")
    return StreamingReverseHloRegionReplacementPlan(
        inputs=inputs,
        outputs=outputs,
        insertion_instruction=insertion_instruction,
        internal_instructions=tuple(sorted(internal, key=source_order.__getitem__)),
        preserved_shared_inputs=tuple(sorted(shared_inputs, key=source_order.__getitem__)),
        external_users=tuple(external_users),
        provenance=StreamingReverseHloProvenance(
            score_contract=score_contract.name,
            reverse_contracts=(
                reverse_score_contract.name,
                value_contract,
                query_contract,
                key_contract,
            ),
            maximum_fold=maximum_fold.name,
            additive_folds=additive_folds,
            domain_restriction=restrictions[0].name if restrictions else None,
            score_scale=physical_scale,
        ),
        state_policy=generated.state_policy,
        semantic_fingerprint=generated.semantic_fingerprint,
        maximum_vjp=program.maximum_vjp.value,
        reassociation=program.reassociation.value,
    )


def _validate_generated_reverse_signature(
    generated: GeneratedStreamingAttentionBackwardFfi,
    *,
    boundary: str,
) -> None:
    if generated.state_policy is not StreamingAttentionBackwardStatePolicy.RECOMPUTE:
        raise ValueError(f"{boundary} reverse replacement requires explicit recompute state policy")
    if tuple(value.name for value in generated.inputs) != ("query", "key", "value", "output_cotangent"):
        raise ValueError(f"{boundary} reverse replacement requires the natural four-buffer input signature")
    if tuple(value.name for value in generated.outputs) != (
        "query_cotangent",
        "key_cotangent",
        "value_cotangent",
    ):
        raise ValueError(f"{boundary} reverse replacement requires three input-cotangent outputs")


def _entry_users(instructions: tuple[HloInstruction, ...]) -> dict[str, tuple[str, ...]]:
    users: dict[str, list[str]] = {instruction.name: [] for instruction in instructions}
    for instruction in instructions:
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)
    return {name: tuple(values) for name, values in users.items()}


def _entry_fold_reducer(module: HloModuleGraph, instruction: HloInstruction) -> str | None:
    match = _CALLED_COMPUTATION.search(instruction.attributes)
    if match is None:
        return None
    return module.computation(match.group("name")).root.opcode


def _entry_ancestors(name: str, instructions: dict[str, HloInstruction]) -> frozenset[str]:
    return _entry_ancestors_many({name}, instructions)


def _entry_ancestors_many(names: Iterable[str], instructions: dict[str, HloInstruction]) -> frozenset[str]:
    pending = list(names)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(instructions[name].operands)
    return frozenset(seen)


def _entry_descendants(name: str, users: dict[str, tuple[str, ...]]) -> frozenset[str]:
    pending = [name]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(users[current])
    return frozenset(seen)


def _nearest_entry_opcode_ancestor_function(
    instructions: dict[str, HloInstruction],
) -> Callable[[str, str], frozenset[str]]:
    memo: dict[tuple[str, str], frozenset[str]] = {}

    def nearest(name: str, opcode: str) -> frozenset[str]:
        key = (name, opcode)
        if key in memo:
            return memo[key]
        instruction = instructions[name]
        if instruction.opcode == opcode:
            result = frozenset({name})
        else:
            result = frozenset().union(*(nearest(operand, opcode) for operand in instruction.operands))
        memo[key] = result
        return result

    return nearest


def _first_entry_opcode_users(
    name: str,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    *,
    opcode: str,
) -> frozenset[str]:
    pending = list(users[name])
    seen: set[str] = set()
    found: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        instruction = instructions[current]
        if instruction.opcode == opcode:
            found.add(current)
            continue
        if instruction.opcode in _REGION_REVERSE_OPCODES:
            pending.extend(users[current])
    return frozenset(found)


def _entry_ancestor_slice(
    name: str,
    instructions: dict[str, HloInstruction],
    *,
    stop: set[str],
) -> frozenset[str]:
    pending = [name]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen or current in stop:
            continue
        seen.add(current)
        pending.extend(instructions[current].operands)
    return frozenset(seen)


def _bind_region_score_inputs(
    score_contract: HloInstruction,
    instructions: dict[str, HloInstruction],
    query: StreamingAttentionBackwardFfiBuffer,
    key: StreamingAttentionBackwardFfiBuffer,
) -> tuple[str, str, float]:
    bindings: list[tuple[str, str, float]] = []
    for query_operand, key_operand in (score_contract.operands, tuple(reversed(score_contract.operands))):
        query_boundary, query_scale = _scaled_boundary_ancestor(query_operand, instructions, query)
        key_boundary, key_scale = _scaled_boundary_ancestor(key_operand, instructions, key)
        if query_boundary is not None and key_boundary is not None:
            bindings.append((query_boundary, key_boundary, query_scale * key_scale))
    if len(bindings) != 1:
        raise ValueError(f"could not uniquely bind score Contract Q/K operands, found {len(bindings)}")
    return bindings[0]


def _scaled_boundary_ancestor(
    name: str,
    instructions: dict[str, HloInstruction],
    specification: StreamingAttentionBackwardFfiBuffer,
) -> tuple[str | None, float]:
    current = name
    scale = 1.0
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        instruction = instructions[current]
        if instruction.opcode == "multiply" and len(instruction.operands) == 2:
            scalar_edges = tuple(
                (value, constant)
                for value, scalar in (
                    (instruction.operands[0], instruction.operands[1]),
                    (instruction.operands[1], instruction.operands[0]),
                )
                if (constant := _entry_broadcast_scalar_constant(scalar, instructions)) is not None
                and math.isfinite(constant)
            )
            if len(scalar_edges) == 1:
                value, constant = scalar_edges[0]
                scale *= constant
                current = value
                continue
        if _buffer_compatible_shape(instruction.shape, specification, allow_singleton_elision=False):
            return current, scale
        if instruction.opcode not in {"broadcast", "convert", "copy", "reshape", "transpose"}:
            break
        if len(instruction.operands) != 1:
            break
        current = instruction.operands[0]
    return None, 1.0


def _entry_broadcast_scalar_constant(name: str, instructions: dict[str, HloInstruction]) -> float | None:
    instruction = instructions[name]
    if instruction.opcode == "constant":
        match = _CONSTANT.search(instruction.attributes)
        return float(match.group("value")) if match is not None else None
    if instruction.opcode in {"broadcast", "copy", "reshape"} and len(instruction.operands) == 1:
        return _entry_broadcast_scalar_constant(instruction.operands[0], instructions)
    return None


def _closest_compatible_ancestor(
    name: str,
    instructions: dict[str, HloInstruction],
    specification: StreamingAttentionBackwardFfiBuffer,
    *,
    allow_singleton_elision: bool,
) -> str | None:
    pending = [(name, 0)]
    seen: set[str] = set()
    matches: list[str] = []
    match_depth: int | None = None
    while pending:
        current, depth = pending.pop(0)
        if current in seen:
            continue
        if match_depth is not None and depth > match_depth:
            break
        seen.add(current)
        instruction = instructions[current]
        if _buffer_compatible_shape(
            instruction.shape,
            specification,
            allow_singleton_elision=allow_singleton_elision,
        ):
            match_depth = depth
            matches.append(current)
            continue
        if instruction.opcode not in _REGION_MAP_OPCODES or instruction.opcode in {
            "add",
            "compare",
            "constant",
            "divide",
            "exponential",
            "select",
            "subtract",
        }:
            continue
        pending.extend((operand, depth + 1) for operand in instruction.operands)
    if len(matches) > 1:
        raise ValueError(f"ambiguous compatible ancestors at depth {match_depth}: {matches}")
    return matches[0] if matches else None


def _unique_compatible_descendant(
    name: str,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    specification: StreamingAttentionBackwardFfiBuffer,
    *,
    allow_singleton_elision: bool,
) -> str:
    pending = list(users[name])
    seen: set[str] = set()
    depth: dict[str, int] = {value: 1 for value in pending}
    matches: list[tuple[int, str]] = []
    while pending:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        instruction = instructions[current]
        if _buffer_compatible_shape(
            instruction.shape,
            specification,
            allow_singleton_elision=allow_singleton_elision,
        ):
            matches.append((depth[current], current))
            continue
        if instruction.opcode not in _REGION_REVERSE_OPCODES or instruction.opcode == "dot":
            continue
        for user in users[current]:
            depth.setdefault(user, depth[current] + 1)
            pending.append(user)
    if not matches:
        raise ValueError(f"Contract %{name} has no compatible cotangent boundary")
    minimum_depth = min(depth_value for depth_value, _ in matches)
    nearest = tuple(value for depth_value, value in matches if depth_value == minimum_depth)
    if len(nearest) != 1:
        raise ValueError(f"Contract %{name} has ambiguous cotangent boundaries {nearest}")
    return nearest[0]


def _buffer_compatible_shape(
    shape: str,
    specification: StreamingAttentionBackwardFfiBuffer,
    *,
    allow_singleton_elision: bool,
) -> bool:
    dtype, dimensions = _shape_signature(shape)
    if dtype != specification.dtype.value:
        return False
    if dimensions == specification.shape:
        return True
    if not allow_singleton_elision:
        return False
    return tuple(value for value in dimensions if value != 1) == tuple(
        value for value in specification.shape if value != 1
    )


def _validate_region_effect_and_control_safety(
    region: set[str],
    instructions: dict[str, HloInstruction],
) -> None:
    side_effects = tuple(
        name
        for name in region
        if instructions[name].opcode in {"infeed", "outfeed", "recv", "send"}
        or "custom_call_has_side_effect=true" in instructions[name].attributes
    )
    if side_effects:
        raise ValueError(f"reverse-only closure contains side effects: {sorted(side_effects)}")
    for instruction in instructions.values():
        match = _CONTROL_PREDECESSORS.search(instruction.attributes)
        if match is None:
            continue
        predecessors = set(re.findall(r"%?([A-Za-z0-9_.-]+)", match.group("values")))
        if (instruction.name in region) != bool(predecessors & region):
            raise ValueError("reverse-only closure crosses an explicit control dependency")


def _boundary_adapter_opcode(source_shape: str, target_shape: str) -> str | None:
    if source_shape == target_shape:
        return None
    source_dtype, source_dimensions = _shape_signature(source_shape)
    target_dtype, target_dimensions = _shape_signature(target_shape)
    if source_dtype != target_dtype:
        raise ValueError(f"boundary adapter cannot change dtype: {source_shape} -> {target_shape}")
    if source_dimensions == target_dimensions:
        return "copy"
    if tuple(value for value in source_dimensions if value != 1) == tuple(
        value for value in target_dimensions if value != 1
    ):
        return "reshape"
    raise ValueError(f"boundary adapter cannot reorder logical dimensions: {source_shape} -> {target_shape}")


def _emit_boundary_adapter(
    lines: list[str],
    *,
    indent: str,
    source: str,
    source_shape: str,
    target_shape: str,
    name: str,
) -> str:
    opcode = _boundary_adapter_opcode(source_shape, target_shape)
    if opcode is None:
        return source
    lines.append(f"{indent}%{name} = {target_shape} {opcode}(%{source})")
    return name


def _replace_entry_operand(hlo_text: str, *, user: str, old: str, new: str) -> str:
    pattern = re.compile(
        rf"^(?P<prefix>\s*(?:ROOT\s+)?%?{re.escape(user)} = )(?P<body>.*?)$",
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one external user definition for {user!r}")
    match = matches[0]
    body = match.group("body")
    operand_pattern = re.compile(rf"%{re.escape(old)}(?![A-Za-z0-9_.-])")
    replaced_body, count = operand_pattern.subn(f"%{new}", body)
    if count == 0:
        raise ValueError(f"external user %{user} does not consume %{old}")
    return hlo_text[: match.start("body")] + replaced_body + hlo_text[match.end("body") :]


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
            ffi_shape=_ffi_shape(specifications[role.value]),
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
        ffi_shape=_ffi_shape(specification),
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


def _ffi_shape(specification: StreamingAttentionBackwardFfiBuffer) -> str:
    dimensions = ",".join(str(value) for value in specification.shape)
    layout = ",".join(str(value) for value in specification.layout)
    return f"{specification.dtype.value}[{dimensions}]{{{layout}}}"


def _shape_layout(shape: str) -> tuple[int, ...]:
    match = _ARRAY_SHAPE.fullmatch(shape.strip())
    if match is None or match.group("layout") is None:
        raise ValueError(f"physical HLO output requires one explicit dense layout: {shape!r}")
    layout = tuple(int(value) for value in match.group("layout").split(","))
    rank = len(tuple(value for value in match.group("dims").split(",") if value))
    if tuple(sorted(layout)) != tuple(range(rank)):
        raise ValueError(f"physical HLO output layout is not a rank-{rank} permutation: {layout}")
    return layout


def _shape_signature(shape: str) -> tuple[str, tuple[int, ...]]:
    match = _ARRAY_SHAPE.fullmatch(shape.strip())
    if match is None:
        raise ValueError(f"unsupported physical HLO array shape: {shape!r}")
    return match.group("dtype"), tuple(int(value) for value in match.group("dims").split(",") if value)


def _buffer_signature(specification: StreamingAttentionBackwardFfiBuffer) -> tuple[str, tuple[int, ...]]:
    return specification.dtype.value, specification.shape
