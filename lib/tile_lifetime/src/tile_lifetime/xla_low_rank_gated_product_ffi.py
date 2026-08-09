# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Form exact typed-FFI boundaries for generic low-rank Contract/Map chains."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, replace

from tile_lifetime.cast_scalar_program import CastScalarProgram
from tile_lifetime.xla_hlo_recovery import (
    EntryRegionValue,
    HloComputation,
    HloInstruction,
    InlinedHloGraph,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)
from tile_lifetime.xla_low_rank_gated_product import (
    LowRankGatedProductForwardPlan,
    LowRankGatedProductReversePlan,
    recover_low_rank_gated_product_training,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_map

_LAYOUT_WRAPPERS = frozenset({"bitcast", "copy", "reshape", "transpose"})
_CONSTANT_DATAFLOW = frozenset({"bitcast", "broadcast", "copy", "convert", "reshape", "transpose"})
_DOT_DIMENSION = re.compile(
    r"(?P<name>lhs_contracting_dims|rhs_contracting_dims|lhs_batch_dims|rhs_batch_dims)=" r"\{(?P<dims>[0-9,]*)\}"
)


@dataclass(frozen=True)
class LowRankContractMapForwardHloReplacementPlan:
    """One maximal bounded forward/rematerialization Contract/Map call."""

    call_name: str
    semantics: LowRankGatedProductForwardPlan
    inputs: tuple[EntryRegionValue, ...]
    outputs: tuple[EntryRegionValue, ...]
    internal_instructions: tuple[str, ...]
    insertion_instruction: str
    dot_instructions: tuple[str, ...]
    dot_flops: int
    scalar_programs: tuple[CastScalarProgram, ...]
    boundary_family_digest: str
    semantic_digest: str
    api_version: int


@dataclass(frozen=True)
class LowRankContractMapReverseHloReplacementPlan:
    """One maximal bounded JAX-owned reverse Contract/Map call."""

    call_name: str
    semantics: LowRankGatedProductReversePlan
    inputs: tuple[EntryRegionValue, ...]
    outputs: tuple[EntryRegionValue, ...]
    internal_instructions: tuple[str, ...]
    insertion_instruction: str
    dot_instructions: tuple[str, ...]
    dot_flops: int
    scalar_programs: tuple[CastScalarProgram, ...]
    cotangent_inputs: tuple[EntryRegionValue, ...]
    upstream_collectives: tuple[EntryRegionValue, ...]
    boundary_family_digest: str
    semantic_digest: str
    api_version: int


@dataclass(frozen=True)
class LowRankContractMapTrainingHloReplacementPlan:
    """All disjoint generic forward/rematerialization and reverse calls."""

    forward: tuple[LowRankContractMapForwardHloReplacementPlan, ...]
    reverse: tuple[LowRankContractMapReverseHloReplacementPlan, ...]
    original_live_dot_count: int
    original_live_dot_flops: int
    replaced_dot_count: int
    replaced_dot_flops: int


@dataclass(frozen=True)
class LowRankContractMapCallAudit:
    """Exact ABI and dead arithmetic evidence for one generated call."""

    call_instruction: str
    target: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    output_users: tuple[tuple[str, tuple[str, ...]], ...]
    dead_internal_instructions: tuple[str, ...]
    replaced_output_instructions: tuple[str, ...]
    removed_dot_instructions: tuple[str, ...]
    removed_dot_flops: int
    api_version: int


@dataclass(frozen=True)
class LowRankContractMapTrainingHloReplacementAudit:
    """Composition-level liveness, collective, and ABI evidence."""

    forward: tuple[LowRankContractMapCallAudit, ...]
    reverse: tuple[LowRankContractMapCallAudit, ...]
    generated_call_count: int
    removed_dot_count: int
    removed_dot_flops: int
    collective_instructions: tuple[str, ...]
    upstream_collective_paths: tuple[tuple[str, tuple[str, ...], str], ...]
    live_old_arithmetic: tuple[str, ...]


def plan_low_rank_contract_map_training_hlo_replacements(
    hlo_text: str,
) -> LowRankContractMapTrainingHloReplacementPlan:
    """Recover ten bounded calls from generic Contract/Map structure."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    graph = inline_elementwise_fusions(module)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _users(entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    report = recover_low_rank_gated_product_training(hlo_text)

    forward = tuple(
        _plan_forward_boundary(index, semantics, instructions, users, source_order)
        for index, semantics in enumerate(report.forward_realizations)
    )
    forward_by_first_contract = {plan.semantics.down_contract.instruction: plan for plan in forward}
    reverse = tuple(
        _plan_reverse_boundary(
            index,
            semantics,
            forward_by_first_contract[semantics.primal.down_contract.instruction],
            graph,
            instructions,
            users,
            source_order,
        )
        for index, semantics in enumerate(report.reverse_families)
    )
    all_internal = [set(plan.internal_instructions) for plan in (*forward, *reverse)]
    for left, right in itertools.combinations(all_internal, 2):
        overlap = left & right
        if overlap:
            raise ValueError(f"low-rank Contract/Map replacement boundaries overlap: {sorted(overlap)}")
    replaced_dots = tuple(dot for plan in (*forward, *reverse) for dot in plan.dot_instructions)
    if len(set(replaced_dots)) != len(replaced_dots):
        raise ValueError("one physical Contract was assigned to multiple generated calls")
    return LowRankContractMapTrainingHloReplacementPlan(
        forward=forward,
        reverse=reverse,
        original_live_dot_count=report.live_contract_count,
        original_live_dot_flops=report.live_contract_flops,
        replaced_dot_count=len(replaced_dots),
        replaced_dot_flops=sum(plan.dot_flops for plan in (*forward, *reverse)),
    )


def replace_low_rank_contract_map_training_hlo_regions_with_custom_calls(
    hlo_text: str,
    plan: LowRankContractMapTrainingHloReplacementPlan,
    *,
    forward_target: str,
    reverse_target: str,
) -> str:
    """Replace every disjoint boundary with one generic typed-FFI call."""
    rewritten = hlo_text
    for boundary in plan.forward:
        rewritten = _replace_boundary(rewritten, boundary, target=forward_target)
    for boundary in plan.reverse:
        rewritten = _replace_boundary(rewritten, boundary, target=reverse_target)
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_low_rank_contract_map_training_hlo_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: LowRankContractMapTrainingHloReplacementPlan,
    *,
    forward_target: str,
    reverse_target: str,
) -> LowRankContractMapTrainingHloReplacementAudit:
    """Prove exact calls, dead old work, and unchanged placement collectives."""
    original_entry = _entry(original_hlo)
    transformed_entry = _entry(transformed_hlo)
    original_instructions = {instruction.name: instruction for instruction in original_entry.instructions}
    transformed_instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    live = _live_instructions(transformed_entry)
    all_boundaries = (*plan.forward, *plan.reverse)
    all_internal = set().union(*(set(boundary.internal_instructions) for boundary in all_boundaries))
    generated_consumers: dict[str, set[str]] = {}
    for boundary in all_boundaries:
        for value in boundary.inputs:
            generated_consumers.setdefault(value.instruction, set()).add(boundary.call_name)
    expected_users = {
        output.instruction: {
            *(user for user in _users(original_entry)[output.instruction] if user not in all_internal),
            *generated_consumers.get(output.instruction, set()),
        }
        for boundary in all_boundaries
        for output in boundary.outputs
    }
    forward = tuple(
        _audit_call(
            transformed_entry,
            boundary,
            target=forward_target,
            expected_users=expected_users,
        )
        for boundary in plan.forward
    )
    reverse = tuple(
        _audit_call(
            transformed_entry,
            boundary,
            target=reverse_target,
            expected_users=expected_users,
        )
        for boundary in plan.reverse
    )
    collective_names = tuple(
        instruction.name for instruction in original_entry.instructions if instruction.opcode == "all-reduce"
    )
    transformed_collectives = tuple(
        instruction.name for instruction in transformed_entry.instructions if instruction.opcode == "all-reduce"
    )
    if transformed_collectives != collective_names:
        raise ValueError("low-rank Contract/Map replacement changed placement all-reduces")
    if any(original_instructions[name].opcode == "all-reduce" for name in all_internal):
        raise ValueError("a placement all-reduce entered a generated Contract/Map boundary")
    upstream_paths: list[tuple[str, tuple[str, ...], str]] = []
    for boundary in plan.reverse:
        for collective in boundary.upstream_collectives:
            if collective.instruction not in transformed_instructions:
                raise ValueError(f"upstream collective %{collective.instruction} was removed")
            cotangents = tuple(
                value.instruction
                for value in boundary.cotangent_inputs
                if _depends_on(value.instruction, collective.instruction, transformed_instructions)
            )
            if cotangents:
                upstream_paths.append((boundary.call_name, cotangents, collective.instruction))
    old_arithmetic = tuple(
        name
        for boundary in (*plan.forward, *plan.reverse)
        for name in boundary.internal_instructions
        if name not in {output.instruction for output in boundary.outputs}
    )
    live_old = tuple(name for name in old_arithmetic if name in live)
    if live_old:
        raise ValueError(f"old low-rank Contract/Map arithmetic remains live: {live_old}")
    removed_dots = sum(len(call.removed_dot_instructions) for call in (*forward, *reverse))
    removed_flops = sum(call.removed_dot_flops for call in (*forward, *reverse))
    if removed_dots != plan.replaced_dot_count or removed_flops != plan.replaced_dot_flops:
        raise ValueError("replacement audit changed the planned Contract accounting")
    return LowRankContractMapTrainingHloReplacementAudit(
        forward=forward,
        reverse=reverse,
        generated_call_count=len(forward) + len(reverse),
        removed_dot_count=removed_dots,
        removed_dot_flops=removed_flops,
        collective_instructions=collective_names,
        upstream_collective_paths=tuple(upstream_paths),
        live_old_arithmetic=live_old,
    )


def mutate_forward_hidden_scalar_program(
    plan: LowRankContractMapForwardHloReplacementPlan,
    program: CastScalarProgram,
) -> LowRankContractMapForwardHloReplacementPlan:
    """Change one scalar Map while retaining the physical boundary family."""
    semantics = replace(plan.semantics, hidden_map=program)
    scalar_programs = (program, semantics.output_map)
    return replace(
        plan,
        semantics=semantics,
        scalar_programs=scalar_programs,
        semantic_digest=_semantic_digest(plan.boundary_family_digest, scalar_programs),
    )


def _plan_forward_boundary(
    index: int,
    semantics: LowRankGatedProductForwardPlan,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
) -> LowRankContractMapForwardHloReplacementPlan:
    internal = {
        semantics.down_contract.instruction,
        semantics.up_contract.instruction,
        *_path_interval(
            semantics.hidden.instruction,
            {semantics.down_contract.instruction},
            instructions,
        ),
        *_path_interval(
            semantics.output.instruction,
            {semantics.input.instruction, semantics.up_contract.instruction},
            instructions,
        ),
    }
    outputs = _boundary_outputs(internal, instructions, users, source_order)
    inputs = _boundary_inputs(internal, instructions, source_order)
    insertion = _insertion_instruction(internal, inputs, outputs, users, source_order)
    _verify_boundary(internal, inputs, outputs, instructions, users)
    scalar_programs = (semantics.hidden_map, semantics.output_map)
    dot_instructions = tuple(
        name for name in sorted(internal, key=source_order.__getitem__) if instructions[name].opcode == "dot"
    )
    family_digest = _family_digest("forward", inputs, outputs, dot_instructions, instructions)
    return LowRankContractMapForwardHloReplacementPlan(
        call_name=f"shuttle.generated.low_rank_contract_map.forward.{index}",
        semantics=semantics,
        inputs=inputs,
        outputs=outputs,
        internal_instructions=tuple(sorted(internal, key=source_order.__getitem__)),
        insertion_instruction=insertion,
        dot_instructions=dot_instructions,
        dot_flops=semantics.down_contract.flops + semantics.up_contract.flops,
        scalar_programs=scalar_programs,
        boundary_family_digest=family_digest,
        semantic_digest=_semantic_digest(family_digest, scalar_programs),
        api_version=1,
    )


def _plan_reverse_boundary(
    index: int,
    semantics: LowRankGatedProductReversePlan,
    forward: LowRankContractMapForwardHloReplacementPlan,
    graph: InlinedHloGraph,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
) -> LowRankContractMapReverseHloReplacementPlan:
    saved = {output.instruction for output in forward.outputs}
    forward_internal = set(forward.internal_instructions)
    cotangent_sources = tuple(
        value.instruction
        for value in semantics.up_input_sources
        if value.instruction != semantics.primal.input.instruction and value.instruction not in forward_internal
    )
    up_core = _strip_layout_wrappers(semantics.up_input_adjoint.lhs.instruction, instructions)
    hidden_core = _strip_layout_wrappers(semantics.down_input_adjoint.lhs.instruction, instructions)
    up_sources, up_map = _select_saved_scalar_program(
        graph,
        candidates=(semantics.primal.input.instruction, *saved, *cotangent_sources),
        target=up_core,
        saved=saved,
        source_order=source_order,
    )
    hidden_sources, hidden_map = _select_saved_scalar_program(
        graph,
        candidates=(*saved, semantics.up_input_adjoint.instruction),
        target=hidden_core,
        saved=saved,
        source_order=source_order,
    )
    residual_sources, residual_map = _select_saved_scalar_program(
        graph,
        candidates=(*saved, semantics.down_input_adjoint.instruction, *cotangent_sources),
        target=semantics.input_adjoint.instruction,
        saved=saved,
        source_order=source_order,
    )
    internal = {
        *_path_interval(up_core, set(up_sources), instructions),
        *_path_interval(semantics.up_input_adjoint.lhs.instruction, {up_core}, instructions),
        semantics.up_input_adjoint.instruction,
        *_path_interval(hidden_core, set(hidden_sources), instructions),
        *_path_interval(semantics.down_input_adjoint.lhs.instruction, {hidden_core}, instructions),
        semantics.down_input_adjoint.instruction,
        *_path_interval(semantics.input_adjoint.instruction, set(residual_sources), instructions),
        *_path_interval(semantics.down_weight_adjoint.rhs.instruction, {hidden_core}, instructions),
        semantics.down_weight_adjoint.instruction,
        *_path_interval(semantics.up_weight_adjoint.rhs.instruction, {up_core}, instructions),
        semantics.up_weight_adjoint.instruction,
    }
    outputs = (
        semantics.input_adjoint,
        semantics.down_weight_adjoint.output,
        semantics.up_weight_adjoint.output,
    )
    inputs = _boundary_inputs(internal, instructions, source_order)
    insertion = _insertion_instruction(internal, inputs, outputs, users, source_order)
    _verify_boundary(internal, inputs, outputs, instructions, users)
    scalar_programs = (up_map, hidden_map, residual_map)
    dot_instructions = tuple(
        name for name in sorted(internal, key=source_order.__getitem__) if instructions[name].opcode == "dot"
    )
    expected_dots = {
        semantics.up_input_adjoint.instruction,
        semantics.down_input_adjoint.instruction,
        semantics.down_weight_adjoint.instruction,
        semantics.up_weight_adjoint.instruction,
    }
    if set(dot_instructions) != expected_dots:
        raise ValueError("reverse boundary does not contain exactly its four recovered Contracts")
    family_digest = _family_digest("reverse", inputs, outputs, dot_instructions, instructions)
    cotangent_inputs = tuple(
        EntryRegionValue(name, instructions[name].shape)
        for name in up_sources
        if name != semantics.primal.input.instruction and name not in saved
    )
    if not cotangent_inputs:
        raise ValueError("reverse boundary has no external cotangent input")
    return LowRankContractMapReverseHloReplacementPlan(
        call_name=f"shuttle.generated.low_rank_contract_map.reverse.{index}",
        semantics=semantics,
        inputs=inputs,
        outputs=outputs,
        internal_instructions=tuple(sorted(internal, key=source_order.__getitem__)),
        insertion_instruction=insertion,
        dot_instructions=dot_instructions,
        dot_flops=sum(
            contract.flops
            for contract in (
                semantics.up_input_adjoint,
                semantics.down_input_adjoint,
                semantics.down_weight_adjoint,
                semantics.up_weight_adjoint,
            )
        ),
        scalar_programs=scalar_programs,
        cotangent_inputs=cotangent_inputs,
        upstream_collectives=semantics.upstream_collectives,
        boundary_family_digest=family_digest,
        semantic_digest=_semantic_digest(family_digest, scalar_programs),
        api_version=1,
    )


def _select_saved_scalar_program(
    graph: InlinedHloGraph,
    *,
    candidates: tuple[str, ...],
    target: str,
    saved: set[str],
    source_order: dict[str, int],
) -> tuple[tuple[str, ...], CastScalarProgram]:
    ordered = tuple(dict.fromkeys(sorted(candidates, key=source_order.__getitem__)))
    matches: list[tuple[tuple[str, ...], CastScalarProgram]] = []
    for count in range(1, len(ordered) + 1):
        for sources in itertools.combinations(ordered, count):
            try:
                program = import_hlo_scalar_map(
                    graph,
                    source_nodes=tuple(graph.entry_value(source) for source in sources),
                    target_node=graph.entry_value(target),
                )
            except (KeyError, ValueError):
                continue
            if len(program.inputs) == len(sources):
                matches.append((sources, program))
    if not matches:
        raise ValueError(f"no scalar Map boundary reaches %{target}")
    matches.sort(
        key=lambda match: (
            -sum(source in saved for source in match[0]),
            len(match[0]),
            tuple(source_order[source] for source in match[0]),
        )
    )
    return matches[0]


def _replace_boundary(
    hlo_text: str,
    plan: LowRankContractMapForwardHloReplacementPlan | LowRankContractMapReverseHloReplacementPlan,
    *,
    target: str,
) -> str:
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    existing = {instruction.name for instruction in entry.instructions}
    if plan.call_name in existing:
        raise ValueError(f"generated call name %{plan.call_name} already exists")
    rewritten = hlo_text
    insertion_line = None
    for instruction in plan.internal_instructions:
        pattern = re.compile(rf"^\s*(?:ROOT\s+)?%?{re.escape(instruction)} = .*?\n", re.MULTILINE)
        matches = tuple(pattern.finditer(rewritten))
        if len(matches) != 1:
            raise ValueError(f"expected one physical definition for %{instruction}")
        match = matches[0]
        if instruction == plan.insertion_instruction:
            insertion_line = (match.start(), match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip())])
        rewritten = rewritten[: match.start()] + rewritten[match.end() :]
    if insertion_line is None:
        raise ValueError("replacement insertion instruction was not removed")
    insertion_offset, indent = insertion_line
    operands = ", ".join(f"%{value.instruction}" for value in plan.inputs)
    constraints = ", ".join(value.shape for value in plan.inputs)
    output_shapes = ", ".join(value.shape for value in plan.outputs)
    lines = [
        f"{indent}%{plan.call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}",
        *(
            f"{indent}%{output.instruction} = {output.shape} get-tuple-element(%{plan.call_name}), index={index}"
            for index, output in enumerate(plan.outputs)
        ),
    ]
    transformed = rewritten[:insertion_offset] + "\n".join(lines) + "\n" + rewritten[insertion_offset:]
    parse_hlo_module_text(transformed)
    return transformed


def _audit_call(
    transformed_entry: HloComputation,
    plan: LowRankContractMapForwardHloReplacementPlan | LowRankContractMapReverseHloReplacementPlan,
    *,
    target: str,
    expected_users: dict[str, set[str]],
) -> LowRankContractMapCallAudit:
    transformed_users = _users(transformed_entry)
    transformed = {instruction.name: instruction for instruction in transformed_entry.instructions}
    call = transformed.get(plan.call_name)
    if call is None or call.opcode != "custom-call" or f'custom_call_target="{target}"' not in call.attributes:
        raise ValueError(f"missing generated low-rank Contract/Map call %{plan.call_name}")
    expected_inputs = tuple(value.instruction for value in plan.inputs)
    if call.operands != expected_inputs:
        raise ValueError(f"generated call %{plan.call_name} changed its exact input ABI")
    if "api_version=API_VERSION_TYPED_FFI" not in call.attributes:
        raise ValueError(f"generated call %{plan.call_name} is not typed FFI API version 1")
    output_names = tuple(value.instruction for value in plan.outputs)
    for index, output in enumerate(plan.outputs):
        generated = transformed.get(output.instruction)
        if generated is None or generated.opcode != "get-tuple-element" or generated.operands != (plan.call_name,):
            raise ValueError(f"generated output %{output.instruction} changed its tuple boundary")
        if generated.shape != output.shape or f"index={index}" not in generated.attributes:
            raise ValueError(f"generated output %{output.instruction} changed its ABI")
        if set(transformed_users[output.instruction]) != expected_users[output.instruction]:
            raise ValueError(f"generated output %{output.instruction} changed its external users")
    replaced_outputs = set(output_names)
    dead = tuple(name for name in plan.internal_instructions if name not in replaced_outputs)
    survivors = tuple(name for name in dead if name in transformed)
    if survivors:
        raise ValueError(f"old Contract/Map instructions remain after %{plan.call_name}: {survivors}")
    for dot in plan.dot_instructions:
        if dot in transformed and transformed[dot].opcode == "dot":
            raise ValueError(f"old Contract %{dot} remains after %{plan.call_name}")
    return LowRankContractMapCallAudit(
        call_instruction=call.name,
        target=target,
        inputs=expected_inputs,
        outputs=output_names,
        output_users=tuple((name, transformed_users[name]) for name in output_names),
        dead_internal_instructions=dead,
        replaced_output_instructions=output_names,
        removed_dot_instructions=plan.dot_instructions,
        removed_dot_flops=plan.dot_flops,
        api_version=plan.api_version,
    )


def _verify_boundary(
    internal: set[str],
    inputs: tuple[EntryRegionValue, ...],
    outputs: tuple[EntryRegionValue, ...],
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
) -> None:
    input_names = {value.instruction for value in inputs}
    output_names = {value.instruction for value in outputs}
    external_outputs = {name for name in internal if any(user not in internal for user in users.get(name, ()))}
    if external_outputs != output_names:
        raise ValueError(
            "Contract/Map boundary does not expose every live result: "
            f"expected {sorted(external_outputs)}, found {sorted(output_names)}"
        )
    unresolved = {
        operand
        for name in internal
        for operand in instructions[name].operands
        if operand not in internal and operand not in input_names and not _constant_derived(operand, instructions)
    }
    if unresolved:
        raise ValueError(f"Contract/Map boundary has unresolved dynamic inputs: {sorted(unresolved)}")


def _insertion_instruction(
    internal: set[str],
    inputs: tuple[EntryRegionValue, ...],
    outputs: tuple[EntryRegionValue, ...],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
) -> str:
    last_input = max(source_order[value.instruction] for value in inputs)
    candidates = tuple(name for name in internal if source_order[name] > last_input)
    if not candidates:
        raise ValueError("no Contract/Map instruction follows every dynamic input")
    insertion = min(candidates, key=source_order.__getitem__)
    external_users = {user for output in outputs for user in users[output.instruction] if user not in internal}
    if any(source_order[user] <= source_order[insertion] for user in external_users):
        raise ValueError("a Contract/Map output is consumed before every call input is ready")
    return insertion


def _boundary_inputs(
    internal: set[str],
    instructions: dict[str, HloInstruction],
    source_order: dict[str, int],
) -> tuple[EntryRegionValue, ...]:
    names = {
        operand
        for name in internal
        for operand in instructions[name].operands
        if operand not in internal and not _constant_derived(operand, instructions)
    }
    return tuple(
        EntryRegionValue(name, instructions[name].shape) for name in sorted(names, key=source_order.__getitem__)
    )


def _boundary_outputs(
    internal: set[str],
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
) -> tuple[EntryRegionValue, ...]:
    names = tuple(
        name
        for name in sorted(internal, key=source_order.__getitem__)
        if any(user not in internal for user in users.get(name, ()))
    )
    return tuple(EntryRegionValue(name, instructions[name].shape) for name in names)


def _path_interval(
    target: str,
    sources: set[str],
    instructions: dict[str, HloInstruction],
) -> set[str]:
    dependence: dict[str, bool] = {}

    def depends_on_source(name: str) -> bool:
        if name in dependence:
            return dependence[name]
        if name in sources:
            dependence[name] = True
            return True
        result = any(depends_on_source(operand) for operand in instructions[name].operands)
        dependence[name] = result
        return result

    interval: set[str] = set()

    def visit(name: str) -> None:
        if name in sources or name in interval or not depends_on_source(name):
            return
        interval.add(name)
        for operand in instructions[name].operands:
            visit(operand)

    visit(target)
    return interval


def _strip_layout_wrappers(name: str, instructions: dict[str, HloInstruction]) -> str:
    current = instructions[name]
    while current.opcode in _LAYOUT_WRAPPERS and len(current.operands) == 1:
        current = instructions[current.operands[0]]
    return current.name


def _constant_derived(
    name: str,
    instructions: dict[str, HloInstruction],
    memo: dict[str, bool] | None = None,
) -> bool:
    if memo is None:
        memo = {}
    if name in memo:
        return memo[name]
    instruction = instructions[name]
    if instruction.opcode == "constant":
        memo[name] = True
        return True
    result = (
        bool(instruction.operands)
        and instruction.opcode in _CONSTANT_DATAFLOW
        and all(_constant_derived(operand, instructions, memo) for operand in instruction.operands)
    )
    memo[name] = result
    return result


def _family_digest(
    kind: str,
    inputs: tuple[EntryRegionValue, ...],
    outputs: tuple[EntryRegionValue, ...],
    dots: tuple[str, ...],
    instructions: dict[str, HloInstruction],
) -> str:
    record = {
        "kind": kind,
        "input_shapes": [value.shape for value in inputs],
        "output_shapes": [value.shape for value in outputs],
        "dot_shapes": [instructions[name].shape for name in dots],
        "dot_dimensions": [
            {
                match.group("name"): tuple(int(value) for value in match.group("dims").split(",") if value)
                for match in _DOT_DIMENSION.finditer(instructions[name].attributes)
            }
            for name in dots
        ],
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _semantic_digest(family_digest: str, scalar_programs: tuple[CastScalarProgram, ...]) -> str:
    record = {"family": family_digest, "scalar_programs": [program.digest for program in scalar_programs]}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _depends_on(name: str, source: str, instructions: dict[str, HloInstruction]) -> bool:
    visited: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current == source:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(instructions[current].operands)
    return False


def _entry(hlo_text: str) -> HloComputation:
    module = parse_hlo_module_text(hlo_text)
    return module.computation(module.entry)


def _users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            values.setdefault(operand, []).append(instruction.name)
    return {name: tuple(users) for name, users in values.items()}


def _live_instructions(entry: HloComputation) -> set[str]:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    live: set[str] = set()
    pending = [entry.root.name]
    while pending:
        name = pending.pop()
        if name in live:
            continue
        live.add(name)
        pending.extend(instructions[name].operands)
    return live
