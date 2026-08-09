# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a compact Contract/normalized-exp/indexed-selection forward."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from tile_lifetime.xla_hlo_recovery import EntryRegionValue, HloComputation, HloInstruction, parse_hlo_module_text
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    NormalizedExpReverseContract,
    recover_normalized_exp_contract_reverse_hlo_regions,
)
from tile_lifetime.xla_relation_program_recovery import ContractDimensionMap

_DOT_DIMENSIONS = re.compile(
    r"(?P<name>lhs_contracting_dims|rhs_contracting_dims|lhs_batch_dims|rhs_batch_dims)=" r"\{(?P<dims>[0-9,]*)\}"
)
_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")


@dataclass(frozen=True)
class NormalizedExpContractForwardHloRegion:
    """Compact forward semantics proven from a padded physical HLO family."""

    physical_score_contract: NormalizedExpReverseContract
    compact_score_contract: NormalizedExpReverseContract
    fold_validity: EntryRegionValue
    selected_indices: EntryRegionValue
    output: EntryRegionValue
    saved_state: EntryRegionValue
    internal_instructions: tuple[str, ...]
    external_users: tuple[tuple[str, tuple[str, ...]], ...]
    semantic_digest: str


@dataclass(frozen=True)
class NormalizedExpContractForwardHloReplacementPlan:
    """A two-output typed-FFI boundary for one compact forward family."""

    region: NormalizedExpContractForwardHloRegion
    inputs: tuple[EntryRegionValue, ...]
    outputs: tuple[EntryRegionValue, ...]
    insertion_instruction: str
    external_users: tuple[tuple[str, tuple[str, ...]], ...]
    api_version: int


@dataclass(frozen=True)
class NormalizedExpContractForwardHloReplacementAudit:
    """Post-replacement liveness evidence for the compact forward call."""

    call_instruction: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    rewired_external_users: tuple[tuple[str, tuple[str, ...]], ...]
    dead_instructions: tuple[str, ...]
    retained_boundary_instructions: tuple[str, ...]
    output_users: tuple[tuple[str, tuple[str, ...]], ...]
    api_version: int


def recover_normalized_exp_contract_forward_hlo_region(hlo_text: str) -> NormalizedExpContractForwardHloRegion:
    """Recover one compact forward using only algebraic/dataflow structure."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry)
    reverse_report = recover_normalized_exp_contract_reverse_hlo_regions(hlo_text)
    if len(reverse_report.regions) != 1:
        raise ValueError(f"expected one normalized-exp reverse witness, found {len(reverse_report.regions)}")
    reverse = reverse_report.regions[0]

    saved_state = instructions[reverse.saved_state.instruction]
    state_slice = _operand_with_opcode(saved_state, instructions, "slice")
    state_select = _operand_with_opcode(state_slice, instructions, "select")
    if len(state_select.operands) != 3:
        raise ValueError("saved-state row validity is not a ternary select")
    log_normalizer = instructions[state_select.operands[1]]
    if log_normalizer.opcode != "add" or len(log_normalizer.operands) != 2:
        raise ValueError("saved normalized-exp state is not log(sum)+maximum")
    maximum_folds = tuple(
        instructions[name] for name in log_normalizer.operands if instructions[name].opcode == "reduce"
    )
    if len(maximum_folds) != 1:
        raise ValueError(f"log-normalizer has {len(maximum_folds)} direct Fold states")
    maximum_fold = maximum_folds[0]
    restricted_scores = instructions[maximum_fold.operands[0]]
    if restricted_scores.opcode != "select" or len(restricted_scores.operands) != 3:
        raise ValueError("normalized-exp maximum does not consume a DomainRestriction")
    converted_scores = instructions[restricted_scores.operands[1]]
    while converted_scores.opcode in {"convert", "reshape"} and len(converted_scores.operands) == 1:
        converted_scores = instructions[converted_scores.operands[0]]
    physical_contract = _contract(instructions, converted_scores)

    forward_lhs_base, forward_lhs_chain = _strip_zero_padding(physical_contract.lhs.instruction, instructions)
    forward_rhs_base, forward_rhs_chain = _strip_zero_padding(physical_contract.rhs.instruction, instructions)
    reverse_lhs_base, _ = _strip_zero_padding(reverse.score_contract.lhs.instruction, instructions)
    reverse_rhs_base, _ = _strip_zero_padding(reverse.score_contract.rhs.instruction, instructions)
    if forward_lhs_base != reverse_lhs_base or forward_rhs_base != reverse_rhs_base:
        raise ValueError("padded forward and compact reverse Contracts do not share primal operands")
    compact_contract = reverse.score_contract

    selected_indices = _raw_selected_indices(reverse.selected_indices.instruction, instructions)
    output = _loss_output(log_normalizer, instructions, users)
    internal = set(_ancestor_interval(output.instruction, instructions, stop={forward_lhs_base, forward_rhs_base}))
    internal.update(_ancestor_interval(saved_state.name, instructions, stop={forward_lhs_base, forward_rhs_base}))
    internal.update(forward_lhs_chain)
    internal.update(forward_rhs_chain)
    internal.add(physical_contract.instruction)
    outputs = (output, EntryRegionValue(saved_state.name, saved_state.shape))
    external_users = tuple(
        (value.instruction, tuple(user for user in users[value.instruction] if user not in internal))
        for value in outputs
    )
    if any(not values for _, values in external_users):
        raise ValueError("normalized-exp forward output has no external consumer")
    semantic_record = {
        "compact_score": _contract_semantics(compact_contract),
        "fold_validity_shape": reverse.fold_validity.shape,
        "selected_indices_shape": selected_indices.shape,
        "output_shape": output.shape,
        "saved_state_shape": saved_state.shape,
        "maps": ("fold_domain_restriction", "exp", "log", "indexed_selection", "subtract"),
        "folds": ("maximum", "sum"),
    }
    return NormalizedExpContractForwardHloRegion(
        physical_score_contract=physical_contract,
        compact_score_contract=compact_contract,
        fold_validity=reverse.fold_validity,
        selected_indices=selected_indices,
        output=output,
        saved_state=EntryRegionValue(saved_state.name, saved_state.shape),
        internal_instructions=tuple(
            instruction.name for instruction in entry.instructions if instruction.name in internal
        ),
        external_users=external_users,
        semantic_digest=hashlib.sha256(json.dumps(semantic_record, sort_keys=True).encode()).hexdigest(),
    )


def plan_normalized_exp_contract_forward_hlo_replacement(
    hlo_text: str,
) -> NormalizedExpContractForwardHloReplacementPlan:
    """Plan a compact two-output forward call before padded computation begins."""
    region = recover_normalized_exp_contract_forward_hlo_region(hlo_text)
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    inputs = (
        region.compact_score_contract.lhs,
        region.compact_score_contract.rhs,
        region.fold_validity,
        region.selected_indices,
    )
    insertion = region.selected_indices.instruction
    if any(source_order[value.instruction] > source_order[insertion] for value in inputs):
        raise ValueError("compact normalized-exp forward inputs are not ready at the selected-index boundary")
    return NormalizedExpContractForwardHloReplacementPlan(
        region=region,
        inputs=inputs,
        outputs=(region.output, region.saved_state),
        insertion_instruction=insertion,
        external_users=region.external_users,
        api_version=1,
    )


def replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
    hlo_text: str,
    plan: NormalizedExpContractForwardHloReplacementPlan,
    *,
    target: str,
) -> str:
    """Insert a compact forward call and redirect output/state users."""
    call_name = "shuttle.generated.normalized_exp_contract_forward"
    output_names = tuple(f"{call_name}.output.{index}" for index in range(len(plan.outputs)))
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    collision = {instruction.name for instruction in entry.instructions} & {call_name, *output_names}
    if collision:
        raise ValueError(f"normalized-exp forward replacement names already exist: {sorted(collision)}")
    pattern = re.compile(
        rf"^(?P<indent>\s*)(?:ROOT\s+)?%?{re.escape(plan.insertion_instruction)} = .*?$",
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one insertion definition for {plan.insertion_instruction!r}")
    match = matches[0]
    indent = match.group("indent")
    output_shapes = ", ".join(value.shape for value in plan.outputs)
    operands = ", ".join(f"%{value.instruction}" for value in plan.inputs)
    constraints = ", ".join(value.shape for value in plan.inputs)
    lines = [
        f"{indent}%{call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}",
        *(
            f"{indent}%{name} = {output.shape} get-tuple-element(%{call_name}), index={index}"
            for index, (name, output) in enumerate(zip(output_names, plan.outputs, strict=True))
        ),
    ]
    rewritten = hlo_text[: match.end()] + "\n" + "\n".join(lines) + hlo_text[match.end() :]
    for (old_name, external_users), new_name in zip(plan.external_users, output_names, strict=True):
        for user in external_users:
            rewritten = _replace_entry_operand(rewritten, user=user, old=old_name, new=new_name)
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_normalized_exp_contract_forward_hlo_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: NormalizedExpContractForwardHloReplacementPlan,
    *,
    target: str,
    expected_output_users: tuple[tuple[str, ...], ...] | None = None,
) -> NormalizedExpContractForwardHloReplacementAudit:
    """Verify one exact generated boundary and removal of old forward work."""
    original_module = parse_hlo_module_text(original_hlo)
    original_entry = original_module.computation(original_module.entry)
    module = parse_hlo_module_text(transformed_hlo)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    calls = tuple(
        instruction
        for instruction in entry.instructions
        if instruction.opcode == "custom-call" and f'custom_call_target="{target}"' in instruction.attributes
    )
    if len(calls) != 1:
        raise ValueError(f"expected one generated forward call for {target!r}, found {len(calls)}")
    call = calls[0]
    expected_inputs = tuple(value.instruction for value in plan.inputs)
    if call.operands != expected_inputs:
        raise ValueError("normalized-exponential forward call changed its input boundary")
    if "api_version=API_VERSION_TYPED_FFI" not in call.attributes:
        raise ValueError("normalized-exponential forward call is not typed FFI API version 1")
    call_outputs = tuple(f"{call.name}.output.{index}" for index in range(len(plan.outputs)))
    for index, (name, expected) in enumerate(zip(call_outputs, plan.outputs, strict=True)):
        output = instructions.get(name)
        if output is None or output.opcode != "get-tuple-element" or output.operands != (call.name,):
            raise ValueError(f"normalized-exponential forward output {index} changed its tuple boundary")
        if output.shape != expected.shape or f"index={index}" not in output.attributes:
            raise ValueError(f"normalized-exponential forward output {index} changed its result ABI")
    for output, users in plan.external_users:
        for user in users:
            if output in instructions[user].operands:
                raise ValueError(f"external user %{user} still consumes old forward output %{output}")
    live = _live_entry_instructions(entry)
    transformed_users = _entry_users(entry)
    actual_output_users = tuple(
        (name, tuple(user for user in transformed_users[name] if user in live)) for name in call_outputs
    )
    required_output_users = expected_output_users or tuple(users for _, users in plan.external_users)
    if len(required_output_users) != len(call_outputs):
        raise ValueError("normalized-exponential forward audit has the wrong output-user arity")
    for (name, actual), expected in zip(actual_output_users, required_output_users, strict=True):
        if actual != expected:
            raise ValueError(f"normalized-exponential forward output %{name} changed its consumers")
    internal = frozenset(plan.region.internal_instructions)
    replaced_outputs = frozenset(value.instruction for value in plan.outputs)
    original_users = _entry_users(original_entry)
    shared_roots = tuple(
        instruction
        for instruction in internal
        if instruction not in replaced_outputs and any(user not in internal for user in original_users[instruction])
    )
    retained_boundary = _ancestor_closure(original_entry, (*expected_inputs, *shared_roots))
    dead = tuple(
        instruction for instruction in plan.region.internal_instructions if instruction not in retained_boundary
    )
    still_live = tuple(instruction for instruction in dead if instruction in live)
    if still_live:
        raise ValueError(f"replaced normalized-exponential forward arithmetic remains live: {still_live}")
    return NormalizedExpContractForwardHloReplacementAudit(
        call_instruction=call.name,
        inputs=expected_inputs,
        outputs=call_outputs,
        rewired_external_users=plan.external_users,
        dead_instructions=dead,
        retained_boundary_instructions=tuple(
            instruction for instruction in plan.region.internal_instructions if instruction in retained_boundary
        ),
        output_users=actual_output_users,
        api_version=plan.api_version,
    )


def _contract(instructions: dict[str, HloInstruction], instruction: HloInstruction) -> NormalizedExpReverseContract:
    if instruction.opcode != "dot" or len(instruction.operands) != 2:
        raise ValueError("normalized-exp score producer is not a binary Contract")
    parsed = {
        match.group("name"): tuple(int(value) for value in match.group("dims").split(",") if value)
        for match in _DOT_DIMENSIONS.finditer(instruction.attributes)
    }
    lhs = instructions[instruction.operands[0]]
    rhs = instructions[instruction.operands[1]]
    lhs_shape = _shape(lhs.shape)
    rhs_shape = _shape(rhs.shape)
    lhs_contracting = parsed.get("lhs_contracting_dims", ())
    rhs_contracting = parsed.get("rhs_contracting_dims", ())
    lhs_batch = parsed.get("lhs_batch_dims", ())
    rhs_batch = parsed.get("rhs_batch_dims", ())
    return NormalizedExpReverseContract(
        instruction.name,
        EntryRegionValue(lhs.name, lhs.shape),
        EntryRegionValue(rhs.name, rhs.shape),
        instruction.shape,
        ContractDimensionMap(
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            tuple(axis for axis in range(len(lhs_shape)) if axis not in lhs_contracting + lhs_batch),
            tuple(axis for axis in range(len(rhs_shape)) if axis not in rhs_contracting + rhs_batch),
        ),
    )


def _operand_with_opcode(
    instruction: HloInstruction, instructions: dict[str, HloInstruction], opcode: str
) -> HloInstruction:
    if len(instruction.operands) != 1:
        raise ValueError(f"%{instruction.name} is not unary")
    operand = instructions[instruction.operands[0]]
    if operand.opcode != opcode:
        raise ValueError(f"%{instruction.name} does not consume a {opcode}")
    return operand


def _strip_zero_padding(name: str, instructions: dict[str, HloInstruction]) -> tuple[str, tuple[str, ...]]:
    chain: list[str] = []
    current = instructions[name]
    while current.opcode == "pad" and len(current.operands) == 2:
        padding_value = instructions[current.operands[1]]
        if padding_value.opcode != "constant" or not re.search(r"constant\((?:0|0\.0+)\)", padding_value.attributes):
            raise ValueError(f"%{current.name} is not zero padding")
        chain.append(current.name)
        current = instructions[current.operands[0]]
    return current.name, tuple(chain)


def _raw_selected_indices(name: str, instructions: dict[str, HloInstruction]) -> EntryRegionValue:
    current = instructions[name]
    if current.opcode != "clamp" or len(current.operands) != 3:
        raise ValueError("selected indices are not a bounded clamp")
    raw = instructions[current.operands[1]]
    return EntryRegionValue(raw.name, raw.shape)


def _loss_output(
    log_normalizer: HloInstruction,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
) -> EntryRegionValue:
    differences = tuple(
        instructions[user]
        for user in users[log_normalizer.name]
        if instructions[user].opcode == "subtract" and instructions[user].operands[0] == log_normalizer.name
    )
    if len(differences) != 1:
        raise ValueError(f"log-normalizer has {len(differences)} indexed-selection differences")
    current = differences[0]
    expected = ("select", "slice", "reshape")
    for opcode in expected:
        next_values = tuple(instructions[user] for user in users[current.name] if instructions[user].opcode == opcode)
        if len(next_values) != 1:
            raise ValueError(f"loss %{current.name} has {len(next_values)} {opcode} users")
        current = next_values[0]
    return EntryRegionValue(current.name, current.shape)


def _ancestor_interval(name: str, instructions: dict[str, HloInstruction], *, stop: set[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current in seen or current in stop:
            continue
        seen.add(current)
        pending.extend(instructions[current].operands)
    return tuple(seen)


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            mutable.setdefault(operand, []).append(instruction.name)
    return {name: tuple(values) for name, values in mutable.items()}


def _ancestor_closure(entry: HloComputation, roots: tuple[str, ...]) -> frozenset[str]:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    ancestors: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in ancestors:
            continue
        ancestors.add(name)
        stack.extend(instructions[name].operands)
    return frozenset(ancestors)


def _live_entry_instructions(entry: HloComputation) -> frozenset[str]:
    return _ancestor_closure(entry, (entry.root.name,))


def _shape(shape: str) -> tuple[int, ...]:
    match = _ARRAY_SHAPE.match(shape)
    if match is None:
        raise ValueError(f"unsupported array shape {shape!r}")
    return tuple(int(value) for value in match.group("dims").split(",") if value)


def _contract_semantics(contract: NormalizedExpReverseContract) -> dict[str, object]:
    return {
        "lhs_shape": contract.lhs.shape,
        "rhs_shape": contract.rhs.shape,
        "output_shape": contract.output_shape,
        "lhs_contracting": contract.dimensions.lhs_contracting,
        "rhs_contracting": contract.dimensions.rhs_contracting,
    }


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
    replaced_body, count = re.subn(rf"%{re.escape(old)}(?![A-Za-z0-9_.-])", f"%{new}", body)
    if count == 0:
        raise ValueError(f"external user %{user} does not consume %{old}")
    return hlo_text[: match.start("body")] + replaced_body + hlo_text[match.end("body") :]
