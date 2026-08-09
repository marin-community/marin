# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover normalized-exponential Contract reverse structure from XLA HLO."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from tile_lifetime.xla_hlo_recovery import EntryRegionValue, HloComputation, HloInstruction, parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import ContractDimensionMap

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_DOT_DIMENSIONS = re.compile(
    r"(?P<name>lhs_contracting_dims|rhs_contracting_dims|lhs_batch_dims|rhs_batch_dims)=" r"\{(?P<dims>[0-9,]*)\}"
)


@dataclass(frozen=True)
class NormalizedExpReverseContract:
    """One Contract participating in a normalized-exponential reverse region."""

    instruction: str
    lhs: EntryRegionValue
    rhs: EntryRegionValue
    output_shape: str
    dimensions: ContractDimensionMap


@dataclass(frozen=True)
class NormalizedExpContractReverseHloRegion:
    """A generic Contract/Map/Fold reverse region recovered from physical HLO."""

    score_contract: NormalizedExpReverseContract
    input_reverse_contract: NormalizedExpReverseContract
    operand_reverse_contract: NormalizedExpReverseContract
    saved_state: EntryRegionValue
    fold_validity: EntryRegionValue
    row_cotangent: EntryRegionValue
    selected_mask: EntryRegionValue
    selected_indices: EntryRegionValue
    row_validity: EntryRegionValue
    score_cotangent: EntryRegionValue
    internal_instructions: tuple[str, ...]
    external_users: tuple[tuple[str, tuple[str, ...]], ...]
    semantic_digest: str


@dataclass(frozen=True)
class NormalizedExpContractReverseRecoveryReport:
    """All structurally recovered normalized-exponential reverse regions."""

    regions: tuple[NormalizedExpContractReverseHloRegion, ...]
    rejected: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class NormalizedExpContractReverseHloReplacementPlan:
    """One local typed-FFI boundary for a recovered reverse region."""

    region: NormalizedExpContractReverseHloRegion
    inputs: tuple[EntryRegionValue, ...]
    outputs: tuple[EntryRegionValue, ...]
    insertion_instruction: str
    external_users: tuple[tuple[str, tuple[str, ...]], ...]
    api_version: int


@dataclass(frozen=True)
class NormalizedExpContractReverseHloReplacementAudit:
    """Post-replacement liveness evidence for a local reverse call."""

    call_instruction: str
    rewired_external_users: tuple[tuple[str, tuple[str, ...]], ...]


def recover_normalized_exp_contract_reverse_hlo_regions(
    hlo_text: str,
) -> NormalizedExpContractReverseRecoveryReport:
    """Recover generic reverse regions without consulting names or metadata."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry)
    regions: list[NormalizedExpContractReverseHloRegion] = []
    rejected: list[tuple[str, str]] = []
    for exponential in entry.instructions:
        if exponential.opcode != "exponential":
            continue
        try:
            regions.append(_recover_from_exponential(exponential, instructions, users))
        except ValueError as error:
            rejected.append((exponential.name, str(error)))
    return NormalizedExpContractReverseRecoveryReport(tuple(regions), tuple(rejected))


def plan_normalized_exp_contract_reverse_hlo_replacement(
    hlo_text: str,
) -> NormalizedExpContractReverseHloReplacementPlan:
    """Form one generated reverse boundary from a unique recovered region."""
    report = recover_normalized_exp_contract_reverse_hlo_regions(hlo_text)
    if len(report.regions) != 1:
        detail = "; ".join(f"%{name}: {reason}" for name, reason in report.rejected)
        raise ValueError(
            f"expected one normalized-exponential Contract reverse region, found {len(report.regions)}"
            + (f" ({detail})" if detail else "")
        )
    region = report.regions[0]
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    users = _entry_users(entry)
    inputs = (
        region.score_contract.lhs,
        region.score_contract.rhs,
        region.saved_state,
        region.fold_validity,
        region.row_cotangent,
        region.selected_indices,
        region.row_validity,
    )
    insertion = region.score_cotangent.instruction
    if any(source_order[value.instruction] >= source_order[insertion] for value in inputs):
        raise ValueError("normalized-exponential reverse inputs are not ready at the score-cotangent boundary")
    outputs = (
        EntryRegionValue(region.input_reverse_contract.instruction, region.input_reverse_contract.output_shape),
        EntryRegionValue(region.operand_reverse_contract.instruction, region.operand_reverse_contract.output_shape),
    )
    internal = set(region.internal_instructions)
    external_users = tuple(
        (output.instruction, tuple(user for user in users[output.instruction] if user not in internal))
        for output in outputs
    )
    if any(not values for _, values in external_users):
        raise ValueError("normalized-exponential reverse output has no external consumer")
    return NormalizedExpContractReverseHloReplacementPlan(
        region=region,
        inputs=inputs,
        outputs=outputs,
        insertion_instruction=insertion,
        external_users=external_users,
        api_version=1,
    )


def replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
    hlo_text: str,
    plan: NormalizedExpContractReverseHloReplacementPlan,
    *,
    target: str,
) -> str:
    """Insert a two-output reverse call and redirect the proven users."""
    call_name = "shuttle.generated.normalized_exp_contract_reverse"
    output_names = tuple(f"{call_name}.output.{index}" for index in range(len(plan.outputs)))
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    existing = {instruction.name for instruction in entry.instructions}
    collision = existing & {call_name, *output_names}
    if collision:
        raise ValueError(f"normalized-exponential replacement names already exist: {sorted(collision)}")
    insertion_pattern = re.compile(
        rf"^(?P<indent>\s*)(?:ROOT\s+)?%?{re.escape(plan.insertion_instruction)} = .*?$",
        re.MULTILINE,
    )
    matches = tuple(insertion_pattern.finditer(hlo_text))
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


def audit_normalized_exp_contract_reverse_hlo_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: NormalizedExpContractReverseHloReplacementPlan,
    *,
    target: str,
) -> NormalizedExpContractReverseHloReplacementAudit:
    """Verify the generated call and removal of old external reverse edges."""
    original_entry = parse_hlo_module_text(original_hlo).computation(parse_hlo_module_text(original_hlo).entry)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    transformed_instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    calls = tuple(
        instruction
        for instruction in transformed_entry.instructions
        if instruction.opcode == "custom-call" and f'custom_call_target="{target}"' in instruction.attributes
    )
    if len(calls) != 1:
        raise ValueError(f"expected one generated reverse call for {target!r}, found {len(calls)}")
    original_names = {instruction.name for instruction in original_entry.instructions}
    for output, users in plan.external_users:
        if output not in original_names:
            raise ValueError(f"unknown original reverse output %{output}")
        for user in users:
            if output in transformed_instructions[user].operands:
                raise ValueError(f"external user %{user} still consumes old reverse output %{output}")
    return NormalizedExpContractReverseHloReplacementAudit(calls[0].name, plan.external_users)


def _recover_from_exponential(
    exponential: HloInstruction,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
) -> NormalizedExpContractReverseHloRegion:
    centered = _only_operand_with_opcode(exponential, instructions, "subtract")
    if len(centered.operands) != 2:
        raise ValueError("normalized-exponential centering is not binary")
    raw_score = instructions[centered.operands[0]]
    state_broadcast = instructions[centered.operands[1]]
    if raw_score.opcode != "convert" or len(raw_score.operands) != 1:
        raise ValueError("centered score is not an explicit converted Contract result")
    score_contract_instruction = instructions[raw_score.operands[0]]
    score_contract = _contract(entry_instructions=instructions, instruction=score_contract_instruction)
    if state_broadcast.opcode != "broadcast" or len(state_broadcast.operands) != 1:
        raise ValueError("normalized-exponential state is not a row broadcast")

    restricted = _unique_user(
        exponential.name,
        users,
        instructions,
        lambda value: value.opcode == "select" and len(value.operands) == 3 and value.operands[1] == exponential.name,
        "validity restriction",
    )
    validity = instructions[restricted.operands[0]]
    false_value = instructions[restricted.operands[2]]
    if false_value.opcode not in {"broadcast", "constant"}:
        raise ValueError("validity restriction does not use a scalar fill")
    if validity.opcode != "broadcast" or len(validity.operands) != 1:
        raise ValueError("Fold validity is not broadcast from its physical fold domain")
    fold_validity = instructions[validity.operands[0]]

    probability_product = _unique_user(
        restricted.name,
        users,
        instructions,
        lambda value: value.opcode == "multiply" and restricted.name in value.operands,
        "probability cotangent product",
    )
    row_cotangent_broadcast_name = next(
        operand for operand in probability_product.operands if operand != restricted.name
    )
    row_cotangent_broadcast = instructions[row_cotangent_broadcast_name]
    if row_cotangent_broadcast.opcode != "broadcast" or len(row_cotangent_broadcast.operands) != 1:
        raise ValueError("probability cotangent is not row-broadcast")

    score_difference = _unique_user(
        probability_product.name,
        users,
        instructions,
        lambda value: value.opcode == "subtract" and value.operands[0] == probability_product.name,
        "selected-coordinate correction",
    )
    selected_product = instructions[score_difference.operands[1]]
    if selected_product.opcode != "multiply" or row_cotangent_broadcast.name not in selected_product.operands:
        raise ValueError("selected-coordinate correction does not share the row cotangent")
    selected_convert_name = next(
        operand for operand in selected_product.operands if operand != row_cotangent_broadcast.name
    )
    selected_convert = instructions[selected_convert_name]
    if selected_convert.opcode != "convert" or len(selected_convert.operands) != 1:
        raise ValueError("selected-coordinate mask has no explicit precision boundary")
    selected_indices, row_validity, selection_internal = _recover_indexed_selection(
        instructions[selected_convert.operands[0]], instructions
    )

    score_cotangent = _unique_user(
        score_difference.name,
        users,
        instructions,
        lambda value: value.opcode == "convert" and len(value.operands) == 1,
        "score cotangent conversion",
    )
    reverse_contracts = tuple(
        _contract(entry_instructions=instructions, instruction=instructions[user])
        for user in users.get(score_cotangent.name, ())
        if instructions[user].opcode == "dot"
    )
    if len(reverse_contracts) != 2:
        raise ValueError(f"score cotangent has {len(reverse_contracts)} reverse Contracts")
    input_reverse, operand_reverse = _classify_reverse_contracts(score_contract, score_cotangent.name, reverse_contracts)

    internal = {
        score_contract.instruction,
        raw_score.name,
        centered.name,
        exponential.name,
        restricted.name,
        probability_product.name,
        selected_convert.name,
        selected_product.name,
        score_difference.name,
        score_cotangent.name,
        input_reverse.instruction,
        operand_reverse.instruction,
        *selection_internal,
    }
    external_users = tuple(
        (name, tuple(user for user in users.get(name, ()) if user not in internal))
        for name in sorted(internal)
        if any(user not in internal for user in users.get(name, ()))
    )
    semantic_record = {
        "score": _contract_semantics(score_contract),
        "input_reverse": _contract_semantics(input_reverse),
        "operand_reverse": _contract_semantics(operand_reverse),
        "saved_state_shape": instructions[state_broadcast.operands[0]].shape,
        "fold_validity_shape": fold_validity.shape,
        "row_cotangent_shape": instructions[row_cotangent_broadcast.operands[0]].shape,
        "selected_mask_shape": instructions[selected_convert.operands[0]].shape,
        "selected_indices_shape": selected_indices.shape,
        "row_validity_shape": row_validity.shape,
        "score_cotangent_shape": score_cotangent.shape,
        "map_opcodes": (
            "subtract",
            "exponential",
            "select",
            "multiply",
            "multiply",
            "subtract",
            "convert",
        ),
    }
    semantic_digest = hashlib.sha256(json.dumps(semantic_record, sort_keys=True).encode()).hexdigest()
    return NormalizedExpContractReverseHloRegion(
        score_contract=score_contract,
        input_reverse_contract=input_reverse,
        operand_reverse_contract=operand_reverse,
        saved_state=EntryRegionValue(state_broadcast.operands[0], instructions[state_broadcast.operands[0]].shape),
        fold_validity=EntryRegionValue(fold_validity.name, fold_validity.shape),
        row_cotangent=EntryRegionValue(
            row_cotangent_broadcast.operands[0], instructions[row_cotangent_broadcast.operands[0]].shape
        ),
        selected_mask=EntryRegionValue(selected_convert.operands[0], instructions[selected_convert.operands[0]].shape),
        selected_indices=selected_indices,
        row_validity=row_validity,
        score_cotangent=EntryRegionValue(score_cotangent.name, score_cotangent.shape),
        internal_instructions=tuple(
            instruction.name for instruction in instructions.values() if instruction.name in internal
        ),
        external_users=external_users,
        semantic_digest=semantic_digest,
    )


def _recover_indexed_selection(
    selected_mask: HloInstruction,
    instructions: dict[str, HloInstruction],
) -> tuple[EntryRegionValue, EntryRegionValue, tuple[str, ...]]:
    if selected_mask.opcode != "select" or len(selected_mask.operands) != 3:
        raise ValueError("selected-coordinate mask is not an indexed select")
    equality = instructions[selected_mask.operands[0]]
    if equality.opcode != "compare" or "direction=EQ" not in equality.attributes or len(equality.operands) != 2:
        raise ValueError("selected-coordinate predicate is not equality against an index")
    equality_operands = tuple(instructions[name] for name in equality.operands)
    iotas = tuple(value for value in equality_operands if value.opcode == "iota")
    index_broadcasts = tuple(value for value in equality_operands if value.opcode == "broadcast")
    if len(iotas) != 1 or len(index_broadcasts) != 1 or len(index_broadcasts[0].operands) != 1:
        raise ValueError("selected-coordinate equality lacks one iota and one index broadcast")
    selected_indices = instructions[index_broadcasts[0].operands[0]]

    true_value = instructions[selected_mask.operands[1]]
    if true_value.opcode != "broadcast" or len(true_value.operands) != 1:
        raise ValueError("selected-coordinate value does not broadcast row validity")
    validity_convert = instructions[true_value.operands[0]]
    if validity_convert.opcode != "convert" or len(validity_convert.operands) != 1:
        raise ValueError("selected-coordinate row validity has no explicit cast")
    row_validity = instructions[validity_convert.operands[0]]
    internal = (
        equality.name,
        iotas[0].name,
        index_broadcasts[0].name,
        true_value.name,
        validity_convert.name,
        selected_mask.name,
    )
    return (
        EntryRegionValue(selected_indices.name, selected_indices.shape),
        EntryRegionValue(row_validity.name, row_validity.shape),
        internal,
    )


def _contract(
    *,
    entry_instructions: dict[str, HloInstruction],
    instruction: HloInstruction,
) -> NormalizedExpReverseContract:
    if instruction.opcode != "dot" or len(instruction.operands) != 2:
        raise ValueError(f"%{instruction.name} is not a binary Contract")
    parsed = {
        match.group("name"): tuple(int(value) for value in match.group("dims").split(",") if value)
        for match in _DOT_DIMENSIONS.finditer(instruction.attributes)
    }
    lhs = entry_instructions[instruction.operands[0]]
    rhs = entry_instructions[instruction.operands[1]]
    lhs_shape = _array_shape(lhs.shape)
    rhs_shape = _array_shape(rhs.shape)
    if lhs_shape is None or rhs_shape is None:
        raise ValueError(f"Contract %{instruction.name} has a non-array operand")
    lhs_contracting = parsed.get("lhs_contracting_dims", ())
    rhs_contracting = parsed.get("rhs_contracting_dims", ())
    lhs_batch = parsed.get("lhs_batch_dims", ())
    rhs_batch = parsed.get("rhs_batch_dims", ())
    return NormalizedExpReverseContract(
        instruction=instruction.name,
        lhs=EntryRegionValue(lhs.name, lhs.shape),
        rhs=EntryRegionValue(rhs.name, rhs.shape),
        output_shape=instruction.shape,
        dimensions=ContractDimensionMap(
            lhs_contracting=lhs_contracting,
            rhs_contracting=rhs_contracting,
            lhs_batch=lhs_batch,
            rhs_batch=rhs_batch,
            lhs_output=tuple(axis for axis in range(len(lhs_shape[1])) if axis not in lhs_contracting + lhs_batch),
            rhs_output=tuple(axis for axis in range(len(rhs_shape[1])) if axis not in rhs_contracting + rhs_batch),
        ),
    )


def _classify_reverse_contracts(
    score_contract: NormalizedExpReverseContract,
    score_cotangent: str,
    reverse_contracts: tuple[NormalizedExpReverseContract, ...],
) -> tuple[NormalizedExpReverseContract, NormalizedExpReverseContract]:
    input_reverse = tuple(
        contract
        for contract in reverse_contracts
        if contract.lhs.instruction == score_cotangent and contract.rhs.instruction == score_contract.rhs.instruction
    )
    operand_reverse = tuple(
        contract
        for contract in reverse_contracts
        if contract.lhs.instruction == score_contract.lhs.instruction and contract.rhs.instruction == score_cotangent
    )
    if len(input_reverse) != 1 or len(operand_reverse) != 1:
        raise ValueError("could not classify input and operand reverse Contracts from dataflow")
    return input_reverse[0], operand_reverse[0]


def _only_operand_with_opcode(
    instruction: HloInstruction,
    instructions: dict[str, HloInstruction],
    opcode: str,
) -> HloInstruction:
    values = tuple(instructions[name] for name in instruction.operands if instructions[name].opcode == opcode)
    if len(values) != 1:
        raise ValueError(f"%{instruction.name} has {len(values)} {opcode} operands")
    return values[0]


def _unique_user(
    source: str,
    users: dict[str, tuple[str, ...]],
    instructions: dict[str, HloInstruction],
    predicate: Callable[[HloInstruction], bool],
    role: str,
) -> HloInstruction:
    values = tuple(instructions[name] for name in users.get(source, ()) if predicate(instructions[name]))
    if len(values) != 1:
        raise ValueError(f"%{source} has {len(values)} users matching {role}")
    return values[0]


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            mutable.setdefault(operand, []).append(instruction.name)
    return {name: tuple(values) for name, values in mutable.items()}


def _array_shape(shape: str) -> tuple[str, tuple[int, ...]] | None:
    match = _ARRAY_SHAPE.match(shape)
    if match is None:
        return None
    return match.group("dtype"), tuple(int(value) for value in match.group("dims").split(",") if value)


def _contract_semantics(contract: NormalizedExpReverseContract) -> dict[str, object]:
    return {
        "lhs_shape": contract.lhs.shape,
        "rhs_shape": contract.rhs.shape,
        "output_shape": contract.output_shape,
        "lhs_contracting": contract.dimensions.lhs_contracting,
        "rhs_contracting": contract.dimensions.rhs_contracting,
        "lhs_batch": contract.dimensions.lhs_batch,
        "rhs_batch": contract.dimensions.rhs_batch,
        "lhs_output": contract.dimensions.lhs_output,
        "rhs_output": contract.dimensions.rhs_output,
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
    operand_pattern = re.compile(rf"%{re.escape(old)}(?![A-Za-z0-9_.-])")
    replaced_body, count = operand_pattern.subn(f"%{new}", body)
    if count == 0:
        raise ValueError(f"external user %{user} does not consume %{old}")
    return hlo_text[: match.start("body")] + replaced_body + hlo_text[match.end("body") :]
