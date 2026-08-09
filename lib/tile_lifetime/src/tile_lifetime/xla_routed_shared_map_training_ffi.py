# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose routed training calls around one shared Contract/multi-Map region.

The shared region produces both a forward Map value and the corresponding
reverse Map value. Generic rank-two Contracts and a deterministic
source-indexed Fold own the remaining input-adjoint arithmetic. XLA retains
only physical view wrappers, relation/index work, and placement collectives.
"""

from __future__ import annotations

from dataclasses import dataclass

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import HloComputation, parse_hlo_module_text
from tile_lifetime.xla_rank_two_contract_ffi import (
    RankTwoBf16ContractTypedFfiPlan,
    RankTwoContractReplacementAudit,
    audit_rank_two_contract_replacement,
    plan_rank_two_bf16_contract_typed_ffi,
    replace_rank_two_contract_with_custom_call,
)
from tile_lifetime.xla_relation_program_recovery import (
    RoutedForwardCodegenDisposition,
    RoutedForwardTypedFfiCodegenPlan,
    RoutedInputAdjointTypedFfiCodegenPlan,
    RoutedWeightGradientTypedFfiCodegenPlan,
    SharedContractMultiMapRegionRecord,
    form_shared_contract_multi_map_region,
    plan_routed_forward_typed_ffi,
    plan_routed_input_adjoint_typed_ffi,
    plan_routed_weight_gradient_typed_ffi,
)
from tile_lifetime.xla_routed_forward_ffi import (
    replace_routed_forward_region_with_custom_call,
)
from tile_lifetime.xla_routed_weight_gradient_ffi import (
    replace_group_batched_contract_with_custom_call,
)
from tile_lifetime.xla_shared_contract_multimap import (
    SharedContractMultiMapReplacementAudit,
    audit_shared_contract_multi_map_replacement,
    replace_shared_contract_multi_map_region_with_custom_call,
)
from tile_lifetime.xla_source_indexed_fold_ffi import (
    SourceIndexedFoldReplacementAudit,
    SourceIndexedFoldTypedFfiPlan,
    audit_source_indexed_fold_replacement,
    plan_source_indexed_fold_typed_ffi,
    replace_source_indexed_fold_with_custom_call,
)


@dataclass(frozen=True)
class RoutedSharedMapTrainingFfiTargets:
    """Typed-FFI targets for one nonoverlapping routed composition."""

    forward: str
    input_contracts: tuple[str, str]
    shared_contract_multi_map: str
    source_fold: str
    weight_gradients: tuple[str, str]


@dataclass(frozen=True)
class RoutedSharedMapTrainingTypedFfiPlan:
    """Generated routed regions with only physical view wrappers left to XLA."""

    forward: RoutedForwardTypedFfiCodegenPlan
    input_contracts: tuple[RankTwoBf16ContractTypedFfiPlan, RankTwoBf16ContractTypedFfiPlan]
    shared_contract_multi_map: SharedContractMultiMapRegionRecord
    source_fold: SourceIndexedFoldTypedFfiPlan
    recovered_input_adjoint: RoutedInputAdjointTypedFfiCodegenPlan
    retained_input_adjoint_wrappers: tuple[str, ...]
    weight_gradients: tuple[RoutedWeightGradientTypedFfiCodegenPlan, RoutedWeightGradientTypedFfiCodegenPlan]


@dataclass(frozen=True)
class RoutedSharedMapTrainingReplacementAudit:
    """Post-roundtrip wiring evidence for the nonoverlapping composition."""

    target_instructions: tuple[str, str, str, str, str, str, str]
    input_contracts: tuple[RankTwoContractReplacementAudit, RankTwoContractReplacementAudit]
    shared_contract_multi_map: SharedContractMultiMapReplacementAudit
    source_fold: SourceIndexedFoldReplacementAudit
    source_fold_collective: str
    weight_gradient_collectives: tuple[str, str]
    retained_input_adjoint_wrappers: tuple[str, ...]
    copy_count: tuple[int, int]
    transpose_count: tuple[int, int]


def plan_routed_shared_map_training_typed_ffi(
    hlo_text: str,
    *,
    shared_contract_numerical_policy: NumericalPolicy = NumericalPolicy.ALLOW_ROUNDING_REORDER,
    weight_gradient_numerical_policy: NumericalPolicy = NumericalPolicy.ALLOW_ROUNDING_REORDER,
) -> RoutedSharedMapTrainingTypedFfiPlan:
    """Recover disjoint generated regions around a shared forward/reverse Map."""
    forward = plan_routed_forward_typed_ffi(hlo_text)
    if forward.disposition is not RoutedForwardCodegenDisposition.READY:
        raise ValueError("routed forward region lacks a verified physical layout")
    shared = form_shared_contract_multi_map_region(
        hlo_text,
        numerical_policy=shared_contract_numerical_policy,
    )
    recovered_input_adjoint = plan_routed_input_adjoint_typed_ffi(hlo_text)
    if len(recovered_input_adjoint.contracts) != 2:
        raise ValueError(f"expected two input-adjoint Contracts, found {len(recovered_input_adjoint.contracts)}")
    input_contracts = tuple(
        plan_rank_two_bf16_contract_typed_ffi(
            hlo_text,
            contract,
            numerical_policy=shared_contract_numerical_policy,
        )
        for contract in recovered_input_adjoint.contracts
    )
    source_fold = plan_source_indexed_fold_typed_ffi(
        hlo_text,
        recovered_input_adjoint,
        recovered_input_adjoint.contracts[1],
    )
    weight_gradients = plan_routed_weight_gradient_typed_ffi(
        hlo_text,
        numerical_policy=weight_gradient_numerical_policy,
    )
    if len(weight_gradients) != 2:
        raise ValueError(f"expected two routed weight-gradient Contracts, found {len(weight_gradients)}")

    shared_internal = set(shared.boundary.internal_instructions)
    generated_internal = (
        set(forward.region.boundary.internal_instructions),
        *(set(weight.region.boundary.internal_instructions) for weight in weight_gradients),
    )
    if any(shared_internal & internal for internal in generated_internal):
        raise ValueError("shared Contract/multi-Map overlaps another selected generated region")

    shared_outputs = {output.value.instruction for output in shared.outputs}
    input_adjoint_outputs = {output.instruction for output in recovered_input_adjoint.region.boundary.outputs}
    shared_reverse_outputs = shared_outputs & input_adjoint_outputs
    if len(shared_reverse_outputs) != 1:
        raise ValueError(
            "shared Contract/multi-Map must replace exactly one output of the recovered input-adjoint region"
        )
    weight_inputs = tuple({operand.value.instruction for operand in weight.operands} for weight in weight_gradients)
    if any(len(shared_outputs & inputs) != 1 for inputs in weight_inputs):
        raise ValueError("each weight-gradient Contract must consume one shared Map output")
    if set().union(*(shared_outputs & inputs for inputs in weight_inputs)) != shared_outputs:
        raise ValueError("weight-gradient Contracts do not consume every shared Map output")

    input_adjoint_internal = set(recovered_input_adjoint.region.boundary.internal_instructions)
    if not shared_internal & input_adjoint_internal:
        raise ValueError("shared reverse Map does not overlap the recovered input-adjoint region")
    generated_input_adjoint = {
        *(contract.instruction for contract in input_contracts),
        source_fold.instruction,
    }
    retained_wrappers = tuple(
        instruction
        for instruction in recovered_input_adjoint.region.boundary.internal_instructions
        if instruction not in shared_internal and instruction not in generated_input_adjoint
    )
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    opcodes = {instruction.name: instruction.opcode for instruction in entry.instructions}
    unsupported_remainder = tuple(
        instruction
        for instruction in retained_wrappers
        if opcodes[instruction] not in {"bitcast", "copy", "reshape", "slice", "transpose"}
    )
    if unsupported_remainder:
        raise ValueError(f"input-adjoint arithmetic remains outside generated calls: {unsupported_remainder}")
    return RoutedSharedMapTrainingTypedFfiPlan(
        forward=forward,
        input_contracts=(input_contracts[0], input_contracts[1]),
        shared_contract_multi_map=shared,
        source_fold=source_fold,
        recovered_input_adjoint=recovered_input_adjoint,
        retained_input_adjoint_wrappers=retained_wrappers,
        weight_gradients=(weight_gradients[0], weight_gradients[1]),
    )


def replace_routed_shared_map_training_regions_with_custom_calls(
    hlo_text: str,
    plan: RoutedSharedMapTrainingTypedFfiPlan,
    *,
    targets: RoutedSharedMapTrainingFfiTargets,
) -> str:
    """Replace seven nonoverlapping regions while retaining physical wrappers."""
    rewritten = replace_routed_forward_region_with_custom_call(
        hlo_text,
        plan.forward,
        target=targets.forward,
    )
    rewritten = replace_rank_two_contract_with_custom_call(
        rewritten,
        plan.input_contracts[0],
        target=targets.input_contracts[0],
    )
    rewritten = replace_shared_contract_multi_map_region_with_custom_call(
        rewritten,
        plan.shared_contract_multi_map,
        target=targets.shared_contract_multi_map,
    )
    rewritten = replace_rank_two_contract_with_custom_call(
        rewritten,
        plan.input_contracts[1],
        target=targets.input_contracts[1],
    )
    rewritten = replace_source_indexed_fold_with_custom_call(
        rewritten,
        plan.source_fold,
        target=targets.source_fold,
    )
    for weight, target in zip(plan.weight_gradients, targets.weight_gradients, strict=True):
        rewritten = replace_group_batched_contract_with_custom_call(rewritten, weight, target=target)
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_routed_shared_map_training_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: RoutedSharedMapTrainingTypedFfiPlan,
    *,
    targets: RoutedSharedMapTrainingFfiTargets,
) -> RoutedSharedMapTrainingReplacementAudit:
    """Verify generated wiring and the exact input-adjoint work left to XLA."""
    original_module = parse_hlo_module_text(original_hlo)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    original_entry = original_module.computation(original_module.entry)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    users = _entry_users(transformed_entry)
    target_names = (
        targets.forward,
        targets.input_contracts[0],
        targets.shared_contract_multi_map,
        targets.input_contracts[1],
        targets.source_fold,
        *targets.weight_gradients,
    )
    target_instructions = tuple(_unique_target_instruction(transformed_entry, target) for target in target_names)

    forward_instruction = instructions[target_instructions[0]]
    expected_forward_operands = tuple(operand.value.instruction for operand in plan.forward.operands)
    if forward_instruction.operands != expected_forward_operands:
        raise ValueError(
            f"routed forward operands changed: expected {expected_forward_operands}, "
            f"found {forward_instruction.operands}"
        )
    for output in plan.forward.region.boundary.outputs:
        transformed_output = instructions[output.instruction]
        if transformed_output.opcode != "get-tuple-element" or transformed_output.operands != (
            forward_instruction.name,
        ):
            raise ValueError(f"routed forward output %{output.instruction} is not extracted from its generated call")

    shared_audit = audit_shared_contract_multi_map_replacement(
        original_hlo,
        transformed_hlo,
        plan.shared_contract_multi_map,
        target=targets.shared_contract_multi_map,
    )
    input_contract_audits = tuple(
        audit_rank_two_contract_replacement(
            original_hlo,
            transformed_hlo,
            contract,
            target=target,
        )
        for contract, target in zip(plan.input_contracts, targets.input_contracts, strict=True)
    )
    source_fold_audit = audit_source_indexed_fold_replacement(
        original_hlo,
        transformed_hlo,
        plan.source_fold,
        target=targets.source_fold,
    )
    fold_users = tuple(instructions[user] for user in source_fold_audit.external_users)
    fold_collectives = tuple(user for user in fold_users if user.opcode == "all-reduce")
    if len(fold_users) != 1 or len(fold_collectives) != 1:
        raise ValueError(
            "source-indexed Fold must feed exactly one external all-reduce, "
            f"found {[(user.name, user.opcode) for user in fold_users]}"
        )
    weight_collectives: list[str] = []
    for instruction_name, weight in zip(target_instructions[5:], plan.weight_gradients, strict=True):
        instruction = instructions[instruction_name]
        expected_operands = tuple(operand.value.instruction for operand in weight.operands)
        if instruction.operands != expected_operands:
            raise ValueError(
                f"weight Contract %{instruction_name} operands changed: "
                f"expected {expected_operands}, found {instruction.operands}"
            )
        direct_users = tuple(instructions[user] for user in users[instruction_name])
        collectives = tuple(user for user in direct_users if user.opcode == "all-reduce")
        if len(direct_users) != 1 or len(collectives) != 1:
            raise ValueError(
                f"weight Contract %{instruction_name} must feed exactly one external all-reduce, "
                f"found {[(user.name, user.opcode) for user in direct_users]}"
            )
        weight_collectives.append(collectives[0].name)

    missing_wrappers = tuple(
        instruction for instruction in plan.retained_input_adjoint_wrappers if instruction not in instructions
    )
    if missing_wrappers:
        raise ValueError(f"input-adjoint physical wrappers were removed: {missing_wrappers}")

    copy_count = (
        _opcode_count(original_entry, "copy"),
        _opcode_count(transformed_entry, "copy"),
    )
    transpose_count = (
        _opcode_count(original_entry, "transpose"),
        _opcode_count(transformed_entry, "transpose"),
    )
    if copy_count[1] > copy_count[0]:
        raise ValueError(f"replacement added copies: {copy_count[0]} -> {copy_count[1]}")
    if transpose_count[1] > transpose_count[0]:
        raise ValueError(f"replacement added transposes: {transpose_count[0]} -> {transpose_count[1]}")
    return RoutedSharedMapTrainingReplacementAudit(
        target_instructions=(
            target_instructions[0],
            target_instructions[1],
            target_instructions[2],
            target_instructions[3],
            target_instructions[4],
            target_instructions[5],
            target_instructions[6],
        ),
        input_contracts=(input_contract_audits[0], input_contract_audits[1]),
        shared_contract_multi_map=shared_audit,
        source_fold=source_fold_audit,
        source_fold_collective=fold_collectives[0].name,
        weight_gradient_collectives=(weight_collectives[0], weight_collectives[1]),
        retained_input_adjoint_wrappers=plan.retained_input_adjoint_wrappers,
        copy_count=copy_count,
        transpose_count=transpose_count,
    )


def _unique_target_instruction(entry: HloComputation, target: str) -> str:
    attribute = f'custom_call_target="{target}"'
    matches = tuple(
        instruction.name
        for instruction in entry.instructions
        if instruction.opcode == "custom-call" and attribute in instruction.attributes
    )
    if len(matches) != 1:
        raise ValueError(f"expected one post-roundtrip custom call for {target!r}, found {len(matches)}")
    return matches[0]


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    users: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)
    return {instruction: tuple(values) for instruction, values in users.items()}


def _opcode_count(entry: HloComputation, opcode: str) -> int:
    return sum(instruction.opcode == opcode for instruction in entry.instructions)
