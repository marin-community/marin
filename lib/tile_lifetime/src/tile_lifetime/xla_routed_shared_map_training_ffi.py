# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose routed training calls around one shared Contract/multi-Map region.

The shared region produces both a forward Map value and the corresponding
reverse Map value.  The surrounding input-adjoint Contract/Fold remains in XLA
at this boundary so no physical instruction is owned by two generated calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import HloComputation, parse_hlo_module_text
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
from tile_lifetime.xla_routed_forward_ffi import replace_routed_forward_region_with_custom_call
from tile_lifetime.xla_routed_weight_gradient_ffi import replace_group_batched_contract_with_custom_call
from tile_lifetime.xla_shared_contract_multimap import (
    SharedContractMultiMapReplacementAudit,
    audit_shared_contract_multi_map_replacement,
    replace_shared_contract_multi_map_region_with_custom_call,
)


@dataclass(frozen=True)
class RoutedSharedMapTrainingFfiTargets:
    """Typed-FFI targets for one nonoverlapping routed composition."""

    forward: str
    shared_contract_multi_map: str
    weight_gradients: tuple[str, str]


@dataclass(frozen=True)
class RoutedSharedMapTrainingTypedFfiPlan:
    """Generated regions plus the input-adjoint region intentionally deferred."""

    forward: RoutedForwardTypedFfiCodegenPlan
    shared_contract_multi_map: SharedContractMultiMapRegionRecord
    deferred_input_adjoint: RoutedInputAdjointTypedFfiCodegenPlan
    weight_gradients: tuple[RoutedWeightGradientTypedFfiCodegenPlan, RoutedWeightGradientTypedFfiCodegenPlan]


@dataclass(frozen=True)
class RoutedSharedMapTrainingReplacementAudit:
    """Post-roundtrip wiring evidence for the nonoverlapping composition."""

    target_instructions: tuple[str, str, str, str]
    shared_contract_multi_map: SharedContractMultiMapReplacementAudit
    weight_gradient_collectives: tuple[str, str]
    deferred_input_adjoint_instructions: tuple[str, ...]
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
    deferred = plan_routed_input_adjoint_typed_ffi(hlo_text)
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
    deferred_outputs = {output.instruction for output in deferred.region.boundary.outputs}
    shared_reverse_outputs = shared_outputs & deferred_outputs
    if len(shared_reverse_outputs) != 1:
        raise ValueError(
            "shared Contract/multi-Map must replace exactly one output of the deferred input-adjoint region"
        )
    weight_inputs = tuple({operand.value.instruction for operand in weight.operands} for weight in weight_gradients)
    if any(len(shared_outputs & inputs) != 1 for inputs in weight_inputs):
        raise ValueError("each weight-gradient Contract must consume one shared Map output")
    if set().union(*(shared_outputs & inputs for inputs in weight_inputs)) != shared_outputs:
        raise ValueError("weight-gradient Contracts do not consume every shared Map output")

    deferred_internal = set(deferred.region.boundary.internal_instructions)
    if not shared_internal & deferred_internal:
        raise ValueError("shared reverse Map does not overlap the deferred input-adjoint region")
    if not deferred_internal - shared_internal:
        raise ValueError("shared replacement unexpectedly owns the complete input-adjoint region")
    return RoutedSharedMapTrainingTypedFfiPlan(
        forward=forward,
        shared_contract_multi_map=shared,
        deferred_input_adjoint=deferred,
        weight_gradients=(weight_gradients[0], weight_gradients[1]),
    )


def replace_routed_shared_map_training_regions_with_custom_calls(
    hlo_text: str,
    plan: RoutedSharedMapTrainingTypedFfiPlan,
    *,
    targets: RoutedSharedMapTrainingFfiTargets,
) -> str:
    """Replace four nonoverlapping regions while retaining the adjoint remainder."""
    rewritten = replace_routed_forward_region_with_custom_call(
        hlo_text,
        plan.forward,
        target=targets.forward,
    )
    rewritten = replace_shared_contract_multi_map_region_with_custom_call(
        rewritten,
        plan.shared_contract_multi_map,
        target=targets.shared_contract_multi_map,
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
    target_names = (targets.forward, targets.shared_contract_multi_map, *targets.weight_gradients)
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
    weight_collectives: list[str] = []
    for instruction_name, weight in zip(target_instructions[2:], plan.weight_gradients, strict=True):
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

    shared_internal = set(plan.shared_contract_multi_map.boundary.internal_instructions)
    deferred = tuple(
        instruction
        for instruction in plan.deferred_input_adjoint.region.boundary.internal_instructions
        if instruction not in shared_internal
    )
    missing_deferred = tuple(instruction for instruction in deferred if instruction not in instructions)
    if missing_deferred:
        raise ValueError(f"input-adjoint instructions outside the shared Map were removed: {missing_deferred}")

    copy_count = (_opcode_count(original_entry, "copy"), _opcode_count(transformed_entry, "copy"))
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
        ),
        shared_contract_multi_map=shared_audit,
        weight_gradient_collectives=(weight_collectives[0], weight_collectives[1]),
        deferred_input_adjoint_instructions=deferred,
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
