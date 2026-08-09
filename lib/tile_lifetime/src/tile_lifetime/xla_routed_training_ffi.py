# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose independent routed training regions under one XLA transformation."""

from __future__ import annotations

from dataclasses import dataclass

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import HloComputation, parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import (
    RoutedForwardCodegenDisposition,
    RoutedForwardTypedFfiCodegenPlan,
    RoutedInputAdjointTypedFfiCodegenPlan,
    RoutedWeightGradientTypedFfiCodegenPlan,
    plan_routed_forward_typed_ffi,
    plan_routed_input_adjoint_typed_ffi,
    plan_routed_weight_gradient_typed_ffi,
)
from tile_lifetime.xla_routed_forward_ffi import replace_routed_forward_region_with_custom_call
from tile_lifetime.xla_routed_input_adjoint_ffi import replace_routed_input_adjoint_region_with_custom_call
from tile_lifetime.xla_routed_weight_gradient_ffi import replace_group_batched_contract_with_custom_call


@dataclass(frozen=True)
class RoutedTrainingFfiTargets:
    """Independent typed-FFI targets used by one routed training transform."""

    forward: str
    input_adjoint: str
    weight_gradients: tuple[str, str]


@dataclass(frozen=True)
class RoutedTrainingTypedFfiPlan:
    """Four generated regions composed without changing their physical bodies."""

    forward: RoutedForwardTypedFfiCodegenPlan
    input_adjoint: RoutedInputAdjointTypedFfiCodegenPlan
    weight_gradients: tuple[RoutedWeightGradientTypedFfiCodegenPlan, RoutedWeightGradientTypedFfiCodegenPlan]


@dataclass(frozen=True)
class RoutedTrainingReplacementAudit:
    """Verified physical wiring after the composed HLO round trip."""

    target_instructions: tuple[str, str, str, str]
    weight_gradient_collectives: tuple[str, str]
    input_adjoint_auxiliary: str
    copy_count: tuple[int, int]
    transpose_count: tuple[int, int]


def plan_routed_training_typed_ffi(
    hlo_text: str,
    *,
    weight_gradient_numerical_policy: NumericalPolicy = NumericalPolicy.ALLOW_ROUNDING_REORDER,
) -> RoutedTrainingTypedFfiPlan:
    """Recover one forward, one input-adjoint, and two weight Contracts."""
    forward = plan_routed_forward_typed_ffi(hlo_text)
    if forward.disposition is not RoutedForwardCodegenDisposition.READY:
        raise ValueError("routed forward region lacks a verified physical layout")
    input_adjoint = plan_routed_input_adjoint_typed_ffi(hlo_text)
    weight_gradients = plan_routed_weight_gradient_typed_ffi(
        hlo_text,
        numerical_policy=weight_gradient_numerical_policy,
    )
    if len(weight_gradients) != 2:
        raise ValueError(f"expected two routed weight-gradient Contracts, found {len(weight_gradients)}")
    first_weight, second_weight = weight_gradients

    input_outputs = {output.instruction: output.shape for output in input_adjoint.region.boundary.outputs}
    first_weight_inputs = {operand.value.instruction: operand.value.shape for operand in first_weight.operands}
    shared_values = set(input_outputs) & set(first_weight_inputs)
    if len(shared_values) != 1:
        raise ValueError("input-adjoint and first weight Contract must share one live auxiliary value")
    shared_value = next(iter(shared_values))
    if input_outputs[shared_value] != first_weight_inputs[shared_value]:
        raise ValueError("input-adjoint auxiliary and weight-Contract operand shapes disagree")
    if any(plan.output_alias_operand is not None for plan in weight_gradients):
        raise ValueError("weight-gradient Contracts require fresh outputs before placement collectives")
    if len({collective for plan in weight_gradients for collective in plan.region.external_collectives}) != 2:
        raise ValueError("weight-gradient Contracts require two distinct external placement collectives")
    return RoutedTrainingTypedFfiPlan(
        forward=forward,
        input_adjoint=input_adjoint,
        weight_gradients=(first_weight, second_weight),
    )


def replace_routed_training_regions_with_custom_calls(
    hlo_text: str,
    plan: RoutedTrainingTypedFfiPlan,
    *,
    targets: RoutedTrainingFfiTargets,
) -> str:
    """Apply four independent generic custom calls to one physical HLO module."""
    rewritten = replace_routed_forward_region_with_custom_call(
        hlo_text,
        plan.forward,
        target=targets.forward,
    )
    rewritten = replace_routed_input_adjoint_region_with_custom_call(
        rewritten,
        plan.input_adjoint,
        target=targets.input_adjoint,
    )
    for weight_plan, target in zip(plan.weight_gradients, targets.weight_gradients, strict=True):
        rewritten = replace_group_batched_contract_with_custom_call(
            rewritten,
            weight_plan,
            target=target,
        )
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_routed_training_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: RoutedTrainingTypedFfiPlan,
    *,
    targets: RoutedTrainingFfiTargets,
) -> RoutedTrainingReplacementAudit:
    """Verify that four independent calls retain their required dataflow.

    This audit intentionally follows post-roundtrip HLO users instead of
    relying on source instruction numbers for the placement collectives.
    """
    original_module = parse_hlo_module_text(original_hlo)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    original_entry = original_module.computation(original_module.entry)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    transformed_instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    users: dict[str, list[str]] = {instruction.name: [] for instruction in transformed_entry.instructions}
    for instruction in transformed_entry.instructions:
        for operand in instruction.operands:
            users[operand].append(instruction.name)

    target_names = (targets.forward, targets.input_adjoint, *targets.weight_gradients)
    target_instructions: list[str] = []
    for target in target_names:
        target_attribute = f'custom_call_target="{target}"'
        matches = tuple(
            instruction
            for instruction in transformed_entry.instructions
            if instruction.opcode == "custom-call" and target_attribute in instruction.attributes
        )
        if len(matches) != 1:
            raise ValueError(f"expected one post-roundtrip custom call for {target!r}, found {len(matches)}")
        target_instructions.append(matches[0].name)

    weight_collectives: list[str] = []
    for target_instruction, weight_plan in zip(target_instructions[2:], plan.weight_gradients, strict=True):
        instruction = transformed_instructions[target_instruction]
        expected_operands = tuple(operand.value.instruction for operand in weight_plan.operands)
        if instruction.operands != expected_operands:
            raise ValueError(
                f"weight Contract %{target_instruction} operands changed: "
                f"expected {expected_operands}, found {instruction.operands}"
            )
        direct_users = tuple(transformed_instructions[user] for user in users[target_instruction])
        collectives = tuple(user for user in direct_users if user.opcode == "all-reduce")
        if len(direct_users) != 1 or len(collectives) != 1:
            raise ValueError(
                f"weight Contract %{target_instruction} must feed exactly one external all-reduce, "
                f"found {[(user.name, user.opcode) for user in direct_users]}"
            )
        weight_collectives.append(collectives[0].name)

    input_outputs = {output.instruction for output in plan.input_adjoint.region.boundary.outputs}
    first_weight_inputs = {operand.value.instruction for operand in plan.weight_gradients[0].operands}
    auxiliary = next(iter(input_outputs & first_weight_inputs))
    auxiliary_instruction = transformed_instructions[auxiliary]
    if auxiliary_instruction.opcode != "get-tuple-element":
        raise ValueError(f"input-adjoint auxiliary %{auxiliary} is not extracted from the generated tuple")
    if auxiliary not in transformed_instructions[target_instructions[2]].operands:
        raise ValueError("first weight Contract no longer consumes the input-adjoint auxiliary")

    def opcode_count(entry: HloComputation, opcode: str) -> int:
        return sum(instruction.opcode == opcode for instruction in entry.instructions)

    copy_count = (opcode_count(original_entry, "copy"), opcode_count(transformed_entry, "copy"))
    transpose_count = (opcode_count(original_entry, "transpose"), opcode_count(transformed_entry, "transpose"))
    if copy_count[1] > copy_count[0]:
        raise ValueError(f"replacement added copies: {copy_count[0]} -> {copy_count[1]}")
    if transpose_count[1] > transpose_count[0]:
        raise ValueError(f"replacement added transposes: {transpose_count[0]} -> {transpose_count[1]}")
    forward_instruction, input_instruction, first_weight_instruction, second_weight_instruction = target_instructions
    first_collective, second_collective = weight_collectives
    return RoutedTrainingReplacementAudit(
        target_instructions=(
            forward_instruction,
            input_instruction,
            first_weight_instruction,
            second_weight_instruction,
        ),
        weight_gradient_collectives=(first_collective, second_collective),
        input_adjoint_auxiliary=auxiliary,
        copy_count=copy_count,
        transpose_count=transpose_count,
    )


def entry_parameter_ancestors(hlo_text: str, values: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Return runtime parameter ancestors for selected entry values."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    def ancestors(name: str) -> tuple[str, ...]:
        pending = [name]
        seen: set[str] = set()
        parameters: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            instruction = instructions[current]
            if instruction.opcode == "parameter":
                parameters.add(current)
            else:
                pending.extend(instruction.operands)
        return tuple(sorted(parameters))

    return {value: ancestors(value) for value in values}
