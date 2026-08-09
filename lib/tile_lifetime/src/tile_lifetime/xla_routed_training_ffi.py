# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose independent routed training regions under one XLA transformation."""

from __future__ import annotations

from dataclasses import dataclass

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
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
