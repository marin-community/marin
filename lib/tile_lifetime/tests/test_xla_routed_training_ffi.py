# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import pytest

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import RoutedForwardFfiOperandRole, RoutedInputAdjointFfiOperandRole
from tile_lifetime.xla_routed_forward_ffi import generate_cuda_routed_forward_ffi
from tile_lifetime.xla_routed_input_adjoint_ffi import generate_cuda_routed_input_adjoint_ffi
from tile_lifetime.xla_routed_training_ffi import (
    RoutedTrainingFfiTargets,
    entry_parameter_ancestors,
    plan_routed_training_typed_ffi,
    replace_routed_training_regions_with_custom_calls,
)
from tile_lifetime.xla_routed_weight_gradient_ffi import generate_cuda_group_batched_contract_ffi

_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_forward_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
)
_TARGETS = RoutedTrainingFfiTargets(
    forward="shuttle.routed_training.forward.test",
    input_adjoint="shuttle.routed_training.input_adjoint.test",
    weight_gradients=(
        "shuttle.routed_training.weight_gradient.0.test",
        "shuttle.routed_training.weight_gradient.1.test",
    ),
)


def _hlo() -> str:
    return gzip.decompress(_ARTIFACT.read_bytes()).decode()


def test_routed_training_plan_recovers_four_independent_regions() -> None:
    plan = plan_routed_training_typed_ffi(_hlo())

    assert plan.forward.region.insertion_instruction == "scatter-add.39"
    assert plan.input_adjoint.region.insertion_instruction == "scatter-add.42"
    assert tuple(weight.region.insertion_instruction for weight in plan.weight_gradients) == ("dot.6", "dot.7")
    assert tuple(weight.region.external_collectives for weight in plan.weight_gradients) == (
        ("psum.52",),
        ("psum.53",),
    )
    input_outputs = {output.instruction for output in plan.input_adjoint.region.boundary.outputs}
    first_weight_inputs = {operand.value.instruction for operand in plan.weight_gradients[0].operands}
    assert input_outputs & first_weight_inputs == {"select.7"}
    assert plan.forward.region.numerical_policy.value == "source_ordered"
    assert plan.input_adjoint.region.numerical_policy.value == "source_ordered"
    assert all(
        weight.numerical_contract.numerical_policy is NumericalPolicy.ALLOW_ROUNDING_REORDER
        for weight in plan.weight_gradients
    )


def test_routed_training_replacement_preserves_wiring_and_collectives() -> None:
    hlo = _hlo()
    plan = plan_routed_training_typed_ffi(hlo)
    rewritten = replace_routed_training_regions_with_custom_calls(hlo, plan, targets=_TARGETS)

    assert all(rewritten.count(target) == 1 for target in (_TARGETS.forward, _TARGETS.input_adjoint))
    assert all(rewritten.count(target) == 1 for target in _TARGETS.weight_gradients)
    assert "%select.7 = bf16[4,512,64]{2,1,0} get-tuple-element" in rewritten
    assert "%dot.6 = bf16[4,32,64]{2,1,0} custom-call(%select.6, %select.7)" in rewritten
    assert "%psum.52 = bf16[4,32,64]{2,1,0} all-reduce(%dot.6)" in rewritten
    assert "%psum.53 = bf16[4,32,32]{2,1,0} all-reduce(%dot.7)" in rewritten
    assert rewritten.count(" copy(") <= hlo.count(" copy(")
    assert rewritten.count(" transpose(") <= hlo.count(" transpose(")
    parse_hlo_module_text(rewritten)


def test_routed_training_all_dynamic_operands_have_parameter_ancestry() -> None:
    hlo = _hlo()
    plan = plan_routed_training_typed_ffi(hlo)
    operand_names = tuple(
        dict.fromkeys(
            (
                *(operand.value.instruction for operand in plan.forward.operands),
                *(operand.value.instruction for operand in plan.input_adjoint.operands),
                *(operand.value.instruction for weight in plan.weight_gradients for operand in weight.operands),
            )
        )
    )
    ancestors = entry_parameter_ancestors(hlo, operand_names)
    static_values = {value for value, parameters in ancestors.items() if not parameters}

    input_roles = {operand.value.instruction: operand.role for operand in plan.input_adjoint.operands}
    assert static_values == {
        value for value, role in input_roles.items() if role is RoutedInputAdjointFfiOperandRole.FOLD_INITIAL
    }
    assert all(
        ancestors[operand.value.instruction]
        for operand in plan.forward.operands
        if operand.role is not RoutedForwardFfiOperandRole.FOLD_INITIAL
    )


def test_routed_training_map_mutation_regenerates_only_affected_body() -> None:
    hlo = _hlo()
    original = "%mul.972 = bf16[16,32]{1,0} multiply(%mul.83, %slice.54)"
    mutated = "%mul.972 = bf16[16,32]{1,0} add(%mul.83, %slice.54)"
    assert hlo.count(original) == 1
    baseline_plan = plan_routed_training_typed_ffi(hlo)
    mutated_plan = plan_routed_training_typed_ffi(hlo.replace(original, mutated, 1))
    baseline_input = generate_cuda_routed_input_adjoint_ffi(
        baseline_plan.input_adjoint,
        target=_TARGETS.input_adjoint,
    )
    mutated_input = generate_cuda_routed_input_adjoint_ffi(
        mutated_plan.input_adjoint,
        target=_TARGETS.input_adjoint,
    )
    baseline_forward = generate_cuda_routed_forward_ffi(baseline_plan.forward, target=_TARGETS.forward)
    mutated_forward = generate_cuda_routed_forward_ffi(mutated_plan.forward, target=_TARGETS.forward)
    baseline_weights = tuple(
        generate_cuda_group_batched_contract_ffi(weight, target=target)
        for weight, target in zip(baseline_plan.weight_gradients, _TARGETS.weight_gradients, strict=True)
    )
    mutated_weights = tuple(
        generate_cuda_group_batched_contract_ffi(weight, target=target)
        for weight, target in zip(mutated_plan.weight_gradients, _TARGETS.weight_gradients, strict=True)
    )

    assert baseline_input.semantic_digest != mutated_input.semantic_digest
    assert baseline_input.source_digest != mutated_input.source_digest
    assert baseline_forward.source_digest == mutated_forward.source_digest
    assert tuple(source.source_digest for source in baseline_weights) == tuple(
        source.source_digest for source in mutated_weights
    )


def test_routed_training_rejects_bitwise_weight_reduction_claim() -> None:
    with pytest.raises(ValueError, match="unspecified bitwise dot reduction tree"):
        plan_routed_training_typed_ffi(
            _hlo(),
            weight_gradient_numerical_policy=NumericalPolicy.BITWISE_EXACT,
        )
