# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json
import statistics
from pathlib import Path

import pytest

from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import RoutedForwardFfiOperandRole, RoutedInputAdjointFfiOperandRole
from tile_lifetime.xla_routed_forward_ffi import generate_cuda_routed_forward_ffi
from tile_lifetime.xla_routed_input_adjoint_ffi import generate_cuda_routed_input_adjoint_ffi
from tile_lifetime.xla_routed_training_ffi import (
    RoutedTrainingFfiTargets,
    audit_routed_training_replacement,
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
_COMBINED_GPU_ARTIFACT = Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0"


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
    audit = audit_routed_training_replacement(hlo, rewritten, plan, targets=_TARGETS)

    assert all(rewritten.count(target) == 1 for target in (_TARGETS.forward, _TARGETS.input_adjoint))
    assert all(rewritten.count(target) == 1 for target in _TARGETS.weight_gradients)
    assert "%select.7 = bf16[4,512,64]{2,1,0} get-tuple-element" in rewritten
    assert "%dot.6 = bf16[4,32,64]{2,1,0} custom-call(%select.6, %select.7)" in rewritten
    assert "%psum.52 = bf16[4,32,64]{2,1,0} all-reduce(%dot.6)" in rewritten
    assert "%psum.53 = bf16[4,32,32]{2,1,0} all-reduce(%dot.7)" in rewritten
    assert rewritten.count(" copy(") <= hlo.count(" copy(")
    assert rewritten.count(" transpose(") <= hlo.count(" transpose(")
    assert audit.target_instructions == (
        "shuttle_generated_routed_forward_region",
        "shuttle_generated_routed_input_adjoint_region",
        "dot.6",
        "dot.7",
    )
    assert audit.weight_gradient_collectives == ("psum.52", "psum.53")
    assert audit.input_adjoint_auxiliary == "select.7"
    assert audit.copy_count[1] <= audit.copy_count[0]
    assert audit.transpose_count[1] <= audit.transpose_count[0]
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


def test_routed_training_gb200_artifact_preserves_acceptance_evidence() -> None:
    checksums = (_COMBINED_GPU_ARTIFACT / "SHA256SUMS").read_text().splitlines()
    assert len(checksums) == 21
    for record in checksums:
        expected, relative_path = record.split("  ", maxsplit=1)
        payload = (_COMBINED_GPU_ARTIFACT / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected

    summary = json.loads((_COMBINED_GPU_ARTIFACT / "summary.json").read_text())
    assert summary["device_kind"] == "NVIDIA GB200"
    assert summary["architecture"] == "sm_100a"
    assert summary["custom_call_occurrences_in_transformed_hlo"] == {
        "forward": 1,
        "input_adjoint": 1,
        "weight_gradients": [1, 1],
    }
    assert summary["custom_call_handler_executions"] == {
        "forward": 35,
        "input_adjoint": 35,
        "weight_gradients": [35, 35],
    }
    assert summary["external_collectives"] == ["psum.58", "psum.59"]
    assert summary["input_adjoint_auxiliary"] == "select.7"
    assert summary["copy_count"] == {"original": 0, "transformed": 0}
    assert summary["transpose_count"] == {"original": 51, "transformed": 50}
    assert not summary["uses_atomic_accumulation"]
    assert summary["output_alias_operands"] == [None, None]
    assert summary["static_operand_roles"] == ["fold_initial"]
    assert summary["outputs_match"]
    assert summary["maximum_absolute_error"] < 4e-9
    assert summary["mean_absolute_error"] < 2e-12
    assert summary["bitwise_equal_leaf_count"] == 49
    assert summary["result_leaf_count"] == 53

    samples = summary["raw_samples"]
    assert len(samples) == 30
    assert sum(sample["order"][0] == "baseline" for sample in samples) == 15
    assert sum(sample["order"][0] == "transformed" for sample in samples) == 15
    baseline_median = statistics.median(sample["baseline"]["latency_ms"] for sample in samples)
    generated_median = statistics.median(sample["transformed"]["latency_ms"] for sample in samples)
    assert baseline_median == summary["baseline_median_ms"]
    assert generated_median == summary["generated_median_ms"]
    assert generated_median / baseline_median == summary["generated_over_baseline"]
    assert summary["generated_over_baseline"] < 1.2
    assert len(summary["generated_unique_output_hashes"]) == 1
