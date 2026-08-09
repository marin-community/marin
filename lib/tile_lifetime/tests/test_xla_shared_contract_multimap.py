# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import re
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lib.tile_lifetime.benchmarks.xla_grug_backward_multi_output_gpu_custom_call_smoke import _tree_hash_evidence
from lib.tile_lifetime.benchmarks.xla_grug_routed_combined_gpu_custom_call import (
    _ROUTED_ATTENTION_TARGETS,
    _SHARED_ROUTED_TARGETS,
    _attention_reverse_program,
    _audit_shared_map_composition,
    _axis_fold_reassociation_report,
    _generate_axis_fold_programs,
    _plan_shared_map_composition,
    _replace_shared_map_composition,
    _single_custom_call_target_occurrences,
)
from tile_lifetime.jax_streaming_attention_backward_ffi import generate_streaming_attention_backward_ffi
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import (
    SharedContractMapDependence,
    SharedContractMultiMapOperandRole,
    form_shared_contract_multi_map_region,
)
from tile_lifetime.xla_routed_shared_map_training_ffi import (
    RoutedSharedMapTrainingFfiTargets,
    audit_routed_shared_map_training_replacement,
    plan_routed_shared_map_training_typed_ffi,
    replace_routed_shared_map_training_regions_with_custom_calls,
)
from tile_lifetime.xla_shared_contract_multimap import (
    evaluate_shared_contract_multi_map_plan,
)
from tile_lifetime.xla_shared_contract_multimap_ffi import (
    compile_shared_contract_multi_map_ffi,
    generate_cuda_shared_contract_multi_map_ffi,
)
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    derive_streaming_attention_backward_ffi_output_layouts,
    plan_streaming_attention_backward_hlo_region_replacement,
)

_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_forward_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
)
_TARGET = "shuttle.shared_contract_multi_map.test"
_COMPOSED_TARGETS = RoutedSharedMapTrainingFfiTargets(
    forward="shuttle.shared_map_training.forward.test",
    input_contracts=(
        "shuttle.shared_map_training.input_contract.0.test",
        "shuttle.shared_map_training.input_contract.1.test",
    ),
    shared_contract_multi_map="shuttle.shared_map_training.maps.test",
    source_fold="shuttle.shared_map_training.source_fold.test",
    weight_gradients=(
        "shuttle.shared_map_training.weight_gradient.0.test",
        "shuttle.shared_map_training.weight_gradient.1.test",
    ),
)


def _hlo() -> str:
    return gzip.decompress(_ARTIFACT.read_bytes()).decode()


def _bf16(value: np.ndarray) -> np.ndarray:
    return np.asarray(jnp.asarray(value, dtype=jnp.bfloat16).astype(jnp.float32))


def test_output_hash_evidence_identifies_the_changed_leaf() -> None:
    original = {"state": (np.asarray([1.0], dtype=np.float32), np.asarray([2], dtype=np.int32))}
    mutated = {"state": (np.asarray([1.0], dtype=np.float32), np.asarray([3], dtype=np.int32))}

    original_hash, original_leaves = _tree_hash_evidence(original)
    mutated_hash, mutated_leaves = _tree_hash_evidence(mutated)

    assert original_hash != mutated_hash
    assert [leaf["path"] for leaf in original_leaves] == ["['state'][0]", "['state'][1]"]
    assert original_leaves[0]["sha256"] == mutated_leaves[0]["sha256"]
    assert original_leaves[1]["sha256"] != mutated_leaves[1]["sha256"]


def test_natural_hlo_recovers_one_shared_contract_with_two_live_scalar_maps() -> None:
    plan = form_shared_contract_multi_map_region(_hlo())

    assert plan.convex
    assert plan.topologically_insertable
    assert plan.contract.output_shape == "bf16[512,64]{1,0}"
    assert plan.contract.dimensions.lhs_contracting == (1,)
    assert plan.contract.dimensions.rhs_contracting == (0,)
    assert tuple(operand.role for operand in plan.operands) == tuple(SharedContractMultiMapOperandRole)
    assert tuple(operand.value.instruction for operand in plan.operands) == (
        "reshape.354",
        "reshape.355",
        "slice.54",
        "and.34",
        "and.35",
    )
    assert tuple(output.dependence for output in plan.outputs) == (
        SharedContractMapDependence.CONTRACT_ONLY,
        SharedContractMapDependence.CONTRACT_AND_AUXILIARY,
    )
    assert tuple(output.value.instruction for output in plan.outputs) == ("select.5", "select.7")
    assert tuple(output.value.shape for output in plan.outputs) == (
        "bf16[4,512,32]{2,1,0}",
        "bf16[4,512,64]{2,1,0}",
    )
    assert tuple(output.external_users for output in plan.outputs) == (
        ("transpose.169", "dot.7"),
        ("transpose.167", "dot.6"),
    )
    assert plan.boundary.internal_instructions.count("dot.66") == 1
    assert "dot.67" not in plan.boundary.internal_instructions
    assert "dot.68" not in plan.boundary.internal_instructions
    assert "dot.69" not in plan.boundary.internal_instructions
    assert plan.numerical_contract.scalar_policy.value == "source_ordered"
    assert plan.numerical_contract.numerical_policy is NumericalPolicy.ALLOW_ROUNDING_REORDER


def test_routed_training_composition_generates_input_adjoint_arithmetic_and_retains_views() -> None:
    plan = plan_routed_shared_map_training_typed_ffi(_hlo())

    shared_internal = set(plan.shared_contract_multi_map.boundary.internal_instructions)
    input_adjoint_internal = set(plan.recovered_input_adjoint.region.boundary.internal_instructions)
    assert shared_internal & input_adjoint_internal
    assert {output.value.instruction for output in plan.shared_contract_multi_map.outputs} == {
        "select.5",
        "select.7",
    }
    assert tuple(contract.instruction for contract in plan.input_contracts) == ("dot.67", "dot.68")
    assert plan.source_fold.instruction == "scatter-add.42"
    assert plan.retained_input_adjoint_wrappers == (
        "slice.54",
        "transpose.167",
        "reshape.360",
        "slice.58",
        "reshape.408",
    )
    assert all(
        not shared_internal & set(weight.region.boundary.internal_instructions) for weight in plan.weight_gradients
    )


def test_routed_training_composition_replaces_all_input_adjoint_arithmetic_once() -> None:
    hlo = _hlo()
    plan = plan_routed_shared_map_training_typed_ffi(hlo)

    rewritten = replace_routed_shared_map_training_regions_with_custom_calls(
        hlo,
        plan,
        targets=_COMPOSED_TARGETS,
    )
    audit = audit_routed_shared_map_training_replacement(
        hlo,
        rewritten,
        plan,
        targets=_COMPOSED_TARGETS,
    )

    assert audit.target_instructions == (
        "shuttle_generated_routed_forward_region",
        "dot.67",
        "shuttle_generated_shared_contract_multi_map",
        "dot.68",
        "scatter-add.42",
        "dot.6",
        "dot.7",
    )
    assert audit.weight_gradient_collectives == ("psum.52", "psum.53")
    assert audit.source_fold_collective == "psum.50"
    assert audit.shared_contract_multi_map.outputs == ("select.5", "select.7")
    assert audit.retained_input_adjoint_wrappers == plan.retained_input_adjoint_wrappers
    assert "shuttle_generated_routed_input_adjoint_region" not in rewritten
    assert audit.copy_count[1] <= audit.copy_count[0]
    assert audit.transpose_count[1] <= audit.transpose_count[0]


def test_shared_map_harness_composes_generated_input_adjoint_calls() -> None:
    hlo = _hlo()
    attention_program, attention_schedule, _ = _attention_reverse_program()
    default_attention = generate_streaming_attention_backward_ffi(
        attention_program,
        attention_schedule,
        target_name=_ROUTED_ATTENTION_TARGETS.attention_backward,
    )
    default_attention_plan = plan_streaming_attention_backward_hlo_region_replacement(
        hlo,
        attention_program,
        default_attention,
    )
    attention = generate_streaming_attention_backward_ffi(
        attention_program,
        attention_schedule,
        target_name=_ROUTED_ATTENTION_TARGETS.attention_backward,
        output_layouts=derive_streaming_attention_backward_ffi_output_layouts(default_attention_plan),
    )
    axis_folds = _generate_axis_fold_programs(hlo)
    plan = _plan_shared_map_composition(hlo, attention_program, attention, axis_folds)

    assert _axis_fold_reassociation_report(plan.axis_folds) == ["deterministic_tree", "deterministic_tree"]

    transformed = _replace_shared_map_composition(hlo, plan)
    audit = _audit_shared_map_composition(hlo, transformed, plan)

    targets = (
        _SHARED_ROUTED_TARGETS.forward,
        *_SHARED_ROUTED_TARGETS.input_contracts,
        _SHARED_ROUTED_TARGETS.shared_contract_multi_map,
        _SHARED_ROUTED_TARGETS.source_fold,
        *_SHARED_ROUTED_TARGETS.weight_gradients,
        _ROUTED_ATTENTION_TARGETS.attention_backward,
    )
    selected_targets = (*targets, *(generated.target_name for generated in axis_folds))
    exact_occurrences = _single_custom_call_target_occurrences(transformed, selected_targets)
    assert set(exact_occurrences.values()) == {1}
    assert transformed.count("shuttle.routed_training.input_adjoint.v2") == 0
    assert audit.routed.retained_input_adjoint_wrappers == plan.routed.retained_input_adjoint_wrappers
    assert len(audit.axis_folds) == 2
    assert audit.routed.shared_contract_multi_map.outputs == ("select.5", "select.7")


def test_shared_contract_multi_map_cpu_reference_matches_source_ordered_formula() -> None:
    plan = form_shared_contract_multi_map_region(_hlo())
    rng = np.random.default_rng(19)
    lhs = _bf16(rng.normal(scale=0.15, size=(512, 128)).astype(np.float32))
    rhs = _bf16(rng.normal(scale=0.15, size=(128, 64)).astype(np.float32))
    cotangent = _bf16(rng.normal(scale=0.2, size=(16, 32)).astype(np.float32))
    forward_validity = np.zeros((4, 512, 32), dtype=np.bool_)
    reverse_validity = np.zeros((4, 512, 64), dtype=np.bool_)
    for row in range(16):
        forward_validity[row % 4, row, :] = True
        reverse_validity[row % 4, row, :] = True

    observed_forward, observed_reverse = evaluate_shared_contract_multi_map_plan(
        plan,
        (lhs, rhs, cotangent, forward_validity, reverse_validity),
    )

    projection = _bf16(lhs.astype(np.float32) @ rhs.astype(np.float32))[:16]
    gate = projection[:, :32]
    up = projection[:, 32:]
    sigmoid = _bf16(1.0 / _bf16(_bf16(np.exp(_bf16(-gate))) + 1.0))
    activated = _bf16(gate * sigmoid)
    forward = _bf16(activated * up)
    upstream_times_up = _bf16(cotangent * up)
    gate_gradient = _bf16(
        _bf16(upstream_times_up * sigmoid)
        + _bf16(_bf16(gate * upstream_times_up) * _bf16(sigmoid * _bf16(1.0 - sigmoid)))
    )
    up_gradient = _bf16(activated * cotangent)
    reverse = np.concatenate((gate_gradient, up_gradient), axis=1)
    expected_forward = np.zeros((4, 512, 32), dtype=np.float32)
    expected_reverse = np.zeros((4, 512, 64), dtype=np.float32)
    for row in range(16):
        expected_forward[row % 4, row] = forward[row]
        expected_reverse[row % 4, row] = reverse[row]

    assert np.array_equal(observed_forward, expected_forward)
    assert np.array_equal(observed_reverse, expected_reverse)
    repeated = evaluate_shared_contract_multi_map_plan(
        plan,
        (lhs, rhs, cotangent, forward_validity, reverse_validity),
    )
    assert all(
        np.array_equal(first, second)
        for first, second in zip((observed_forward, observed_reverse), repeated, strict=True)
    )


def test_scalar_map_mutations_change_only_the_affected_generated_ast() -> None:
    hlo = _hlo()
    baseline = form_shared_contract_multi_map_region(hlo)
    forward_mutation = hlo.replace(
        "%mul.960 = bf16[16,32]{1,0} multiply(%mul.83, %split.11)",
        "%mul.960 = bf16[16,32]{1,0} add(%mul.83, %split.11)",
        1,
    )
    reverse_mutation = hlo.replace(
        "%mul.972 = bf16[16,32]{1,0} multiply(%mul.83, %slice.54)",
        "%mul.972 = bf16[16,32]{1,0} add(%mul.83, %slice.54)",
        1,
    )
    assert forward_mutation != hlo
    assert reverse_mutation != hlo
    forward = form_shared_contract_multi_map_region(forward_mutation)
    reverse = form_shared_contract_multi_map_region(reverse_mutation)

    baseline_digests = tuple(
        tuple(scalar.scalar_program.digest for scalar in output.scalar_outputs) for output in baseline.outputs
    )
    forward_digests = tuple(
        tuple(scalar.scalar_program.digest for scalar in output.scalar_outputs) for output in forward.outputs
    )
    reverse_digests = tuple(
        tuple(scalar.scalar_program.digest for scalar in output.scalar_outputs) for output in reverse.outputs
    )
    assert forward_digests[0] != baseline_digests[0]
    assert forward_digests[1] == baseline_digests[1]
    assert reverse_digests[0] == baseline_digests[0]
    assert reverse_digests[1] != baseline_digests[1]


def test_cuda_typed_ffi_generation_tracks_semantics_not_frontend_names() -> None:
    hlo = _hlo()
    plan = form_shared_contract_multi_map_region(hlo)
    generated = generate_cuda_shared_contract_multi_map_ffi(plan, target=_TARGET)
    renamed = re.sub(r'op_name="[^"]*"', 'op_name="unrelated_label"', hlo)
    renamed_generated = generate_cuda_shared_contract_multi_map_ffi(
        form_shared_contract_multi_map_region(renamed),
        target=_TARGET,
    )

    assert generated.output_count == 2
    assert generated.scalar_semantic_digests == (
        (plan.outputs[0].scalar_outputs[0].scalar_program.digest,),
        tuple(scalar.scalar_program.digest for scalar in plan.outputs[1].scalar_outputs),
    )
    assert generated.semantic_digest == renamed_generated.semantic_digest
    assert generated.source_digest == renamed_generated.source_digest


def test_cuda_typed_ffi_output_count_and_body_come_from_mutated_plan() -> None:
    hlo = _hlo()
    baseline_plan = form_shared_contract_multi_map_region(hlo)
    forward_mutation = hlo.replace(
        "%mul.960 = bf16[16,32]{1,0} multiply(%mul.83, %split.11)",
        "%mul.960 = bf16[16,32]{1,0} add(%mul.83, %split.11)",
        1,
    )
    mutated_plan = form_shared_contract_multi_map_region(forward_mutation)
    baseline = generate_cuda_shared_contract_multi_map_ffi(baseline_plan, target=_TARGET)
    mutated = generate_cuda_shared_contract_multi_map_ffi(mutated_plan, target=_TARGET)

    assert baseline.scalar_semantic_digests[0] != mutated.scalar_semantic_digests[0]
    assert baseline.scalar_semantic_digests[1] == mutated.scalar_semantic_digests[1]
    assert baseline.semantic_digest != mutated.semantic_digest
    assert baseline.source_digest != mutated.source_digest

    reverse_only_plan = replace(baseline_plan, outputs=(baseline_plan.outputs[1],))
    reverse_only = generate_cuda_shared_contract_multi_map_ffi(reverse_only_plan, target=_TARGET)
    assert reverse_only.output_count == 1
    assert reverse_only.scalar_semantic_digests == (baseline.scalar_semantic_digests[1],)

    rng = np.random.default_rng(47)
    operands = (
        _bf16(rng.normal(size=(512, 128)).astype(np.float32)),
        _bf16(rng.normal(size=(128, 64)).astype(np.float32)),
        _bf16(rng.normal(size=(16, 32)).astype(np.float32)),
        np.ones((4, 512, 32), dtype=np.bool_),
        np.ones((4, 512, 64), dtype=np.bool_),
    )
    full_outputs = evaluate_shared_contract_multi_map_plan(baseline_plan, operands)
    reverse_only_outputs = evaluate_shared_contract_multi_map_plan(reverse_only_plan, operands)
    assert len(reverse_only_outputs) == 1
    assert np.array_equal(reverse_only_outputs[0], full_outputs[1])


def test_cuda_typed_ffi_accepts_independent_segmented_output_layouts() -> None:
    plan = form_shared_contract_multi_map_region(_hlo())
    contract_only, auxiliary = plan.outputs
    operands = tuple(
        (
            replace(operand, value=replace(operand.value, shape="pred[4,512,64]{0,2,1}"))
            if operand.role is SharedContractMultiMapOperandRole.CONTRACT_AND_AUXILIARY_VALIDITY
            else operand
        )
        for operand in plan.operands
    )
    layout_mutation = replace(
        plan,
        operands=operands,
        outputs=(
            contract_only,
            replace(auxiliary, value=replace(auxiliary.value, shape="bf16[4,512,64]{1,2,0}")),
        ),
    )

    baseline = generate_cuda_shared_contract_multi_map_ffi(plan, target=_TARGET)
    generated = generate_cuda_shared_contract_multi_map_ffi(layout_mutation, target=_TARGET)

    assert generated.output_count == baseline.output_count
    assert generated.scalar_semantic_digests == baseline.scalar_semantic_digests
    assert generated.semantic_digest != baseline.semantic_digest
    assert generated.source_digest != baseline.source_digest


def test_shared_contract_multi_map_replacement_preserves_both_consumer_sets() -> None:
    hlo = _hlo()
    compilation = compile_shared_contract_multi_map_ffi(hlo, target=_TARGET)
    generated = compilation.generated
    transformed = compilation.transformed_hlo
    audit = compilation.replacement_audit

    assert transformed.count(f'custom_call_target="{_TARGET}"') == 1
    assert "%dot.66 =" not in transformed
    assert "%dot.67 =" in transformed
    assert "%dot.68 =" in transformed
    assert "%dot.69 =" in transformed
    assert "%select.5 = bf16[4,512,32]{2,1,0} get-tuple-element" in transformed
    assert "%select.7 = bf16[4,512,64]{2,1,0} get-tuple-element" in transformed
    assert audit.external_users == (
        ("select.5", ("transpose.169", "dot.7")),
        ("select.7", ("transpose.167", "dot.6")),
    )
    assert audit.copy_count[0] == audit.copy_count[1]
    assert audit.transpose_count[0] == audit.transpose_count[1]
    assert generated.target == _TARGET
    assert generated.output_count == len(audit.outputs)
    parse_hlo_module_text(transformed)


def test_shared_contract_multi_map_recovery_ignores_frontend_metadata() -> None:
    hlo = _hlo()
    renamed = re.sub(r'op_name="[^"]*"', 'op_name="unrelated_label"', hlo)

    baseline = form_shared_contract_multi_map_region(hlo)
    mutated = form_shared_contract_multi_map_region(renamed)

    assert tuple(output.value for output in mutated.outputs) == tuple(output.value for output in baseline.outputs)
    assert tuple(
        scalar.scalar_program.digest for output in mutated.outputs for scalar in output.scalar_outputs
    ) == tuple(scalar.scalar_program.digest for output in baseline.outputs for scalar in output.scalar_outputs)
