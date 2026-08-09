# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_routed_shared_map_training_ffi import (
    RoutedSharedMapTrainingFfiTargets,
    plan_routed_shared_map_training_typed_ffi,
    replace_routed_shared_map_training_regions_with_custom_calls,
)
from tile_lifetime.xla_weighted_relation_reverse_ffi import (
    RelationPayloadPolicy,
    audit_weighted_relation_reverse_replacement,
    evaluate_weighted_relation_reverse_plan,
    generate_cuda_relation_edge_fold_ffi,
    plan_weighted_relation_reverse_typed_ffi,
    replace_weighted_relation_reverse_with_custom_calls,
)

_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_forward_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
)


def _hlo() -> str:
    return gzip.decompress(_ARTIFACT.read_bytes()).decode()


def _bf16(value: np.ndarray) -> np.ndarray:
    return np.asarray(jnp.asarray(value, dtype=jnp.bfloat16).astype(jnp.float32))


def test_natural_hlo_recovers_generic_weighted_relation_reverse() -> None:
    plan = plan_weighted_relation_reverse_typed_ffi(_hlo())

    assert plan.relation_plan.stable_permutation.endswith("/sort.9")
    assert plan.relation_plan.edge_count == 16
    assert plan.payload_policy is RelationPayloadPolicy.RECOMPUTE_CONTRACT
    assert plan.legal_payload_policies == (RelationPayloadPolicy.RECOMPUTE_CONTRACT,)
    assert plan.payload_contract.instruction == "dot.69"
    assert plan.payload_contract.lhs.instruction == "reshape.363"
    assert plan.payload_contract.rhs.instruction == "reshape.364"
    assert plan.payload_contract.output_shape == "bf16[512,32]{1,0}"
    assert plan.edge_fold.instruction == "scatter-add.41"
    assert plan.edge_fold.payload_logical_shape == "bf16[16,32]{1,0}"
    assert plan.edge_fold.payload_wrappers == ("slice.35",)
    assert plan.edge_fold.edge_cotangent.instruction == "reshape.735"
    assert plan.edge_fold.internal_instructions == (
        "slice.35",
        "mul.967",
        "reduce_sum.710",
        "reshape.409",
    )
    assert not plan.edge_fold.numerical_contract.atomic_accumulation


def test_generated_calls_own_contract_map_and_nested_folds_before_collective() -> None:
    hlo = _hlo()
    plan = plan_weighted_relation_reverse_typed_ffi(hlo)
    contract_target = "shuttle.weighted_relation_reverse.contract.test"
    fold_target = "shuttle.weighted_relation_reverse.fold.test"
    generated = generate_cuda_relation_edge_fold_ffi(plan.edge_fold, target=fold_target)
    transformed = replace_weighted_relation_reverse_with_custom_calls(
        hlo,
        plan,
        contract_target=contract_target,
        fold_target=fold_target,
    )
    audit = audit_weighted_relation_reverse_replacement(
        hlo,
        transformed,
        plan,
        contract_target=contract_target,
        fold_target=fold_target,
    )

    assert audit.contract_instruction == "dot.69"
    assert audit.fold_instruction == "scatter-add.41"
    assert audit.fold_operands == (
        "broadcast.102",
        "broadcast_in_dim.427",
        "dot.69",
        "reshape.735",
    )
    assert audit.dead_replaced_instructions == plan.edge_fold.internal_instructions
    assert audit.placement_wrappers == ("reshape.230",)
    assert audit.placement_collective == "psum.51"
    assert 'custom_call_target="shuttle.weighted_relation_reverse.contract.test"' in transformed
    assert 'custom_call_target="shuttle.weighted_relation_reverse.fold.test"' in transformed
    assert "atomicAdd(" not in generated.source
    assert "generated_edge_contribution" in generated.source
    assert "generated_inner_fold_update" in generated.source
    assert "generated_outer_fold_update" in generated.source
    parse_hlo_module_text(transformed)


def test_weighted_reverse_composes_after_existing_generated_routed_regions() -> None:
    hlo = _hlo()
    routed_targets = RoutedSharedMapTrainingFfiTargets(
        forward="shuttle.composed.forward",
        input_contracts=("shuttle.composed.input.0", "shuttle.composed.input.1"),
        shared_contract_multi_map="shuttle.composed.shared_maps",
        source_fold="shuttle.composed.source_fold",
        weight_gradients=("shuttle.composed.weight.0", "shuttle.composed.weight.1"),
    )
    routed_plan = plan_routed_shared_map_training_typed_ffi(hlo)
    routed_hlo = replace_routed_shared_map_training_regions_with_custom_calls(
        hlo,
        routed_plan,
        targets=routed_targets,
    )
    reverse_plan = plan_weighted_relation_reverse_typed_ffi(routed_hlo)
    transformed = replace_weighted_relation_reverse_with_custom_calls(
        routed_hlo,
        reverse_plan,
        contract_target="shuttle.composed.edge_contract",
        fold_target="shuttle.composed.edge_fold",
    )
    audit = audit_weighted_relation_reverse_replacement(
        routed_hlo,
        transformed,
        reverse_plan,
        contract_target="shuttle.composed.edge_contract",
        fold_target="shuttle.composed.edge_fold",
    )

    assert transformed.count('custom_call_target="shuttle.composed.') == 9
    assert reverse_plan.relation_plan.stable_permutation.endswith("/sort.9")
    assert audit.dead_replaced_instructions == (
        "slice.35",
        "mul.967",
        "reduce_sum.710",
        "reshape.409",
    )
    assert audit.placement_collective == "psum.51"


def test_weighted_relation_reverse_cpu_semantics_match_independent_reference() -> None:
    plan = plan_weighted_relation_reverse_typed_ffi(_hlo())
    rng = np.random.default_rng(73)
    lhs = _bf16(rng.normal(scale=0.15, size=(512, 128)).astype(np.float32))
    rhs = _bf16(rng.normal(scale=0.15, size=(128, 32)).astype(np.float32))
    edge_cotangent = _bf16(rng.normal(scale=0.15, size=(16, 32)).astype(np.float32))
    source_indices = np.asarray([3, 0, 3, 7, 1, 3, 0, 2, 7, 7, 4, 6, 3, 5, 0, 1], dtype=np.int32)[:, None]
    initial = _bf16(rng.normal(scale=0.05, size=(16,)).astype(np.float32))

    observed = evaluate_weighted_relation_reverse_plan(
        plan,
        lhs,
        rhs,
        initial,
        source_indices,
        edge_cotangent,
    )
    payload = _bf16(lhs.astype(np.float32) @ rhs.astype(np.float32))
    expected = initial.copy()
    for source in range(16):
        outer = expected[source]
        for edge in range(16):
            if source_indices[edge, 0] != source:
                continue
            inner = np.float32(0.0)
            for feature in range(32):
                product = _bf16(np.asarray([payload[edge, feature] * edge_cotangent[edge, feature]]))[0]
                inner = _bf16(np.asarray([inner + product]))[0]
            outer = _bf16(np.asarray([outer + inner]))[0]
        expected[source] = outer

    assert np.array_equal(observed, expected)
    assert np.array_equal(
        observed,
        evaluate_weighted_relation_reverse_plan(
            plan,
            lhs,
            rhs,
            initial,
            source_indices,
            edge_cotangent,
        ),
    )


def test_scalar_map_mutation_regenerates_same_generic_nested_fold() -> None:
    hlo = _hlo()
    original = "%mul.967 = bf16[16,32]{1,0} multiply(%slice.35, %reshape.735)"
    mutated = "%mul.967 = bf16[16,32]{1,0} add(%slice.35, %reshape.735)"
    assert hlo.count(original) == 1
    baseline_plan = plan_weighted_relation_reverse_typed_ffi(hlo)
    mutated_plan = plan_weighted_relation_reverse_typed_ffi(hlo.replace(original, mutated, 1))
    baseline = generate_cuda_relation_edge_fold_ffi(
        baseline_plan.edge_fold,
        target="shuttle.weighted_relation_reverse.mutation",
    )
    regenerated = generate_cuda_relation_edge_fold_ffi(
        mutated_plan.edge_fold,
        target="shuttle.weighted_relation_reverse.mutation",
    )

    assert baseline_plan.payload_contract == mutated_plan.payload_contract
    assert baseline_plan.edge_fold.contribution_program.digest != (mutated_plan.edge_fold.contribution_program.digest)
    assert baseline.semantic_digest != regenerated.semantic_digest
    assert baseline.source_digest != regenerated.source_digest
    assert "__fmul_rn" in baseline.source
    assert "__fadd_rn" in regenerated.source
