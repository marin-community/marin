# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.xla_contract_relation_fold_ffi import (
    audit_contract_relation_fold_replacement,
    evaluate_contract_relation_fold_plan,
    generate_cuda_contract_relation_fold_ffi,
    replace_contract_relation_fold_with_custom_call,
)
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_rank_two_contract_ffi import (
    audit_rank_two_contract_replacement,
    evaluate_rank_two_contract_plan,
    generate_cuda_rank_two_contract_ffi,
    narrow_rank_two_contract_to_consumer_row_domain,
    plan_rank_two_bf16_contract_typed_ffi,
    replace_rank_two_contract_with_custom_call,
)
from tile_lifetime.xla_relation_program_recovery import ContractDimensionMap, RoutedForwardContractStage
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
    assert plan.payload_contract.instruction == "slice.35"
    assert plan.payload_contract.source_instruction == "dot.69"
    assert plan.payload_contract.lhs_row_start == 0
    assert plan.payload_contract.lhs.instruction == "reshape.363"
    assert plan.payload_contract.rhs.instruction == "reshape.364"
    assert plan.payload_contract.output_shape == "bf16[16,32]{1,0}"
    assert plan.edge_fold.instruction == "scatter-add.41"
    assert plan.edge_fold.payload_logical_shape == "bf16[16,32]{1,0}"
    assert plan.edge_fold.payload_wrappers == ("slice.35",)
    assert plan.edge_fold.edge_cotangent.instruction == "reshape.735"
    assert plan.edge_fold.internal_instructions == (
        "dot.69",
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
    generated_contract = generate_cuda_rank_two_contract_ffi(plan.payload_contract, target=contract_target)
    generated_fold = generate_cuda_relation_edge_fold_ffi(plan.edge_fold, target=fold_target)
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

    assert audit.contract_instruction == "slice.35"
    assert audit.fold_instruction == "scatter-add.41"
    assert audit.fold_operands == (
        "broadcast.102",
        "broadcast_in_dim.427",
        "slice.35",
        "reshape.735",
    )
    assert audit.dead_replaced_instructions == plan.edge_fold.internal_instructions
    assert audit.placement_wrappers == ("reshape.230",)
    assert audit.placement_collective == "psum.51"
    assert 'custom_call_target="shuttle.weighted_relation_reverse.contract.test"' in transformed
    assert 'custom_call_target="shuttle.weighted_relation_reverse.fold.test"' in transformed
    assert "constexpr int kRows = 16;" in generated_contract.source
    assert "constexpr int kRows = 512;" not in generated_contract.source
    assert "atomicAdd(" not in generated_fold.source
    assert "generated_edge_contribution" in generated_fold.source
    assert "generated_inner_fold_update" in generated_fold.source
    assert "generated_outer_fold_update" in generated_fold.source
    parse_hlo_module_text(transformed)


def test_bounded_contract_relation_fold_candidate_erases_payload_materialization_and_launch() -> None:
    hlo = _hlo()
    plan = plan_weighted_relation_reverse_typed_ffi(hlo)
    target = "shuttle.contract_relation_fold.test"
    generated = generate_cuda_contract_relation_fold_ffi(
        plan.payload_contract,
        plan.edge_fold,
        target=target,
    )
    transformed = replace_contract_relation_fold_with_custom_call(
        hlo,
        plan.payload_contract,
        plan.edge_fold,
        target=target,
    )
    audit = audit_contract_relation_fold_replacement(
        hlo,
        transformed,
        plan.payload_contract,
        plan.edge_fold,
        target=target,
    )

    assert generated.cost.contract_fma_count == 16 * 32 * 128
    assert generated.cost.payload_elements == 16 * 32
    assert generated.cost.payload_global_bytes == 0
    assert generated.cost.kernel_launches == 1
    assert generated.cost.threads_per_block == 512
    assert generated.cost.shared_bytes == 16 * 32 * 2 + 16 * 4
    assert transformed.count(f'custom_call_target="{target}"') == 1
    assert audit.call_instruction == "scatter-add.41"
    assert audit.operands == (
        "reshape.363",
        "reshape.364",
        "broadcast.102",
        "broadcast_in_dim.427",
        "reshape.735",
    )
    assert audit.external_users == ("reshape.230",)
    assert "dot.69" in audit.dead_instructions
    assert "slice.35" in audit.dead_instructions
    assert "cublas" not in generated.source
    assert "atomicAdd(" not in generated.source


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
        "dot.69",
        "mul.967",
        "reduce_sum.710",
        "reshape.409",
    )
    assert audit.placement_collective == "psum.51"


def _rank_two_slice_hlo(*, input_rows: int, row_start: int, row_count: int) -> str:
    row_limit = row_start + row_count
    return f"""HloModule row_domain

ENTRY %main (%lhs: bf16[{input_rows},5], %rhs: bf16[5,3]) -> bf16[{row_count},3] {{
  %lhs = bf16[{input_rows},5]{{1,0}} parameter(0)
  %rhs = bf16[5,3]{{1,0}} parameter(1)
  %dot = bf16[{input_rows},3]{{1,0}} dot(%lhs, %rhs), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
  %window = bf16[{row_count},3]{{1,0}} slice(%dot), slice={{[{row_start}:{row_limit}], [0:3]}}
  ROOT %root = bf16[{row_count},3]{{1,0}} copy(%window)
}}
"""


@pytest.mark.parametrize(("input_rows", "row_start", "row_count"), ((7, 0, 4), (9, 2, 3)))
def test_contract_row_domain_projection_uses_generic_slice_relation(
    input_rows: int,
    row_start: int,
    row_count: int,
) -> None:
    hlo = _rank_two_slice_hlo(input_rows=input_rows, row_start=row_start, row_count=row_count)
    stage = RoutedForwardContractStage(
        node="main/dot",
        lhs="main/lhs",
        rhs="main/rhs",
        output_shape=f"bf16[{input_rows},3]{{1,0}}",
        dimensions=ContractDimensionMap(
            lhs_contracting=(1,),
            rhs_contracting=(0,),
            lhs_batch=(),
            rhs_batch=(),
            lhs_output=(0,),
            rhs_output=(1,),
        ),
    )
    full = plan_rank_two_bf16_contract_typed_ffi(hlo, stage)
    narrowed = narrow_rank_two_contract_to_consumer_row_domain(hlo, full, consumer_value="window")
    generated = generate_cuda_rank_two_contract_ffi(narrowed, target="shuttle.row_domain.test")
    transformed = replace_rank_two_contract_with_custom_call(hlo, narrowed, target="shuttle.row_domain.test")
    audit = audit_rank_two_contract_replacement(hlo, transformed, narrowed, target="shuttle.row_domain.test")

    lhs = np.arange(input_rows * 5, dtype=np.float32).reshape(input_rows, 5) / 16
    rhs = np.arange(15, dtype=np.float32).reshape(5, 3) / 8
    observed = evaluate_rank_two_contract_plan(narrowed, _bf16(lhs), _bf16(rhs))
    expected = _bf16(_bf16(lhs)[row_start : row_start + row_count] @ _bf16(rhs))

    assert np.array_equal(observed, expected)
    assert narrowed.source_instruction == "dot"
    assert narrowed.instruction == "window"
    assert narrowed.output_shape == f"bf16[{row_count},3]{{1,0}}"
    assert narrowed.lhs_row_start == row_start
    assert narrowed.numerical_contract == full.numerical_contract
    assert audit.call_instruction == "window"
    assert audit.output_shape == f"bf16[{row_count},3]{{1,0}}"
    assert transformed.count('custom_call_target="shuttle.row_domain.test"') == 1
    assert f"constexpr int kRows = {row_count};" in generated.source
    assert f"+ {row_start} * kReduction" in generated.source


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
        evaluate_contract_relation_fold_plan(
            plan.payload_contract,
            plan.edge_fold,
            lhs,
            rhs,
            initial,
            source_indices,
            edge_cotangent,
        ),
        expected,
    )
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

    baseline_fused = generate_cuda_contract_relation_fold_ffi(
        baseline_plan.payload_contract,
        baseline_plan.edge_fold,
        target="shuttle.contract_relation_fold.mutation",
    )
    regenerated_fused = generate_cuda_contract_relation_fold_ffi(
        mutated_plan.payload_contract,
        mutated_plan.edge_fold,
        target="shuttle.contract_relation_fold.mutation",
    )
    assert baseline_fused.cost == regenerated_fused.cost
    assert baseline_fused.semantic_digest != regenerated_fused.semantic_digest
    assert baseline_fused.source_digest != regenerated_fused.source_digest
    assert "__fmul_rn" in baseline_fused.source
    assert "__fadd_rn" in regenerated_fused.source


def test_bounded_contract_relation_fold_rejects_non_source_ordered_fold() -> None:
    plan = plan_weighted_relation_reverse_typed_ffi(_hlo())
    unsafe_fold = replace(
        plan.edge_fold,
        numerical_contract=replace(plan.edge_fold.numerical_contract, deterministic=False),
    )

    with pytest.raises(ValueError, match="deterministic atomic-free Fold ownership"):
        generate_cuda_contract_relation_fold_ffi(
            plan.payload_contract,
            unsafe_fold,
            target="shuttle.contract_relation_fold.unsafe",
        )
