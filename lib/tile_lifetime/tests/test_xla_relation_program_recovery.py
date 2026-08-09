# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.cast_scalar_program import (
    CastScalarExpression,
    CastScalarKind,
    evaluate_cast_scalar_program,
)
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import (
    ContractChainRole,
    RoutedForwardCodegenDisposition,
    RoutedForwardFfiOperandRole,
    SegmentedLayoutRelation,
    plan_routed_forward_typed_ffi,
    recover_relation_programs,
)
from tile_lifetime.xla_routed_forward_ffi import (
    evaluate_routed_forward_plan,
    generate_cuda_routed_forward_ffi,
    replace_routed_forward_region_with_custom_call,
)

_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz"
)
_GPU_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/grug_contract_map_gpu_gb200_v0/final/original-gpu-pre-scheduler-hlo.txt.gz"
)


def _frozen_hlo() -> str:
    return gzip.decompress(_ARTIFACT.read_bytes()).decode()


def _frozen_gpu_hlo() -> str:
    return gzip.decompress(_GPU_ARTIFACT.read_bytes()).decode()


def test_gpu_grug_hlo_generates_bf16_routed_forward_executor() -> None:
    plan = plan_routed_forward_typed_ffi(_frozen_gpu_hlo())
    generated = generate_cuda_routed_forward_ffi(plan, target="shuttle.routed_forward.bf16.v1")

    assert plan.disposition is RoutedForwardCodegenDisposition.READY
    assert tuple(contract.output_shape for contract in plan.contracts) == (
        "bf16[512,64]{1,0}",
        "bf16[512,32]{1,0}",
    )
    assert tuple(operand.role for operand in plan.operands) == tuple(RoutedForwardFfiOperandRole)
    assert "CUDA_R_16BF" in generated.source
    assert "ffi::Result<ffi::Buffer<ffi::BF16, 2>>" in generated.source
    assert "__float2bfloat16_rn" in generated.source
    assert "fold_input[edge * kOutputFeatures + feature]" in generated.source
    assert "atomicAdd" not in generated.source


def test_grug_hlo_recovers_generic_routed_forward_and_backward_program() -> None:
    report = recover_relation_programs(_frozen_hlo())

    assert len(report.relation_plans) == 2
    assert {
        (plan.token_count, plan.slots_per_token, plan.edge_count, plan.destination_count)
        for plan in report.relation_plans
    } == {(8, 2, 16, 4)}
    assert all(plan.destination_sort != plan.selection for plan in report.relation_plans)
    assert all(plan.destination_counts != plan.destination_offsets for plan in report.relation_plans)

    assert len(report.contract_chains) == 3
    chains_by_role = {chain.role: chain for chain in report.contract_chains}
    assert set(chains_by_role) == {
        ContractChainRole.FORWARD,
        ContractChainRole.FORWARD_RECOMPUTE,
        ContractChainRole.INPUT_GRADIENT,
    }
    forward = chains_by_role[ContractChainRole.FORWARD]
    assert forward.first.output_shape == "f32[512,64]{1,0}"
    assert forward.second.output_shape == "f32[512,32]{1,0}"
    assert "slice" in forward.map.opcodes
    assert "multiply" in forward.map.opcodes
    assert any(source.startswith("f32") and result.startswith("bf16") for source, result in forward.map.cast_shapes)
    assert len(forward.map.scalar_outputs) == 1
    forward_output = forward.map.scalar_outputs[0]
    assert tuple(
        (value.input_index.row_offset, value.input_index.feature_offset)
        for value in forward_output.scalar_program.inputs
        if value.input_index is not None
    ) == ((0, 0), (0, 32))
    assert _count_kind(forward_output.scalar_program.expression, CastScalarKind.CONVERT) == 30
    assert "__float2bfloat16_rn" in forward_output.generated_cuda.source
    assert "MoE" not in forward_output.generated_cuda.source
    recompute = chains_by_role[ContractChainRole.FORWARD_RECOMPUTE]
    assert len(recompute.map.scalar_outputs) == 1
    assert recompute.map.scalar_outputs[0].scalar_program.digest == forward_output.scalar_program.digest

    input_gradient = chains_by_role[ContractChainRole.INPUT_GRADIENT]
    assert "concatenate" in input_gradient.map.opcodes
    assert "multiply" in input_gradient.map.opcodes
    assert input_gradient.first.output_shape == input_gradient.second.output_shape == "f32[512,32]{1,0}"
    assert tuple((output.feature_offset, output.feature_extent) for output in input_gradient.map.scalar_outputs) == (
        (0, 32),
        (32, 32),
    )
    assert tuple(
        tuple(
            (value.input_name, value.input_index.feature_offset)
            for value in output.scalar_program.inputs
            if value.input_index is not None
        )
        for output in input_gradient.map.scalar_outputs
    ) == (
        (("input0_r0_f0", 0), ("input1_r0_f0", 0), ("input1_r0_f32", 32)),
        (("input0_r0_f0", 0), ("input1_r0_f0", 0)),
    )
    assert tuple(
        _count_kind(output.scalar_program.expression, CastScalarKind.CONVERT)
        for output in input_gradient.map.scalar_outputs
    ) == (86, 30)
    assert all(
        output.scalar_program.expression.kind is CastScalarKind.CONVERT
        and output.scalar_program.expression.operands[0].kind is CastScalarKind.CONVERT
        for output in input_gradient.map.scalar_outputs
    )
    assert all("__float2bfloat16_rn" in output.generated_cuda.source for output in input_gradient.map.scalar_outputs)

    assert len(report.folds) == 2
    assert {fold.reducer for fold in report.folds} == {"add"}
    assert {fold.output_shape for fold in report.folds} == {"f32[8,32]{1,0}"}
    assert all(fold.reducer_opcodes == ("add", "convert", "convert") for fold in report.folds)
    assert any("multiply" in fold.contribution_opcodes for fold in report.folds)
    weighted_fold = next(fold for fold in report.folds if "multiply" in fold.contribution_opcodes)
    plain_fold = next(fold for fold in report.folds if "multiply" not in fold.contribution_opcodes)
    assert len(weighted_fold.contribution_inputs) == 2
    assert len(plain_fold.contribution_inputs) == 1
    assert _count_kind(weighted_fold.contribution_scalar_program.expression, CastScalarKind.MULTIPLY) == 1
    assert _count_kind(plain_fold.contribution_scalar_program.expression, CastScalarKind.MULTIPLY) == 0
    assert all(_count_kind(fold.reducer_scalar_program.expression, CastScalarKind.ADD) == 1 for fold in report.folds)
    assert all(_count_kind(fold.reducer_scalar_program.expression, CastScalarKind.CONVERT) == 2 for fold in report.folds)
    assert all("__float2bfloat16_rn" in fold.generated_reducer_cuda.source for fold in report.folds)
    assert all("MoE" not in fold.generated_contribution_cuda.source for fold in report.folds)

    assert {gradient.contract.output_shape for gradient in report.weight_gradients} == {
        "f32[4,32,64]{2,1,0}",
        "f32[4,32,32]{2,1,0}",
    }
    assert all(gradient.external_collectives for gradient in report.weight_gradients)
    assert report.external_collectives


def test_grug_hlo_relation_recovery_ignores_frontend_metadata() -> None:
    hlo = _frozen_hlo()
    renamed_metadata = re.sub(r'op_name="[^"]*"', 'op_name="unrelated_diagnostic_label"', hlo)

    baseline = recover_relation_programs(hlo).to_dict()
    renamed = recover_relation_programs(renamed_metadata).to_dict()

    assert renamed == baseline


def test_grug_hlo_forward_map_ast_preserves_source_bf16_boundaries() -> None:
    forward = next(
        chain
        for chain in recover_relation_programs(_frozen_hlo()).contract_chains
        if chain.role is ContractChainRole.FORWARD
    )
    assert len(forward.map.scalar_outputs) == 1
    program = forward.map.scalar_outputs[0].scalar_program
    left = 0.78125
    right = -1.3125

    def bf16(value):
        return jnp.asarray(value, dtype=jnp.bfloat16).astype(jnp.float32)

    left_source = bf16(bf16(bf16(jnp.float32(left))))
    right_source = bf16(bf16(bf16(jnp.float32(right))))
    denominator = bf16(jnp.exp(bf16(-left_source)))
    denominator = bf16(denominator + jnp.float32(1.0))
    sigmoid = bf16(jnp.float32(1.0) / denominator)
    activated = bf16(left_source * sigmoid)
    expected = bf16(activated * right_source)

    observed = evaluate_cast_scalar_program(
        program,
        {"input_r0_f0": left, "input_r0_f32": right},
    )

    assert float(observed) == pytest.approx(float(expected), abs=0.0)


def test_grug_hlo_forward_map_mutation_regenerates_scalar_cuda() -> None:
    baseline = next(
        chain
        for chain in recover_relation_programs(_frozen_hlo()).contract_chains
        if chain.role is ContractChainRole.FORWARD
    )
    mutated_hlo = _frozen_hlo().replace(
        "multiply(%convert.3758, %convert.3756)",
        "add(%convert.3758, %convert.3756)",
        1,
    )
    mutated = next(
        chain
        for chain in recover_relation_programs(mutated_hlo).contract_chains
        if chain.role is ContractChainRole.FORWARD
    )
    baseline_output = baseline.map.scalar_outputs[0]
    mutated_output = mutated.map.scalar_outputs[0]

    assert mutated_output.scalar_program.digest != baseline_output.scalar_program.digest
    assert mutated_output.generated_cuda.source_digest != baseline_output.generated_cuda.source_digest
    assert _count_kind(mutated_output.scalar_program.expression, CastScalarKind.ADD) == (
        _count_kind(baseline_output.scalar_program.expression, CastScalarKind.ADD) + 1
    )
    assert _count_kind(mutated_output.scalar_program.expression, CastScalarKind.MULTIPLY) == (
        _count_kind(baseline_output.scalar_program.expression, CastScalarKind.MULTIPLY) - 1
    )


def test_grug_hlo_input_adjoint_mutation_regenerates_only_affected_scalar_output() -> None:
    baseline = next(
        chain
        for chain in recover_relation_programs(_frozen_hlo()).contract_chains
        if chain.role is ContractChainRole.INPUT_GRADIENT
    )
    mutated_hlo = _frozen_hlo().replace(
        "multiply(%convert.3441, %convert.3459)",
        "add(%convert.3441, %convert.3459)",
        1,
    )
    mutated = next(
        chain
        for chain in recover_relation_programs(mutated_hlo).contract_chains
        if chain.role is ContractChainRole.INPUT_GRADIENT
    )

    assert mutated.map.scalar_outputs[0].scalar_program.digest == baseline.map.scalar_outputs[0].scalar_program.digest
    assert mutated.map.scalar_outputs[1].scalar_program.digest != baseline.map.scalar_outputs[1].scalar_program.digest
    assert mutated.map.scalar_outputs[1].generated_cuda.source_digest != (
        baseline.map.scalar_outputs[1].generated_cuda.source_digest
    )


def test_grug_hlo_fold_contribution_mutation_regenerates_only_contribution_body() -> None:
    baseline = next(
        fold for fold in recover_relation_programs(_frozen_hlo()).folds if "multiply" in fold.contribution_opcodes
    )
    mutated_hlo = _frozen_hlo().replace(
        "multiply(%convert.3742, %broadcast.964)",
        "add(%convert.3742, %broadcast.964)",
        1,
    )
    mutated = next(
        fold for fold in recover_relation_programs(mutated_hlo).folds if fold.source_contract == baseline.source_contract
    )

    assert mutated.contribution_scalar_program.digest != baseline.contribution_scalar_program.digest
    assert mutated.generated_contribution_cuda.source_digest != baseline.generated_contribution_cuda.source_digest
    assert mutated.reducer_scalar_program.digest == baseline.reducer_scalar_program.digest
    assert mutated.generated_reducer_cuda.source_digest == baseline.generated_reducer_cuda.source_digest
    assert _count_kind(mutated.contribution_scalar_program.expression, CastScalarKind.ADD) == (
        _count_kind(baseline.contribution_scalar_program.expression, CastScalarKind.ADD) + 1
    )
    assert _count_kind(mutated.contribution_scalar_program.expression, CastScalarKind.MULTIPLY) == 0


def test_grug_hlo_forms_ready_convex_routed_forward_region_with_verified_segmented_layout() -> None:
    plan = plan_routed_forward_typed_ffi(_frozen_hlo())
    region = plan.region

    assert region.convex
    assert region.topologically_insertable
    assert not region.requires_auxiliary_output_split
    assert len(region.boundary.internal_instructions) == 6
    assert len(region.boundary.inputs) == 7
    assert tuple(value.shape for value in region.boundary.outputs) == ("f32[8,32]{1,0}",)
    assert tuple(contract.backend for contract in plan.contracts) == ("cublas", "cublas")
    assert all(contract.dimensions.lhs_contracting == (1,) for contract in plan.contracts)
    assert all(contract.dimensions.rhs_contracting == (0,) for contract in plan.contracts)
    assert all(contract.dimensions.lhs_output == (0,) for contract in plan.contracts)
    assert all(contract.dimensions.rhs_output == (1,) for contract in plan.contracts)
    assert plan.api_version == 1
    assert plan.disposition is RoutedForwardCodegenDisposition.READY
    assert tuple(operand.role for operand in plan.operands) == tuple(RoutedForwardFfiOperandRole)
    assert tuple(operand.value.instruction for operand in plan.operands) == (
        "copy_bitcast_fusion.13",
        "copy.1575",
        "select_bitcast_fusion.2",
        "gather_convert_fusion.1",
        "compare_and_fusion.1",
        "copy_bitcast_fusion.24",
        "copy_bitcast_fusion.23",
    )
    assert plan.map_stage.logical_row_extent == 16
    assert plan.map_stage.logical_feature_extent == 32
    assert plan.map_stage.physical_output_shape == "f32[512,128]{1,0}"
    assert plan.map_stage.has_segmented_layout
    assert plan.missing_segmented_layout is None
    assert plan.segmented_layout is not None
    layout = plan.segmented_layout
    assert layout.verified_relations == (
        SegmentedLayoutRelation.EDGE_ROW_TO_PADDED_ROW,
        SegmentedLayoutRelation.SEGMENT_TO_FEATURE_PANEL,
        SegmentedLayoutRelation.VALIDITY_AND_FILL,
        SegmentedLayoutRelation.SOURCE_FOLD_INVERSE,
    )
    assert layout.index_map.padded_row_extent == 512
    assert layout.index_map.segment_count == 4
    assert layout.index_map.feature_stride == 4
    assert layout.index_map.segment_stride == 1
    assert layout.fill_value == 0.0
    assert layout.runtime_index_inputs == (layout.segment_ends, layout.inverse.stable_permutation)
    segment_ends = (3, 7, 12, 16)
    first = layout.index_map.physical_index(
        edge_row=0,
        feature=5,
        segment=0,
        segment_ends=segment_ends,
    )
    next_segment = layout.index_map.physical_index(
        edge_row=3,
        feature=5,
        segment=1,
        segment_ends=segment_ends,
    )
    wrong_segment = layout.index_map.physical_index(
        edge_row=3,
        feature=5,
        segment=0,
        segment_ends=segment_ends,
    )
    assert (first.physical_row, first.physical_k, first.valid) == (0, 20, True)
    assert (next_segment.physical_row, next_segment.physical_k, next_segment.valid) == (3, 21, True)
    assert (wrong_segment.physical_row, wrong_segment.physical_k, wrong_segment.valid) == (3, 20, False)
    permutation = (5, 0, 3, 2, 1, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14)
    assert layout.inverse.source_coordinate(0, permutation) == (2, 1)
    assert layout.inverse.source_coordinate(1, permutation) == (0, 0)


def test_grug_hlo_routed_forward_map_mutation_changes_generic_codegen_plan() -> None:
    baseline = plan_routed_forward_typed_ffi(_frozen_hlo())
    mutated = plan_routed_forward_typed_ffi(
        _frozen_hlo().replace(
            "multiply(%convert.3758, %convert.3756)",
            "add(%convert.3758, %convert.3756)",
            1,
        )
    )

    assert mutated.region.boundary == baseline.region.boundary
    assert mutated.contracts == baseline.contracts
    assert mutated.map_stage.scalar_outputs[0].scalar_program.digest != (
        baseline.map_stage.scalar_outputs[0].scalar_program.digest
    )
    assert mutated.fold_stage.contribution_program.digest == baseline.fold_stage.contribution_program.digest
    assert mutated.fold_stage.reducer_program.digest == baseline.fold_stage.reducer_program.digest
    assert mutated.disposition is RoutedForwardCodegenDisposition.READY


def test_grug_hlo_routed_forward_fold_mutation_changes_generic_codegen_plan() -> None:
    baseline = plan_routed_forward_typed_ffi(_frozen_hlo())
    mutated = plan_routed_forward_typed_ffi(
        _frozen_hlo().replace(
            "multiply(%convert.3742, %broadcast.964)",
            "add(%convert.3742, %broadcast.964)",
            1,
        )
    )

    assert mutated.region.boundary == baseline.region.boundary
    assert mutated.contracts == baseline.contracts
    assert mutated.map_stage.scalar_outputs[0].scalar_program.digest == (
        baseline.map_stage.scalar_outputs[0].scalar_program.digest
    )
    assert mutated.fold_stage.contribution_program.digest != baseline.fold_stage.contribution_program.digest
    assert mutated.fold_stage.generated_contribution_cuda.source_digest != (
        baseline.fold_stage.generated_contribution_cuda.source_digest
    )
    assert mutated.fold_stage.reducer_program.digest == baseline.fold_stage.reducer_program.digest
    assert mutated.disposition is RoutedForwardCodegenDisposition.READY


def test_grug_hlo_segmented_layout_shape_mutation_changes_physical_row_map() -> None:
    baseline = plan_routed_forward_typed_ffi(_frozen_hlo())
    mutated_hlo = _frozen_hlo().replace("512", "256").replace("496", "240")
    mutated = plan_routed_forward_typed_ffi(mutated_hlo)

    assert baseline.disposition is RoutedForwardCodegenDisposition.READY
    assert mutated.disposition is RoutedForwardCodegenDisposition.READY
    assert baseline.segmented_layout is not None
    assert mutated.segmented_layout is not None
    assert baseline.segmented_layout.index_map.padded_row_extent == 512
    assert mutated.segmented_layout.index_map.padded_row_extent == 256
    assert mutated.segmented_layout.physical_shape == "f32[256,128]{1,0}"
    assert (
        mutated.segmented_layout.index_map.physical_index(
            edge_row=15,
            feature=31,
            segment=3,
            segment_ends=(3, 7, 12, 16),
        ).physical_k
        == 127
    )


def test_grug_hlo_segmented_layout_rejects_mismatched_weight_flattening() -> None:
    mutated_hlo = _frozen_hlo().replace(
        "transpose(%convert.3419), dimensions={1,0,2}",
        "transpose(%convert.3419), dimensions={0,1,2}",
        1,
    )
    plan = plan_routed_forward_typed_ffi(mutated_hlo)

    assert plan.disposition is RoutedForwardCodegenDisposition.MISSING_SEGMENTED_LAYOUT
    assert plan.segmented_layout is None
    assert plan.missing_segmented_layout is not None
    assert plan.missing_segmented_layout.required_relations == (SegmentedLayoutRelation.SEGMENT_TO_FEATURE_PANEL,)
    assert len(plan.missing_segmented_layout.reasons) == 1


def test_routed_forward_reference_executes_generated_map_and_fold_semantics() -> None:
    plan = plan_routed_forward_typed_ffi(_frozen_hlo())
    operands = _routed_operands(plan)

    observed = evaluate_routed_forward_plan(plan, operands)
    expected = _direct_routed_forward(operands)

    np.testing.assert_array_equal(observed, expected)


def test_routed_forward_cuda_generation_tracks_map_and_fold_mutations() -> None:
    baseline = plan_routed_forward_typed_ffi(_frozen_hlo())
    map_mutation = plan_routed_forward_typed_ffi(
        _frozen_hlo().replace(
            "multiply(%convert.3758, %convert.3756)",
            "add(%convert.3758, %convert.3756)",
            1,
        )
    )
    fold_mutation = plan_routed_forward_typed_ffi(
        _frozen_hlo().replace(
            "multiply(%convert.3742, %broadcast.964)",
            "add(%convert.3742, %broadcast.964)",
            1,
        )
    )
    operands = _routed_operands(baseline)
    generated = tuple(
        generate_cuda_routed_forward_ffi(plan, target="shuttle.routed_forward.v1")
        for plan in (baseline, map_mutation, fold_mutation)
    )
    outputs = tuple(evaluate_routed_forward_plan(plan, operands) for plan in (baseline, map_mutation, fold_mutation))

    assert len({value.semantic_digest for value in generated}) == 3
    assert len({value.source_digest for value in generated}) == 3
    assert generated[0].source.count("cublasGemmEx(") == 1
    assert "atomicAdd" not in generated[0].source
    assert "for (int edge = 0; edge < kLogicalEdges; ++edge)" in generated[0].source
    assert not np.array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[0], outputs[2])


def test_routed_forward_replacement_preserves_runtime_operand_order() -> None:
    plan = plan_routed_forward_typed_ffi(_frozen_hlo())
    transformed = replace_routed_forward_region_with_custom_call(
        _frozen_hlo(),
        plan,
        target="shuttle.routed_forward.v1",
    )
    module = parse_hlo_module_text(transformed)
    entry = module.computation(module.entry)
    call = next(
        instruction
        for instruction in entry.instructions
        if 'custom_call_target="shuttle.routed_forward.v1"' in instruction.attributes
    )

    assert call.operands == tuple(operand.value.instruction for operand in plan.operands)
    assert entry.root.name == parse_hlo_module_text(_frozen_hlo()).computation(module.entry).root.name
    assert transformed.count("shuttle.routed_forward.v1") == 1


def _routed_operands(plan) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(11)
    values = {
        RoutedForwardFfiOperandRole.SECOND_CONTRACT_RHS: rng.normal(0.0, 0.05, (128, 32)).astype(np.float32),
        RoutedForwardFfiOperandRole.FOLD_INITIAL: rng.normal(0.0, 0.05, (8, 32)).astype(np.float32),
        RoutedForwardFfiOperandRole.FOLD_INDICES: (np.arange(16, dtype=np.int32) % 8).reshape(16, 1),
        RoutedForwardFfiOperandRole.FOLD_CONTRIBUTION_INPUT: np.asarray(
            rng.normal(0.0, 0.05, (16, 1)), dtype=jnp.bfloat16
        ),
        RoutedForwardFfiOperandRole.SEGMENT_VALIDITY: np.zeros((4, 512, 32), dtype=np.bool_),
        RoutedForwardFfiOperandRole.FIRST_CONTRACT_LHS: rng.normal(0.0, 0.05, (512, 128)).astype(np.float32),
        RoutedForwardFfiOperandRole.FIRST_CONTRACT_RHS: rng.normal(0.0, 0.05, (128, 64)).astype(np.float32),
    }
    for edge in range(16):
        values[RoutedForwardFfiOperandRole.SEGMENT_VALIDITY][edge // 4, edge, :] = True
    return tuple(values[operand.role] for operand in plan.operands)


def _direct_routed_forward(operands: tuple[np.ndarray, ...]) -> np.ndarray:
    second_weight, initial, source_indices, route_weights, validity, activation, first_weight = operands
    projection = (activation @ first_weight).astype(np.float32)
    mapped = np.zeros((512, 128), dtype=np.float32)
    for edge in range(16):
        for feature in range(32):
            left = _bf16(_bf16(_bf16(projection[edge, feature])))
            right = _bf16(_bf16(_bf16(projection[edge, feature + 32])))
            denominator = _bf16(np.exp(_bf16(-left)))
            denominator = _bf16(denominator + np.float32(1.0))
            sigmoid = _bf16(np.float32(1.0) / denominator)
            activated = _bf16(left * sigmoid)
            value = _bf16(activated * right)
            for segment in range(4):
                if validity[segment, edge, feature]:
                    mapped[edge, feature * 4 + segment] = value
    routed = (mapped @ second_weight).astype(np.float32)
    output = initial.copy()
    for edge in range(16):
        source = int(source_indices[edge, 0])
        for feature in range(32):
            contribution = _bf16(_bf16(_bf16(routed[edge, feature])) * np.float32(route_weights[edge, 0]))
            output[source, feature] = _bf16(output[source, feature] + contribution)
    return output.astype(np.float32)


def _bf16(value):
    return np.asarray(jnp.asarray(value, dtype=jnp.bfloat16), dtype=np.float32)


def _count_kind(expression: CastScalarExpression, kind: CastScalarKind) -> int:
    return int(expression.kind is kind) + sum(_count_kind(operand, kind) for operand in expression.operands)
