# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import re
from pathlib import Path

import jax.numpy as jnp
import ml_dtypes
import numpy as np

from tile_lifetime.cast_scalar_program import (
    CastScalarDType,
    CastScalarExpression,
    CastScalarKind,
    CastScalarProgram,
    ScalarIndexRelation,
)
from tile_lifetime.cuda_partitioned_gemm_codegen import (
    audit_cuda_partitioned_gemm_source,
    generate_cuda_partitioned_gemm_ffi,
)
from tile_lifetime.jax_partitioned_gemm_ffi import evaluate_partitioned_gemm_jax
from tile_lifetime.partitioned_gemm_program import (
    AccumulatorPartition,
    AuxiliaryPartitionFold,
    PartitionedGemmProgram,
    PartitionFoldReassociation,
    PassthroughPartitionFinalization,
)
from tile_lifetime.partitioned_gemm_reference import evaluate_partitioned_gemm_reference
from tile_lifetime.xla_fold_consumer_preparation import plan_fold_consumer_preparations
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    plan_generated_low_rank_contract_map_training,
    replace_generated_low_rank_contract_map_training,
)
from tile_lifetime.xla_partitioned_contract_fold import (
    audit_attached_partition_folds,
    plan_attached_partition_folds,
    replace_attached_partition_folds,
)

_TARGET_PREFIX = "shuttle.generic.partitioned_contract_fold.test"


def _post_gated_product_grug_hlo() -> str:
    artifact = (
        Path(__file__).parents[1]
        / "benchmarks/artifacts/xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0/"
        "transformed-gpu-pre-scheduler-hlo.txt.gz"
    )
    hlo = gzip.decompress(artifact.read_bytes()).decode()
    gated = plan_generated_low_rank_contract_map_training(
        hlo,
        forward_target_prefix="shuttle.generic.low_rank_contract_map.generated.forward.test",
        reverse_target_prefix="shuttle.generic.low_rank_contract_map.generated.reverse.test",
    )
    return replace_generated_low_rank_contract_map_training(hlo, gated)


def _negated_self_product_mutation(hlo: str) -> str:
    pattern = re.compile(r"^(?P<indent>\s*)%square\.12 = (?P<shape>f32\[[^ ]+) multiply\(%convert\.1, %convert\.1\),")
    lines: list[str] = []
    replaced = False
    for line in hlo.splitlines(keepends=True):
        match = pattern.match(line)
        if match is not None:
            indent = match.group("indent")
            shape = match.group("shape")
            lines.append(f"{indent}%partition_fold_mutation = {shape} negate(%convert.1)\n")
            line = line.replace("multiply(%convert.1, %convert.1)", "multiply(%partition_fold_mutation, %convert.1)")
            replaced = True
        lines.append(line)
    assert replaced
    return "".join(lines)


def _ambiguous_second_partition_fold(hlo: str) -> str:
    marker = "  %reduce_sum.632 = f32[2,4,1]{2,1,0} reduce(%square.13, %constant.266),"
    lines: list[str] = []
    replaced = False
    for line in hlo.splitlines(keepends=True):
        lines.append(line)
        if line.startswith(marker):
            lines.append(
                "  %partition_fold_ambiguous = f32[2,4,1]{2,1,0} "
                "reduce(%square.13, %constant.266), dimensions={3}, to_apply=%region_0.1\n"
            )
            replaced = True
    assert replaced
    return "".join(lines)


def _epsilon_mutation(hlo: str) -> str:
    original = "%constant.246 = f32[] constant(1e-06)"
    assert original in hlo
    return hlo.replace(original, "%constant.246 = f32[] constant(2e-06)")


def _self_product_program() -> CastScalarProgram:
    source = CastScalarExpression(
        CastScalarKind.INPUT,
        CastScalarDType.BF16,
        input_name="value",
        input_index=ScalarIndexRelation(0, 0),
    )
    converted = CastScalarExpression(CastScalarKind.CONVERT, CastScalarDType.F32, operands=(source,))
    return CastScalarProgram(
        CastScalarExpression(CastScalarKind.MULTIPLY, CastScalarDType.F32, operands=(converted, converted))
    )


def _sum_program() -> CastScalarProgram:
    left = CastScalarExpression(
        CastScalarKind.INPUT,
        CastScalarDType.F32,
        input_name="input0",
        input_index=ScalarIndexRelation(0, 0),
    )
    right = CastScalarExpression(
        CastScalarKind.INPUT,
        CastScalarDType.F32,
        input_name="input1",
        input_index=ScalarIndexRelation(0, 0),
    )
    return CastScalarProgram(CastScalarExpression(CastScalarKind.ADD, CastScalarDType.F32, operands=(left, right)))


def _kinds(expression: CastScalarExpression) -> set[CastScalarKind]:
    return {expression.kind}.union(*(_kinds(operand) for operand in expression.operands))


def _tiny_program() -> PartitionedGemmProgram:
    return PartitionedGemmProgram(
        shape=(2, 8, 3),
        partitioned_operand=1,
        operand_shapes=(
            "bf16[2,3]{1,0}",
            "bf16[4,3]{0,1}",
            "bf16[2,3]{0,1}",
            "bf16[2,3]{0,1}",
        ),
        partitions=(
            AccumulatorPartition(0, 4, "bf16[2,4]{1,0}"),
            AccumulatorPartition(4, 6, "bf16[2,2]{1,0}"),
            AccumulatorPartition(6, 8, "bf16[2,2]{1,0}"),
        ),
        scalar_finalizations=(),
        passthrough_finalizations=(
            PassthroughPartitionFinalization(0, "bf16[2,4]{1,0}"),
            PassthroughPartitionFinalization(1, "bf16[2,2]{1,0}"),
            PassthroughPartitionFinalization(2, "bf16[2,2]{1,0}"),
        ),
        input_dtype="bf16",
        accumulation_dtype="f32",
        partition_dtype="bf16",
        output_dtype="bf16",
        output_rounding="round_to_nearest_even",
        auxiliary_folds=(
            AuxiliaryPartitionFold(
                source_partition=0,
                input_shape="bf16[2,2,2]{2,1,0}",
                contribution=_self_product_program(),
                reducer=_sum_program(),
                initializer=0.0,
                output_shape="f32[2,2]{1,0}",
                accumulator_dtype="f32",
                output_dtype="f32",
                reassociation=PartitionFoldReassociation.ALLOW_ROUNDING_REORDER,
            ),
            AuxiliaryPartitionFold(
                source_partition=2,
                input_shape="bf16[2,1,2]{2,1,0}",
                contribution=_self_product_program(),
                reducer=_sum_program(),
                initializer=0.0,
                output_shape="f32[2,1]{1,0}",
                accumulator_dtype="f32",
                output_dtype="f32",
                reassociation=PartitionFoldReassociation.ALLOW_ROUNDING_REORDER,
            ),
        ),
    )


def test_natural_hlo_recovers_two_generic_auxiliary_folds_and_retains_raw_partitions() -> None:
    plan = plan_attached_partition_folds(_post_gated_product_grug_hlo(), target_prefix=_TARGET_PREFIX)

    assert len(plan.families) == 1
    assert len(plan.calls) == 1
    call = plan.calls[0]
    family = plan.families[0]
    program = family.program
    assert family.generated.template == "partitioned_gemm_auxiliary_fold_finalization"
    assert call.base.recovered.entry_instruction == "dot.87"
    assert tuple(output.instruction for output in call.base.outputs) == (
        "slice.66",
        "slice.67",
        "slice.68",
        "slice.69",
    )
    assert tuple(output.instruction for output in call.fold_outputs) == ("reduce_sum.630", "reduce_sum.632")
    assert program.shape == (8, 66, 32)
    assert tuple((partition.start, partition.limit) for partition in program.partitions) == (
        (0, 32),
        (32, 48),
        (48, 64),
        (64, 66),
    )
    assert tuple(value.source_partition for value in program.passthrough_finalizations) == (0, 1, 2, 3)
    assert tuple(value.source_partition for value in program.auxiliary_folds) == (0, 2)
    assert tuple(value.input_shape for value in program.auxiliary_folds) == (
        "bf16[2,4,2,16]{3,2,1,0}",
        "bf16[2,4,1,16]{3,2,1,0}",
    )
    assert all(value.contribution.expression.kind is CastScalarKind.MULTIPLY for value in program.auxiliary_folds)
    assert all(value.reducer.expression.kind is CastScalarKind.ADD for value in program.auxiliary_folds)
    assert all(
        value.reassociation is PartitionFoldReassociation.ALLOW_ROUNDING_REORDER for value in program.auxiliary_folds
    )


def test_recovery_does_not_require_a_workload_specific_fold_count() -> None:
    hlo = _ambiguous_second_partition_fold(_post_gated_product_grug_hlo())

    plan = plan_attached_partition_folds(hlo, target_prefix=_TARGET_PREFIX)

    assert len(plan.families) == 1
    assert tuple(fold.source_partition for fold in plan.families[0].program.auxiliary_folds) == (0,)


def test_replacement_preserves_raw_users_fold_users_and_collectives() -> None:
    hlo = _post_gated_product_grug_hlo()
    plan = plan_attached_partition_folds(hlo, target_prefix=_TARGET_PREFIX)
    transformed = replace_attached_partition_folds(hlo, plan)
    audit = audit_attached_partition_folds(hlo, transformed, plan)

    assert audit.removed_contract_count == 1
    assert audit.target_occurrences == ((plan.families[0].target, 1),)
    assert audit.retained_partition_users == (
        ("slice.66", ("reshape.1033",)),
        ("slice.67", ("broadcast.190", "mul.106")),
        ("slice.68", ("reshape.1034",)),
        ("slice.69", ("reshape.1022",)),
    )
    assert audit.fold_output_users == (
        ("reduce_sum.630", ("multiply.42",)),
        ("reduce_sum.632", ("multiply.43",)),
    )
    assert len(audit.collective_instructions) == 10
    assert "%square.12 =" not in transformed
    assert "%square.13 =" not in transformed
    assert "%convert.1 =" in transformed
    assert "%reduce_sum.630 = f32[2,4,2]{2,1,0} get-tuple-element" in transformed


def test_scalar_mutation_reuses_partition_and_fold_structure_but_regenerates_bodies() -> None:
    hlo = _post_gated_product_grug_hlo()
    original = plan_attached_partition_folds(hlo, target_prefix=_TARGET_PREFIX).families[0]
    mutated_plan = plan_attached_partition_folds(_negated_self_product_mutation(hlo), target_prefix=_TARGET_PREFIX)
    mutated = mutated_plan.families[0]

    assert original.program.shape == mutated.program.shape
    assert original.program.partitions == mutated.program.partitions
    assert original.program.output_shapes == mutated.program.output_shapes
    assert original.program.auxiliary_folds[1] == mutated.program.auxiliary_folds[1]
    assert original.program.auxiliary_folds[0].contribution != mutated.program.auxiliary_folds[0].contribution
    assert original.program.semantic_digest != mutated.program.semantic_digest
    assert original.target != mutated.target
    assert "(-input_r0_f0)" in mutated.generated.auxiliary_contribution_bodies[0].source


def test_auxiliary_folds_feed_generated_consumer_contract_preparation() -> None:
    hlo = _post_gated_product_grug_hlo()
    folds = plan_attached_partition_folds(hlo, target_prefix=_TARGET_PREFIX)

    plan = plan_fold_consumer_preparations(hlo, folds)

    assert len(plan.attachments) == 2
    assert tuple(attachment.raw_partition for attachment in plan.attachments) == ("slice.66", "slice.68")
    assert tuple(attachment.fold_output for attachment in plan.attachments) == ("reduce_sum.630", "reduce_sum.632")
    assert tuple(attachment.prepared_value for attachment in plan.attachments) == (
        "convert_element_type.361",
        "convert_element_type.367",
    )
    assert tuple(attachment.consumer_contract for attachment in plan.attachments) == ("dot.16", "dot.16")
    assert tuple(attachment.consumer_operand for attachment in plan.attachments) == (0, 1)
    assert all(attachment.scalar_program.expression.kind is CastScalarKind.CONVERT for attachment in plan.attachments)
    assert all(CastScalarKind.RSQRT in _kinds(attachment.scalar_program.expression) for attachment in plan.attachments)
    assert tuple(step.instruction for step in plan.attachments[0].consumer_steps) == (
        "mul.725",
        "transpose.16",
    )
    assert tuple(step.instruction for step in plan.attachments[1].consumer_steps) == (
        "reshape.485",
        "broadcast.196",
        "transpose.17",
    )


def test_fold_consumer_preparation_mutation_reuses_attachment_structure() -> None:
    hlo = _post_gated_product_grug_hlo()
    original_folds = plan_attached_partition_folds(hlo, target_prefix=_TARGET_PREFIX)
    mutated_hlo = _epsilon_mutation(hlo)
    mutated_folds = plan_attached_partition_folds(mutated_hlo, target_prefix=_TARGET_PREFIX)

    original = plan_fold_consumer_preparations(hlo, original_folds)
    mutated = plan_fold_consumer_preparations(mutated_hlo, mutated_folds)

    assert tuple(
        (attachment.raw_partition, attachment.fold_output, attachment.consumer_contract, attachment.consumer_operand)
        for attachment in original.attachments
    ) == tuple(
        (attachment.raw_partition, attachment.fold_output, attachment.consumer_contract, attachment.consumer_operand)
        for attachment in mutated.attachments
    )
    assert tuple(attachment.scalar_program.digest for attachment in original.attachments) != tuple(
        attachment.scalar_program.digest for attachment in mutated.attachments
    )


def test_reference_and_jax_keep_raw_outputs_and_emit_source_ordered_fold_values() -> None:
    program = _tiny_program()
    operands = (
        np.asarray([[0.5, -1.0, 0.25], [1.5, 0.75, -0.5]], dtype=ml_dtypes.bfloat16),
        np.asarray(
            [[1.0, 0.5, -0.25], [-0.5, 1.25, 0.75], [0.25, -0.75, 1.0], [1.5, 0.5, -1.25]],
            dtype=ml_dtypes.bfloat16,
        ),
        np.asarray([[0.5, 0.25, 2.0], [-1.0, 0.5, 0.75]], dtype=ml_dtypes.bfloat16),
        np.asarray([[1.0, -0.25, 0.5], [0.25, 1.5, -0.75]], dtype=ml_dtypes.bfloat16),
    )

    actual = evaluate_partitioned_gemm_reference(program, operands)
    jax_actual = evaluate_partitioned_gemm_jax(program, tuple(jnp.asarray(value) for value in operands))

    assert len(actual) == 5
    for reference, generated in zip(actual, jax_actual, strict=True):
        np.testing.assert_array_equal(np.asarray(generated), reference)
    np.testing.assert_array_equal(actual[3], np.sum(actual[0].astype(np.float32).reshape(2, 2, 2) ** 2, axis=-1))
    np.testing.assert_array_equal(actual[4], np.sum(actual[2].astype(np.float32).reshape(2, 1, 2) ** 2, axis=-1))


def test_bounded_cuda_source_emits_generic_contribution_and_reducer_asts() -> None:
    generated = generate_cuda_partitioned_gemm_ffi(
        _tiny_program(),
        target="shuttle.generic.partitioned_contract_fold.physical.test",
    )
    audit = audit_cuda_partitioned_gemm_source(generated)

    assert audit.kernel_count == 1
    assert audit.segmented_rhs_count == 3
    assert audit.direct_output_count == 5
    assert audit.has_ordered_fp32_mainloop
    assert audit.has_bf16_rne_partition_boundary
    assert not audit.has_atomics
    assert audit.opaque_semantic_dependencies == ()
    assert generated.source.count("generated_partition_fold_contribution_") >= 4
    assert generated.source.count("generated_partition_fold_reducer_") >= 4
    assert "ffi::Result<ffi::Buffer<ffi::F32, 2>> output3_buffer" in generated.source
    assert "fold_state = generated_partition_fold_reducer_0(fold_state, contribution);" in generated.source
