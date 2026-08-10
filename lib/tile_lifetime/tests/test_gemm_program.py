# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    GENERIC_H100_GEMM_BACKEND,
    AttachmentSite,
    DType,
    GemmSkeleton,
    TileOp,
    TilePrimitive,
    TileProgramError,
    TileProgramStage,
    compile_gemm_program,
    optimize_tile_program,
)
from tile_lifetime.cuda_prepared_contract_codegen import (
    PreparedContractOperandDelivery,
    audit_cuda_prepared_contract_source,
    execute_preparation_reference,
    execute_prepared_contract_reference,
    generate_cuda_prepared_contract,
)
from tile_lifetime.dense_flow import pairwise_silu_product_expression
from tile_lifetime.plan import Attachment
from tile_lifetime.quack_gemm_codegen import QuackOperandKind, generate_quack_gemm
from tile_lifetime.tensor_program import serialize_scalar_expression


def _gemm(
    *,
    prologue: tuple[Attachment, ...] = (),
    epilogue: tuple[Attachment, ...],
    output_layout: str = "row_major_mn",
) -> GemmSkeleton:
    return GemmSkeleton(
        name="name_must_not_select_the_kernel",
        input="x",
        weight="weight",
        output="accumulator",
        shape=(8, 16, 32),
        accumulation_dtype=DType.FP32,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout=output_layout,
        prologue=prologue,
        epilogue=epilogue,
    )


def test_gemm_attachment_mutation_changes_program_without_changing_backend() -> None:
    residual = Attachment(
        operation="residual_add",
        site=AttachmentSite.GEMM_EPILOGUE,
        inputs=("accumulator", "residual"),
        outputs=("sum",),
    )
    baseline = _gemm(
        epilogue=(
            residual,
            Attachment(
                operation="multiply_gamma",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=("sum", "gamma"),
                outputs=("scaled",),
            ),
        )
    )
    mutated = replace(
        baseline,
        name="still_not_a_dispatch_key",
        epilogue=(
            residual,
            Attachment(
                operation="partial_sum_square",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=("sum",),
                outputs=("partials",),
            ),
        ),
    )

    baseline_program = compile_gemm_program(baseline)
    mutated_program = compile_gemm_program(mutated)

    assert baseline.backend == mutated.backend == GENERIC_H100_GEMM_BACKEND
    assert TilePrimitive.MULTIPLY_GAMMA in baseline_program.tile_program.primitives_at(TileProgramStage.FINALIZATION)
    assert TilePrimitive.MULTIPLY_GAMMA not in mutated_program.tile_program.primitives_at(TileProgramStage.FINALIZATION)
    assert mutated_program.stored_values == ("partials",)


def test_tile_program_optimizes_dataflow_and_reports_liveness() -> None:
    operations = (
        TileOp(TilePrimitive.ADD, TileProgramStage.PREPARATION, ("left", "right"), ("sum",)),
        TileOp(TilePrimitive.ADD, TileProgramStage.PREPARATION, ("left", "right"), ("duplicate_sum",)),
        TileOp(TilePrimitive.MULTIPLY, TileProgramStage.PREPARATION, ("duplicate_sum", "scale"), ("result",)),
        TileOp(TilePrimitive.ADD, TileProgramStage.PREPARATION, ("left", "unused"), ("dead",)),
    )

    program = optimize_tile_program(operations, required_outputs=("result",))

    assert program.primitives_at(TileProgramStage.PREPARATION) == (TilePrimitive.ADD, TilePrimitive.MULTIPLY)
    assert program.aliases == (("duplicate_sum", "sum"),)
    assert {lifetime.value for lifetime in program.lifetimes if not lifetime.external} == {"sum", "result"}
    assert program.peak_live_values == 2


def test_tile_program_core_represents_relation_weighted_state_update() -> None:
    operations = (
        TileOp(TilePrimitive.LOAD_EDGE_WEIGHT, TileProgramStage.PREPARATION, ("edge",), ("weight",)),
        TileOp(TilePrimitive.LOAD_STATE, TileProgramStage.PREPARATION, ("state",), ("old_state",)),
        TileOp(TilePrimitive.MULTIPLY, TileProgramStage.PREPARATION, ("input", "weight"), ("weighted",)),
        TileOp(TilePrimitive.ADD, TileProgramStage.PREPARATION, ("old_state", "weighted"), ("new_state",)),
        TileOp(
            TilePrimitive.CONVERT,
            TileProgramStage.PREPARATION,
            ("new_state",),
            ("new_state_bf16",),
            (("dtype", "bf16"),),
        ),
    )

    program = optimize_tile_program(operations, required_outputs=("new_state_bf16",))

    assert program.primitives_at(TileProgramStage.PREPARATION) == (
        TilePrimitive.LOAD_EDGE_WEIGHT,
        TilePrimitive.LOAD_STATE,
        TilePrimitive.MULTIPLY,
        TilePrimitive.ADD,
        TilePrimitive.CONVERT,
    )


def test_consumer_prologue_scale_exposes_bf16_conversion_before_mainloop() -> None:
    scale = Attachment(
        operation="scale_row",
        site=AttachmentSite.GEMM_PROLOGUE,
        inputs=("x", "inverse_rms"),
        outputs=("normalized",),
    )

    program = compile_gemm_program(_gemm(prologue=(scale,), epilogue=()))

    assert program.tile_program.primitives_at(TileProgramStage.PREPARATION) == (
        TilePrimitive.SCALE_ROW,
        TilePrimitive.CONVERT,
    )
    assert program.mainloop_input == "normalized.mainloop_bf16"
    assert program.tile_program.primitives_at(TileProgramStage.FINALIZATION) == (
        TilePrimitive.CONVERT,
        TilePrimitive.STORE,
    )


def test_gemm_program_rejects_rope_layout_that_attention_cannot_consume() -> None:
    rope = Attachment(
        operation="pairwise_rope_q",
        site=AttachmentSite.GEMM_EPILOGUE,
        inputs=("accumulator", "sine", "cosine"),
        outputs=("rotated",),
    )

    with pytest.raises(TileProgramError):
        compile_gemm_program(_gemm(epilogue=(rope,)))


def test_quack_codegen_is_invariant_to_workload_and_value_names() -> None:
    def residual_program(name: str, accumulator: str, residual: str, gamma: str) -> GemmSkeleton:
        return replace(
            _gemm(
                epilogue=(
                    Attachment(
                        operation="residual_add",
                        site=AttachmentSite.GEMM_EPILOGUE,
                        inputs=(accumulator, residual),
                        outputs=("summed",),
                    ),
                    Attachment(
                        operation="multiply_gamma",
                        site=AttachmentSite.GEMM_EPILOGUE,
                        inputs=("summed", gamma),
                        outputs=("accumulator",),
                    ),
                    Attachment(
                        operation="partial_sum_square",
                        site=AttachmentSite.GEMM_EPILOGUE,
                        inputs=("summed",),
                        outputs=("partials",),
                    ),
                )
            ),
            name=name,
        )

    first = generate_quack_gemm(compile_gemm_program(residual_program("transformer_block", "projected", "x", "g")))
    renamed = generate_quack_gemm(
        compile_gemm_program(residual_program("unrelated_workload", "arbitrary_dot", "skip", "row_weight"))
    )

    assert first.source == renamed.source
    assert first.digest == renamed.digest
    assert first.c_source == "x"
    assert renamed.c_source == "skip"
    assert tuple(operand.kind for operand in first.operands) == (QuackOperandKind.ROW,)
    assert {output.destination for output in first.outputs} == {"summed", "partials"}
    assert "value_0 = acc + c" in first.source
    assert "transformer" not in first.source


def test_quack_codegen_composes_generated_a_transform_and_pairwise_map() -> None:
    scale = Attachment(
        operation="scale_row",
        site=AttachmentSite.GEMM_PROLOGUE,
        inputs=("x", "row_scale"),
        outputs=("scaled",),
    )
    pairwise_map = Attachment(
        operation="pairwise_map",
        site=AttachmentSite.GEMM_EPILOGUE,
        inputs=("raw_pairs",),
        outputs=("accumulator",),
        attributes=(("expression_ast", serialize_scalar_expression(pairwise_silu_product_expression())),),
    )

    generated = generate_quack_gemm(compile_gemm_program(_gemm(prologue=(scale,), epilogue=(pairwise_map,))))

    assert generated.has_transform
    assert not generated.writes_main_output
    assert "(activation) * (operand_0)" in generated.source
    assert "cute.exp" in generated.source
    assert "swiglu" not in generated.source
    assert tuple(operand.kind for operand in generated.operands) == (QuackOperandKind.COLUMN,)


def _centered_affine_preparation(*, bias_operation: str = "add") -> GemmSkeleton:
    return _gemm(
        prologue=(
            Attachment(
                operation="subtract",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=("x", "row_center"),
                outputs=("centered",),
                attributes=(("input.1_delivery", "row"),),
            ),
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=("centered", "row_scale"),
                outputs=("scaled",),
            ),
            Attachment(
                operation="multiply",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=("scaled", "feature_scale"),
                outputs=("affine_scaled",),
                attributes=(("input.1_delivery", "feature"),),
            ),
            Attachment(
                operation=bias_operation,
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=("affine_scaled", "feature_bias"),
                outputs=("prepared",),
                attributes=(("input.1_delivery", "feature"),),
            ),
        ),
        epilogue=(),
    )


def test_centered_affine_preparation_generates_explicit_physical_program() -> None:
    program = compile_gemm_program(_centered_affine_preparation())
    quack = generate_quack_gemm(program)
    bounded = generate_cuda_prepared_contract(program)
    audit = audit_cuda_prepared_contract_source(bounded)

    assert program.tile_program.primitives_at(TileProgramStage.PREPARATION) == (
        TilePrimitive.SUBTRACT,
        TilePrimitive.SCALE_ROW,
        TilePrimitive.MULTIPLY,
        TilePrimitive.ADD,
        TilePrimitive.CONVERT,
    )
    assert tuple(operand.kind for operand in quack.operands) == (
        QuackOperandKind.COLUMN,
        QuackOperandKind.COLUMN,
        QuackOperandKind.FEATURE,
        QuackOperandKind.FEATURE,
    )
    assert quack.transform_auxiliary_count == 4
    assert not quack.executable_with_single_auxiliary_transform_backend
    assert "'operand_0': 'colvec_ktile_fp32'" in quack.source
    assert "'operand_2': 'kvec_mtile_fp32'" in quack.source
    assert "(((activation) - (operand_0)) * (operand_1)) * (operand_2)" in quack.source
    compile(quack.source, "<generated-quack>", "exec")

    assert tuple(operand.delivery for operand in bounded.operands) == (
        PreparedContractOperandDelivery.ROW,
        PreparedContractOperandDelivery.ROW,
        PreparedContractOperandDelivery.FEATURE,
        PreparedContractOperandDelivery.FEATURE,
    )
    assert audit.kernel_count == 1
    assert audit.row_operand_count == 2
    assert audit.feature_operand_count == 2
    assert audit.has_explicit_fp32_preparation
    assert audit.has_bf16_rne_mainloop_boundary
    assert audit.has_ordered_fp32_accumulation
    assert not audit.has_atomics
    assert not audit.opaque_semantic_dependencies


def test_centered_affine_quack_source_imports_against_optional_backend_surface(monkeypatch) -> None:
    def decorator(*_args, **_kwargs):
        return lambda function: function

    cutlass = types.ModuleType("cutlass")
    cute = types.ModuleType("cutlass.cute")
    cutlass.cute = cute
    quack = types.ModuleType("quack")
    epilogue = types.ModuleType("quack.epilogue")
    epilogue.gemm_epilogue = decorator
    epilogue.pack = lambda left, right: (left, right)
    epilogue.unpack = lambda pair: pair
    epilogue_ops = types.ModuleType("quack.epilogue.ops")
    epilogue_ops.ColVecLoad = object
    epilogue_ops.ColVecReduce = object
    epilogue_ops.RowVecLoad = object
    epilogue_ops.TileLoad = object
    operand_transform = types.ModuleType("quack.operand_transform")
    operand_transform.a_transform = decorator
    for name, module in {
        "cutlass": cutlass,
        "cutlass.cute": cute,
        "quack": quack,
        "quack.epilogue": epilogue,
        "quack.epilogue.ops": epilogue_ops,
        "quack.operand_transform": operand_transform,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    generated = generate_quack_gemm(compile_gemm_program(_centered_affine_preparation()))
    namespace: dict[str, object] = {}
    exec(compile(generated.source, "<generated-quack>", "exec"), namespace)
    transform = namespace["generated_transform"]

    assert callable(transform)
    assert transform(2.0, 0.5, 2.0, 3.0, 4.0) == 13.0


def test_centered_affine_preparation_reference_matches_natural_jax_boundary() -> None:
    rng = np.random.default_rng(61)
    program = compile_gemm_program(_centered_affine_preparation())
    activation = np.asarray(jnp.asarray(rng.normal(size=(8, 32)), dtype=jnp.bfloat16), dtype=np.float32)
    weight = np.asarray(jnp.asarray(rng.normal(size=(16, 32)), dtype=jnp.bfloat16), dtype=np.float32)
    row_center = rng.normal(size=(8,)).astype(np.float32)
    row_scale = (rng.random(size=(8,)) + 0.25).astype(np.float32)
    feature_scale = np.asarray(jnp.asarray(rng.normal(size=(32,)), dtype=jnp.bfloat16), dtype=np.float32)
    feature_bias = np.asarray(jnp.asarray(rng.normal(size=(32,)), dtype=jnp.bfloat16), dtype=np.float32)
    values = {
        "row_center": row_center,
        "row_scale": row_scale,
        "feature_scale": feature_scale,
        "feature_bias": feature_bias,
    }

    actual_boundary = execute_preparation_reference(program, activation, values)
    expected_boundary = np.asarray(
        jnp.asarray(
            (
                (jnp.asarray(activation, dtype=jnp.float32) - row_center[:, None])
                * row_scale[:, None]
                * jnp.asarray(feature_scale, dtype=jnp.float32)[None, :]
                + jnp.asarray(feature_bias, dtype=jnp.float32)[None, :]
            ),
            dtype=jnp.bfloat16,
        ),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual_boundary, expected_boundary)

    actual_output = execute_prepared_contract_reference(program, activation, weight, values)
    jax_output = np.asarray(
        jnp.asarray(
            jnp.asarray(expected_boundary, dtype=jnp.bfloat16) @ jnp.asarray(weight, dtype=jnp.bfloat16).T,
            dtype=jnp.bfloat16,
        ),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual_output, jax_output)


def test_centered_affine_map_mutation_reuses_prepared_contract_family() -> None:
    baseline = generate_cuda_prepared_contract(compile_gemm_program(_centered_affine_preparation()))
    mutated = generate_cuda_prepared_contract(
        compile_gemm_program(_centered_affine_preparation(bias_operation="subtract"))
    )

    assert tuple(operand.delivery for operand in baseline.operands) == tuple(
        operand.delivery for operand in mutated.operands
    )
    assert baseline.semantic_digest != mutated.semantic_digest
    assert baseline.source_digest != mutated.source_digest
    assert "__fadd_rn(prepared_2, operand_3[reduction])" in baseline.source
    assert "__fsub_rn(prepared_2, operand_3[reduction])" in mutated.source


def test_quack_codegen_rope_uses_runtime_tables_as_data() -> None:
    skeleton = _gemm(
        epilogue=(
            Attachment(
                operation="partition_qkv_segment_views_bshd",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=("raw_qkv",),
                outputs=("query", "key", "value"),
            ),
            Attachment(
                operation="pairwise_rope_q",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=("query", "arbitrary_sine", "arbitrary_cosine"),
                outputs=("rotated_query",),
            ),
            Attachment(
                operation="pairwise_rope_k",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=("key", "arbitrary_sine", "arbitrary_cosine"),
                outputs=("rotated_key",),
            ),
        ),
        output_layout="fa3_bshd_last_dimension_contiguous",
    )

    generated = generate_quack_gemm(compile_gemm_program(skeleton))

    assert tuple(operand.kind for operand in generated.operands) == (QuackOperandKind.PAIR_COEFFICIENT_TILE,)
    assert generated.operands[0].sources == ("arbitrary_sine", "arbitrary_cosine")
    assert "TileLoad('operand_0')" in generated.source
    assert "10000" not in generated.source
