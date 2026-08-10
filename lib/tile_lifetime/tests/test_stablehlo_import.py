# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import replace
from pathlib import Path

import pytest

from tile_lifetime.ir import DType
from tile_lifetime.pipeline import (
    FrontendSourceKind,
    compile_stablehlo_streaming_attention_program,
    validate_stablehlo_streaming_attention_compilation,
)
from tile_lifetime.plan import (
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    ReductionSkeleton,
    StreamingAttentionSkeleton,
)
from tile_lifetime.reference_pipeline import (
    compile_reference_stablehlo_attention_region,
    compile_reference_stablehlo_rms_attention_program,
    compile_reference_stablehlo_rms_region,
)
from tile_lifetime.semantic_erasure import SemanticErasureError, validate_plan_semantic_erasure
from tile_lifetime.semantic_recovery import recover_attention_region, recover_rms_region
from tile_lifetime.stablehlo_import import CompareAttributes, DotAttributes, ReductionAttributes, import_stablehlo
from tile_lifetime.streaming_attention import StreamingTileSchedule

FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "rms_region_v1_14_1.mlir.bc.b64"
ATTENTION_FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "causal_gqa_attention_v1_14_1.mlir.bc.b64"
PROGRAM_FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "coda_fa3_program_v1_14_1.mlir.bc.b64"
INPUT_NAMES = ("x", "residual", "weight_0", "gamma", "weight_1")
ATTENTION_INPUT_NAMES = ("query", "key", "value")
PROGRAM_INPUT_NAMES = (*INPUT_NAMES, *ATTENTION_INPUT_NAMES)


def test_import_stablehlo_recovers_static_values_and_dataflow() -> None:
    graph = import_stablehlo(base64.b64decode(FIXTURE.read_text()), input_names=INPUT_NAMES)

    assert [
        (graph.value(value_id).name, graph.value(value_id).shape, graph.value(value_id).dtype)
        for value_id in graph.inputs
    ] == [
        ("x", (2, 4), DType.BF16),
        ("residual", (2, 3), DType.BF16),
        ("weight_0", (4, 3), DType.BF16),
        ("gamma", (3,), DType.BF16),
        ("weight_1", (3, 5), DType.BF16),
    ]
    assert graph.value(graph.outputs[0]).shape == (2, 5)
    assert graph.value(graph.outputs[0]).dtype is DType.BF16

    dots = tuple(operation for operation in graph.operations if operation.kind == "dot_general")
    assert len(dots) == 2
    assert isinstance(dots[0].attributes, DotAttributes)
    assert dots[0].attributes.lhs_contracting_dimensions == (1,)
    assert dots[0].attributes.rhs_contracting_dimensions == (0,)
    assert graph.producer(graph.outputs[0]) == dots[1]
    assert graph.value(dots[0].inputs[0]).name == "x"
    assert graph.value(dots[1].inputs[1]).name == "weight_1"


def test_import_stablehlo_preserves_reduction_semantics_and_provenance() -> None:
    graph = import_stablehlo(base64.b64decode(FIXTURE.read_text()), input_names=INPUT_NAMES)

    reductions = tuple(operation for operation in graph.operations if operation.kind == "reduce")
    assert len(reductions) == 1
    assert isinstance(reductions[0].attributes, ReductionAttributes)
    assert reductions[0].attributes.dimensions == (1,)
    assert reductions[0].attributes.reducer == "add"
    assert "reduce_sum" in reductions[0].source_location


def test_reference_stablehlo_rms_region_selects_delayed_scale_plan() -> None:
    artifact = base64.b64decode(FIXTURE.read_text())
    imported = import_stablehlo(artifact, input_names=INPUT_NAMES)
    recovered = recover_rms_region(
        imported,
        gemm_accumulation_dtype=DType.FP32,
        output_name="output",
    )
    plan = compile_reference_stablehlo_rms_region(
        artifact,
        input_names=INPUT_NAMES,
        output_name="output",
        gemm_accumulation_dtype=DType.FP32,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert len(recovered.source_operation_ids) == len(imported.operations)
    assert all(operation.source_location is not None for operation in recovered.graph.operations)
    assert [type(skeleton) for skeleton in plan.skeletons] == [GemmSkeleton, ReductionSkeleton, GemmSkeleton]
    assert plan.materialization("normalized").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.rewrites[0].applied
    assert plan.semantic_erasure_report is not None
    assert plan.semantic_erasure_report.is_clean
    validate_plan_semantic_erasure(plan)


def test_import_stablehlo_preserves_attention_axes_and_causal_comparison() -> None:
    graph = import_stablehlo(base64.b64decode(ATTENTION_FIXTURE.read_text()), input_names=ATTENTION_INPUT_NAMES)

    assert [graph.value(value_id).shape for value_id in graph.inputs] == [
        (1, 5, 6, 64),
        (1, 5, 2, 64),
        (1, 5, 2, 64),
    ]
    dots = tuple(operation for operation in graph.operations if operation.kind == "dot_general")
    assert len(dots) == 2
    assert isinstance(dots[0].attributes, DotAttributes)
    assert dots[0].attributes.lhs_batching_dimensions == (0, 2)
    assert dots[0].attributes.lhs_contracting_dimensions == (3,)
    comparisons = tuple(operation for operation in graph.operations if operation.kind == "compare")
    assert len(comparisons) == 1
    assert comparisons[0].attributes == CompareAttributes(direction="LE", compare_type="SIGNED")


def test_reference_stablehlo_attention_selects_opaque_streaming_plan() -> None:
    artifact = base64.b64decode(ATTENTION_FIXTURE.read_text())
    imported = import_stablehlo(artifact, input_names=ATTENTION_INPUT_NAMES)
    recovered = recover_attention_region(imported, output_name="attention_output")
    plan = compile_reference_stablehlo_attention_region(
        artifact,
        input_names=ATTENTION_INPUT_NAMES,
        output_name="attention_output",
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert len(recovered.source_operation_ids) == len(imported.operations)
    assert len(recovered.graph.operations) == 1
    skeleton = plan.skeletons[0]
    assert isinstance(skeleton, StreamingAttentionSkeleton)
    assert skeleton.head_dimension == 64
    assert skeleton.query_heads == 6
    assert skeleton.key_value_heads == 2
    assert skeleton.causal
    assert skeleton.backend == "official_flashattention_3_hopper"
    assert skeleton.pipeline_stages == 2
    assert skeleton.producer_threads == 32
    assert skeleton.consumer_threads == 384
    assert plan.materialization("attention_output.scores").disposition is (
        MaterializationDisposition.INTERNAL_ATTENTION_STATE
    )
    assert plan.materialization("attention_output.probabilities").disposition is (
        MaterializationDisposition.INTERNAL_ATTENTION_STATE
    )
    assert plan.sequence_squared_materializations == ()
    assert not any(record.shape[-2:] == (5, 5) for record in plan.activation_materializations)


def test_stablehlo_attention_lowers_to_backend_neutral_streaming_program() -> None:
    compilation = compile_stablehlo_streaming_attention_program(
        base64.b64decode(ATTENTION_FIXTURE.read_text()),
        input_names=ATTENTION_INPUT_NAMES,
        output_name="attention_output",
        schedule=StreamingTileSchedule(query_tile_size=64, key_value_tile_size=128, pipeline_depth=2),
    )

    program = compilation.program
    assert program.qk.inputs[0].shape == (1, 5, 6, 64)
    assert program.qk.inputs[1].shape == (1, 5, 2, 64)
    assert program.pv.inputs[1].shape == (1, 5, 2, 64)
    assert program.qk.index_maps_for_input(1)[0].divisor == 3
    assert program.pv.index_maps_for_input(1)[0].divisor == 3
    assert program.schedule == StreamingTileSchedule(64, 128, 2)
    assert not hasattr(program, "backend")
    assert compilation.provenance.source_kind is FrontendSourceKind.STABLEHLO_ARTIFACT
    assert compilation.semantic_erasure_report.is_clean


def test_current_streaming_frontend_rejects_hand_authored_provenance() -> None:
    compilation = compile_stablehlo_streaming_attention_program(
        base64.b64decode(ATTENTION_FIXTURE.read_text()),
        input_names=ATTENTION_INPUT_NAMES,
        output_name="attention_output",
        schedule=StreamingTileSchedule(query_tile_size=64, key_value_tile_size=128, pipeline_depth=2),
    )
    bypass = replace(
        compilation,
        provenance=replace(
            compilation.provenance,
            source_kind=FrontendSourceKind.HAND_AUTHORED_SEMANTIC_IR,
        ),
    )

    with pytest.raises(SemanticErasureError, match="must originate from a StableHLO artifact"):
        validate_stablehlo_streaming_attention_compilation(bypass)


def test_current_streaming_frontend_rejects_named_kernel_scheduling_key() -> None:
    compilation = compile_stablehlo_streaming_attention_program(
        base64.b64decode(ATTENTION_FIXTURE.read_text()),
        input_names=ATTENTION_INPUT_NAMES,
        output_name="attention_output",
        schedule=StreamingTileSchedule(query_tile_size=64, key_value_tile_size=128, pipeline_depth=2),
    )
    named = replace(
        compilation,
        semantic_erasure_report=replace(
            compilation.semantic_erasure_report,
            scheduling_keys=(*compilation.semantic_erasure_report.scheduling_keys, "flashattention_3"),
        ),
    )

    with pytest.raises(SemanticErasureError, match="retains named semantics"):
        validate_stablehlo_streaming_attention_compilation(named)


def test_reference_stablehlo_program_recovers_coda_and_fa3() -> None:
    plan = compile_reference_stablehlo_rms_attention_program(
        base64.b64decode(PROGRAM_FIXTURE.read_text()),
        input_names=PROGRAM_INPUT_NAMES,
        rms_output_name="rms_output",
        attention_output_name="attention_output",
        gemm_accumulation_dtype=DType.FP32,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [
        GemmSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
        StreamingAttentionSkeleton,
    ]
    assert [rewrite.name for rewrite in plan.rewrites] == [
        "move_row_scalar_through_right_contract",
        "stream_exact_attention",
    ]
    assert all(rewrite.applied for rewrite in plan.rewrites)
    assert plan.materialization("normalized").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("attention_output.scores").disposition is (
        MaterializationDisposition.INTERNAL_ATTENTION_STATE
    )
