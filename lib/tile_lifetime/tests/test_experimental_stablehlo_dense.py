# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from shuttle.experimental.stablehlo_import import ConcatenateAttributes, SliceAttributes, import_stablehlo
from shuttle.ir import DType
from tile_lifetime import (
    GemmSkeleton,
    NumericalPolicy,
    ReductionSkeleton,
    StreamingAttentionSkeleton,
    TransformSkeleton,
)
from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.experimental_pipeline import (
    FrontendCompilationStatus,
    FrontendSourceKind,
    compile_experimental_stablehlo_dense_transformer_region,
    require_architecturally_conforming_dense_compilation,
    validate_experimental_stablehlo_dense_compilation,
)
from tile_lifetime.experimental_semantic_recovery import recover_dense_transformer_region
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.reference import DENSE_REGION_INPUT_NAMES, DenseDebugConfig, export_debug_dense_region
from tile_lifetime.semantic_erasure import SemanticErasureError

FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "dense_region_v1_14_1.mlir.bc.b64"


def _fixture_artifact() -> bytes:
    return base64.b64decode(FIXTURE.read_text())


def test_parameterized_jax_export_keeps_parameters_as_inputs() -> None:
    config = DenseDebugConfig(sequence=5)
    graph = import_stablehlo(export_debug_dense_region(config), input_names=DENSE_REGION_INPUT_NAMES)

    assert [graph.value(value_id).shape for value_id in graph.inputs] == [
        (5, 128),
        (128, 256),
        (128, 128),
        (128,),
        (128, 512),
        (256, 128),
        (128,),
        (128, 256),
        (5, 32),
        (5, 32),
    ]
    assert len(tuple(operation for operation in graph.operations if operation.kind == "constant")) == 6
    assert [graph.value(value_id).shape for value_id in graph.outputs] == [
        (5, 128),
        (1, 5, 2, 64),
        (1, 5, 1, 64),
        (1, 5, 1, 64),
    ]


def test_frozen_dense_fixture_preserves_partition_and_rope_semantics() -> None:
    graph = import_stablehlo(_fixture_artifact(), input_names=DENSE_REGION_INPUT_NAMES)

    slices = tuple(operation for operation in graph.operations if operation.kind == "slice")
    concatenations = tuple(operation for operation in graph.operations if operation.kind == "concatenate")
    assert len(slices) == 16
    assert len(concatenations) == 4
    assert all(isinstance(operation.attributes, SliceAttributes) for operation in slices)
    assert all(isinstance(operation.attributes, ConcatenateAttributes) for operation in concatenations)
    assert all("reference.py" in operation.source_location for operation in (*slices, *concatenations))


def test_recover_dense_fixture_builds_connected_semantic_graph() -> None:
    imported = import_stablehlo(_fixture_artifact(), input_names=DENSE_REGION_INPUT_NAMES)
    recovered = recover_dense_transformer_region(imported, gemm_accumulation_dtype=DType.FP32)

    assert len(recovered.source_operation_ids) == len(imported.operations) == 184
    assert [type(operation).__name__ for operation in recovered.graph.operations] == [
        "ViewOp",
        "QKVProjectionOp",
        "RoPEOp",
        "ScaledDotProductAttentionOp",
        "ViewOp",
        "LinearOp",
        "ResidualAddOp",
        "RMSNormOp",
        "LinearOp",
        "PairwiseSwiGLUOp",
        "LinearOp",
        "ResidualAddOp",
        "RMSNormOp",
        "ViewOp",
        "QKVProjectionOp",
        "RoPEOp",
    ]
    assert all(operation.source_location is not None for operation in recovered.graph.operations)


def test_experimental_dense_stablehlo_path_selects_eight_skeleton_plan() -> None:
    artifact = _fixture_artifact()
    compilation = compile_experimental_stablehlo_dense_transformer_region(
        artifact,
        input_names=DENSE_REGION_INPUT_NAMES,
        gemm_accumulation_dtype=DType.FP32,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    plan = compilation.plan

    assert compilation.provenance.source_kind is FrontendSourceKind.STABLEHLO_ARTIFACT
    assert compilation.status is FrontendCompilationStatus.EXPERIMENTAL_EXACT_RECOGNIZER
    assert compilation.provenance.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert compilation.provenance.source_operation_ids == tuple(range(184))
    assert [type(skeleton) for skeleton in plan.skeletons] == [
        GemmSkeleton,
        StreamingAttentionSkeleton,
        GemmSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
        GemmSkeleton,
        ReductionSkeleton,
        GemmSkeleton,
    ]
    assert not any(isinstance(skeleton, TransformSkeleton) for skeleton in plan.skeletons)
    assert plan.sequence_squared_materializations == ()
    assert all(rewrite.applied for rewrite in plan.rewrites)


def test_experimental_dense_stablehlo_path_exposes_delayed_rms_alternative() -> None:
    compilation = compile_experimental_stablehlo_dense_transformer_region(
        _fixture_artifact(),
        input_names=DENSE_REGION_INPUT_NAMES,
        gemm_accumulation_dtype=DType.FP32,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        rms_scale_placement=RowScalePlacement.CONSUMER_EPILOGUE,
    )
    plan = compilation.plan

    gate_up = plan.skeletons[4]
    next_qkv = plan.skeletons[7]
    assert isinstance(gate_up, GemmSkeleton)
    assert isinstance(next_qkv, GemmSkeleton)
    assert gate_up.backend == GENERIC_H100_GEMM_BACKEND
    assert next_qkv.backend == GENERIC_H100_GEMM_BACKEND


def test_experimental_dense_stablehlo_path_rejects_unverified_provenance_and_named_schedule_keys() -> None:
    compilation = compile_experimental_stablehlo_dense_transformer_region(
        _fixture_artifact(),
        input_names=DENSE_REGION_INPUT_NAMES,
        gemm_accumulation_dtype=DType.FP32,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    hand_authored = replace(
        compilation,
        provenance=replace(compilation.provenance, source_kind=FrontendSourceKind.HAND_AUTHORED_SEMANTIC_IR),
    )
    with pytest.raises(SemanticErasureError, match="StableHLO artifact"):
        validate_experimental_stablehlo_dense_compilation(hand_authored)

    report = compilation.plan.semantic_erasure_report
    assert report is not None
    named_report = replace(report, scheduling_keys=(*report.scheduling_keys, "moe_forward"))
    named_plan = replace(compilation.plan, semantic_erasure_report=named_report)
    with pytest.raises(SemanticErasureError, match="named semantics"):
        validate_experimental_stablehlo_dense_compilation(replace(compilation, plan=named_plan))

    with pytest.raises(SemanticErasureError, match="requires in-pipeline Shuttle MLIR provenance"):
        require_architecturally_conforming_dense_compilation(compilation)
