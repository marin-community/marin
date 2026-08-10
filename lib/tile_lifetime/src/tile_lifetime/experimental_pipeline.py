# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental Python StableHLO recovery pipelines."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from shuttle.experimental.stablehlo_import import import_stablehlo
from shuttle.ir import DType
from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.dense_region import compile_dense_transformer_region
from tile_lifetime.experimental_msa_recovery import (
    NaturalProjectedRoutedAttentionCompilation,
    RecoveredProjectedRoutedAttentionProgram,
    compile_natural_projected_routed_attention,
    recover_projected_routed_attention_program,
)
from tile_lifetime.experimental_routed_attention_recovery import (
    NaturalRoutedAttentionCompilation,
    RecoveredRoutedAttentionProgram,
    compile_natural_routed_attention,
    recover_routed_attention_program,
)
from tile_lifetime.experimental_semantic_recovery import (
    recover_attention_region,
    recover_dense_transformer_region,
)
from tile_lifetime.ir import ScaledDotProductAttentionOp
from tile_lifetime.plan import NumericalPolicy, RegionPlan, SemanticErasureReport, SemanticLoweringStep
from tile_lifetime.routed_attention_plan import RoutedAttentionPlanConfig
from tile_lifetime.semantic_erasure import (
    ErasedTensorProgram,
    SemanticErasureError,
    build_tensor_erasure_report,
    validate_erased_tensor_program,
    validate_plan_semantic_erasure,
)
from tile_lifetime.streaming_attention import (
    StreamingAttentionProgram,
    StreamingTileSchedule,
    streaming_attention_from_semantic_operation,
)


class FrontendSourceKind(StrEnum):
    """Source boundary retained by an experimental recovery result."""

    STABLEHLO_ARTIFACT = "stablehlo_artifact"
    HAND_AUTHORED_SEMANTIC_IR = "hand_authored_semantic_ir"


class FrontendCompilationStatus(StrEnum):
    """Architecture status of an experimental frontend result."""

    EXPERIMENTAL_EXACT_RECOGNIZER = "experimental_exact_recognizer"


@dataclass(frozen=True)
class FrontendProvenance:
    """Source evidence carried across semantic erasure."""

    source_kind: FrontendSourceKind
    artifact_sha256: str
    source_operation_ids: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentalWholePatternStreamingAttentionCompilation:
    """Historical named-attention recovery retained for bounded comparisons."""

    program: StreamingAttentionProgram
    provenance: FrontendProvenance
    semantic_erasure_report: SemanticErasureReport


@dataclass(frozen=True)
class StableHLODenseCompilation:
    """Dense plan with source evidence and an experimental classification."""

    plan: RegionPlan
    provenance: FrontendProvenance
    status: FrontendCompilationStatus


def compile_experimental_whole_pattern_stablehlo_streaming_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    schedule: StreamingTileSchedule,
) -> ExperimentalWholePatternStreamingAttentionCompilation:
    """Reconstruct a named attention operation before erasing it.

    This whole-pattern path is not accepted clean frontend provenance. The
    operation-by-operation StableHLO importer is reusable substrate. Attention
    selectors remain experimental.
    """
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_attention_region(stablehlo_graph, output_name=output_name)
    operations = recovered.graph.operations
    if len(operations) != 1 or not isinstance(operations[0], ScaledDotProductAttentionOp):
        raise ValueError("expected exactly one recovered semantic attention operation")
    program = streaming_attention_from_semantic_operation(operations[0], schedule=schedule)
    report = build_tensor_erasure_report(
        program.source,
        source_semantics=("stablehlo.normalized_exponential_attention",),
        lowering_steps=(
            SemanticLoweringStep(
                source_semantic="stablehlo.normalized_exponential_attention",
                generic_primitives=("Contract", "Map", "DomainRestriction", "Fold", "Contract", "Map"),
            ),
        ),
    )
    compilation = ExperimentalWholePatternStreamingAttentionCompilation(
        program=program,
        provenance=FrontendProvenance(
            source_kind=FrontendSourceKind.STABLEHLO_ARTIFACT,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            source_operation_ids=recovered.source_operation_ids,
        ),
        semantic_erasure_report=report,
    )
    validate_experimental_whole_pattern_streaming_attention_compilation(compilation)
    return compilation


def validate_experimental_whole_pattern_streaming_attention_compilation(
    compilation: ExperimentalWholePatternStreamingAttentionCompilation,
) -> None:
    """Reject hand-authored inputs even on the historical comparison path."""
    if compilation.provenance.source_kind is not FrontendSourceKind.STABLEHLO_ARTIFACT:
        raise SemanticErasureError("experimental StableHLO candidates must originate from a StableHLO artifact")
    erased = ErasedTensorProgram(compilation.program.source, compilation.semantic_erasure_report)
    validate_erased_tensor_program(erased)


def recover_experimental_stablehlo_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
) -> RecoveredRoutedAttentionProgram:
    """Recover natural selected-attention math and erase names into generic algebra."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    return recover_routed_attention_program(stablehlo_graph)


def compile_experimental_stablehlo_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    runtime_inputs: dict[str, np.ndarray],
    schedule: StreamingTileSchedule,
    config: RoutedAttentionPlanConfig,
    padding_quantum: int = 1,
) -> NaturalRoutedAttentionCompilation:
    """Recover a routed-attention prototype through Python dataflow objects."""
    recovered = recover_experimental_stablehlo_routed_attention_program(artifact, input_names=input_names)
    return compile_natural_routed_attention(
        recovered,
        runtime_inputs=runtime_inputs,
        schedule=schedule,
        config=config,
        padding_quantum=padding_quantum,
    )


def recover_experimental_stablehlo_projected_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
) -> RecoveredProjectedRoutedAttentionProgram:
    """Recover projected token routing into generic sparse-relation algebra."""
    graph = import_stablehlo(artifact, input_names=input_names)
    return recover_projected_routed_attention_program(graph)


def compile_experimental_stablehlo_projected_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    runtime_inputs: dict[str, np.ndarray],
    schedule: StreamingTileSchedule,
    config: RoutedAttentionPlanConfig,
    padding_quantum: int = 1,
) -> NaturalProjectedRoutedAttentionCompilation:
    """Recover a projected-selection prototype through Python dataflow objects."""
    recovered = recover_experimental_stablehlo_projected_routed_attention_program(artifact, input_names=input_names)
    return compile_natural_projected_routed_attention(
        recovered,
        runtime_inputs=runtime_inputs,
        schedule=schedule,
        config=config,
        padding_quantum=padding_quantum,
    )


def compile_experimental_stablehlo_dense_transformer_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RowScalePlacement = RowScalePlacement.CONSUMER_PROLOGUE,
) -> StableHLODenseCompilation:
    """Recover a bounded dense prototype with an exact Python recognizer."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_dense_transformer_region(
        stablehlo_graph,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
    )
    plan = compile_dense_transformer_region(
        recovered.graph,
        numerical_policy=numerical_policy,
        rms_scale_placement=rms_scale_placement,
    )
    compilation = StableHLODenseCompilation(
        plan=plan,
        provenance=FrontendProvenance(
            source_kind=FrontendSourceKind.STABLEHLO_ARTIFACT,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            source_operation_ids=recovered.source_operation_ids,
        ),
        status=FrontendCompilationStatus.EXPERIMENTAL_EXACT_RECOGNIZER,
    )
    validate_experimental_stablehlo_dense_compilation(compilation)
    return compilation


def validate_experimental_stablehlo_dense_compilation(compilation: StableHLODenseCompilation) -> None:
    """Validate evidence for the bounded experimental dense recognizer."""
    if compilation.provenance.source_kind is not FrontendSourceKind.STABLEHLO_ARTIFACT:
        raise SemanticErasureError("experimental StableHLO candidates must originate from a StableHLO artifact")
    if not compilation.provenance.source_operation_ids:
        raise SemanticErasureError("experimental StableHLO candidates must retain source-operation provenance")
    validate_plan_semantic_erasure(compilation.plan)


def require_architecturally_conforming_dense_compilation(compilation: StableHLODenseCompilation) -> None:
    """Reject the Python recovery path at the architecture-acceptance gate."""
    validate_experimental_stablehlo_dense_compilation(compilation)
    raise SemanticErasureError(
        "Python StableHLO recovery is experimental; current acceptance requires in-pipeline Shuttle MLIR provenance"
    )
