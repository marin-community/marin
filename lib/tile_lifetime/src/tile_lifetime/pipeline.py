# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end compiler entry points."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.compiler import RMSScalePlacement
from tile_lifetime.dense_region import compile_dense_transformer_region
from tile_lifetime.expert_parallel import ExpertParallelConfig, compile_expert_parallel_region
from tile_lifetime.expert_parallel_plan import ExpertParallelPlan
from tile_lifetime.ir import DType, ScaledDotProductAttentionOp
from tile_lifetime.moe_recovery import RecoveredMoERegion, recover_moe_region
from tile_lifetime.msa_recovery import (
    NaturalProjectedRoutedAttentionCompilation,
    RecoveredProjectedRoutedAttentionProgram,
    compile_natural_projected_routed_attention,
    recover_projected_routed_attention_program,
)
from tile_lifetime.plan import NumericalPolicy, RegionPlan, SemanticErasureReport, SemanticLoweringStep
from tile_lifetime.routed_attention_plan import RoutedAttentionPlanConfig
from tile_lifetime.routed_attention_recovery import (
    NaturalRoutedAttentionCompilation,
    RecoveredRoutedAttentionProgram,
    compile_natural_routed_attention,
    recover_routed_attention_program,
)
from tile_lifetime.semantic_erasure import (
    ErasedTensorProgram,
    SemanticErasureError,
    build_tensor_erasure_report,
    validate_erased_tensor_program,
    validate_plan_semantic_erasure,
)
from tile_lifetime.semantic_recovery import (
    recover_attention_region,
    recover_dense_transformer_region,
)
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.streaming_attention import (
    StreamingAttentionProgram,
    StreamingTileSchedule,
    streaming_attention_from_semantic_operation,
)


class FrontendSourceKind(StrEnum):
    """Verified source boundary for a current Shuttle compilation."""

    STABLEHLO_ARTIFACT = "stablehlo_artifact"
    HAND_AUTHORED_SEMANTIC_IR = "hand_authored_semantic_ir"


class FrontendCompilationStatus(StrEnum):
    """Whether frontend recovery is eligible for a clean-synthesis claim."""

    EXPERIMENTAL_EXACT_RECOGNIZER = "experimental_exact_recognizer"
    GENERIC_HLO_DATAFLOW = "generic_hlo_dataflow"


@dataclass(frozen=True)
class FrontendProvenance:
    """Source evidence carried across semantic erasure."""

    source_kind: FrontendSourceKind
    artifact_sha256: str
    source_operation_ids: tuple[int, ...]


@dataclass(frozen=True)
class StableHLOStreamingAttentionCompilation:
    """Generic streaming program with frontend and erasure evidence."""

    program: StreamingAttentionProgram
    provenance: FrontendProvenance
    semantic_erasure_report: SemanticErasureReport


@dataclass(frozen=True)
class StableHLODenseCompilation:
    """Dense plan with source evidence and an explicit acceptance status."""

    plan: RegionPlan
    provenance: FrontendProvenance
    status: FrontendCompilationStatus


def recover_stablehlo_moe_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
) -> RecoveredMoERegion:
    """Import and recover the bounded ordinary JAX MoE semantic region."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    return recover_moe_region(stablehlo_graph, gemm_accumulation_dtype=gemm_accumulation_dtype)


def compile_stablehlo_expert_parallel_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
    config: ExpertParallelConfig,
    numerical_policy: NumericalPolicy,
) -> ExpertParallelPlan:
    """Recover ordinary StableHLO MoE math and lower it to generic EP stages."""
    recovered = recover_stablehlo_moe_region(
        artifact,
        input_names=input_names,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
    )
    return compile_expert_parallel_region(
        recovered.graph,
        config=config,
        numerical_policy=numerical_policy,
    )


def compile_stablehlo_streaming_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    schedule: StreamingTileSchedule,
) -> StableHLOStreamingAttentionCompilation:
    """Recover ordinary StableHLO attention into generic Contract/Map/Fold."""
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
    compilation = StableHLOStreamingAttentionCompilation(
        program=program,
        provenance=FrontendProvenance(
            source_kind=FrontendSourceKind.STABLEHLO_ARTIFACT,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            source_operation_ids=recovered.source_operation_ids,
        ),
        semantic_erasure_report=report,
    )
    validate_stablehlo_streaming_attention_compilation(compilation)
    return compilation


def validate_stablehlo_streaming_attention_compilation(
    compilation: StableHLOStreamingAttentionCompilation,
) -> None:
    """Reject hand-authored or named scheduling inputs on the current path."""
    if compilation.provenance.source_kind is not FrontendSourceKind.STABLEHLO_ARTIFACT:
        raise SemanticErasureError("current frontend candidates must originate from a StableHLO artifact")
    erased = ErasedTensorProgram(compilation.program.source, compilation.semantic_erasure_report)
    validate_erased_tensor_program(erased)


def recover_stablehlo_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
) -> RecoveredRoutedAttentionProgram:
    """Recover natural selected-attention math and erase names into generic algebra."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    return recover_routed_attention_program(stablehlo_graph)


def compile_stablehlo_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    runtime_inputs: dict[str, np.ndarray],
    schedule: StreamingTileSchedule,
    config: RoutedAttentionPlanConfig,
    padding_quantum: int = 1,
) -> NaturalRoutedAttentionCompilation:
    """Compile ordinary StableHLO through runtime RelationPlan and streaming skeletons."""
    recovered = recover_stablehlo_routed_attention_program(artifact, input_names=input_names)
    return compile_natural_routed_attention(
        recovered,
        runtime_inputs=runtime_inputs,
        schedule=schedule,
        config=config,
        padding_quantum=padding_quantum,
    )


def recover_stablehlo_projected_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
) -> RecoveredProjectedRoutedAttentionProgram:
    """Recover projected token routing into generic sparse-relation algebra."""
    graph = import_stablehlo(artifact, input_names=input_names)
    return recover_projected_routed_attention_program(graph)


def compile_stablehlo_projected_routed_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    runtime_inputs: dict[str, np.ndarray],
    schedule: StreamingTileSchedule,
    config: RoutedAttentionPlanConfig,
    padding_quantum: int = 1,
) -> NaturalProjectedRoutedAttentionCompilation:
    """Compile projected Selection through RelationPlan and schedule synthesis."""
    recovered = recover_stablehlo_projected_routed_attention_program(artifact, input_names=input_names)
    return compile_natural_projected_routed_attention(
        recovered,
        runtime_inputs=runtime_inputs,
        schedule=schedule,
        config=config,
        padding_quantum=padding_quantum,
    )


def compile_stablehlo_dense_transformer_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_PROLOGUE,
) -> StableHLODenseCompilation:
    """Recover, erase, and compile the connected dense StableHLO region."""
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
    validate_stablehlo_dense_compilation(compilation)
    return compilation


def validate_stablehlo_dense_compilation(compilation: StableHLODenseCompilation) -> None:
    """Validate evidence for the bounded experimental dense recognizer."""
    if compilation.provenance.source_kind is not FrontendSourceKind.STABLEHLO_ARTIFACT:
        raise SemanticErasureError("current frontend candidates must originate from a StableHLO artifact")
    if not compilation.provenance.source_operation_ids:
        raise SemanticErasureError("current frontend candidates must retain source-operation provenance")
    validate_plan_semantic_erasure(compilation.plan)


def require_current_stablehlo_dense_compilation(compilation: StableHLODenseCompilation) -> None:
    """Fail closed until dense recovery uses the shared generic HLO importer."""
    validate_stablehlo_dense_compilation(compilation)
    if compilation.status is not FrontendCompilationStatus.GENERIC_HLO_DATAFLOW:
        raise SemanticErasureError(
            "exact named dense-region reconstruction is experimental; current acceptance requires generic HLO dataflow"
        )
