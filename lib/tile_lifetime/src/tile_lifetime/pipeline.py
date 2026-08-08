# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end compiler entry points."""

import numpy as np

from tile_lifetime.attention import compile_attention_region
from tile_lifetime.compiler import RMSScalePlacement, compile_region
from tile_lifetime.dense_region import compile_dense_transformer_region
from tile_lifetime.expert_parallel import ExpertParallelConfig, compile_expert_parallel_region
from tile_lifetime.expert_parallel_plan import ExpertParallelPlan
from tile_lifetime.ir import DType, ScaledDotProductAttentionOp
from tile_lifetime.moe_recovery import RecoveredMoERegion, recover_moe_region
from tile_lifetime.plan import NumericalPolicy, RegionPlan
from tile_lifetime.routed_attention_plan import RoutedAttentionPlanConfig
from tile_lifetime.routed_attention_recovery import (
    NaturalRoutedAttentionCompilation,
    RecoveredRoutedAttentionProgram,
    compile_natural_routed_attention,
    recover_routed_attention_program,
)
from tile_lifetime.semantic_recovery import (
    recover_attention_region,
    recover_dense_transformer_region,
    recover_rms_region,
)
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.streaming_attention import (
    StreamingAttentionProgram,
    StreamingTileSchedule,
    streaming_attention_from_semantic_operation,
)


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


def compile_stablehlo_rms_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Compile the first supported StableHLO region into an execution plan."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_rms_region(
        stablehlo_graph,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
        output_name=output_name,
    )
    return compile_region(
        recovered.graph,
        numerical_policy=numerical_policy,
        rms_scale_placement=rms_scale_placement,
    )


def compile_stablehlo_attention_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    numerical_policy: NumericalPolicy,
) -> RegionPlan:
    """Recover and lower exact causal GQA from portable StableHLO."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_attention_region(stablehlo_graph, output_name=output_name)
    return compile_attention_region(recovered.graph, numerical_policy=numerical_policy)


def compile_stablehlo_streaming_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    schedule: StreamingTileSchedule,
) -> StreamingAttentionProgram:
    """Recover ordinary StableHLO attention into generic Contract/Map/Fold."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_attention_region(stablehlo_graph, output_name=output_name)
    operations = recovered.graph.operations
    if len(operations) != 1 or not isinstance(operations[0], ScaledDotProductAttentionOp):
        raise ValueError("expected exactly one recovered semantic attention operation")
    return streaming_attention_from_semantic_operation(operations[0], schedule=schedule)


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


def compile_stablehlo_rms_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    rms_output_name: str,
    attention_output_name: str,
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Compile one StableHLO module containing RMS/GEMM and attention regions."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    recovered_rms = recover_rms_region(
        stablehlo_graph,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
        output_name=rms_output_name,
        output_index=0,
    )
    recovered_attention = recover_attention_region(
        stablehlo_graph,
        output_name=attention_output_name,
        output_index=1,
    )
    coda_plan = compile_region(
        recovered_rms.graph,
        numerical_policy=numerical_policy,
        rms_scale_placement=rms_scale_placement,
    )
    attention_plan = compile_attention_region(recovered_attention.graph, numerical_policy=numerical_policy)
    return RegionPlan(
        skeletons=(*coda_plan.skeletons, *attention_plan.skeletons),
        materializations=(*coda_plan.materializations, *attention_plan.materializations),
        rewrites=(*coda_plan.rewrites, *attention_plan.rewrites),
    )


def compile_stablehlo_dense_transformer_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_PROLOGUE,
) -> RegionPlan:
    """Recover and compile the connected dense debug region."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_dense_transformer_region(
        stablehlo_graph,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
    )
    return compile_dense_transformer_region(
        recovered.graph,
        numerical_policy=numerical_policy,
        rms_scale_placement=rms_scale_placement,
    )
