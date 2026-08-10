# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical named-operation planners retained for reference comparisons.

These entrypoints recover ordinary StableHLO, but schedule named semantic
operations through prototype physical planners. They are excluded from the
current Shuttle frontend because their accepted paths can select an opaque
workload kernel.
"""

from shuttle.experimental.stablehlo_import import import_stablehlo
from shuttle.ir import DType
from tile_lifetime.attention import compile_reference_attention_region
from tile_lifetime.compiler import RowScalePlacement, compile_reference_region
from tile_lifetime.expert_parallel import ExpertParallelConfig, compile_expert_parallel_region
from tile_lifetime.expert_parallel_plan import ExpertParallelPlan
from tile_lifetime.moe_recovery import RecoveredMoERegion, recover_moe_region
from tile_lifetime.plan import NumericalPolicy, RegionPlan
from tile_lifetime.reference_semantic_recovery import recover_reference_rms_region
from tile_lifetime.semantic_recovery import recover_attention_region


def recover_reference_stablehlo_moe_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
) -> RecoveredMoERegion:
    """Recover the bounded named MoE TensorGraph used by the reference planner."""
    graph = import_stablehlo(artifact, input_names=input_names)
    return recover_moe_region(graph, gemm_accumulation_dtype=gemm_accumulation_dtype)


def compile_reference_stablehlo_expert_parallel_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
    config: ExpertParallelConfig,
    numerical_policy: NumericalPolicy,
) -> ExpertParallelPlan:
    """Run the historical named MoE planner for comparison tests."""
    recovered = recover_reference_stablehlo_moe_region(
        artifact,
        input_names=input_names,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
    )
    return compile_expert_parallel_region(
        recovered.graph,
        config=config,
        numerical_policy=numerical_policy,
    )


def compile_reference_stablehlo_rms_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RowScalePlacement = RowScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Run the historical named RMS/GEMM planner for reference tests."""
    graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_reference_rms_region(
        graph,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
        output_name=output_name,
    )
    return compile_reference_region(
        recovered.graph,
        numerical_policy=numerical_policy,
        rms_scale_placement=rms_scale_placement,
    )


def compile_reference_stablehlo_attention_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    output_name: str,
    numerical_policy: NumericalPolicy,
) -> RegionPlan:
    """Run the historical named attention planner for reference tests."""
    graph = import_stablehlo(artifact, input_names=input_names)
    recovered = recover_attention_region(graph, output_name=output_name)
    return compile_reference_attention_region(recovered.graph, numerical_policy=numerical_policy)


def compile_reference_stablehlo_rms_attention_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    rms_output_name: str,
    attention_output_name: str,
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RowScalePlacement = RowScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Run the historical combined named RMS and attention planners."""
    graph = import_stablehlo(artifact, input_names=input_names)
    recovered_rms = recover_reference_rms_region(
        graph,
        gemm_accumulation_dtype=gemm_accumulation_dtype,
        output_name=rms_output_name,
        output_index=0,
    )
    recovered_attention = recover_attention_region(
        graph,
        output_name=attention_output_name,
        output_index=1,
    )
    rms_plan = compile_reference_region(
        recovered_rms.graph,
        numerical_policy=numerical_policy,
        rms_scale_placement=rms_scale_placement,
    )
    attention_plan = compile_reference_attention_region(
        recovered_attention.graph,
        numerical_policy=numerical_policy,
    )
    return RegionPlan(
        skeletons=(*rms_plan.skeletons, *attention_plan.skeletons),
        materializations=(*rms_plan.materializations, *attention_plan.materializations),
        rewrites=(*rms_plan.rewrites, *attention_plan.rewrites),
    )
