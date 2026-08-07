# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end compiler entry points."""

from tile_lifetime.attention import compile_attention_region
from tile_lifetime.compiler import RMSScalePlacement, compile_region
from tile_lifetime.dense_region import compile_dense_transformer_region
from tile_lifetime.ir import DType
from tile_lifetime.moe_recovery import RecoveredMoERegion, recover_moe_region
from tile_lifetime.plan import NumericalPolicy, RegionPlan
from tile_lifetime.semantic_recovery import (
    recover_attention_region,
    recover_dense_transformer_region,
    recover_rms_region,
)
from tile_lifetime.stablehlo_import import import_stablehlo


def recover_stablehlo_moe_region(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    gemm_accumulation_dtype: DType,
) -> RecoveredMoERegion:
    """Import and recover the bounded ordinary JAX MoE semantic region."""
    stablehlo_graph = import_stablehlo(artifact, input_names=input_names)
    return recover_moe_region(stablehlo_graph, gemm_accumulation_dtype=gemm_accumulation_dtype)


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


def compile_stablehlo_coda_fa3_program(
    artifact: bytes,
    *,
    input_names: tuple[str, ...],
    rms_output_name: str,
    attention_output_name: str,
    gemm_accumulation_dtype: DType,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Compile one StableHLO module containing the initial CODA and FA3 regions."""
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
