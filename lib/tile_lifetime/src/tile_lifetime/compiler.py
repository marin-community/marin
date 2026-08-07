# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic region planning for the first tile-lifetime prototype."""

from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.ir import (
    DType,
    LinearOp,
    ResidualAddOp,
    RMSNormOp,
    ScaledDotProductAttentionOp,
    SemanticOp,
    TensorGraph,
)
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    GemmSkeleton,
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    NumericalPolicy,
    ReductionSkeleton,
    RegionPlan,
    RewriteExplanation,
    TransformSkeleton,
)


@dataclass(frozen=True)
class _RMSRegion:
    producer_gemm: LinearOp
    residual_add: ResidualAddOp
    rms_norm: RMSNormOp
    consumer_gemm: LinearOp
    residual: str


class RMSScalePlacement(StrEnum):
    """Physical placement of the inverse-RMS row scale around the consumer GEMM."""

    CONSUMER_EPILOGUE = "consumer_epilogue"
    CONSUMER_PROLOGUE = "consumer_prologue"


def compile_region(
    graph: TensorGraph,
    *,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Compile a semantic graph into a bounded tile-lifetime execution plan."""
    candidate, rejection_reasons = _find_rms_region(graph, numerical_policy=numerical_policy)
    if candidate is None:
        return _materialized_fallback(graph, rejection_reasons=rejection_reasons)
    return _rms_plan(candidate, scale_placement=rms_scale_placement)


def _find_rms_region(
    graph: TensorGraph, *, numerical_policy: NumericalPolicy
) -> tuple[_RMSRegion | None, tuple[str, ...]]:
    rms_operations = tuple(operation for operation in graph.operations if isinstance(operation, RMSNormOp))
    if len(rms_operations) != 1:
        return None, (f"expected one RMSNorm operation, found {len(rms_operations)}",)

    rms_norm = rms_operations[0]
    reasons: list[str] = []
    residual_add = graph.producer(rms_norm.input)
    if not isinstance(residual_add, ResidualAddOp):
        reasons.append("RMSNorm input is not produced by a residual addition")
        return None, tuple(reasons)

    producer_gemm, residual = _producer_gemm_and_residual(graph, residual_add)
    if producer_gemm is None or residual is None:
        reasons.append("residual addition does not combine one GEMM output with one residual tensor")

    consumers = graph.consumers(rms_norm.output)
    if len(consumers) != 1:
        reasons.append(f"normalized activation has {len(consumers)} consumers; delayed scaling requires exactly one")
        consumer_gemm = None
    else:
        consumer_gemm = consumers[0]
        if not isinstance(consumer_gemm, LinearOp) or consumer_gemm.input != rms_norm.output:
            reasons.append("normalized activation's only consumer is not a right-multiplication GEMM")
            consumer_gemm = None

    if rms_norm.axis != len(rms_norm.input.shape) - 1:
        reasons.append("RMSNorm reduction axis is not the consumer GEMM reduction dimension")
    if rms_norm.reduction_dtype not in {DType.FP32, DType.FP64}:
        reasons.append("RMSNorm partial reduction must accumulate in FP32 or FP64")
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        reasons.append("delayed scaling reorders finite-precision rounding under the bitwise-exact policy")

    if reasons or producer_gemm is None or residual is None or consumer_gemm is None:
        return None, tuple(reasons)
    return _RMSRegion(producer_gemm, residual_add, rms_norm, consumer_gemm, residual), ()


def _producer_gemm_and_residual(graph: TensorGraph, residual_add: ResidualAddOp) -> tuple[LinearOp | None, str | None]:
    left_producer = graph.producer(residual_add.left)
    right_producer = graph.producer(residual_add.right)
    if isinstance(left_producer, LinearOp) and right_producer is None:
        return left_producer, residual_add.right.name
    if isinstance(right_producer, LinearOp) and left_producer is None:
        return right_producer, residual_add.left.name
    return None, None


def _rms_plan(region: _RMSRegion, *, scale_placement: RMSScalePlacement) -> RegionPlan:
    first = region.producer_gemm
    norm = region.rms_norm
    second = region.consumer_gemm
    scaled_residual = f"{region.residual_add.output.name}_times_{norm.gamma.name}"
    partials = f"{region.residual_add.output.name}_rms_partials"
    inverse_rms = f"{region.residual_add.output.name}_inverse_rms"
    normalized_for_gemm = f"{scaled_residual}_times_{inverse_rms}"

    gemm_0 = GemmSkeleton(
        name="gemm_residual_rms_partials",
        input=first.input.name,
        weight=first.weight.name,
        output=scaled_residual,
        shape=(first.output.shape[0], first.output.shape[1], first.input.shape[1]),
        accumulation_dtype=first.accumulation_dtype,
        backend="coda_cute_h100",
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        epilogue=(
            Attachment(
                operation="residual_add",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(first.output.name, region.residual),
                outputs=(region.residual_add.output.name,),
            ),
            Attachment(
                operation="multiply_gamma",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(region.residual_add.output.name, norm.gamma.name),
                outputs=(scaled_residual,),
            ),
            Attachment(
                operation="partial_sum_square",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(region.residual_add.output.name,),
                outputs=(partials,),
            ),
        ),
    )
    reduction = ReductionSkeleton(
        name="combine_rms_partials",
        input=partials,
        output=inverse_rms,
        operator=f"rsqrt(sum / {norm.input.shape[norm.axis]} + {norm.epsilon})",
        reduction_dtype=norm.reduction_dtype,
    )
    consumer_prologue = ()
    consumer_epilogue = ()
    consumer_name = "gemm_delayed_rms_scale"
    prologue_cluster_shape = (1, 2 if second.output.shape[1] >= 16_384 else 1, 1)
    if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE:
        consumer_name = "gemm_prologue_rms_scale"
        consumer_prologue = (
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=(scaled_residual, inverse_rms),
                outputs=(normalized_for_gemm,),
            ),
        )
    else:
        consumer_epilogue = (
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(second.output.name, inverse_rms),
                outputs=(second.output.name,),
            ),
        )

    gemm_1 = GemmSkeleton(
        name=consumer_name,
        input=scaled_residual,
        weight=second.weight.name,
        output=second.output.name,
        shape=(second.output.shape[0], second.output.shape[1], second.input.shape[1]),
        accumulation_dtype=second.accumulation_dtype,
        backend=(
            "quack_sm90_fp32_a_transform" if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE else "coda_cute_h100"
        ),
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=(128, 256, 64) if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE else None,
        cluster_shape=prologue_cluster_shape if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE else None,
        pingpong=False if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE else None,
        prologue=consumer_prologue,
        epilogue=consumer_epilogue,
    )

    materializations = (
        MaterializationRecord(
            value=first.output.name,
            shape=first.output.shape,
            dtype=first.output.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="consumed by residual and RMS-partial attachments before the first GEMM tile is stored",
        ),
        MaterializationRecord(
            value=region.residual_add.output.name,
            shape=region.residual_add.output.shape,
            dtype=region.residual_add.output.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="replaced by the gamma-scaled representation consumed by the next GEMM",
        ),
        MaterializationRecord(
            value=norm.output.name,
            shape=norm.output.shape,
            dtype=norm.output.dtype,
            disposition=(
                MaterializationDisposition.PROLOGUE_ONLY
                if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE
                else MaterializationDisposition.EPILOGUE_ONLY
            ),
            reason=(
                "represented by an on-chip BF16 row-scale transform before the consumer WGMMA"
                if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE
                else "inverse-RMS scaling is delayed through the following right-multiplication"
            ),
        ),
        MaterializationRecord(
            value=scaled_residual,
            shape=norm.output.shape,
            dtype=norm.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="cross-skeleton activation consumed by the second GEMM mainloop",
        ),
        MaterializationRecord(
            value=partials,
            shape=(norm.output.shape[0],),
            dtype=norm.reduction_dtype,
            disposition=MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            reason="small per-row RMS statistic buffer",
        ),
        MaterializationRecord(
            value=second.output.name,
            shape=second.output.shape,
            dtype=second.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="region output",
        ),
    )
    if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE:
        rewrite_name = "scale_rms_in_consumer_gemm_prologue"
        transformed_consumer = (
            f"{normalized_for_gemm} = bf16({scaled_residual} * {inverse_rms}) inside the GEMM prologue",
            f"{second.output.name} = {normalized_for_gemm} @ {second.weight.name}",
        )
        semantic_properties = (
            "residual addition is tile-local",
            "gamma multiplication is tile-local",
            "sum of squares decomposes into tile partials",
            "inverse RMS is a row scalar available before the consumer WGMMA",
        )
        estimated_benefit = (
            "eliminates the materialized normalized activation while retaining source-like pre-GEMM scaling; "
            "the input transform adds shared-memory traffic, synchronization, and repeated scale work per N tile"
        )
        numerical_effect = (
            "places inverse-RMS scaling and BF16 conversion before WGMMA, matching a source GEMM that consumes "
            "a BF16 normalized activation more closely than delayed epilogue scaling; storing u*gamma before the "
            "prologue can still introduce an additional BF16 rounding"
        )
    else:
        rewrite_name = "delay_rms_row_scale_through_gemm"
        transformed_consumer = (
            f"{second.output.name} = scale_row({scaled_residual} @ {second.weight.name}, {inverse_rms})",
        )
        semantic_properties = (
            "residual addition is tile-local",
            "gamma multiplication is tile-local",
            "sum of squares decomposes into tile partials",
            "inverse RMS is a row scalar",
            "row scaling commutes with right multiplication over real arithmetic",
        )
        estimated_benefit = "eliminates the materialized normalized activation and its full-tensor normalization kernel"
        numerical_effect = (
            "moves inverse-RMS scaling after the second GEMM, changing where BF16 rounding occurs; "
            "the generated plan requires differential validation against the declared reference"
        )

    explanation = RewriteExplanation(
        name=rewrite_name,
        applied=True,
        original_fragment=(
            f"{first.output.name} = {first.input.name} @ {first.weight.name}",
            f"{region.residual_add.output.name} = {first.output.name} + {region.residual}",
            f"{norm.output.name} = rms_norm({region.residual_add.output.name}, {norm.gamma.name})",
            f"{second.output.name} = {norm.output.name} @ {second.weight.name}",
        ),
        transformed_fragment=(
            f"{scaled_residual}, {partials} = gemm_epilogue_residual_rms_partials(...)",
            f"{inverse_rms} = combine_rms_partials({partials})",
            *transformed_consumer,
        ),
        semantic_properties=semantic_properties,
        legality_checks=(
            "RMS reduction covers the GEMM reduction dimension",
            "normalized activation has exactly one consumer",
            "consumer is a right-multiplication GEMM",
            "RMS partials accumulate in FP32 or FP64",
            "numerical policy permits reordered finite-precision rounding",
        ),
        estimated_benefit=estimated_benefit,
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect=numerical_effect,
    )
    return RegionPlan(skeletons=(gemm_0, reduction, gemm_1), materializations=materializations, rewrites=(explanation,))


def _materialized_fallback(graph: TensorGraph, *, rejection_reasons: tuple[str, ...]) -> RegionPlan:
    skeletons = tuple(_fallback_skeleton(operation) for operation in graph.operations)
    materializations = tuple(
        MaterializationRecord(
            value=operation.output.name,
            shape=operation.output.shape,
            dtype=operation.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="standalone fallback operation boundary",
        )
        for operation in graph.operations
    )
    explanation = RewriteExplanation(
        name="delay_rms_row_scale_through_gemm",
        applied=False,
        original_fragment=tuple(type(operation).__name__ for operation in graph.operations),
        transformed_fragment=(),
        semantic_properties=(),
        legality_checks=(),
        estimated_benefit="none; the materialized reference path is retained",
        numerical_equivalence=NumericalEquivalence.BITWISE_EXACT,
        numerical_effect="none",
        rejection_reasons=rejection_reasons,
    )
    return RegionPlan(skeletons=skeletons, materializations=materializations, rewrites=(explanation,))


def _fallback_skeleton(operation: SemanticOp) -> GemmSkeleton | TransformSkeleton:
    if isinstance(operation, LinearOp):
        return GemmSkeleton(
            name=f"materialized_{operation.output.name}",
            input=operation.input.name,
            weight=operation.weight.name,
            output=operation.output.name,
            shape=(operation.output.shape[0], operation.output.shape[1], operation.input.shape[1]),
            accumulation_dtype=operation.accumulation_dtype,
        )
    if isinstance(operation, ResidualAddOp):
        return TransformSkeleton(
            name=f"materialized_{operation.output.name}",
            operation="residual_add",
            inputs=(operation.left.name, operation.right.name),
            output=operation.output.name,
        )
    if isinstance(operation, RMSNormOp):
        return TransformSkeleton(
            name=f"materialized_{operation.output.name}",
            operation="rms_norm",
            inputs=(operation.input.name, operation.gamma.name),
            output=operation.output.name,
        )
    assert isinstance(operation, ScaledDotProductAttentionOp)
    return TransformSkeleton(
        name=f"materialized_{operation.output.name}",
        operation="scaled_dot_product_attention",
        inputs=(operation.query.name, operation.key.name, operation.value.name),
        output=operation.output.name,
    )
