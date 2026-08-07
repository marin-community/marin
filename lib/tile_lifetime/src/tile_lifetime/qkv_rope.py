# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan a combined QKV projection and RoPE boundary into streaming attention."""

from dataclasses import dataclass

from tile_lifetime.attention import compile_attention_region, select_hopper_attention_config
from tile_lifetime.ir import QKVProjectionOp, RoPEOp, ScaledDotProductAttentionOp, TensorGraph
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    GemmSkeleton,
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    NumericalPolicy,
    RegionPlan,
    RewriteExplanation,
    TransformSkeleton,
)


@dataclass(frozen=True)
class _QKVRoPEAttentionRegion:
    projection: QKVProjectionOp
    rope: RoPEOp
    attention: ScaledDotProductAttentionOp


def compile_qkv_rope_attention_region(
    graph: TensorGraph,
    *,
    numerical_policy: NumericalPolicy,
) -> RegionPlan:
    """Compile QKV projection, adjacent-pair RoPE, and exact attention."""
    region, rejection_reasons = _find_region(graph)
    if region is None:
        raise ValueError("QKV/RoPE/attention region is not legal: " + "; ".join(rejection_reasons))
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        return _materialized_plan(graph, region)
    return _fused_plan(graph, region)


def _find_region(graph: TensorGraph) -> tuple[_QKVRoPEAttentionRegion | None, tuple[str, ...]]:
    projections = tuple(operation for operation in graph.operations if isinstance(operation, QKVProjectionOp))
    ropes = tuple(operation for operation in graph.operations if isinstance(operation, RoPEOp))
    attentions = tuple(operation for operation in graph.operations if isinstance(operation, ScaledDotProductAttentionOp))
    if len(projections) != 1 or len(ropes) != 1 or len(attentions) != 1:
        return None, (
            f"expected one QKV projection, RoPE, and attention; found {len(projections)}, {len(ropes)}, "
            f"and {len(attentions)}",
        )

    projection = projections[0]
    rope = ropes[0]
    attention = attentions[0]
    reasons: list[str] = []
    if rope.query != projection.query or rope.key != projection.key:
        reasons.append("RoPE does not consume this QKV projection's Q and K outputs")
    if attention.query != rope.output or attention.key != rope.key_output:
        reasons.append("attention does not consume the rotated Q and K outputs")
    if attention.value != projection.value:
        reasons.append("attention does not consume this QKV projection's V output")

    required_consumers = (
        (projection.query, (rope,)),
        (projection.key, (rope,)),
        (projection.value, (attention,)),
        (rope.output, (attention,)),
        (rope.key_output, (attention,)),
    )
    for value, expected in required_consumers:
        consumers = graph.consumers(value)
        if consumers != expected:
            reasons.append(f"{value.name} has consumers {[type(item).__name__ for item in consumers]}")

    if reasons:
        return None, tuple(reasons)
    return _QKVRoPEAttentionRegion(projection=projection, rope=rope, attention=attention), ()


def _fused_plan(graph: TensorGraph, region: _QKVRoPEAttentionRegion) -> RegionPlan:
    projection = region.projection
    rope = region.rope
    attention = region.attention
    attention_plan = compile_attention_region(graph, numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER)
    config = select_hopper_attention_config(attention)
    batch, sequence, hidden = projection.input.shape

    qkv_gemm = GemmSkeleton(
        name=f"{projection.output.name}.qkv_projection_rope",
        input=projection.input.name,
        weight=projection.weight.name,
        output=projection.output.name,
        shape=(batch * sequence, projection.weight.shape[1], hidden),
        accumulation_dtype=projection.accumulation_dtype,
        backend="coda_cute_h100",
        input_layout="bsh_contiguous",
        output_layout=config.input_layout,
        epilogue=(
            Attachment(
                operation="partition_qkv_segment_views_bshd",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(projection.output.name,),
                outputs=(projection.query.name, projection.key.name, projection.value.name),
            ),
            Attachment(
                operation="pairwise_rope_q",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(projection.query.name, rope.sine.name, rope.cosine.name),
                outputs=(rope.output.name,),
            ),
            Attachment(
                operation="pairwise_rope_k",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(projection.key.name, rope.sine.name, rope.cosine.name),
                outputs=(rope.key_output.name,),
            ),
        ),
    )
    materializations = (
        MaterializationRecord(
            value=projection.output.name,
            shape=projection.output.shape,
            dtype=projection.output.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="combined QKV accumulators are partitioned before the epilogue stores separate FA3 inputs",
        ),
        MaterializationRecord(
            value=projection.query.name,
            shape=projection.query.shape,
            dtype=projection.query.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="unrotated Q exists only within the projection epilogue",
        ),
        MaterializationRecord(
            value=projection.key.name,
            shape=projection.key.shape,
            dtype=projection.key.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="unrotated K exists only within the projection epilogue",
        ),
        MaterializationRecord(
            value=rope.output.name,
            shape=rope.output.shape,
            dtype=rope.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="rotated Q is a BSHD segment view with a contiguous head dimension accepted by FA3",
        ),
        MaterializationRecord(
            value=rope.key_output.name,
            shape=rope.key_output.shape,
            dtype=rope.key_output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="rotated K is a BSHD segment view with a contiguous head dimension accepted by FA3",
        ),
        MaterializationRecord(
            value=projection.value.name,
            shape=projection.value.shape,
            dtype=projection.value.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="V is a BSHD segment view with a contiguous head dimension accepted by FA3",
        ),
        *attention_plan.materializations,
    )
    rewrite = RewriteExplanation(
        name="fuse_rope_into_qkv_projection_epilogue",
        applied=True,
        original_fragment=(
            "combined QKV projection",
            "partition Q, K, and V",
            "apply adjacent-pair RoPE to Q and K",
            "exact scaled dot-product attention",
        ),
        transformed_fragment=(
            "run one QKV GEMM mainloop",
            "partition accumulator regions and rotate Q/K in the epilogue",
            "store packed QKV with BSHD segment views accepted directly by FA3",
            "consume them with the selected official FA3 streaming skeleton",
        ),
        semantic_properties=(
            "QKV partitioning is an indexed epilogue extraction",
            "RoPE is pairwise-local within each Q/K head",
            "the attention boundary requires a contiguous head dimension but permits strided leading dimensions",
        ),
        legality_checks=(
            "Q and K have no consumer before RoPE",
            "rotated Q, rotated K, and V are consumed only by attention",
            "rotary pairs are adjacent and the rotary dimension is even",
            "the projection output layout equals the selected FA3 input layout",
        ),
        estimated_benefit="eliminates unrotated Q/K writes and any QKV-to-attention layout-conversion kernel",
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect=(
            "RoPE consumes FP32 projection accumulators before the BF16 store, changing rounding relative to "
            "materialized unrotated BF16 Q and K"
        ),
    )
    return RegionPlan(
        skeletons=(qkv_gemm, *attention_plan.skeletons),
        materializations=materializations,
        rewrites=(rewrite, *attention_plan.rewrites),
    )


def _materialized_plan(graph: TensorGraph, region: _QKVRoPEAttentionRegion) -> RegionPlan:
    projection = region.projection
    rope = region.rope
    attention_plan = compile_attention_region(graph, numerical_policy=NumericalPolicy.BITWISE_EXACT)
    batch, sequence, hidden = projection.input.shape
    qkv_gemm = GemmSkeleton(
        name=f"{projection.output.name}.materialized_qkv_projection",
        input=projection.input.name,
        weight=projection.weight.name,
        output=projection.output.name,
        shape=(batch * sequence, projection.weight.shape[1], hidden),
        accumulation_dtype=projection.accumulation_dtype,
    )
    transforms = (
        TransformSkeleton(
            name=f"materialized_{rope.output.name}",
            operation="pairwise_rope_q",
            inputs=(projection.query.name, rope.sine.name, rope.cosine.name),
            output=rope.output.name,
        ),
        TransformSkeleton(
            name=f"materialized_{rope.key_output.name}",
            operation="pairwise_rope_k",
            inputs=(projection.key.name, rope.sine.name, rope.cosine.name),
            output=rope.key_output.name,
        ),
    )
    materialized_values = (
        projection.output,
        projection.query,
        projection.key,
        projection.value,
        rope.output,
        rope.key_output,
    )
    return RegionPlan(
        skeletons=(qkv_gemm, *transforms, *attention_plan.skeletons),
        materializations=tuple(
            MaterializationRecord(
                value=value.name,
                shape=value.shape,
                dtype=value.dtype,
                disposition=MaterializationDisposition.MATERIALIZE,
                reason="bitwise-exact policy retains the source QKV and RoPE boundaries",
            )
            for value in materialized_values
        )
        + attention_plan.materializations,
        rewrites=(
            RewriteExplanation(
                name="fuse_rope_into_qkv_projection_epilogue",
                applied=False,
                original_fragment=("QKV projection", "materialized Q/K", "RoPE", "attention"),
                transformed_fragment=(),
                semantic_properties=("pairwise-local RoPE",),
                legality_checks=("source floating-point operation order must be preserved",),
                estimated_benefit="none; retained materialized boundaries",
                numerical_equivalence=NumericalEquivalence.BITWISE_EXACT,
                numerical_effect="none",
                rejection_reasons=("bitwise-exact policy rejects pre-store RoPE",),
            ),
            *attention_plan.rewrites,
        ),
    )
