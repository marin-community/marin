# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower semantic attention to a bounded Hopper streaming template family."""

from dataclasses import dataclass

from tile_lifetime.ir import DType, ScaledDotProductAttentionOp, TensorGraph
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    NumericalPolicy,
    RegionPlan,
    RewriteExplanation,
    StreamingAttentionSkeleton,
    TransformSkeleton,
)


@dataclass(frozen=True)
class AttentionPhysicalConfig:
    """One supported physical configuration in the official FA3 template family."""

    backend: str
    query_block_size: int
    key_value_block_size: int
    pipeline_stages: int
    producer_threads: int
    consumer_threads: int
    pack_gqa: bool
    mma_pv_is_rs: bool
    intra_warpgroup_overlap: bool
    persistent_scheduler: bool
    register_estimate: int | None
    input_layout: str = "fa3_bshd_last_dimension_contiguous"
    output_layout: str = "bshd_contiguous"


def select_hopper_attention_config(operation: ScaledDotProductAttentionOp) -> AttentionPhysicalConfig:
    """Select a conservative initial H100 configuration for a supported head dimension."""
    if operation.head_dimension not in (64, 128):
        raise ValueError(f"FA3-style template does not support head dimension {operation.head_dimension}")
    if operation.head_dimension == 64:
        query_block_size = 192
        key_value_block_size = 128 if operation.causal else 192
    else:
        query_block_size = 128
        key_value_block_size = 128 if operation.causal else 176
    mma_pv_is_rs = operation.causal
    return AttentionPhysicalConfig(
        backend="official_flashattention_3_hopper",
        query_block_size=query_block_size,
        key_value_block_size=key_value_block_size,
        pipeline_stages=2,
        producer_threads=32,
        consumer_threads=(query_block_size // 64) * 128,
        pack_gqa=operation.query_heads != operation.key_value_heads,
        mma_pv_is_rs=mma_pv_is_rs,
        intra_warpgroup_overlap=True,
        persistent_scheduler=True,
        register_estimate=168 if operation.head_dimension == 128 else None,
    )


def compile_attention_region(graph: TensorGraph, *, numerical_policy: NumericalPolicy) -> RegionPlan:
    """Compile the graph's single attention operation into a streaming or fallback plan."""
    operations = tuple(operation for operation in graph.operations if isinstance(operation, ScaledDotProductAttentionOp))
    if len(operations) != 1:
        raise ValueError(f"expected one semantic attention operation, found {len(operations)}")
    operation = operations[0]
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        return _materialized_attention_plan(operation)

    config = select_hopper_attention_config(operation)
    batch, query_length, query_heads, value_dimension = operation.output.shape
    key_length = operation.key.shape[1]
    score_shape = (batch, query_heads, query_length, key_length)
    statistic_shape = (batch, query_heads, query_length)
    state_prefix = f"{operation.output.name}.online"
    online_max = f"{state_prefix}.max"
    online_sum = f"{state_prefix}.sum"
    online_output = f"{state_prefix}.output"
    scores = f"{operation.output.name}.scores"
    probabilities = f"{operation.output.name}.probabilities"

    skeleton = StreamingAttentionSkeleton(
        name=f"{operation.output.name}.streaming_attention",
        query=operation.query.name,
        key=operation.key.name,
        value=operation.value.name,
        output=operation.output.name,
        score_value=scores,
        probability_value=probabilities,
        query_block_size=config.query_block_size,
        key_value_block_size=config.key_value_block_size,
        head_dimension=operation.head_dimension,
        query_heads=operation.query_heads,
        key_value_heads=operation.key_value_heads,
        causal=operation.causal,
        scale=operation.scale,
        backend=config.backend,
        input_layout=config.input_layout,
        output_layout=config.output_layout,
        pipeline_stages=config.pipeline_stages,
        producer_threads=config.producer_threads,
        consumer_threads=config.consumer_threads,
        pack_gqa=config.pack_gqa,
        mma_pv_is_rs=config.mma_pv_is_rs,
        intra_warpgroup_overlap=config.intra_warpgroup_overlap,
        persistent_scheduler=config.persistent_scheduler,
        register_estimate=config.register_estimate,
        online_state=(online_max, online_sum, online_output),
        attachments=(
            Attachment(
                operation="scale_and_causal_mask",
                site=AttachmentSite.ATTENTION_SCORE_TRANSFORM,
                inputs=(operation.query.name, operation.key.name),
                outputs=(scores,),
            ),
            Attachment(
                operation="online_softmax_max_sum_output_update",
                site=AttachmentSite.ATTENTION_ONLINE_UPDATE,
                inputs=(scores, operation.value.name),
                outputs=(online_max, online_sum, online_output),
            ),
            Attachment(
                operation="normalize_online_output",
                site=AttachmentSite.ATTENTION_OUTPUT_TRANSFORM,
                inputs=(online_output, online_sum),
                outputs=(operation.output.name,),
            ),
        ),
    )
    internal_reason = "kept in registers/shared memory by the online-softmax skeleton"
    materializations = (
        MaterializationRecord(
            value=scores,
            shape=score_shape,
            dtype=operation.accumulation_dtype,
            disposition=MaterializationDisposition.INTERNAL_ATTENTION_STATE,
            reason=internal_reason,
        ),
        MaterializationRecord(
            value=probabilities,
            shape=score_shape,
            dtype=operation.accumulation_dtype,
            disposition=MaterializationDisposition.INTERNAL_ATTENTION_STATE,
            reason="represented by transient exponentials and never formed as a full tensor",
        ),
        MaterializationRecord(
            value=online_max,
            shape=statistic_shape,
            dtype=DType.FP32,
            disposition=MaterializationDisposition.INTERNAL_ATTENTION_STATE,
            reason=internal_reason,
        ),
        MaterializationRecord(
            value=online_sum,
            shape=statistic_shape,
            dtype=DType.FP32,
            disposition=MaterializationDisposition.INTERNAL_ATTENTION_STATE,
            reason=internal_reason,
        ),
        MaterializationRecord(
            value=online_output,
            shape=(batch, query_heads, query_length, value_dimension),
            dtype=DType.FP32,
            disposition=MaterializationDisposition.INTERNAL_ATTENTION_STATE,
            reason=internal_reason,
        ),
        MaterializationRecord(
            value=operation.output.name,
            shape=operation.output.shape,
            dtype=operation.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="semantic attention result crosses the skeleton boundary",
        ),
    )
    rewrite = RewriteExplanation(
        name="stream_exact_attention",
        applied=True,
        original_fragment=("Q @ K^T", "scale and causal mask", "softmax", "probabilities @ V"),
        transformed_fragment=(
            "keep one Q tile resident",
            "stream K/V tiles",
            "update online row maximum, normalizer, and output accumulator",
            "normalize and store the output tile",
        ),
        semantic_properties=(
            "cascaded row reductions",
            "associative max and sum state",
            "online softmax recurrence",
            "GQA head mapping",
        ),
        legality_checks=(
            "softmax reduces the key-sequence dimension",
            "scale and causal mask semantics are explicit",
            "query heads map evenly onto KV heads",
            "score and probability tensors have no external consumers",
            "head dimension is supported by the Hopper template family",
        ),
        estimated_benefit="eliminates both sequence-squared activation materializations",
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect="changes floating-point reduction order while preserving exact real-number attention semantics",
    )
    return RegionPlan(skeletons=(skeleton,), materializations=materializations, rewrites=(rewrite,))


def _materialized_attention_plan(operation: ScaledDotProductAttentionOp) -> RegionPlan:
    batch, query_length, query_heads, _ = operation.output.shape
    score_shape = (batch, query_heads, query_length, operation.key.shape[1])
    scores = f"{operation.output.name}.scores"
    probabilities = f"{operation.output.name}.probabilities"
    return RegionPlan(
        skeletons=(
            TransformSkeleton(
                name=f"{operation.output.name}.materialized_attention",
                operation="scaled_dot_product_attention",
                inputs=(operation.query.name, operation.key.name, operation.value.name),
                output=operation.output.name,
            ),
        ),
        materializations=(
            MaterializationRecord(
                value=scores,
                shape=score_shape,
                dtype=operation.accumulation_dtype,
                disposition=MaterializationDisposition.MATERIALIZE,
                reason="strict bitwise policy forbids the online reduction reorder",
            ),
            MaterializationRecord(
                value=probabilities,
                shape=score_shape,
                dtype=operation.accumulation_dtype,
                disposition=MaterializationDisposition.MATERIALIZE,
                reason="strict bitwise policy preserves the source reduction order",
            ),
            MaterializationRecord(
                value=operation.output.name,
                shape=operation.output.shape,
                dtype=operation.output.dtype,
                disposition=MaterializationDisposition.MATERIALIZE,
                reason="semantic attention result crosses the region boundary",
            ),
        ),
        rewrites=(
            RewriteExplanation(
                name="stream_exact_attention",
                applied=False,
                original_fragment=("Q @ K^T", "scale and causal mask", "softmax", "probabilities @ V"),
                transformed_fragment=(),
                semantic_properties=("online softmax recurrence",),
                legality_checks=("source floating-point reduction order must be preserved",),
                estimated_benefit="none; retained the executable materialized reference path",
                numerical_equivalence=NumericalEquivalence.BITWISE_EXACT,
                numerical_effect="none",
                rejection_reasons=("bitwise-exact numerical policy rejects reduction reassociation",),
            ),
        ),
    )
