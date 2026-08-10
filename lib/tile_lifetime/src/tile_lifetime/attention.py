# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic attention Fold algebra and historical named-operation planning."""

from dataclasses import dataclass

import numpy as np

from shuttle.ir import DType
from tile_lifetime.ir import ScaledDotProductAttentionOp, TensorGraph
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
class AttentionPartial:
    """Mergeable exact-attention state over a subset of key/value positions."""

    row_max: np.ndarray
    row_sum_exp: np.ndarray
    weighted_value_accumulator: np.ndarray


@dataclass(frozen=True)
class NormalizedAttentionPartial:
    """Compact physical form of an attention partial.

    ``row_log_normalizer`` is ``row_max + log(row_sum_exp)`` and
    ``normalized_weighted_value`` is the already normalized partial output.
    This representation is useful when partial states cross a kernel boundary:
    it stores one scalar per row rather than separate maximum and sum fields.
    """

    row_log_normalizer: np.ndarray
    normalized_weighted_value: np.ndarray


def summarize_attention_partial(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    scale: float,
    causal: bool = False,
    query_positions: np.ndarray | None = None,
    key_positions: np.ndarray | None = None,
    query_valid: np.ndarray | None = None,
    key_valid: np.ndarray | None = None,
) -> AttentionPartial:
    """Summarize one Q-by-KV block as online-softmax state."""
    query, key, value = _validate_attention_block(query, key, value, scale)
    query_position, key_position = _resolve_positions(
        query.shape[0],
        key.shape[0],
        causal=causal,
        query_positions=query_positions,
        key_positions=key_positions,
    )
    scores = _attention_scores(query, key, scale)
    resolved_query_valid = _resolve_token_validity(query_valid, query.shape[0], "query")
    resolved_key_valid = _resolve_token_validity(key_valid, key.shape[0], "key")
    score_valid = resolved_query_valid[:, None] & resolved_key_valid[None, :]
    if causal:
        score_valid &= query_position[:, None] >= key_position[None, :]
    scores = np.where(score_valid[:, None, :], scores, -np.inf)
    return _summarize_scores(scores, value, query.shape[1])


def merge_attention_partials(left: AttentionPartial, right: AttentionPartial) -> AttentionPartial:
    """Merge disjoint attention states after rescaling them to one row maximum."""
    _validate_partial_pair(left, right)
    row_max = np.maximum(left.row_max, right.row_max)
    left_scale = _state_rescale(left.row_max, row_max, left.row_sum_exp)
    right_scale = _state_rescale(right.row_max, row_max, right.row_sum_exp)
    row_sum_exp = left_scale * left.row_sum_exp + right_scale * right.row_sum_exp
    weighted_value_accumulator = (
        left_scale[..., None] * left.weighted_value_accumulator
        + right_scale[..., None] * right.weighted_value_accumulator
    )
    return AttentionPartial(
        row_max=row_max,
        row_sum_exp=row_sum_exp,
        weighted_value_accumulator=weighted_value_accumulator,
    )


def finalize_attention_partial(partial: AttentionPartial) -> np.ndarray:
    """Normalize a complete attention state, rejecting rows with an empty fold domain."""
    empty_rows = partial.row_sum_exp <= 0
    if np.any(empty_rows):
        locations = np.argwhere(empty_rows)
        raise ValueError(f"attention rows have no valid selected keys at indices {locations.tolist()}")
    return partial.weighted_value_accumulator / partial.row_sum_exp[..., None]


def normalize_attention_partial(partial: AttentionPartial) -> NormalizedAttentionPartial:
    """Change physical state coordinates without changing Fold semantics."""
    empty_rows = partial.row_sum_exp <= 0
    safe_sum = np.where(empty_rows, 1.0, partial.row_sum_exp)
    row_log_normalizer = np.where(empty_rows, -np.inf, partial.row_max + np.log(safe_sum))
    normalized_weighted_value = np.where(
        empty_rows[..., None],
        0.0,
        partial.weighted_value_accumulator / safe_sum[..., None],
    )
    return NormalizedAttentionPartial(
        row_log_normalizer=row_log_normalizer.astype(np.float32),
        normalized_weighted_value=normalized_weighted_value.astype(np.float32),
    )


def merge_normalized_attention_partials(
    left: NormalizedAttentionPartial,
    right: NormalizedAttentionPartial,
) -> NormalizedAttentionPartial:
    """Merge compact partial states using log-normalizer weights."""
    if left.row_log_normalizer.shape != right.row_log_normalizer.shape:
        raise ValueError("attention log-normalizer shapes must match")
    if left.normalized_weighted_value.shape != right.normalized_weighted_value.shape:
        raise ValueError("normalized attention value shapes must match")
    if left.normalized_weighted_value.shape[:-1] != left.row_log_normalizer.shape:
        raise ValueError("normalized attention values must have one vector per log-normalizer row")

    common = np.maximum(left.row_log_normalizer, right.row_log_normalizer)
    finite_common = np.isfinite(common)
    safe_common = np.where(finite_common, common, 0.0)
    left_weight = np.where(
        np.isfinite(left.row_log_normalizer),
        np.exp(left.row_log_normalizer - safe_common),
        0.0,
    ).astype(np.float32)
    right_weight = np.where(
        np.isfinite(right.row_log_normalizer),
        np.exp(right.row_log_normalizer - safe_common),
        0.0,
    ).astype(np.float32)
    total_weight = left_weight + right_weight
    safe_weight = np.where(total_weight > 0, total_weight, 1.0)
    normalized = (
        left_weight[..., None] * left.normalized_weighted_value
        + right_weight[..., None] * right.normalized_weighted_value
    ) / safe_weight[..., None]
    row_log_normalizer = np.where(total_weight > 0, safe_common + np.log(safe_weight), -np.inf)
    return NormalizedAttentionPartial(
        row_log_normalizer=row_log_normalizer.astype(np.float32),
        normalized_weighted_value=normalized.astype(np.float32),
    )


def finalize_normalized_attention_partial(partial: NormalizedAttentionPartial) -> np.ndarray:
    """Return the normalized value, rejecting an empty Fold domain."""
    empty_rows = ~np.isfinite(partial.row_log_normalizer)
    if np.any(empty_rows):
        locations = np.argwhere(empty_rows)
        raise ValueError(f"attention rows have no valid selected keys at indices {locations.tolist()}")
    return partial.normalized_weighted_value


@dataclass(frozen=True)
class ReferenceAttentionPhysicalConfig:
    """Historical physical configuration for the opaque FA3 comparison path."""

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


def select_reference_hopper_attention_config(
    operation: ScaledDotProductAttentionOp,
) -> ReferenceAttentionPhysicalConfig:
    """Select the historical opaque-FA3 comparison configuration."""
    if operation.head_dimension not in (64, 128):
        raise ValueError(f"FA3-style template does not support head dimension {operation.head_dimension}")
    if operation.head_dimension == 64:
        query_block_size = 192
        key_value_block_size = 128 if operation.causal else 192
    else:
        query_block_size = 128
        key_value_block_size = 128 if operation.causal else 176
    mma_pv_is_rs = operation.causal
    return ReferenceAttentionPhysicalConfig(
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


def compile_reference_attention_region(graph: TensorGraph, *, numerical_policy: NumericalPolicy) -> RegionPlan:
    """Compile one named attention operation through the historical physical planner."""
    operations = tuple(operation for operation in graph.operations if isinstance(operation, ScaledDotProductAttentionOp))
    if len(operations) != 1:
        raise ValueError(f"expected one semantic attention operation, found {len(operations)}")
    operation = operations[0]
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        return _materialized_attention_plan(operation)

    config = select_reference_hopper_attention_config(operation)
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


def _summarize_scores(scores: np.ndarray, value: np.ndarray, query_head_count: int) -> AttentionPartial:
    row_max = np.max(scores, axis=-1)
    finite_rows = np.isfinite(row_max)
    centered = np.full(scores.shape, -np.inf, dtype=np.float32)
    centered[finite_rows] = scores[finite_rows] - row_max[finite_rows, None]
    exponentials = np.exp(centered)
    row_sum_exp = np.sum(exponentials, axis=-1)
    head_map = _query_to_kv_head(query_head_count, value.shape[1])
    expanded_value = value.astype(np.float32)[:, head_map, :]
    weighted_value = np.einsum("qhk,khv->qhv", exponentials, expanded_value)
    return AttentionPartial(
        row_max=row_max,
        row_sum_exp=row_sum_exp,
        weighted_value_accumulator=weighted_value,
    )


def _state_rescale(old_max: np.ndarray, new_max: np.ndarray, old_sum: np.ndarray) -> np.ndarray:
    scale = np.zeros(old_sum.shape, dtype=np.float32)
    populated = old_sum > 0
    scale[populated] = np.exp(old_max[populated] - new_max[populated])
    return scale


def _validate_partial_pair(left: AttentionPartial, right: AttentionPartial) -> None:
    if left.row_max.shape != right.row_max.shape or left.row_sum_exp.shape != left.row_max.shape:
        raise ValueError("attention partial statistics must have matching shapes")
    if right.row_sum_exp.shape != right.row_max.shape:
        raise ValueError("attention partial statistics must have matching shapes")
    expected_value_shape = (*left.row_max.shape, left.weighted_value_accumulator.shape[-1])
    if left.weighted_value_accumulator.shape != expected_value_shape:
        raise ValueError("left weighted-value accumulator has incompatible shape")
    if right.weighted_value_accumulator.shape != left.weighted_value_accumulator.shape:
        raise ValueError("attention partial accumulators must have matching shapes")


def _validate_attention_block(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query)
    key = np.asarray(key)
    value = np.asarray(value)
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("attention blocks must have shapes [token, head, feature]")
    if key.shape[:2] != value.shape[:2]:
        raise ValueError("key and value blocks must have matching token and head dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    _query_to_kv_head(query.shape[1], key.shape[1])
    if not np.isfinite(scale):
        raise ValueError("attention scale must be finite")
    return query, key, value


def _resolve_positions(
    query_count: int,
    key_count: int,
    *,
    causal: bool,
    query_positions: np.ndarray | None,
    key_positions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if query_positions is None:
        query_positions = np.arange(query_count, dtype=np.int64)
    else:
        query_positions = np.asarray(query_positions)
    if key_positions is None:
        key_positions = np.arange(key_count, dtype=np.int64)
    else:
        key_positions = np.asarray(key_positions)
    if query_positions.shape != (query_count,) or key_positions.shape != (key_count,):
        raise ValueError("query and key positions must match their block token counts")
    if causal and (
        not np.issubdtype(query_positions.dtype, np.integer) or not np.issubdtype(key_positions.dtype, np.integer)
    ):
        raise ValueError("causal query and key positions must use integer indices")
    return query_positions, key_positions


def _resolve_token_validity(validity: np.ndarray | None, token_count: int, name: str) -> np.ndarray:
    if validity is None:
        return np.ones(token_count, dtype=np.bool_)
    resolved = np.asarray(validity, dtype=np.bool_)
    if resolved.shape != (token_count,):
        raise ValueError(f"{name} token validity must have shape {(token_count,)}, got {resolved.shape}")
    return resolved


def _attention_scores(query: np.ndarray, key: np.ndarray, scale: float) -> np.ndarray:
    head_map = _query_to_kv_head(query.shape[1], key.shape[1])
    expanded_key = key.astype(np.float32)[:, head_map, :]
    return np.einsum("qhd,khd->qhk", query.astype(np.float32), expanded_key) * np.float32(scale)


def _query_to_kv_head(query_head_count: int, kv_head_count: int) -> np.ndarray:
    if kv_head_count <= 0 or query_head_count % kv_head_count:
        raise ValueError("query heads must map evenly onto KV heads")
    return np.arange(query_head_count, dtype=np.int32) // (query_head_count // kv_head_count)
