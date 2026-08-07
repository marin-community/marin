# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from tile_lifetime import (
    DType,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    StreamingAttentionSkeleton,
    TensorGraph,
    TransformSkeleton,
    compile_qkv_rope_attention_region,
)

BATCH = 1
SEQUENCE = 128
HIDDEN = 512
QUERY_HEADS = 8
KEY_VALUE_HEADS = 2
HEAD_DIMENSION = 64


def _qkv_rope_attention_region() -> TensorGraph:
    graph = TensorGraph()
    x = graph.input("x", shape=(BATCH, SEQUENCE, HIDDEN), dtype=DType.BF16)
    projection_width = (QUERY_HEADS + 2 * KEY_VALUE_HEADS) * HEAD_DIMENSION
    weight = graph.parameter("qkv_weight", shape=(HIDDEN, projection_width), dtype=DType.BF16)
    sine = graph.parameter("rope_sine", shape=(SEQUENCE, HEAD_DIMENSION // 2), dtype=DType.BF16)
    cosine = graph.parameter("rope_cosine", shape=(SEQUENCE, HEAD_DIMENSION // 2), dtype=DType.BF16)

    query, key, value = graph.qkv_projection(
        x,
        weight,
        name="qkv",
        query_heads=QUERY_HEADS,
        key_value_heads=KEY_VALUE_HEADS,
        head_dimension=HEAD_DIMENSION,
        accumulation_dtype=DType.FP32,
    )
    rotated_query, rotated_key = graph.rope(
        query,
        key,
        sine,
        cosine,
        name="rotated",
        rotary_dimension=HEAD_DIMENSION,
    )
    graph.scaled_dot_product_attention(
        rotated_query,
        rotated_key,
        value,
        name="attention_output",
        scale=HEAD_DIMENSION**-0.5,
        causal=True,
        accumulation_dtype=DType.FP32,
    )
    return graph


def test_qkv_rope_plan_feeds_selected_fa3_layout_without_conversion() -> None:
    plan = compile_qkv_rope_attention_region(
        _qkv_rope_attention_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert [type(skeleton) for skeleton in plan.skeletons] == [GemmSkeleton, StreamingAttentionSkeleton]
    qkv = plan.skeletons[0]
    attention = plan.skeletons[1]
    assert isinstance(qkv, GemmSkeleton)
    assert isinstance(attention, StreamingAttentionSkeleton)
    assert qkv.backend == "coda_cute_h100"
    assert qkv.shape == (BATCH * SEQUENCE, 768, HIDDEN)
    assert qkv.output_layout == attention.input_layout == "fa3_bshd_last_dimension_contiguous"
    assert [attachment.operation for attachment in qkv.epilogue] == [
        "partition_qkv_segment_views_bshd",
        "pairwise_rope_q",
        "pairwise_rope_k",
    ]
    assert attention.query == "rotated.query"
    assert attention.key == "rotated.key"
    assert attention.value == "qkv.value"
    assert attention.query_block_size == 192
    assert attention.key_value_block_size == 128
    assert not any(isinstance(skeleton, TransformSkeleton) for skeleton in plan.skeletons)
    assert plan.sequence_squared_materializations == ()


def test_qkv_rope_plan_keeps_only_unrotated_qk_inside_epilogue() -> None:
    plan = compile_qkv_rope_attention_region(
        _qkv_rope_attention_region(),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert plan.materialization("qkv.query").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("qkv.key").disposition is MaterializationDisposition.EPILOGUE_ONLY
    assert plan.materialization("rotated.query").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.materialization("rotated.key").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.materialization("qkv.value").disposition is MaterializationDisposition.MATERIALIZE
    assert all("layout-conversion" not in record.reason for record in plan.materializations)
    assert plan.rewrites[0].applied


def test_bitwise_qkv_rope_plan_retains_source_boundaries() -> None:
    plan = compile_qkv_rope_attention_region(
        _qkv_rope_attention_region(),
        numerical_policy=NumericalPolicy.BITWISE_EXACT,
    )

    assert any(isinstance(skeleton, TransformSkeleton) for skeleton in plan.skeletons)
    assert plan.materialization("qkv.query").disposition is MaterializationDisposition.MATERIALIZE
    assert plan.materialization("qkv.key").disposition is MaterializationDisposition.MATERIALIZE
    assert not plan.rewrites[0].applied


def test_combined_qkv_pairwise_rope_matches_separate_real_computation() -> None:
    rng = np.random.default_rng(23)
    tokens = 5
    hidden = 7
    query_heads = 2
    key_value_heads = 1
    head_dimension = 4
    query_width = query_heads * head_dimension
    key_value_width = key_value_heads * head_dimension
    x = rng.normal(size=(tokens, hidden))
    query_weight = rng.normal(size=(hidden, query_width))
    key_weight = rng.normal(size=(hidden, key_value_width))
    value_weight = rng.normal(size=(hidden, key_value_width))
    angle = rng.normal(size=(tokens, head_dimension // 2))
    sine = np.sin(angle)
    cosine = np.cos(angle)

    def rotate(values: np.ndarray) -> np.ndarray:
        pairs = values.reshape(tokens, -1, head_dimension // 2, 2)
        even = pairs[..., 0] * cosine[:, None, :] - pairs[..., 1] * sine[:, None, :]
        odd = pairs[..., 0] * sine[:, None, :] + pairs[..., 1] * cosine[:, None, :]
        return np.stack((even, odd), axis=-1).reshape(values.shape)

    reference_query = rotate((x @ query_weight).reshape(tokens, query_heads, head_dimension))
    reference_key = rotate((x @ key_weight).reshape(tokens, key_value_heads, head_dimension))
    reference_value = (x @ value_weight).reshape(tokens, key_value_heads, head_dimension)

    combined_weight = np.concatenate((query_weight, key_weight, value_weight), axis=1)
    combined = x @ combined_weight
    query_end = query_width
    key_end = query_end + key_value_width
    fused_query = rotate(combined[:, :query_end].reshape(tokens, query_heads, head_dimension))
    fused_key = rotate(combined[:, query_end:key_end].reshape(tokens, key_value_heads, head_dimension))
    fused_value = combined[:, key_end:].reshape(tokens, key_value_heads, head_dimension)

    np.testing.assert_allclose(fused_query, reference_query, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fused_key, reference_key, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fused_value, reference_value, rtol=1e-12, atol=1e-12)
