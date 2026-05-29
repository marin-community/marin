# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.data.text import GrugLmExample
from levanter.grug.attention import (
    AttentionMask,
    attention,
    reference_attention,
    thd_segment_metadata_from_segment_ids,
)


def _make_qkv(*, batch: int = 2, q_len: int = 6, k_len: int = 6, q_heads: int = 4, kv_heads: int = 2):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, q_len, q_heads, 8), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    return q, k, v


def test_reference_attention_matches_manual_segment_mask():
    q, k, v = _make_qkv(batch=1, q_len=5, k_len=5, q_heads=2, kv_heads=1)
    segment_ids = jnp.array([[3, 3, 8, 8, -1]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(segment_ids)

    actual = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    dense = jnp.array(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, False, False],
            [False, False, True, True, False],
            [False, False, False, False, True],
        ],
        dtype=jnp.bool_,
    )[None, :, :]
    expected = reference_attention(q, k, v, dense, logits_dtype=jnp.float32)

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_thd_segment_metadata_includes_padding_run():
    segment_ids = jnp.array([7, 7, 8, 8, 8, -1], dtype=jnp.int32)
    metadata = thd_segment_metadata_from_segment_ids(segment_ids, max_segments=3)

    np.testing.assert_array_equal(metadata.segment_lengths, jnp.array([2, 3, 1], dtype=jnp.int32))
    np.testing.assert_array_equal(metadata.num_segments, jnp.array(3, dtype=jnp.int32))


def test_grug_lm_example_with_max_segments_stacks():
    first = GrugLmExample.causal(
        jnp.arange(8, dtype=jnp.int32),
        segment_ids=jnp.array([0, 0, 1, 1, 2, 2, -1, -1], dtype=jnp.int32),
        max_segments=4,
    )
    second = GrugLmExample.causal(
        jnp.arange(8, dtype=jnp.int32),
        segment_ids=jnp.array([4, 4, 4, 5, 5, -1, -1, -1], dtype=jnp.int32),
        max_segments=4,
    )

    batch = jax.tree.map(lambda *xs: jnp.stack(xs), first, second)
    q_segment_ids, kv_segment_ids = batch.attn_mask.segment_ids

    assert batch.tokens.shape == (2, 8)
    np.testing.assert_array_equal(q_segment_ids, kv_segment_ids)


def test_thd_segment_metadata_rejects_mismatched_q_kv_segments():
    if jax.default_backend() == "tpu":
        pytest.skip("TPU runtime does not support raising eqx.error_if as a Python exception.")

    q_segment_ids = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    kv_segment_ids = jnp.array([0, 0, 2, 2], dtype=jnp.int32)

    with pytest.raises(Exception, match="matching q/kv segment_ids"):
        AttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids, max_segments=2)


def test_attention_rejects_unknown_implementation():
    q, k, v = _make_qkv()

    with pytest.raises(ValueError, match="Unknown Grug attention implementation"):
        attention(q, k, v, AttentionMask.causal(), implementation="nope")  # type: ignore[arg-type]
