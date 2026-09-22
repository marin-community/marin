# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import (
    AttentionMask,
    attention,
    reference_attention,
    token_validity_from_attention_mask,
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


def test_reference_attention_supports_model_sharded_head_dimension():
    q, k, v = _make_qkv(batch=1, q_len=5, k_len=5, q_heads=2, kv_heads=1)
    mask = AttentionMask.causal()
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]),
        ("model",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    qkv_sharding = NamedSharding(mesh, P(None, None, None, "model"))
    sharded_q, sharded_k, sharded_v = (jax.device_put(x, qkv_sharding) for x in (q, k, v))

    actual = jax.jit(reference_attention, static_argnames=("mask", "logits_dtype"))(
        sharded_q,
        sharded_k,
        sharded_v,
        mask=mask,
        logits_dtype=jnp.float32,
    )

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert isinstance(actual.sharding, NamedSharding)
    assert actual.sharding.spec == qkv_sharding.spec


def test_reference_attention_eval_shape_supports_model_sharded_grouped_query_heads():
    mesh = AbstractMesh(
        axis_sizes=(2, 2),
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    q_sharding = NamedSharding(mesh, P("data", None, "model", None))
    kv_sharding = NamedSharding(mesh, P("data", None, None, None))
    q = jax.ShapeDtypeStruct((8, 3, 4, 4), jnp.float32, sharding=q_sharding)
    k = jax.ShapeDtypeStruct((8, 3, 2, 4), jnp.float32, sharding=kv_sharding)
    v = jax.ShapeDtypeStruct((8, 3, 2, 4), jnp.float32, sharding=kv_sharding)

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(lambda q, k, v: reference_attention(q, k, v, None, logits_dtype=jnp.float32), q, k, v)

    assert output.sharding == q_sharding


def test_real_tpu_splash_attention_matches_reference():
    if jax.default_backend() != "tpu":
        pytest.skip("Splash attention requires a TPU backend.")

    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, -1),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sharding = NamedSharding(mesh, P(None, None, "model", None))
    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.device_put(jax.random.normal(q_key, (1, 256, 8, 128), dtype=jnp.float32) * 0.02, sharding)
    k = jax.device_put(jax.random.normal(k_key, (1, 256, 8, 128), dtype=jnp.float32) * 0.02, sharding)
    v = jax.device_put(jax.random.normal(v_key, (1, 256, 8, 128), dtype=jnp.float32) * 0.02, sharding)
    mask = AttentionMask.causal()

    with jax.set_mesh(mesh):
        actual = jax.jit(lambda q, k, v: attention(q, k, v, mask, implementation="tpu_splash"))(q, k, v)
        expected = jax.jit(lambda q, k, v: reference_attention(q, k, v, mask, logits_dtype=jnp.float32))(q, k, v)

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_token_validity_uses_padding_ids_without_excluding_packed_boundaries():
    segment_ids = jnp.array(
        [
            [0, 0, 1, 1, -1, -1],
            [7, 8, 8, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    mask = AttentionMask.causal().with_segment_ids(segment_ids)

    valid = token_validity_from_attention_mask(mask, batch_size=3, sequence_length=6)

    np.testing.assert_array_equal(
        valid,
        jnp.array(
            [
                [True, True, True, True, False, False],
                [True, True, True, False, False, False],
                [False, False, False, False, False, False],
            ]
        ),
    )


def test_token_validity_uses_empty_rows_from_dense_boolean_masks():
    dense_mask = jnp.array(
        [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ],
        dtype=jnp.bool_,
    )

    valid = token_validity_from_attention_mask(dense_mask, batch_size=2, sequence_length=3)

    np.testing.assert_array_equal(valid, jnp.array([[True, True, False], [True, True, False]]))


def test_attention_rejects_unknown_implementation():
    q, k, v = _make_qkv()

    with pytest.raises(ValueError, match="Unknown Grug attention implementation"):
        attention(q, k, v, AttentionMask.causal(), implementation="nope")  # type: ignore[arg-type]
