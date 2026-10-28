# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import (
    AttentionMask,
    attention,
    inkling_rel_bias,
    reference_attention,
    token_validity_from_attention_mask,
)
from levanter.grug.attention._inkling_relpos import band_to_compact, compact_to_band, dense_rel_bias
from levanter.grug.attention._core import blocked_inkling_attention


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 37])
@pytest.mark.parametrize("segments", [0, 1, 2])
def test_structured_mask_query_tiles(causal, window, segments):
    mask = AttentionMask(is_causal=causal, sliding_window=window)
    if segments:
        q_ids = (jnp.arange(256) // 73).at[-19:].set(-1)
        k_ids = (jnp.arange(384) // 73).at[-17:].set(-1)
        if segments == 2:
            q_ids = jnp.stack((q_ids, q_ids // 2))
            k_ids = jnp.stack((k_ids, k_ids // 2))
        mask = mask.with_segment_ids(q_ids, k_ids)
    reference = mask.materialize_mask(256, 384)
    compute = eqx.filter_jit(lambda mask, start: mask.materialize_mask(64, 384, query_offset=start))
    for offset in (0, 64, 192):
        actual = compute(mask, jnp.asarray(offset, jnp.int32))
        if reference is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, reference[..., offset : offset + 64, :])


@pytest.mark.parametrize("dense_mask", [False, True])
@pytest.mark.parametrize(
    "block_size,precision", [(128, None), (256, None), (64, "highest"), (128, "highest"), (256, "highest")]
)
def test_blocked_inkling_matches_reference_values_and_gradients(dense_mask, block_size, precision):
    q, k, v = _make_qkv(batch=1, q_len=256, k_len=256, q_heads=2, kv_heads=1)
    v = jax.random.normal(jax.random.key(4), (*v.shape[:-1], 12))
    # A document boundary inside the second query tile catches tile-relative
    # causal indices and loss of packed-document masking.
    segments = jnp.asarray([[0] * 173 + [1] * 83])
    mask = AttentionMask.causal(sliding_window=80).with_segment_ids(segments)
    if dense_mask:
        mask = mask.materialize_mask(256, 256)
    states = jax.random.normal(jax.random.key(1), (1, 2, 256, 4)) * 0.1
    bank = jax.random.normal(jax.random.key(2), (4, 128)) * 0.1
    band = inkling_rel_bias(states, bank)
    weights = jax.random.normal(jax.random.key(3), (*q.shape[:-1], v.shape[-1]))

    def loss(q, k, v, band, *, blocked):
        output = (
            blocked_inkling_attention(q, k, v, mask, band, block_size=block_size)
            if blocked
            else reference_attention(q, k, v, mask, logits_dtype=jnp.float32, rel_bias=band)
        )
        return jnp.sum(output * weights), output

    compute = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True), static_argnames="blocked")
    # Highest precision isolates indexing/transpose parity across tile shapes;
    # TPU default matmul precision introduces tile-dependent float32 rounding.
    # Keep the production 128-row path covered at default precision as well.
    with jax.default_matmul_precision(precision):
        (value, output), grads = compute(q, k, v, band, blocked=True)
        (expected_value, expected_output), expected_grads = compute(q, k, v, band, blocked=False)
    np.testing.assert_allclose(value, expected_value, atol=1e-4, rtol=1e-4)
    for actual, expected in zip((output, *grads), (expected_output, *expected_grads), strict=True):
        difference = np.abs(np.asarray(actual) - np.asarray(expected))
        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"max error {difference.max()}, mean error {difference.mean()}",
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


def _distance_bias(relative_states, proj, seq_len):
    """bias[b, h, i, j] = R[b, h, i] . proj[:, i - j] for 0 <= i - j < rel_extent, else 0."""
    rel_extent = proj.shape[1]
    per_distance = jnp.einsum("bhsr,rl->bhsl", relative_states, proj)
    distance = jnp.arange(seq_len)[:, None] - jnp.arange(seq_len)[None, :]
    in_band = (distance >= 0) & (distance < rel_extent)
    index = jnp.broadcast_to(jnp.clip(distance, 0, rel_extent - 1), per_distance.shape[:3] + (seq_len,))
    return jnp.where(in_band, jnp.take_along_axis(per_distance, index, axis=-1), 0.0)


def test_inkling_rel_bias_band_matches_distance_bias_and_its_gradients():
    batch, heads, seq_len, rel_dim, rel_extent = 2, 3, 512, 4, 256
    relative_states = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq_len, rel_dim))
    proj = jax.random.normal(jax.random.PRNGKey(1), (rel_dim, rel_extent))
    weights = jax.random.normal(jax.random.PRNGKey(3), (batch, heads, seq_len, seq_len))

    def banded(r, p):
        return dense_rel_bias(inkling_rel_bias(r, p), seq_len, seq_len)

    np.testing.assert_allclose(
        banded(relative_states, proj), _distance_bias(relative_states, proj, seq_len), rtol=1e-5, atol=1e-5
    )
    grads = jax.grad(lambda r, p: jnp.sum(banded(r, p) * weights), argnums=(0, 1))(relative_states, proj)
    expected = jax.grad(lambda r, p: jnp.sum(_distance_bias(r, p, seq_len) * weights), argnums=(0, 1))(
        relative_states, proj
    )
    for grad, expected_grad in zip(grads, expected, strict=True):
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-4, atol=1e-4)


def test_inkling_rel_bias_compact_and_band_layouts_round_trip():
    compact = jax.random.normal(jax.random.PRNGKey(2), (1, 2, 256, 128))
    band = compact_to_band(compact)
    np.testing.assert_array_equal(np.asarray(band_to_compact(band)), np.asarray(compact))
    np.testing.assert_array_equal(np.asarray(compact_to_band(band_to_compact(band))), np.asarray(band))
