# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import (
    xla_flash_attention,
    AttentionMask,
    attention,
    fa4_cute_segment_bounds,
    reference_attention,
    token_validity_from_attention_mask,
)


from levanter.layers.attention import AttentionMask as NamedAttentionMask
from levanter.layers.attention import _tpu_splash_attention as named_splash_attention
from levanter.layers.rotary import DefaultRotaryEmbeddings, DefaultRotaryEmbeddingsConfig
from levanter.layers.rotary_native import RotaryConfig, apply_rotary


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


def test_reference_attention_applies_the_window_in_fa4_bounds():
    # Grug layer scans pass a full-causal mask and carry each layer's window only in the bounds.
    q, k, v = _make_qkv(batch=2, q_len=12, k_len=12, q_heads=2, kv_heads=1)
    # The second row ends in padding, whose bounds hold a sentinel rather than a lower bound.
    segment_ids = jnp.array([[0] * 7 + [1] * 5, [0] * 9 + [-1] * 3], dtype=jnp.int32)
    windowed = AttentionMask.causal(sliding_window=4).with_segment_ids(segment_ids)
    lower_bounds, valid = fa4_cute_segment_bounds(windowed, batch_size=2, seq_len=12, sliding_window=4)
    bounds_mask = AttentionMask.causal().with_segment_ids(segment_ids).with_fa4_bounds(lower_bounds, valid)

    actual = reference_attention(q, k, v, bounds_mask, logits_dtype=jnp.float32)
    expected = reference_attention(q, k, v, windowed, logits_dtype=jnp.float32)

    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.tpu
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


@pytest.mark.multidevice
def test_splash_head_sharding_partitions_sparse_mask_metadata():
    if len(jax.devices()) < 2:
        pytest.skip("Sparse metadata partitioning needs two devices.")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("model",), axis_types=(AxisType.Explicit,))
    qkv = jax.ShapeDtypeStruct(
        (1, 128, 4, 128), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, None, "model", None))
    )
    segments = jax.ShapeDtypeStruct((1, 128), jnp.int32, sharding=NamedSharding(mesh, P()))
    with jax.set_mesh(mesh):
        traced = jax.make_jaxpr(
            lambda q, k, v, ids: attention(
                q, k, v, AttentionMask.causal().with_segment_ids(ids), implementation="tpu_splash"
            )
        )(qkv, qkv, qkv, segments)
    mapped = next(eqn for eqn in traced.jaxpr.eqns if eqn.primitive.name == "shard_map")
    metadata = [
        var.aval for var in mapped.params["jaxpr"].invars if var.aval.dtype == jnp.int8 and len(var.aval.shape) == 3
    ]
    # Identical causal heads need one sparse mask group on each head shard. Global
    # mask groups must not reach a shard-local Splash invocation.
    assert metadata
    assert all(aval.shape[0] == 1 for aval in metadata)


@pytest.mark.tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("sequence_length", [256, 1024])
@pytest.mark.parametrize("segmented", [False, True])
def test_real_tpu_splash_attention_matches_reference(dtype, sequence_length, segmented):
    if jax.default_backend() != "tpu":
        pytest.skip("Splash attention requires a TPU backend.")

    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, -1),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sharding = NamedSharding(mesh, P(None, None, "model", None))
    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
    shape = (1, sequence_length, 8, 128)
    q, k, v = (
        jax.device_put((jax.random.normal(key, shape) * 0.02).astype(dtype), sharding) for key in (q_key, k_key, v_key)
    )
    mask = AttentionMask.causal()
    if segmented:
        ids = jax.device_put(jnp.arange(sequence_length, dtype=jnp.int32) // 64, NamedSharding(mesh, P()))
        mask = mask.with_segment_ids(ids)

    with jax.set_mesh(mesh), jax.default_matmul_precision("default"):
        actual = jax.jit(lambda q, k, v: attention(q, k, v, mask, implementation="tpu_splash"))(q, k, v)
        expected = jax.jit(lambda q, k, v: reference_attention(q, k, v, mask, logits_dtype=jnp.float32))(q, k, v)

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.tpu
def test_real_tpu_splash_attention_gradients_match_reference():
    if jax.default_backend() != "tpu":
        pytest.skip("Splash attention requires a TPU backend.")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("model",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, P(None, None, "model", None))
    keys = jax.random.split(jax.random.PRNGKey(114), 3)
    q, k, v = (jax.device_put(jax.random.normal(key, (1, 128, 32, 128)) * 0.5, sharding) for key in keys)
    ids = jax.device_put(jnp.arange(128, dtype=jnp.int32) // 64, NamedSharding(mesh, P()))
    mask = AttentionMask.causal().with_segment_ids(ids)

    def objective(q, k, v, implementation):
        output = attention(q, k, v, mask, implementation=implementation)
        return jnp.sum(jnp.sin(output))

    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        actual = jax.jit(jax.grad(lambda q, k, v: objective(q, k, v, "tpu_splash"), argnums=(0, 1, 2)))(q, k, v)
        expected = jax.jit(jax.grad(lambda q, k, v: objective(q, k, v, "reference"), argnums=(0, 1, 2)))(q, k, v)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.tpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("cotangent_scale", [1.0, 1e-6])
def test_splash_rotary_bfloat16_q_gradient_matches_named(cotangent_scale):
    if jax.default_backend() != "tpu":
        pytest.skip("Splash attention requires a TPU backend.")
    devices = np.asarray(jax.devices()).reshape(1, -1)
    explicit = jax.sharding.Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    auto = jax.sharding.Mesh(devices, ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto))
    batch, sequence, heads, width = 1, 128, 32, 128
    B, S, K, H, G, D = (
        hax.Axis(name, size)
        for name, size in (
            ("batch", batch),
            ("position", sequence),
            ("key_position", sequence),
            ("kv_head", heads),
            ("q_heads_per_group", 1),
            ("head_size", width),
        )
    )
    shape = (batch, sequence, heads, width)
    qkey, kkey, vkey, ckey = jax.random.split(jax.random.PRNGKey(141), 4)
    q, k, v = (jax.random.normal(key, shape).astype(jnp.bfloat16) for key in (qkey, kkey, vkey))
    cotangent = (jax.random.normal(ckey, shape) * jnp.float32(cotangent_scale)).astype(jnp.bfloat16)
    ids = jnp.broadcast_to(jnp.arange(sequence, dtype=jnp.int32) // 64, (batch, sequence))
    mapping = {"batch": "data", "kv_head": "model"}

    def native_vjp(q, k, v, cotangent, ids):
        positions = jnp.broadcast_to(jnp.arange(sequence, dtype=jnp.int32), (batch, sequence))
        k = apply_rotary(k, positions, RotaryConfig())
        mask = AttentionMask.causal().with_segment_ids(ids)

        def endpoint(q):
            q = apply_rotary(q, positions, RotaryConfig())
            return attention(q, k, v, mask, implementation="tpu_splash")

        output, pullback = jax.vjp(endpoint, q)
        return output, pullback(cotangent)[0]

    def named_vjp(q, k, v, cotangent, ids):
        rotary = DefaultRotaryEmbeddings(D, DefaultRotaryEmbeddingsConfig())
        k = rotary(k, hax.arange(S)).astype(k.dtype).rearrange((B, H, S, D)).rename({"position": "key_position"})
        v = v.rearrange((B, H, S, D)).rename({"position": "key_position"})
        mask = NamedAttentionMask.causal().with_segment_ids(ids)

        def endpoint(q):
            q = rotary(q, hax.arange(S)).astype(q.dtype).rearrange((B, H, G, S, D))
            output = named_splash_attention(
                S, K, D, q, k, v, mask, attention_dtype=q.dtype, scaling_factor=1.0 / math.sqrt(width)
            )
            return output.rearrange((B, S, H, G, D)).flatten_axes((H, G), H)

        output, pullback = jax.vjp(endpoint, q)
        return output, pullback(cotangent)[0].flatten_axes((H, G), H)

    spec = P(None, None, "model", None)
    # Compare the BF16 rounding stages used by training on both TPU generations.
    compiler_options = {"xla_allow_excess_precision": False}
    with jax.default_matmul_precision("default"):
        with jax.set_mesh(explicit):
            raw_args = tuple(
                jax.device_put(value, NamedSharding(explicit, spec)) for value in (q, k, v, cotangent)
            ) + (
                jax.device_put(ids, NamedSharding(explicit, P())),
            )
            actual = jax.jit(native_vjp, compiler_options=compiler_options)(*raw_args)
        with jax.set_mesh(auto), hax.axis_mapping(mapping):
            raw_args = tuple(jax.device_put(value, NamedSharding(auto, spec)) for value in (q, k, v, cotangent)) + (
                jax.device_put(ids, NamedSharding(auto, P())),
            )
            named_args = (
                hax.named(raw_args[0].reshape(batch, sequence, heads, 1, width), (B, S, H, G, D)),
                hax.named(raw_args[1], (B, S, H, D)),
                hax.named(raw_args[2], (B, S, H, D)),
                hax.named(raw_args[3], (B, S, H, D)),
                hax.named(raw_args[4], (B, S)),
            )
            expected = hax.named_jit(
                named_vjp, axis_resources=mapping, out_axis_resources=mapping, compiler_options=compiler_options
            )(*named_args)
    for observed, reference in zip(actual, expected, strict=True):
        assert np.isfinite(np.asarray(observed, dtype=np.float32)).all()
        # Numerical equality permits the signed-zero differences of the inverse rotation.
        np.testing.assert_array_equal(observed, reference.array)


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


def _flash_inputs():
    key = jax.random.key(3)
    kq, kk, kv = jax.random.split(key, 3)
    batch, seq, q_heads, kv_heads, head_dim = 2, 64, 4, 2, 8
    q = jax.random.normal(kq, (batch, seq, q_heads, head_dim), jnp.float32)
    k = jax.random.normal(kk, (batch, seq, kv_heads, head_dim), jnp.float32)
    v = jax.random.normal(kv, (batch, seq, kv_heads, head_dim), jnp.float32)
    # Two packed segments per row with different boundaries, under a causal sliding window.
    seg = jnp.stack([jnp.where(jnp.arange(seq) < 40, 0, 1), jnp.where(jnp.arange(seq) < 24, 0, 1)])
    mask = AttentionMask.causal(sliding_window=20).with_segment_ids(seg)
    return q, k, v, mask


def _assert_matches_reference(q, k, v, mask, *, block_size=None):
    # TPU matmuls default to bf16 precision; compare both paths at full precision so only the attention
    # computation, not accumulation order, can make them differ.
    with jax.default_matmul_precision("highest"):
        expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
        actual = xla_flash_attention(q, k, v, mask, block_size=block_size)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_xla_flash_attention_matches_reference_with_window_and_segments():
    q, k, v, mask = _flash_inputs()
    _assert_matches_reference(q, k, v, mask, block_size=16)


def test_xla_flash_attention_gradients_match_reference():
    q, k, v, mask = _flash_inputs()
    cot = jax.random.normal(jax.random.key(9), q.shape, jnp.float32)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v, mask) * cot)

    with jax.default_matmul_precision("highest"):
        ref_grads = jax.grad(
            lambda q, k, v: loss(lambda *a: reference_attention(*a, logits_dtype=jnp.float32), q, k, v),
            argnums=(0, 1, 2),
        )(q, k, v)
        flash_grads = jax.grad(
            lambda q, k, v: loss(lambda *a: xla_flash_attention(*a, block_size=16), q, k, v), argnums=(0, 1, 2)
        )(q, k, v)
    # TPU's emulated f32 backward differs from the reference by about 1e-4 on values of order one.
    for ref, flash in zip(ref_grads, flash_grads, strict=True):
        np.testing.assert_allclose(np.asarray(flash), np.asarray(ref), rtol=1e-3, atol=1e-3)


def test_xla_flash_attention_accepts_dense_boolean_mask():
    q, k, v, mask = _flash_inputs()
    dense = mask.materialize_mask(q.shape[1], k.shape[1])
    _assert_matches_reference(q, k, v, dense, block_size=16)


@pytest.mark.parametrize("seq_len", [48, 1025])
def test_xla_flash_attention_handles_lengths_that_do_not_divide_the_default_block(seq_len):
    """48 blocks at 16; 1025 has no usable power-of-two block and falls back to the reference path."""
    kq, kk, kv = jax.random.split(jax.random.key(5), 3)
    q = jax.random.normal(kq, (1, seq_len, 2, 8), jnp.float32)
    k = jax.random.normal(kk, (1, seq_len, 1, 8), jnp.float32)
    v = jax.random.normal(kv, (1, seq_len, 1, 8), jnp.float32)
    _assert_matches_reference(q, k, v, AttentionMask.causal(sliding_window=20))


def test_xla_flash_attention_broadcasts_shared_segment_ids_across_the_batch():
    q, k, v, _ = _flash_inputs()
    shared = jnp.where(jnp.arange(q.shape[1]) < 40, 0, 1)[None, :]  # one row of segment ids for a batch of 2
    mask = AttentionMask.causal(sliding_window=20).with_segment_ids(shared)
    _assert_matches_reference(q, k, v, mask, block_size=16)
