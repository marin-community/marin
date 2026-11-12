# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import AttentionMask, fa4_cute_segment_bounds, reference_attention

# The kernels are Triton launched through jax_triton, from the gpu extra; Triton has no CPU backend.
pytest.importorskip("jax_triton")
if jax.default_backend() != "gpu":
    pytest.skip("gpu_triton_flash runs only on the JAX GPU backend", allow_module_level=True)

from levanter.grug.attention._triton_flash import TritonFlashBlockSizes, triton_flash_attention  # noqa: E402

SEQ_LEN = 256
# Small tiles so the short test sequence still crosses masked, unmasked and skipped key blocks.
SMALL_BLOCKS = TritonFlashBlockSizes(
    block_q=32, block_k=16, block_q_dq=32, block_k_dq=16, block_q_dkv=16, block_k_dkv=32
)
# (tiles, head_dim, query heads, KV heads): the small tiles; the tuned tiles (None) at the head_dim training uses;
# and a GQA ratio that is not a power of two.
CASES = {
    "small_blocks": (SMALL_BLOCKS, 64, 8, 2),
    "tuned_blocks": (None, 128, 8, 2),
    "gqa_ratio_3": (SMALL_BLOCKS, 64, 6, 2),
}
# Rows 0-89 are one document, 90-199 another, 200-255 padding (segment -1, its own run).
SEGMENTS = jnp.concatenate([jnp.zeros(90), jnp.ones(110), -jnp.ones(56)]).astype(jnp.int32)

MASKS = {
    "causal": AttentionMask.causal(),
    "window": AttentionMask.causal(sliding_window=40),
    "window_segments": AttentionMask.causal(sliding_window=40).with_segment_ids(
        jnp.stack([SEGMENTS, jnp.roll(SEGMENTS, 37).at[:37].set(0)])
    ),
}


def _qkv(dtype, head_dim, q_heads, kv_heads):
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (2, SEQ_LEN, q_heads, head_dim), jnp.float32).astype(dtype)
    k = jax.random.normal(kk, (2, SEQ_LEN, kv_heads, head_dim), jnp.float32).astype(dtype)
    v = jax.random.normal(kv, (2, SEQ_LEN, kv_heads, head_dim), jnp.float32).astype(dtype)
    return q, k, v


def _reference(q, k, v, mask):
    """f32 reference on the same (possibly bf16-rounded) inputs."""
    with jax.default_matmul_precision("highest"):
        f32 = [x.astype(jnp.float32) for x in (q, k, v)]
        return reference_attention(*f32, mask, logits_dtype=jnp.float32)


# Worst cases over these tests on MI300X and MI350X, as the smallest allclose atol=rtol that passes: 4e-6 in f32; in
# bf16, 3.0e-3 for the output and 1.7e-2 for the gradients, about one bf16 step of each value, since P and dS are
# rounded to bf16 before their matmuls as in every flash kernel.
TOLERANCE = {jnp.float32: 1e-4, jnp.bfloat16: 1e-2}
GRADIENT_TOLERANCE = {jnp.float32: 1e-4, jnp.bfloat16: 2e-2}
# Worst mean absolute errors measured the same way: 2e-7 in f32 and 1.0e-3 in bf16.
MEAN_ERROR = {jnp.float32: 1e-6, jnp.bfloat16: 2e-3}


def _assert_matches(actual, expected, *, tol, mean_tol, name):
    """allclose, plus a bound on the mean absolute error that catches a bias allclose's rtol would absorb."""
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    error = np.abs(actual - expected)
    assert (
        error.mean() <= mean_tol
    ), f"{name}: mean |error| {error.mean():.2e} > {mean_tol:.0e} (max {error.max():.2e})"
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol, err_msg=name)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_forward_matches_reference(case, mask_name, dtype):
    block_sizes, *shape = CASES[case]
    q, k, v = _qkv(dtype, *shape)
    mask = MASKS[mask_name]
    actual = triton_flash_attention(q, k, v, mask, block_sizes=block_sizes)
    assert actual.dtype == dtype
    expected = _reference(q, k, v, mask)
    _assert_matches(actual, expected, tol=TOLERANCE[dtype], mean_tol=MEAN_ERROR[dtype], name="out")


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_gradients_match_reference(case, mask_name, dtype):
    block_sizes, *shape = CASES[case]
    q, k, v = _qkv(dtype, *shape)
    mask = MASKS[mask_name]
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def flash(q, k, v, mask):
        return triton_flash_attention(q, k, v, mask, block_sizes=block_sizes)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v, mask).astype(jnp.float32) * cotangent)

    expected = jax.grad(lambda *a: loss(_reference, *a), argnums=(0, 1, 2))(q, k, v)
    actual = jax.grad(lambda *a: loss(flash, *a), argnums=(0, 1, 2))(q, k, v)
    for name, want, got in zip("qkv", expected, actual, strict=True):
        _assert_matches(got, want, tol=GRADIENT_TOLERANCE[dtype], mean_tol=MEAN_ERROR[dtype], name=f"d{name}")


def _hero_mask(segment_ids, window):
    """The moe_hero_ep layer mask: a full causal mask whose precomputed bounds carry the sliding window."""
    full = AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids))
    windowed = AttentionMask(is_causal=True, sliding_window=window, segment_ids=(segment_ids, segment_ids))
    lower, valid = fa4_cute_segment_bounds(
        windowed, batch_size=segment_ids.shape[0], seq_len=segment_ids.shape[1], sliding_window=window
    )
    return full.with_fa4_bounds(lower, valid)


def test_flash_attention_on_single_device_grug_mesh_matches_reference():
    # A one-device Grug mesh keeps its length-1 axes, so every spec names mesh axes that partition nothing.
    names = ("replica_dcn", "data", "context", "expert", "model")
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 5), names, axis_types=(AxisType.Explicit,) * 5)
    batch_axes = ("replica_dcn", "data", "expert")
    q, k, v = _qkv(jnp.bfloat16, 64, 8, 2)
    segments = MASKS["window_segments"].segment_ids[0]
    reference_mask = MASKS["window_segments"]
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)
    expected = _reference(q, k, v, reference_mask)
    expected_grads = jax.grad(lambda *a: jnp.sum(_reference(*a, reference_mask) * cotangent), argnums=(0, 1, 2))(
        q, k, v
    )
    valid_rows = np.asarray(segments >= 0)[:, :, None, None]

    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P(batch_axes, None, "model", None))
        q, k, v, cotangent = (jax.device_put(x, sharding) for x in (q, k, v, cotangent))
        ids = jax.device_put(segments, NamedSharding(mesh, P(batch_axes, None)))
        # Separate q and kv arrays, as a batch from the data loader carries them.
        segment_mask = AttentionMask.causal(sliding_window=40).with_segment_ids(ids, jnp.copy(ids))

        def flash(q, k, v, mask):
            return triton_flash_attention(q, k, v, mask, block_sizes=SMALL_BLOCKS)

        def loss(q, k, v, mask):
            return jnp.sum(flash(q, k, v, mask).astype(jnp.float32) * cotangent)

        actual = jax.jit(flash)(q, k, v, segment_mask)
        actual_grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v, segment_mask)
        hero = jax.jit(flash)(q, k, v, _hero_mask(ids, 40))

    assert actual.sharding.spec == sharding.spec
    _assert_matches(actual, expected, tol=TOLERANCE[jnp.bfloat16], mean_tol=MEAN_ERROR[jnp.bfloat16], name="out")
    for name, want, got in zip("qkv", expected_grads, actual_grads, strict=True):
        tol = GRADIENT_TOLERANCE[jnp.bfloat16]
        _assert_matches(got, want, tol=tol, mean_tol=MEAN_ERROR[jnp.bfloat16], name=f"d{name}")
    # Padding queries attend only to themselves under precomputed bounds, so compare document tokens.
    hero = np.where(valid_rows, np.asarray(hero, np.float32), np.asarray(expected))
    _assert_matches(hero, expected, tol=TOLERANCE[jnp.bfloat16], mean_tol=MEAN_ERROR[jnp.bfloat16], name="hero out")


def test_flash_attention_batch_sharded_mesh_keeps_output_and_gradient_sharding():
    mesh = AbstractMesh(axis_sizes=(2,), axis_names=("data",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, P("data", None, None, None))
    q = jax.ShapeDtypeStruct((4, SEQ_LEN, 8, 64), jnp.bfloat16, sharding=sharding)
    kv = jax.ShapeDtypeStruct((4, SEQ_LEN, 2, 64), jnp.bfloat16, sharding=sharding)
    ids = jax.ShapeDtypeStruct((4, SEQ_LEN), jnp.int32, sharding=NamedSharding(mesh, P("data", None)))

    def attend(q, k, v, q_ids, kv_ids):
        mask = AttentionMask.causal(sliding_window=40).with_segment_ids(q_ids, kv_ids)
        return triton_flash_attention(q, k, v, mask, block_sizes=SMALL_BLOCKS)

    def loss(q, k, v, q_ids, kv_ids):
        return jnp.sum(attend(q, k, v, q_ids, kv_ids).astype(jnp.float32))

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(attend, q, kv, kv, ids, ids)
        grads = jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv, ids, ids)
    assert output.sharding == sharding
    assert [g.sharding for g in grads] == [sharding] * 3
