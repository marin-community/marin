# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.attention._pallas_triton_flash import TritonFlashBlockSizes, pallas_triton_flash_attention

# On a GPU the kernels compile through Triton; elsewhere they run in the Pallas interpreter.
INTERPRET = jax.default_backend() != "gpu"
SEQ_LEN = 256
# Small tiles so the short test sequence still crosses masked, unmasked and skipped key blocks.
SMALL_BLOCKS = TritonFlashBlockSizes(
    block_q=32, block_k=16, block_q_dq=32, block_k_dq=16, block_q_dkv=16, block_k_dkv=32
)
# Rows 0-89 are one document, 90-199 another, 200-255 padding (segment -1, its own run).
SEGMENTS = jnp.concatenate([jnp.zeros(90), jnp.ones(110), -jnp.ones(56)]).astype(jnp.int32)

MASKS = {
    "causal": AttentionMask.causal(),
    "window": AttentionMask.causal(sliding_window=40),
    "window_segments": AttentionMask.causal(sliding_window=40).with_segment_ids(
        jnp.stack([SEGMENTS, jnp.roll(SEGMENTS, 37).at[:37].set(0)])
    ),
}


def _qkv(dtype):
    """Two sequences with 8 query heads sharing 2 KV heads, head_dim 64."""
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (2, SEQ_LEN, 8, 64), jnp.float32).astype(dtype)
    k = jax.random.normal(kk, (2, SEQ_LEN, 2, 64), jnp.float32).astype(dtype)
    v = jax.random.normal(kv, (2, SEQ_LEN, 2, 64), jnp.float32).astype(dtype)
    return q, k, v


@pytest.fixture(autouse=True)
def _interpreter_matmul_precision():
    """Run interpreted kernels at full f32 precision; TPU's default runs f32 matmuls in bf16 passes."""
    if not INTERPRET:
        yield
        return
    with jax.default_matmul_precision("highest"):
        yield


def _reference(q, k, v, mask):
    """f32 reference on the same (possibly bf16-rounded) inputs."""
    with jax.default_matmul_precision("highest"):
        f32 = [x.astype(jnp.float32) for x in (q, k, v)]
        return reference_attention(*f32, mask, logits_dtype=jnp.float32)


def _flash(q, k, v, mask):
    return pallas_triton_flash_attention(q, k, v, mask, block_sizes=SMALL_BLOCKS, interpret=INTERPRET)


# Worst cases measured on MI350X and in the CPU interpreter: 9e-7 in f32, and 3.5e-3 in bf16, where P and dS are
# rounded to bf16 before their matmuls as in every flash kernel.
TOLERANCE = {jnp.float32: 1e-4, jnp.bfloat16: 1e-2}


@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_forward_matches_reference(mask_name, dtype):
    q, k, v = _qkv(dtype)
    mask = MASKS[mask_name]
    actual = _flash(q, k, v, mask)
    expected = _reference(q, k, v, mask)
    assert actual.dtype == dtype
    tol = TOLERANCE[dtype]
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected), atol=tol, rtol=tol)


@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_gradients_match_reference(mask_name, dtype):
    q, k, v = _qkv(dtype)
    mask = MASKS[mask_name]
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v, mask).astype(jnp.float32) * cotangent)

    expected = jax.grad(lambda *a: loss(_reference, *a), argnums=(0, 1, 2))(q, k, v)
    actual = jax.grad(lambda *a: loss(_flash, *a), argnums=(0, 1, 2))(q, k, v)
    tol = TOLERANCE[dtype]
    for name, want, got in zip("qkv", expected, actual, strict=True):
        want = np.asarray(want, np.float32)
        got = np.asarray(got, np.float32)
        # Gradients of K/V sum over a GQA group and many queries, so compare relative to their scale.
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(got / scale, want / scale, atol=tol, rtol=tol, err_msg=f"d{name}")


def test_flash_attention_batch_sharded_mesh_keeps_output_and_gradient_sharding():
    mesh = AbstractMesh(axis_sizes=(2,), axis_names=("data",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, P("data", None, None, None))
    q = jax.ShapeDtypeStruct((4, SEQ_LEN, 8, 64), jnp.bfloat16, sharding=sharding)
    kv = jax.ShapeDtypeStruct((4, SEQ_LEN, 2, 64), jnp.bfloat16, sharding=sharding)
    mask = AttentionMask.causal(sliding_window=40)

    def attend(q, k, v):
        return pallas_triton_flash_attention(q, k, v, mask, block_sizes=SMALL_BLOCKS, interpret=True)

    def loss(q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32))

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(attend, q, kv, kv)
        grads = jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv)
    assert output.sharding == sharding
    assert [g.sharding for g in grads] == [sharding] * 3
