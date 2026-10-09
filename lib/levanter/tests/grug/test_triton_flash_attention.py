# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import AttentionMask, reference_attention

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
# (tiles, head_dim): the small tiles, and the default tiles (None) at the head_dim training uses.
CASES = {"small_blocks": (SMALL_BLOCKS, 64), "tuned_blocks": (None, 128)}
# Rows 0-89 are one document, 90-199 another, 200-255 padding (segment -1, its own run).
SEGMENTS = jnp.concatenate([jnp.zeros(90), jnp.ones(110), -jnp.ones(56)]).astype(jnp.int32)

MASKS = {
    "causal": AttentionMask.causal(),
    "window": AttentionMask.causal(sliding_window=40),
    "window_segments": AttentionMask.causal(sliding_window=40).with_segment_ids(
        jnp.stack([SEGMENTS, jnp.roll(SEGMENTS, 37).at[:37].set(0)])
    ),
}


def _qkv(dtype, head_dim):
    """Two sequences with 8 query heads sharing 2 KV heads."""
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (2, SEQ_LEN, 8, head_dim), jnp.float32).astype(dtype)
    k = jax.random.normal(kk, (2, SEQ_LEN, 2, head_dim), jnp.float32).astype(dtype)
    v = jax.random.normal(kv, (2, SEQ_LEN, 2, head_dim), jnp.float32).astype(dtype)
    return q, k, v


def _reference(q, k, v, mask):
    """f32 reference on the same (possibly bf16-rounded) inputs."""
    with jax.default_matmul_precision("highest"):
        f32 = [x.astype(jnp.float32) for x in (q, k, v)]
        return reference_attention(*f32, mask, logits_dtype=jnp.float32)


# Largest absolute errors measured on MI300X and MI350X: 2.6e-6 in f32, and 8.9e-3 in bf16, in the forward output,
# which is itself rounded to bf16 (half a bf16 step is 7.8e-3 between 2 and 4).
TOLERANCE = {jnp.float32: 1e-4, jnp.bfloat16: 1e-2}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_forward_matches_reference(case, mask_name, dtype):
    block_sizes, head_dim = CASES[case]
    q, k, v = _qkv(dtype, head_dim)
    mask = MASKS[mask_name]
    actual = triton_flash_attention(q, k, v, mask, block_sizes=block_sizes)
    expected = _reference(q, k, v, mask)
    assert actual.dtype == dtype
    tol = TOLERANCE[dtype]
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected), atol=tol, rtol=tol)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_gradients_match_reference(case, mask_name, dtype):
    block_sizes, head_dim = CASES[case]
    q, k, v = _qkv(dtype, head_dim)
    mask = MASKS[mask_name]
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def flash(q, k, v, mask):
        return triton_flash_attention(q, k, v, mask, block_sizes=block_sizes)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v, mask).astype(jnp.float32) * cotangent)

    expected = jax.grad(lambda *a: loss(_reference, *a), argnums=(0, 1, 2))(q, k, v)
    actual = jax.grad(lambda *a: loss(flash, *a), argnums=(0, 1, 2))(q, k, v)
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
        return triton_flash_attention(q, k, v, mask, block_sizes=SMALL_BLOCKS)

    def loss(q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32))

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(attend, q, kv, kv)
        grads = jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv)
    assert output.sharding == sharding
    assert [g.sharding for g in grads] == [sharding] * 3
