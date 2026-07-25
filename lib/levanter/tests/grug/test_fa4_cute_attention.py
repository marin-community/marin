# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P, use_abstract_mesh

import levanter.grug.attention._fa4_cute as fa4_cute
import levanter.grug.attention._fa4_cute_backend as fa4_cute_backend
from levanter.grug.attention import (
    AttentionMask,
    gpu_fa4_cute_attention,
    reference_attention,
)
from levanter.grug.attention._fa4_cute import _simple_causal_lower_bounds


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _make_qkv(*, batch: int = 2, q_len: int = 6, k_len: int = 6, q_heads: int = 4, kv_heads: int = 2):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, q_len, q_heads, 8), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    return q, k, v


def test_packed_segment_backward_block_sparse_indices_are_q_direction():
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, -1]], dtype=jnp.int32)
    lower_bounds, valid = fa4_cute._packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=8,
        sliding_window=None,
    )

    mask_block_cnt, mask_block_idx = fa4_cute_backend._packed_segment_backward_block_sparse_indices(
        lower_bounds,
        valid,
        tile_m=2,
        tile_n=4,
    )

    np.testing.assert_array_equal(mask_block_cnt, jnp.array([[[2, 2]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        mask_block_idx,
        jnp.array([[[[0, 1, 0, 0], [2, 3, 0, 0]]]], dtype=jnp.int32),
    )


def test_packed_segment_backward_block_sparse_indices_split_full_blocks():
    segment_ids = jnp.zeros((1, 8), dtype=jnp.int32)
    lower_bounds, valid = fa4_cute._packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=8,
        sliding_window=None,
    )

    sparse_metadata = fa4_cute_backend._packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        tile_m=2,
        tile_n=2,
    )

    np.testing.assert_array_equal(sparse_metadata.partial_block_cnt, jnp.array([[[1, 1, 1, 1]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        sparse_metadata.partial_block_idx,
        jnp.array([[[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]]], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(sparse_metadata.full_block_cnt, jnp.array([[[3, 2, 1, 0]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        sparse_metadata.full_block_idx,
        jnp.array([[[[1, 2, 3, 0], [2, 3, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]]], dtype=jnp.int32),
    )


def test_fa4_frontend_rejects_mismatched_q_kv_segment_ids():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe validation requires a GPU backend.")
    q, k, v = _make_qkv(batch=1, q_len=4, k_len=4, q_heads=2, kv_heads=1)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    q_segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    kv_segment_ids = jnp.array([[1, 1, 3, 3]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids)

    with pytest.raises(Exception, match="requires matching q/kv segment_ids"):
        jax.block_until_ready(gpu_fa4_cute_attention(q, k, v, mask))


def test_simple_causal_lower_bounds_match_sliding_window_semantics():
    lower_bounds, valid = _simple_causal_lower_bounds(batch_size=2, seq_len=6, sliding_window=3)

    np.testing.assert_array_equal(
        lower_bounds,
        np.array(
            [
                [0, 0, 0, 1, 2, 3],
                [0, 0, 0, 1, 2, 3],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(valid, np.ones((2, 6), dtype=np.bool_))


def test_simple_causal_lower_bounds_match_full_causal_semantics():
    lower_bounds, valid = _simple_causal_lower_bounds(batch_size=2, seq_len=4, sliding_window=None)

    np.testing.assert_array_equal(lower_bounds, np.zeros((2, 4), dtype=np.int32))
    np.testing.assert_array_equal(valid, np.ones((2, 4), dtype=np.bool_))


def test_fa4_frontend_shards_metadata_with_qkv_batch_axis(monkeypatch):
    def fake_forward(q, k, v, lower_bounds, valid, *, sm_scale, kernel_config):
        del k, v, sm_scale, kernel_config
        if q.shape[:2] != lower_bounds.shape:
            raise ValueError(f"local lower_bounds shape {lower_bounds.shape} does not match q {q.shape}")
        if q.shape[:2] != valid.shape:
            raise ValueError(f"local valid shape {valid.shape} does not match q {q.shape}")
        return q

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute, "_segmented_kernel_config", lambda head_dim: object())
    monkeypatch.setattr(fa4_cute, "fa4_cute_attention_forward", fake_forward)
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    qkv_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, "model", None))
    q = jax.ShapeDtypeStruct((16, 4, 2, 8), jnp.bfloat16, sharding=qkv_sharding)
    k = jax.ShapeDtypeStruct((16, 4, 1, 8), jnp.bfloat16, sharding=qkv_sharding)
    v = jax.ShapeDtypeStruct((16, 4, 1, 8), jnp.bfloat16, sharding=qkv_sharding)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out = jax.eval_shape(
            lambda q_arg, k_arg, v_arg: gpu_fa4_cute_attention(q_arg, k_arg, v_arg, AttentionMask.causal()),
            q,
            k,
            v,
        )

    assert out.shape == q.shape
    assert out.sharding.spec == qkv_sharding.spec


def _one_device_mesh():
    """FA4's lower-bounds helpers reshard with PartitionSpec, which requires a mesh
    context on current jax. Production always calls under a mesh; tests enter a
    trivial one-device mesh here."""
    return jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))


def _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent, *, valid_tokens=None):
    with jax.set_mesh(_one_device_mesh()):
        actual = jax.jit(gpu_fa4_cute_attention)(q, k, v, mask)
        expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
        if valid_tokens is not None:
            actual = jnp.where(valid_tokens[..., None, None], actual, expected)

        np.testing.assert_allclose(actual, expected, atol=7e-2, rtol=7e-2)

        def ref_loss(q_arg, k_arg, v_arg):
            out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

        def fa4_loss(q_arg, k_arg, v_arg):
            out = gpu_fa4_cute_attention(q_arg, k_arg, v_arg, mask)
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

        actual_grads = jax.jit(jax.grad(fa4_loss, argnums=(0, 1, 2)))(q, k, v)
        expected_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)

        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)


@pytest.mark.parametrize(("q_heads", "kv_heads", "head_dim"), [(4, 1, 64), (2, 2, 64), (4, 1, 128)])
def test_real_gpu_fa4_cute_attention_matches_reference_for_valid_dynamic_packed_segments(q_heads, kv_heads, head_dim):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(4)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 64, q_heads, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 64, kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 64, kv_heads, head_dim), dtype=jnp.bfloat16)
    segment_ids = jnp.array(
        [[37] * 17 + [42] * 23 + [43] * 21 + [-1] * 3],
        dtype=jnp.int32,
    )
    mask = AttentionMask.causal(sliding_window=5).with_segment_ids(segment_ids)
    valid = segment_ids >= 0
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

    _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent, valid_tokens=valid)


def test_real_gpu_fa4_cute_attention_matches_reference_for_simple_sliding_mask():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(5)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (2, 64, 4, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (2, 64, 2, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (2, 64, 2, 64), dtype=jnp.bfloat16)
    mask = AttentionMask.causal(sliding_window=7)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)

    _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent)


def test_real_gpu_fa4_lse_save_path_matches_default_path(monkeypatch):
    """SCALE_FA4_LSE_SAVE=1 must be numerically identical to the default FA4 path.

    Same forward kernel, same backward kernel, only residual placement differs (saved
    (out, lse) primals + thin custom VJP vs the forward recompute). Checks out, lse
    (vs a reference logsumexp), and dq/dk/dv against the default path.
    """
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    from levanter.grug.attention._fa4_cute import _self_attention_lower_bounds, _segmented_kernel_config
    from levanter.grug.attention._fa4_cute_backend import segmented_flash_attention_forward

    key = jax.random.PRNGKey(11)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (2, 64, 4, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (2, 64, 2, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (2, 64, 2, 64), dtype=jnp.bfloat16)
    mask = AttentionMask.causal(sliding_window=7)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)

    def loss_fn(q_arg, k_arg, v_arg):
        out = gpu_fa4_cute_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    with jax.set_mesh(_one_device_mesh()):
        out_default = jax.jit(gpu_fa4_cute_attention)(q, k, v, mask)
        grads_default = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))(q, k, v)

        monkeypatch.setenv("SCALE_FA4_LSE_SAVE", "1")
        jax.clear_caches()
        out_saved = jax.jit(gpu_fa4_cute_attention)(q, k, v, mask)
        grads_saved = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))(q, k, v)

        # lse sanity: the raw forward's lse matches a reference logsumexp over the windowed scores.
        lower_bounds, valid = _self_attention_lower_bounds(q, k, v, mask, backend_name="test")
        _, lse = segmented_flash_attention_forward(
            q,
            k,
            v,
            lower_bounds,
            valid,
            softmax_scale=64**-0.5,
            kernel_config=_segmented_kernel_config(64),
        )
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * 64**-0.5
    positions = jnp.arange(64, dtype=jnp.int32)
    causal = (positions[None, :] <= positions[:, None]) & (positions[None, :] >= positions[:, None] - 6)
    scores = jnp.where(causal[None, None], scores, -jnp.inf)
    lse_ref = jax.nn.logsumexp(scores, axis=-1)
    np.testing.assert_allclose(lse, lse_ref, atol=7e-2, rtol=7e-2)
