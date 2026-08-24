# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

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
from levanter.grug.sharding import compact_grug_mesh


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


def test_packed_segment_causal_lower_bounds_carry_next_valid_bound_through_padding():
    segment_ids = jnp.array([[-1, -1, 7, 7, 8, 8, -1]], dtype=jnp.int32)

    lower_bounds, valid = fa4_cute._packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=7,
        sliding_window=None,
    )

    np.testing.assert_array_equal(lower_bounds, jnp.array([[2, 2, 2, 2, 4, 4, 7]], dtype=jnp.int32))
    np.testing.assert_array_equal(valid, jnp.array([[False, False, True, True, True, True, False]]))


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


_CONTEXT_PARALLEL_PARITY_SCRIPT = """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    import levanter.grug.attention._fa4_cute as fa4_cute
    from levanter.grug.attention import AttentionMask
    from levanter.grug.sharding import compact_grug_mesh

    BATCH, SEQ, Q_HEADS, KV_HEADS, HEAD_DIM = 8, 32, 4, 2, 8
    BATCH_AXES = ("replica_dcn", "data", "expert")
    SM_SCALE = HEAD_DIM**-0.5
    local_q_lengths = []

    def reference_forward(q, k, v, lower_bounds, valid, *, sm_scale, kernel_config, q_offset=None):
        # Stand-in for the CUTLASS kernel with the same metadata contract: local query i is at
        # global position i + q_offset, and lower_bounds/keys are in global positions.
        del kernel_config
        local_q_lengths.append(q.shape[1])
        offset = 0 if q_offset is None else q_offset[0]
        q_positions = jnp.arange(q.shape[1], dtype=jnp.int32)[None, :, None] + offset
        k_positions = jnp.arange(k.shape[1], dtype=jnp.int32)[None, None, :]
        allowed = valid[:, :, None] & (lower_bounds[:, :, None] <= k_positions) & (k_positions <= q_positions)
        repeats = q.shape[2] // k.shape[2]
        k_full = jnp.repeat(k, repeats, axis=2)
        v_full = jnp.repeat(v, repeats, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_full) * sm_scale
        scores = jnp.where(allowed[:, None, :, :], scores, -1e30)
        weights = jnp.where(allowed[:, None, :, :], jax.nn.softmax(scores, axis=-1), 0.0)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v_full)

    fa4_cute.fa4_cute_attention_forward = reference_forward

    def unsharded_attention(q, k, v, lower_bounds, valid):
        return reference_forward(q, k, v, lower_bounds, valid, sm_scale=SM_SCALE, kernel_config=None)

    def sharded_attention(q, k, v, lower_bounds, valid):
        return fa4_cute._fa4_cute_attention_forward_sharded(
            q, k, v, lower_bounds, valid, sm_scale=SM_SCALE, kernel_config=None
        )

    def cotangent_loss(attention_fn):
        def loss(q, k, v, lower_bounds, valid, cotangent):
            return jnp.sum(attention_fn(q, k, v, lower_bounds, valid) * cotangent)

        return jax.grad(loss, argnums=(0, 1, 2))

    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (BATCH, SEQ, Q_HEADS, HEAD_DIM))
    k = jax.random.normal(keys[1], (BATCH, SEQ, KV_HEADS, HEAD_DIM))
    v = jax.random.normal(keys[2], (BATCH, SEQ, KV_HEADS, HEAD_DIM))
    cotangent = jax.random.normal(keys[3], (BATCH, SEQ, Q_HEADS, HEAD_DIM))
    # Packed documents of unequal length plus trailing padding, so the bounds vary per row.
    segment_ids = jnp.asarray(
        [[3] * 7 + [4] * 13 + [5] * 9 + [-1] * 3] * (BATCH // 2) + [[6] * 20 + [7] * 12] * (BATCH // 2),
        dtype=jnp.int32,
    )

    for sliding_window in (None, 5):
        mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(segment_ids)
        lower_bounds, valid = fa4_cute.fa4_cute_segment_bounds(
            mask, batch_size=BATCH, seq_len=SEQ, sliding_window=sliding_window
        )

        # The same bounds must come out of context-sharded segment ids, which is how the
        # model hands them over once the residual stream itself is sequence-sharded.
        sharded_mesh = compact_grug_mesh(context_axis_size=4)
        with jax.set_mesh(sharded_mesh):
            sharded_ids = jax.device_put(segment_ids, NamedSharding(sharded_mesh, P(BATCH_AXES, "context")))
            sharded_bounds, sharded_valid = jax.jit(
                lambda ids: fa4_cute.fa4_cute_segment_bounds(
                    AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(ids),
                    batch_size=BATCH,
                    seq_len=SEQ,
                    sliding_window=sliding_window,
                )
            )(sharded_ids)
        np.testing.assert_array_equal(np.asarray(sharded_bounds), np.asarray(lower_bounds))
        np.testing.assert_array_equal(np.asarray(sharded_valid), np.asarray(valid))
        expected = unsharded_attention(q, k, v, lower_bounds, valid)
        expected_grads = cotangent_loss(unsharded_attention)(q, k, v, lower_bounds, valid, cotangent)

        for context_axis_size in (1, 2, 4):
            mesh = compact_grug_mesh(context_axis_size=context_axis_size)
            seq_axis = "context" if context_axis_size > 1 else None
            q_sharding = NamedSharding(mesh, P(BATCH_AXES, seq_axis, "model", None))
            kv_sharding = NamedSharding(mesh, P(BATCH_AXES, None, "model", None))
            metadata_sharding = NamedSharding(mesh, P(BATCH_AXES, None))
            args = (
                jax.device_put(q, q_sharding),
                jax.device_put(k, kv_sharding),
                jax.device_put(v, kv_sharding),
                jax.device_put(lower_bounds, metadata_sharding),
                jax.device_put(valid, metadata_sharding),
            )
            del local_q_lengths[:]
            with jax.set_mesh(mesh):
                actual = jax.jit(sharded_attention)(*args)
                actual_grads = jax.jit(cotangent_loss(sharded_attention))(
                    *args, jax.device_put(cotangent, q_sharding)
                )

            case = f"sliding_window={sliding_window} context_axis_size={context_axis_size}"
            assert local_q_lengths[0] == SEQ // context_axis_size, f"{case}: {local_q_lengths}"
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
            for actual_grad, expected_grad, name in zip(actual_grads, expected_grads, "qkv", strict=True):
                np.testing.assert_allclose(
                    np.asarray(actual_grad),
                    np.asarray(expected_grad),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"d{name} mismatch for {case}",
                )
"""


def test_context_sharded_attention_matches_unsharded_values_and_gradients():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CONTEXT_PARALLEL_PARITY_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent, *, valid_tokens=None):
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


@pytest.mark.parametrize("sliding_window", [None, 31])
def test_real_gpu_fa4_cute_attention_matches_reference_with_leading_padding(sliding_window):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(6)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 128, 20, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 128, 5, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 128, 5, 128), dtype=jnp.bfloat16)
    segment_ids = jnp.array([[-1] * 19 + [37] * 109], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(segment_ids)
    valid = segment_ids >= 0
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

    _assert_real_gpu_fa4_cute_matches_reference(q, k, v, mask, cotangent, valid_tokens=valid)


def test_real_gpu_fa4_cute_attention_matches_reference_with_context_sharded_queries():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    if jax.device_count() < 2:
        pytest.skip("Context-parallel FA4/CuTe needs at least two devices.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(7)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 128, 4, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 128, 1, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 128, 1, 64), dtype=jnp.bfloat16)
    segment_ids = jnp.array([[11] * 53 + [12] * 71 + [-1] * 4], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=17).with_segment_ids(segment_ids)
    valid = segment_ids >= 0
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

    mesh = compact_grug_mesh(context_axis_size=2)
    batch_axes = ("replica_dcn", "data", "expert")
    q_sharding = NamedSharding(mesh, P(batch_axes, "context", "model", None))
    kv_sharding = NamedSharding(mesh, P(batch_axes, None, "model", None))
    with jax.set_mesh(mesh):
        _assert_real_gpu_fa4_cute_matches_reference(
            jax.device_put(q, q_sharding),
            jax.device_put(k, kv_sharding),
            jax.device_put(v, kv_sharding),
            mask,
            jax.device_put(cotangent, q_sharding),
            valid_tokens=valid,
        )


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
