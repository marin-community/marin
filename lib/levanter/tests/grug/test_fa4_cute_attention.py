# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh

import levanter.grug.attention._fa4_cute as fa4_cute
import levanter.grug.attention._fa4_cute_backend as fa4_cute_backend
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    attention,
    gpu_fa4_cute_attention,
    reference_attention,
)
from levanter.grug.attention._fa4_cute import _segmented_kernel_config, _simple_causal_lower_bounds
from levanter.grug.attention._fa4_cute_config import SM100_GQA_RATIOS, SM100_HEAD_DIM, sm100_flash4_cute_kernel_config
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.cpu_devices import run_on_cpu_devices


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
        key_sequence_length=8,
        q_offset=jnp.zeros((1,), dtype=jnp.int32),
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


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("tile", [(2, 4), (4, 2), (4, 4)])
@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("query_slice", [(0, 11), (0, 5), (3, 8), (7, 11)])
def test_packed_sparse_blocks_match_dense_mask(direction, tile, window, query_slice):
    ids = np.array([[-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1], [-1] * 11], dtype=np.int32)
    lower, valid = fa4_cute._packed_segment_causal_lower_bounds(
        jnp.asarray(ids), batch_size=2, seq_len=11, sliding_window=window
    )
    build = (
        fa4_cute_backend._packed_segment_forward_block_sparse_indices_with_full
        if direction == "forward"
        else fa4_cute_backend._packed_segment_backward_block_sparse_indices_with_full
    )
    tile_m, tile_n = tile
    start, stop = query_slice
    sparse = build(
        lower[:, start:stop],
        valid[:, start:stop],
        key_sequence_length=ids.shape[1],
        q_offset=jnp.array([start], dtype=jnp.int32),
        tile_m=tile_m,
        tile_n=tile_n,
    )
    query = np.arange(ids.shape[1])[:, None]
    key = np.arange(ids.shape[1])[None, :]
    dense = (ids[:, :, None] == ids[:, None, :]) & (ids[:, :, None] >= 0) & (key <= query)
    if window is not None:
        dense &= key >= query - window + 1
    dense = dense[:, start:stop, :]
    query_blocks = (stop - start + tile_m - 1) // tile_m
    key_blocks = (ids.shape[1] + tile_n - 1) // tile_n
    partial = np.zeros((2, key_blocks, query_blocks), dtype=bool)
    full = np.zeros_like(partial)
    for batch in range(2):
        for q_block in range(query_blocks):
            for k_block in range(key_blocks):
                block = dense[
                    batch, q_block * tile_m : (q_block + 1) * tile_m, k_block * tile_n : (k_block + 1) * tile_n
                ]
                full[batch, k_block, q_block] = block.shape[0] == tile_m and block.all()
                partial[batch, k_block, q_block] = block.any() and not full[batch, k_block, q_block]
    if direction == "forward":
        partial, full = partial.swapaxes(1, 2), full.swapaxes(1, 2)
    for expected, counts, indices in (
        (partial, sparse.partial_block_cnt, sparse.partial_block_idx),
        (full, sparse.full_block_cnt, sparse.full_block_idx),
    ):
        counts, indices = np.asarray(counts), np.asarray(indices)
        for batch in range(2):
            for block in range(expected.shape[1]):
                np.testing.assert_array_equal(
                    indices[batch, 0, block, : counts[batch, 0, block]],
                    np.flatnonzero(expected[batch, block]),
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
    def fake_forward(q, k, v, lower_bounds, valid, *, sm_scale, kernel_config, q_offset):
        del k, v, sm_scale, kernel_config, q_offset
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


def _fake_unsharded_forward(q, k, v, lower_bounds, valid, *, sm_scale, kernel_config, q_offset):
    del k, v, lower_bounds, valid, sm_scale, kernel_config, q_offset
    return q


def test_fa4_rejects_kv_batch_sharded_over_query_sequence_axis(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute, "_segmented_kernel_config", lambda head_dim: object())
    monkeypatch.setattr(fa4_cute, "fa4_cute_attention_forward", _fake_unsharded_forward)
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    q = jax.ShapeDtypeStruct(
        (16, 4, 2, 8), jnp.bfloat16, sharding=NamedSharding(mesh, P(("replica_dcn", "expert"), "data", "model", None))
    )
    kv_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, "model", None))
    k = jax.ShapeDtypeStruct((16, 4, 1, 8), jnp.bfloat16, sharding=kv_sharding)
    v = jax.ShapeDtypeStruct((16, 4, 1, 8), jnp.bfloat16, sharding=kv_sharding)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        with pytest.raises(ValueError, match="match q's batch/head sharding"):
            jax.eval_shape(
                lambda q_arg, k_arg, v_arg: gpu_fa4_cute_attention(q_arg, k_arg, v_arg, AttentionMask.causal()),
                q,
                k,
                v,
            )


@pytest.mark.parametrize("context_entry", ["context", ("context", "model")])
def test_fa4_accepts_unit_context_axis(monkeypatch, context_entry):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute, "_segmented_kernel_config", lambda head_dim: object())
    monkeypatch.setattr(fa4_cute, "fa4_cute_attention_forward", _fake_unsharded_forward)
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 1, 8, 1),
        axis_names=("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    batch_axes = ("replica_dcn", "data", "expert")
    q_sharding = NamedSharding(mesh, P(batch_axes, context_entry, None, None))
    # Naming the length-1 axis on K/V's batch dim partitions nothing either.
    kv_sharding = NamedSharding(mesh, P((*batch_axes, "context"), None, "model", None))
    q = jax.ShapeDtypeStruct((16, 4, 2, 8), jnp.bfloat16, sharding=q_sharding)
    k = jax.ShapeDtypeStruct((16, 4, 1, 8), jnp.bfloat16, sharding=kv_sharding)
    v = jax.ShapeDtypeStruct((16, 4, 1, 8), jnp.bfloat16, sharding=kv_sharding)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out = jax.eval_shape(
            lambda q_arg, k_arg, v_arg: gpu_fa4_cute_attention(q_arg, k_arg, v_arg, AttentionMask.causal()),
            q,
            k,
            v,
        )

    assert out.shape == q.shape
    assert out.sharding.spec == q_sharding.spec


def test_fa4_precomputed_bounds_reject_mismatched_context_lengths(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute, "_segmented_kernel_config", lambda head_dim: object())
    mesh = AbstractMesh((2,), ("context",), axis_types=(AxisType.Explicit,))
    q = jax.ShapeDtypeStruct((1, 8, 2, 8), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, "context")))
    k = jax.ShapeDtypeStruct((1, 4, 1, 8), jnp.bfloat16, sharding=NamedSharding(mesh, P()))
    v = jax.ShapeDtypeStruct((1, 4, 1, 8), jnp.bfloat16, sharding=NamedSharding(mesh, P()))

    def forward(q, k, v):
        # Precomputed metadata bypasses the ordinary mask-building validation.
        mask = AttentionMask.causal().with_fa4_bounds(jnp.zeros((1, 8), jnp.int32), jnp.ones((1, 8), jnp.bool_))
        return gpu_fa4_cute_attention(q, k, v, mask)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        with pytest.raises(ValueError, match="q_len == k_len globally"):
            jax.eval_shape(forward, q, k, v)


_CONTEXT_METADATA_SCRIPT = """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

    from levanter.grug.attention import AttentionMask
    from levanter.grug.attention._fa4_cute import fa4_cute_segment_bounds

    segment_ids = jnp.asarray(
        [[3] * 7 + [4] * 13 + [5] * 9 + [-1] * 3, [6] * 20 + [7] * 12],
        dtype=jnp.int32,
    )
    for window in (None, 5):
        def bounds(ids):
            return fa4_cute_segment_bounds(
                AttentionMask.causal(sliding_window=window).with_segment_ids(ids),
                batch_size=2, seq_len=32, sliding_window=window,
            )

        expected = bounds(segment_ids)
        for context_size in (1, 2, 4):
            mesh = Mesh(
                np.asarray(jax.devices()).reshape(8 // context_size, context_size),
                ("data", "context"), axis_types=(AxisType.Explicit,) * 2,
            )
            with jax.set_mesh(mesh):
                ids = jax.device_put(segment_ids, NamedSharding(mesh, P(None, "context")))
                actual = jax.jit(bounds)(ids)
            for actual_array, expected_array in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(np.asarray(actual_array), np.asarray(expected_array))
"""


def test_context_sharded_segment_ids_preserve_global_bounds():
    run_on_cpu_devices(_CONTEXT_METADATA_SCRIPT, device_count=8)


@pytest.mark.parametrize(
    ("arch", "q_heads", "head_dim", "dtype"),
    [
        (90, 8, 128, jnp.bfloat16),
        (120, 8, 128, jnp.bfloat16),
        (100, 2, 128, jnp.bfloat16),
        (100, 8, 64, jnp.bfloat16),
        (100, 8, 128, jnp.float16),
    ],
)
def test_fa4_sm100_attention_rejects_unsupported_layouts(monkeypatch, arch, q_heads, head_dim, dtype):
    q = jnp.zeros((1, 1, q_heads, head_dim), dtype=dtype)
    kv = jnp.zeros((1, 1, 1, head_dim), dtype=dtype)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute, "gpu_compute_capability", lambda: arch)
    monkeypatch.setattr(fa4_cute, "fa4_cute_attention_forward", lambda q, *_args, **_kwargs: q)

    with pytest.raises(ValueError, match="gpu_fa4_cute_sm100"):
        attention(q, kv, kv, AttentionMask.causal(), implementation="gpu_fa4_cute_sm100")


@pytest.mark.parametrize("arch", [100, 103])
def test_fa4_sm100_attention_accepts_blackwell_compute_capabilities(monkeypatch, arch):
    q = jnp.zeros((1, 1, 8, 128), dtype=jnp.bfloat16)
    kv = jnp.zeros((1, 1, 1, 128), dtype=jnp.bfloat16)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fa4_cute, "gpu_compute_capability", lambda: arch)
    monkeypatch.setattr(fa4_cute, "fa4_cute_attention_forward", lambda q, *_args, **_kwargs: q)

    out = attention(q, kv, kv, AttentionMask.causal(), implementation="gpu_fa4_cute_sm100")

    assert out.shape == q.shape


@pytest.mark.parametrize(
    ("q_heads", "head_dim", "dtype"), [(2, 128, jnp.bfloat16), (8, 64, jnp.bfloat16), (8, 128, jnp.float16)]
)
def test_fa4_sm100_backward_rejects_unsupported_layouts(q_heads, head_dim, dtype):
    # The native SM100 config must not fall back to the port backward, whose fields it leaves unset.
    seq_len = 4
    q = jnp.zeros((1, seq_len, q_heads, head_dim), dtype=dtype)
    kv = jnp.zeros((1, seq_len, 1, head_dim), dtype=dtype)
    lower_bounds = jnp.zeros((1, seq_len), dtype=jnp.int32)
    valid = jnp.ones((1, seq_len), dtype=jnp.bool_)
    lse = jnp.zeros((1, q_heads, seq_len), dtype=jnp.float32)

    with pytest.raises(ValueError, match="gpu_fa4_cute_sm100"):
        fa4_cute_backend.segmented_flash_attention_backward(
            q,
            kv,
            kv,
            q,
            q,
            lse,
            lower_bounds,
            valid,
            softmax_scale=1.0,
            kernel_config=sm100_flash4_cute_kernel_config(),
            q_offset=jnp.zeros((1,), dtype=jnp.int32),
        )


def _assert_real_gpu_fa4_cute_matches_reference(
    q,
    k,
    v,
    mask,
    cotangent,
    *,
    valid_tokens=None,
    implementation: GrugAttentionImplementation = "gpu_fa4_cute",
):
    def fa4(q_arg, k_arg, v_arg):
        return attention(q_arg, k_arg, v_arg, mask, implementation=implementation)

    actual = jax.jit(fa4)(q, k, v)
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    if valid_tokens is not None:
        actual = jnp.where(valid_tokens[..., None, None], actual, expected)

    np.testing.assert_allclose(actual, expected, atol=7e-2, rtol=7e-2)

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def fa4_loss(q_arg, k_arg, v_arg):
        out = attention(q_arg, k_arg, v_arg, mask, implementation=implementation)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual_grads = jax.jit(jax.grad(fa4_loss, argnums=(0, 1, 2)))(q, k, v)
    expected_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)


def test_real_gpu_fa4_cute_sm100_attention_matches_reference():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    if fa4_cute.gpu_compute_capability() != 100:
        pytest.skip("This FA4 backend requires SM100.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(7)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 128, 4, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 128, 1, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 128, 1, 128), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)

    _assert_real_gpu_fa4_cute_matches_reference(
        q,
        k,
        v,
        AttentionMask.causal(),
        cotangent,
        implementation="gpu_fa4_cute_sm100",
    )


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


_SEQUENCE_SHARDED_LAYOUTS = [(4, 1, 64), (8, 2, 128), (4, 4, 128), (6, 1, 128), (8, 1, 128)]
_SEQUENCE_SHARDED_CASES = [("gpu_fa4_cute", *layout) for layout in _SEQUENCE_SHARDED_LAYOUTS] + [
    ("gpu_fa4_cute_sm100", q_heads, kv_heads, head_dim)
    for q_heads, kv_heads, head_dim in _SEQUENCE_SHARDED_LAYOUTS
    if head_dim == SM100_HEAD_DIM and q_heads // kv_heads in SM100_GQA_RATIOS
]


@pytest.mark.parametrize(("implementation", "q_heads", "kv_heads", "head_dim"), _SEQUENCE_SHARDED_CASES)
@pytest.mark.parametrize(
    ("context_size", "sequence_axes"),
    [
        (1, ("context",)),
        (2, ("context",)),
        (4, ("context",)),
        (2, ("data",)),
        (4, ("context", "data")),
        (4, ("data", "context")),
    ],
)
@pytest.mark.parametrize("mask_kind", ["causal", "window", "packed"])
def test_real_gpu_fa4_cute_attention_matches_reference_with_sequence_sharded_queries(
    q_heads, kv_heads, head_dim, context_size, sequence_axes, implementation, mask_kind
):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    if implementation == "gpu_fa4_cute_sm100" and fa4_cute.gpu_compute_capability() != 100:
        pytest.skip("Native SM100 requires SM100.")
    if jax.device_count() < context_size:
        pytest.skip(f"Context-parallel FA4/CuTe needs at least {context_size} devices.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    if (
        context_size > 1
        and head_dim == 128
        and q_heads != kv_heads
        and _segmented_kernel_config(head_dim).sm90_backward is not None
    ):
        pytest.skip("The native SM90 GQA backward carries no context-parallel query offset.")
    # Multiple query tiles exercise offset bounds in both forward and backward kernels.
    seq_len = 512
    if sequence_axes == ("context",):
        mesh = compact_grug_mesh(replica_axis_size=1, context_axis_size=context_size)
        batch_axes = ("replica_dcn", "data", "expert")
        head_axis = "model"
    else:
        sequence_shape = (context_size,) if len(sequence_axes) == 1 else (2, 2)
        # Fix mesh order so reversing sequence_axes tests PartitionSpec ordering.
        axis_names = ("batch", *sorted(sequence_axes))
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(-1, *sequence_shape),
            axis_names,
            axis_types=(AxisType.Explicit,) * len(axis_names),
        )
        batch_axes = ("batch",)
        head_axis = None
    # Use one sequence per batch coordinate.
    batch = math.prod(mesh.shape[axis] for axis in batch_axes)
    key = jax.random.PRNGKey(7)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (batch, seq_len, q_heads, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (batch, seq_len, kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (batch, seq_len, kv_heads, head_dim), dtype=jnp.bfloat16)
    segment_ids = jnp.broadcast_to(jnp.array([[11] * 213 + [12] * 291 + [-1] * 8], dtype=jnp.int32), (batch, seq_len))
    mask = AttentionMask.causal(sliding_window=129 if mask_kind != "causal" else None)
    if mask_kind == "packed":
        mask = mask.with_segment_ids(segment_ids)
    valid = segment_ids >= 0 if mask_kind == "packed" else jnp.ones_like(segment_ids, dtype=jnp.bool_)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

    q_sharding = NamedSharding(mesh, P(batch_axes, sequence_axes, head_axis, None))
    kv_sharding = NamedSharding(mesh, P(batch_axes, None, head_axis, None))
    with jax.set_mesh(mesh):
        _assert_real_gpu_fa4_cute_matches_reference(
            jax.device_put(q, q_sharding),
            jax.device_put(k, kv_sharding),
            jax.device_put(v, kv_sharding),
            mask,
            jax.device_put(cotangent, q_sharding),
            valid_tokens=valid,
            implementation=implementation,
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


@pytest.mark.parametrize("sliding_window", [None, 31])
@pytest.mark.slow
@pytest.mark.timeout(180)
def test_real_gpu_fa4_cute_zeroes_padding_tiles_before_reusing_query_storage(sliding_window):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    keys = jax.random.split(jax.random.PRNGKey(73), 4)
    sequence_length = 8520
    q = jax.random.normal(keys[0], (2, sequence_length, 20, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (2, sequence_length, 5, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (2, sequence_length, 5, 128), dtype=jnp.bfloat16)
    # The full query grid reproduces the asynchronous copy race; small grids may
    # finish their copies before O overwrites shared storage even without a wait.
    valid_prefix = [37] * 17 + [42] * 23
    ids = jnp.array(
        [valid_prefix + [-1] * (sequence_length - len(valid_prefix)), [-1] * sequence_length], dtype=jnp.int32
    )
    mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(ids)
    valid = ids >= 0

    def forward(q, k, v):
        return attention(q, k, v, mask, implementation="gpu_fa4_cute")

    compiled = jax.jit(forward)
    first = compiled(q, k, v)
    np.testing.assert_array_equal(np.asarray(first)[~np.asarray(valid)], 0)
    for _ in range(10):
        repeated = compiled(q, k, v)
        np.testing.assert_array_equal(repeated, first)
    cotangent = jax.random.normal(keys[3], q.shape, dtype=jnp.bfloat16)
    gradients = jax.jit(
        jax.grad(lambda q, k, v: jnp.sum(forward(q, k, v).astype(jnp.float32) * cotangent), argnums=(0, 1, 2))
    )(q, k, v)
    for gradient in gradients:
        np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(valid)], 0)
    # Only the first 40 tokens are valid. A bounded dense reference covers every
    # active output and gradient without constructing an 8520-squared score map.
    reference_mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(ids[:1, :40])
    short_qkv = (q[:1, :40], k[:1, :40], v[:1, :40])
    expected = reference_attention(*short_qkv, reference_mask, logits_dtype=jnp.float32)
    np.testing.assert_allclose(first[:1, :40], expected, atol=7e-2, rtol=7e-2)

    def reference_loss(q, k, v):
        output = reference_attention(q, k, v, reference_mask, logits_dtype=jnp.float32)
        return jnp.sum(output.astype(jnp.float32) * cotangent[:1, :40])

    expected_gradients = jax.jit(jax.grad(reference_loss, argnums=(0, 1, 2)))(*short_qkv)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(actual[:1, :40], expected, atol=7e-2, rtol=7e-2)


@pytest.mark.parametrize(("query_heads", "kv_heads"), [(48, 6), (48, 12), (6, 1)])
@pytest.mark.parametrize(("sequence_length", "sliding_window"), [(257, None), (257, 31), (2305, 2048)])
@pytest.mark.timeout(300)
@pytest.mark.parametrize("implementation", ["gpu_fa4_cute", "gpu_fa4_cute_sm100"])
def test_real_gpu_fa4_cute_sm100_gradients_with_changing_packed_segments(
    query_heads, kv_heads, sequence_length, sliding_window, implementation
):
    if jax.default_backend() != "gpu" or fa4_cute.gpu_compute_capability() != 100:
        pytest.skip("Native SM100 backward correctness requires an SM100 GPU.")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_sm100")

    def output_and_gradients(q, k, v, cotangent, ids, *, implementation):
        mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(ids)

        def loss(q, k, v):
            output = attention(q, k, v, mask, implementation=implementation)
            # Reference attention uses a finite softmax sentinel for fully masked
            # rows. Zero those outputs to match the packed attention contract.
            if implementation == "reference":
                output = jnp.where((ids >= 0)[..., None, None], output, 0)
            return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32)), output

        (_, output), gradients = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(q, k, v)
        return (output, *gradients)

    actual_call = jax.jit(lambda *args: output_and_gradients(*args, implementation=implementation))
    reference_call = jax.jit(lambda *args: output_and_gradients(*args, implementation="reference"))
    batch = 2 if sequence_length == 257 else 1
    for iteration in range(3):
        positions = np.arange(sequence_length)
        boundaries = np.array([101] if sequence_length > 2048 else [31, 129, 193])
        ids = np.stack([np.searchsorted(boundaries + iteration + row * 7, positions) for row in range(batch)])
        ids[:, : 19 + iteration] = -1
        ids[:, -17:] = -1
        if batch == 2 and iteration == 2:
            ids[1, :] = -1
        query_shape = (batch, sequence_length, query_heads, 128)
        kv_shape = (batch, sequence_length, kv_heads, 128)
        keys = jax.random.split(jax.random.key(20260916 + iteration), 4)
        q, k, v, cotangent = (
            jax.random.normal(key, shape, dtype=jnp.bfloat16)
            for key, shape in zip(keys, (query_shape, kv_shape, kv_shape, query_shape), strict=True)
        )
        # Reuse each executable with changed masks and nonzero padded cotangents
        # to expose stale accumulator contents between invocations.
        args = (q, k, v, cotangent, jnp.asarray(ids, dtype=jnp.int32))
        actual = actual_call(*args)
        expected = reference_call(*args)
        for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected, strict=True):
            got = np.asarray(got, dtype=np.float32)
            want = np.asarray(want, dtype=np.float32)
            difference = np.abs(got - want)
            error = f"{name}: max absolute error {difference.max()}, mean {difference.mean()}"
            np.testing.assert_allclose(got, want, atol=7e-2, rtol=7e-2, err_msg=error)
            np.testing.assert_array_equal(got[ids < 0], 0, err_msg=name)
