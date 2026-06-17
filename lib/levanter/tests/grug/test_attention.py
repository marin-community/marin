# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.data.text import GrugLmExample
import levanter.grug.attention._fa4_thd as fa4_thd
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


def test_grug_lm_example_with_max_segments_stacks_padded_packed_rows():
    first = GrugLmExample.causal(
        jnp.arange(8, dtype=jnp.int32),
        segment_ids=jnp.array([0, 0, 1, 1, -1, -1, -1, -1], dtype=jnp.int32),
        max_segments=3,
    )
    second = GrugLmExample.causal(
        jnp.arange(8, dtype=jnp.int32),
        segment_ids=jnp.array([4, 4, 4, 5, 5, -1, -1, -1], dtype=jnp.int32),
        max_segments=3,
    )

    batch = jax.tree.map(lambda *xs: jnp.stack(xs), first, second)
    q_segment_ids, kv_segment_ids = batch.attn_mask.segment_ids
    metadata = batch.attn_mask.thd_segment_metadata

    assert batch.tokens.shape == (2, 8)
    np.testing.assert_array_equal(q_segment_ids, kv_segment_ids)
    assert metadata is not None
    np.testing.assert_array_equal(metadata.segment_lengths, jnp.array([[2, 2, 4], [3, 2, 3]], dtype=jnp.int32))
    np.testing.assert_array_equal(metadata.num_segments, jnp.array([3, 3], dtype=jnp.int32))


def test_thd_segment_metadata_sharding_follows_token_stream():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("data", "expert", "model"))
    token_sharding = NamedSharding(mesh, P(("data", "expert"), None, "model", None))
    q = jax.device_put(jnp.zeros((2, 4, 2, 8), dtype=jnp.float32), token_sharding)
    segment_lengths = jnp.array([[2, 2], [3, 1]], dtype=jnp.int32)
    num_segments = jnp.array([2, 2], dtype=jnp.int32)

    sharding = fa4_thd._segment_lengths_sharding(segment_lengths, num_segments, q)

    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == P(("data", "expert"), None)


def test_thd_global_prefix_sum_input_is_replicated():
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "expert"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    sharded = jax.device_put(
        jnp.array([[2, 2], [3, 1]], dtype=jnp.int32),
        NamedSharding(mesh, P(("data", "expert"), None)),
    )

    replicated = fa4_thd._replicate_for_global_prefix_sum(sharded)

    sharding = replicated.sharding
    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == P(None, None)


def test_thd_segment_metadata_rejects_mismatched_q_kv_segments():
    if jax.default_backend() == "tpu":
        pytest.skip("TPU checkify reports the error but does not raise a Python exception.")

    q_segment_ids = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    kv_segment_ids = jnp.array([0, 0, 2, 2], dtype=jnp.int32)

    @eqx.filter_jit
    def build_mask(q_ids, kv_ids):
        return AttentionMask.causal().with_segment_ids(q_ids, kv_ids, max_segments=2)

    with pytest.raises(Exception, match="matching q/kv segment_ids"):
        mask = build_mask(q_segment_ids, kv_segment_ids)
        assert mask.thd_segment_metadata is not None
        jax.block_until_ready(mask.thd_segment_metadata.segment_lengths)


def test_gpu_fa4_thd_rejects_mha_before_kernel_config(monkeypatch):
    monkeypatch.setattr(fa4_thd.jax, "default_backend", lambda: "gpu")

    q = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    k = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    v = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(segment_ids, max_segments=2)

    with pytest.raises(NotImplementedError, match="supports only GQA"):
        attention(q, k, v, mask, implementation="gpu_fa4_thd")


def test_gpu_fa4_thd_rejects_nonpositive_sliding_window():
    q = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    k = jnp.ones((1, 4, 1, 8), dtype=jnp.float32)
    v = jnp.ones((1, 4, 1, 8), dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 1, 1]], dtype=jnp.int32)

    zero_window = AttentionMask.causal(sliding_window=0).with_segment_ids(segment_ids, max_segments=2)
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        fa4_thd._validate_simple_causal_self_attention(q, k, v, zero_window, backend_name="gpu_fa4_thd_attention")


def test_gpu_fa4_thd_supports_hopper_kernel_config(monkeypatch):
    monkeypatch.setattr(fa4_thd, "_gpu_compute_arch", lambda: 90)

    config = fa4_thd._thd_kernel_config(128)

    assert config.forward_tile == (128, 64)
    assert config.backward_tile == (64, 128)
    assert config.num_threads == 384


def test_gpu_fa4_thd_hopper_postprocess_uses_mma_compatible_tile():
    assert fa4_thd._sm90_postprocess_tile_m(90, 128) == 64
    assert fa4_thd._sm90_postprocess_tile_m(90, 64) == 64
    assert fa4_thd._sm90_postprocess_tile_m(100, 128) == 128


def test_gpu_fa4_thd_hopper_backward_passes_smem_safe_options_to_kernel():
    captured_kwargs: dict[str, object] = {}

    # The upstream CUDA kernels are optional in unit tests; exercise the launcher
    # boundary where Marin passes SM90-safe options into flash-attn-4.
    class FakeCutlass:
        BFloat16 = object()
        Float16 = object()
        Float32 = object()

    class FakeCute:
        Tensor = object()

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def kernel(fn):
            return fn

    class FakeCuda:
        CUstream = object()

    class FakePreprocess:
        def __init__(self, *args, **kwargs):
            pass

    class FakeBackward:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

    class FakePostprocess:
        def __init__(self, *args, **kwargs):
            pass

    modules = fa4_thd._UpstreamFa4CuteModules(
        arch=90,
        cutlass=FakeCutlass,
        cute=FakeCute,
        cjax=object(),
        cuda=FakeCuda,
        FlashAttentionForward=object(),
        FlashAttentionBackward=FakeBackward,
        FlashAttentionBackwardPreprocess=FakePreprocess,
        FlashAttentionBackwardPostprocess=FakePostprocess,
    )

    fa4_thd._upstream_fa4_thd_backward_launcher(
        modules,
        dtype=jnp.dtype(jnp.bfloat16),
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=2,
        kernel_config=fa4_thd.Flash4CuteKernelConfig(
            forward_tile=(128, 64),
            backward_tile=(64, 128),
            num_threads=384,
        ),
        sliding_window=None,
    )

    assert captured_kwargs["PdS_stage"] == 1
    assert captured_kwargs["SdP_swapAB"] is True
    assert captured_kwargs["AtomLayoutNdKV"] == 2
    assert captured_kwargs["num_threads"] == 384


def test_attention_rejects_unknown_implementation():
    q, k, v = _make_qkv()

    with pytest.raises(ValueError, match="Unknown Grug attention implementation"):
        attention(q, k, v, AttentionMask.causal(), implementation="nope")  # type: ignore[arg-type]
