# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.data.text.examples import GrugLmExample
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


class _FakeLaunchable:
    def launch(self, **kwargs):
        return None


class _FakeFloat32:
    def __init__(self, value):
        self.value = value


class _FakeCutlass:
    BFloat16 = object()
    Float16 = object()
    Float32 = _FakeFloat32


class _FakeCute:
    Tensor = object

    @staticmethod
    def jit(fn):
        return fn

    @staticmethod
    def kernel(fn):
        return lambda *args, **kwargs: _FakeLaunchable()

    @staticmethod
    def ceil_div(numerator, denominator):
        return 1

    @staticmethod
    def size(tensor):
        return 1


class _FakeCuda:
    CUstream = object


class _Fa4KernelRecorder:
    """Fake FA4 kernel class recording the options it is built with and every traced invocation."""

    def __init__(self):
        self.constructor_kwargs: dict = {}
        self.invocations: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.constructor_kwargs.update(kwargs)
        return self._invoke

    def _invoke(self, *args, **kwargs):
        self.invocations.append(kwargs)


def _noop_kernel(*args, **kwargs):
    return None


def _fa4_thd_fake_modules(*, arch, forward=None, backward=None):
    """Build `_UpstreamFa4CuteModules` with the upstream CUDA kernels replaced by recorders.

    flash-attn-4 and the CuTe DSL are optional in unit tests, so the launcher boundary is exercised
    against fakes whose `cute.jit` is the identity.
    """
    return fa4_thd._UpstreamFa4CuteModules(
        arch=arch,
        cutlass=_FakeCutlass,
        cute=_FakeCute,
        cjax=object(),
        cuda=_FakeCuda,
        FlashAttentionForward=forward or _Fa4KernelRecorder(),
        FlashAttentionBackward=backward or _Fa4KernelRecorder(),
        FlashAttentionBackwardPreprocess=lambda *args, **kwargs: _noop_kernel,
        FlashAttentionBackwardPostprocess=lambda *args, **kwargs: _noop_kernel,
    )


_BLACKWELL_THD_KERNEL_CONFIG = fa4_thd.Flash4CuteKernelConfig(
    forward_tile=(128, 128),
    backward_tile=(128, 128),
    num_threads=384,
)


def test_gpu_fa4_thd_hopper_backward_passes_smem_safe_options_to_kernel():
    backward = _Fa4KernelRecorder()

    fa4_thd._upstream_fa4_thd_backward_launcher(
        _fa4_thd_fake_modules(arch=90, backward=backward),
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

    assert backward.constructor_kwargs["PdS_stage"] == 1
    assert backward.constructor_kwargs["SdP_swapAB"] is True
    assert backward.constructor_kwargs["AtomLayoutNdKV"] == 2
    assert backward.constructor_kwargs["num_threads"] == 384


@pytest.mark.parametrize(
    ("sliding_window", "window_size_left", "window_size_right"),
    [(None, None, None), (5, 4, 0)],
    ids=["full-causal", "sliding-window"],
)
def test_gpu_fa4_thd_launchers_pass_resolved_window_sizes(sliding_window, window_size_left, window_size_right):
    """Both launchers hand FA4 an already-resolved window pair from a single call site."""
    forward = _Fa4KernelRecorder()
    forward_launcher = fa4_thd._upstream_fa4_thd_forward_launcher(
        _fa4_thd_fake_modules(arch=100, forward=forward),
        dtype=jnp.dtype(jnp.bfloat16),
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=2,
        kernel_config=_BLACKWELL_THD_KERNEL_CONFIG,
        sliding_window=sliding_window,
    )
    forward_launcher(object(), object(), object(), object(), object(), object(), object(), softmax_scale=0.1)

    assert len(forward.invocations) == 1
    assert forward.invocations[0]["window_size_left"] == window_size_left
    assert forward.invocations[0]["window_size_right"] == window_size_right

    backward = _Fa4KernelRecorder()
    backward_launcher = fa4_thd._upstream_fa4_thd_backward_launcher(
        _fa4_thd_fake_modules(arch=100, backward=backward),
        dtype=jnp.dtype(jnp.bfloat16),
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=2,
        kernel_config=_BLACKWELL_THD_KERNEL_CONFIG,
        sliding_window=sliding_window,
    )
    backward_launcher(*[object() for _ in range(16)], softmax_scale=0.1)

    assert len(backward.invocations) == 1
    assert backward.invocations[0]["window_size_left"] == window_size_left
    assert backward.invocations[0]["window_size_right"] == window_size_right


@pytest.mark.parametrize("sliding_window", [None, 5], ids=["full-causal", "sliding-window"])
def test_real_gpu_fa4_thd_attention_matches_reference_value_and_gradients(sliding_window):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4 THD correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")

    q_key, k_key, v_key, cotangent_key = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(q_key, (1, 64, 4, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 64, 2, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 64, 2, 128), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    segment_ids = jnp.array([[3] * 31 + [8] * 33], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(segment_ids, max_segments=2)

    def fa4(q_arg, k_arg, v_arg):
        return attention(q_arg, k_arg, v_arg, mask, implementation="gpu_fa4_thd")

    def reference(q_arg, k_arg, v_arg):
        return reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)

    def weighted_sum(fn):
        return lambda *args: jnp.sum(fn(*args).astype(jnp.float32) * cotangent.astype(jnp.float32))

    np.testing.assert_allclose(jax.jit(fa4)(q, k, v), jax.jit(reference)(q, k, v), atol=7e-2, rtol=7e-2)

    actual_grads = jax.jit(jax.grad(weighted_sum(fa4), argnums=(0, 1, 2)))(q, k, v)
    expected_grads = jax.jit(jax.grad(weighted_sum(reference), argnums=(0, 1, 2)))(q, k, v)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)


def test_attention_rejects_unknown_implementation():
    q, k, v = _make_qkv()

    with pytest.raises(ValueError, match="Unknown Grug attention implementation"):
        attention(q, k, v, AttentionMask.causal(), implementation="nope")  # type: ignore[arg-type]
