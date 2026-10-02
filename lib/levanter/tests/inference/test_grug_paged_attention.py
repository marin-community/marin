# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention import ragged_paged_attention
from levanter.grug.attention._paged_gpu import gpu_paged_attention
from levanter.grug.attention._paged_tpu import _exp_nonpositive, tpu_paged_decode


class _PagedCase(NamedTuple):
    q: jax.Array
    kv_pages: jax.Array
    kv_lens: jax.Array
    page_indices: jax.Array
    cu_q_lens: jax.Array
    num_seqs: jax.Array


def _mixed_case(dtype):
    rng = np.random.default_rng(12)
    # Noncontiguous physical pages, partial final pages, inactive sequence and query padding.
    q = rng.normal(size=(8, 2, 3, 32)).astype(np.float32)
    pages = rng.normal(size=(7, 4, 4, 32)).astype(np.float32)
    pages[0] = np.nan
    indices = np.array([[4, 2, -1], [6, 1, 3], [-1, -1, -1]], np.int32)
    lengths = np.array([7, 10, -1], np.int32)
    offsets = np.array([0, 3, 4, -1], np.int32)
    args = (q, pages, lengths, indices, offsets, np.array(2, np.int32))
    return _PagedCase(*(jnp.asarray(x, dtype if i < 2 else None) for i, x in enumerate(args)))


def _dense_oracle(args, window, cap, scale):
    scale = np.float64(np.float32(scale))
    q, pages, lengths, indices, offsets, num_seqs = (
        np.asarray(x, dtype=np.float64 if i < 2 else None) for i, x in enumerate(args)
    )
    output = np.zeros_like(q)
    for seq in range(int(num_seqs)):
        length = int(lengths[seq])
        tokens = pages[indices[seq, : (length + pages.shape[1] - 1) // pages.shape[1]]].reshape(
            -1, pages.shape[2], pages.shape[3]
        )
        for token in range(int(offsets[seq]), int(offsets[seq + 1])):
            position = length - int(offsets[seq + 1]) + token
            begin = 0 if window is None else max(0, position - window + 1)
            keys = tokens[begin : position + 1, 0::2]
            values = tokens[begin : position + 1, 1::2]
            for head in range(q.shape[1]):
                logits = q[token, head] @ keys[:, head].T * scale
                if cap is not None:
                    logits = cap * np.tanh(logits / cap)
                probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                output[token, head] = probabilities @ values[:, head]
    return output


@pytest.mark.parametrize("window,cap", [(None, None), (1, None), (5, 1.5)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_grug_paged_attention_mixed_prefixes_match_dense(window, cap, dtype):
    args = _mixed_case(dtype)
    scale = 0.17
    expected = _dense_oracle(args, window, cap, scale).astype(np.asarray(args.q).dtype)
    fn = jax.jit(partial(ragged_paged_attention, sliding_window=window, soft_cap=cap, implementation="reference"))
    actual = fn(*args, sm_scale=jnp.array(scale))
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(np.asarray(actual, np.float32) - np.asarray(expected, np.float32))) < 1e-5
    empty = fn(*args._replace(num_seqs=jnp.array(0, jnp.int32)), sm_scale=jnp.array(scale))
    np.testing.assert_array_equal(empty, jnp.zeros_like(args.q))


def test_grug_paged_attention_explicit_head_sharding_matches_dense():
    args = _mixed_case(jnp.float32)
    devices = np.asarray(jax.devices()[:2])
    mesh = Mesh(devices, ("model",), axis_types=(AxisType.Explicit,))
    specs = (P(None, "model", None, None), P(None, None, "model", None), P(), P(), P(), P())
    with jax.set_mesh(mesh):
        sharded = tuple(jax.device_put(arg, NamedSharding(mesh, spec)) for arg, spec in zip(args, specs, strict=True))
        actual = jax.jit(partial(ragged_paged_attention, sm_scale=0.17, sliding_window=5, implementation="reference"))(
            *sharded
        )
        np.testing.assert_allclose(actual, _dense_oracle(args, 5, None, 0.17), atol=1e-5, rtol=1e-5)
        assert actual.sharding == sharded[0].sharding


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("runtime_scale", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("implementation", ["tpu", "tpu_fp32_tiles"])
def test_grug_tpu_paged_attention_matches_dense(window, runtime_scale, dtype, implementation):
    args = _mixed_case(dtype)
    fn = partial(ragged_paged_attention, sliding_window=window, implementation=implementation)
    actual = (
        jax.jit(fn)(*args, sm_scale=jnp.array(0.17)) if runtime_scale else jax.jit(partial(fn, sm_scale=0.17))(*args)
    )
    expected = _dense_oracle(args, window, None, 0.17).astype(np.asarray(args[0]).dtype)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-4, rtol=1e-4)


def _decode_case(dtype):
    rng = np.random.default_rng(41)
    q = jnp.asarray(rng.normal(size=(4, 2, 3, 32)), dtype)
    pages = jnp.asarray(rng.normal(size=(7, 16, 4, 32)), dtype).at[0].set(jnp.nan)
    return _PagedCase(
        q,
        pages,
        jnp.array([37, 18, -1], jnp.int32),
        jnp.array([[4, 2, 6], [1, 3, -1], [-1, -1, -1]], jnp.int32),
        jnp.array([0, 1, 2, -1], jnp.int32),
        jnp.array(2, jnp.int32),
    )


@pytest.mark.parametrize("window,cap", [(None, None), (1, None), (29, 1.5)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("av_precision", ["ieee", "bf16_3x"])
def test_grug_gpu_paged_attention_interpreter_matches_dense(window, cap, dtype, av_precision):
    with jax.default_device(jax.devices("cpu")[0]):
        args = _decode_case(dtype)
        q, pages, _, indices, _, _ = args
        upper = jnp.array([37, 18, 0, 0], jnp.int32)
        lower = jnp.zeros_like(upper) if window is None else jnp.maximum(0, upper - window)
        bounds = jnp.stack((lower, upper), axis=-1).at[2].set(jnp.array([1, 1], jnp.int32))
        table = jnp.maximum(indices[jnp.array([0, 1, 0, 0])], 0)
        actual = jax.jit(partial(gpu_paged_attention, soft_cap=cap, av_precision=av_precision, interpret=True))(
            q, pages, table, bounds, 0.17
        )
        expected = jnp.asarray(_dense_oracle(args, window, cap, 0.17), dtype).astype(jnp.float32)
        np.testing.assert_allclose(actual.astype(jnp.float32), expected, atol=1e-5, rtol=1e-5)
        assert np.max(np.abs(np.asarray(actual, np.float32) - np.asarray(expected))) < 1e-5


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU Pallas kernel")
@pytest.mark.parametrize("window", [None, 29])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("av_precision", ["ieee", "bf16_3x"])
def test_grug_gpu_paged_attention_decode_matches_dense(window, dtype, av_precision):
    args = _decode_case(dtype)
    compiled = jax.jit(
        partial(
            ragged_paged_attention,
            sm_scale=0.17,
            sliding_window=window,
            implementation="gpu_pallas" if av_precision == "ieee" else "gpu_pallas_bf16_3x",
        )
    )
    actual = compiled(*args)
    expected = jnp.asarray(_dense_oracle(args, window, None, 0.17), dtype).astype(jnp.float32)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected, atol=1e-4, rtol=1e-4)
    empty = compiled(*args._replace(num_seqs=jnp.array(0, jnp.int32)))
    np.testing.assert_array_equal(empty, jnp.zeros_like(args.q))
    prefill_args = args._replace(cu_q_lens=jnp.array([0, 2, 4, -1], jnp.int32))
    prefill = compiled(*prefill_args)
    expected_prefill = jnp.asarray(_dense_oracle(prefill_args, window, None, 0.17), dtype).astype(jnp.float32)
    np.testing.assert_allclose(prefill.astype(jnp.float32), expected_prefill, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kv_splits", [8, 16])
def test_grug_gpu_split_reduction_matches_dense(kv_splits):
    with jax.default_device(jax.devices("cpu")[0]):
        rng = np.random.default_rng(41)
        q = jnp.asarray(rng.normal(size=(1, 2, 3, 32)), jnp.float32)
        pages = jnp.asarray(rng.normal(size=(19, 16, 4, 32)), jnp.float32)
        table = jnp.arange(18, -1, -1, dtype=jnp.int32)[None]
        args = _PagedCase(q, pages, jnp.array([291]), table, jnp.array([0, 1]), jnp.array(1))
        actual = jax.jit(partial(gpu_paged_attention, kv_splits=kv_splits, interpret=True))(
            q, pages, table, jnp.array([[28, 291]]), 0.17
        )
        np.testing.assert_allclose(actual, _dense_oracle(args, 263, None, 0.17), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "implementation,dma_buffers",
    [
        pytest.param(
            "gpu_pallas_bf16_3x",
            1,
            marks=pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU Pallas kernel"),
        ),
        pytest.param(
            "tpu_fp32_tiles", 1, marks=pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
        ),
        pytest.param(
            "tpu_fp32_tiles", 2, marks=pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
        ),
    ],
)
def test_grug_full_shape_bf16_decode_matches_float64(implementation, dma_buffers):
    # This measured Hero shape crosses BF16 rounding boundaries where the
    # FP32 page-streaming reference and IEEE split-K kernel can disagree.
    batch, context, heads, groups, dim, page_size = 8, 4096, 12, 4, 128, 128
    q_key, cache_key = jax.random.split(jax.random.key(42))
    q = jax.random.normal(q_key, (batch, heads, groups, dim), jnp.bfloat16)
    page_count = batch * context // page_size
    pages = jax.random.normal(cache_key, (page_count, page_size, 2 * heads, dim), jnp.bfloat16)
    args = _PagedCase(
        q,
        pages,
        jnp.full((batch,), context, jnp.int32),
        jnp.arange(page_count, dtype=jnp.int32).reshape(batch, -1),
        jnp.arange(batch + 1, dtype=jnp.int32),
        jnp.array(batch, jnp.int32),
    )
    actual = jax.jit(
        partial(
            ragged_paged_attention,
            sm_scale=dim**-0.5,
            sliding_window=2048,
            implementation=implementation,
            tpu_dma_buffers=dma_buffers,
        )
    )(*args)
    expected = _dense_oracle(args, 2048, None, dim**-0.5).astype(np.asarray(q).dtype)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
@pytest.mark.parametrize("heads,groups", [(3, 4), (5, 1), (5, 4), (6, 8), (12, 4)])
@pytest.mark.parametrize("dma_buffers", [1, 2])
def test_grug_tpu_bf16_unaligned_heads_match_float64(heads, groups, dma_buffers):
    args = _unaligned_head_case(heads, groups)
    q = args.q
    actual = jax.jit(
        partial(
            ragged_paged_attention,
            sm_scale=128**-0.5,
            sliding_window=17,
            implementation="tpu_fp32_tiles",
            tpu_dma_buffers=dma_buffers,
        )
    )(*args)
    expected = _dense_oracle(args, 17, None, 128**-0.5).astype(np.asarray(q).dtype)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-4, rtol=1e-4)


def _unaligned_head_case(heads, groups):
    rng = np.random.default_rng(38)
    q = jnp.asarray(rng.normal(size=(2, heads, groups, 128)), jnp.bfloat16)
    pages = jnp.asarray(rng.normal(size=(7, 16, 2 * heads, 128)), jnp.bfloat16)
    # Poison slots outside the two windows, including both partial edge pages.
    pages = pages.at[2, :4].set(jnp.nan).at[6, 5:].set(jnp.nan).at[3, 0].set(jnp.nan).at[1, 2:].set(jnp.nan)
    return _PagedCase(
        q,
        pages,
        jnp.array([37, 18], jnp.int32),
        jnp.array([[4, 2, 6], [3, 1, 0]], jnp.int32),
        jnp.array([0, 1, 2], jnp.int32),
        jnp.array(2, jnp.int32),
    )


@pytest.mark.parametrize("dma_buffers", [1, 2])
def test_tpu_decode_interpreter_matches_float64(dma_buffers):
    with jax.default_device(jax.devices("cpu")[0]):
        args = _unaligned_head_case(5, 4)
        bounds = jnp.array([[20, 37], [1, 18]], jnp.int32)
        fn = jax.jit(partial(tpu_paged_decode, dma_buffers=dma_buffers, interpret=True))
        actual = fn(args.q, args.kv_pages, args.page_indices, bounds, 128**-0.5)
        expected = _dense_oracle(args, 17, None, 128**-0.5).astype(np.asarray(args.q).dtype)
        np.testing.assert_allclose(
            np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-4, rtol=1e-4
        )
        empty = fn(args.q, args.kv_pages, args.page_indices, jnp.zeros_like(bounds), 128**-0.5)
        np.testing.assert_array_equal(empty, 0)


def test_tpu_softmax_exp_matches_float64():
    # Include exponent range-reduction boundaries as well as the softmax range.
    values = np.concatenate((np.linspace(-87, 0, 10001), (np.arange(-125, 0) + 0.5) * np.log(2))).astype(np.float32)
    with jax.default_device(jax.devices("cpu")[0]):
        actual = jax.jit(_exp_nonpositive)(values)
        masked = jax.jit(_exp_nonpositive)(jnp.array([-jnp.inf, -100.0, 0.0], jnp.float32))
    np.testing.assert_allclose(actual, np.exp(values.astype(np.float64)), atol=0, rtol=1e-7)
    np.testing.assert_array_equal(masked, [0, 0, 1])
