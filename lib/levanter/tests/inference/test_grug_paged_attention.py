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
from levanter.inference.utils import INVALID
from levanter.testing.devices import use_test_devices


class _PagedCase(NamedTuple):
    q: jax.Array
    kv_pages: jax.Array
    kv_lens: jax.Array
    page_indices: jax.Array
    cu_q_lens: jax.Array
    num_seqs: jax.Array


_SCALE = 0.17


def _mixed_case(dtype, kv_heads=2):
    rng = np.random.default_rng(12)
    q = rng.normal(size=(8, kv_heads, 3, 32)).astype(np.float32)
    pages = rng.normal(size=(7, 4, 2 * kv_heads, 32)).astype(np.float32)
    pages[0] = np.nan
    indices = np.array([[4, 2, -1], [6, 1, 3], [-1, -1, -1]], np.int32)
    lengths = np.array([7, 10, -1], np.int32)
    offsets = np.array([0, 3, 4, -1], np.int32)
    args = (q, pages, lengths, indices, offsets, np.array(2, np.int32))
    return _PagedCase(*(jnp.asarray(x, dtype if i < 2 else None) for i, x in enumerate(args)))


def _dense_reference(args, window, cap, scale):
    q, pages, lengths, indices, offsets, num_seqs = (np.asarray(x) for x in args)
    output = np.zeros_like(q, dtype=np.float32)
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
                logits = q[token, head].astype(np.float32) @ keys[:, head].T * np.float32(scale)
                if cap is not None:
                    logits = np.float32(cap) * np.tanh(logits / np.float32(cap))
                logits -= np.max(logits, axis=-1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                output[token, head] = probabilities @ values[:, head]
    return output.astype(q.dtype)


@pytest.mark.parametrize("window,cap", [(None, None), (1, None), (5, 1.5)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_grug_paged_attention_mixed_prefixes_match_reference(window, cap, dtype):
    args = _mixed_case(dtype)
    fn = jax.jit(partial(ragged_paged_attention, sliding_window=window, soft_cap=cap, implementation="reference"))
    actual = fn(*args, sm_scale=jnp.array(_SCALE))
    expected = _dense_reference(args, window, cap, _SCALE)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-5, rtol=1e-5)


def test_grug_paged_attention_empty_batch_returns_zeros():
    args = _mixed_case(jnp.float32)._replace(num_seqs=jnp.array(0, jnp.int32))
    fn = jax.jit(partial(ragged_paged_attention, implementation="reference", sm_scale=_SCALE))
    empty = fn(*args)
    np.testing.assert_array_equal(empty, jnp.zeros_like(args.q))


@pytest.mark.parametrize("num_devices,cache_head_axis", [(2, "model"), (1, None)])
def test_grug_paged_attention_explicit_head_sharding_matches_reference(num_devices, cache_head_axis):
    with use_test_devices(num_devices) as devices:
        args = _mixed_case(jnp.float32)
        mesh = Mesh(np.asarray(devices), ("model",), axis_types=(AxisType.Explicit,))
        specs = (P(None, "model", None, None), P(None, None, cache_head_axis, None), P(), P(), P(), P())
        with jax.set_mesh(mesh):
            sharded = tuple(
                jax.device_put(arg, NamedSharding(mesh, spec)) for arg, spec in zip(args, specs, strict=True)
            )
            actual = jax.jit(
                partial(ragged_paged_attention, sm_scale=_SCALE, sliding_window=5, implementation="reference")
            )(*sharded)
        np.testing.assert_allclose(actual, _dense_reference(args, 5, None, _SCALE), atol=1e-5, rtol=1e-5)
        assert actual.sharding == sharded[0].sharding


@pytest.mark.tpu
@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
@pytest.mark.parametrize("runtime_scale", [False, True])
@pytest.mark.parametrize("kv_heads", [2, 3])
def test_grug_tpu_paged_attention_matches_reference(runtime_scale, kv_heads):
    args = _mixed_case(jnp.float32, kv_heads)
    fn = partial(ragged_paged_attention, sliding_window=5, implementation="tpu")
    actual = (
        jax.jit(fn)(*args, sm_scale=jnp.array(_SCALE))
        if runtime_scale
        else jax.jit(partial(fn, sm_scale=_SCALE))(*args)
    )
    expected = _dense_reference(args, 5, None, _SCALE)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-4, rtol=1e-4)


@pytest.mark.tpu
@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
def test_grug_tpu_paged_attention_with_evicted_window_page():
    page_size, head_dim = 4, 128
    pages = jnp.zeros((2, page_size, 2, head_dim), dtype=jnp.float32)
    pages = pages.at[1, :, 1].set(jnp.arange(1, page_size + 1, dtype=jnp.float32)[:, None])
    q = jnp.zeros((1, 1, 4, head_dim), dtype=jnp.float32)
    attend = jax.jit(
        partial(ragged_paged_attention, sm_scale=head_dim**-0.5, implementation="tpu", sliding_window=page_size)
    )
    result = attend(
        q,
        pages,
        jnp.array([2 * page_size], dtype=jnp.int32),
        jnp.array([[INVALID, 1]], dtype=jnp.int32),
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(result, jnp.full(q.shape, (page_size + 1) / 2))


@pytest.mark.tpu
@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
def test_grug_tpu_paged_attention_fp32_64k():
    # Zero scores make the output the mean of visible cache values. Reverse the
    # page table so the expected logical positions differ from physical order.
    num_pages, page_size, head_dim = 1024, 64, 128
    num_queries = 129
    context = num_pages * page_size
    values = jnp.arange(context, dtype=jnp.float32).reshape(num_pages, page_size) / context
    pages = jnp.zeros((num_pages, page_size, 2, head_dim), dtype=jnp.float32)
    pages = pages.at[:, :, 1].set(values[::-1, :, None])
    q = jnp.zeros((num_queries, 1, 4, head_dim), dtype=jnp.float32)
    fn = jax.jit(partial(ragged_paged_attention, sm_scale=head_dim**-0.5, implementation="tpu"))
    actual = fn(
        q,
        pages,
        jnp.array([context], dtype=jnp.int32),
        jnp.arange(num_pages - 1, -1, -1, dtype=jnp.int32)[None],
        jnp.array([0, num_queries], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    last_visible = np.arange(context - num_queries, context, dtype=np.float32)
    expected = np.broadcast_to((last_visible / (2 * context))[:, None, None, None], actual.shape)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
