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
from levanter.grug.attention._paged import _tpu_v3_available, ragged_paged_attention_append
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


@pytest.mark.tpu
@pytest.mark.skipif(not _tpu_v3_available(), reason="v5p or v6e Pallas configuration")
@pytest.mark.parametrize("window,offsets", [(None, [0, 1, 2, -1]), (None, [0, 129, 130, -1]), (96, [0, 129, 130, -1])])
def test_grug_tpu_v3_paged_attention_matches_dense(window, offsets):
    rng = np.random.default_rng(21)
    args = _PagedCase(
        q=jnp.asarray(rng.normal(size=(256, 5, 4, 128)), jnp.bfloat16),
        kv_pages=jnp.asarray(rng.normal(size=(32, 128, 10, 128)), jnp.bfloat16),
        kv_lens=jnp.array([643, 1191, -1], jnp.int32),
        page_indices=jnp.array([rng.choice(32, 16, replace=False) for _ in range(2)] + [[-1] * 16], jnp.int32),
        cu_q_lens=jnp.array(offsets, jnp.int32),
        num_seqs=jnp.array(2, jnp.int32),
    )
    scale = 128**-0.5
    attention = jax.jit(
        partial(ragged_paged_attention, sm_scale=scale, sliding_window=window, implementation="tpu_v3")
    )
    actual = attention(*args)
    expected = _dense_reference(args, window, None, scale)
    assert actual.dtype == args.q.dtype
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), rtol=0.008, atol=2e-4)
    np.testing.assert_array_equal(actual[offsets[2] :], jnp.zeros_like(actual[offsets[2] :]))


@pytest.mark.tpu
@pytest.mark.skipif(not _tpu_v3_available(), reason="v5p or v6e Pallas configuration")
@pytest.mark.parametrize("window", [None, 2048])
@pytest.mark.parametrize(
    "width,queries,heads",
    [
        (4, (1,), 5),
        (16, (1, 1, 1), 5),
        (32, (1,) * 5, 5),
        (512, (361, 82), 5),
        (2048, (2048,), 5),
        (4096, (3207,), 5),
        pytest.param(16, (1, 1, 1), 1, id="single-kv-head"),
        pytest.param(16, (1, 1, 1), 2, id="two-kv-heads"),
    ],
)
def test_grug_tpu_fused_append_updates_only_addressed_tokens(window, width, queries, heads):
    rng = np.random.default_rng(25)
    q = jnp.asarray(rng.normal(size=(width, heads, 4, 128)), jnp.bfloat16)
    k = jnp.asarray(rng.normal(size=(width, heads, 128)), jnp.bfloat16)
    v = jnp.asarray(rng.normal(size=(width, heads, 128)), jnp.bfloat16)
    page_size = 128
    pages_per_seq, first_length = (64, 6208) if heads <= 2 else (32, 2208)
    active = len(queries)
    num_pages = active * pages_per_seq
    pages = jnp.asarray(rng.normal(size=(num_pages, page_size, 2 * heads, 128)), jnp.bfloat16)
    indices = jnp.asarray(rng.permutation(num_pages).reshape(active, pages_per_seq), jnp.int32)
    if heads == 2:
        indices = indices.at[1:, :4].set(indices[0, :4])
    lengths = jnp.maximum(first_length + 31 * jnp.arange(active, dtype=jnp.int32), jnp.asarray(queries) + 127)
    offsets = jnp.asarray(np.cumsum((0, *queries)), jnp.int32)
    num_seqs = jnp.array(active, jnp.int32)
    scale = 128**-0.5
    attention = jax.jit(partial(ragged_paged_attention_append, sm_scale=scale, sliding_window=window))
    actual, updated = attention(q, k, v, pages, lengths, indices, offsets, num_seqs)
    positions = jnp.concatenate([jnp.arange(lengths[i] - count, lengths[i]) for i, count in enumerate(queries)])
    logical_pages, slots = jnp.divmod(positions, page_size)
    physical_pages = indices[np.repeat(np.arange(active), queries), logical_pages]
    expected_pages = pages.at[physical_pages, slots, 0::2].set(k[: positions.size])
    expected_pages = expected_pages.at[physical_pages, slots, 1::2].set(v[: positions.size])
    np.testing.assert_array_equal(updated, expected_pages)
    args = _PagedCase(q, expected_pages, lengths, indices, offsets, num_seqs)
    expected = _dense_reference(args, window, None, scale)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), rtol=0.008, atol=2e-4)


@pytest.mark.tpu
@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
def test_grug_tpu_paged_attention_compacts_distinct_packed_windows():
    rng = np.random.default_rng(21)
    args = _PagedCase(
        q=jnp.asarray(rng.normal(size=(8, 5, 4, 128)), jnp.float32),
        kv_pages=jnp.asarray(rng.normal(size=(32, 128, 10, 128)), jnp.float32),
        kv_lens=jnp.array([643, 1191, -1], jnp.int32),
        page_indices=jnp.array([rng.choice(32, 12, replace=False) for _ in range(2)] + [[-1] * 12], jnp.int32),
        cu_q_lens=jnp.array([0, 3, 4, -1], jnp.int32),
        num_seqs=jnp.array(2, jnp.int32),
    )
    attention = jax.jit(partial(ragged_paged_attention, sm_scale=128**-0.5, sliding_window=128, implementation="tpu"))
    actual = attention(*args)
    # One extra empty row makes a compact cache larger than the original.
    # Keep the active queries, physical pages and tile size identical.
    untrimmed = args._replace(
        kv_lens=jnp.pad(args.kv_lens, (0, 1), constant_values=-1),
        page_indices=jnp.pad(args.page_indices, ((0, 1), (0, 0)), constant_values=-1),
        cu_q_lens=jnp.pad(args.cu_q_lens, (0, 1), constant_values=-1),
    )
    np.testing.assert_array_equal(actual, attention(*untrimmed))
