# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention import ragged_paged_attention, ragged_paged_attention_append
from levanter.grug.attention._paged_gpu import gpu_paged_attention

_SUPPORTS_RPA_V3 = jax.default_backend() == "tpu" and pltpu.get_tpu_info().chip_version in (
    pltpu.ChipVersion.TPU_V5P,
    pltpu.ChipVersion.TPU_V6E,
)


class _PagedCase(NamedTuple):
    q: jax.Array
    kv_pages: jax.Array
    kv_lens: jax.Array
    page_indices: jax.Array
    cu_q_lens: jax.Array
    num_seqs: jax.Array


def _mixed_case(dtype, *, kv_heads=2, queries_per_head=3, head_size=32):
    rng = np.random.default_rng(12)
    q = rng.normal(size=(8, kv_heads, queries_per_head, head_size)).astype(np.float32)
    pages = rng.normal(size=(7, 4, 2 * kv_heads, head_size)).astype(np.float32)
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
    actual = fn(*args, sm_scale=jnp.array(0.17))
    expected = _dense_reference(args, window, cap, 0.17)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-5, rtol=1e-5)
    empty = fn(*args._replace(num_seqs=jnp.array(0, jnp.int32)), sm_scale=jnp.array(0.17))
    np.testing.assert_array_equal(empty, jnp.zeros_like(args.q))


@pytest.mark.parametrize("num_devices,cache_head_axis", [(2, "model"), (1, None)])
def test_grug_paged_attention_explicit_head_sharding_matches_reference(num_devices, cache_head_axis):
    args = _mixed_case(jnp.float32)
    devices = np.asarray(jax.devices()[:num_devices])
    mesh = Mesh(devices, ("model",), axis_types=(AxisType.Explicit,))
    specs = (P(None, "model", None, None), P(None, None, cache_head_axis, None), P(), P(), P(), P())
    with jax.set_mesh(mesh):
        sharded = tuple(jax.device_put(arg, NamedSharding(mesh, spec)) for arg, spec in zip(args, specs, strict=True))
        actual = jax.jit(partial(ragged_paged_attention, sm_scale=0.17, sliding_window=5, implementation="reference"))(
            *sharded
        )
    np.testing.assert_allclose(actual, _dense_reference(args, 5, None, 0.17), atol=1e-5, rtol=1e-5)
    assert actual.sharding == sharded[0].sharding


@pytest.mark.parametrize("window", [None, 257])
def test_grug_gpu_split_decode_matches_reference_in_interpret_mode(window):
    rng = np.random.default_rng(37)
    args = _PagedCase(
        jnp.asarray(rng.normal(size=(4, 1, 4, 128)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(8, 128, 2, 128)), jnp.bfloat16),
        jnp.array([643], jnp.int32),
        jnp.array([[6, 1, 3, 2, 5, 4, -1]], jnp.int32),
        jnp.array([0, 1], jnp.int32),
        jnp.array(1, jnp.int32),
    )
    upper = jnp.array([643, 0, 0, 0], jnp.int32)
    lower = jnp.zeros_like(upper) if window is None else jnp.maximum(0, upper - window)
    bounds = jnp.stack((lower, upper), axis=-1)
    pages = jnp.broadcast_to(jnp.maximum(args.page_indices, 0), (args.q.shape[0], args.page_indices.shape[1]))
    scale = 128**-0.5
    with jax.default_device(jax.devices("cpu")[0]):
        actual = jax.jit(partial(gpu_paged_attention, max_splits=16, interpret=True))(
            args.q, args.kv_pages, pages, bounds, scale
        )
    expected = _dense_reference(args, window, None, scale)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(np.float32), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="CUDA paged attention")
@pytest.mark.parametrize("window", [None, 129])
@pytest.mark.parametrize("query_capacity, query_length", [(8, 1), (256, 31)])
def test_grug_gpu_paged_attention_matches_reordered_cached_prefix(window, query_capacity, query_length):
    rng = np.random.default_rng(37)
    pages = rng.normal(size=(8, 128, 2, 128))
    pages[0] = np.nan  # Unallocated page used as a dummy by gathers.
    pages[4, 3:] = np.nan  # Last allocated page extends past the live sequence.
    args = _PagedCase(
        jnp.asarray(rng.normal(size=(query_capacity, 1, 4, 128)), jnp.bfloat16),
        jnp.asarray(pages, jnp.bfloat16),
        jnp.array([643], jnp.int32),
        jnp.array([[6, 1, 3, 2, 5, 4, -1, -1]], jnp.int32),
        jnp.array([0, query_length], jnp.int32),
        jnp.array(1, jnp.int32),
    )
    scale = 128**-0.5
    actual = jax.jit(partial(ragged_paged_attention, implementation="gpu", sm_scale=scale, sliding_window=window))(
        *args
    )
    expected = _dense_reference(args, window, None, scale)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(np.float32), atol=2e-4, rtol=8e-3)
    np.testing.assert_array_equal(actual[query_length:], jnp.zeros_like(actual[query_length:]))


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="CUDA paged attention")
@pytest.mark.parametrize("query_capacity", [128, 256, 1024])
@pytest.mark.parametrize("window", [None, 129])
def test_grug_gpu_paged_attention_matches_distinct_packed_prefixes(query_capacity, window):
    rng = np.random.default_rng(12)
    pages = rng.normal(size=(10, 128, 2, 128))
    pages[0] = np.nan
    args = _PagedCase(
        jnp.asarray(rng.normal(size=(query_capacity, 1, 4, 128)), jnp.bfloat16),
        jnp.asarray(pages, jnp.bfloat16),
        jnp.array([385, 262, -1], jnp.int32),
        jnp.array([[3, 1, 7, 2], [6, 4, 5, -1], [-1, -1, -1, -1]], jnp.int32),
        jnp.array([0, 16, 49, -1], jnp.int32),
        jnp.array(2, jnp.int32),
    )
    scale = 128**-0.5
    actual = jax.jit(partial(ragged_paged_attention, implementation="gpu", sm_scale=scale, sliding_window=window))(
        *args
    )
    expected = _dense_reference(args, window, None, scale)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=2e-4, rtol=8e-3)
    np.testing.assert_array_equal(actual[49:], jnp.zeros_like(actual[49:]))


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
@pytest.mark.parametrize("runtime_scale", [False, True])
def test_grug_tpu_paged_attention_matches_reference(runtime_scale):
    args = _mixed_case(jnp.float32)
    fn = partial(ragged_paged_attention, sliding_window=5, implementation="tpu")
    actual = (
        jax.jit(fn)(*args, sm_scale=jnp.array(0.17)) if runtime_scale else jax.jit(partial(fn, sm_scale=0.17))(*args)
    )
    expected = _dense_reference(args, 5, None, 0.17)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
def test_grug_tpu_fp32_paged_attention_matches_bf16_reference_with_five_kv_heads():
    args = _mixed_case(jnp.bfloat16, kv_heads=5, queries_per_head=4, head_size=128)
    attention = partial(ragged_paged_attention, sm_scale=128**-0.5, sliding_window=5)
    expected = jax.jit(partial(attention, implementation="reference"))(*args)
    actual = jax.jit(partial(attention, implementation="tpu_fp32"))(*args)
    assert actual.dtype == args.q.dtype
    # Accumulation order may change how the final value rounds to BF16.
    nearest = jnp.nextafter(expected, actual)
    np.testing.assert_array_equal(actual, nearest)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernel")
def test_grug_tpu_paged_attention_compacts_distinct_packed_windows():
    rng = np.random.default_rng(21)
    args = _PagedCase(
        q=jnp.asarray(rng.normal(size=(8, 5, 4, 128)), jnp.bfloat16),
        kv_pages=jnp.asarray(rng.normal(size=(32, 128, 10, 128)), jnp.bfloat16),
        kv_lens=jnp.array([643, 1191, -1], jnp.int32),
        page_indices=jnp.array([rng.choice(32, 12, replace=False) for _ in range(2)] + [[-1] * 12], jnp.int32),
        cu_q_lens=jnp.array([0, 3, 4, -1], jnp.int32),
        num_seqs=jnp.array(2, jnp.int32),
    )
    attention = jax.jit(
        partial(ragged_paged_attention, sm_scale=128**-0.5, sliding_window=128, implementation="tpu_fp32")
    )
    actual = attention(*args)
    # One extra empty row makes a compact cache larger than the original.
    # Keep the active queries, physical pages and tile size identical.
    untrimmed = args._replace(
        kv_lens=jnp.pad(args.kv_lens, (0, 1), constant_values=-1),
        page_indices=jnp.pad(args.page_indices, ((0, 1), (0, 0)), constant_values=-1),
        cu_q_lens=jnp.pad(args.cu_q_lens, (0, 1), constant_values=-1),
    )
    np.testing.assert_array_equal(actual, attention(*untrimmed))


@pytest.mark.skipif(not _SUPPORTS_RPA_V3, reason="v5p or v6e Pallas configuration")
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


@pytest.mark.skipif(not _SUPPORTS_RPA_V3, reason="v5p or v6e Pallas configuration")
@pytest.mark.parametrize("window", [None, 2048])
@pytest.mark.parametrize(
    "width,queries,heads",
    [
        (4, (1,), 5),
        (16, (1, 1, 1), 5),
        (32, (1,) * 5, 5),
        (512, (361, 82), 5),
        (2048, (2048,), 5),
        pytest.param(16, (1, 1, 1), 1, id="single-kv-head"),
    ],
)
def test_grug_tpu_fused_append_updates_only_addressed_tokens(window, width, queries, heads):
    rng = np.random.default_rng(25)
    q = jnp.asarray(rng.normal(size=(width, heads, 4, 128)), jnp.bfloat16)
    k = jnp.asarray(rng.normal(size=(width, heads, 128)), jnp.bfloat16)
    v = jnp.asarray(rng.normal(size=(width, heads, 128)), jnp.bfloat16)
    page_size = 128
    pages_per_seq, first_length = (64, 6208) if heads == 1 else (32, 2208)
    active = len(queries)
    num_pages = active * pages_per_seq
    pages = jnp.asarray(rng.normal(size=(num_pages, page_size, 2 * heads, 128)), jnp.bfloat16)
    indices = jnp.asarray(rng.permutation(num_pages).reshape(active, pages_per_seq), jnp.int32)
    lengths = first_length + 31 * jnp.arange(active, dtype=jnp.int32)
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
