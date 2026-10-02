# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention import ragged_paged_attention


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
    return tuple(jnp.asarray(x, dtype if i < 2 else None) for i, x in enumerate(args))


def _dense_oracle(args, window, cap, scale):
    q, pages, lengths, indices, offsets, num_seqs = (
        np.asarray(x, dtype=np.float32 if i < 2 else None) for i, x in enumerate(args)
    )
    output = np.zeros_like(q)
    for seq in range(int(num_seqs)):
        length = int(lengths[seq])
        tokens = pages[indices[seq, : (length + pages.shape[1] - 1) // pages.shape[1]]].reshape(-1, 4, 32)
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
    expected = _dense_oracle(args, window, cap, scale).astype(np.asarray(args[0]).dtype)
    fn = jax.jit(partial(ragged_paged_attention, sliding_window=window, soft_cap=cap, implementation="reference"))
    actual = fn(*args, sm_scale=jnp.array(scale))
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(np.asarray(actual, np.float32) - np.asarray(expected, np.float32))) < 1e-5
    empty = fn(*args[:-1], jnp.array(0, jnp.int32), sm_scale=jnp.array(scale))
    np.testing.assert_array_equal(empty, jnp.zeros_like(args[0]))


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
def test_grug_tpu_paged_attention_matches_dense(window, runtime_scale):
    args = _mixed_case(jnp.float32)
    fn = partial(ragged_paged_attention, sliding_window=window, implementation="tpu")
    actual = (
        jax.jit(fn)(*args, sm_scale=jnp.array(0.17)) if runtime_scale else jax.jit(partial(fn, sm_scale=0.17))(*args)
    )
    np.testing.assert_allclose(actual, _dense_oracle(args, window, None, 0.17), atol=1e-4, rtol=1e-4)
