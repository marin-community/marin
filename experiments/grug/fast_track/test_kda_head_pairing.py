# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``kda_head_pairing``: adjacent KDA heads share one delta-rule state over an interleaved sequence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import (
    KDA_CHUNK_SIZE,
    _kda_kernel,
    _kda_kernel_paired,
    _pair_heads,
    _unpair_heads,
)

_HEADS, _DIM = 4, 8


def test_pairing_interleaves_adjacent_heads_and_inverts():
    x = jnp.arange(2 * 3 * _HEADS).reshape(2, 3, _HEADS)
    paired = _pair_heads(x)
    assert paired.shape == (2, 6, _HEADS // 2)
    # Pair 0 reads head 0 then head 1 at each token.
    np.testing.assert_array_equal(np.asarray(paired[0, :, 0]), [0, 1, 4, 5, 8, 9])
    np.testing.assert_array_equal(np.asarray(_unpair_heads(paired)), np.asarray(x))


def _value_jacobian(kernel, token: int, head: int, source_head: int) -> np.ndarray:
    """|d o[token, head] / d v[s, source_head]| summed over channels, for every source token s."""
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    shape = (1, KDA_CHUNK_SIZE, _HEADS, _DIM)
    q, k, v = (jax.random.normal(key, shape) for key in keys[:3])
    g = -0.1 * jnp.ones(shape)
    beta = jax.nn.sigmoid(jax.random.normal(keys[3], shape[:3]))

    def out(v):
        return kernel(q, k, v, g, beta, {}, save_chunk_states=False, chunk_size=KDA_CHUNK_SIZE)[0, token, head].sum()

    return np.abs(np.asarray(jax.grad(out)(v)[0, :, source_head])).sum(axis=-1)


def test_paired_heads_read_each_others_writes_causally():
    token = 5
    a_from_b = _value_jacobian(_kda_kernel_paired, token, head=0, source_head=1)
    b_from_a = _value_jacobian(_kda_kernel_paired, token, head=1, source_head=0)
    # A's query at token t sits before B's write at token t; B's sits after A's.
    assert a_from_b[token - 1] > 0 and a_from_b[token] == 0 and not a_from_b[token + 1 :].any()
    assert b_from_a[token] > 0 and not b_from_a[token + 1 :].any()
    # Heads in different pairs stay independent, as does every head without pairing.
    assert not _value_jacobian(_kda_kernel_paired, token, head=0, source_head=2).any()
    assert not _value_jacobian(_kda_kernel, token, head=0, source_head=1).any()


@pytest.mark.parametrize("pairing", ["half_decay", "full_decay"])
def test_paired_model_trains(pairing):
    mesh, model = t._model(kda_head_pairing=pairing)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        loss = jax.jit(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32)))(model)
    assert np.isfinite(float(loss))


def test_pairing_needs_kda():
    with pytest.raises(ValueError):
        t._config(kda_head_pairing="half_decay", local_mixer="attention")
