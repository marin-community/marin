# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The count-sketch sign trick on the hashed n-gram table (``embed2_sign_trick``) and the momentum-free
per-row Adam for that table (``embed2_row_sparse_adam``), both after modded-nanogpt record #360."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import _ngram_sign_ids
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig, RowAdamState, scale_by_row_adam

_ROWS = 64
_POOL = 16
_BIGRAM = dict(ngram_stat_rows=0, second_embed=True, second_embed_bigram=True, embed2_rows=_ROWS)


def _record_sign_ids(tokens: np.ndarray, sentinel: int, ngram: int, pool: int, segments=None, salt: int = 0):
    """The record's ``((c0 x_t) ^ (c1 x_{t-1}) ^ (c2 x_{t-2})) & (pool - 1)`` in wrapping uint32 arithmetic."""
    mults = {2: (48271, 30011), 3: (58699, 39779, 26801)}[ngram]
    out = np.zeros(tokens.shape, np.int64)
    for b in range(tokens.shape[0]):
        for s in range(tokens.shape[1]):
            x = (int(tokens[b, s]) * mults[0]) & 0xFFFFFFFF
            for lag in range(1, ngram):
                missing = s < lag or (segments is not None and segments[b, s - lag] != segments[b, s])
                prev = sentinel if missing else int(tokens[b, s - lag])
                x ^= (prev * mults[lag]) & 0xFFFFFFFF
            x ^= (salt * 0x27D4EB2D) & 0xFFFFFFFF
            out[b, s] = x & (pool - 1)
    return out


@pytest.mark.parametrize("ngram", [2, 3])
def test_sign_ids_match_record_hash(ngram):
    tokens = np.asarray(jax.random.randint(jax.random.PRNGKey(0), (2, t._SEQ), 0, t._VOCAB))
    segments = np.repeat(np.array([[0] * 5 + [1] * (t._SEQ - 5)]), 2, axis=0)
    got = _ngram_sign_ids(jnp.asarray(tokens), jnp.asarray(segments), _ROWS, ngram, _POOL, salt=1)
    np.testing.assert_array_equal(np.asarray(got), _record_sign_ids(tokens, _ROWS, ngram, _POOL, segments, salt=1))


class _Capture(eqx.Module):
    """Stands in for ``embed2_norm``: records its input (the gathered, signed rows) and applies the real norm."""

    inner: eqx.Module
    sink: list = eqx.field(static=True)

    def __call__(self, x):
        jax.debug.callback(lambda v: self.sink.append(np.asarray(v)), x)
        return self.inner(x)


def _embed2_rows(model, tokens):
    sink = []
    model = eqx.tree_at(lambda m: m.embed2_norm, model, _Capture(model.embed2_norm, sink))
    normed = model.embed2_norm.inner
    jax.block_until_ready(eqx.filter_jit(lambda m, x: m(x)[0])(model, tokens))
    (rows,) = sink
    return rows, np.asarray(normed(jnp.asarray(rows)))


@pytest.mark.parametrize(
    "heads", [dict(), dict(embed2_hash_heads=2, embed2_head_orders=(2, 3))], ids=["single", "bigram_trigram"]
)
def test_sign_trick_multiplies_rows_by_hashed_signs(heads):
    """Signed rows equal the plain rows times the pool rows the record's hash picks (per head), and the norm output
    keeps its per-token RMS."""
    mesh, signed = t._model(**_BIGRAM, **heads, embed2_sign_trick=True, embed2_sign_pool=_POOL)
    _, plain = t._model(**_BIGRAM, **heads)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        signed_rows, signed_normed = _embed2_rows(signed, tokens)
        plain_rows, plain_normed = _embed2_rows(plain, tokens)

    pool = np.asarray(signed.embed2_sign_table.astype(jnp.float32))
    assert pool.shape == (_POOL, signed.token_embed2.shape[1]) and set(np.unique(pool)) == {-1.0, 1.0}
    orders = heads.get("embed2_head_orders", (2,))
    signs = np.concatenate(
        [pool[_record_sign_ids(np.asarray(tokens), _ROWS, order, _POOL, salt=h)] for h, order in enumerate(orders)],
        axis=-1,
    )
    np.testing.assert_array_equal(signed_rows, plain_rows * signs)
    rms = lambda x: np.sqrt(np.mean(np.square(x), -1))  # noqa: E731
    np.testing.assert_allclose(rms(signed_normed), rms(plain_normed), rtol=1e-5)


def test_sign_trick_model_trains_and_routes_pool_to_frozen():
    mesh, model = t._model(**_BIGRAM, embed2_hash_heads=2, embed2_head_orders=(2, 3), embed2_sign_trick=True)
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, t._SEQ), 0, t._VOCAB)
    weights = jnp.ones(tokens.shape, jnp.float32)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, weights)))(model)
    assert np.isfinite(float(loss))
    assert np.abs(np.asarray(grads.token_embed2)).max() > 0
    assert np.abs(np.asarray(grads.embed2_sign_table)).max() == 0

    labels = GrugMoeMuonHConfig(embed2_row_sparse_adam=True).create_mask(eqx.filter(model, eqx.is_inexact_array))
    assert labels.embed2_sign_table == "frozen"
    assert labels.token_embed2 == "embed2"
    assert GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_inexact_array)).token_embed2 == "adam"


def test_row_adam_matches_numpy_reference():
    """Zero-gradient rows keep their value while their second moment decays; touched rows move by
    ``lr sqrt(1 - b2^t) g / (sqrt(v) + eps)`` with ``v`` the EMA of the row's mean square gradient."""
    beta2, eps, lr = 0.9, 1e-8, 0.1
    rng = np.random.default_rng(0)
    params = rng.normal(size=(6, 4)).astype(np.float32)
    grads = [rng.normal(size=(6, 4)).astype(np.float32) for _ in range(3)]
    grads[1][[0, 3]] = 0.0
    grads[2][[0, 5]] = 0.0

    opt = optax.chain(scale_by_row_adam(beta2, eps), optax.scale(-lr))
    state = opt.init(jnp.asarray(params))
    p, v = params.copy(), np.zeros((6, 1), np.float32)
    got = jnp.asarray(params)
    for step, g in enumerate(grads, start=1):
        updates, state = opt.update(jnp.asarray(g), state, got)
        got = optax.apply_updates(got, updates)
        before = p.copy()
        v = beta2 * v + (1 - beta2) * np.mean(g * g, -1, keepdims=True)
        p = p - lr * np.sqrt(1 - beta2**step) * g / (np.sqrt(v) + eps)
        np.testing.assert_allclose(np.asarray(got), p, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(p[np.all(g == 0, -1)], before[np.all(g == 0, -1)])
    row_state = state[0]
    assert isinstance(row_state, RowAdamState) and row_state.nu.shape == (6, 1)
    np.testing.assert_allclose(np.asarray(row_state.nu), v, rtol=1e-5)
    assert float(row_state.nu[0, 0]) == pytest.approx(
        float(beta2**2 * (1 - beta2) * np.mean(grads[0][0] ** 2)), rel=1e-5
    )


def test_row_sparse_optimizer_state_is_one_float_per_table_row():
    mesh, model = t._model(**_BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    with jax.set_mesh(mesh):
        sparse = GrugMoeMuonHConfig(embed2_row_sparse_adam=True, embed2_lr_mult=10.0).build(10).init(params)
        dense = GrugMoeMuonHConfig(embed2_lr_mult=10.0).build(10).init(params)
    is_row = lambda x: isinstance(x, RowAdamState)  # noqa: E731
    (row_state,) = [x for x in jax.tree.leaves(sparse, is_leaf=is_row) if is_row(x)]
    (nu,) = jax.tree.leaves(row_state.nu)
    assert nu.shape == (_ROWS, 1) and nu.dtype == jnp.float32
    state_floats = lambda s: sum(x.size for x in jax.tree.leaves(s))  # noqa: E731
    assert state_floats(dense) - state_floats(sparse) == 2 * model.token_embed2.size - _ROWS
