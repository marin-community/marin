# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fixed-encoder n-gram statistic table: writes are exact sums of the fixed next-token code, the table
never receives a gradient or an optimizer update, the model stays causal, and the reader trains."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track.model import (
    GrugModelConfig,
    LocalMixer,
    Transformer,
    _ngram_stat_ids,
    write_ngram_stats,
)
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_SEQ = 16
_VOCAB = 32
_ROWS = 97


def _config(**overrides) -> GrugModelConfig:
    kwargs = dict(
        vocab_size=_VOCAB,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=16,
        max_seq_len=_SEQ,
        sliding_window=_SEQ,
        global_every=2,
        local_mixer=LocalMixer.KDA,
        attn_res=True,
        attn_res_num_blocks=2,
        ngram_stat_rows=_ROWS,
        ngram_stat_dim=8,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _model(**overrides) -> tuple[Mesh, Transformer]:
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Transformer.init(_config(**overrides), key=jax.random.PRNGKey(0))
    return mesh, model


def test_write_adds_code_of_next_token_per_ngram_row():
    """With orders (2, 3), each position adds [code(next token), 1] to its bigram row and to its trigram row."""
    mesh, model = _model(ngram_stat_orders=(2, 3))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, _SEQ), 0, _VOCAB)
    weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
    with jax.set_mesh(mesh):
        written = eqx.filter_jit(write_ngram_stats)(model, tokens, weight, None)
        ids = np.asarray(_ngram_stat_ids(model.config, tokens, None))
    code = np.asarray(model.ngram_stat_code)
    expected = np.zeros((2 * _ROWS, code.shape[1] + 1), np.float64)
    toks = np.asarray(tokens)
    for b in range(toks.shape[0]):
        for t in range(_SEQ - 1):
            for k in range(2):
                expected[ids[b, t, k], :-1] += code[toks[b, t + 1]]
                expected[ids[b, t, k], -1] += 1
    np.testing.assert_allclose(np.asarray(written.ngram_stat_table), expected, rtol=1e-5, atol=1e-5)
    assert ids[..., 0].max() < _ROWS <= ids[..., 1].min()
    assert float(expected[:_ROWS, -1].sum()) == float(expected[_ROWS:, -1].sum()) == 2 * (_SEQ - 1)


def test_model_with_stat_table_is_causal():
    mesh, model = _model()
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, _SEQ), 0, _VOCAB)
    with jax.set_mesh(mesh):
        weight = jnp.ones(tokens.shape, jnp.float32)
        model = eqx.filter_jit(write_ngram_stats)(model, tokens, weight, None)
        perturbed = tokens.at[:, -1].set((tokens[:, -1] + 1) % _VOCAB)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        hidden, hidden_perturbed = forward(model, tokens), forward(model, perturbed)
    np.testing.assert_allclose(np.asarray(hidden[:, :-1]), np.asarray(hidden_perturbed[:, :-1]), rtol=1e-5, atol=1e-5)


def test_table_is_frozen_and_reader_trains():
    mesh, model = _model(ngram_stat_mlp_dim=16)
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, _SEQ), 0, _VOCAB)
    weights = jnp.ones(tokens.shape, jnp.float32)
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(write_ngram_stats)(model, tokens, weights, None)
        _, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, weights)))(model)
    assert np.abs(np.asarray(grads.ngram_stat_table)).max() == 0
    assert np.abs(np.asarray(grads.ngram_stat_up)).max() > 0
    assert np.abs(np.asarray(grads.ngram_stat_hidden)).max() > 0

    opt_config = GrugMoeMuonHConfig()
    labels = opt_config.create_mask(model)
    assert labels.ngram_stat_table == "frozen" and labels.ngram_stat_code == "frozen"
    assert labels.ngram_stat_gate_w == "adam"
