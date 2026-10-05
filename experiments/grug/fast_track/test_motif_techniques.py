# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Motif 3 techniques: grouped differential attention (``mla_grouped_diff``) and PolyNorm experts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import polynorm, polynorm_over_input
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def _tokens():
    return jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)


def _loss(model, tokens):
    return model.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32))


def test_polynorm_is_scale_free_and_matches_its_tied_form():
    u = jax.random.normal(jax.random.PRNGKey(0), (5, 16))
    out = np.asarray(polynorm(u))
    # Each power is RMS-normalized per row, so scaling a row leaves the output unchanged.
    np.testing.assert_allclose(np.asarray(polynorm(3.0 * u)), out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(polynorm_over_input(u) * u), out, rtol=1e-4, atol=1e-5)
    assert np.isfinite(np.asarray(polynorm_over_input(jnp.zeros((2, 16))))).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_ungated_relu2": True, "moe_ungated_activation": "polynorm"},
        {"shared_ungated_relu2": True, "shared_ungated_activation": "polynorm"},
    ],
)
def test_polynorm_experts_train(overrides):
    relu2 = {k: v for k, v in overrides.items() if k.endswith("relu2")}
    mesh, plain = t._model(**relu2)
    _, model = t._model(**overrides)
    tokens = _tokens()
    with jax.set_mesh(mesh):
        loss = float(eqx.filter_jit(_loss)(model, tokens))
        assert np.isfinite(loss) and loss != pytest.approx(float(eqx.filter_jit(_loss)(plain, tokens)), abs=1e-6)


def test_grouped_diff_shares_noise_maps_and_learns_lambda():
    mesh, model = t._model(mla=True, mla_grouped_diff=2)
    attn = model.stacked_blocks.stacked.attn
    heads, head_dim = model.config.num_heads, model.config.inferred_head_dim
    # One noise map per group of 2 heads: half the query columns of a full projection.
    assert attn.w_qn.shape[-1] == heads // 2 * head_dim
    assert attn.gda_lambda.shape[-2:] == (model.config.hidden_dim, heads)
    tokens = _tokens()
    with jax.set_mesh(mesh):
        grads = eqx.filter_jit(eqx.filter_grad(_loss))(model, tokens)
    for leaf in (grads.stacked_blocks.stacked.attn.gda_lambda, grads.stacked_blocks.stacked.attn.w_qn):
        assert np.abs(np.asarray(leaf)).sum() > 0
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    assert mask.stacked_blocks.stacked.attn.gda_lambda == "adam"


def test_grouped_diff_needs_a_divisible_head_count():
    with pytest.raises(ValueError):
        t._config(mla=True, mla_grouped_diff=3)


def test_gdla_noise_heads_come_from_the_head_budget():
    # 4 query heads = 1 group of 3 signal heads + 1 noise head sharing one value head.
    mesh, model = t._model(mla=True, num_heads=4, mla_gdla_noise_heads=1)
    attn = model.stacked_blocks.stacked.attn
    head_dim, hidden = model.config.inferred_head_dim, model.config.hidden_dim
    assert attn.w_q.shape[-1] == 4 * head_dim
    assert attn.w_uv.shape[-1] == head_dim
    assert attn.w_o.shape[-2:] == (3 * head_dim, hidden)
    assert attn.gda_lambda.shape[-2:] == (hidden, 3)
    tokens = _tokens()
    perturbed = tokens.at[:, -1].set((tokens[:, -1] + 1) % t._VOCAB)
    with jax.set_mesh(mesh):
        grads = eqx.filter_jit(eqx.filter_grad(_loss))(model, tokens)
        forward = eqx.filter_jit(lambda m, x: m(x)[0])
        hidden_out, hidden_perturbed = forward(model, tokens), forward(model, perturbed)
    np.testing.assert_allclose(
        np.asarray(hidden_out[:, :-1]), np.asarray(hidden_perturbed[:, :-1]), rtol=1e-5, atol=1e-5
    )
    attn_grads = grads.stacked_blocks.stacked.attn
    # The noise head (the last query head of its group) reaches the loss only through the subtraction.
    noise_q = np.asarray(attn_grads.w_q)[..., 3 * head_dim :]
    assert np.abs(noise_q).sum() > 0
    assert np.abs(np.asarray(attn_grads.gda_lambda)).sum() > 0
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    assert mask.stacked_blocks.stacked.attn.gda_lambda == "adam"


@pytest.mark.parametrize("overrides", [{"num_heads": 4, "mla_gdla_noise_heads": 3}, {"mla_gdla_noise_heads": 2}])
def test_gdla_needs_whole_signal_groups(overrides):
    with pytest.raises(ValueError):
        t._config(mla=True, **overrides)
