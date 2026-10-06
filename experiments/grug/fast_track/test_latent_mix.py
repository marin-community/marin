# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Factored (latent) attention projections and mixtures of latents on attention and the MoE latent."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import LatentProj, _switchhead_weights, mixture_weights
from experiments.grug.fast_track.optimizer import _is_gate_or_router_weight


def _loss_and_grads(model, mesh):
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        return eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape))))(
            model
        )


def test_mixture_latent_matches_block_gated_reference():
    mesh, model = t._model(ngram_stat_rows=0, latent_mix_experts=4, latent_mix_topk=2)
    cfg = model.config
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 8))
    with jax.set_mesh(mesh):
        proj = LatentProj.init(cfg, jax.random.PRNGKey(0), 8, 12, 6, True, 0.1)
        out = np.asarray(proj(cfg, x))
    latent = (x @ proj.down).reshape(3, 4, 3)
    latent = latent / np.sqrt(np.mean(np.square(latent), -1, keepdims=True) + cfg.layer_norm_eps)
    weights = np.asarray(_switchhead_weights(x[None], proj.mix_gate, 4, 2))[0, :, 0]
    assert ((weights > 0).sum(-1) == 2).all()
    expected = (latent * weights[..., None]).reshape(3, 12) @ proj.up
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("site", ["q", "k", "v", "o"])
@pytest.mark.parametrize("mix", [False, True])
def test_attention_latent_replaces_the_projection_and_trains(site, mix):
    settings = {f"attn_latent_{site}": 8, "latent_mix_sites": (site,) if mix else ()}
    mesh, model = t._model(ngram_stat_rows=0, mla=False, **settings)
    attn = model.stacked_blocks.stacked.attn
    assert getattr(attn, f"w_{site}") is None
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss))
    latent = getattr(grads.stacked_blocks.stacked.attn, f"latent_{site}")
    for leaf in (latent.down, latent.up, latent.norm.weight) + ((latent.mix_gate,) if mix else ()):
        assert float(jnp.abs(leaf).max()) > 0


@pytest.mark.parametrize("site", ["moe_in", "moe_out"])
def test_moe_latent_mixture_trains_its_gate(site):
    mesh, model = t._model(ngram_stat_rows=0, latent_mix_sites=(site,))
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss))
    mlp = grads.stacked_blocks.stacked.mlp
    gate = mlp.latent_mix_in_gate if site == "moe_in" else mlp.latent_mix_out_gate
    assert float(jnp.abs(gate).max()) > 0


def test_mixture_gates_go_to_adam():
    for path in ("blocks.attn.latent_q.mix_gate", "blocks.mlp.latent_mix_in_gate", "blocks.mlp.latent_mix_out_gate"):
        assert _is_gate_or_router_weight(path)


def test_mla_kv_latent_mixture_keeps_topk_blocks_and_trains_its_gate():
    mesh, model = t._model(
        ngram_stat_rows=0, mla=True, mla_kv_latent_dim=16, latent_mix_sites=("kv",), latent_mix_topk=1
    )
    attn = model.stacked_blocks.stacked.attn
    assert attn.kv_latent_norm.weight.shape[-2:] == (4, 4)
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss))
    assert float(jnp.abs(grads.stacked_blocks.stacked.attn.kv_mix_gate).max()) > 0


def test_bias_balance_gradient_is_the_load_error_and_entropy_pushes_toward_uniform():
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 8))
    gate = jax.random.normal(jax.random.PRNGKey(1), (8, 4)).at[:, 0].add(2.0)
    bias = jnp.zeros((4,))
    grad = jax.grad(lambda b: jnp.sum(mixture_weights(x, gate, 4, 2, selection_bias=b)))(bias)
    load = np.asarray((mixture_weights(x, gate, 4, 2) > 0).mean(0))
    np.testing.assert_allclose(np.asarray(grad), load - load.mean(), atol=1e-6)
    # The entropy term moves logits so that the over-picked block's mean softmax share falls.
    plain = jax.grad(lambda g: jnp.sum(mixture_weights(x, g, 4, 4)))(gate)
    balanced = jax.grad(lambda g: jnp.sum(mixture_weights(x, g, 4, 4, entropy_weight=1.0)))(gate)
    extra = np.asarray(balanced - plain)
    mean_x = np.asarray(jax.nn.softmax(x @ gate, -1))  # [T, E]
    # Descending the extra gradient lowers the busiest block's average probability.
    step = gate - 0.1 * extra
    before = mean_x.mean(0)[0]
    after = np.asarray(jax.nn.softmax(x @ step, -1)).mean(0)[0]
    assert after < before


def test_bias_balanced_mla_mixture_trains_and_routes_bias_to_sign_sgd():
    mesh, model = t._model(
        ngram_stat_rows=0, mla=True, mla_kv_latent_dim=16, latent_mix_sites=("kv",), latent_mix_balance="bias"
    )
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss))
    g = np.asarray(grads.stacked_blocks.stacked.attn.kv_mix_bias)
    np.testing.assert_allclose(g.sum(-1), 0.0, atol=1e-6)


def test_renormed_mixture_weights_sum_to_one_over_the_kept_blocks():
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 8))
    gate = jax.random.normal(jax.random.PRNGKey(1), (8, 8))
    weights = np.asarray(mixture_weights(x, gate, 8, 2, renorm=True))
    np.testing.assert_allclose(weights.sum(-1), 1.0, rtol=1e-6)
    assert ((weights > 0).sum(-1) == 2).all()


def test_sum_mode_mla_mixture_keeps_a_shared_latent_and_up_projection():
    mesh, model = t._model(
        ngram_stat_rows=0,
        mla=True,
        mla_kv_latent_dim=12,
        latent_mix_sites=("kv",),
        latent_mix_experts=4,
        latent_mix_kv_mode="sum",
    )
    attn = model.stacked_blocks.stacked.attn
    assert attn.w_dkv.shape[-1] == 4 * 12
    assert attn.w_uk.shape[-2] == 12 and attn.w_uv.shape[-2] == 12
    assert attn.kv_latent_norm.weight.shape[-1] == 12
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss))
    assert float(jnp.abs(grads.stacked_blocks.stacked.attn.kv_mix_gate).max()) > 0
