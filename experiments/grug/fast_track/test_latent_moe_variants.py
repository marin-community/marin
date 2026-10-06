# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LatentMoE variants: multi-head latents, a private per-expert channel, Matryoshka widths, token-conditioned W_down."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.optimizer import _is_gate_or_router_weight


def _loss_and_grads(model, mesh):
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        return eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape))))(
            model
        )


def _mlp(tree):
    return tree.stacked_blocks.stacked.mlp


def _trains(leaf):
    return float(jnp.abs(leaf).max()) > 0


def test_multi_head_latent_experts_read_one_latent_group():
    mesh, model = t._model(ngram_stat_rows=0, latent_dim=32, expert_read_groups=2, latent_out_dim=16)
    assert model.config.expert_in_dim == 16
    assert _mlp(model).expert_mlp.w_up.shape[-2] == 16
    assert _mlp(model).w_latent_down.shape[-1] == 32
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss)) and _trains(_mlp(grads).w_latent_down)


def test_private_channel_widens_the_expert_input_and_trains_its_projection():
    mesh, model = t._model(ngram_stat_rows=0, expert_private_dim=4, expert_private_groups=2)
    cfg = model.config
    assert cfg.expert_in_dim == 16 + 4 and cfg.expert_out_dim == 16
    assert _mlp(model).expert_mlp.w_up.shape[-2] == 20
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss))
    assert _trains(_mlp(grads).expert_private_proj) and _trains(_mlp(grads).expert_private_norm.weight)


@pytest.mark.parametrize("blocks", [2, 4])
def test_matryoshka_latent_trains_its_width_gate_and_bias(blocks):
    mesh, model = t._model(ngram_stat_rows=0, latent_matryoshka_blocks=blocks)
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss))
    assert _trains(_mlp(grads).latent_width_gate)
    bias_grad = np.asarray(_mlp(grads).latent_width_bias)
    np.testing.assert_allclose(bias_grad.sum(-1), 0.0, atol=1e-6)
    assert _is_gate_or_router_weight("stacked_blocks.stacked.mlp.latent_width_gate")


def test_token_conditioned_w_down_mixes_full_width_bases():
    mesh, model = t._model(
        ngram_stat_rows=0,
        latent_mix_sites=("moe_in",),
        latent_mix_moe_in_mode="sum",
        latent_mix_experts=4,
        latent_mix_topk=4,
    )
    assert _mlp(model).w_latent_down.shape[-1] == 4 * 16
    assert _mlp(model).latent_norm.weight.shape[-1] == 16
    loss, grads = _loss_and_grads(model, mesh)
    assert np.isfinite(float(loss)) and _trains(_mlp(grads).latent_mix_in_gate)


def test_matryoshka_scale_is_finite_when_the_chosen_gate_saturates():
    _, model = t._model(ngram_stat_rows=0, latent_matryoshka_blocks=4)
    mlp = jax.tree.map(lambda a: a[0] if eqx.is_array(a) else a, _mlp(model))
    mlp = eqx.tree_at(lambda m: m.latent_width_gate, mlp, -1e4 * jnp.ones_like(mlp.latent_width_gate))
    x = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (8, model.config.hidden_dim)))
    latent = jnp.ones((8, 16))

    def out(gate):
        return jnp.sum(eqx.tree_at(lambda m: m.latent_width_gate, mlp, gate)._matryoshka_latent(latent, x, {}))

    value, grad = jax.value_and_grad(out)(mlp.latent_width_gate)
    assert np.isfinite(float(value)) and np.isfinite(np.asarray(grad)).all()


def test_write_groups_mask_w_down_block_diagonally_and_keep_it_under_training():
    mesh, model = t._model(ngram_stat_rows=0, latent_out_dim=32, expert_write_groups=2)
    w_down = np.asarray(_mlp(model).expert_mlp.w_down)  # [L, E, I, O]
    neurons, out = w_down.shape[-2:]
    assert np.all(w_down[..., : neurons // 2, out // 2 :] == 0) and np.all(w_down[..., neurons // 2 :, : out // 2] == 0)
    assert np.abs(w_down[..., : neurons // 2, : out // 2]).max() > 0
    loss, grads = _loss_and_grads(model, mesh)
    g = np.asarray(_mlp(grads).expert_mlp.w_down)
    assert np.isfinite(float(loss)) and np.all(g[..., : neurons // 2, out // 2 :] == 0)
