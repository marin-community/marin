# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eigenbasis MuonH variants: SNR weighting in Shampoo's eigenbasis, and their optimizer group."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_ngram_stat as t
import experiments.grug.fast_track.test_optimizer_group_knobs as knobs
from experiments.grug.fast_track.eig_muon import EIG_MODES, scale_by_eig_direction
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def _run(mode: str, grads: list[np.ndarray], **kwargs) -> np.ndarray:
    opt = scale_by_eig_direction(mode, **kwargs)
    params = jnp.zeros(grads[0].shape, jnp.float32)
    state = opt.init(params)
    out = None
    for g in grads:
        out, state = opt.update(jnp.asarray(g, jnp.float32), state, params)
    return np.asarray(out)


def _polar(x: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(x, full_matrices=False)
    return u @ vt


@pytest.mark.parametrize("shape", [(8, 6), (6, 8)])
def test_snr_of_a_constant_gradient_is_the_polar_factor(shape):
    g = np.random.default_rng(0).standard_normal(shape)
    # The basis comes from a float32, slightly damped eigh, so agreement is to about 1e-3.
    np.testing.assert_allclose(_run("snr", [g, g, g]), _polar(g), atol=3e-3)


def test_snr_mutes_a_direction_whose_sign_keeps_flipping():
    rng = np.random.default_rng(1)
    u, _ = np.linalg.qr(rng.standard_normal((8, 2)))
    v, _ = np.linalg.qr(rng.standard_normal((6, 2)))
    a, b = np.outer(u[:, 0], v[:, 0]), np.outer(u[:, 1], v[:, 1])
    grads = [a + 3.0 * rng.choice([-1.0, 1.0]) * b for _ in range(60)]
    out = _run("snr", grads)
    assert np.sum(out * a) > 0.9
    assert abs(np.sum(out * b)) < 0.4
    # Muon's own direction keeps a full weight on the flipping direction.
    assert abs(np.sum(_run("muon", grads[-1:]) * b)) > 0.9


def test_basis_refresh_carries_the_momentum_exactly():
    rng = np.random.default_rng(2)
    grads = [rng.standard_normal((8, 6)) for _ in range(12)]
    every_step = _run("muon", grads, refresh_every=1)
    rarely = _run("muon", grads, refresh_every=100)
    # The momentum is a change of coordinates away from the plain EMA, whatever the basis.
    np.testing.assert_allclose(every_step, rarely, atol=2e-4)


@pytest.mark.parametrize("mode", EIG_MODES)
def test_every_mode_changes_only_its_matrix_types(mode):
    mesh, model = t._model(**knobs._BIGRAM)
    params = eqx.filter(model, eqx.is_inexact_array)
    config = GrugMoeMuonHConfig(eig_families=("kda", "latent"), eig_mode=mode)
    mask = config.create_mask(params)
    assert mask.kda_blocks.stacked.attn.w_q == "eig"
    assert mask.kda_blocks.stacked.mlp.w_latent_up == "eig"
    assert mask.kda_blocks.stacked.mlp.expert_mlp.w_up == "muonh"
    with jax.set_mesh(mesh):
        base = knobs._two_steps(GrugMoeMuonHConfig(), params)
        out = knobs._two_steps(config, params)
    q = np.asarray(out.kda_blocks.stacked.attn.w_q)
    assert np.all(np.isfinite(q)) and not np.allclose(q, np.asarray(base.kda_blocks.stacked.attn.w_q))
    np.testing.assert_allclose(
        np.asarray(out.kda_blocks.stacked.mlp.expert_mlp.w_up),
        np.asarray(base.kda_blocks.stacked.mlp.expert_mlp.w_up),
        rtol=1e-5,
        atol=1e-7,
    )


def test_eig_settings_are_validated():
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(eig_families=("lm_head",))
    with pytest.raises(ValueError):
        GrugMoeMuonHConfig(eig_mode="adam")
