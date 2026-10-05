# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``muonh_step``: every mode keeps each matrix's norm; ``spectral`` takes Muon's own step size and ``grad`` scales
the relative step by the smoothed gradient norm over its early mean."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t
import experiments.grug.fast_track.test_optimizer_group_knobs as knobs
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig, MuonHStep, scale_with_grug_muonh


def _setup(shape=(16, 32)):
    w = jax.random.normal(jax.random.PRNGKey(0), shape) * 0.05
    return {"w": w}


def _run(mode, grads, **kwargs):
    params = _setup(grads[0].shape)
    opt = scale_with_grug_muonh(learning_rate=0.02, momentum_schedule=lambda _: 0.0, step_mode=mode, **kwargs)
    state = opt.init(params)
    out = []
    for g in grads:
        updates, state = opt.update({"w": g}, state, params)
        out.append(updates["w"])
    return params["w"], out, state


def test_every_mode_keeps_the_norm():
    g = [jax.random.normal(jax.random.PRNGKey(i), (16, 32)) for i in range(3)]
    for mode in MuonHStep:
        w, ups, _ = _run(mode, g)
        for u in ups:
            np.testing.assert_allclose(jnp.linalg.norm(w + u), jnp.linalg.norm(w), rtol=1e-5)


def test_spectral_step_ignores_the_matrix_norm():
    """Doubling |W| doubles the relative step under ``relative`` but halves it under ``spectral``."""
    g = [jax.random.normal(jax.random.PRNGKey(3), (16, 32))]
    rel = {}
    for mode in (MuonHStep.RELATIVE, MuonHStep.SPECTRAL):
        steps = []
        for scale in (1.0, 2.0):
            params = {"w": _setup()["w"] * scale}
            opt = scale_with_grug_muonh(learning_rate=0.02, momentum_schedule=lambda _: 0.0, step_mode=mode)
            u, _ = opt.update({"w": g[0]}, opt.init(params), params)
            # Small steps: the chord length before/after projection is about the step length.
            steps.append(float(jnp.linalg.norm(u["w"]) / jnp.linalg.norm(params["w"])))
        rel[mode] = steps[1] / steps[0]
    np.testing.assert_allclose(rel[MuonHStep.RELATIVE], 1.0, rtol=1e-3)
    np.testing.assert_allclose(rel[MuonHStep.SPECTRAL], 0.5, rtol=2e-2)


def test_grad_step_follows_the_smoothed_gradient_norm_after_the_reference_window():
    base = jax.random.normal(jax.random.PRNGKey(4), (16, 32))
    grads = [base] * 3 + [0.25 * base]
    _, rel_ups, _ = _run(MuonHStep.RELATIVE, grads)
    _, grad_ups, state = _run(MuonHStep.GRAD, grads, step_ref_steps=3)
    for r, g in zip(rel_ups[:3], grad_ups[:3], strict=True):
        np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-8)
    ratio = float(jnp.linalg.norm(grad_ups[3]) / jnp.linalg.norm(rel_ups[3]))
    # The gradient-norm EMA (0.95) moves 5% of the way toward the quartered gradient.
    np.testing.assert_allclose(ratio, 0.95 + 0.05 * 0.25, rtol=2e-2)
    assert int(state.step_count) == 4


def test_model_optimizer_builds_and_steps_in_every_mode():
    mesh, model = t._model(ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)
    for mode in ("spectral", "grad"):
        with jax.set_mesh(mesh):
            updates = knobs._two_steps(GrugMoeMuonHConfig(muon_bimaxwell=True, muonh_step=mode), params)
        for leaf in jax.tree.leaves(eqx.filter(updates, eqx.is_inexact_array)):
            assert np.all(np.isfinite(np.asarray(leaf)))
