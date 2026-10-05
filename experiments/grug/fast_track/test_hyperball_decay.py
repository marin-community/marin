# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``log_hyperball_decay``: the logged decay is the shrink the hyperball re-projection applies to ``W + u``, and
logging leaves the MuonH update unchanged."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.optimizer import (
    GrugMoeMuonHConfig,
    _scale_invariant_hyperball_updates,
    hyperball_metrics,
)


def _tree():
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    params = {"w": jax.random.normal(k1, (8, 6)), "stack": jax.random.normal(k2, (3, 8, 6))}
    directions = {"w": jax.random.normal(k3, (8, 6)), "stack": jax.random.normal(k4, (3, 8, 6))}
    return params, directions


def test_logging_leaves_the_update_unchanged():
    params, directions = _tree()
    plain = _scale_invariant_hyperball_updates(params, directions, 0.05)
    logged, _ = _scale_invariant_hyperball_updates(params, directions, 0.05, with_stats=True)
    for key in params:
        np.testing.assert_allclose(logged[key], plain[key], rtol=1e-6, atol=1e-7)


def test_decay_is_the_reprojection_shrink_and_orthogonal_steps_barely_decay():
    lr = 0.05
    w = jax.random.normal(jax.random.PRNGKey(1), (8, 6))
    d = jax.random.normal(jax.random.PRNGKey(2), (8, 6))
    d_orth = d - jnp.sum(d * w) / jnp.sum(w * w) * w
    for direction, expect_cos in ((d_orth, 0.0), (w, -1.0), (-w, 1.0)):
        _, stats = _scale_invariant_hyperball_updates({"w": w}, {"w": direction}, lr, with_stats=True)
        decay, cos, _ = (float(v) for v in stats["w"].reshape(3))
        np.testing.assert_allclose(cos, expect_cos, atol=1e-5)
        # |u| = lr |W|, so |W + u| / |W| = sqrt(1 + 2 lr cos + lr^2).
        np.testing.assert_allclose(decay, 1 - 1 / np.sqrt(1 + 2 * lr * cos + lr**2), rtol=1e-4, atol=1e-7)
    # An outward step (direction -W moves along +W) decays by about lr; an inward one is a negative decay.
    _, out = _scale_invariant_hyperball_updates({"w": w}, {"w": -w}, lr, with_stats=True)
    assert float(out["w"][0].reshape(())) > 0.04


def test_stacked_leaves_get_one_stat_per_layer():
    params, directions = _tree()
    _, stats = _scale_invariant_hyperball_updates(params, directions, 0.05, with_stats=True)
    assert stats["stack"].shape == (3, 3, 1, 1)
    assert stats["w"].shape == (3, 1, 1)


def test_rescale_is_the_factor_on_the_direction():
    w = jax.random.normal(jax.random.PRNGKey(3), (8, 6))
    d = 7.0 * jax.random.normal(jax.random.PRNGKey(4), (8, 6))
    _, stats = _scale_invariant_hyperball_updates({"w": w}, {"w": d}, 0.05, with_stats=True)
    np.testing.assert_allclose(
        float(stats["w"][2].reshape(())), 0.05 * jnp.linalg.norm(w) / jnp.linalg.norm(d), rtol=1e-5
    )


def test_metrics_cover_muonh_matrices_and_the_adamh_lm_head():
    mesh, model = t._model(ngram_stat_rows=0)
    params = eqx.filter(model, eqx.is_inexact_array)
    config = GrugMoeMuonHConfig(muon_bimaxwell=True, log_hyperball_decay=True)
    with jax.set_mesh(mesh):
        params = jax.tree.map(lambda p: reshard(p, P(*(None,) * p.ndim)), params)
        grads = jax.tree.map(lambda p: 0.01 * jnp.ones_like(p), params)
        opt = config.build(10)
        _, state = eqx.filter_jit(opt.update)(grads, opt.init(params), params)
        metrics = hyperball_metrics(state)
    names = set(metrics)
    assert any(n.startswith("train/hyperball/output_proj/") for n in names), sorted(names)[:5]
    assert "train/hyperball/kda_blocks.stacked.attn.w_q/L0/decay" in names, sorted(names)[:20]
    for n, v in metrics.items():
        assert np.isfinite(float(v)), n
