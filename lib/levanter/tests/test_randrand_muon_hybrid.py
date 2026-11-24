# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Notebook parity, switch boundaries, and the inactive-blend control."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from levanter.optim import randrand_core as rr
from levanter.optim.randrand_muon_hybrid import MuonBlendConfig, randrand_muon_hybrid


@pytest.mark.timeout(180)
def test_notebook_trajectory_switch_and_latched_blend(monkeypatch):
    data = np.load(Path(__file__).parent / "data/randrand/switch_reference.npz")
    actual_basis = rr.lowrank_basis_from_sketch

    def shared_basis(gram, sketch, rank):
        shape = sketch.shape
        shared = np.random.default_rng(shape[0] * 1000 + shape[1]).standard_normal(shape).astype(np.float32)
        return actual_basis(gram, jnp.asarray(shared), rank)

    monkeypatch.setattr(rr, "lowrank_basis_from_sketch", shared_basis)
    config = replace(
        rr.RandRANDConfig(),
        matrix_lr=0.006,
        matrix_betas=(0.95, 0.95),
        embed_lr=0.1,
        vector_lr=0.06,
        low_rank_correction=False,
        correction_scale=0.0,
        rank=2,
        rank_sketch=3,
        pair_k=1,
        min_dim_for_deflation=1,
    )
    labels = ("matrix", "matrix", "matrix", "embed", "vector", "head")
    tx = randrand_muon_hybrid(labels, config)
    params = tuple(jnp.asarray(data[f"initial_{i}"]) for i in range(6))
    state = tx.init(params)

    @jax.jit
    def step(p, s, g):
        u, s = tx.update(g, s, p)
        return optax.apply_updates(p, u), s

    checkpoints = set(data["checkpoints"].tolist())
    max_error = 0.0
    for t in range(1450):
        params, state = step(params, state, tuple(jnp.asarray(data[f"grads_{i}"][t]) for i in range(6)))
        if t + 1 not in checkpoints:
            continue
        expected = data[f"{t+1}_blend"]
        np.testing.assert_allclose(state.blend.weight, expected[0], atol=1e-5, rtol=1e-5)
        if np.isfinite(expected[1]):
            np.testing.assert_allclose(state.blend.cosine, expected[1], atol=1e-5, rtol=1e-5)
        if np.isfinite(expected[2]):
            np.testing.assert_allclose(state.blend.cosine_ema, expected[2], atol=1e-5, rtol=1e-5)
        for i, p in enumerate(params):
            ref = data[f"{t+1}_p_{i}"]
            max_error = max(max_error, float(np.max(np.abs(np.asarray(p) - ref))))
            np.testing.assert_allclose(p, ref, atol=1e-5, rtol=1e-5)
            for field, values in [
                ("exp_avg", state.randrand.first),
                ("exp_avg_sq", state.randrand.second),
                ("mh_buf", state.momentum),
                ("prev_update", state.previous_update),
            ]:
                if values[i] is not None:
                    np.testing.assert_allclose(values[i], data[f"{t+1}_{field}_{i}"], atol=1e-6, rtol=1e-5)
        if t + 1 == 1000:
            assert float(state.blend.weight) == 0.0
        if t + 1 == 1001:
            np.testing.assert_allclose(state.blend.weight, 1 / 400, atol=1e-7)
    assert 0.7 < float(state.blend.weight) < 1.0
    assert float(state.blend.cosine_ema) > 0
    print(f"Notebook switch: max parameter absolute error {max_error:.3e}")


def test_disabled_blend_is_exact_existing_randrand():
    config = replace(
        rr.RandRANDConfig(),
        rank=2,
        rank_sketch=3,
        pair_k=1,
        min_dim_for_deflation=1,
        randrand_warmup=2,
        basis_every=2,
        warmup_steps=2,
        max_grad_norm=None,
    )
    labels = ("matrix", "vector")
    params = (jnp.arange(24, dtype=jnp.float32).reshape(6, 4) * 0.01, jnp.ones(4, jnp.float32))
    base = rr.randrand(labels, config, total_steps=12)
    hybrid = randrand_muon_hybrid(labels, config, MuonBlendConfig(min_step=1000), total_steps=12)
    a, b = base.init(params), hybrid.init(params)
    base_step, hybrid_step = jax.jit(base.update), jax.jit(hybrid.update)
    for t in range(8):
        grads = jax.tree.map(lambda p: jnp.sin(p + t), params)
        ua, a = base_step(grads, a, params)
        ub, b = hybrid_step(grads, b, params)
        for x, y in zip(jax.tree.leaves((ua, a)), jax.tree.leaves((ub, b.randrand)), strict=True):
            np.testing.assert_array_equal(x, y)
        params = optax.apply_updates(params, ua)
    assert float(b.blend.weight) == 0
