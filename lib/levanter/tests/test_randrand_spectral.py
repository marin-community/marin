# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Independent notebook replay for spectral filtering and adaptive blending."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from levanter.optim import randrand_core as rr
from levanter.optim import randrand_spectral as spectral
from levanter.optim.randrand_muon_hybrid import MuonBlendConfig, randrand_muon_hybrid


def _reference_banks(data, config):
    # Replay the *observed* notebook probes at each independently sampled bank.
    # Framework PRNG streams differ. Match probes, but leave QR/SVD/filter/Adam
    # arithmetic under test; the fixture was exported from unchanged notebook cells.
    keys = []
    key = jax.random.PRNGKey(config.basis_seed)
    for _ in range(3):
        key, active, standby = jax.random.split(key, 3)
        keys.extend((active, standby))
    for refresh in range(1, 11):
        key, refresh_key = jax.random.split(key)
        if refresh % 5 == 0:
            keys.extend(jax.random.fold_in(refresh_key, i) for i in range(3))
    return keys


@pytest.mark.timeout(180)
def test_spectral_notebook_trajectory_bank_handovers_and_blend(monkeypatch):
    data = {}
    for path in sorted((Path(__file__).parent / "data/randrand").glob("spectral_switch_reference_*.npz")):
        with np.load(path) as shard:
            data.update({key: shard[key] for key in shard.files})
    config = replace(
        rr.RandRANDConfig(),
        rank=3,
        rank_sketch=4,
        min_dim_for_deflation=1,
        randrand_warmup=200,
        basis_every=200,
        basis_ema_beta=0.98,
    )
    keys = _reference_banks(data, config)

    def shared_bank(key, rows, columns, width):
        matches = [
            i
            for i in range(len(keys))
            if data[f"probe_left_{i}"].shape == (rows, width) and data[f"probe_right_{i}"].shape == (columns, width)
        ]
        candidates = jnp.stack([keys[i] for i in matches])
        index = jnp.argmax(jnp.all(candidates == key, axis=1))
        left = jnp.stack([jnp.asarray(data[f"probe_left_{i}"]) for i in matches])[index]
        right = jnp.stack([jnp.asarray(data[f"probe_right_{i}"]) for i in matches])[index]
        return spectral.SketchBank(left, right, jnp.zeros_like(left), jnp.zeros_like(right))

    monkeypatch.setattr(spectral, "draw_bank", shared_bank)
    labels = ("matrix", "matrix", "matrix", "embed", "vector", "head")
    tx = randrand_muon_hybrid(labels, config, spectral=spectral.SpectralSettings())
    params = tuple(jnp.asarray(data[f"initial_{i}"]) for i in range(6))
    state = tx.init(params)

    @jax.jit
    def step(p, s, g):
        updates, s = tx.update(g, s, p)
        return optax.apply_updates(p, updates), s

    checkpoints = set(data["checkpoints"].tolist())
    max_error = 0.0
    for t in range(2002):
        params, state = step(params, state, tuple(jnp.asarray(data[f"grads_{i}"][t]) for i in range(6)))
        if t + 1 not in checkpoints:
            continue
        expected = data[f"{t+1}_blend"]
        actual = (state.blend.weight, state.blend.cosine, state.blend.cosine_ema)
        for a, b in zip(actual, expected, strict=True):
            if np.isfinite(b):
                np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        for i, p in enumerate(params):
            ref = data[f"{t+1}_p_{i}"]
            max_error = max(max_error, float(np.max(np.abs(np.asarray(p) - ref))))
            np.testing.assert_allclose(p, ref, atol=1e-5, rtol=1e-5)
            for field, values in (
                ("exp_avg", state.randrand.first),
                ("exp_avg_sq", state.randrand.second),
                ("mh_buf", state.momentum),
                ("prev_update", state.previous_update),
            ):
                if values[i] is not None:
                    np.testing.assert_allclose(values[i], data[f"{t+1}_{field}_{i}"], atol=1e-6, rtol=1e-5)
            ms = state.randrand.matrices[i]
            if ms is not None:
                for suffix, bank in (("", ms.active), ("_standby", ms.standby)):
                    for side in ("left", "right"):
                        np.testing.assert_allclose(
                            getattr(bank, f"{side}_ema"),
                            data[f"{t+1}_basis_{side}_ema{suffix}_{i}"],
                            atol=1e-6,
                            rtol=1e-5,
                        )
        if t + 1 == 1000:
            assert float(state.blend.weight) == 0
        if t + 1 == 1001:
            np.testing.assert_allclose(state.blend.weight, 1 / 400, atol=1e-7)
    assert 0.7 < float(state.blend.weight) < 1
    assert float(state.blend.cosine_ema) > 0
    print(f"Spectral notebook: max sampled parameter absolute error {max_error:.3e}")


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (4, 4)])
def test_spectral_filter_matches_dense_operators(shape):
    rng = np.random.default_rng(42)
    x = rng.normal(size=shape).astype(np.float32)
    left = np.linalg.qr(rng.normal(size=(shape[0], 3)))[0].astype(np.float32)
    right = np.linalg.qr(rng.normal(size=(shape[1], 3)))[0].astype(np.float32)
    values_left, values_right = np.array([100.0, 4.0, 1.0]), np.array([9.0, 2.0, 0.1])

    def operator(basis, values):
        mu = 1e-6 * values[0]
        scales = ((values[-1] + mu) / (values[:-1] + mu)) ** 0.25
        return np.eye(basis.shape[0]) + (basis[:, :-1] * (scales - 1)) @ basis[:, :-1].T

    expected = operator(left, values_left) @ x @ operator(right, values_right)
    bank = spectral.draw_bank(jax.random.PRNGKey(0), *shape, 3)
    state = spectral.SpectralMatrixState(
        bank,
        bank,
        jnp.asarray(left),
        jnp.asarray(right),
        jnp.asarray(values_left, dtype=jnp.float32),
        jnp.asarray(values_right, dtype=jnp.float32),
        jnp.int32(1),
        jnp.bool_(True),
    )
    actual = jax.jit(spectral.spectral_filter)(jnp.asarray(x), state, 1e-6)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_spectral_disabled_blend_and_zero_history():
    config = replace(
        rr.RandRANDConfig(),
        rank=2,
        rank_sketch=3,
        min_dim_for_deflation=1,
        randrand_warmup=2,
        basis_every=2,
        warmup_steps=2,
        max_grad_norm=None,
    )
    settings = spectral.SpectralSettings(probe_every=2)
    labels = ("matrix", "head")
    params = (jnp.arange(24, dtype=jnp.float32).reshape(6, 4) * 0.01, jnp.ones((5, 4), jnp.float32))
    base = spectral.randrand(labels, config, settings, total_steps=12)
    blend = randrand_muon_hybrid(labels, config, MuonBlendConfig(min_step=1000), total_steps=12, spectral=settings)
    a, b = base.init(params), blend.init(params)
    base_step, blend_step = jax.jit(base.update), jax.jit(blend.update)
    for t in range(8):
        grads = jax.tree.map(lambda p: jnp.zeros_like(p) if t < 4 else jnp.sin(p + t), params)
        ua, a = base_step(grads, a, params)
        ub, b = blend_step(grads, b, params)
        for x, y in zip(jax.tree.leaves((ua, a)), jax.tree.leaves((ub, b.randrand)), strict=True):
            np.testing.assert_array_equal(x, y)
        if t < 4:
            # An empty sketch must not turn on boosted decay.
            expected = (
                -rr.lr_multiplier(t, total_steps=12, warmup_steps=2)
                * config.matrix_lr
                * config.matrix_weight_decay
                * params[0]
            )
            np.testing.assert_allclose(ua[0], expected, atol=1e-7, rtol=1e-5)
        params = optax.apply_updates(params, ua)
    assert float(b.blend.weight) == 0
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(a))


def test_sketch_banks_track_raw_gradient_second_moments():
    config = replace(
        rr.RandRANDConfig(),
        rank=2,
        rank_sketch=3,
        min_dim_for_deflation=1,
        randrand_warmup=1,
        basis_every=1,
        warmup_steps=1,
        max_grad_norm=None,
    )
    tx = spectral.randrand(("matrix",), config, spectral.SpectralSettings(probe_every=100), total_steps=12)
    params = (jnp.ones((5, 4), jnp.float32),)
    state = tx.init(params)
    step = jax.jit(tx.update)
    left, right = np.zeros((5, 5)), np.zeros((4, 4))
    rng = np.random.default_rng(5)
    for _ in range(6):
        g = rng.normal(size=(5, 4)).astype(np.float32)
        beta = config.basis_ema_beta
        left = beta * left + (1 - beta) * (g @ g.T)
        right = beta * right + (1 - beta) * (g.T @ g)
        updates, state = step((jnp.asarray(g),), state, params)
        params = optax.apply_updates(params, updates)
        for bank in (state.matrices[0].active, state.matrices[0].standby):
            np.testing.assert_allclose(bank.left_probe.T @ bank.left_probe, np.eye(3), atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(bank.right_probe.T @ bank.right_probe, np.eye(3), atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(bank.left_ema, left @ bank.left_probe, atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(bank.right_ema, right @ bank.right_probe, atol=1e-6, rtol=1e-5)
