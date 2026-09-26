# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Online KL-Shampoo optimizer (experiments/grug/fast_track/okls.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.grug.fast_track.okls import (
    _okls_warm_start,
    scale_with_grug_okls,
    scaled_cans_inv_sqrt,
)


def _random_spd(d: int, seed: int, cond_scale: float = 1.0) -> jax.Array:
    """Random symmetric-PD ``(d, d)`` matrix (A Aᵀ / d + I)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d)).astype(np.float32) * cond_scale
    m = a @ a.T / d + np.eye(d, dtype=np.float32)
    return jnp.asarray(m)


@pytest.mark.parametrize("d", [8, 64, 256])
def test_scaled_cans_returns_inverse_sqrt_fp32(d):
    """ScaledCANS computes S^{-1/2}: P S P ≈ I on random SPD (fp32 GEMMs, exact-ish)."""
    s = _random_spd(d, seed=d)
    p = scaled_cans_inv_sqrt(s, steps=10, matmul_dtype=jnp.float32)
    residual = p @ s @ p - jnp.eye(d)
    err = float(jnp.linalg.norm(residual) / jnp.sqrt(d))
    # Reference against a true symmetric inverse-sqrt.
    evals, evecs = jnp.linalg.eigh(s)
    p_true = evecs @ jnp.diag(evals**-0.5) @ evecs.T
    rel = float(jnp.linalg.norm(p - p_true) / jnp.linalg.norm(p_true))
    assert err < 5e-3, f"P S P deviates from I: rms residual {err}"
    assert rel < 5e-3, f"P deviates from true S^-1/2: rel err {rel}"


@pytest.mark.parametrize("d", [64, 256])
def test_scaled_cans_fp16_accuracy(d):
    """fp16 GEMMs (the fast device lever, matching the released kernel) stay close to S^{-1/2}."""
    s = _random_spd(d, seed=d + 1)
    p = scaled_cans_inv_sqrt(s, steps=10, matmul_dtype=jnp.float16)
    err = float(jnp.linalg.norm(p @ s @ p - jnp.eye(d)) / jnp.sqrt(d))
    assert err < 3e-2, f"fp16 P S P deviates from I: rms residual {err}"
    assert jnp.all(jnp.isfinite(p))


def test_warm_start_is_symmetric_pd():
    """Warm-start factors are symmetric and positive-definite (eigenvalues > 0)."""
    rng = np.random.default_rng(0)
    grad = jnp.asarray(rng.standard_normal((12, 20)).astype(np.float32))
    s_a, s_b = _okls_warm_start(grad, eps=1e-9)
    assert s_a.shape == (12, 12) and s_b.shape == (20, 20)
    for s in (s_a, s_b):
        assert float(jnp.max(jnp.abs(s - s.T))) < 1e-4
        assert float(jnp.min(jnp.linalg.eigvalsh(s))) > 0.0
    # ScaledCANS on the warm-start factor also recovers a valid inverse root.
    p_a = scaled_cans_inv_sqrt(s_a, matmul_dtype=jnp.float32)
    err = float(jnp.linalg.norm(p_a @ s_a @ p_a - jnp.eye(12)) / jnp.sqrt(12))
    assert err < 1e-2


def _okls_transform(hyperball: bool = False):
    return scale_with_grug_okls(
        beta1=0.9684,
        beta2=0.9482,
        eps=1e-9,
        weight_decay=0.0303,
        cans_steps=10,
        matmul_dtype=jnp.float32,
        learning_rate=jnp.asarray(0.09, jnp.float32),
        lr_peak=0.09,
        hyperball=hyperball,
    )


def test_okls_step_shapes_and_finite_2d_3d_4d():
    """A single OKLS step on 2D/3D/4D matrix leaves: finite deltas, state shapes, count advances."""
    rng = np.random.default_rng(3)

    def rnd(shape):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * 0.02)

    params = {"mat2d": rnd((8, 6)), "stack3d": rnd((3, 8, 6)), "experts4d": rnd((2, 4, 6, 6))}
    grads = {k: rnd(v.shape) for k, v in params.items()}

    tx = _okls_transform()
    state = tx.init(params)
    assert state.S_a["mat2d"].shape == (8, 8)
    assert state.S_b["stack3d"].shape == (3, 6, 6)
    assert state.S_a["experts4d"].shape == (2, 4, 6, 6)

    updates, state2 = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    for k in params:
        assert updates[k].shape == params[k].shape
        assert jnp.all(jnp.isfinite(updates[k])), f"non-finite update for {k}"
        assert jnp.all(jnp.isfinite(new_params[k]))
    assert int(state2.count) == 1

    # Second step exercises the non-warm-start path (count > 0) and must stay finite.
    grads2 = {k: rnd(v.shape) for k, v in params.items()}
    updates2, state3 = tx.update(grads2, state2, new_params)
    assert int(state3.count) == 2
    for k in params:
        assert jnp.all(jnp.isfinite(updates2[k]))


def test_okls_whitening_conditions_anisotropic_gradient():
    """OKLS whitening should reduce gradient anisotropy: the whitened update is better-conditioned
    (smaller top/bottom singular-value ratio) than the raw gradient for a skewed-spectrum gradient."""
    rng = np.random.default_rng(7)
    u = jnp.asarray(np.linalg.qr(rng.standard_normal((16, 16)))[0].astype(np.float32))
    v = jnp.asarray(np.linalg.qr(rng.standard_normal((16, 16)))[0].astype(np.float32))
    spectrum = jnp.asarray(np.geomspace(1.0, 1e-3, 16).astype(np.float32))
    grad = (u * spectrum) @ v.T  # highly anisotropic

    params = {"m": jnp.zeros((16, 16))}
    grads = {"m": grad}
    tx = _okls_transform()
    state = tx.init(params)
    # Warm-started first step already whitens with fresh roots.
    updates, _ = tx.update(grads, state, params)

    def cond_number(x):
        sv = jnp.linalg.svd(x, compute_uv=False)
        return float(sv[0] / sv[-1])

    # Whitening meaningfully reduces anisotropy; the warm-start k·I floor (Shampoo-style) keeps it
    # from fully inverting the 1000:1 spectrum, so expect a solid-but-partial reduction.
    assert cond_number(updates["m"]) < cond_number(grad) / 3.0


def test_okls_hyperball_step_is_norm_preserving():
    """The hyperball ablation moves the parameter while preserving its Frobenius norm (per matrix)."""
    rng = np.random.default_rng(21)
    params = {"stack3d": jnp.asarray(rng.standard_normal((3, 8, 6)).astype(np.float32))}
    grads = {"stack3d": jnp.asarray(rng.standard_normal((3, 8, 6)).astype(np.float32) * 0.05)}
    tx = _okls_transform(hyperball=True)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    new_p = optax.apply_updates(params, updates)
    assert jnp.all(jnp.isfinite(updates["stack3d"]))
    for i in range(3):
        old_norm = float(jnp.linalg.norm(params["stack3d"][i]))
        new_norm = float(jnp.linalg.norm(new_p["stack3d"][i]))
        assert abs(new_norm - old_norm) / old_norm < 1e-4, f"layer {i} norm not preserved"
        assert float(jnp.linalg.norm(new_p["stack3d"][i] - params["stack3d"][i])) > 0
