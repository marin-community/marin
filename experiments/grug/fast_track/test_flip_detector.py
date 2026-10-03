# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.flip_detector import flagged_cosines, flip_directions, side_stats, subspace_overlap

_A, _B = 16, 12


def _updates(steps=200, seed=0):
    """Steady drift, plus one input direction ``u`` whose update flips sign every step. The drift has no component
    along ``u``, so the purely alternating direction the detector flags is ``u`` itself."""
    rng = np.random.default_rng(seed)
    u = np.zeros(_A)
    u[3] = 1.0
    drift = rng.normal(size=(_A, _B)) * (1.0 - u)[:, None]
    w = rng.normal(size=_B)
    return [drift + 3.0 * (-1) ** t * np.outer(u, w) + 0.3 * rng.normal(size=(_A, _B)) for t in range(steps)], u


def test_detector_finds_the_flipping_input_direction():
    ds, u = _updates()
    beta = 0.95
    cross = np.zeros((_A, _A))
    energy = np.zeros((_A, _A))
    for d, dp in zip(ds[1:], ds[:-1], strict=True):
        c, e = side_stats(jnp.asarray(d), jnp.asarray(dp), "in")
        cross = beta * cross + (1 - beta) * np.asarray(c)
        energy = beta * energy + (1 - beta) * np.asarray(e)
    rho, basis = flip_directions(cross, energy, k=1)
    assert rho[0] < -0.8 and rho[1] > -0.2 and rho[-1] > 0.8  # one flipping direction; the rest drift or are noise
    assert subspace_overlap(basis, u[:, None]) > 0.95
    f_cos, r_cos, share = flagged_cosines(jnp.asarray(ds[-1]), jnp.asarray(ds[-2]), jnp.asarray(basis), "in")
    assert float(f_cos) < -0.5 and float(r_cos) > 0.5 and 0.0 < float(share) < 1.0


def test_stats_and_cosines_handle_a_layer_axis_and_the_output_side():
    ds, _ = _updates(steps=3)
    d = jnp.stack([jnp.asarray(ds[2]), jnp.asarray(ds[1])])
    dp = jnp.stack([jnp.asarray(ds[1]), jnp.asarray(ds[0])])
    c, e = side_stats(d, dp, "out")
    assert c.shape == (2, _B, _B) and e.shape == (2, _B, _B)
    basis = jnp.broadcast_to(jnp.eye(_B)[:, :2], (2, _B, 2))
    f_cos, r_cos, share = flagged_cosines(d, dp, basis, "out")
    assert f_cos.shape == r_cos.shape == share.shape == (2,)
