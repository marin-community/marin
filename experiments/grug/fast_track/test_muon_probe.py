# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Muon probe separates reliable singular directions from noise on held-out batches."""

import itertools

import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.muon_probe as mp


def _signal(rank: int, m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((m, rank)))
    v, _ = np.linalg.qr(rng.standard_normal((n, rank)))
    return (u * np.geomspace(4.0, 1.0, rank)) @ v.T


def test_probe_finds_signal_directions_reliable_and_noise_directions_not(tmp_path):
    m, n, rank = 12, 10, 3
    signal = _signal(rank, m, n, 0)
    rng = np.random.default_rng(1)

    def gradients(params, batch, step):
        # Batch-dependent noise on top of a fixed low-rank signal.
        noise = np.random.default_rng(int(batch)).standard_normal((m, n)) * 0.5
        return {"w": jnp.asarray(signal + noise, jnp.float32)}, {"w": params["w"]}

    params = {"w": jnp.asarray(rng.standard_normal((m, n)), jnp.float32)}
    probe = mp.MuonProbe(5, str(tmp_path), lambda p: {"w": p["w"]}, itertools.count(100))
    for step in range(4, 5 + mp.TRAJECTORY_STEPS + 1):
        probe.before_step(step, params, jnp.asarray(step), gradients)
        params = {"w": params["w"] - 0.01 * jnp.asarray(signal, jnp.float32)}  # the "optimizer step"

    out = np.load(tmp_path / "muon_probe_step5.npz")
    s = out["heldout_grad/w"]
    assert s.shape == (mp.HELDOUT_BATCHES, n)
    consistency = s.mean(0) / np.sqrt((s**2).mean(0))
    assert np.all(consistency[:rank] > 0.8)
    assert np.all(np.abs(consistency[rank:]) < 0.5)
    assert out["trajectory_grad/w"].shape == (mp.TRAJECTORY_STEPS + 1, n)
    assert out["sigma_ns/w"].shape == (n,) and out["sigma_update/w"].shape == (n,)
