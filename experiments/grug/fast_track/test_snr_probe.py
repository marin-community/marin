# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SNR probe: the candidate updates weight directions by their gradient SNR, and probe deltas land on the right
matrices."""

import jax
import jmp
import numpy as np
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.snr_probe as sp
import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.grad_capture import add_to_captured, capture_matrices
from experiments.grug.fast_track.train import _make_probe_loss


def _polar(x: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(x, full_matrices=False)
    return u @ vt


def _stream(steps: int = 48, m: int = 10, n: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients with a steady direction ``a`` and a direction ``b`` whose sign is random every step."""
    rng = np.random.default_rng(0)
    u, _ = np.linalg.qr(rng.standard_normal((m, 2)))
    v, _ = np.linalg.qr(rng.standard_normal((n, 2)))
    a, b = np.outer(u[:, 0], v[:, 0]), np.outer(u[:, 1], v[:, 1])
    grads = np.stack([a + 3.0 * rng.choice([-1.0, 1.0]) * b for _ in range(steps)])
    return grads, a, b


def test_svd_snr_keeps_the_steady_direction_and_mutes_the_flipping_one():
    grads, a, b = _stream()
    cands = sp.candidate_directions(grads, sp.BETA, _polar)
    muon, snr = cands["muon"], cands["svd_snr_diag"]
    # Muon gives both directions a unit weight; the SNR weights follow each direction's sign agreement over time.
    assert abs(np.sum(muon * b)) > 0.9
    assert np.sum(snr * a) > 0.95
    assert abs(np.sum(snr * b)) < 0.5


def test_eig_and_elementwise_snr_are_bounded_by_one():
    grads, _, _ = _stream()
    cands = sp.candidate_directions(grads, sp.BETA, _polar)
    assert np.max(np.abs(cands["adam_snr"])) <= 1.0 + 1e-9
    # In the rotated basis every coordinate is an SNR, so the rotated-back matrix has Frobenius norm <= sqrt(mn).
    assert np.linalg.norm(cands["eig_snr"]) <= np.sqrt(grads[0].size) + 1e-9


def test_sphere_step_moves_by_the_relative_step_and_keeps_the_norm():
    rng = np.random.default_rng(1)
    w, d = rng.standard_normal((6, 5)), rng.standard_normal((6, 5))
    d -= np.sum(d * w) / np.sum(w * w) * w  # tangent to the sphere, so the projection back is second order
    delta = sp.sphere_step(w, d, 0.01)
    np.testing.assert_allclose(np.linalg.norm(w + delta), np.linalg.norm(w), rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(delta) / np.linalg.norm(w), 0.01, rtol=1e-3)
    assert np.sum(delta * d) < 0


def test_deltas_land_only_on_their_captured_matrix():
    mesh, model = t._model(mla=True)
    sites = ("L2.kda.w_v", "L3.mla.w_uk", "L0.latent.w_up", "L5.expert2.w_down", "L0.shared.w_up")
    with jax.set_mesh(mesh):
        before = jax.device_get(jax.jit(capture_matrices)(model))
        deltas = {s: np.full(before[s].shape, 0.5, np.float32) for s in sites}
        after = jax.device_get(jax.jit(lambda m, d: capture_matrices(add_to_captured(m, d)))(model, deltas))
    for name in before:
        expected = before[name] + (0.5 if name in sites else 0.0)
        np.testing.assert_allclose(after[name], expected, rtol=0, atol=1e-6)


def test_probe_loss_changes_only_with_the_deltas():
    mesh, model = t._model(mla=True)
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, t._SEQ), 0, t._VOCAB)
    batch = GrugLmExample(tokens=tokens, loss_weight=np.ones(tokens.shape, np.float32), attn_mask=AttentionMask.causal())
    loss = _make_probe_loss(jmp.get_policy("params=float32,compute=float32,output=float32"))
    cfg = model.config
    betas = np.zeros((cfg.num_layers, cfg.num_experts), np.float32)
    with jax.set_mesh(mesh):
        shape = jax.device_get(jax.jit(capture_matrices)(model))["L2.kda.w_v"].shape
        untouched = float(loss(model, batch, betas, {}))
        zero = float(loss(model, batch, betas, {"L2.kda.w_v": np.zeros(shape, np.float32)}))
        moved = float(loss(model, batch, betas, {"L2.kda.w_v": np.full(shape, 0.3, np.float32)}))
    np.testing.assert_allclose(zero, untouched, rtol=1e-6)
    assert abs(moved - zero) > 1e-4


def test_probe_scores_every_candidate_on_a_quadratic(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "WINDOW", 8)
    monkeypatch.setattr(sp, "HELDOUT_BATCHES", 4)
    monkeypatch.setattr(sp, "EVAL_BATCHES", 2)
    rng = np.random.default_rng(3)
    target = rng.standard_normal((6, 5))
    site = "L2.kda.w_v"

    def gradients(params, batch, step):
        noise = np.random.default_rng(int(batch)).standard_normal((6, 5)) * 0.1
        return {site: jax.numpy.asarray(params[site] - target + noise, jax.numpy.float32)}, None

    def losses(params, batch, deltas):
        return 0.5 * jax.numpy.sum(jax.numpy.square(params[site] + deltas[site] - target))

    params = {site: jax.numpy.asarray(rng.standard_normal((6, 5)), jax.numpy.float32)}
    probe = sp.SnrProbe(10, str(tmp_path), lambda p: dict(p), iter(range(100, 1000)))
    for step in range(0, 11):
        probe.before_step(step, params, np.asarray(step), gradients, losses)
        params = {site: params[site] - 0.05 * (params[site] - target)}

    out = np.load(tmp_path / "snr_probe_step10.npz")
    assert out[f"payoff/{site}"].shape == (len(sp.CANDIDATES), 4)
    base = out["loss/base"]
    for cand in sp.CANDIDATES:
        # Every candidate is a descent direction on this quadratic, so a step lowers the loss.
        assert np.all(out[f"loss/kda/{cand}/x1"] < base)
