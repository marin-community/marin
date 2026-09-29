# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-diagnostic capture: sites map to the right layers, gradients match the train step's, files round-trip."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jmp
import numpy as np
import pytest
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.analyze_grad_capture as a
import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.grad_capture import capture_matrices, capture_steps, write_capture
from experiments.grug.fast_track.train import _accumulated_loss_and_grads, _loss_and_grads, _make_grad_capture_step

_MP = jmp.get_policy("params=float32,compute=float32,output=float32")


def _setup():
    mesh, model = t._model(mla=True)
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, t._SEQ), 0, t._VOCAB)
    batch = GrugLmExample(tokens=tokens, loss_weight=np.ones(tokens.shape, np.float32), attn_mask=AttentionMask.causal())
    return mesh, model, batch


def test_sites_are_the_named_layers_matrices():
    mesh, model, _ = _setup()
    with jax.set_mesh(mesh):
        sites = jax.device_get(jax.jit(capture_matrices)(model))
    layers = model.layers()
    np.testing.assert_array_equal(sites["L2.kda.w_v"], np.asarray(layers[2].attn.w_v))
    np.testing.assert_array_equal(sites["L3.mla.w_uv"], np.asarray(layers[3].attn.w_uv))
    np.testing.assert_array_equal(sites["L5.router"], np.asarray(layers[5].mlp.router))
    np.testing.assert_array_equal(sites["L0.expert3.w_down"], np.asarray(layers[0].mlp.expert_mlp.w_down[3]))
    np.testing.assert_array_equal(sites["L5.shared.w_up"], np.asarray(layers[5].shared[0].w_up))


def test_captured_gradient_is_the_train_step_gradient():
    mesh, model, batch = _setup()
    cfg = model.config
    betas = np.zeros((cfg.num_layers, cfg.num_experts), np.float32)
    step = np.asarray(7, np.int32)
    with jax.set_mesh(mesh):
        grads, params = jax.device_get(
            _make_grad_capture_step(_MP, z_loss_weight=1e-4)(
                model, batch, betas, step, loop_active=None, router_tie_active=None
            )
        )
        (_, _), full = jax.jit(lambda m: _loss_and_grads(m, batch, _MP, 1e-4, step))(model)
        expected = jax.device_get(jax.jit(capture_matrices)(full))
    assert grads.keys() == expected.keys() == params.keys()
    for name in expected:
        np.testing.assert_allclose(grads[name], expected[name], rtol=1e-5, atol=1e-7)
    assert all(np.abs(grads[name]).sum() > 0 for name in ("L0.kda.w_q", "L3.mla.w_o", "L5.latent.w_up"))


def test_capture_windows_and_file_round_trip(tmp_path):
    assert capture_steps((10, 100), 3) == {10, 11, 12, 100, 101, 102}
    with pytest.raises(ValueError):
        capture_steps((0,), 0)
    grads = {"a": np.ones((2, 3), np.float32)}
    updates = {"a": np.full((2, 3), -0.5, np.float32)}
    path = str(tmp_path / "grad_capture_step10.npz")
    write_capture(path, 10, grads, updates, params=grads)
    loaded = np.load(path)
    assert int(loaded["step"]) == 10
    np.testing.assert_array_equal(loaded["update/a"], updates["a"])
    assert "param/a" in loaded and "grad/a" in loaded


def _kronecker_stream(rng, steps, m, n, row_scale, col_scale, mean_scale=0.0):
    """Gradients with covariance exactly B (x) A (diagonal factors) plus an optional persistent mean."""
    mean = mean_scale * rng.standard_normal((m, n))
    return np.stack(
        [mean + (row_scale[:, None] * rng.standard_normal((m, n))) * col_scale[None, :] for _ in range(steps)]
    )


def test_kl_whitening_removes_kronecker_anisotropy():
    rng = np.random.default_rng(0)
    m, n = 24, 16
    grads = _kronecker_stream(rng, 48, m, n, np.geomspace(0.1, 10, m), np.geomspace(0.3, 3, n))
    w = a.whitening_test(grads, rng)
    # Anisotropic before, and whitened down to the finite-sample spread of an exact Kronecker Gaussian.
    assert w["none"] > 1.0
    assert w["kl"] < 0.6 * w["none"]
    assert abs(w["kl"] - w["gaussian"]) < 0.25
    spectra = a.spectrum_stats(a.kl_factors(grads)[0])
    assert spectra["log10_cond90"] > 1.0


def test_direction_scores_rebuild_muon_and_see_persistent_signal():
    rng = np.random.default_rng(1)
    m, n = 20, 12
    grads = _kronecker_stream(rng, 40, m, n, np.ones(m), np.ones(n), mean_scale=1.0)
    # An applied update that is exactly -polar(Nesterov momentum), as MuonH applies (up to scale).
    buffer, applied = np.zeros((m, n)), []
    for g in grads:
        buffer = a.MOMENTUM * buffer + g
        applied.append(-0.01 * a._polar(g + a.MOMENTUM * buffer))
    scores = a.direction_scores(grads, np.stack(applied))
    assert scores["cos_applied_muon"] > 0.999
    # A persistent mean makes every direction a descent direction on future batches.
    assert all(scores[name]["gain_fro"] > 0 for name in ("sgd", "muon", "okls", "shampoo"))
    # Muon's polar factor has a flat spectrum: stable rank min(m, n).
    assert scores["muon"]["stable_rank"] > 0.99 * n


_EP_SCRIPT = """
import jax, numpy as np
from jax.sharding import AxisType, Mesh
import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.grad_capture import capture_matrices
from experiments.grug.fast_track.model import Transformer

mesh = Mesh(np.array(jax.devices()[:4]).reshape(1, 1, 4, 1), ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4)
with jax.set_mesh(mesh):
    model = Transformer.init(t._config(mla=True), key=jax.random.PRNGKey(0))
    sites = jax.device_get(jax.jit(capture_matrices)(model))
kda = jax.device_get(model.kda_blocks.stacked)
mla = jax.device_get(model.stacked_blocks.stacked)
np.testing.assert_array_equal(sites["L5.expert2.w_up"], np.asarray(mla.mlp.expert_mlp.w_up)[1, 2])
np.testing.assert_array_equal(sites["L0.expert1.w_down"], np.asarray(kda.mlp.expert_mlp.w_down)[0, 1])
np.testing.assert_array_equal(sites["L3.mla.w_q"], np.asarray(mla.attn.w_q)[0])
print("EP_OK")
"""


def test_capture_on_an_expert_sharded_mesh():
    root = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [root, f"{root}/lib/levanter/src", f"{root}/lib/haliax/src", env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_EP_SCRIPT)], env=env, cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "EP_OK" in result.stdout


def test_accumulated_gradients_average_the_microbatches():
    mesh, model, _ = _setup()
    tokens = jax.random.randint(jax.random.PRNGKey(4), (4, t._SEQ), 0, t._VOCAB)
    batch = GrugLmExample(
        tokens=tokens, loss_weight=jax.numpy.ones(tokens.shape, np.float32), attn_mask=AttentionMask.causal()
    )
    step = np.asarray(3, np.int32)
    halves = [
        GrugLmExample(tokens=tokens[i : i + 2], loss_weight=batch.loss_weight[i : i + 2], attn_mask=batch.attn_mask)
        for i in (0, 2)
    ]
    with jax.set_mesh(mesh):
        (loss, _), grads = jax.jit(lambda m: _accumulated_loss_and_grads(2, m, batch, _MP, None, step, None, None))(
            model
        )
        parts = [jax.jit(lambda m, b=b: _loss_and_grads(m, b, _MP, None, step))(model) for b in halves]
    np.testing.assert_allclose(float(loss), np.mean([float(p[0][0]) for p in parts]), rtol=1e-6)
    expected = jax.tree.map(lambda x, y: (np.asarray(x) + np.asarray(y)) / 2, parts[0][1], parts[1][1])
    got = jax.tree.leaves(jax.device_get(grads))
    for g, e in zip(got, jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(g, e, rtol=1e-4, atol=1e-7)


def test_overshoot_ratio_reads_the_step_size_on_a_quadratic():
    """Gradient descent on 0.5 * h * x^2 with step eta: r = 1 - eta * h (0 at the optimum, -1 at twice it)."""
    for eta_h, expected in ((0.5, 0.5), (1.0, 0.0), (2.0, -1.0)):
        x, grads, applied = np.ones((1, 3, 2)), [], []
        for _ in range(6):
            g = x.copy()  # h = 1
            grads.append(g[0])
            applied.append(-eta_h * g[0])
            x = x - eta_h * g
        result = a.overshoot(np.stack(grads), np.stack(applied))
        assert abs(result["median_ratio"] - expected) < 1e-9
        assert result["frac_overshoot"] == (1.0 if expected < 0 else 0.0)
