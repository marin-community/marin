# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-diagnostic capture: sites map to the right layers, gradients match the train step's, files round-trip."""

import jax
import jmp
import numpy as np
import pytest
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.grad_capture import capture_matrices, capture_steps, write_capture
from experiments.grug.fast_track.train import _loss_and_grads, _make_grad_capture_step

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
