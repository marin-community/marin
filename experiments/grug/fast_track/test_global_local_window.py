# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``global_local_window``: a short sliding-window softmax branch in the global layers, a no-op at init."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t


def _tokens():
    return jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)


def test_branch_only_in_global_layers_and_a_no_op_at_init():
    mesh, plain = t._model()
    _, local = t._model(global_local_window=8, global_local_heads=2, global_local_head_dim=8)
    assert [layer.local_attn is not None for layer in local.layers()] == [False, False, False, True, False, True]
    tokens = _tokens()
    with jax.set_mesh(mesh):
        forward = eqx.filter_jit(lambda m, x: m(x)[0])
        np.testing.assert_allclose(np.asarray(forward(local, tokens)), np.asarray(forward(plain, tokens)), atol=1e-5)


def test_branch_gate_gets_a_gradient_and_then_changes_the_output():
    mesh, model = t._model(global_local_window=8, global_local_heads=2, global_local_head_dim=8)
    tokens = _tokens()
    loss = lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32))  # noqa: E731
    with jax.set_mesh(mesh):
        grads = eqx.filter_jit(eqx.filter_grad(loss))(model)
        gate_grad = np.asarray(grads.stacked_blocks.stacked.local_attn.out_gate)
        assert np.abs(gate_grad).sum() > 0
        opened = eqx.tree_at(lambda m: m.stacked_blocks.stacked.local_attn.out_gate, model, jnp.ones_like(gate_grad))
        forward = eqx.filter_jit(lambda m, x: m(x)[0])
        assert not np.allclose(np.asarray(forward(opened, tokens)), np.asarray(forward(model, tokens)), atol=1e-6)


def test_window_needs_kda_local_layers():
    with pytest.raises(ValueError):
        t._config(global_local_window=8, local_mixer="attention")
