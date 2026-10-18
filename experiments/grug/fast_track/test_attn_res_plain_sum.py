# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``attn_res_plain_sum``: every AttnRes gate returns the plain sum of its sources, so the model is the standard
residual-stream model, in the unrolled AttnRes loop and under ``scan_layers``."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
import experiments.grug.fast_track.test_scan_layers as ts
from experiments.grug.fast_track.model import LocalMixer

_SLIDING = dict(local_mixer=LocalMixer.SLIDING_WINDOW, num_layers=8, attn_res_num_blocks=4)


def _with_params_of(target, source):
    """``target`` with every leaf that ``source`` has at the same path replaced by ``source``'s."""
    values = {jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(source)}
    return jax.tree_util.tree_map_with_path(lambda path, leaf: values.get(jax.tree_util.keystr(path), leaf), target)


def _loss_and_grads(model, mesh, tokens):
    def loss_fn(m, x):
        return m.next_token_loss(x, jnp.ones(x.shape, jnp.float32))

    with jax.set_mesh(mesh):
        return eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model, tokens)


@pytest.mark.timeout(240)
def test_plain_sum_matches_the_standard_residual_model():
    mesh, standard = t._model(attn_res=False, **_SLIDING)
    _, plain = t._model(attn_res_plain_sum=True, **_SLIDING)
    plain = _with_params_of(plain, standard)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    loss, grads = _loss_and_grads(standard, mesh, tokens)
    plain_loss, plain_grads = _loss_and_grads(plain, mesh, tokens)
    np.testing.assert_allclose(float(plain_loss), float(loss), rtol=1e-6)
    shared = _with_params_of(jax.tree.map(jnp.zeros_like, eqx.filter(plain_grads, eqx.is_inexact_array)), grads)
    for got, want in zip(
        jax.tree.leaves(eqx.filter(plain_grads, eqx.is_inexact_array)), jax.tree.leaves(shared), strict=True
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-6)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("two_phase", [False, True])
def test_scanned_plain_sum_matches_the_unrolled_plain_sum(two_phase):
    loss, grads, _ = ts._loss_grads_metrics(attn_res_plain_sum=True, **_SLIDING)
    scan_loss, scan_grads, _ = ts._loss_grads_metrics(
        scan_layers=True, attn_res_plain_sum=True, attn_res_two_phase=two_phase, **_SLIDING
    )
    np.testing.assert_allclose(scan_loss, loss, rtol=1e-6)
    for unrolled, scanned in zip(grads, scan_grads, strict=True):
        np.testing.assert_allclose(scanned, unrolled, rtol=1e-4, atol=1e-6 * max(1.0, np.max(np.abs(unrolled))))
