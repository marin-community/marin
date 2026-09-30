# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router designs from the Belay thread: a residual tiny-MLP router and router-orthogonal expert reads."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import _orthogonalize_expert_reads


def _tokens():
    return jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)


def _forward(model, tokens):
    return eqx.filter_jit(lambda m, x: m(x)[0])(model, tokens)


def test_each_expert_ignores_its_own_router_direction():
    mesh, model = t._model()
    mlp = jax.tree.map(lambda x: x[0], model.kda_blocks.stacked.mlp)  # layer 0's MoE
    with jax.set_mesh(mesh):
        em = eqx.filter_jit(_orthogonalize_expert_reads)(mlp.expert_mlp, mlp.router, mlp.w_latent_down)
    u = np.asarray(mlp.router).T @ np.asarray(mlp.w_latent_down)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    w_new, w_old = np.asarray(em.w_up), np.asarray(mlp.expert_mlp.w_up)
    for e in range(w_new.shape[0]):
        np.testing.assert_allclose(u[e] @ w_new[e], 0.0, atol=1e-6)
        # Only the router direction was removed: the change is rank one.
        assert np.linalg.matrix_rank(w_new[e] - w_old[e], tol=1e-6) == 1


def test_orthogonal_reads_run_and_change_the_model():
    mesh, plain = t._model()
    _, ortho = t._model(expert_router_orthogonal=True)
    tokens = _tokens()
    with jax.set_mesh(mesh):
        a, b = np.asarray(_forward(plain, tokens)), np.asarray(_forward(ortho, tokens))
    assert np.isfinite(b).all() and not np.allclose(a, b, atol=1e-6)


def test_mlp_router_starts_as_the_linear_router_and_learns():
    mesh, plain = t._model()
    _, mlp_router = t._model(router_mlp_hidden=8)
    tokens = _tokens()
    with jax.set_mesh(mesh):
        np.testing.assert_allclose(
            np.asarray(_forward(mlp_router, tokens)), np.asarray(_forward(plain, tokens)), rtol=1e-5, atol=1e-5
        )
        grads = eqx.filter_jit(
            eqx.filter_grad(lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32)))
        )(mlp_router)
    assert float(jnp.abs(grads.kda_blocks.stacked.mlp.router_mlp_b).sum()) > 0
