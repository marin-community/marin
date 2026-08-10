# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import haliax as hax
import haliax.nn as hnn
from haliax import Axis

from levanter.models.moe import dense_router_delta
from levanter.utils.activation import ActivationFunctionEnum
from test_utils import use_test_mesh


def _inputs(Token: Axis, Embed: Axis, Mlp: Axis, Experts: Axis):
    ks = random.split(random.PRNGKey(0), 5)
    x = hax.random.normal(ks[0], (Token, Embed))
    logits = hax.random.normal(ks[1], (Token, Experts))
    gate_w = hax.random.normal(ks[2], (Experts, Embed, Mlp)) * 0.1
    up_w = hax.random.normal(ks[3], (Experts, Embed, Mlp)) * 0.1
    down_w = hax.random.normal(ks[4], (Experts, Mlp, Embed)) * 0.1
    return x, logits, gate_w, up_w, down_w


def _naive_dense(x, logits, gate_w, up_w, down_w, act, *, Experts, Embed, Mlp):
    """Reference dense forward: sum_e stop_gradient(E_e(x)) * softmax(logits)_e over all experts.

    Materializes the Token x Experts x Mlp intermediates (the memory-naive layout) and is an
    independent oracle for the router gradient the fold-based helper must reproduce.
    """
    weights = hnn.softmax(logits.astype(jnp.float32), axis=Experts).astype(x.dtype)
    hidden = act(x.dot(gate_w, axis=Embed)) * x.dot(up_w, axis=Embed)
    expert_out = jax.lax.stop_gradient(hidden.dot(down_w, axis=Mlp))
    return (expert_out * weights).sum(axis=Experts)


def test_dense_router_delta_value_is_zero():
    Token, Embed, Mlp, Experts = Axis("token", 6), Axis("embed", 8), Axis("mlp", 5), Axis("experts", 4)
    act = ActivationFunctionEnum.silu.to_fn()
    x, logits, gate_w, up_w, down_w = _inputs(Token, Embed, Mlp, Experts)

    with use_test_mesh():
        delta = jax.jit(
            lambda x, lg, g, u, d: dense_router_delta(x, lg, g, u, d, act, Experts=Experts, Embed=Embed, Mlp=Mlp)
        )(x, logits, gate_w, up_w, down_w)

    # Straight-through: the delta must be exactly zero in value so the forward is unchanged.
    np.testing.assert_array_equal(np.asarray(delta.array), np.zeros((Token.size, Embed.size), dtype=np.float32))


def test_dense_router_delta_gradient_matches_naive_dense():
    Token, Embed, Mlp, Experts = Axis("token", 6), Axis("embed", 8), Axis("mlp", 5), Axis("experts", 4)
    act = ActivationFunctionEnum.silu.to_fn()
    x, logits, gate_w, up_w, down_w = _inputs(Token, Embed, Mlp, Experts)
    cotangent = hax.random.normal(random.PRNGKey(7), (Token, Embed))

    def helper_grad(logits):
        def scalar(lg):
            delta = dense_router_delta(x, lg, gate_w, up_w, down_w, act, Experts=Experts, Embed=Embed, Mlp=Mlp)
            return hax.sum(delta * cotangent).scalar()

        return jax.grad(scalar)(logits)

    def naive_grad(logits):
        def scalar(lg):
            out = _naive_dense(x, lg, gate_w, up_w, down_w, act, Experts=Experts, Embed=Embed, Mlp=Mlp)
            return hax.sum(out * cotangent).scalar()

        return jax.grad(scalar)(logits)

    with use_test_mesh():
        g_helper = jax.jit(helper_grad)(logits)
        g_naive = jax.jit(naive_grad)(logits)

    np.testing.assert_allclose(np.asarray(g_helper.array), np.asarray(g_naive.array), rtol=1e-5, atol=1e-6)


def test_dense_router_delta_gradient_reaches_expert_outputs_not_weights():
    # The delta's gradient must reach only the router logits: expert weights and the input get none.
    Token, Embed, Mlp, Experts = Axis("token", 6), Axis("embed", 8), Axis("mlp", 5), Axis("experts", 4)
    act = ActivationFunctionEnum.silu.to_fn()
    x, logits, gate_w, up_w, down_w = _inputs(Token, Embed, Mlp, Experts)
    cotangent = hax.random.normal(random.PRNGKey(7), (Token, Embed))

    def scalar(x, gate_w, up_w, down_w):
        delta = dense_router_delta(x, logits, gate_w, up_w, down_w, act, Experts=Experts, Embed=Embed, Mlp=Mlp)
        return hax.sum(delta * cotangent).scalar()

    with use_test_mesh():
        gx, gg, gu, gd = jax.jit(jax.grad(scalar, argnums=(0, 1, 2, 3)))(x, gate_w, up_w, down_w)

    for g in (gx, gg, gu, gd):
        np.testing.assert_array_equal(np.asarray(g.array), np.zeros_like(np.asarray(g.array)))


def test_dense_router_delta_memory_is_bounded_in_expert_count():
    # The remat'd fold keeps peak temp memory at O(Token * Mlp): growing the expert count 16x must
    # not blow up the dense activation footprint (it would with a Token x Experts x Mlp layout).
    # memory_analysis() reports the compiled scratch (temp) allocation only on CPU; on TPU it comes
    # back as 0 because buffers are assigned in HBM under a different accounting. The property is a
    # graph-structural one, so the CPU lane covers it.
    if jax.default_backend() != "cpu":
        pytest.skip("memory_analysis temp sizes are only populated on the CPU backend")
    Token, Embed, Mlp = Axis("token", 256), Axis("embed", 128), Axis("mlp", 512)
    act = ActivationFunctionEnum.silu.to_fn()

    def peak_temp_bytes(num_experts: int) -> int:
        Experts = Axis("experts", num_experts)
        x, logits, gate_w, up_w, down_w = _inputs(Token, Embed, Mlp, Experts)
        cotangent = hax.random.normal(random.PRNGKey(7), (Token, Embed))

        def scalar(lg):
            delta = dense_router_delta(x, lg, gate_w, up_w, down_w, act, Experts=Experts, Embed=Embed, Mlp=Mlp)
            return hax.sum(delta * cotangent).scalar()

        return jax.jit(jax.grad(scalar)).lower(logits).compile().memory_analysis().temp_size_in_bytes

    with use_test_mesh():
        small = peak_temp_bytes(8)
        large = peak_temp_bytes(128)

    # 16x the experts must stay well under 2x the temp memory (weights aside, activations are flat).
    assert large < 2 * small
