# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, set_mesh

from experiments.grug.moe_hero_ep import model as hero
from experiments.grug.moe_latent_gated_router import model


@pytest.fixture
def mesh():
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def config(routing):
    return model.GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        latent_dim=4,
        latent_routing=routing,
        intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=4,
        sliding_window=4,
        initializer_std=0.3,
        attention_implementation="reference",
        moe_implementation="scatter",
    )


def test_control_matches_hero_outputs_and_routing(mesh):
    cfg = config(model.LatentRouting.FULL_WIDTH_RMS)
    hero_cfg = hero.GrugModelConfig(**{k: v for k, v in dataclasses.asdict(cfg).items() if k != "latent_routing"})
    with set_mesh(mesh):
        control = model.MoEMLP.init(cfg, key=jax.random.key(0))
        original = hero.MoEMLP.init(hero_cfg, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        actual = control(x)
        expected = original(x)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def test_router_and_experts_ignore_hidden_components_removed_by_latent_projection(mesh):
    # The last four hidden coordinates lie exactly in the down-projection kernel.
    # Both the routed output and router statistics must be invariant to changing them.
    with set_mesh(mesh):
        mlp = model.MoEMLP.init(config(model.LatentRouting.GATED_LATENT), key=jax.random.key(0))
        mlp = eqx.tree_at(lambda m: m.w_latent_down, mlp, jnp.eye(8, 4))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        perturbed = x.at[..., 4:].add(100)
        actual = mlp(x)
        expected = mlp(perturbed)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def test_zero_latent_gate_halves_normalized_projection_and_router_gradient_reaches_it(mesh):
    with set_mesh(mesh):
        mlp = model.MoEMLP.init(config(model.LatentRouting.GATED_LATENT), key=jax.random.key(2))
        mlp = eqx.tree_at(lambda m: m.latent_gated_norm.w_up, mlp, jnp.zeros_like(mlp.latent_gated_norm.w_up))
        mlp = eqx.tree_at(lambda m: m.latent_norm.weight, mlp, jnp.array([0.5, 1.0, 1.5, 2.0]))
        x = jax.random.normal(jax.random.key(3), (1, 4, 8))
        _, stats = mlp(x)
        # A zero gate matrix must halve the normalized feature, including its
        # learned scale. This catches a missing norm or norm placed after the gate.
        projected = np.asarray(x).reshape(-1, 8) @ np.asarray(mlp.w_latent_down)
        normalized = projected / np.sqrt(np.mean(projected**2, axis=-1, keepdims=True) + mlp.cfg.layer_norm_eps)
        z = jnp.asarray(0.5 * normalized * np.array([0.5, 1.0, 1.5, 2.0]))
        logits = z @ mlp.router
        expected = jax.nn.softmax(logits, axis=-1).sum(axis=0)
        np.testing.assert_allclose(stats["router_prob_sum_local"][0], expected, rtol=1e-5, atol=1e-6)
        grads = eqx.filter_grad(lambda m: m(x)[1]["router_z_sq_sum_local"].sum())(mlp)
    assert np.linalg.norm(grads.w_latent_down) > 0
    assert np.linalg.norm(grads.latent_norm.weight) > 0
    assert np.linalg.norm(grads.latent_gated_norm.w_up) > 0


def test_output_matches_selected_dense_experts_on_gated_latent(mesh):
    with set_mesh(mesh):
        mlp = model.MoEMLP.init(config(model.LatentRouting.GATED_LATENT), key=jax.random.key(4))
        x = jax.random.normal(jax.random.key(5), (1, 4, 8))
        actual, _ = mlp(x)
        assert mlp.latent_gated_norm is not None
        assert mlp.latent_norm is not None
        assert mlp.w_latent_down is not None
        z = mlp.latent_gated_norm(mlp.latent_norm(x.reshape(-1, 8) @ mlp.w_latent_down))
        logits = z @ mlp.router
        # Compute all experts densely, then select the per-token winners. This is
        # independent of the grouped dispatch/combine implementation.
        bank = mlp.expert_mlp
        gate = np.einsum("tl,eli->tei", np.asarray(z), np.asarray(bank.w_gate))
        up = np.einsum("tl,eli->tei", np.asarray(z), np.asarray(bank.w_up))
        outputs = np.einsum("tei,eil->tel", gate / (1 + np.exp(-gate)) * up, np.asarray(bank.w_down))
        logits = np.asarray(logits)
        selected = np.argsort(logits + np.asarray(mlp.router_bias), axis=-1)[:, -2:]
        weights = 1 / (1 + np.exp(-np.take_along_axis(logits, selected, axis=-1)))
        weights = 2.5 * weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)
        chosen = np.take_along_axis(outputs, selected[..., None], axis=1)
        expected = ((chosen * weights[..., None]).sum(axis=1) @ np.asarray(mlp.w_latent_up)).reshape(x.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
