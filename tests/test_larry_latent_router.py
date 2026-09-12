# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, set_mesh

from experiments.grug.moe_latent_gated_router.larry_launch import build_config
from experiments.grug.moe_latent_gated_router.larry_model import GrugModelConfig, MoEMLP


def test_router_uses_normalized_gated_latent_and_backpropagates():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        moe_latent_dim=8,
        moe_latent_norm=True,
        moe_latent_gated_norm=True,
        router_input="gated_latent",
        num_layers=1,
        num_heads=1,
        moe_implementation="scatter",
    )
    with set_mesh(mesh):
        mlp = MoEMLP.init(config, key=jax.random.key(0))
        mlp = eqx.tree_at(lambda m: m.moe_down, mlp, jnp.eye(16, 8))
        mlp = eqx.tree_at(lambda m: m.moe_latent_gate.w_up, mlp, jnp.zeros_like(mlp.moe_latent_gate.w_up))
        x = jax.random.normal(jax.random.key(1), (1, 4, 16))
        actual = mlp(x)
        perturbed = mlp(x.at[..., 8:].add(100))
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(perturbed), strict=True):
            np.testing.assert_array_equal(left, right)
        projected = x.reshape(-1, 16)[:, :8]
        normalized = projected / jnp.sqrt(jnp.mean(projected**2, axis=-1, keepdims=True) + config.layer_norm_eps)
        logits = 0.5 * normalized @ mlp.router
        expected = jnp.mean(jax.scipy.special.logsumexp(logits, axis=-1) ** 2)
        np.testing.assert_allclose(actual[1]["router_z_loss"], expected, rtol=1e-5, atol=1e-6)
        grads = eqx.filter_grad(lambda m: m(x)[1]["router_z_loss"])(mlp)
    assert np.linalg.norm(grads.moe_down) > 0
    assert np.linalg.norm(grads.moe_latent_rms.weight) > 0
    assert np.linalg.norm(grads.moe_latent_gate.w_up) > 0


def test_recorded_optimizer_assigns_raw_gate_matrices_to_muonh():
    config = build_config(512, "optimizer-mask-check")
    params = {
        "blocks": {
            "mlp": {
                "moe_latent_gate": {"w_down": jnp.ones((8, 4)), "w_up": jnp.ones((4, 8))},
                "moe_latent_rms": {"weight": jnp.ones((8,))},
                "router": jnp.ones((8, 4)),
                "expert_mlp": {"w_gate": jnp.ones((4, 8, 16))},
            }
        },
        "lm_head": jnp.ones((8, 32)),
    }
    mask = config.optimizer.create_mask(params)
    mlp = mask["blocks"]["mlp"]
    assert mlp["moe_latent_gate"] == {"w_down": "muonh", "w_up": "muonh"}
    assert mlp["moe_latent_rms"]["weight"] == "adam"
    assert mlp["router"] == "adam"
    assert mlp["expert_mlp"]["w_gate"] == "muonh"
    assert mask["lm_head"] == "adamh"
