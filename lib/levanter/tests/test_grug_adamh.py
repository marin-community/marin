# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GrugAdamHConfig: masking routes weight matrices to AdamH, scalars/embeddings to Adam."""

import jax
import jax.numpy as jnp

from levanter.optim.grug_adamh import GrugAdamHConfig


def _make_fake_grug_params():
    """Minimal pytree mimicking grug MoE parameter structure."""
    return {
        "embed": jnp.zeros((128, 64)),  # embedding: 2D but name contains "embed"
        "layers": {
            "attn_w_q": jnp.zeros((64, 64)),  # weight matrix
            "attn_w_k": jnp.zeros((64, 64)),  # weight matrix
            "mlp_w1": jnp.zeros((64, 192)),  # weight matrix
            "mlp_w2": jnp.zeros((192, 64)),  # weight matrix
            "norm_weight": jnp.zeros((64,)),  # 1D norm scale
        },
        "router_weight": jnp.zeros((64, 8)),  # router: 2D but name contains "router"
        "lm_head": jnp.zeros((64, 128)),  # weight matrix
    }


def test_grug_adamh_mask_routes_correctly():
    config = GrugAdamHConfig(learning_rate=0.01, adam_lr=1e-3)
    params = _make_fake_grug_params()
    mask = config.create_mask(params)

    assert mask["embed"] == "adam", "embeddings should use adam"
    assert mask["router_weight"] == "adam", "router weights should use adam"
    assert mask["layers"]["norm_weight"] == "adam", "1D norm params should use adam"
    assert mask["layers"]["attn_w_q"] == "adamh", "attention weight matrices should use adamh"
    assert mask["layers"]["mlp_w1"] == "adamh", "MLP weight matrices should use adamh"
    assert mask["lm_head"] == "adamh", "lm_head weight matrix should use adamh"


def test_grug_adamh_builds_optimizer():
    config = GrugAdamHConfig(
        learning_rate=0.01,
        adam_lr=1e-3,
        lr_schedule="linear",
        warmup=0.1,
        max_grad_norm=1.0,
    )
    params = _make_fake_grug_params()
    opt = config.build(num_train_steps=100)
    state = opt.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    updates, new_state = opt.update(grads, state, params)
    # Verify updates have same structure as params
    flat_updates = jax.tree.leaves(updates)
    flat_params = jax.tree.leaves(params)
    assert len(flat_updates) == len(flat_params)
