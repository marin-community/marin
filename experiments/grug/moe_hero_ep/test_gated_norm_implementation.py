# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, set_mesh

from experiments.grug.moe_hero_ep import model


def _tiny_config():
    return model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(32),
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )


def test_fused_gated_norm_matches_the_reference_logits_and_parameter_gradients():
    """Every RMSNorm -> GatedNorm pair (embedding, both block sites, final) takes the fused op."""
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    base = _tiny_config()
    tokens = jax.random.randint(jax.random.key(3), (2, 8), 0, base.vocab_size)
    cotangent = jax.random.normal(jax.random.key(4), (2, 8, base.vocab_size))

    def loss_and_grads(cfg):
        transformer = model.Transformer.init(cfg, key=jax.random.key(5))
        loss = lambda t: jnp.sum(t.logits(tokens) * cotangent)  # noqa: E731
        return eqx.filter_jit(eqx.filter_value_and_grad(loss))(transformer)

    with set_mesh(mesh), jax.default_matmul_precision("highest"):
        want_loss, want_grads = loss_and_grads(base)
        fused = dataclasses.replace(base, gated_norm_implementation=model.GatedNormImplementation.FUSED)
        got_loss, got_grads = loss_and_grads(fused)

    np.testing.assert_allclose(float(got_loss), float(want_loss), rtol=1e-5)
    want_leaves = jax.tree.leaves(eqx.filter(want_grads, eqx.is_inexact_array))
    got_leaves = jax.tree.leaves(eqx.filter(got_grads, eqx.is_inexact_array))
    # Some leaves' gradients are cancellation noise near zero in both, so the absolute tolerance is
    # set by the largest gradient anywhere in the model.
    scale = max(float(np.max(np.abs(np.asarray(leaf)))) for leaf in want_leaves)
    for got, want in zip(got_leaves, want_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-5 * scale)
