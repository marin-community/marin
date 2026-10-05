# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SFT export must retain the effective QB routes and model behavior."""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug_sft.export_science_sft_for_eval import serving_model, snowball_config
from experiments.grug_sft.head_only_train import _add_qb_betas_to_residual, _apply_qb_betas
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


def test_export_preserves_effective_router_bias_and_logits():
    training = GrugModelConfig(
        vocab_size=24,
        hidden_dim=12,
        intermediate_dim=16,
        shared_expert_intermediate_dim=12,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=5,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        qk_mult=1.37,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
        moe_implementation="ring",
    )
    pending = jnp.arange(training.num_layers * training.num_experts, dtype=jnp.float32).reshape(
        training.num_layers, training.num_experts
    )
    tokens = jnp.arange(10, dtype=jnp.int32).reshape(1, 10) % training.vocab_size
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        source = Transformer.init(training, key=jax.random.key(7))
        exported = serving_model(source, pending, snowball_config(training))
        expected = np.asarray(jax.jit(lambda model, ids: model.logits(ids))(_apply_qb_betas(source, pending), tokens))
        actual = np.asarray(
            hax.named_jit(lambda model, ids: model(ids))(
                exported,
                hax.named(tokens[0], (Axis("position", tokens.shape[1]),)),
            ).array
        )
    np.testing.assert_allclose(actual, expected[0], rtol=1e-5, atol=1e-5)
    routes = exported.to_state_dict()
    for layer in range(training.num_layers):
        bias = np.asarray(routes[f"model.layers.{layer}.mlp.router.bias"])
        np.testing.assert_allclose(bias, -np.asarray(pending[layer] - pending[layer].mean()))


def test_export_adds_learned_router_bias_residual_to_qb_bias():
    training = GrugModelConfig(
        vocab_size=24,
        hidden_dim=12,
        intermediate_dim=16,
        shared_expert_intermediate_dim=12,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=5,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=4,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
        moe_implementation="ring",
        trainable_router_bias=True,
    )
    pending = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        source = Transformer.init(training, key=jax.random.key(8))
        residual = jnp.full((5, 4), 0.125, dtype=jnp.float32)
        source = eqx.tree_at(lambda model: model.stacked_blocks.stacked.mlp.router_bias, source, residual)
        exported = serving_model(source, pending, snowball_config(training), train_router_bias_residual=True)
        expected = _add_qb_betas_to_residual(source, pending)
    routes = exported.to_state_dict()
    for layer in range(training.num_layers):
        actual_bias = np.asarray(routes[f"model.layers.{layer}.mlp.router.bias"])
        expected_bias = np.asarray(expected.stacked_blocks.stacked.mlp.router_bias[layer])
        np.testing.assert_allclose(actual_bias, expected_bias)
