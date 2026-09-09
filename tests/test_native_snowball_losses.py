# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from marin.testing.inference.snowball_checkpoint import apply_pending_qb_betas

from experiments.evaluation.native_snowball_losses import native_token_losses, prepare_native_model
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


@pytest.mark.parametrize("stacked", [False, True])
def test_native_document_nll_matches_logits_and_excludes_router_regularization(stacked):
    config = GrugModelConfig(
        vocab_size=24,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_token=2,
        max_seq_len=8,
        sliding_window=4,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=stacked,
        moe_implementation="ring",
        attention_implementation="reference",
        router_z_loss_coef=100.0,
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.key(7))
        betas = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        model = eqx.filter_jit(apply_pending_qb_betas)(model, betas)
        biases = (
            model.stacked_blocks.stacked.mlp.router_bias
            if stacked
            else jnp.stack([block.mlp.router_bias for block in model.blocks])
        )
        np.testing.assert_allclose(biases, [[1.5, 0.5, -0.5, -1.5]] * 2)
        sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        tokens = jax.device_put(jnp.arange(8, dtype=jnp.int32)[None, :], sharding)
        weights = jax.device_put(jnp.array([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.float32), sharding)
        batch = GrugLmExample(tokens=tokens, loss_weight=weights, attn_mask=AttentionMask.causal())
        actual = eqx.filter_jit(native_token_losses)(model, batch)
        logits = eqx.filter_jit(lambda model: model.logits(tokens))(model)
        labels = np.roll(np.asarray(tokens), -1, axis=1)
        expected = -np.take_along_axis(np.asarray(jax.nn.log_softmax(logits)), labels[..., None], axis=-1)[
            ..., 0
        ] * np.asarray(weights)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_bfloat16_router_preparation_matches_export_order():
    config = GrugModelConfig(
        vocab_size=24,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_token=2,
        use_array_stacked_blocks=True,
        moe_implementation="ring",
        disable_pko=True,
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.key(8))
        betas = jnp.array([[0.015, 0.124, 0.267, 0.401], [0.318, 0.702, 0.505, 0.201]])
        expected_biases = np.asarray((-betas + betas.mean(axis=-1, keepdims=True)).astype(jnp.bfloat16))
        expected_weights = [np.asarray(x.astype(jnp.bfloat16)) for x in jax.tree.leaves(model)]
        policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=float32")
        prepared = prepare_native_model(model, betas, policy)
        np.testing.assert_array_equal(prepared.stacked_blocks.stacked.mlp.router_bias, expected_biases)
        # The preparation must change only router bias and parameter precision.
        without_bias = eqx.tree_at(
            lambda tree: tree.stacked_blocks.stacked.mlp.router_bias,
            prepared,
            jnp.zeros_like(prepared.stacked_blocks.stacked.mlp.router_bias),
        )
        for actual, expected in zip(jax.tree.leaves(without_bias), expected_weights, strict=True):
            np.testing.assert_array_equal(actual, expected)
