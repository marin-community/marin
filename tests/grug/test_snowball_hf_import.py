# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballTransformer

from experiments.grug.moe_hero_ep.model import GrugModelConfig as TrainingConfig
from experiments.grug.moe_hero_ep.train import _apply_qb_betas
from experiments.grug_sft.snowball_hf_import import import_snowball_hf_weights


def _snowball_config() -> SnowballConfig:
    return SnowballConfig(
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
        moe_implementation="ring",
    )


def _training_config() -> TrainingConfig:
    source = _snowball_config()
    return TrainingConfig(
        vocab_size=source.vocab_size,
        hidden_dim=source.hidden_dim,
        intermediate_dim=source.intermediate_dim,
        shared_expert_intermediate_dim=source.shared_expert_intermediate_dim,
        num_experts=source.num_experts,
        num_experts_per_token=source.num_experts_per_token,
        num_layers=source.num_layers,
        num_heads=source.num_heads,
        num_kv_heads=source.num_kv_heads,
        head_dim=source.head_dim,
        max_seq_len=source.max_seq_len,
        sliding_window=source.sliding_window,
        qk_mult=source.qk_mult,
        num_shared_experts=1,
        moe_implementation="ring",
    )


def test_hf_import_preserves_effective_weights_and_greedy_tokens_in_stacked_trainer():
    with set_mesh(compact_grug_mesh(expert_axis_size=1)):
        exported = SnowballTransformer.init(_snowball_config(), key=jax.random.key(7))
        biases = jnp.remainder(
            jnp.arange(exported.config.num_layers * exported.config.num_experts, dtype=jnp.float32).reshape(
                exported.config.num_layers, exported.config.num_experts
            ),
            5,
        )
        biases = biases - jnp.mean(biases, axis=-1, keepdims=True)
        blocks = tuple(
            eqx.tree_at(lambda block: block.mlp.router_bias, block, biases[index])
            for index, block in enumerate(exported.blocks)
        )
        exported = eqx.tree_at(lambda model: model.blocks, exported, blocks)

        imported, pending_qb_betas = import_snowball_hf_weights(
            _snowball_config(),
            _training_config(),
            exported.to_state_dict(),
            key=jax.random.key(11),
        )
        effective = _apply_qb_betas(imported, pending_qb_betas)

        imported_blocks = tuple(effective.stacked_blocks.unstacked())
        assert np.array_equal(np.asarray(pending_qb_betas), np.asarray(-biases))
        for expected, actual in zip(exported.blocks, imported_blocks, strict=True):
            expected_leaves = jax.tree.leaves(expected)
            actual_leaves = jax.tree.leaves(actual)
            assert len(expected_leaves) == len(actual_leaves)
            for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves, strict=True):
                assert np.array_equal(np.asarray(expected_leaf), np.asarray(actual_leaf))

        np.testing.assert_array_equal(effective.token_embed, exported.token_embed)
        np.testing.assert_array_equal(effective.output_proj, exported.output_proj)

        tokens = (jnp.arange(10, dtype=jnp.int32).reshape(1, 10) * 7 + 3) % exported.config.vocab_size
        expected_logits = np.asarray(jax.jit(lambda model, value: model(value) @ model.output_proj)(exported, tokens))
        actual_logits = np.asarray(jax.jit(lambda model, value: model.logits(value))(effective, tokens))

    np.testing.assert_array_equal(np.argmax(actual_logits, axis=-1), np.argmax(expected_logits, axis=-1))
