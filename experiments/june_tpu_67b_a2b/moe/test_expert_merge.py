# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn import ArrayStacked
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.expert_merge import convert_one_expert_pair, permute_pending_qb_beta
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


def _tiny_config(mapping: tuple[int, ...]) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=len(mapping),
        expert_bank_for_layer=mapping,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="ring",
        disable_pko=True,
        use_array_stacked_blocks=True,
    )


def test_router_and_qb_permutation_preserve_routing_under_renamed_ids():
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int32)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config((0, 1)), key=jax.random.key(0))
        assert model.stacked_blocks is not None
        old_router = model.stacked_blocks.get_layer(1).mlp
        inputs = jnp.linspace(-1.0, 1.0, 192, dtype=jnp.float32).reshape(4, 3, 16)
        old_routing = old_router.route(inputs)

        converted = convert_one_expert_pair(
            model,
            representative_layer=0,
            source_layer=1,
            source_to_shared=permutation,
        )
        assert converted.stacked_blocks is not None
        new_routing = converted.stacked_blocks.get_layer(1).mlp.route(inputs)

        pending_qb = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        converted_qb = permute_pending_qb_beta(
            pending_qb,
            layer_index=1,
            source_to_shared=permutation,
        )

    np.testing.assert_array_equal(new_routing.selected_experts, permutation[np.asarray(old_routing.selected_experts)])
    np.testing.assert_array_equal(new_routing.combine_weights, old_routing.combine_weights)
    np.testing.assert_array_equal(converted_qb[0], pending_qb[0])
    np.testing.assert_array_equal(converted_qb[1, permutation], pending_qb[1])


def test_identity_merge_is_exact_when_source_and_representative_banks_match():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config((0, 1)), key=jax.random.key(1))
        assert isinstance(model.expert_banks, ArrayStacked)
        identical_banks = jax.tree.map(
            lambda value: value.at[1].set(value[0]),
            model.expert_banks.stacked,
        )
        model = eqx.tree_at(lambda current: current.expert_banks.stacked, model, identical_banks)
        tokens = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (4, 8))
        original_logits = model.logits(tokens)

        converted = convert_one_expert_pair(
            model,
            representative_layer=0,
            source_layer=1,
            source_to_shared=np.arange(4, dtype=np.int32),
        )
        converted_logits = converted.logits(tokens)

    assert converted.config.resolved_expert_bank_for_layer == (0, 0)
    assert isinstance(converted.expert_banks, ArrayStacked)
    assert converted.expert_banks.num_layers == 1
    np.testing.assert_array_equal(converted_logits, original_logits)
