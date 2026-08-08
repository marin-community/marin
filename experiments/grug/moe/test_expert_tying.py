# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.model import (
    GrugModelConfig,
    GrugMoeHfConfig,
    Transformer,
    _cross_loop_agreement,
    grugmoe_inference_state_dict,
)
from experiments.grug.moe.train import apply_qb_betas


def _tiny_config(
    *,
    mapping: tuple[int, ...],
    shared_intermediate_dim: int = 16,
    adapter_ranks: tuple[int, ...] | None = None,
) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=shared_intermediate_dim,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=len(mapping),
        expert_bank_for_layer=mapping,
        expert_adapter_rank_for_layer=adapter_ranks,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="ring",
    )


def _array_leaves(tree) -> list[jax.Array]:
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))


def _parameter_count(tree) -> int:
    return sum(leaf.size for leaf in _array_leaves(tree))


def test_untied_default_matches_explicit_untied_topology():
    implicit_config = _tiny_config(mapping=(0, 1, 2))
    implicit_config = dataclasses.replace(implicit_config, expert_bank_for_layer=None)
    explicit_config = _tiny_config(mapping=(0, 1, 2))

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        implicit = Transformer.init(implicit_config, key=jax.random.key(0))
        explicit = Transformer.init(explicit_config, key=jax.random.key(0))
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        implicit_logits = implicit.logits(tokens)
        explicit_logits = explicit.logits(tokens)

    assert implicit_config.resolved_expert_bank_for_layer == (0, 1, 2)
    for implicit_leaf, explicit_leaf in zip(_array_leaves(implicit), _array_leaves(explicit), strict=True):
        np.testing.assert_array_equal(implicit_leaf, explicit_leaf)
    np.testing.assert_array_equal(implicit_logits, explicit_logits)


def test_explicit_rank_zero_adapters_match_disabled_adapters():
    disabled_config = _tiny_config(mapping=(0, 0), adapter_ranks=None)
    rank_zero_config = _tiny_config(mapping=(0, 0), adapter_ranks=(0, 0))

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        disabled = Transformer.init(disabled_config, key=jax.random.key(10))
        rank_zero = Transformer.init(rank_zero_config, key=jax.random.key(10))
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        disabled_logits = disabled.logits(tokens)
        rank_zero_logits = rank_zero.logits(tokens)

    assert all(block.routed_expert_adapter is None for block in disabled.blocks)
    assert all(block.routed_expert_adapter is None for block in rank_zero.blocks)
    for disabled_leaf, rank_zero_leaf in zip(_array_leaves(disabled), _array_leaves(rank_zero), strict=True):
        np.testing.assert_array_equal(disabled_leaf, rank_zero_leaf)
    np.testing.assert_array_equal(disabled_logits, rank_zero_logits)


def test_zero_initialized_adapter_preserves_function_and_has_live_b_gradients():
    disabled_config = _tiny_config(mapping=(0, 0))
    adapter_config = _tiny_config(mapping=(0, 0), adapter_ranks=(0, 4))

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        disabled = Transformer.init(disabled_config, key=jax.random.key(11))
        adapted = Transformer.init(adapter_config, key=jax.random.key(11))
        adapter = adapted.blocks[1].routed_expert_adapter
        assert adapter is not None
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)

        def adapter_loss(current_adapter):
            current = eqx.tree_at(
                lambda model: model.blocks[1].routed_expert_adapter,
                adapted,
                current_adapter,
            )
            return jnp.sum(jnp.square(current.logits(tokens)))

        adapter_grad = jax.grad(adapter_loss)(adapter)
        disabled_logits = disabled.logits(tokens)
        adapted_logits = adapted.logits(tokens)

    np.testing.assert_array_equal(adapted_logits, disabled_logits)
    assert float(jnp.linalg.norm(adapter.input_a)) > 0
    assert float(jnp.linalg.norm(adapter.output_a)) > 0
    np.testing.assert_array_equal(adapter.input_b, jnp.zeros_like(adapter.input_b))
    np.testing.assert_array_equal(adapter.output_b, jnp.zeros_like(adapter.output_b))
    assert float(jnp.linalg.norm(adapter_grad.input_b)) > 0
    assert float(jnp.linalg.norm(adapter_grad.output_b)) > 0

    bf16_input = jnp.linspace(-1.0, 1.0, 32, dtype=jnp.bfloat16).reshape(2, 16)
    adapted_input = adapter.adapt_input(bf16_input)
    adapted_output = adapter.adapt_output(bf16_input)
    assert adapted_input.dtype == jnp.bfloat16
    assert adapted_output.dtype == jnp.bfloat16
    np.testing.assert_array_equal(adapted_input, bf16_input)
    np.testing.assert_array_equal(adapted_output, bf16_input)


def test_nonzero_adapter_cannot_change_routes_or_qb_statistics():
    adapter_config = _tiny_config(mapping=(0, 0), adapter_ranks=(0, 4))
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(adapter_config, key=jax.random.key(15))
        block = model.blocks[1]
        adapter = block.routed_expert_adapter
        assert adapter is not None
        nonzero_adapter = dataclasses.replace(
            adapter,
            input_b=jnp.ones_like(adapter.input_b) * 0.1,
            output_b=jnp.ones_like(adapter.output_b) * 0.1,
        )
        x = jnp.linspace(-1.0, 1.0, 64, dtype=jnp.float32).reshape(1, 4, 16)
        bank = model.expert_banks[block.expert_bank_index]
        without_adapter = block.mlp.forward_with_trace(x, bank)
        with_adapter = block.mlp.forward_with_trace(x, bank, nonzero_adapter)

    np.testing.assert_array_equal(with_adapter.routing.selected_experts, without_adapter.routing.selected_experts)
    np.testing.assert_array_equal(with_adapter.routing.combine_weights, without_adapter.routing.combine_weights)
    np.testing.assert_array_equal(with_adapter.router_stats["qb_beta"], without_adapter.router_stats["qb_beta"])
    assert not np.array_equal(with_adapter.routed_output, without_adapter.routed_output)


def test_inference_state_dict_carries_layer_adapter_tensors():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(
            _tiny_config(mapping=(0, 0), adapter_ranks=(0, 4)),
            key=jax.random.key(16),
        )
        state_dict = grugmoe_inference_state_dict(model)

    adapter_prefix = "model.layers.1.mlp.expert_adapter"
    assert state_dict[f"{adapter_prefix}.input_a.weight"].shape == (4, 16)
    assert state_dict[f"{adapter_prefix}.input_b.weight"].shape == (16, 4)
    assert state_dict[f"{adapter_prefix}.output_a.weight"].shape == (4, 16)
    assert state_dict[f"{adapter_prefix}.output_b.weight"].shape == (16, 4)


def test_layer_adapter_rank_and_parameter_saving_are_explicit():
    rank = 4
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        untied = Transformer.init(_tiny_config(mapping=(0, 1)), key=jax.random.key(12))
        adapted_tied = Transformer.init(
            _tiny_config(mapping=(0, 0), adapter_ranks=(0, rank)),
            key=jax.random.key(12),
        )

    adapter = adapted_tied.blocks[1].routed_expert_adapter
    assert adapter is not None
    assert adapted_tied.config.resolved_expert_adapter_rank_for_layer == (0, rank)
    assert sum(block.routed_expert_adapter is not None for block in adapted_tied.blocks) == 1
    assert adapter.input_a.shape == (adapted_tied.config.hidden_dim, rank)
    assert adapter.input_b.shape == (rank, adapted_tied.config.hidden_dim)
    assert adapter.output_a.shape == (adapted_tied.config.hidden_dim, rank)
    assert adapter.output_b.shape == (rank, adapted_tied.config.hidden_dim)

    bank_parameters = (
        3 * adapted_tied.config.num_experts * adapted_tied.config.hidden_dim * adapted_tied.config.intermediate_dim
    )
    adapter_parameters = 4 * adapted_tied.config.hidden_dim * rank
    assert _parameter_count(untied) - _parameter_count(adapted_tied) == bank_parameters - adapter_parameters


def test_expert_bank_mapping_stores_each_bank_once():
    mapping = (0, 1, 1, 2)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=mapping), key=jax.random.key(1))

    paths = leaf_key_paths(model)
    block_paths = [str(path) for path in jax.tree_util.tree_leaves(paths.blocks)]

    assert tuple(block.expert_bank_index for block in model.blocks) == mapping
    assert len(model.expert_banks) == len(set(mapping))
    assert not any("expert_banks" in path or "expert_mlp" in path for path in block_paths)


def test_shared_bank_gradient_equals_sum_of_layer_use_site_gradients():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        key = jax.random.key(2)
        tied_model = Transformer.init(_tiny_config(mapping=(0, 0)), key=key)
        untied_model = Transformer.init(_tiny_config(mapping=(0, 1)), key=key)
        shared_bank = tied_model.expert_banks[0]
        untied_banks = (shared_bank, shared_bank)
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)

        def tied_loss(expert_bank):
            current = eqx.tree_at(lambda model: model.expert_banks, tied_model, (expert_bank,))
            return jnp.sum(jnp.square(current.logits(tokens)))

        def untied_loss(expert_banks):
            current = eqx.tree_at(lambda model: model.expert_banks, untied_model, expert_banks)
            return jnp.sum(jnp.square(current.logits(tokens)))

        tied_logits = tied_model.logits(tokens)
        untied_logits = eqx.tree_at(lambda model: model.expert_banks, untied_model, untied_banks).logits(tokens)
        joint_grad = jax.grad(tied_loss)(shared_bank)
        separate_grads = jax.grad(untied_loss)(untied_banks)
        expected_grad = jax.tree.map(lambda first, second: first + second, *separate_grads)

    np.testing.assert_array_equal(tied_logits, untied_logits)
    assert all(float(jnp.linalg.norm(leaf)) > 0 for grad in separate_grads for leaf in _array_leaves(grad))
    for actual, expected in zip(_array_leaves(joint_grad), _array_leaves(expected_grad), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_tied_checkpoint_round_trip_preserves_topology_and_values(tmp_path):
    saved_config = _tiny_config(mapping=(0, 1, 1, 2))
    hf_payload = json.loads(json.dumps(saved_config.to_hf_config(saved_config.vocab_size).to_dict()))
    restored_config = GrugModelConfig.from_hf_config(GrugMoeHfConfig(**hf_payload))

    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        saved_model = Transformer.init(saved_config, key=jax.random.key(3))
        template = Transformer.init(restored_config, key=jax.random.key(4))
        save_checkpoint(saved_model, step=0, checkpoint_path=tmp_path)
        restored_model = load_checkpoint(template, checkpoint_path=tmp_path, mesh=mesh)

    assert restored_config.resolved_expert_bank_for_layer == (0, 1, 1, 2)
    assert tuple(block.expert_bank_index for block in restored_model.blocks) == (0, 1, 1, 2)
    assert len(restored_model.expert_banks) == 3
    for actual, expected in zip(_array_leaves(restored_model), _array_leaves(saved_model), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_qb_updates_and_router_statistics_remain_layer_specific():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=(0, 0)), key=jax.random.key(5))
        original_bank = _array_leaves(model.expert_banks)
        qb_betas = jnp.array(
            [
                [-100.0, 0.0, 0.0, 0.0],
                [0.0, -100.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        updated = apply_qb_betas(model, qb_betas)
        x = jnp.ones((1, 4, 16), dtype=jnp.float32)
        _, first_stats = updated.blocks[0].mlp(x, updated.expert_banks[0])
        _, second_stats = updated.blocks[1].mlp(x, updated.expert_banks[0])
        _, model_stats = updated(jnp.arange(8, dtype=jnp.int32).reshape(1, 8))

    expected_biases = -qb_betas + jnp.mean(qb_betas, axis=-1, keepdims=True)
    np.testing.assert_array_equal(updated.blocks[0].mlp.router_bias, expected_biases[0])
    np.testing.assert_array_equal(updated.blocks[1].mlp.router_bias, expected_biases[1])
    assert int(jnp.argmax(first_stats["routing_counts"])) == 0
    assert int(jnp.argmax(second_stats["routing_counts"])) == 1
    assert not np.array_equal(first_stats["routing_counts"], second_stats["routing_counts"])
    assert model_stats["routing_counts_per_layer"].shape == (2, 4)
    assert model_stats["qb_beta_per_layer"].shape == (2, 4)
    for before, after in zip(original_bank, _array_leaves(updated.expert_banks), strict=True):
        np.testing.assert_array_equal(before, after)


def test_cross_loop_metrics_match_known_top1_and_topk_overlap():
    router_stats = [
        {"selected_experts": jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)},
        {"selected_experts": jnp.array([[0, 2], [3, 1]], dtype=jnp.int32)},
    ]

    agreement = _cross_loop_agreement(router_stats, bank_for_layer=(0, 0))

    assert agreement.top1 == 0.5
    assert agreement.topk_set_overlap == 0.5


def test_shared_dense_mlps_remain_per_layer_when_experts_are_tied():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=(0, 0)), key=jax.random.key(6))
        x = jnp.linspace(-1.0, 1.0, 32, dtype=jnp.float32).reshape(1, 2, 16)
        block_zero_shared = model.blocks[0].shared
        block_one_shared = model.blocks[1].shared
        assert block_zero_shared is not None
        assert block_one_shared is not None
        block_one_before = block_one_shared(x)
        zero_shared = jax.tree.map(jnp.zeros_like, block_zero_shared)
        changed = eqx.tree_at(lambda current: current.blocks[0].shared, model, zero_shared)
        changed_block_zero_shared = changed.blocks[0].shared
        changed_block_one_shared = changed.blocks[1].shared
        assert changed_block_zero_shared is not None
        assert changed_block_one_shared is not None
        block_zero_after = changed_block_zero_shared(x)
        block_one_after = changed_block_one_shared(x)

    np.testing.assert_array_equal(block_zero_after, jnp.zeros_like(block_zero_after))
    np.testing.assert_array_equal(block_one_after, block_one_before)
