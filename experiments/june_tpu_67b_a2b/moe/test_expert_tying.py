# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.nn import ArrayStacked
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.jax_utils import leaf_key_paths

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer, _cross_loop_agreement
from experiments.june_tpu_67b_a2b.moe.train import _apply_qb_betas


def _tiny_config(*, mapping: tuple[int, ...], stacked: bool = False) -> GrugModelConfig:
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
        use_array_stacked_blocks=stacked,
    )


def _array_leaves(tree) -> list[jax.Array]:
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))


def test_untied_default_matches_explicit_untied_topology():
    explicit_config = _tiny_config(mapping=(0, 1, 2))
    implicit_config = dataclasses.replace(explicit_config, expert_bank_for_layer=None)

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


@pytest.mark.parametrize(
    "mapping",
    [
        (0, 1),
        (0, -1, 1),
        (0, 2, 2),
    ],
)
def test_expert_bank_mapping_rejects_invalid_topologies(mapping: tuple[int, ...]):
    config = _tiny_config(mapping=(0, 1, 2))
    with pytest.raises(ValueError):
        dataclasses.replace(config, expert_bank_for_layer=mapping)


@pytest.mark.parametrize("stacked", [False, True])
def test_expert_bank_mapping_stores_each_bank_once(stacked: bool):
    mapping = (0, 1, 1, 2)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=mapping, stacked=stacked), key=jax.random.key(1))

    if stacked:
        assert isinstance(model.expert_banks, ArrayStacked)
        assert model.expert_banks.num_layers == 3
        assert all(leaf.shape[0] == 3 for leaf in _array_leaves(model.expert_banks.stacked))
    else:
        assert isinstance(model.expert_banks, tuple)
        assert len(model.expert_banks) == 3
        assert len(_array_leaves(model.expert_banks)) == 9

    block_tree = model.stacked_blocks if stacked else model.blocks
    block_paths = jax.tree_util.tree_leaves(leaf_key_paths(block_tree))
    assert not any("expert_mlp" in str(path) for path in block_paths)


def test_shared_bank_gradient_equals_sum_of_layer_use_site_gradients():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=(0, 0)), key=jax.random.key(2))
        assert model.blocks is not None
        assert isinstance(model.expert_banks, tuple)
        bank = model.expert_banks[0]
        inputs = (
            jnp.linspace(-1.0, 1.0, 32, dtype=jnp.float32).reshape(1, 2, 16),
            jnp.linspace(1.0, -0.5, 32, dtype=jnp.float32).reshape(1, 2, 16),
        )

        def layer_loss(expert_bank, layer_index):
            routed, _ = model.blocks[layer_index].mlp(inputs[layer_index], expert_bank)
            return jnp.sum(jnp.square(routed))

        separate_grads = tuple(jax.grad(layer_loss)(bank, layer_index) for layer_index in range(2))
        joint_grad = jax.grad(lambda expert_bank: layer_loss(expert_bank, 0) + layer_loss(expert_bank, 1))(bank)
        expected_grad = jax.tree.map(lambda first, second: first + second, *separate_grads)

    assert all(float(jnp.linalg.norm(leaf)) > 0 for grad in separate_grads for leaf in _array_leaves(grad))
    for actual, expected in zip(_array_leaves(joint_grad), _array_leaves(expected_grad), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_moe_trace_exposes_routed_inputs_without_changing_dispatch():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=(0,)), key=jax.random.key(7))
        assert model.blocks is not None
        assert isinstance(model.expert_banks, tuple)
        mlp = model.blocks[0].mlp
        bank = model.expert_banks[0]
        inputs = jnp.linspace(-1.0, 1.0, 48, dtype=jnp.float32).reshape(1, 3, 16)

        trace = mlp.forward_with_trace(inputs, bank)
        routed, router_stats = mlp(inputs, bank)

    np.testing.assert_array_equal(trace.routed_output, routed)
    assert trace.routing.x_flat.shape == (3, 16)
    assert trace.routing.selected_experts.shape == (3, 2)
    np.testing.assert_allclose(jnp.sum(trace.routing.combine_weights, axis=-1), 2.5, rtol=1e-6, atol=1e-6)
    assert router_stats.keys() == trace.router_stats.keys()
    for key in router_stats:
        np.testing.assert_array_equal(router_stats[key], trace.router_stats[key])


def test_tied_mapping_matches_under_array_stacked_execution():
    mapping = (0, 1, 1, 2)
    unstacked_config = _tiny_config(mapping=mapping)
    stacked_config = dataclasses.replace(unstacked_config, use_array_stacked_blocks=True)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        unstacked = Transformer.init(unstacked_config, key=jax.random.key(3))
        stacked = Transformer.init(stacked_config, key=jax.random.key(3))
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        unstacked_hidden, unstacked_metrics = unstacked(tokens)
        stacked_hidden, stacked_metrics = stacked(tokens)

    np.testing.assert_allclose(stacked_hidden, unstacked_hidden, rtol=1e-5, atol=1e-6)
    assert stacked_metrics.keys() == unstacked_metrics.keys()
    for key in stacked_metrics:
        np.testing.assert_allclose(stacked_metrics[key], unstacked_metrics[key], rtol=1e-5, atol=1e-6)


def test_cross_loop_metrics_match_known_top1_and_topk_overlap():
    selected_experts = jnp.array(
        [
            [[0, 1], [2, 3]],
            [[0, 2], [3, 1]],
        ],
        dtype=jnp.int32,
    )

    agreement = _cross_loop_agreement(selected_experts, bank_for_layer=(0, 0))

    assert agreement.top1 == 0.5
    assert agreement.topk_set_overlap == 0.5


def test_stacked_shared_bank_gradient_equals_sum_of_layer_use_site_gradients():
    tied_config = _tiny_config(mapping=(0, 0), stacked=True)
    untied_config = dataclasses.replace(tied_config, expert_bank_for_layer=(0, 1))
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        tied_model = Transformer.init(tied_config, key=jax.random.key(8))
        untied_model = Transformer.init(untied_config, key=jax.random.key(8))
        assert isinstance(tied_model.expert_banks, ArrayStacked)
        assert isinstance(untied_model.expert_banks, ArrayStacked)

        duplicated_bank_values = jax.tree.map(
            lambda value: jnp.repeat(value, 2, axis=0),
            tied_model.expert_banks.stacked,
        )
        duplicated_banks = dataclasses.replace(
            untied_model.expert_banks,
            stacked=duplicated_bank_values,
        )
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)

        def tied_loss(expert_banks):
            current = eqx.tree_at(lambda model: model.expert_banks, tied_model, expert_banks)
            return jnp.sum(jnp.square(current.logits(tokens)))

        def untied_loss(expert_banks):
            current = eqx.tree_at(lambda model: model.expert_banks, untied_model, expert_banks)
            return jnp.sum(jnp.square(current.logits(tokens)))

        tied_logits = tied_model.logits(tokens)
        untied_logits = eqx.tree_at(
            lambda model: model.expert_banks,
            untied_model,
            duplicated_banks,
        ).logits(tokens)
        joint_grad = jax.grad(tied_loss)(tied_model.expert_banks)
        separate_grads = jax.grad(untied_loss)(duplicated_banks)

    np.testing.assert_array_equal(tied_logits, untied_logits)
    for actual, per_layer in zip(
        _array_leaves(joint_grad.stacked),
        _array_leaves(separate_grads.stacked),
        strict=True,
    ):
        assert all(float(jnp.linalg.norm(layer_grad)) > 0 for layer_grad in per_layer)
        np.testing.assert_allclose(actual[0], jnp.sum(per_layer, axis=0), rtol=1e-5, atol=1e-6)


def test_stacked_qb_updates_and_router_statistics_remain_layer_specific():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=(0, 0), stacked=True), key=jax.random.key(9))
        assert isinstance(model.expert_banks, ArrayStacked)
        assert model.stacked_blocks is not None
        original_bank = _array_leaves(model.expert_banks)
        qb_betas = jnp.array(
            [
                [-100.0, 0.0, 0.0, 0.0],
                [0.0, -100.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        updated = _apply_qb_betas(model, qb_betas)
        assert updated.stacked_blocks is not None
        first_router = updated.stacked_blocks.get_layer(0).mlp
        second_router = updated.stacked_blocks.get_layer(1).mlp
        x = jnp.ones((1, 4, 16), dtype=jnp.float32)
        bank = updated.expert_banks.get_layer(0)
        _, first_stats = first_router(x, bank)
        _, second_stats = second_router(x, bank)
        _, model_stats = updated(jnp.arange(8, dtype=jnp.int32).reshape(1, 8))

    expected_biases = -qb_betas + jnp.mean(qb_betas, axis=-1, keepdims=True)
    np.testing.assert_array_equal(first_router.router_bias, expected_biases[0])
    np.testing.assert_array_equal(second_router.router_bias, expected_biases[1])
    assert int(jnp.argmax(first_stats["routing_counts"])) == 0
    assert int(jnp.argmax(second_stats["routing_counts"])) == 1
    assert not np.array_equal(first_stats["routing_counts"], second_stats["routing_counts"])
    assert model_stats["routing_counts_per_layer"].shape == (2, 4)
    assert model_stats["qb_beta_per_layer"].shape == (2, 4)
    for before, after in zip(original_bank, _array_leaves(updated.expert_banks), strict=True):
        np.testing.assert_array_equal(before, after)


def test_stacked_shared_dense_mlps_remain_per_layer_when_experts_are_tied():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(_tiny_config(mapping=(0, 0), stacked=True), key=jax.random.key(10))
        assert model.stacked_blocks is not None
        stacked_shared = model.stacked_blocks.stacked.shared
        assert stacked_shared is not None
        x = jnp.linspace(-1.0, 1.0, 32, dtype=jnp.float32).reshape(1, 2, 16)
        block_one_before = model.stacked_blocks.get_layer(1).shared
        assert block_one_before is not None
        block_one_before_output = block_one_before(x)
        changed_shared = jax.tree.map(lambda value: value.at[0].set(jnp.zeros_like(value[0])), stacked_shared)
        changed = eqx.tree_at(lambda current: current.stacked_blocks.stacked.shared, model, changed_shared)
        assert changed.stacked_blocks is not None
        changed_block_zero = changed.stacked_blocks.get_layer(0).shared
        changed_block_one = changed.stacked_blocks.get_layer(1).shared
        assert changed_block_zero is not None
        assert changed_block_one is not None

        block_zero_after = changed_block_zero(x)
        block_one_after = changed_block_one(x)

    np.testing.assert_array_equal(block_zero_after, jnp.zeros_like(block_zero_after))
    np.testing.assert_array_equal(block_one_after, block_one_before_output)


def test_array_stacked_tied_forward_runs_on_data_and_expert_mesh_axes():
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from levanter.grug.sharding import compact_grug_mesh
        from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

        assert jax.device_count() == 4
        config = GrugModelConfig(
            vocab_size=32,
            hidden_dim=16,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=2,
            expert_bank_for_layer=(0, 0),
            num_heads=2,
            num_kv_heads=1,
            max_seq_len=8,
            sliding_window=4,
            moe_implementation="ring",
            disable_pko=True,
            use_array_stacked_blocks=True,
        )
        mesh = compact_grug_mesh(expert_axis_size=2, replica_axis_size=1, model_axis_size=1)
        with jax.set_mesh(mesh):
            model = Transformer.init(config, key=jax.random.key(0))
            tokens = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (4, 8))
            hidden, metrics = jax.jit(lambda current, token_ids: current(token_ids))(model, tokens)
            jax.block_until_ready(hidden)

        assert hidden.shape == (4, 8, 16)
        assert metrics["routing_counts_per_layer"].shape == (2, 4)
        assert metrics["activation_norm_per_layer"].shape == (2,)
        assert np.isfinite(np.asarray(metrics["top1_cross_loop_agreement"]))
        """
    )
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
