# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from haliax.nn import ArrayStacked
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.june_tpu_67b_a2b.moe.expert_merge import (
    convert_one_expert_pair,
    forward_with_moe_traces,
    permute_pending_qb_beta,
)
from experiments.june_tpu_67b_a2b.moe.merge_checkpoint import (
    OnePairMergeCheckpointSpec,
    convert_june_state_for_one_pair_merge,
)
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainState


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


def _test_mesh():
    return compact_grug_mesh(expert_axis_size=jax.device_count())


def _data_test_mesh():
    return compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)


def test_router_and_qb_permutation_preserve_routing_under_renamed_ids():
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int32)
    with jax.set_mesh(_test_mesh()):
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
    with jax.set_mesh(_test_mesh()):
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


def test_stacked_trace_matches_ordinary_forward_and_samples_before_returning():
    with jax.set_mesh(_data_test_mesh()):
        model = Transformer.init(_tiny_config((0, 1)), key=jax.random.key(2))
        tokens = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (4, 8))
        expected_hidden, _ = model(tokens)

        actual_hidden, traces, capacity_overflow = forward_with_moe_traces(
            model,
            tokens,
            target_layers=(0, 1),
            token_indices=jnp.asarray([0, 7, 16, 31], dtype=jnp.int32),
        )

    np.testing.assert_allclose(actual_hidden, expected_hidden, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(capacity_overflow, np.zeros((2,), dtype=np.float32))
    assert traces.keys() == {0, 1}
    for trace in traces.values():
        assert trace.mlp_input.shape == (4, 16)
        assert trace.selected_experts.shape == (4, 2)
        assert trace.combine_weights.shape == (4, 2)
        assert trace.routed_output.shape == (4, 16)


def test_state_conversion_resets_optimizer_and_records_topology():
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int32)
    with jax.set_mesh(_test_mesh()):
        model = Transformer.init(_tiny_config((0, 1)), key=jax.random.key(3))
        pending_qb = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        state = GrugTrainState(
            step=jnp.asarray(105_149, dtype=jnp.int32),
            params=model,
            opt_state={"old": jnp.asarray(1.0)},
            ema_params=None,
            pending_qb_betas=pending_qb,
        )
        optimizer = optax.adam(1e-4)
        spec = OnePairMergeCheckpointSpec(
            representative_layer=0,
            source_layer=1,
            source_to_shared=tuple(int(index) for index in permutation),
            assignment_mode=AssignmentMode.NATIVE,
            source_checkpoint="gs://marin-us-central2/example/checkpoints/step-105149",
        )

        converted = convert_june_state_for_one_pair_merge(
            state,
            spec=spec,
            init_optimizer_state=optimizer.init,
        )

    assert int(converted.state.step) == 0
    assert converted.state.params.config.resolved_expert_bank_for_layer == (0, 0)
    assert converted.manifest.source_topology == (0, 1)
    assert converted.manifest.target_topology == (0, 0)
    assert converted.manifest.source_step == 105_149
    np.testing.assert_array_equal(converted.state.pending_qb_betas[0], pending_qb[0])
    np.testing.assert_array_equal(converted.state.pending_qb_betas[1, permutation], pending_qb[1])
    assert not isinstance(converted.state.opt_state, dict)
