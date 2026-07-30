# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.launch_augdk_nested_sft import SFTArm, sft_routing_model
from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe.train import GrugTrainState, extract_expert_range, init_weights_only_from_checkpoint


def _nested_model(*, count: int, layer_fraction: float) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=128,
        intermediate_dim=64,
        shared_expert_intermediate_dim=128,
        num_experts=256,
        num_experts_per_token=4,
        nested_expert_counts=(count,),
        nested_batch_fraction=0.25,
        nested_layer_fraction=layer_fraction,
        num_layers=4,
        num_heads=1,
        num_kv_heads=1,
    )


def test_sft_retains_layerwise_routing_for_matching_pretraining_arm() -> None:
    model = _nested_model(count=128, layer_fraction=0.25)

    configured = sft_routing_model(model, SFTArm.E128_LAYER25, "nested")

    assert configured is model
    assert configured.nested_batch_fraction == 0.25
    assert configured.nested_layer_fraction == 0.25


def test_sft_can_disable_training_restriction_without_changing_router_state_shape() -> None:
    model = _nested_model(count=16, layer_fraction=1.0)

    configured = sft_routing_model(model, SFTArm.E16_NAIVE25, "full")

    assert configured.nested_batch_fraction == 0.0
    assert configured.nested_expert_counts == (16,)


def test_sft_accepts_matched_standalone_e128_model() -> None:
    model = GrugModelConfig(
        vocab_size=128,
        hidden_dim=128,
        intermediate_dim=64,
        shared_expert_intermediate_dim=128,
        num_experts=128,
        num_experts_per_token=4,
        num_layers=4,
        num_heads=1,
        num_kv_heads=1,
    )

    configured = sft_routing_model(model, SFTArm.E128_STANDALONE, "nested")

    assert configured is model
    assert configured.num_experts == 128
    assert configured.nested_expert_counts == ()


def test_fixed_banks_can_be_retained_when_sft_uses_only_full_routing() -> None:
    model = GrugModelConfig(
        vocab_size=128,
        hidden_dim=128,
        intermediate_dim=64,
        shared_expert_intermediate_dim=128,
        num_experts=32,
        num_experts_per_token=4,
        nested_expert_counts=(16, 8),
        nested_batch_fraction=0.0,
        num_layers=2,
        num_heads=1,
        num_kv_heads=1,
    )

    assert model.router_balance_group_counts == (32, 16, 8)


def test_weights_only_initialization_retains_fresh_training_state() -> None:
    initial_params = object()
    loaded_params = object()
    fresh_optimizer = object()
    state = GrugTrainState(
        step=jnp.array(0),
        params=initial_params,  # type: ignore[arg-type]
        opt_state=fresh_optimizer,  # type: ignore[arg-type]
        ema_params=None,
        pending_qb_betas=jnp.zeros((2, 3, 32)),
    )

    def load(tree, checkpoint_path, *, mesh, allow_partial):
        assert tree["params"] is initial_params
        assert checkpoint_path == "s3://checkpoint/step-10"
        assert mesh is None
        assert allow_partial
        return {
            "params": loaded_params,
            "pending_qb_betas": jnp.ones((2, 3, 32)),
        }

    restored = init_weights_only_from_checkpoint(
        state,
        "s3://checkpoint/step-10",
        mesh=None,
        load_ema=False,
        _load_fn=load,
    )

    assert restored.params is loaded_params
    assert restored.opt_state is fresh_optimizer
    assert int(restored.step) == 0
    assert restored.pending_qb_betas.shape == (2, 3, 32)


def test_expert_range_extraction_preserves_selected_router_experts_and_qb_state() -> None:
    source_config = GrugModelConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=32,
        shared_expert_intermediate_dim=64,
        num_experts=8,
        num_experts_per_token=2,
        nested_expert_counts=(4,),
        nested_batch_fraction=0.25,
        num_layers=2,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=16,
    )
    target_config = dataclasses.replace(
        source_config,
        num_experts=4,
        nested_expert_counts=(),
        nested_batch_fraction=0.0,
    )
    mesh = compact_grug_mesh()
    with jax.set_mesh(mesh):
        source = Transformer.init(source_config, key=jax.random.PRNGKey(0))
        target_exemplar = Transformer.init(target_config, key=jax.random.PRNGKey(1))
    pending = jnp.arange(2 * 2 * 8, dtype=jnp.float32).reshape(2, 2, 8)
    target_pending_exemplar = jnp.zeros((2, 4), dtype=jnp.float32)

    target, target_pending = extract_expert_range(
        source,
        pending,
        target_model=target_exemplar,
        target_pending_qb_betas=target_pending_exemplar,
        expert_offset=0,
    )

    assert target.config is target_config
    assert jax.tree.structure(target) == jax.tree.structure(target_exemplar)
    assert target.stacked_blocks.stacked.mlp.router.shape == (2, 64, 4)
    assert target.stacked_blocks.stacked.mlp.router_bias.shape == (2, 4)
    assert target.stacked_blocks.stacked.mlp.expert_mlp.w_down.shape == (2, 4, 32, 64)
    np.testing.assert_array_equal(
        target.stacked_blocks.stacked.mlp.router,
        source.stacked_blocks.stacked.mlp.router[..., :4],
    )
    np.testing.assert_array_equal(target_pending, pending[:, 1, :4])
