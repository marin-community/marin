# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe.launch_augdk_nested_sft import SFTArm, sft_routing_model
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugTrainState, init_weights_only_from_checkpoint


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
