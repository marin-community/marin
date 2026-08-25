# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix import launch_delphi_3e18_historical_frontier_prefix_replay as frontier
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base


def test_historical_frontier_spec_preserves_training_shape_and_restores_original_seed() -> None:
    canonical = base.DelphiSwarmRunSpec(
        run_order=7,
        run_id=123,
        run_name="canonical",
        source_run_name="canonical",
        source_experiment="canonical",
        panel_source="canonical",
        target_flops=3e18,
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        batch_size=128,
        train_steps=3007,
        realized_train_tokens=1_576_534_016,
        expected_checkpoint_step=3006,
        model_hidden_dim=1536,
        model_layers=16,
        non_embedding_params=350_000_000,
        total_trainable_params=360_000_000,
        tensor_parallel_size=1,
        data_seed=123,
        trainer_seed=0,
        phase_boundary=0.8,
        phase_0_fraction=0.8,
        phase_1_fraction=0.2,
        simulated_epoch_target_budget=base.SIMULATED_EPOCH_TARGET_BUDGET,
        available_top_level_tokens=base.TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        max_simulated_epoch=1.0,
        q95_simulated_epoch=1.0,
        mean_phase_tv_to_proportional=0.0,
        phase_weights={"phase_0": base._proportional_weights(), "phase_1": base._proportional_weights()},
    )
    phase_weights = {
        "phase_0": base._proportional_weights(),
        "phase_1": base._proportional_weights(),
    }

    spec = frontier.historical_frontier_spec(canonical, phase_weights)

    assert spec.train_steps == canonical.train_steps
    assert spec.realized_train_tokens == canonical.realized_train_tokens
    assert spec.model_hidden_dim == canonical.model_hidden_dim
    assert spec.run_id == frontier.HISTORICAL_RUN_ID
    assert spec.data_seed == frontier.HISTORICAL_DATA_SEED
    assert spec.trainer_seed == frontier.HISTORICAL_TRAINER_SEED
    assert spec.tpu_type == "v6e-8"
    assert spec.tpu_zone == "us-east5-b"


def test_historical_phase_weights_rejects_changed_bytes() -> None:
    with pytest.raises(ValueError, match="mixture bytes changed"):
        frontier.historical_phase_weights(b"domain,phase_0_weight,phase_1_weight\n")
