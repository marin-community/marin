# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix.launch_delphi_augmented_swarm_3e18 import DelphiSwarmRunSpec
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES


def _template_spec() -> DelphiSwarmRunSpec:
    weights = {domain: 1.0 / len(DOMAIN_NAMES) for domain in DOMAIN_NAMES}
    return DelphiSwarmRunSpec(
        run_order=0,
        run_id=1,
        run_name="template",
        source_run_name="template",
        source_experiment="source",
        panel_source="source",
        target_flops=3e18,
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        batch_size=128,
        train_steps=3007,
        realized_train_tokens=1_576_534_016,
        expected_checkpoint_step=3006,
        model_hidden_dim=2048,
        model_layers=16,
        non_embedding_params=1,
        total_trainable_params=2,
        tensor_parallel_size=1,
        data_seed=1,
        trainer_seed=0,
        phase_boundary=0.8,
        phase_0_fraction=0.8,
        phase_1_fraction=0.2,
        simulated_epoch_target_budget=1_576_728_576,
        available_top_level_tokens=2_000_000_000,
        max_simulated_epoch=1.0,
        q95_simulated_epoch=1.0,
        mean_phase_tv_to_proportional=0.0,
        phase_weights={"phase_0": weights, "phase_1": weights},
    )


def test_frozen_candidate_table_has_exact_expected_alias_structure():
    candidates, alias_map = sweep.load_candidate_mixtures(
        sweep.DEFAULT_CANDIDATE_WEIGHTS,
        sweep.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
    )

    assert len(candidates) == sweep.EXPECTED_RUN_COUNT
    assert alias_map == sweep.EXPECTED_ALIAS_MAP
    assert [candidate.candidate_id for candidate in candidates] == [
        "uncheatable_cap02",
        "uncheatable_cap04",
        "uncheatable_cap06",
        "uncheatable_cap08",
        "uncheatable_cap10",
        "table9_macro_cap02",
        "table9_macro_cap04",
        "table9_macro_cap06",
        "table9_macro_cap08",
        "table9_macro_cap10",
        "table9_macro_cap12",
    ]
    assert all(sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE for candidate in candidates)
    assert all(candidate.max_materialized_epoch <= candidate.epoch_cap for candidate in candidates)


def test_candidate_hash_drift_is_rejected(tmp_path):
    changed = tmp_path / "candidate_weights.csv"
    changed.write_bytes(sweep.DEFAULT_CANDIDATE_WEIGHTS.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Candidate weights changed"):
        sweep.load_candidate_mixtures(changed, sweep.EXPECTED_CANDIDATE_WEIGHTS_SHA256)


def test_completed_adamh_builds_current_optimizer_config():
    optimizer = sweep.current_completed_adamh_heuristic.build_optimizer_config(
        batch_size=128,
        tokens=1_576_534_016,
    )

    assert optimizer.learning_rate > 0
    assert optimizer.adam_lr > 0


def test_training_wrapper_installs_current_optimizer_adapter(monkeypatch):
    observed = {}

    def observe_training(config):
        observed["config"] = config
        observed["heuristic"] = sweep.base.completed_adamh_heuristic

    monkeypatch.setattr(sweep.base, "run_delphi_swarm_training", observe_training)
    monkeypatch.setattr(sweep.base, "completed_adamh_heuristic", object())
    config = object()

    sweep.run_one_phase_training(config)

    assert observed["config"] is config
    assert observed["heuristic"] is sweep.current_completed_adamh_heuristic


def test_run_specs_are_full_horizon_tied_and_common_seeded():
    candidates, _ = sweep.load_candidate_mixtures(
        sweep.DEFAULT_CANDIDATE_WEIGHTS,
        sweep.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
    )
    specs = sweep.build_run_specs(
        template=_template_spec(),
        candidates=candidates,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
    )

    assert len(specs) == sweep.EXPECTED_RUN_COUNT
    assert len({spec.run_id for spec in specs}) == sweep.EXPECTED_RUN_COUNT
    assert {spec.data_seed for spec in specs} == {sweep.COMMON_DATA_SEED}
    assert {spec.trainer_seed for spec in specs} == {sweep.TRAINER_SEED}
    assert {spec.train_steps for spec in specs} == {3007}
    assert {spec.expected_checkpoint_step for spec in specs} == {3006}
    assert {(spec.tpu_type, spec.tpu_region, spec.tpu_zone) for spec in specs} == {
        (sweep.TPU_TYPE, sweep.TPU_REGION, sweep.TPU_ZONE)
    }
    assert all(spec.phase_weights["phase_0"] == spec.phase_weights["phase_1"] for spec in specs)


def test_run_spec_binding_preserves_architecture_and_optimizer_horizon():
    candidates, _ = sweep.load_candidate_mixtures(
        sweep.DEFAULT_CANDIDATE_WEIGHTS,
        sweep.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
    )
    template = _template_spec()

    spec = sweep.build_run_specs(
        template=template,
        candidates=candidates,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
    )[0]

    assert (
        replace(
            spec,
            run_order=template.run_order,
            run_id=template.run_id,
            run_name=template.run_name,
            source_run_name=template.source_run_name,
            source_experiment=template.source_experiment,
            panel_source=template.panel_source,
            tpu_type=template.tpu_type,
            tpu_region=template.tpu_region,
            tpu_zone=template.tpu_zone,
            tensor_parallel_size=template.tensor_parallel_size,
            data_seed=template.data_seed,
            trainer_seed=template.trainer_seed,
            max_simulated_epoch=template.max_simulated_epoch,
            q95_simulated_epoch=template.q95_simulated_epoch,
            mean_phase_tv_to_proportional=template.mean_phase_tv_to_proportional,
            phase_weights=template.phase_weights,
        )
        == template
    )
