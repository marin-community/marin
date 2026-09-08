# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.domain_phase_mix import launch_delphi_one_phase_surrogate_challenger_validations_3e18 as launch
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


def test_frozen_candidate_contract_selects_eight_distinct_aggregate_mixtures():
    launch.validate_candidate_manifest(
        launch.DEFAULT_CANDIDATE_MANIFEST,
        launch.EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )
    candidates = launch.load_candidate_mixtures(
        launch.DEFAULT_CANDIDATE_WEIGHTS,
        launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
    )

    assert [candidate.candidate_id for candidate in candidates] == list(launch.CANDIDATE_IDS)
    assert len(candidates) == launch.EXPECTED_RUN_COUNT
    assert len({tuple(candidate.runtime_counts.values()) for candidate in candidates}) == launch.EXPECTED_RUN_COUNT
    assert {candidate.target for candidate in candidates} == set(launch.TARGETS)
    assert {candidate.epoch_cap for candidate in candidates} == set(launch.CAPS)
    assert all(sum(candidate.runtime_counts.values()) == launch.MIXTURE_BLOCK_SIZE for candidate in candidates)
    assert all(candidate.max_materialized_epoch <= candidate.epoch_cap for candidate in candidates)


def test_candidate_hash_drift_is_rejected(tmp_path):
    changed = tmp_path / "candidate_weights.csv"
    changed.write_bytes(launch.DEFAULT_CANDIDATE_WEIGHTS.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Candidate weights changed"):
        launch.load_candidate_mixtures(changed, launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256)


def test_manifest_hash_drift_is_rejected(tmp_path):
    changed = tmp_path / "manifest.json"
    changed.write_bytes(launch.DEFAULT_CANDIDATE_MANIFEST.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Candidate manifest changed"):
        launch.validate_candidate_manifest(changed, launch.EXPECTED_CANDIDATE_MANIFEST_SHA256)


def test_completed_adamh_builds_current_optimizer_config():
    optimizer = launch.current_completed_adamh_heuristic.build_optimizer_config(
        batch_size=128,
        tokens=1_576_534_016,
    )

    assert optimizer.learning_rate > 0
    assert optimizer.adam_lr > 0


def test_run_specs_are_full_horizon_tied_and_common_seeded():
    candidates = launch.load_candidate_mixtures(
        launch.DEFAULT_CANDIDATE_WEIGHTS,
        launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
    )
    specs = launch.build_run_specs(
        template=_template_spec(),
        candidates=candidates,
        tpu_type=launch.TPU_TYPE,
        tpu_region=launch.TPU_REGION,
        tpu_zone=launch.TPU_ZONE,
    )

    assert len(specs) == launch.EXPECTED_RUN_COUNT
    assert len({spec.run_id for spec in specs}) == launch.EXPECTED_RUN_COUNT
    assert {spec.data_seed for spec in specs} == {launch.COMMON_DATA_SEED}
    assert {spec.trainer_seed for spec in specs} == {launch.TRAINER_SEED}
    assert {spec.train_steps for spec in specs} == {3007}
    assert {spec.expected_checkpoint_step for spec in specs} == {3006}
    assert {(spec.tpu_type, spec.tpu_region, spec.tpu_zone) for spec in specs} == {
        (launch.TPU_TYPE, launch.TPU_REGION, launch.TPU_ZONE)
    }
    assert all(spec.phase_weights["phase_0"] == spec.phase_weights["phase_1"] for spec in specs)


def test_run_spec_binding_changes_only_policy_and_runtime_identity():
    candidates = launch.load_candidate_mixtures(
        launch.DEFAULT_CANDIDATE_WEIGHTS,
        launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
    )
    template = _template_spec()
    spec = launch.build_run_specs(
        template=template,
        candidates=candidates,
        tpu_type=launch.TPU_TYPE,
        tpu_region=launch.TPU_REGION,
        tpu_zone=launch.TPU_ZONE,
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
