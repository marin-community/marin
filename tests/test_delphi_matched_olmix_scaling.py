# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from marin.execution.context import executor_context

from experiments.domain_phase_mix import launch_delphi_frozen_procedure_scaling as frozen
from experiments.domain_phase_mix import launch_delphi_matched_olmix_scaling as launcher

REFERENCE_OUTPUTS = Path(launcher.__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
ANALYSIS_PATH = str(REFERENCE_OUTPUTS / "delphi_baseline_mixtures_issue6607_20260623/isoflop_analysis_preflight")


def test_matched_olmix_rungs_pair_with_mariner_scaling(tmp_path):
    rows = launcher.planned_rows(ANALYSIS_PATH)
    mariner = frozen.planned_rows(ANALYSIS_PATH)
    launcher.save_manifest(
        launcher.SaveManifestConfig(str(tmp_path), ANALYSIS_PATH, json.dumps([asdict(r) for r in rows]))
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["optimization_epoch_cap"] == 4
    assert manifest["kl_coefficients"] == {"uncheatable": 0.05, "table9": 0.005}
    paired = {(r.policy.target, r.run.target_flops): r.run for r in mariner}
    validations = {}
    seeds = {"uncheatable": launcher.matched.UNCHEATABLE_DATA_SEED, "table9": launcher.matched.TABLE9_DATA_SEED}
    for target, seed in seeds.items():
        path = (
            REFERENCE_OUTPUTS
            / f"delphi_matched_olmix_3e18_20260908/launch_dry_run/{target}_seed{seed}_t0/run_specs.json"
        )
        validations.update({row["source_run_name"]: row for row in json.loads(path.read_text())})
    for row in rows:
        reference = validations[row.policy.candidate_id]
        twin = paired[(row.policy.target, row.run.target_flops)]
        assert row.run.phase_weights == reference["phase_weights"]
        assert row.run.simulated_epoch_target_budget == reference["simulated_epoch_target_budget"]
        # Same seed, size, tokens, batch and hardware as MARINER's run at this rung; only the mixture differs.
        assert (row.run.data_seed, row.run.trainer_seed) == (twin.data_seed, twin.trainer_seed)
        assert (row.run.batch_size, row.run.expected_checkpoint_step, row.run.tpu_type) == (
            twin.batch_size,
            twin.expected_checkpoint_step,
            twin.tpu_type,
        )
        assert row.run.total_trainable_params == twin.total_trainable_params
        assert row.policy.max_materialized_epoch <= 4
    assert len(rows) == 6
    assert not {r.run.run_id for r in rows} & {r.run.run_id for r in mariner}


def test_remote_worker_rejects_changed_embedded_weights():
    policy = launcher.selected_policies()[0]
    with pytest.raises(ValueError, match="Embedded weights changed"):
        launcher.register_policy(replace(policy, weights_csv=policy.weights_csv.replace("0.", "0.9", 1)), 666202)


def test_native_evaluations_depend_on_their_own_final_checkpoints():
    rows = launcher.planned_rows(ANALYSIS_PATH)
    with executor_context():
        graph = launcher.build_launch_artifacts(rows, ANALYSIS_PATH, validation_configs={}, max_task_failures=4)
    assert len(graph.training_steps) == len(graph.eval_steps) == len(rows) == 6
    assert len({step.name for step in graph.steps}) == 13
    for row, training, evaluation in zip(rows, graph.training_steps, graph.eval_steps, strict=True):
        checkpoint = evaluation.config.eval_config.checkpoint_path
        assert checkpoint.step is training
        assert checkpoint.name == f"hf/step-{row.run.expected_checkpoint_step}"
        assert training.config.training.mixture == row.policy.candidate_id
        assert training.fn.max_task_failures == 4
