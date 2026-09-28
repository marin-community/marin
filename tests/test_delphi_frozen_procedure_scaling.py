# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from marin.execution.context import executor_context

from experiments.domain_phase_mix import launch_delphi_frozen_procedure_scaling as launcher

REFERENCE_OUTPUTS = Path(launcher.__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
ANALYSIS_PATH = str(REFERENCE_OUTPUTS / "delphi_baseline_mixtures_issue6607_20260623/isoflop_analysis_preflight")


def test_scaling_manifest_preserves_validated_weights_and_original_ladder(tmp_path):
    rows = launcher.planned_rows(ANALYSIS_PATH)
    launcher.save_manifest(
        launcher.SaveManifestConfig(str(tmp_path), ANALYSIS_PATH, json.dumps([asdict(r) for r in rows]))
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    validations = {}
    for target in ("uncheatable", "table9"):
        path = (
            REFERENCE_OUTPUTS
            / f"delphi_frozen_procedure_validation_3e18_20260908/launch_dry_run/{target}/run_specs.json"
        )
        validations.update({row["source_run_name"]: row for row in json.loads(path.read_text())})

    observed = {}
    for row in manifest["runs"]:
        run, policy = row["run"], row["policy"]
        reference = validations[policy["candidate_id"]]
        assert run["phase_weights"] == reference["phase_weights"]
        assert run["simulated_epoch_target_budget"] == reference["simulated_epoch_target_budget"]
        assert run["trainer_seed"] == reference["trainer_seed"] == 0
        assert run["phase_weights"]["phase_0"] == run["phase_weights"]["phase_1"]
        observed[(policy["target"], run["target_flops"])] = (
            policy["candidate_id"],
            run["data_seed"],
            run["batch_size"],
            run["expected_checkpoint_step"],
            run["total_trainable_params"],
            run["tpu_type"],
        )
    assert observed == {
        ("uncheatable", 2e19): ("lwspu_u_snc_cap06", 666202, 128, 9901, 669157120, "v6e-16"),
        ("table9", 2e19): ("lwspu_t9_snc_cap08", 662001, 128, 9901, 669157120, "v6e-16"),
        ("uncheatable", 3e20): ("lwspu_u_snc_cap06", 666204, 256, 23531, 1934710784, "v6e-32"),
        ("table9", 3e20): ("lwspu_t9_snc_cap08", 662003, 256, 23531, 1934710784, "v6e-32"),
        ("uncheatable", 1e21): ("lwspu_u_snc_cap06", 666206, 512, 22056, 3383104000, "v6e-64"),
        ("table9", 1e21): ("lwspu_t9_snc_cap08", 662005, 512, 22056, 3383104000, "v6e-64"),
    }
    assert manifest["optimization_epoch_cap"] is None
    assert manifest["kl_coefficient"] == 0


def test_remote_worker_rejects_changed_embedded_weights():
    policy = launcher.selected_policies()[0]
    with pytest.raises(ValueError, match="Embedded weights changed"):
        launcher.register_policy(replace(policy, weights_csv=policy.weights_csv.replace("0.", "0.9", 1)), 666202)


def test_native_evaluations_depend_on_their_own_final_checkpoints():
    rows = launcher.planned_rows(ANALYSIS_PATH)
    with executor_context():
        graph = launcher.build_launch_artifacts(rows, ANALYSIS_PATH, validation_configs={})
    assert len(graph.training_steps) == len(graph.eval_steps) == len(rows) == 6
    assert len({step.name for step in graph.steps}) == 13
    for row, training, evaluation in zip(rows, graph.training_steps, graph.eval_steps, strict=True):
        checkpoint = evaluation.config.eval_config.checkpoint_path
        assert checkpoint.step is training
        assert checkpoint.name == f"hf/step-{row.run.expected_checkpoint_step}"
        assert training.config.training.mixture == row.policy.candidate_id
