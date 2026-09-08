# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from marin.execution.context import executor_context

from experiments.domain_phase_mix import launch_delphi_one_phase_wspu_scaling as launcher

LOCAL_ANALYSIS_OUTPUT_PATH = str(
    Path(launcher.__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_baseline_mixtures_issue6607_20260623"
    / "isoflop_analysis_preflight"
)


def test_planned_rows_match_frozen_olmix_comparators():
    rows = launcher.planned_rows(LOCAL_ANALYSIS_OUTPUT_PATH)
    observed = {
        (str(row["target"]), float(row["target_flops"])): (
            str(row["candidate_id"]),
            int(row["data_seed"]),
            str(row["tpu_type"]),
        )
        for row in rows
    }

    assert observed == {
        ("uncheatable", 3e18): ("wspu_uncheatable_cap06", 666_200, "v6e-8"),
        ("table9", 3e18): ("wspu_table9_cap06", 662_009, "v6e-8"),
        ("uncheatable", 2e19): ("wspu_uncheatable_cap06", 666_202, "v6e-16"),
        ("table9", 2e19): ("wspu_table9_cap06", 662_001, "v6e-16"),
        ("uncheatable", 3e20): ("wspu_uncheatable_cap06", 666_204, "v6e-32"),
        ("table9", 3e20): ("wspu_table9_cap06", 662_003, "v6e-32"),
        ("uncheatable", 1e21): ("wspu_uncheatable_cap06", 666_206, "v6e-64"),
        ("table9", 1e21): ("wspu_table9_cap06", 662_005, "v6e-64"),
    }
    assert {int(row["trainer_seed"]) for row in rows} == {0}
    assert len({int(row["run_id"]) for row in rows}) == 8
    resolved_ladder = {
        float(row["target_flops"]): (
            int(row["expected_checkpoint_step"]),
            int(row["total_trainable_params"]),
            int(row["model_hidden_dim"]),
            int(row["model_layers"]),
        )
        for row in rows
    }
    assert resolved_ladder == {
        3e18: (3006, 358_304_128, 896, 10),
        2e19: (9901, 669_157_120, 1280, 13),
        3e20: (23531, 1_934_710_784, 2048, 21),
        1e21: (22056, 3_383_104_000, 2560, 26),
    }
    fit_preflight = {
        float(row["target_flops"]): (
            int(row["device_count"]),
            int(row["examples_per_device"]),
            float(row["estimated_persistent_state_gib_per_device"]),
            int(row["tensor_parallel_size"]),
        )
        for row in rows
    }
    assert {flops: values[:2] for flops, values in fit_preflight.items()} == {
        3e18: (8, 16),
        2e19: (16, 8),
        3e20: (32, 8),
        1e21: (64, 8),
    }
    assert all(state_gib < 0.7 for _, _, state_gib, _ in fit_preflight.values())
    assert {tp for _, _, _, tp in fit_preflight.values()} == {1}


def test_embedded_policies_round_trip_through_shared_runtime_accounting():
    rows = launcher.planned_rows(LOCAL_ANALYSIS_OUTPUT_PATH)
    for policy in launcher.selected_policies():
        row = next(row for row in rows if row["candidate_id"] == policy.candidate_id)
        config = launcher._training_config(row, LOCAL_ANALYSIS_OUTPUT_PATH, validation_configs=None)
        _, source = launcher._register_policy(config)
        try:
            phase_weights, diagnostics = launcher.base._read_phase_weights(source)

            assert phase_weights["phase_0"] == phase_weights["phase_1"]
            assert abs(sum(phase_weights["phase_0"].values()) - 1.0) < 1e-12
            assert abs(diagnostics.max_simulated_epoch - policy.max_materialized_epoch) < 1e-5
            assert diagnostics.max_simulated_epoch <= policy.epoch_cap
        finally:
            launcher.base.MIXTURE_SOURCES.pop(source.key, None)
            launcher.base._EMBEDDED_MIXTURE_WEIGHT_CSVS.pop(source.key, None)


def test_launch_graph_has_eight_independent_train_eval_pairs():
    rows = launcher.planned_rows(LOCAL_ANALYSIS_OUTPUT_PATH)
    with executor_context():
        artifacts = launcher.build_launch_artifacts(rows, LOCAL_ANALYSIS_OUTPUT_PATH, validation_configs={})

    assert len(artifacts.training_steps) == 8
    assert len(artifacts.eval_steps) == 8
    assert len({step.name for step in artifacts.training_steps}) == 8
    assert len({step.name for step in artifacts.eval_steps}) == 8
    assert max(len(step.name.rsplit("/", 1)[-1]) for step in artifacts.eval_steps) <= 32
    for training_step, eval_step, row in zip(artifacts.training_steps, artifacts.eval_steps, rows, strict=True):
        checkpoint = eval_step.config.eval_config.checkpoint_path
        assert checkpoint.step is training_step
        assert checkpoint.name == f"hf/step-{row['expected_checkpoint_step']}"

    manifest_rows = json.loads(artifacts.manifest_step.config.rows_json)
    assert len(manifest_rows) == 8
