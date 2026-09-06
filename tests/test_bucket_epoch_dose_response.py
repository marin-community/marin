# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.domain_phase_mix import launch_bucket_epoch_dose_response as launch


def _run_spec(run_name: str = "p000_proportional_anchor") -> launch.EpochSweepRunSpec:
    return launch.EpochSweepRunSpec(
        scale="delphi_3e18",
        stage="full",
        run_order=0,
        run_id=7_291_000,
        run_name=run_name,
        point_id="proportional_anchor",
        point_kind="proportional_anchor",
        seed_block="main_grid",
        replicate_index=0,
        focal_index=None,
        focal_domain=None,
        epoch_multiplier=1.0,
        target_simulated_epochs=1.0,
        focal_weight=None,
        complement_scale=None,
        trainer_seed=0,
        data_seed=20_260_729,
        simulated_epoch_subset_seed=20_260_729,
        experiment_budget=1,
        target_budget=1,
        num_train_steps=3_007,
        expected_checkpoint_step=3_006,
        phase_weights={"phase_0": {"bucket": 1.0}, "phase_1": {"bucket": 1.0}},
    )


def test_load_table9_recovery_entries_is_strict(tmp_path):
    manifest = tmp_path / "recovery.csv"
    manifest.write_text(
        "run_name,checkpoint\n"
        "p000_proportional_anchor,pinlin_calvin_xu/data_mixture/bed3_20260729/"
        "p000_proportional_anchor-3e4d7e/hf/step-3006\n"
    )

    entries = launch._load_table9_recovery_entries(str(manifest))

    assert entries == [
        launch.Table9RecoveryEntry(
            run_name="p000_proportional_anchor",
            checkpoint=("pinlin_calvin_xu/data_mixture/bed3_20260729/" "p000_proportional_anchor-3e4d7e/hf/step-3006"),
        )
    ]


def test_load_table9_recovery_entries_rejects_duplicates(tmp_path):
    manifest = tmp_path / "recovery.csv"
    manifest.write_text("run_name,checkpoint\n" "p000_proportional_anchor,a\n" "p000_proportional_anchor,b\n")

    with pytest.raises(ValueError, match="duplicate run names"):
        launch._load_table9_recovery_entries(str(manifest))


def test_validate_table9_recovery_checkpoint_binds_run_and_step():
    spec = _run_spec()
    checkpoint = "pinlin_calvin_xu/data_mixture/bed3_20260729/" "p000_proportional_anchor-3e4d7e/hf/step-3006"

    launch._validate_table9_recovery_checkpoint(
        spec,
        checkpoint=checkpoint,
        scale=launch.Scale.DELPHI_3E18,
        stage=launch.Stage.FULL,
    )

    with pytest.raises(ValueError, match="does not belong"):
        launch._validate_table9_recovery_checkpoint(
            replace(spec, run_name="p001_d00_m0"),
            checkpoint=checkpoint,
            scale=launch.Scale.DELPHI_3E18,
            stage=launch.Stage.FULL,
        )
    with pytest.raises(ValueError, match="step-3006"):
        launch._validate_table9_recovery_checkpoint(
            spec,
            checkpoint=checkpoint.removesuffix("3006") + "3005",
            scale=launch.Scale.DELPHI_3E18,
            stage=launch.Stage.FULL,
        )
    with pytest.raises(ValueError, match="does not belong"):
        launch._validate_table9_recovery_checkpoint(
            spec,
            checkpoint=checkpoint.replace("/hf/", "/extra/hf/"),
            scale=launch.Scale.DELPHI_3E18,
            stage=launch.Stage.FULL,
        )
