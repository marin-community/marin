# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

from experiments.domain_phase_mix import launch_starcoder_heteroskedastic_snr as launcher


def test_default_anchor_panel_has_expected_repeat_specs():
    anchors = launcher.build_anchor_specs()
    run_specs = launcher.build_run_specs(anchors=anchors, repeats=5, run_id_base=810_000)

    assert len(anchors) == 10
    assert [anchor.anchor_id for anchor in anchors] == [
        "proportional",
        "nemotron_only",
        "balanced",
        "starcoder_only",
        "observed_global_best",
        "observed_p0_zero_slice_best",
        "late_code_moderate",
        "two_stage_default",
        "early_code_high_late_low",
        "high_both_moderate_late",
    ]
    assert len(run_specs) == 50
    assert [spec.run_id for spec in run_specs] == list(range(810_000, 810_050))
    assert run_specs[0].anchor_id == "proportional"
    assert run_specs[0].repeat_index == 0
    assert run_specs[4].anchor_id == "proportional"
    assert run_specs[4].repeat_index == 4
    assert run_specs[5].anchor_id == "nemotron_only"
    assert all(spec.trainer_seed is None for spec in run_specs)
    assert all(spec.data_seed == spec.run_id for spec in run_specs)
    assert all(spec.simulated_epoch_subset_seed is None for spec in run_specs)


def test_extra_anchor_csv_appends_without_shifting_default_ids(tmp_path: Path):
    extra = tmp_path / "anchors.csv"
    extra.write_text(
        "anchor_id,phase_0_starcoder,phase_1_starcoder,description\n" "extra_mid,0.25,0.75,extra midpoint\n"
    )

    anchors = launcher.build_anchor_specs(extra_anchor_csv=extra)
    run_specs = launcher.build_run_specs(anchors=anchors, repeats=2, run_id_base=820_000)

    assert len(anchors) == 11
    assert anchors[-1].anchor_id == "extra_mid"
    assert run_specs[0].anchor_id == "proportional"
    assert run_specs[-2].anchor_id == "extra_mid"
    assert run_specs[-2].run_id == 820_020
    assert run_specs[-1].run_id == 820_021


def test_no_default_anchors_supports_followup_only_csv(tmp_path: Path):
    extra = tmp_path / "anchors.csv"
    extra.write_text("anchor_id,phase_0_starcoder,phase_1_starcoder\n" "followup_corner,1.0,0.0\n")

    anchors = launcher.build_anchor_specs(extra_anchor_csv=extra, include_default_anchors=False)
    run_specs = launcher.build_run_specs(anchors=anchors, repeats=5, run_id_base=830_000)

    assert [anchor.anchor_id for anchor in anchors] == ["followup_corner"]
    assert len(run_specs) == 5
    assert run_specs[0].phase_weights["phase_0"] == {"nemotron_full": 0.0, "starcoder": 1.0}
    assert run_specs[0].phase_weights["phase_1"] == {"nemotron_full": 1.0, "starcoder": 0.0}


def test_missing_default_source_panel_uses_frozen_observed_anchors(tmp_path: Path, monkeypatch):
    missing_source = tmp_path / "missing_source_panel.csv"
    monkeypatch.setattr(launcher, "SOURCE_PANEL_CSV", missing_source)

    anchors = launcher.build_anchor_specs(source_panel_csv=missing_source)

    assert anchors[4] == replace(launcher.DEFAULT_OBSERVED_GLOBAL_BEST, anchor_index=4)
    assert anchors[5] == replace(launcher.DEFAULT_OBSERVED_P0_ZERO_SLICE_BEST, anchor_index=5)
    assert launcher._source_panel_csv_sha256(missing_source) == launcher.SOURCE_PANEL_CSV_SHA256


def test_build_launch_artifacts_dry_run_invariants():
    artifacts = launcher.build_launch_artifacts()

    assert len(artifacts.anchors) == 10
    assert len(artifacts.training_steps) == 50
    assert all(task.name != "wsc273" for task in launcher.EVAL_TASKS)
    assert all("wsc273" not in metric for metric in launcher.ANALYSIS_METRICS)
    assert artifacts.cache_eval_datasets_step.name.endswith("cache_eval_datasets")
    assert artifacts.analysis_step.name.endswith("/analysis")
    first_config = artifacts.training_steps[0].config
    assert str(artifacts.training_steps[0].override_output_path).startswith("gs://marin-us-central1/")
    assert first_config.env_vars["MARIN_PREFIX"] == "gs://marin-us-central1"
    assert first_config.env_vars["MARIN_TOKENIZER_CACHE_PATH"] == "gs://marin-us-central1/raw/tokenizers"
    assert first_config.env_vars["HF_ALLOW_CODE_EVAL"] == "1"
    assert artifacts.training_steps[0].fn.pip_dependency_groups == ["eval"]


def test_accepts_east5_launch_with_region_local_paths():
    artifacts = launcher.build_launch_artifacts(tpu_region="us-east5", tpu_zone="us-east5-a")

    first_config = artifacts.training_steps[0].config
    assert str(artifacts.training_steps[0].override_output_path).startswith("gs://marin-us-east5/")
    assert first_config.env_vars["MARIN_PREFIX"] == "gs://marin-us-east5"
    assert first_config.env_vars["MARIN_TOKENIZER_CACHE_PATH"] == "gs://marin-us-east5/raw/tokenizers"
    assert artifacts.cache_eval_datasets_step.config.gcs_path == "gs://marin-us-east5/raw/eval-datasets/code-tasks"


def test_skip_inline_eval_harness_sets_training_env_var():
    artifacts = launcher.build_launch_artifacts(
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        skip_inline_eval_harness=True,
    )

    env_vars = artifacts.training_steps[0].config.env_vars
    assert env_vars["LEVANTER_SKIP_EVAL_HARNESS"] == "1"
    assert env_vars["MARIN_PREFIX"] == "gs://marin-us-east5"
    assert env_vars["MARIN_EVAL_DATASETS_CACHE_DEPENDENCY"]


def test_rejects_cross_region_eval_cache_override():
    try:
        launcher.build_launch_artifacts(
            tpu_region="us-east5",
            tpu_zone="us-east5-a",
            eval_datasets_cache_path="gs://marin-us-central1/raw/eval-datasets/code-tasks",
        )
    except ValueError as exc:
        assert "Eval cache path must be local to us-east5" in str(exc)
    else:
        raise AssertionError("expected cross-region eval cache path to be rejected")


def test_rejects_executor_prefix_override():
    try:
        launcher._executor_prefix("gs://marin-us-east5/scratch/custom-prefix")
    except ValueError as exc:
        assert "--executor-prefix is disabled" in str(exc)
    else:
        raise AssertionError("expected executor prefix override to be rejected")
