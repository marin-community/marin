# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd

import experiments.domain_phase_mix.determinism_analysis as determinism_analysis
from experiments.domain_phase_mix.determinism_analysis import (
    PANEL_VS_NOISE_SUMMARY_CSV,
    PANEL_VS_NOISE_SUMMARY_JSON,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    _build_panel_vs_noise_summary_rows,
    create_panel_vs_noise_report,
)
from experiments.domain_phase_mix.launch_two_phase_many_first10_fixed_subset_panel import (
    PANEL_TRAINER_SEED,
    SOURCE_RUN_NAMES as PANEL_SOURCE_RUN_NAMES,
    build_run_specs as build_first10_panel_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    SIMULATED_EPOCH_SUBSET_SEED,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import (
    RUN_00097_PHASE_WEIGHTS,
    SOURCE_RUN_NAME,
)


def test_first10_fixed_subset_panel_builds_expected_manifest_and_weights():
    run_specs = build_first10_panel_run_specs()

    assert len(run_specs) == 10
    assert [spec.run_id for spec in run_specs] == list(range(10))
    assert [spec.source_run_name for spec in run_specs] == list(PANEL_SOURCE_RUN_NAMES)
    assert [spec.run_name for spec in run_specs] == [f"panel_{name}" for name in PANEL_SOURCE_RUN_NAMES]
    assert all(spec.cohort == "observed_panel" for spec in run_specs)
    assert all(spec.trainer_seed == PANEL_TRAINER_SEED for spec in run_specs)
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(spec.simulated_epoch_subset_seed == SIMULATED_EPOCH_SUBSET_SEED for spec in run_specs)
    for spec in run_specs:
        assert abs(sum(spec.phase_weights["phase_0"].values()) - 1.0) < 1e-12
        assert abs(sum(spec.phase_weights["phase_1"].values()) - 1.0) < 1e-12


def test_build_panel_vs_noise_summary_rows_reports_pairwise_exceedance():
    panel_df = pd.DataFrame(
        [
            {
                "run_name": "panel_run_00002",
                "lm_eval/mmlu_5shot/bpb": 2.10,
                "lm_eval/mmlu_5shot/choice_logprob": -1.40,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.55,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.266,
                "eval/paloma/c4_en/bpb": 1.18,
            },
            {
                "run_name": "panel_run_00003",
                "lm_eval/mmlu_5shot/bpb": 2.20,
                "lm_eval/mmlu_5shot/choice_logprob": -1.47,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.62,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.260,
                "eval/paloma/c4_en/bpb": 1.19,
            },
            {
                "run_name": "panel_run_00004",
                "lm_eval/mmlu_5shot/bpb": 2.35,
                "lm_eval/mmlu_5shot/choice_logprob": -1.58,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.76,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.252,
                "eval/paloma/c4_en/bpb": 1.20,
            },
        ]
    )
    baseline_df = pd.DataFrame(
        [
            {
                "run_name": "trainer_seed_10000",
                "lm_eval/mmlu_5shot/bpb": 2.18,
                "lm_eval/mmlu_5shot/choice_logprob": -1.49,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.66,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.259,
                "eval/paloma/c4_en/bpb": 1.19,
            },
            {
                "run_name": "trainer_seed_10001",
                "lm_eval/mmlu_5shot/bpb": 2.21,
                "lm_eval/mmlu_5shot/choice_logprob": -1.51,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.68,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.258,
                "eval/paloma/c4_en/bpb": 1.20,
            },
            {
                "run_name": "trainer_seed_10002",
                "lm_eval/mmlu_5shot/bpb": 2.19,
                "lm_eval/mmlu_5shot/choice_logprob": -1.50,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.67,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.2585,
                "eval/paloma/c4_en/bpb": 1.18,
            },
        ]
    )

    rows = _build_panel_vs_noise_summary_rows(
        panel_runs_df=panel_df,
        baseline_runs_df=baseline_df,
        primary_metrics=(
            "lm_eval/mmlu_5shot/bpb",
            "lm_eval/mmlu_5shot/choice_logprob",
            "lm_eval/mmlu_5shot/choice_logprob_norm",
            "lm_eval/mmlu_5shot/choice_prob_norm",
        ),
        secondary_metrics=("eval/paloma/c4_en/bpb",),
    )
    summary_df = pd.DataFrame(rows)

    bpb_row = summary_df[summary_df["metric"] == "lm_eval/mmlu_5shot/bpb"].iloc[0]
    assert bpb_row["panel_n"] == 3
    assert bpb_row["pairwise_n"] == 3
    assert bpb_row["panel_range_ratio_vs_noise"] > 1.0
    assert bpb_row["frac_pairwise_gt_1x_noise_std"] >= bpb_row["frac_pairwise_gt_2x_noise_std"]


def test_create_panel_vs_noise_report_writes_summary_outputs(tmp_path, monkeypatch):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "report"
    analysis_dir.mkdir()
    output_dir.mkdir()

    manifest = {
        "experiment_name": "pinlin_calvin_xu/data_mixture/ngd3dm2_first10_fixed_subset_panel",
        "runs": [
            {
                "run_id": 0,
                "run_name": "panel_run_00002",
                "cohort": "observed_panel",
                "trainer_seed": 0,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "source_run_name": "run_00002",
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
            {
                "run_id": 1,
                "run_name": "panel_run_00003",
                "cohort": "observed_panel",
                "trainer_seed": 0,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "source_run_name": "run_00003",
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest))

    results = pd.DataFrame(
        [
            {
                "run_id": 0,
                "run_name": "panel_run_00002",
                "status": "completed",
                "wandb_run_id": "panel-0",
                "lm_eval/mmlu_5shot/bpb": 2.10,
                "lm_eval/mmlu_5shot/choice_logprob": -1.40,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.55,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.266,
                "eval/paloma/c4_en/bpb": 1.18,
                "eval/paloma/macro_bpb": 1.39,
                "eval/uncheatable_eval/macro_bpb": 1.11,
            },
            {
                "run_id": 1,
                "run_name": "panel_run_00003",
                "status": "completed",
                "wandb_run_id": "panel-1",
                "lm_eval/mmlu_5shot/bpb": 2.30,
                "lm_eval/mmlu_5shot/choice_logprob": -1.55,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.70,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.255,
                "eval/paloma/c4_en/bpb": 1.20,
                "eval/paloma/macro_bpb": 1.40,
                "eval/uncheatable_eval/macro_bpb": 1.12,
            },
        ]
    )
    results.to_csv(analysis_dir / RESULTS_CSV, index=False)

    baseline_manifest = {
        "experiment_name": "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_fixed_subset_study",
        "runs": [
            {
                "run_id": 0,
                "run_name": "trainer_seed_10000",
                "cohort": "seed_sweep",
                "trainer_seed": 10_000,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "source_run_name": SOURCE_RUN_NAME,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
            {
                "run_id": 1,
                "run_name": "trainer_seed_10001",
                "cohort": "seed_sweep",
                "trainer_seed": 10_001,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "source_run_name": SOURCE_RUN_NAME,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
        ],
    }

    monkeypatch.setattr(
        determinism_analysis,
        "_collect_manifest_results_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "run_id": 0,
                    "run_name": "trainer_seed_10000",
                    "status": "completed",
                    "wandb_run_id": "baseline-0",
                    "lm_eval/mmlu_5shot/bpb": 2.18,
                    "lm_eval/mmlu_5shot/choice_logprob": -1.49,
                    "lm_eval/mmlu_5shot/choice_logprob_norm": -1.66,
                    "lm_eval/mmlu_5shot/choice_prob_norm": 0.259,
                    "eval/paloma/c4_en/bpb": 1.19,
                    "eval/paloma/macro_bpb": 1.39,
                    "eval/uncheatable_eval/macro_bpb": 1.11,
                },
                {
                    "run_id": 1,
                    "run_name": "trainer_seed_10001",
                    "status": "completed",
                    "wandb_run_id": "baseline-1",
                    "lm_eval/mmlu_5shot/bpb": 2.21,
                    "lm_eval/mmlu_5shot/choice_logprob": -1.51,
                    "lm_eval/mmlu_5shot/choice_logprob_norm": -1.68,
                    "lm_eval/mmlu_5shot/choice_prob_norm": 0.258,
                    "eval/paloma/c4_en/bpb": 1.20,
                    "eval/paloma/macro_bpb": 1.40,
                    "eval/uncheatable_eval/macro_bpb": 1.12,
                },
            ]
        ),
    )

    create_panel_vs_noise_report(
        determinism_analysis.PanelVsNoiseReportConfig(
            output_path=str(output_dir),
            run_manifest_path=str(manifest_path),
            analysis_output_path=str(analysis_dir),
            wandb_entity="unit",
            wandb_project="unit",
            baseline_manifest_json=json.dumps(baseline_manifest),
            primary_metrics=(
                "lm_eval/mmlu_5shot/bpb",
                "lm_eval/mmlu_5shot/choice_logprob",
                "lm_eval/mmlu_5shot/choice_logprob_norm",
                "lm_eval/mmlu_5shot/choice_prob_norm",
            ),
            secondary_metrics=(
                "eval/paloma/c4_en/bpb",
                "eval/paloma/macro_bpb",
                "eval/uncheatable_eval/macro_bpb",
            ),
        )
    )

    assert (output_dir / RUN_MANIFEST_FILE).exists()
    assert (output_dir / RESULTS_CSV).exists()
    assert (output_dir / PANEL_VS_NOISE_SUMMARY_CSV).exists()
    assert (output_dir / PANEL_VS_NOISE_SUMMARY_JSON).exists()

    summary = pd.read_csv(output_dir / PANEL_VS_NOISE_SUMMARY_CSV)
    bpb_row = summary[summary["metric"] == "lm_eval/mmlu_5shot/bpb"].iloc[0]
    assert bpb_row["panel_n"] == 2
    assert bpb_row["pairwise_n"] == 1
    assert bpb_row["panel_range_ratio_vs_noise"] > 1.0

    payload = json.loads((output_dir / PANEL_VS_NOISE_SUMMARY_JSON).read_text())
    assert payload["baseline_experiment"] == baseline_manifest["experiment_name"]
