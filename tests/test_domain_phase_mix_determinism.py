# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd

import experiments.domain_phase_mix.determinism_analysis as determinism_analysis
from experiments.domain_phase_mix.determinism_analysis import (
    CONTROL_RUNS_CSV,
    DETERMINISM_CONTROL_REPORT_JSON,
    FINAL_BPB_STATS_JSON,
    JitterReportConfig,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    SEED_RUNS_CSV,
    TRAJECTORY_BPB_STATS_CSV,
    create_jitter_report,
)
from experiments.domain_phase_mix.two_phase_starcoder_determinism_wsd import (
    FIXED_PHASE_WEIGHTS,
    OBJECTIVE_METRIC,
    build_run_specs,
    default_sweep_config,
    resolve_wsd_schedule,
)


def test_build_run_specs_has_expected_seed_and_control_cohorts():
    cfg = default_sweep_config("unit-test")
    run_specs = build_run_specs(cfg, seed_start=10_000, control_seed=424_242)

    assert len(run_specs) == 22
    assert [spec.run_id for spec in run_specs] == list(range(22))

    seed_specs = [spec for spec in run_specs if spec.cohort == "seed_sweep"]
    control_specs = [spec for spec in run_specs if spec.cohort == "determinism_control"]

    assert len(seed_specs) == 20
    assert len(control_specs) == 2
    assert len({spec.data_seed for spec in seed_specs}) == 20
    assert {spec.data_seed for spec in control_specs} == {424_242}
    assert all(spec.phase_weights == FIXED_PHASE_WEIGHTS for spec in run_specs)


def test_resolve_wsd_schedule_aligns_boundary_and_decay():
    cfg = default_sweep_config("unit-test")
    schedule = resolve_wsd_schedule(cfg)

    assert schedule.total_steps == 3814
    assert schedule.warmup_steps == 38
    assert schedule.phase_boundary_step == 1904
    assert schedule.decay_steps == 1910


def test_create_jitter_report_writes_ci_and_control_outputs(tmp_path, monkeypatch):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "report"
    analysis_dir.mkdir()
    output_dir.mkdir()

    manifest = {
        "runs": [
            {
                "run_id": 0,
                "run_name": "run_00000",
                "cohort": "seed_sweep",
                "data_seed": 10_000,
                "phase_weights": FIXED_PHASE_WEIGHTS,
            },
            {
                "run_id": 1,
                "run_name": "run_00001",
                "cohort": "seed_sweep",
                "data_seed": 10_001,
                "phase_weights": FIXED_PHASE_WEIGHTS,
            },
            {
                "run_id": 2,
                "run_name": "run_00002",
                "cohort": "determinism_control",
                "data_seed": 424_242,
                "phase_weights": FIXED_PHASE_WEIGHTS,
            },
            {
                "run_id": 3,
                "run_name": "run_00003",
                "cohort": "determinism_control",
                "data_seed": 424_242,
                "phase_weights": FIXED_PHASE_WEIGHTS,
            },
        ]
    }
    manifest_path.write_text(json.dumps(manifest))

    results = pd.DataFrame(
        [
            {
                "run_id": 0,
                "wandb_run_id": "wb-0",
                "status": "completed",
                OBJECTIVE_METRIC: 0.81,
            },
            {
                "run_id": 1,
                "wandb_run_id": "wb-1",
                "status": "completed",
                OBJECTIVE_METRIC: 0.83,
            },
            {
                "run_id": 2,
                "wandb_run_id": "wb-2",
                "status": "completed",
                OBJECTIVE_METRIC: 0.82,
            },
            {
                "run_id": 3,
                "wandb_run_id": "wb-3",
                "status": "completed",
                OBJECTIVE_METRIC: 0.82,
            },
        ]
    )
    results.to_csv(analysis_dir / RESULTS_CSV, index=False)

    def _fake_collect(runs_df: pd.DataFrame, **_: object) -> pd.DataFrame:
        rows = []
        for row in runs_df.itertuples(index=False):
            if row.cohort == "determinism_control":
                values = [1.00, 0.95]
            elif int(row.run_id) == 0:
                values = [1.10, 0.90]
            else:
                values = [1.00, 0.80]

            for step, value in zip((1000, 2000), values, strict=True):
                rows.append(
                    {
                        "run_id": int(row.run_id),
                        "run_name": row.run_name,
                        "cohort": row.cohort,
                        "data_seed": int(row.data_seed),
                        "wandb_run_id": row.wandb_run_id,
                        "step": step,
                        "metric_value": value,
                        "total_tokens": float(step) * 1_024.0,
                    }
                )

        return pd.DataFrame(rows)

    monkeypatch.setattr(determinism_analysis, "_collect_trajectory_rows", _fake_collect)

    create_jitter_report(
        JitterReportConfig(
            output_path=str(output_dir),
            objective_metric=OBJECTIVE_METRIC,
            wandb_entity="unit",
            wandb_project="unit",
            run_manifest_path=str(manifest_path),
            analysis_output_path=str(analysis_dir),
            bootstrap_samples=200,
            bootstrap_seed=0,
        )
    )

    assert (output_dir / SEED_RUNS_CSV).exists()
    assert (output_dir / CONTROL_RUNS_CSV).exists()
    assert (output_dir / FINAL_BPB_STATS_JSON).exists()
    assert (output_dir / TRAJECTORY_BPB_STATS_CSV).exists()
    assert (output_dir / DETERMINISM_CONTROL_REPORT_JSON).exists()

    final_stats = json.loads((output_dir / FINAL_BPB_STATS_JSON).read_text())
    assert final_stats["n_runs"] == 2
    assert final_stats["objective_metric"] == OBJECTIVE_METRIC

    traj_stats = pd.read_csv(output_dir / TRAJECTORY_BPB_STATS_CSV)
    assert set(["step", "metric_mean", "mean_ci_low", "mean_ci_high"]).issubset(traj_stats.columns)
    assert len(traj_stats) == 2

    control_report = json.loads((output_dir / DETERMINISM_CONTROL_REPORT_JSON).read_text())
    assert control_report["status"] == "ok"
    assert control_report["final_abs_diff"] == 0.0
    assert control_report["max_abs_step_diff"] == 0.0
