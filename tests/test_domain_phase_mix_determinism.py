# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fray.cluster import ResourceConfig

import experiments.domain_phase_mix.determinism_analysis as determinism_analysis
from experiments.domain_phase_mix.determinism_analysis import (
    CollectManifestResultsConfig,
    CONTROL_RUNS_CSV,
    DETERMINISM_CONTROL_REPORT_JSON,
    FINAL_BPB_STATS_JSON,
    FIXED_SUBSET_NOISE_SUMMARY_CSV,
    FIXED_SUBSET_NOISE_SUMMARY_JSON,
    JitterReportConfig,
    RESULTS_CSV,
    RUN_MANIFEST_FILE,
    SEED_RUNS_CSV,
    SWARM_COMPARISON_CSV,
    SWARM_COMPARISON_JSON,
    TRAJECTORY_BPB_STATS_CSV,
    _build_compute_scaling_summary_rows,
    _build_fixed_subset_summary_rows,
    collect_manifest_results,
    create_fixed_subset_noise_report,
    create_jitter_report,
)
from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import (
    EXACT_CONTROL_DATA_SEED,
    EXACT_CONTROL_NAMES,
    RUN_00097_PHASE_WEIGHTS,
    SEED_SWEEP_START,
    SOURCE_RUN_NAME,
    build_run_specs as build_run_00097_seed_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_compute_scaling_study import (
    OLMO3_30M_3B_BUDGET,
    OLMO3_30M_3B_LADDER,
    REGMIX60M_6B_BUDGET,
    REGMIX60M_6B_LADDER,
    build_run_specs as build_compute_scaling_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    SIMULATED_EPOCH_SUBSET_SEED,
    build_run_specs as build_fixed_subset_run_specs,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
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


def test_default_train_uses_trainer_seed_and_preserves_data_seed(monkeypatch):
    monkeypatch.setattr("experiments.defaults._prepare_data_config", lambda tokenized, use_default_validation: object())
    monkeypatch.setattr("experiments.defaults._get_vocab_size", lambda _: 32_000)
    monkeypatch.setattr("experiments.defaults._validate_train_length", lambda train_seq_len, model_config: 2048)
    monkeypatch.setattr("experiments.defaults.compute_num_parameters", lambda model_config, vocab_size: 123)

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
        train_batch_size=8,
        num_train_steps=10,
        learning_rate=1e-3,
        trainer_seed=123,
        data_seed=None,
    )
    step = default_train(
        name="unit/test",
        tokenized=object(),
        model_config=SimpleNamespace(max_seq_len=2048),
        train_config=train_config,
        eval_harness_tasks=(),
        use_default_validation=False,
    )

    assert step.config.train_config.trainer.seed == 123
    assert step.config.train_config.data_seed is None


def test_mixture_experiment_create_train_config_allows_none_data_seed_override():
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="unit-test")
    train_config = experiment.create_train_config(run_id=42, data_seed=None, trainer_seed=123)

    assert train_config.data_seed is None
    assert train_config.trainer_seed == 123


def test_mixture_experiment_create_training_step_forwards_simulated_epoch_subset_seed(monkeypatch):
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="unit-test")
    captured = {}

    def fake_simulated_epoching_train(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(config=SimpleNamespace())

    monkeypatch.setattr(
        "experiments.domain_phase_mix.experiment.simulated_epoching_train",
        fake_simulated_epoching_train,
    )

    experiment.create_training_step(
        weight_config=WeightConfig(run_id=7, phase_weights=RUN_00097_PHASE_WEIGHTS),
        run_name="unit_run",
        trainer_seed=123,
        data_seed=None,
        simulated_epoch_subset_seed=97,
    )

    assert captured["simulated_epoch_subset_seed"] == 97
    assert captured["train_config"].trainer_seed == 123
    assert captured["train_config"].data_seed is None


def test_run_00097_seed_study_builds_expected_manifest_and_weights():
    run_specs = build_run_00097_seed_specs()

    assert len(run_specs) == 12
    assert [spec.run_id for spec in run_specs] == list(range(12))

    seed_specs = [spec for spec in run_specs if spec.cohort == "seed_sweep"]
    control_specs = [spec for spec in run_specs if spec.cohort == "exact_replay_control"]

    assert len(seed_specs) == 10
    assert len(control_specs) == 2
    assert [spec.trainer_seed for spec in seed_specs] == list(range(SEED_SWEEP_START, SEED_SWEEP_START + 10))
    assert all(spec.data_seed is None for spec in seed_specs)
    assert [spec.run_name for spec in control_specs] == list(EXACT_CONTROL_NAMES)
    assert {spec.data_seed for spec in control_specs} == {EXACT_CONTROL_DATA_SEED}
    assert all(spec.source_run_name == SOURCE_RUN_NAME for spec in run_specs)
    assert all(abs(sum(RUN_00097_PHASE_WEIGHTS[phase].values()) - 1.0) < 1e-12 for phase in RUN_00097_PHASE_WEIGHTS)
    assert all(spec.phase_weights == RUN_00097_PHASE_WEIGHTS for spec in run_specs)


def test_run_00097_fixed_subset_study_builds_expected_manifest_and_weights():
    run_specs = build_fixed_subset_run_specs()

    assert len(run_specs) == 12
    assert [spec.run_id for spec in run_specs] == list(range(12))

    seed_specs = [spec for spec in run_specs if spec.cohort == "seed_sweep"]
    control_specs = [spec for spec in run_specs if spec.cohort == "exact_replay_control"]

    assert len(seed_specs) == 10
    assert len(control_specs) == 2
    assert [spec.trainer_seed for spec in seed_specs] == list(range(SEED_SWEEP_START, SEED_SWEEP_START + 10))
    assert all(spec.data_seed is None for spec in seed_specs)
    assert all(spec.simulated_epoch_subset_seed == SIMULATED_EPOCH_SUBSET_SEED for spec in run_specs)
    assert [spec.run_name for spec in control_specs] == list(EXACT_CONTROL_NAMES)
    assert {spec.data_seed for spec in control_specs} == {EXACT_CONTROL_DATA_SEED}
    assert all(spec.phase_weights == RUN_00097_PHASE_WEIGHTS for spec in run_specs)


def test_run_00097_compute_scaling_study_builds_expected_manifest_and_weights():
    run_specs = build_compute_scaling_run_specs()

    assert len(run_specs) == 20
    assert [spec.run_id for spec in run_specs] == list(range(20))
    assert [spec.trainer_seed for spec in run_specs[:10]] == list(range(SEED_SWEEP_START, SEED_SWEEP_START + 10))
    assert [spec.trainer_seed for spec in run_specs[10:]] == list(range(SEED_SWEEP_START, SEED_SWEEP_START + 10))
    assert all(spec.data_seed is None for spec in run_specs)
    assert {spec.ladder for spec in run_specs[:10]} == {REGMIX60M_6B_LADDER}
    assert {spec.ladder for spec in run_specs[10:]} == {OLMO3_30M_3B_LADDER}
    assert {spec.experiment_budget for spec in run_specs[:10]} == {REGMIX60M_6B_BUDGET}
    assert {spec.experiment_budget for spec in run_specs[10:]} == {OLMO3_30M_3B_BUDGET}
    assert {spec.num_train_steps for spec in run_specs[:10]} == {22_888}
    assert {spec.num_train_steps for spec in run_specs[10:]} == {11_444}
    assert all(spec.phase_weights == RUN_00097_PHASE_WEIGHTS for spec in run_specs)


def test_collect_manifest_results_falls_back_to_name_lookup_when_tags_are_missing(tmp_path, monkeypatch):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    manifest = {
        "experiment_name": "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study",
        "runs": [
            {
                "run_id": 0,
                "run_name": "trainer_seed_10000",
                "cohort": "seed_sweep",
                "trainer_seed": 10_000,
                "data_seed": None,
                "source_run_name": SOURCE_RUN_NAME,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr("experiments.domain_phase_mix.analysis.query_wandb_runs", lambda **_: [])
    monkeypatch.setattr(
        "experiments.domain_phase_mix.analysis.query_wandb_runs_by_name_substrings",
        lambda **_: [
            {
                "wandb_run_id": "trainer_seed_10000-5bf683",
                "wandb_run_name": "pinlin_calvin_xu/data_mixture/ngd3dm~6fb11cac/trainer_seed_10000",
                "status": "finished",
                OBJECTIVE_METRIC: 2.34,
            }
        ],
    )

    collect_manifest_results(
        CollectManifestResultsConfig(
            output_path=str(output_dir),
            run_manifest_path=str(manifest_path),
            wandb_entity="marin-community",
            wandb_project="marin",
        )
    )

    results_df = pd.read_csv(output_dir / RESULTS_CSV)
    assert len(results_df) == 1
    assert results_df.loc[0, "status"] == "completed"
    assert results_df.loc[0, "wandb_run_id"] == "trainer_seed_10000-5bf683"
    assert results_df.loc[0, OBJECTIVE_METRIC] == pytest.approx(2.34)


def test_collect_manifest_results_prefers_exact_truncated_display_name_when_suffixes_collide(tmp_path, monkeypatch):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    manifest = {
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
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr("experiments.domain_phase_mix.analysis.query_wandb_runs", lambda **_: [])
    monkeypatch.setattr(
        "experiments.domain_phase_mix.analysis.query_wandb_runs_by_name_substrings",
        lambda **_: [
            {
                "wandb_run_id": "trainer_seed_10000-5bf683",
                "wandb_run_name": "pinlin_calvin_xu/data_mixture/ngd3dm~6fb11cac/trainer_seed_10000",
                "status": "finished",
                OBJECTIVE_METRIC: 2.34,
            },
            {
                "wandb_run_id": "trainer_seed_10000-084313",
                "wandb_run_name": "pinlin_calvin_xu/data_mixture/ngd3dm~06b6bfd6/trainer_seed_10000",
                "status": "finished",
                OBJECTIVE_METRIC: 2.12,
            },
        ],
    )

    collect_manifest_results(
        CollectManifestResultsConfig(
            output_path=str(output_dir),
            run_manifest_path=str(manifest_path),
            wandb_entity="marin-community",
            wandb_project="marin",
        )
    )

    results_df = pd.read_csv(output_dir / RESULTS_CSV)
    assert len(results_df) == 1
    assert results_df.loc[0, "wandb_run_id"] == "trainer_seed_10000-084313"
    assert results_df.loc[0, OBJECTIVE_METRIC] == pytest.approx(2.12)


def test_build_compute_scaling_summary_rows_computes_primary_ratios():
    current_df = pd.DataFrame(
        [
            {
                "run_name": "regmix60m_6b_trainer_seed_10000",
                "ladder": REGMIX60M_6B_LADDER,
                "lm_eval/mmlu_5shot/bpb": 2.20,
                "lm_eval/mmlu_5shot/choice_logprob": -1.50,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.60,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.260,
                "eval/paloma/c4_en/bpb": 1.19,
            },
            {
                "run_name": "regmix60m_6b_trainer_seed_10001",
                "ladder": REGMIX60M_6B_LADDER,
                "lm_eval/mmlu_5shot/bpb": 2.22,
                "lm_eval/mmlu_5shot/choice_logprob": -1.52,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.62,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.258,
                "eval/paloma/c4_en/bpb": 1.18,
            },
            {
                "run_name": "olmo3_30m_3b_trainer_seed_10000",
                "ladder": OLMO3_30M_3B_LADDER,
                "lm_eval/mmlu_5shot/bpb": 2.30,
                "lm_eval/mmlu_5shot/choice_logprob": -1.58,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.69,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.255,
                "eval/paloma/c4_en/bpb": 1.17,
            },
            {
                "run_name": "olmo3_30m_3b_trainer_seed_10001",
                "ladder": OLMO3_30M_3B_LADDER,
                "lm_eval/mmlu_5shot/bpb": 2.31,
                "lm_eval/mmlu_5shot/choice_logprob": -1.59,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.70,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.254,
                "eval/paloma/c4_en/bpb": 1.16,
            },
        ]
    )
    baseline_df = pd.DataFrame(
        [
            {
                "run_name": "trainer_seed_10000",
                "lm_eval/mmlu_5shot/bpb": 2.10,
                "lm_eval/mmlu_5shot/choice_logprob": -1.45,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.55,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.265,
                "eval/paloma/c4_en/bpb": 1.20,
            },
            {
                "run_name": "trainer_seed_10001",
                "lm_eval/mmlu_5shot/bpb": 2.30,
                "lm_eval/mmlu_5shot/choice_logprob": -1.55,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.75,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.255,
                "eval/paloma/c4_en/bpb": 1.21,
            },
        ]
    )

    rows = _build_compute_scaling_summary_rows(
        current_runs_df=current_df,
        baseline_runs_df=baseline_df,
        primary_metrics=(
            "lm_eval/mmlu_5shot/bpb",
            "lm_eval/mmlu_5shot/choice_logprob",
            "lm_eval/mmlu_5shot/choice_logprob_norm",
            "lm_eval/mmlu_5shot/choice_prob_norm",
        ),
        secondary_metrics=("eval/paloma/c4_en/bpb",),
        bootstrap_samples=64,
        bootstrap_seed=0,
    )
    summary_df = pd.DataFrame(rows)

    assert set(summary_df["cohort"]) == {"baseline_1x", REGMIX60M_6B_LADDER, OLMO3_30M_3B_LADDER}
    regmix_bpb = summary_df[
        (summary_df["cohort"] == REGMIX60M_6B_LADDER) & (summary_df["metric"] == "lm_eval/mmlu_5shot/bpb")
    ].iloc[0]
    olmo3_bpb = summary_df[
        (summary_df["cohort"] == OLMO3_30M_3B_LADDER) & (summary_df["metric"] == "lm_eval/mmlu_5shot/bpb")
    ].iloc[0]

    assert regmix_bpb["variance_ratio_vs_baseline_1x"] < 1.0
    assert regmix_bpb["range_ratio_vs_baseline_1x"] < 1.0
    assert np.isfinite(regmix_bpb["variance_ratio_ci_low"])
    assert np.isfinite(regmix_bpb["variance_ratio_ci_high"])
    assert np.isnan(olmo3_bpb["variance_ratio_vs_baseline_1x"])
    assert np.isnan(olmo3_bpb["range_ratio_vs_baseline_1x"])


def test_build_fixed_subset_summary_rows_computes_primary_ratios():
    current_df = pd.DataFrame(
        [
            {
                "run_name": "trainer_seed_10000",
                "lm_eval/mmlu_5shot/bpb": 2.15,
                "lm_eval/mmlu_5shot/choice_logprob": -1.46,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.58,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.263,
                "eval/paloma/c4_en/bpb": 1.19,
            },
            {
                "run_name": "trainer_seed_10001",
                "lm_eval/mmlu_5shot/bpb": 2.17,
                "lm_eval/mmlu_5shot/choice_logprob": -1.47,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.60,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.262,
                "eval/paloma/c4_en/bpb": 1.18,
            },
        ]
    )
    baseline_df = pd.DataFrame(
        [
            {
                "run_name": "trainer_seed_10000",
                "lm_eval/mmlu_5shot/bpb": 2.10,
                "lm_eval/mmlu_5shot/choice_logprob": -1.45,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.55,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.265,
                "eval/paloma/c4_en/bpb": 1.20,
            },
            {
                "run_name": "trainer_seed_10001",
                "lm_eval/mmlu_5shot/bpb": 2.30,
                "lm_eval/mmlu_5shot/choice_logprob": -1.55,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.75,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.255,
                "eval/paloma/c4_en/bpb": 1.21,
            },
        ]
    )

    rows = _build_fixed_subset_summary_rows(
        current_runs_df=current_df,
        baseline_runs_df=baseline_df,
        primary_metrics=(
            "lm_eval/mmlu_5shot/bpb",
            "lm_eval/mmlu_5shot/choice_logprob",
            "lm_eval/mmlu_5shot/choice_logprob_norm",
            "lm_eval/mmlu_5shot/choice_prob_norm",
        ),
        secondary_metrics=("eval/paloma/c4_en/bpb",),
        bootstrap_samples=64,
        bootstrap_seed=0,
    )
    summary_df = pd.DataFrame(rows)

    assert set(summary_df["cohort"]) == {"baseline_1x", "fixed_subset_1x"}
    fixed_subset_bpb = summary_df[
        (summary_df["cohort"] == "fixed_subset_1x") & (summary_df["metric"] == "lm_eval/mmlu_5shot/bpb")
    ].iloc[0]

    assert fixed_subset_bpb["variance_ratio_vs_baseline_1x"] < 1.0
    assert fixed_subset_bpb["range_ratio_vs_baseline_1x"] < 1.0
    assert np.isfinite(fixed_subset_bpb["variance_ratio_ci_low"])
    assert np.isfinite(fixed_subset_bpb["variance_ratio_ci_high"])


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
                "trainer_seed": 10_000,
                "data_seed": 10_000,
                "source_run_name": SOURCE_RUN_NAME,
                "phase_weights": FIXED_PHASE_WEIGHTS,
            },
            {
                "run_id": 1,
                "run_name": "run_00001",
                "cohort": "seed_sweep",
                "trainer_seed": 10_001,
                "data_seed": 10_001,
                "source_run_name": SOURCE_RUN_NAME,
                "phase_weights": FIXED_PHASE_WEIGHTS,
            },
            {
                "run_id": 2,
                "run_name": "exact_replay_control_a",
                "cohort": "exact_replay_control",
                "trainer_seed": 0,
                "data_seed": 424_242,
                "source_run_name": SOURCE_RUN_NAME,
                "phase_weights": FIXED_PHASE_WEIGHTS,
            },
            {
                "run_id": 3,
                "run_name": "exact_replay_control_b",
                "cohort": "exact_replay_control",
                "trainer_seed": 0,
                "data_seed": 424_242,
                "source_run_name": SOURCE_RUN_NAME,
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

    swarm_results = pd.DataFrame(
        [
            {
                OBJECTIVE_METRIC: 0.70,
                "lm_eval/mmlu_5shot/choice_logprob": -1.10,
                "eval/paloma/c4_en/bpb": 1.15,
                "eval/paloma/macro_bpb": 1.35,
                "eval/uncheatable_eval/macro_bpb": 1.05,
            },
            {
                OBJECTIVE_METRIC: 0.90,
                "lm_eval/mmlu_5shot/choice_logprob": -1.30,
                "eval/paloma/c4_en/bpb": 1.20,
                "eval/paloma/macro_bpb": 1.40,
                "eval/uncheatable_eval/macro_bpb": 1.10,
            },
            {
                OBJECTIVE_METRIC: 1.10,
                "lm_eval/mmlu_5shot/choice_logprob": -1.50,
                "eval/paloma/c4_en/bpb": 1.25,
                "eval/paloma/macro_bpb": 1.45,
                "eval/uncheatable_eval/macro_bpb": 1.15,
            },
        ]
    )
    swarm_path = tmp_path / "two_phase_many.csv"
    swarm_results.to_csv(swarm_path, index=False)

    def _fake_collect(runs_df: pd.DataFrame, **_: object) -> pd.DataFrame:
        rows = []
        for row in runs_df.itertuples(index=False):
            if row.cohort == "exact_replay_control":
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
                        "trainer_seed": int(row.trainer_seed),
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
            swarm_results_csv_path=str(swarm_path),
            comparison_metrics=(
                OBJECTIVE_METRIC,
                "lm_eval/mmlu_5shot/choice_logprob",
                "eval/paloma/c4_en/bpb",
                "eval/paloma/macro_bpb",
                "eval/uncheatable_eval/macro_bpb",
            ),
            bootstrap_samples=200,
            bootstrap_seed=0,
        )
    )

    assert (output_dir / SEED_RUNS_CSV).exists()
    assert (output_dir / CONTROL_RUNS_CSV).exists()
    assert (output_dir / FINAL_BPB_STATS_JSON).exists()
    assert (output_dir / TRAJECTORY_BPB_STATS_CSV).exists()
    assert (output_dir / DETERMINISM_CONTROL_REPORT_JSON).exists()
    assert (output_dir / SWARM_COMPARISON_JSON).exists()
    assert (output_dir / SWARM_COMPARISON_CSV).exists()

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

    swarm_comparison = pd.read_csv(output_dir / SWARM_COMPARISON_CSV)
    assert set(swarm_comparison["metric"]) == {
        OBJECTIVE_METRIC,
        "lm_eval/mmlu_5shot/choice_logprob",
        "eval/paloma/c4_en/bpb",
        "eval/paloma/macro_bpb",
        "eval/uncheatable_eval/macro_bpb",
    }
    objective_row = swarm_comparison.loc[swarm_comparison["metric"] == OBJECTIVE_METRIC].iloc[0]
    assert objective_row["repeat_n"] == 2
    assert objective_row["swarm_n"] == 3
    assert objective_row["variance_ratio"] == pytest.approx(0.005)


def test_create_fixed_subset_noise_report_writes_summary_and_control_context(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "report"
    analysis_dir.mkdir()
    output_dir.mkdir()

    seed_runs = pd.DataFrame(
        [
            {
                "run_id": 0,
                "run_name": "trainer_seed_10000",
                "status": "completed",
                "wandb_run_id": "wb-0",
                "lm_eval/mmlu_5shot/bpb": 2.15,
                "lm_eval/mmlu_5shot/choice_logprob": -1.46,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.58,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.263,
                "eval/paloma/c4_en/bpb": 1.19,
                "eval/paloma/macro_bpb": 1.39,
                "eval/uncheatable_eval/macro_bpb": 1.11,
            },
            {
                "run_id": 1,
                "run_name": "trainer_seed_10001",
                "status": "completed",
                "wandb_run_id": "wb-1",
                "lm_eval/mmlu_5shot/bpb": 2.17,
                "lm_eval/mmlu_5shot/choice_logprob": -1.47,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.60,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.262,
                "eval/paloma/c4_en/bpb": 1.18,
                "eval/paloma/macro_bpb": 1.38,
                "eval/uncheatable_eval/macro_bpb": 1.10,
            },
        ]
    )
    seed_runs.to_csv(analysis_dir / SEED_RUNS_CSV, index=False)
    control_report = {
        "status": "ok",
        "final_abs_diff": 0.0,
        "max_abs_step_diff": 0.0,
    }
    (analysis_dir / DETERMINISM_CONTROL_REPORT_JSON).write_text(json.dumps(control_report))

    baseline_manifest = {
        "experiment_name": "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study",
        "runs": [
            {
                "run_id": 0,
                "run_name": "trainer_seed_10000",
                "cohort": "seed_sweep",
                "trainer_seed": 10_000,
                "data_seed": None,
                "source_run_name": SOURCE_RUN_NAME,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
            {
                "run_id": 1,
                "run_name": "trainer_seed_10001",
                "cohort": "seed_sweep",
                "trainer_seed": 10_001,
                "data_seed": None,
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
                    "lm_eval/mmlu_5shot/bpb": 2.10,
                    "lm_eval/mmlu_5shot/choice_logprob": -1.45,
                    "lm_eval/mmlu_5shot/choice_logprob_norm": -1.55,
                    "lm_eval/mmlu_5shot/choice_prob_norm": 0.265,
                    "eval/paloma/c4_en/bpb": 1.20,
                    "eval/paloma/macro_bpb": 1.40,
                    "eval/uncheatable_eval/macro_bpb": 1.12,
                },
                {
                    "run_id": 1,
                    "run_name": "trainer_seed_10001",
                    "status": "completed",
                    "wandb_run_id": "baseline-1",
                    "lm_eval/mmlu_5shot/bpb": 2.30,
                    "lm_eval/mmlu_5shot/choice_logprob": -1.55,
                    "lm_eval/mmlu_5shot/choice_logprob_norm": -1.75,
                    "lm_eval/mmlu_5shot/choice_prob_norm": 0.255,
                    "eval/paloma/c4_en/bpb": 1.21,
                    "eval/paloma/macro_bpb": 1.41,
                    "eval/uncheatable_eval/macro_bpb": 1.13,
                },
            ]
        ),
    )

    create_fixed_subset_noise_report(
        determinism_analysis.FixedSubsetNoiseReportConfig(
            output_path=str(output_dir),
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
            bootstrap_samples=64,
            bootstrap_seed=0,
        )
    )

    assert (output_dir / FIXED_SUBSET_NOISE_SUMMARY_CSV).exists()
    assert (output_dir / FIXED_SUBSET_NOISE_SUMMARY_JSON).exists()

    summary = pd.read_csv(output_dir / FIXED_SUBSET_NOISE_SUMMARY_CSV)
    assert set(summary["cohort"]) == {"baseline_1x", "fixed_subset_1x"}

    payload = json.loads((output_dir / FIXED_SUBSET_NOISE_SUMMARY_JSON).read_text())
    assert payload["baseline_experiment"] == baseline_manifest["experiment_name"]
    assert payload["control_report"]["status"] == "ok"
