# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fray.cluster import ResourceConfig

import experiments.domain_phase_mix.determinism_analysis as determinism_analysis
import experiments.domain_phase_mix.launch_two_phase_many_qsplit240_fixed_subset_seedpanel_n3 as qsplit240_seedpanel
from experiments.domain_phase_mix.determinism_analysis import (
    CollectManifestResultsConfig,
    CONTROL_RUNS_CSV,
    DETERMINISM_CONTROL_REPORT_JSON,
    FINAL_BPB_STATS_JSON,
    FIT_DATASET_CSV,
    FIT_DATASET_SUMMARY_JSON,
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
    _build_model_size_noise_summary_rows,
    collect_manifest_results,
    create_fit_dataset_export,
    create_fixed_subset_noise_report,
    create_jitter_report,
    create_model_size_noise_report,
    NOISE_SUMMARY_CSV,
    NOISE_SUMMARY_JSON,
    MMLU_NOISE_VS_RUNTIME_300M_6B_PNG,
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
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_fixed_subset_seedpanel_n3 import (
    NAME as QSPLIT240_SEEDPANEL_NAME,
    TRAINER_SEEDS,
    build_run_specs as build_qsplit240_seedpanel_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_fixed_subset_study import (
    EXPERIMENT_BUDGET as REGMIX300M_6B_BUDGET,
    MODEL_FAMILY as REGMIX300M_6B_MODEL_FAMILY,
    build_run_specs as build_regmix300m_6b_run_specs,
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
from experiments.domain_phase_mix.proxy_sweep import regmix_300m_muonh_base, regmix_300m_proxy
from experiments.domain_phase_mix.two_phase_many_observed_runs import (
    CORE_BASELINE_RUN_NAMES,
    ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
    load_original_qsplit240_runs,
    load_original_qsplit240_with_core_baselines,
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


def test_regmix_300m_proxy_matches_expected_domain_mix_settings():
    assert regmix_300m_proxy.max_seq_len == 2048
    assert regmix_300m_proxy.tie_word_embeddings is True
    assert regmix_300m_proxy.scan_layers is True
    assert regmix_300m_proxy.gradient_checkpointing is True
    assert regmix_300m_muonh_base.learning_rate == pytest.approx(0.01)
    assert regmix_300m_muonh_base.adam_lr == pytest.approx(0.002)
    assert regmix_300m_muonh_base.momentum == pytest.approx(0.98)


def test_run_00097_300m_6b_fixed_subset_study_builds_expected_manifest_and_weights():
    run_specs = build_regmix300m_6b_run_specs()

    assert len(run_specs) == 10
    assert [spec.run_id for spec in run_specs] == list(range(10))
    assert [spec.trainer_seed for spec in run_specs] == list(range(SEED_SWEEP_START, SEED_SWEEP_START + 10))
    assert all(spec.run_name == f"regmix300m_6b_trainer_seed_{spec.trainer_seed}" for spec in run_specs)
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(spec.simulated_epoch_subset_seed == SIMULATED_EPOCH_SUBSET_SEED for spec in run_specs)
    assert {spec.model_family for spec in run_specs} == {REGMIX300M_6B_MODEL_FAMILY}
    assert {spec.experiment_budget for spec in run_specs} == {REGMIX300M_6B_BUDGET}
    assert {spec.num_train_steps for spec in run_specs} == {22_888}
    assert all(spec.phase_weights == RUN_00097_PHASE_WEIGHTS for spec in run_specs)


def test_load_original_qsplit240_runs_returns_expected_candidate_set():
    observed_runs = load_original_qsplit240_runs()

    assert len(observed_runs) == 238
    assert {run.source_experiment for run in observed_runs} == {ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT}
    assert observed_runs[0].run_name == "run_00002"
    assert observed_runs[-1].run_name == "run_00239"
    assert observed_runs[0].run_id == 2
    assert observed_runs[-1].run_id == 239
    assert all(run.status == "completed" for run in observed_runs)
    assert all(abs(sum(run.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for run in observed_runs)
    assert all(abs(sum(run.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for run in observed_runs)


def test_load_original_qsplit240_with_core_baselines_returns_expected_candidate_set():
    observed_runs = load_original_qsplit240_with_core_baselines()

    assert len(observed_runs) == 240
    assert tuple(run.run_name for run in observed_runs[:2]) == CORE_BASELINE_RUN_NAMES
    assert observed_runs[0].run_id == 0
    assert observed_runs[1].run_id == 1
    assert observed_runs[2].run_name == "run_00002"
    assert observed_runs[-1].run_name == "run_00239"
    assert {run.source_experiment for run in observed_runs[2:]} == {ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT}
    assert all(run.status == "completed" for run in observed_runs)
    assert all(abs(sum(run.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for run in observed_runs)
    assert all(abs(sum(run.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for run in observed_runs)


def test_qsplit240_seedpanel_builds_expected_manifest():
    run_specs = build_qsplit240_seedpanel_run_specs()

    assert len(run_specs) == 720
    assert [spec.run_id for spec in run_specs] == list(range(720))
    assert {spec.cohort for spec in run_specs} == {"replicated_swarm"}
    assert {spec.trainer_seed for spec in run_specs} == set(TRAINER_SEEDS)
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(spec.simulated_epoch_subset_seed == SIMULATED_EPOCH_SUBSET_SEED for spec in run_specs)
    assert len({spec.run_name for spec in run_specs}) == len(run_specs)
    assert run_specs[0].run_name == "seed10000_baseline_proportional"
    assert run_specs[1].run_name == "seed10001_baseline_proportional"
    assert run_specs[3].run_name == "seed10000_baseline_unimax"
    assert run_specs[-1].run_name == "seed10002_run_00239"

    candidate_counts = pd.Series([spec.candidate_run_name for spec in run_specs]).value_counts()
    assert candidate_counts.nunique() == 1
    assert int(candidate_counts.iloc[0]) == 3


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


def test_collect_manifest_results_preserves_candidate_metadata(tmp_path, monkeypatch):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    manifest = {
        "experiment_name": QSPLIT240_SEEDPANEL_NAME,
        "runs": [
            {
                "run_id": 0,
                "run_name": "seed10000_run_00002",
                "cohort": "replicated_swarm",
                "trainer_seed": 10_000,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "candidate_run_id": 2,
                "candidate_run_name": "run_00002",
                "candidate_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
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
                "wandb_run_id": "seed10000_run_00002-abc123",
                "wandb_run_name": "pinlin_calvin_xu/data_mixture/ngd3dm~12345678/seed10000_run_00002",
                "status": "finished",
                OBJECTIVE_METRIC: 2.11,
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
    assert results_df.loc[0, "candidate_run_id"] == 2
    assert results_df.loc[0, "candidate_run_name"] == "run_00002"
    assert results_df.loc[0, "candidate_source_experiment"] == ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT
    assert results_df.loc[0, "trainer_seed"] == 10_000
    assert results_df.loc[0, "simulated_epoch_subset_seed"] == SIMULATED_EPOCH_SUBSET_SEED


def test_create_fit_dataset_export_writes_sorted_completed_long_csv(tmp_path):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "fit"
    analysis_dir.mkdir()
    output_dir.mkdir()

    manifest = {
        "experiment_name": QSPLIT240_SEEDPANEL_NAME,
        "runs": [
            {
                "run_id": 0,
                "run_name": "seed10000_run_00002",
                "cohort": "replicated_swarm",
                "trainer_seed": 10_000,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "candidate_run_id": 2,
                "candidate_run_name": "run_00002",
                "candidate_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
            {
                "run_id": 1,
                "run_name": "seed10001_run_00002",
                "cohort": "replicated_swarm",
                "trainer_seed": 10_001,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "candidate_run_id": 2,
                "candidate_run_name": "run_00002",
                "candidate_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
            {
                "run_id": 2,
                "run_name": "seed10000_run_00003",
                "cohort": "replicated_swarm",
                "trainer_seed": 10_000,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "candidate_run_id": 3,
                "candidate_run_name": "run_00003",
                "candidate_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest))

    results = pd.DataFrame(
        [
            {
                "wandb_run_id": "wb-2",
                "run_id": 2,
                "run_name": "seed10000_run_00003",
                "status": "completed",
                "candidate_run_id": 3,
                "candidate_run_name": "run_00003",
                "candidate_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                "trainer_seed": 10_000,
                "data_seed": np.nan,
                "simulated_epoch_subset_seed": 97,
                "phase_0_domain_a": 0.4,
                "phase_1_domain_a": 0.6,
                "lm_eval/mmlu_5shot/choice_logprob": -1.50,
            },
            {
                "wandb_run_id": "wb-0",
                "run_id": 0,
                "run_name": "seed10000_run_00002",
                "status": "completed",
                "candidate_run_id": 2,
                "candidate_run_name": "run_00002",
                "candidate_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                "trainer_seed": 10_000,
                "data_seed": np.nan,
                "simulated_epoch_subset_seed": 97,
                "phase_0_domain_a": 0.4,
                "phase_1_domain_a": 0.6,
                "lm_eval/mmlu_5shot/choice_logprob": -1.49,
            },
            {
                "wandb_run_id": "wb-1",
                "run_id": 1,
                "run_name": "seed10001_run_00002",
                "status": "not_found",
                "candidate_run_id": 2,
                "candidate_run_name": "run_00002",
                "candidate_source_experiment": ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
                "trainer_seed": 10_001,
                "data_seed": np.nan,
                "simulated_epoch_subset_seed": 97,
                "phase_0_domain_a": 0.4,
                "phase_1_domain_a": 0.6,
                "lm_eval/mmlu_5shot/choice_logprob": np.nan,
            },
        ]
    )
    results.to_csv(analysis_dir / RESULTS_CSV, index=False)

    create_fit_dataset_export(
        determinism_analysis.FitDatasetExportConfig(
            output_path=str(output_dir),
            run_manifest_path=str(manifest_path),
            analysis_output_path=str(analysis_dir),
        )
    )

    fit_df = pd.read_csv(output_dir / FIT_DATASET_CSV)
    assert list(fit_df["candidate_run_id"]) == [2, 3]
    assert list(fit_df["trainer_seed"]) == [10_000, 10_000]
    assert set(fit_df["status"]) == {"completed"}
    assert fit_df.columns.tolist()[:10] == [
        "wandb_run_id",
        "run_id",
        "run_name",
        "status",
        "candidate_run_id",
        "candidate_run_name",
        "candidate_source_experiment",
        "trainer_seed",
        "data_seed",
        "simulated_epoch_subset_seed",
    ]

    summary = json.loads((output_dir / FIT_DATASET_SUMMARY_JSON).read_text())
    assert summary["total_candidates_expected"] == 2
    assert summary["total_seeds_expected"] == 2
    assert summary["total_rows_expected"] == 3
    assert summary["total_rows_completed"] == 2
    candidate_counts = {
        row["candidate_run_id"]: row["n_completed"] for row in summary["per_candidate_completion_counts"]
    }
    seed_counts = {row["trainer_seed"]: row["n_completed"] for row in summary["per_seed_completion_counts"]}
    assert candidate_counts == {2: 1, 3: 1}
    assert seed_counts == {10_000: 2, 10_001: 0}


def test_qsplit240_seedpanel_main_wires_max_concurrent(monkeypatch):
    captured: dict[str, object] = {}

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        qsplit240_seedpanel,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        qsplit240_seedpanel,
        "create_manifest_results_step",
        lambda **_: SimpleNamespace(name="collect_results"),
    )
    monkeypatch.setattr(
        qsplit240_seedpanel,
        "create_fit_dataset_export_step",
        lambda **_: SimpleNamespace(name="fit_dataset_export"),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(qsplit240_seedpanel, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        qsplit240_seedpanel,
        "create_run_manifest_step",
        lambda **_: SimpleNamespace(name="run_manifest"),
    )
    monkeypatch.setattr(
        qsplit240_seedpanel,
        "build_run_specs",
        lambda: build_qsplit240_seedpanel_run_specs()[:6],
    )
    monkeypatch.setattr(qsplit240_seedpanel.sys, "argv", ["launcher", "--max-concurrent", "17"])

    qsplit240_seedpanel.main()

    assert captured["config"].max_concurrent == 17
    assert len(captured["steps"]) == 9
    assert "replicated swarm" in captured["description"]
    assert qsplit240_seedpanel.DEFAULT_MAX_CONCURRENT == 256


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


def test_build_model_size_noise_summary_rows_computes_dual_baseline_ratios():
    current_df = pd.DataFrame(
        [
            {
                "run_name": "regmix300m_6b_trainer_seed_10000",
                "lm_eval/mmlu_5shot/bpb": 2.11,
                "lm_eval/mmlu_5shot/choice_logprob": -1.44,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.56,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.265,
                "eval/paloma/c4_en/bpb": 1.16,
            },
            {
                "run_name": "regmix300m_6b_trainer_seed_10001",
                "lm_eval/mmlu_5shot/bpb": 2.13,
                "lm_eval/mmlu_5shot/choice_logprob": -1.45,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.57,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.264,
                "eval/paloma/c4_en/bpb": 1.15,
            },
        ]
    )
    fixed_subset_df = pd.DataFrame(
        [
            {
                "run_name": "trainer_seed_10000",
                "lm_eval/mmlu_5shot/bpb": 2.20,
                "lm_eval/mmlu_5shot/choice_logprob": -1.50,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.64,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.260,
                "eval/paloma/c4_en/bpb": 1.18,
            },
            {
                "run_name": "trainer_seed_10001",
                "lm_eval/mmlu_5shot/bpb": 2.24,
                "lm_eval/mmlu_5shot/choice_logprob": -1.52,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.66,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.258,
                "eval/paloma/c4_en/bpb": 1.19,
            },
        ]
    )
    compute_df = pd.DataFrame(
        [
            {
                "run_name": "regmix60m_6b_trainer_seed_10000",
                "lm_eval/mmlu_5shot/bpb": 2.18,
                "lm_eval/mmlu_5shot/choice_logprob": -1.48,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.61,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.262,
                "eval/paloma/c4_en/bpb": 1.17,
            },
            {
                "run_name": "regmix60m_6b_trainer_seed_10001",
                "lm_eval/mmlu_5shot/bpb": 2.22,
                "lm_eval/mmlu_5shot/choice_logprob": -1.50,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.63,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.260,
                "eval/paloma/c4_en/bpb": 1.18,
            },
        ]
    )

    rows = _build_model_size_noise_summary_rows(
        current_runs_df=current_df,
        fixed_subset_baseline_df=fixed_subset_df,
        compute_baseline_df=compute_df,
        runtime_by_cohort={
            "fixed_subset_1x": {
                "active_compute_hours_mean": 0.73,
                "active_compute_hours_std": 0.01,
                "active_compute_hours_n": 10,
            },
            "regmix60m_6b": {
                "active_compute_hours_mean": 3.63,
                "active_compute_hours_std": 0.02,
                "active_compute_hours_n": 10,
            },
            "regmix300m_6b": {
                "active_compute_hours_mean": 4.20,
                "active_compute_hours_std": 0.03,
                "active_compute_hours_n": 10,
            },
        },
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

    assert set(summary_df["cohort"]) == {"fixed_subset_1x", "regmix60m_6b", "regmix300m_6b"}
    current_bpb = summary_df[
        (summary_df["cohort"] == "regmix300m_6b") & (summary_df["metric"] == "lm_eval/mmlu_5shot/bpb")
    ].iloc[0]

    assert current_bpb["active_compute_hours_mean"] == pytest.approx(4.20)
    assert current_bpb["variance_ratio_vs_fixed_subset_1x"] < 1.0
    assert current_bpb["variance_ratio_vs_regmix60m_6b"] < 1.0
    assert np.isfinite(current_bpb["variance_ratio_vs_fixed_subset_1x_ci_low"])
    assert np.isfinite(current_bpb["variance_ratio_vs_regmix60m_6b_ci_high"])


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


def test_create_model_size_noise_report_writes_summary_and_runtime_context(tmp_path, monkeypatch):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    analysis_dir = tmp_path / "analysis"
    output_dir = tmp_path / "report"
    analysis_dir.mkdir()
    output_dir.mkdir()

    manifest = {
        "experiment_name": "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_fixed_subset",
        "runs": [
            {
                "run_id": 0,
                "run_name": "regmix300m_6b_trainer_seed_10000",
                "cohort": "seed_sweep",
                "trainer_seed": 10_000,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "source_run_name": SOURCE_RUN_NAME,
                "experiment_budget": REGMIX300M_6B_BUDGET,
                "num_train_steps": 22_888,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
            {
                "run_id": 1,
                "run_name": "regmix300m_6b_trainer_seed_10001",
                "cohort": "seed_sweep",
                "trainer_seed": 10_001,
                "data_seed": None,
                "simulated_epoch_subset_seed": SIMULATED_EPOCH_SUBSET_SEED,
                "source_run_name": SOURCE_RUN_NAME,
                "experiment_budget": REGMIX300M_6B_BUDGET,
                "num_train_steps": 22_888,
                "phase_weights": RUN_00097_PHASE_WEIGHTS,
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest))

    results = pd.DataFrame(
        [
            {
                "run_id": 0,
                "run_name": "regmix300m_6b_trainer_seed_10000",
                "status": "completed",
                "wandb_run_id": "current-0",
                "lm_eval/mmlu_5shot/bpb": 2.10,
                "lm_eval/mmlu_5shot/choice_logprob": -1.44,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.56,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.265,
                "eval/paloma/c4_en/bpb": 1.16,
                "eval/paloma/macro_bpb": 1.35,
                "eval/uncheatable_eval/macro_bpb": 1.08,
            },
            {
                "run_id": 1,
                "run_name": "regmix300m_6b_trainer_seed_10001",
                "status": "completed",
                "wandb_run_id": "current-1",
                "lm_eval/mmlu_5shot/bpb": 2.12,
                "lm_eval/mmlu_5shot/choice_logprob": -1.45,
                "lm_eval/mmlu_5shot/choice_logprob_norm": -1.57,
                "lm_eval/mmlu_5shot/choice_prob_norm": 0.264,
                "eval/paloma/c4_en/bpb": 1.15,
                "eval/paloma/macro_bpb": 1.34,
                "eval/uncheatable_eval/macro_bpb": 1.07,
            },
        ]
    )
    results.to_csv(analysis_dir / RESULTS_CSV, index=False)

    monkeypatch.setattr(
        determinism_analysis,
        "_collect_trajectory_rows",
        lambda *_, **__: pd.DataFrame(
            [
                {"run_id": 0, "step": 1000, "metric_value": 2.20, "total_tokens": 1.0e8},
                {"run_id": 1, "step": 1000, "metric_value": 2.18, "total_tokens": 1.0e8},
            ]
        ),
    )
    monkeypatch.setattr(
        determinism_analysis,
        "_collect_manifest_results_frame",
        lambda manifest_payload, **_: pd.DataFrame(
            [
                {
                    "run_id": idx,
                    "run_name": run["run_name"],
                    "status": "completed",
                    "wandb_run_id": f"{manifest_payload['experiment_name']}-{idx}",
                    "lm_eval/mmlu_5shot/bpb": 2.20 + 0.02 * idx,
                    "lm_eval/mmlu_5shot/choice_logprob": -1.50 - 0.02 * idx,
                    "lm_eval/mmlu_5shot/choice_logprob_norm": -1.62 - 0.02 * idx,
                    "lm_eval/mmlu_5shot/choice_prob_norm": 0.260 - 0.002 * idx,
                    "eval/paloma/c4_en/bpb": 1.18 + 0.01 * idx,
                    "eval/paloma/macro_bpb": 1.38 + 0.01 * idx,
                    "eval/uncheatable_eval/macro_bpb": 1.10 + 0.01 * idx,
                }
                for idx, run in enumerate(manifest_payload["runs"])
            ]
        ),
    )
    monkeypatch.setattr(
        determinism_analysis,
        "_active_compute_hours_summary",
        lambda run_ids, **_: {
            "active_compute_hours_mean": float(len(run_ids)),
            "active_compute_hours_std": 0.25,
            "active_compute_hours_n": len(run_ids),
        },
    )

    fixed_subset_manifest = {
        "experiment_name": "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_fixed_subset_study",
        "runs": [
            {"run_id": 0, "run_name": "trainer_seed_10000"},
            {"run_id": 1, "run_name": "trainer_seed_10001"},
        ],
    }
    compute_manifest = {
        "experiment_name": "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_compute_scaling",
        "runs": [
            {"run_id": 0, "run_name": "regmix60m_6b_trainer_seed_10000"},
            {"run_id": 1, "run_name": "regmix60m_6b_trainer_seed_10001"},
        ],
    }

    create_model_size_noise_report(
        determinism_analysis.ModelSizeNoiseReportConfig(
            output_path=str(output_dir),
            run_manifest_path=str(manifest_path),
            analysis_output_path=str(analysis_dir),
            wandb_entity="unit",
            wandb_project="unit",
            fixed_subset_baseline_manifest_json=json.dumps(fixed_subset_manifest),
            compute_baseline_manifest_json=json.dumps(compute_manifest),
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

    assert (output_dir / NOISE_SUMMARY_CSV).exists()
    assert (output_dir / NOISE_SUMMARY_JSON).exists()
    assert (output_dir / MMLU_NOISE_VS_RUNTIME_300M_6B_PNG).exists()

    summary = pd.read_csv(output_dir / NOISE_SUMMARY_CSV)
    assert set(summary["cohort"]) == {"fixed_subset_1x", "regmix60m_6b", "regmix300m_6b"}
    current_bpb = summary[(summary["cohort"] == "regmix300m_6b") & (summary["metric"] == "lm_eval/mmlu_5shot/bpb")].iloc[
        0
    ]
    assert current_bpb["active_compute_hours_mean"] == pytest.approx(2.0)
    assert np.isfinite(current_bpb["variance_ratio_vs_fixed_subset_1x"])
    assert np.isfinite(current_bpb["variance_ratio_vs_regmix60m_6b"])

    payload = json.loads((output_dir / NOISE_SUMMARY_JSON).read_text())
    assert payload["fixed_subset_baseline_experiment"] == fixed_subset_manifest["experiment_name"]
    assert payload["compute_baseline_experiment"] == compute_manifest["experiment_name"]
