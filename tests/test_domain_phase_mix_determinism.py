# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fray.cluster import ResourceConfig

import experiments.domain_phase_mix.determinism_analysis as determinism_analysis
from experiments.domain_phase_mix import (
    launch_two_phase_many_genericfamily_subset_optima as genericfamily_subset_optima_launch,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_genericfamily_retuned_subset_optima as genericfamily_retuned_subset_optima_launch,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_genericfamily_recovered_hull_subset_optima as genericfamily_recovered_hull_launch,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_genericfamily_top8actual_hull_subset_optima as genericfamily_top8actual_hull_launch,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_genericfamily_observed_only_trustblend_baseline as obs_trustblend_baseline_launch,
    launch_two_phase_many_genericfamily_power_observed_only_trustblend_baseline as power_obs_trustblend_baseline_launch,
    launch_two_phase_many_genericfamily_observed_only_trustblend_subset_optima as obs_trustblend_subset_launch,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_genericfamily_no_penalty_baseline as genericfamily_no_penalty_launch,
    launch_two_phase_many_genericfamily_penalty_raw_optima_300m_6b as genericfamily_penalty_raw_optima_300m_launch,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_regmix_raw_subset_optima as regmix_raw_subset_optima_launch,
)
import experiments.domain_phase_mix.launch_two_phase_many_genericfamily_power_family_penalty_no_l2_raw_subset_optima as grp_no_l2_raw_subset_optima_launch  # noqa: E501
from experiments.domain_phase_mix import (
    launch_two_phase_many_qsplit240_300m_6b_parity_rerun as qsplit240_300m_parity_rerun,
    launch_two_phase_many_run_00097_300m_6b_parity_rerun as run00097_300m_parity_rerun,
)

from experiments.domain_phase_mix import (
    launch_two_phase_many_olmix_loglinear_subset_optima as olmix_loglinear_subset_optima_launch,
)
import experiments.domain_phase_mix.launch_two_phase_many_qsplit240_fixed_subset_seedpanel_n3 as qsplit240_seedpanel
import experiments.domain_phase_mix.launch_two_phase_many_stratified_baseline as stratified_baseline_launch
import experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b as qsplit240_300m
import experiments.domain_phase_mix.launch_two_phase_many_qsplit240_1_2b_chinchilla_pilot as qsplit240_1_2b_pilot
import experiments.domain_phase_mix.launch_two_phase_many_qsplit240_520m_chinchilla_pilot as qsplit240_520m_pilot
import experiments.domain_phase_mix.launch_two_phase_many_strong_tier_scaling_study as strong_tier_scaling_study_launch
import experiments.domain_phase_mix.qsplit240_replay as qsplit240_replay
import experiments.domain_phase_mix.scaling_study_recipes as scaling_study_recipes
from experiments.domain_phase_mix import (
    launch_two_phase_many_qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_rerun as qsplit240_seedpanel_slverb_rerun,
)
import experiments.domain_phase_mix.exploratory.two_phase_many.analyze_qsplit240_520m_pilot as qsplit240_520m_analysis
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
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import (
    EXPERIMENT_BUDGET as QSPLIT240_300M_6B_BUDGET,
    MODEL_FAMILY as QSPLIT240_300M_6B_MODEL_FAMILY,
    NUM_TRAIN_STEPS as QSPLIT240_300M_6B_NUM_TRAIN_STEPS,
    QSPLIT240_300M_EVAL_TASKS,
    build_run_specs as build_qsplit240_300m_run_specs,
    create_experiment as create_qsplit240_300m_experiment,
    select_run_specs_for_shard,
    shard_execution_name_prefix,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b_parity_rerun import (
    build_run_specs as build_qsplit240_300m_parity_rerun_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_1_2b_chinchilla_pilot import (
    BATCH_SIZE as QSPLIT240_1_2B_BATCH_SIZE,
    DEFAULT_PANEL as QSPLIT240_1_2B_DEFAULT_PANEL,
    DEFAULT_TPU_REGIONS as QSPLIT240_1_2B_DEFAULT_TPU_REGIONS,
    DEFAULT_TPU_ZONE as QSPLIT240_1_2B_DEFAULT_TPU_ZONE,
    EXPERIMENT_BUDGET as QSPLIT240_1_2B_BUDGET,
    MODEL_FAMILY as QSPLIT240_1_2B_MODEL_FAMILY,
    NUM_TRAIN_STEPS as QSPLIT240_1_2B_NUM_TRAIN_STEPS,
    build_run_specs as build_qsplit240_1_2b_pilot_run_specs,
    create_experiment as create_qsplit240_1_2b_pilot_experiment,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_520m_chinchilla_pilot import (
    BATCH_SIZE as QSPLIT240_520M_BATCH_SIZE,
    DEFAULT_PANEL as QSPLIT240_520M_DEFAULT_PANEL,
    DEFAULT_TPU_REGIONS as QSPLIT240_520M_DEFAULT_TPU_REGIONS,
    DEFAULT_TPU_ZONE as QSPLIT240_520M_DEFAULT_TPU_ZONE,
    EXPERIMENT_BUDGET as QSPLIT240_520M_BUDGET,
    MODEL_FAMILY as QSPLIT240_520M_MODEL_FAMILY,
    NUM_TRAIN_STEPS as QSPLIT240_520M_NUM_TRAIN_STEPS,
    build_run_specs as build_qsplit240_520m_pilot_run_specs,
    create_experiment as create_qsplit240_520m_pilot_experiment,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_mmlu_sl_verb_rerun import (
    build_run_specs as build_qsplit240_mmlu_sl_verb_rerun_specs,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_qsplit240_olmo_base_easy_overlap_rerun as qsplit240_olmo_base_easy_overlap_rerun,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_olmo_base_easy_overlap_rerun import (
    build_run_specs as build_qsplit240_olmo_base_easy_overlap_rerun_specs,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_selected_baselines_olmo_base_easy_overlap_rerun as selected_overlap_rerun,
)
from experiments.domain_phase_mix.launch_two_phase_many_selected_baselines_olmo_base_easy_overlap_rerun import (
    build_run_specs as build_selected_baselines_olmo_base_easy_overlap_rerun_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_rerun import (
    build_run_specs as build_qsplit240_seedpanel_mmlu_sl_verb_rerun_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_olmix_loglinear_mmlu_sl_verb_rerun import (
    build_run_specs as build_olmix_loglinear_mmlu_sl_verb_rerun_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_olmix_sl_verb_choice_logprob_norm_mmlu_sl_verb_rerun import (
    build_run_specs as build_fitted_olmix_sl_verb_mmlu_sl_verb_rerun_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_fixed_subset_study import (
    EXPERIMENT_BUDGET as REGMIX300M_6B_BUDGET,
    MODEL_FAMILY as REGMIX300M_6B_MODEL_FAMILY,
    build_run_specs as build_regmix300m_6b_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_300m_6b_parity_rerun import (
    _eval_model_name as run00097_300m_parity_eval_model_name,
    build_run_specs as build_run_00097_300m_parity_rerun_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_mmlu_sl_verb_rerun import (
    build_run_specs as build_run_00097_mmlu_sl_verb_rerun_specs,
)
from experiments.domain_phase_mix import (
    launch_two_phase_many_run_00097_olmo_base_easy_overlap_rerun as run00097_olmo_base_easy_overlap_rerun,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_olmo_base_easy_overlap_rerun import (
    build_run_specs as build_run_00097_olmo_base_easy_overlap_rerun_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_mmlu_sl_verb_rerun import (
    _eval_model_name as fixed_subset_run_00097_mmlu_sl_verb_eval_model_name,
    build_run_specs as build_run_00097_fixed_subset_mmlu_sl_verb_rerun_specs,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    STRATIFIED_RUN_NAME,
    create_stratified_domain_weights,
    create_stratified_weight_config,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_starcoder_determinism_wsd import (
    FIXED_PHASE_WEIGHTS,
    OBJECTIVE_METRIC,
    build_run_specs,
    default_sweep_config,
    resolve_wsd_schedule,
)
from experiments.domain_phase_mix.proxy_sweep import (
    REGMIX_130M_CHINCHILLA_BUDGET,
    REGMIX_300M_CHINCHILLA_BUDGET,
    regmix_130m_muonh_base,
    regmix_130m_proxy,
    regmix_60m_proxy,
    regmix_300m_muonh_base,
    regmix_300m_proxy,
)
from experiments.domain_phase_mix.proxy_sweep import (
    REGMIX_1_2B_CHINCHILLA_BUDGET,
    regmix_1_2b_muonh_base,
    regmix_1_2b_proxy,
    REGMIX_520M_CHINCHILLA_BUDGET,
    regmix_520m_muonh_base,
    regmix_520m_proxy,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import (
    OLMIX_LOGLINEAR_PHASE_WEIGHTS,
    OLMIX_LOGLINEAR_RUN_NAME,
    OLMIX_LOGLINEAR_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.two_phase_many_regmix_raw_subset_optima import (
    REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
    regmix_raw_subset_optima_summaries,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_power_family_penalty_no_l2_raw_subset_optima import (
    GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
    genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_subset_optima import (
    OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES,
    olmix_loglinear_subset_optima_summaries,
)
from experiments.domain_phase_mix.two_phase_many_power_ridge_single import (
    POWER_RIDGE_SINGLE_OBJECTIVE_METRIC,
    POWER_RIDGE_SINGLE_RUN_NAME,
    POWER_RIDGE_SINGLE_SOURCE_EXPERIMENT,
    create_power_ridge_single_weight_config,
    power_ridge_single_summary,
)
from experiments.domain_phase_mix.two_phase_many_dsre_predicted_baselines import (
    DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME,
    DSRE_CEQ_PREDICTED_RUN_NAME,
    DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT,
    create_dsre_ceq_predicted_quality_collapsed_weight_config,
    create_dsre_ceq_predicted_weight_config,
    dsre_ceq_predicted_quality_collapsed_summary,
    dsre_ceq_predicted_summary,
)
from experiments.domain_phase_mix.two_phase_many_dsre_predicted_topic_collapsed import (
    DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME,
    DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_SOURCE_EXPERIMENT,
    create_dsre_ceq_predicted_topic_collapsed_weight_config,
    dsre_ceq_predicted_topic_collapsed_summary,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level_topic_collapsed import (
    build_top_level_domains_with_collapsed_cc_topics,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_uncheatable import (
    OBJECTIVE_METRIC as OLMIX_UNCHEATABLE_OBJECTIVE_METRIC,
    fit_olmix_uncheatable_from_frame,
)
from experiments.domain_phase_mix.two_phase_many_thresholdtotal_overfit import (
    THRESHOLDTOTAL_OVERFIT_OBJECTIVE_METRIC,
    THRESHOLDTOTAL_OVERFIT_RUN_NAME,
    THRESHOLDTOTAL_OVERFIT_SOURCE_EXPERIMENT,
    create_thresholdtotal_overfit_weight_config,
    thresholdtotal_overfit_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME,
    CCGLOBALPREMIUM_SOURCE_EXPERIMENT,
    CCGLOBALPREMIUM_THRESHOLD_RUN_NAME,
    ccglobalpremium_retainedtotal_summary,
    ccglobalpremium_threshold_summary,
    create_ccglobalpremium_retainedtotal_weight_config,
    create_ccglobalpremium_threshold_weight_config,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME,
    CCPAIRTOTAL_RETAINEDTOTAL_SOURCE_EXPERIMENT,
    ccpairtotal_retainedtotal_summary,
    create_ccpairtotal_retainedtotal_weight_config,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline import (
    GENERICFAMILY_TUNED_RUN_NAME,
    GENERICFAMILY_TUNED_SOURCE_EXPERIMENT,
    create_genericfamily_tuned_weight_config,
    genericfamily_tuned_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_subset_optima import (
    GENERICFAMILY_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_subset_optimum_run_name,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_retuned_subset_optimum_run_name,
    parse_subset_sizes as parse_retuned_subset_sizes,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_recovered_hull_subset_optima import (
    GENERICFAMILY_RECOVERED_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_recovered_hull_subset_optimum_run_name,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_top8actual_hull_subset_optima import (
    GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_top8actual_hull_subset_optimum_run_name,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_observed_only_trustblend_subset_optimum_run_name,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_baseline import (
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME,
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_power_observed_only_trustblend_baseline import (
    GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME,
    GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_no_penalty_baseline import (
    GENERICFAMILY_NO_PENALTY_RUN_NAME,
    GENERICFAMILY_NO_PENALTY_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import (
    NEGATED_OBJECTIVE_METRIC,
    OBJECTIVE_METRIC as OLMIX_SL_VERB_OBJECTIVE_METRIC,
    aggregate_candidate_mean_frame,
    fit_olmix_sl_verb_from_frame,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import (
    CORE_BASELINE_RUN_NAMES,
    ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT,
    REPRESENTATIVE12_PANEL_RUN_NAMES,
    load_original_qsplit240_named_panel,
    load_original_qsplit240_runs,
    load_original_qsplit240_with_core_baselines,
)
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import Executor, InputName, collect_dependencies_and_version
from marin.processing.tokenize import TokenizeConfig
from rigging.filesystem import marin_prefix


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
    monkeypatch.setattr(
        "experiments.defaults._prepare_data_config",
        lambda tokenized, use_default_validation: SimpleNamespace(tokenizer="unit-test-tokenizer"),
    )
    monkeypatch.setattr("experiments.defaults._validate_train_length", lambda train_seq_len, model_config: 2048)

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


def test_run_00097_300m_parity_rerun_builds_expected_manifest():
    run_specs = build_run_00097_300m_parity_rerun_specs()

    assert len(run_specs) == 10
    assert [spec.run_id for spec in run_specs] == list(range(10))
    assert {spec.cohort for spec in run_specs} == {"seed_sweep"}
    assert [spec.trainer_seed for spec in run_specs] == list(range(SEED_SWEEP_START, SEED_SWEEP_START + 10))
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(
        spec.source_experiment == "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_fixed_subset"
        for spec in run_specs
    )
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert {spec.num_train_steps for spec in run_specs} == {22_888}


def test_run_00097_300m_parity_rerun_namespaces_eval_model_names():
    eval_model_name = run00097_300m_parity_eval_model_name(
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_parity_rerun",
        run_name="regmix300m_6b_trainer_seed_10000",
    )

    assert eval_model_name.startswith("pinlin_calvin_xu__data_mixture__ngd3dm2_run00097_300m_6b_parity_rerun__")
    assert eval_model_name.endswith("regmix300m_6b_trainer_seed_10000")
    assert eval_model_name != "regmix300m_6b_trainer_seed_10000"


def test_run_00097_mmlu_sl_verb_rerun_builds_expected_manifest():
    run_specs = build_run_00097_mmlu_sl_verb_rerun_specs()

    assert len(run_specs) == 10
    assert [spec.run_id for spec in run_specs] == list(range(10))
    assert {spec.cohort for spec in run_specs} == {"seed_sweep"}
    assert [spec.run_name for spec in run_specs] == [f"trainer_seed_{seed}" for seed in range(10_000, 10_010)]
    assert [spec.trainer_seed for spec in run_specs] == list(range(10_000, 10_010))
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(spec.source_run_name == SOURCE_RUN_NAME for spec in run_specs)
    assert all(
        spec.source_experiment == "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study" for spec in run_specs
    )
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert all(spec.phase_weights == RUN_00097_PHASE_WEIGHTS for spec in run_specs)


def test_run_00097_olmo_base_easy_overlap_rerun_builds_expected_manifest():
    run_specs = build_run_00097_olmo_base_easy_overlap_rerun_specs()

    assert len(run_specs) == 10
    assert [spec.run_id for spec in run_specs] == list(range(10))
    assert {spec.cohort for spec in run_specs} == {"seed_sweep"}
    assert [spec.run_name for spec in run_specs] == [f"trainer_seed_{seed}" for seed in range(10_000, 10_010)]
    assert [spec.trainer_seed for spec in run_specs] == list(range(10_000, 10_010))
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(spec.source_run_name == SOURCE_RUN_NAME for spec in run_specs)
    assert all(
        spec.source_experiment == "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study" for spec in run_specs
    )
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert all(spec.phase_weights == RUN_00097_PHASE_WEIGHTS for spec in run_specs)


def test_run_00097_fixed_subset_mmlu_sl_verb_rerun_builds_expected_manifest():
    run_specs = build_run_00097_fixed_subset_mmlu_sl_verb_rerun_specs()

    assert len(run_specs) == 10
    assert [spec.run_id for spec in run_specs] == list(range(10))
    assert {spec.cohort for spec in run_specs} == {"seed_sweep"}
    assert [spec.run_name for spec in run_specs] == [f"trainer_seed_{seed}" for seed in range(10_000, 10_010)]
    assert [spec.trainer_seed for spec in run_specs] == list(range(10_000, 10_010))
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(spec.simulated_epoch_subset_seed == SIMULATED_EPOCH_SUBSET_SEED for spec in run_specs)
    assert all(spec.source_run_name == SOURCE_RUN_NAME for spec in run_specs)
    assert all(
        spec.source_experiment == "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_fixed_subset_study"
        for spec in run_specs
    )
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert all(spec.phase_weights == RUN_00097_PHASE_WEIGHTS for spec in run_specs)


def test_run_00097_fixed_subset_mmlu_sl_verb_rerun_namespaces_eval_model_names():
    eval_model_name = fixed_subset_run_00097_mmlu_sl_verb_eval_model_name(
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_fixed_subset_study_mmlu_sl_verb_rerun",
        run_name="trainer_seed_10000",
    )

    assert eval_model_name.startswith(
        "pinlin_calvin_xu__data_mixture__ngd3dm2_run00097_fixed_subset_study_mmlu_sl_verb_rerun__"
    )
    assert eval_model_name.endswith("trainer_seed_10000")
    assert eval_model_name != "trainer_seed_10000"


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


def test_two_phase_many_experiment_allows_eval_task_override():
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name="unit-test",
        eval_harness_tasks=QSPLIT240_300M_EVAL_TASKS,
    )

    assert tuple(task.task_alias for task in experiment.eval_harness_tasks) == tuple(
        task.task_alias for task in QSPLIT240_300M_EVAL_TASKS
    )


def test_qsplit240_300m_6b_builds_expected_manifest_and_weights():
    run_specs = build_qsplit240_300m_run_specs()
    observed_runs = load_original_qsplit240_with_core_baselines()
    observed_by_name = {run.run_name: run for run in observed_runs}

    assert len(run_specs) == 240
    assert [spec.run_id for spec in run_specs] == list(range(240))
    assert {spec.cohort for spec in run_specs} == {"original_swarm_300m"}
    assert run_specs[0].run_name == "baseline_proportional"
    assert run_specs[1].run_name == "baseline_unimax"
    assert run_specs[2].run_name == "run_00002"
    assert run_specs[-1].run_name == "run_00239"
    assert {spec.model_family for spec in run_specs} == {QSPLIT240_300M_6B_MODEL_FAMILY}
    assert {spec.experiment_budget for spec in run_specs} == {QSPLIT240_300M_6B_BUDGET}
    assert {spec.target_budget for spec in run_specs} == {qsplit240_replay.DEFAULT_TARGET_BUDGET}
    assert {spec.target_budget_multiplier for spec in run_specs} == {qsplit240_replay.DEFAULT_TARGET_BUDGET_MULTIPLIER}
    assert {spec.num_train_steps for spec in run_specs} == {QSPLIT240_300M_6B_NUM_TRAIN_STEPS}
    assert all(spec.trainer_seed is None for spec in run_specs)
    assert all(spec.data_seed == spec.run_id for spec in run_specs)
    assert all(spec.simulated_epoch_subset_seed is None for spec in run_specs)
    assert all(spec.candidate_run_id == spec.run_id for spec in run_specs)
    assert all(spec.candidate_run_name == spec.run_name for spec in run_specs)
    assert all(
        spec.candidate_source_experiment == observed_by_name[spec.run_name].source_experiment for spec in run_specs
    )
    assert all(spec.phase_weights == observed_by_name[spec.run_name].phase_weights for spec in run_specs)


def test_qsplit240_300m_parity_rerun_builds_expected_manifest():
    run_specs = build_qsplit240_300m_parity_rerun_specs(panel="baselines3")

    assert len(run_specs) == 3
    assert [spec.run_name for spec in run_specs] == [
        "baseline_proportional",
        "baseline_unimax",
        "baseline_olmix_loglinear_uncheatable_bpb",
    ]
    assert {spec.cohort for spec in run_specs} == {"original_swarm_300m_parity_rerun"}
    assert {spec.source_experiment for spec in run_specs} == {qsplit240_300m.NAME}
    assert {spec.num_train_steps for spec in run_specs} == {QSPLIT240_300M_6B_NUM_TRAIN_STEPS}
    assert all(spec.checkpoint_root is None for spec in run_specs)


def test_qsplit240_300m_parity_rerun_resolve_checkpoint_roots_skips_unfinished(monkeypatch):
    run_specs = build_qsplit240_300m_parity_rerun_specs(panel="baselines3")

    def fake_resolve_completed_checkpoint_root(*, run_name: str, **_: object) -> str | None:
        if run_name == "baseline_unimax":
            return None
        return f"gs://unit-test/checkpoints/{run_name}"

    monkeypatch.setattr(
        qsplit240_300m_parity_rerun,
        "resolve_completed_checkpoint_root",
        fake_resolve_completed_checkpoint_root,
    )

    resolved = qsplit240_300m_parity_rerun.resolve_checkpoint_roots(run_specs, checkpoint_regions=("us-east5",))

    assert [spec.run_name for spec in resolved] == [
        "baseline_proportional",
        "baseline_olmix_loglinear_uncheatable_bpb",
    ]
    assert all(spec.checkpoint_root is not None for spec in resolved)


def test_genericfamily_penalty_raw_optimum_300m_builds_expected_manifest_and_weights():
    run_specs = genericfamily_penalty_raw_optima_300m_launch.build_run_specs(variants=("power_family_penalty",))

    assert len(run_specs) == 1
    run_spec = run_specs[0]
    assert run_spec.run_id == 412
    assert run_spec.run_name == "baseline_genericfamily_power_family_penalty_raw_optimum_300m_6b"
    assert run_spec.cohort == "grp_penalty_raw_optimum_300m_6b"
    assert run_spec.model_family == QSPLIT240_300M_6B_MODEL_FAMILY
    assert run_spec.experiment_budget == QSPLIT240_300M_6B_BUDGET
    assert run_spec.target_budget == qsplit240_replay.DEFAULT_TARGET_BUDGET
    assert run_spec.target_budget_multiplier == qsplit240_replay.DEFAULT_TARGET_BUDGET_MULTIPLIER
    assert run_spec.num_train_steps == QSPLIT240_300M_6B_NUM_TRAIN_STEPS
    assert run_spec.trainer_seed is None
    assert run_spec.data_seed == 0
    assert run_spec.simulated_epoch_subset_seed is None
    assert run_spec.candidate_run_id == 412
    assert run_spec.candidate_run_name == "baseline_genericfamily_power_family_penalty_raw_optimum"
    assert (
        run_spec.candidate_source_experiment
        == "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_uncheatable_bpb"
    )
    assert abs(sum(run_spec.phase_weights["phase_0"].values()) - 1.0) < 1e-12
    assert abs(sum(run_spec.phase_weights["phase_1"].values()) - 1.0) < 1e-12


def test_regmix_raw_subset_optima_cover_full_convergence_schedule():
    summaries = regmix_raw_subset_optima_summaries()

    assert regmix_raw_subset_optima_launch.parse_subset_sizes("all") == REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES
    assert tuple(summary.subset_size for summary in summaries) == REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES
    assert summaries[0].run_id == 450
    assert summaries[-1].run_id == 458
    assert summaries[0].run_name == "baseline_regmix_raw_optimum_k020_uncheatable_bpb"
    assert summaries[-1].run_name == "baseline_regmix_raw_optimum_k242_uncheatable_bpb"
    assert all(abs(sum(summary.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for summary in summaries)
    assert all(abs(sum(summary.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for summary in summaries)


def test_grp_no_l2_raw_subset_optima_cover_full_convergence_schedule():
    summaries = genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries()

    assert (
        grp_no_l2_raw_subset_optima_launch.parse_subset_sizes("all")
        == GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES
    )
    assert tuple(summary.subset_size for summary in summaries) == (
        GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES
    )
    assert summaries[0].run_id == 470
    assert summaries[-1].run_id == 478
    assert summaries[0].run_name == "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_k020_uncheatable_bpb"
    assert summaries[-1].run_name == "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_k242_uncheatable_bpb"
    assert all(abs(sum(summary.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for summary in summaries)
    assert all(abs(sum(summary.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for summary in summaries)


def test_olmix_loglinear_subset_optima_cover_full_convergence_schedule():
    summaries = olmix_loglinear_subset_optima_summaries()

    assert olmix_loglinear_subset_optima_launch.parse_subset_sizes("all") == OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES
    assert tuple(summary.subset_size for summary in summaries) == OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES
    assert summaries[0].run_id == 460
    assert summaries[-1].run_id == 468
    assert summaries[0].run_name == "baseline_olmix_loglinear_optimum_k020_uncheatable_bpb"
    assert summaries[-1].run_name == "baseline_olmix_loglinear_optimum_k242_uncheatable_bpb"
    assert all(abs(sum(summary.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for summary in summaries)
    assert all(abs(sum(summary.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for summary in summaries)


def test_load_original_qsplit240_named_panel_returns_expected_runs():
    panel_runs = load_original_qsplit240_named_panel(REPRESENTATIVE12_PANEL_RUN_NAMES)

    assert [run.run_name for run in panel_runs] == list(REPRESENTATIVE12_PANEL_RUN_NAMES)
    assert len(panel_runs) == 12
    assert all(abs(sum(run.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for run in panel_runs)
    assert all(abs(sum(run.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for run in panel_runs)


def test_qsplit240_300m_6b_experiment_uses_expected_model_resources_and_tasks():
    experiment = create_qsplit240_300m_experiment(name=qsplit240_300m.NAME, tpu_type="v5p-8")
    stack_edu_domain = next(domain for domain in experiment.domains if domain.name == "dolma3_stack_edu")

    assert experiment.name == qsplit240_300m.NAME
    assert experiment.model_config is regmix_300m_proxy
    assert experiment.num_train_steps == QSPLIT240_300M_6B_NUM_TRAIN_STEPS
    assert experiment.experiment_budget == QSPLIT240_300M_6B_NUM_TRAIN_STEPS * experiment.tokens_per_step
    assert experiment.target_budget == qsplit240_replay.DEFAULT_TARGET_BUDGET
    assert experiment.optimizer_config.learning_rate == pytest.approx(regmix_300m_muonh_base.learning_rate)
    assert experiment.optimizer_config.adam_lr == pytest.approx(regmix_300m_muonh_base.adam_lr)
    assert experiment.optimizer_config.momentum == pytest.approx(regmix_300m_muonh_base.momentum)
    assert list(experiment.resources.regions or []) == ["us-east5"]
    assert experiment.resources.zone == "us-east5-a"
    assert "existing finished us-east5 merged cache" in stack_edu_domain.description
    assert tuple(task.task_alias for task in experiment.eval_harness_tasks) == tuple(
        task.task_alias for task in QSPLIT240_300M_EVAL_TASKS
    )


def test_qsplit240_300m_6b_sharding_uses_contiguous_prefixes_and_keeps_training_prefix_stable():
    run_specs = build_qsplit240_300m_run_specs()

    shard0 = select_run_specs_for_shard(run_specs, shard_index=0, shard_count=8)
    shard1 = select_run_specs_for_shard(run_specs, shard_index=1, shard_count=8)
    shard7 = select_run_specs_for_shard(run_specs, shard_index=7, shard_count=8)

    assert len(shard0) == 30
    assert len(shard1) == 30
    assert len(shard7) == 30
    assert [spec.run_id for spec in shard0] == list(range(30))
    assert [spec.run_id for spec in shard1] == list(range(30, 60))
    assert [spec.run_id for spec in shard7] == list(range(210, 240))
    assert shard_execution_name_prefix(name_prefix=qsplit240_300m.NAME, shard_index=2, shard_count=8) == (
        f"{qsplit240_300m.NAME}/shard_03of08"
    )
    assert shard_execution_name_prefix(name_prefix=qsplit240_300m.NAME, shard_index=0, shard_count=1) == (
        qsplit240_300m.NAME
    )


def test_qsplit240_300m_6b_training_step_blocks_on_eval_dataset_cache():
    experiment = create_qsplit240_300m_experiment(name=qsplit240_300m.NAME, tpu_type="v5p-8")
    run_spec = build_qsplit240_300m_run_specs()[0]
    cache_step = create_cache_eval_datasets_step(
        eval_tasks=QSPLIT240_300M_EVAL_TASKS,
        gcs_path=qsplit240_300m.EVAL_DATASETS_CACHE_PATH,
        name_prefix=qsplit240_300m.NAME,
    )
    training_step = experiment.create_training_step(
        weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
        name_prefix=qsplit240_300m.NAME,
        run_name=run_spec.run_name,
        data_seed=run_spec.data_seed,
    )
    base_step_executor = Executor(
        prefix=marin_prefix(),
        executor_info_base_path=f"{marin_prefix()}/experiments",
    )
    base_step_executor.compute_version(training_step, is_pseudo_dep=False)
    original_output_path = base_step_executor.output_paths[training_step]

    dependent_step = qsplit240_300m.add_eval_cache_dependency_to_training_step(training_step, cache_step)

    dep_marker = dependent_step.config.env_vars[qsplit240_300m.EVAL_DATASETS_CACHE_DEP_ENV_VAR]
    assert isinstance(dep_marker, InputName)
    assert dep_marker.step is cache_step
    assert dependent_step.override_output_path == original_output_path

    deps = collect_dependencies_and_version(dependent_step.config)
    assert cache_step in deps.dependencies


def test_qsplit240_300m_parity_rerun_main_wires_cache_and_max_concurrent(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        qsplit240_300m_parity_rerun,
        "resolve_checkpoint_roots",
        lambda run_specs, checkpoint_regions=("us-east5",): [
            replace(spec, checkpoint_root=f"gs://unit-test/checkpoints/{spec.run_name}") for spec in run_specs
        ],
    )
    monkeypatch.setattr(
        qsplit240_300m_parity_rerun,
        "evaluate_levanter_lm_evaluation_harness",
        lambda **kwargs: SimpleNamespace(name=kwargs["model_name"]),
    )
    monkeypatch.setattr(
        qsplit240_300m_parity_rerun,
        "output_path_of",
        lambda step, filename=None: (
            f"gs://unit-test/{step.name}" if filename is None else f"gs://unit-test/{step.name}/{filename}"
        ),
    )
    monkeypatch.setattr(
        qsplit240_300m_parity_rerun,
        "create_run_manifest_step",
        lambda **_: SimpleNamespace(name="run_manifest"),
    )
    monkeypatch.setattr(
        qsplit240_300m_parity_rerun,
        "create_cache_eval_datasets_step",
        lambda **_: SimpleNamespace(name="cache_eval_datasets"),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(qsplit240_300m_parity_rerun, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        qsplit240_300m_parity_rerun,
        "build_run_specs",
        lambda panel="all": build_qsplit240_300m_parity_rerun_specs(panel="baselines3"),
    )
    monkeypatch.setattr(
        qsplit240_300m_parity_rerun.sys,
        "argv",
        ["launcher", "--panel", "baselines3", "--max-concurrent", "5", "--shard-count", "3", "--shard-index", "1"],
    )

    qsplit240_300m_parity_rerun.main()

    assert captured["config"].max_concurrent == 5
    assert len(captured["steps"]) == 4
    assert "shard_02of03" in captured["description"]


def test_qsplit240_520m_pilot_builds_expected_manifest_and_weights():
    run_specs = build_qsplit240_520m_pilot_run_specs()
    panel_runs = load_original_qsplit240_named_panel(REPRESENTATIVE12_PANEL_RUN_NAMES)
    panel_by_name = {run.run_name: run for run in panel_runs}

    assert len(run_specs) == 12
    assert [spec.run_name for spec in run_specs] == list(REPRESENTATIVE12_PANEL_RUN_NAMES)
    assert [spec.run_id for spec in run_specs] == [run.run_id for run in panel_runs]
    assert {spec.cohort for spec in run_specs} == {"original_swarm_520m_chinchilla_pilot"}
    assert {spec.model_family for spec in run_specs} == {QSPLIT240_520M_MODEL_FAMILY}
    assert {spec.experiment_budget for spec in run_specs} == {QSPLIT240_520M_BUDGET}
    assert {spec.target_budget for spec in run_specs} == {qsplit240_replay.DEFAULT_TARGET_BUDGET}
    assert {spec.target_budget_multiplier for spec in run_specs} == {qsplit240_replay.DEFAULT_TARGET_BUDGET_MULTIPLIER}
    assert {spec.num_train_steps for spec in run_specs} == {QSPLIT240_520M_NUM_TRAIN_STEPS}
    assert QSPLIT240_520M_NUM_TRAIN_STEPS == REGMIX_520M_CHINCHILLA_BUDGET // (QSPLIT240_520M_BATCH_SIZE * 2048)
    assert all(spec.candidate_run_name == spec.run_name for spec in run_specs)
    assert all(spec.phase_weights == panel_by_name[spec.run_name].phase_weights for spec in run_specs)
    assert QSPLIT240_520M_DEFAULT_PANEL == "representative12"


def test_qsplit240_520m_pilot_experiment_uses_expected_model_resources_region_and_tasks():
    experiment = create_qsplit240_520m_pilot_experiment(
        name=qsplit240_520m_pilot.NAME,
        tpu_type="v5p-32",
        tpu_regions=QSPLIT240_520M_DEFAULT_TPU_REGIONS,
        tpu_zone=None,
    )
    stack_edu_domain = next(domain for domain in experiment.domains if domain.name == "dolma3_stack_edu")
    tokenized_step = next(
        step
        for domain in experiment.domains
        for component in domain.components
        if (
            hasattr((step := component.get_step()), "config")
            and isinstance(step.config, TokenizeConfig)
            and step.name.startswith(("tokenized/dolma3_pool/", "tokenized/dolma3_dolmino_pool/"))
        )
    )

    assert experiment.name == qsplit240_520m_pilot.NAME
    assert experiment.model_config is regmix_520m_proxy
    assert experiment.batch_size == QSPLIT240_520M_BATCH_SIZE
    assert experiment.seq_len == qsplit240_520m_pilot.SEQ_LEN
    assert experiment.num_train_steps == QSPLIT240_520M_NUM_TRAIN_STEPS
    assert experiment.experiment_budget == QSPLIT240_520M_NUM_TRAIN_STEPS * experiment.tokens_per_step
    assert experiment.target_budget == qsplit240_replay.DEFAULT_TARGET_BUDGET
    assert experiment.optimizer_config.learning_rate == pytest.approx(regmix_520m_muonh_base.learning_rate)
    assert experiment.optimizer_config.adam_lr == pytest.approx(regmix_520m_muonh_base.adam_lr)
    assert experiment.optimizer_config.momentum == pytest.approx(regmix_520m_muonh_base.momentum)
    assert list(experiment.resources.regions or []) == list(QSPLIT240_520M_DEFAULT_TPU_REGIONS)
    assert experiment.resources.zone == QSPLIT240_520M_DEFAULT_TPU_ZONE
    assert list(tokenized_step.config.worker_resources.regions or []) == list(QSPLIT240_520M_DEFAULT_TPU_REGIONS)
    assert "mirror://" in stack_edu_domain.description
    assert stack_edu_domain.components[0].get_step().cache_path == (
        "mirror://tokenized/merged/dolma3_dolmino_top_level/dolma3_stack_edu-a7297b"
    )


def test_qsplit240_1_2b_pilot_builds_expected_manifest_and_weights():
    run_specs = build_qsplit240_1_2b_pilot_run_specs()
    panel_runs = qsplit240_replay.load_panel_observed_runs(qsplit240_replay.BASELINES3_PANEL)
    panel_by_name = {run.run_name: run for run in panel_runs}

    assert len(run_specs) == 3
    assert [spec.run_name for spec in run_specs] == [
        "baseline_proportional",
        "baseline_unimax",
        "baseline_olmix_loglinear_uncheatable_bpb",
    ]
    assert [spec.run_id for spec in run_specs] == [run.run_id for run in panel_runs]
    assert {spec.cohort for spec in run_specs} == {"original_swarm_1_2b_chinchilla_pilot"}
    assert {spec.model_family for spec in run_specs} == {QSPLIT240_1_2B_MODEL_FAMILY}
    assert {spec.experiment_budget for spec in run_specs} == {QSPLIT240_1_2B_BUDGET}
    assert {spec.target_budget for spec in run_specs} == {qsplit240_replay.DEFAULT_TARGET_BUDGET}
    assert {spec.target_budget_multiplier for spec in run_specs} == {qsplit240_replay.DEFAULT_TARGET_BUDGET_MULTIPLIER}
    assert {spec.num_train_steps for spec in run_specs} == {QSPLIT240_1_2B_NUM_TRAIN_STEPS}
    assert QSPLIT240_1_2B_NUM_TRAIN_STEPS == REGMIX_1_2B_CHINCHILLA_BUDGET // (QSPLIT240_1_2B_BATCH_SIZE * 2048)
    assert all(spec.candidate_run_name == spec.run_name for spec in run_specs)
    assert all(spec.phase_weights == panel_by_name[spec.run_name].phase_weights for spec in run_specs)
    assert QSPLIT240_1_2B_DEFAULT_PANEL == qsplit240_replay.BASELINES3_PANEL


def test_qsplit240_1_2b_pilot_experiment_uses_expected_model_resources_region_and_tasks():
    experiment = create_qsplit240_1_2b_pilot_experiment(
        name=qsplit240_1_2b_pilot.NAME,
        tpu_type="v5p-64",
        tpu_regions=QSPLIT240_1_2B_DEFAULT_TPU_REGIONS,
        tpu_zone=None,
    )
    stack_edu_domain = next(domain for domain in experiment.domains if domain.name == "dolma3_stack_edu")
    tokenized_step = next(
        step
        for domain in experiment.domains
        for component in domain.components
        if (
            hasattr((step := component.get_step()), "config")
            and isinstance(step.config, TokenizeConfig)
            and step.name.startswith(("tokenized/dolma3_pool/", "tokenized/dolma3_dolmino_pool/"))
        )
    )

    assert experiment.name == qsplit240_1_2b_pilot.NAME
    assert experiment.model_config is regmix_1_2b_proxy
    assert experiment.batch_size == QSPLIT240_1_2B_BATCH_SIZE
    assert experiment.seq_len == qsplit240_1_2b_pilot.SEQ_LEN
    assert experiment.num_train_steps == QSPLIT240_1_2B_NUM_TRAIN_STEPS
    assert experiment.experiment_budget == QSPLIT240_1_2B_NUM_TRAIN_STEPS * experiment.tokens_per_step
    assert experiment.target_budget == qsplit240_replay.DEFAULT_TARGET_BUDGET
    assert experiment.optimizer_config.learning_rate == pytest.approx(regmix_1_2b_muonh_base.learning_rate)
    assert experiment.optimizer_config.adam_lr == pytest.approx(regmix_1_2b_muonh_base.adam_lr)
    assert experiment.optimizer_config.momentum == pytest.approx(regmix_1_2b_muonh_base.momentum)
    assert experiment.optimizer_config.max_grad_norm == pytest.approx(regmix_1_2b_muonh_base.max_grad_norm)
    assert list(experiment.resources.regions or []) == list(QSPLIT240_1_2B_DEFAULT_TPU_REGIONS)
    assert experiment.resources.zone == QSPLIT240_1_2B_DEFAULT_TPU_ZONE
    assert list(tokenized_step.config.worker_resources.regions or []) == list(QSPLIT240_1_2B_DEFAULT_TPU_REGIONS)
    assert "mirror://" in stack_edu_domain.description
    assert stack_edu_domain.components[0].get_step().cache_path == (
        "mirror://tokenized/merged/dolma3_dolmino_top_level/dolma3_stack_edu-a7297b"
    )


def test_stratified_weight_config_is_uniform_across_domains_and_phases():
    domain_weights = create_stratified_domain_weights()
    weight_config = create_stratified_weight_config()

    assert set(domain_weights) == set(weight_config.phase_weights["phase_0"])
    assert sum(domain_weights.values()) == pytest.approx(1.0)
    assert len(set(domain_weights.values())) == 1
    assert weight_config.phase_weights["phase_0"] == weight_config.phase_weights["phase_1"]
    assert weight_config.run_id == 3


def test_regmix_130m_proxy_matches_expected_domain_mix_defaults():
    assert regmix_130m_proxy.max_seq_len == 2048
    assert regmix_130m_proxy.tie_word_embeddings is True
    assert regmix_130m_proxy.scan_layers is True
    assert regmix_130m_proxy.gradient_checkpointing is True
    assert REGMIX_130M_CHINCHILLA_BUDGET == 2_600_000_000
    assert regmix_130m_muonh_base.learning_rate == pytest.approx(0.02)
    assert regmix_130m_muonh_base.adam_lr == pytest.approx(0.008)
    assert regmix_130m_muonh_base.momentum == pytest.approx(0.95)
    assert regmix_130m_muonh_base.max_grad_norm == pytest.approx(1.0)


def test_stratified_scale_specs_match_expected_model_and_resource_defaults():
    spec_60m = stratified_baseline_launch.resolve_scale_spec(stratified_baseline_launch.StratifiedScale.REGMIX_60M_1P2B)
    spec_130m = stratified_baseline_launch.resolve_scale_spec(
        stratified_baseline_launch.StratifiedScale.REGMIX_130M_2P6B
    )
    spec_300m = stratified_baseline_launch.resolve_scale_spec(stratified_baseline_launch.StratifiedScale.REGMIX_300M_6B)
    spec_520m = stratified_baseline_launch.resolve_scale_spec(
        stratified_baseline_launch.StratifiedScale.REGMIX_520M_10P4B
    )
    spec_1_2b = stratified_baseline_launch.resolve_scale_spec(stratified_baseline_launch.StratifiedScale.REGMIX_1_2B_24B)

    assert spec_60m.model_config is regmix_60m_proxy
    assert spec_60m.experiment_budget == 1_200_000_000
    assert spec_60m.tpu_type == "v5p-8"
    assert spec_60m.tpu_regions == ("us-east5", "us-central1")
    assert spec_130m.model_config is regmix_130m_proxy
    assert spec_130m.experiment_budget == REGMIX_130M_CHINCHILLA_BUDGET
    assert spec_130m.optimizer_config is regmix_130m_muonh_base
    assert spec_130m.tpu_type == "v5p-8"
    assert spec_130m.tpu_regions == ("us-east5", "us-central1")
    assert spec_300m.model_config is regmix_300m_proxy
    assert spec_300m.experiment_budget == REGMIX_300M_CHINCHILLA_BUDGET
    assert spec_300m.optimizer_config is regmix_300m_muonh_base
    assert spec_300m.tpu_regions == ("us-east5", "us-central1")
    assert spec_520m.model_config is regmix_520m_proxy
    assert spec_520m.optimizer_config is regmix_520m_muonh_base
    assert spec_520m.tpu_type == "v5p-32"
    assert spec_520m.tpu_regions == ("us-east5", "us-central1")
    assert spec_1_2b.model_config is regmix_1_2b_proxy
    assert spec_1_2b.optimizer_config is regmix_1_2b_muonh_base
    assert spec_1_2b.tpu_type == "v5p-64"
    assert spec_1_2b.tpu_zone is None


def test_stratified_launch_artifacts_record_explicit_scaled_budgets():
    spec = stratified_baseline_launch.resolve_scale_spec(stratified_baseline_launch.StratifiedScale.REGMIX_130M_2P6B)
    experiment_budget = spec.experiment_budget_for_multiplier(0.5)
    target_budget = spec.target_budget_for_multiplier(0.5)
    artifacts = stratified_baseline_launch.build_launch_artifacts(
        scale=stratified_baseline_launch.StratifiedScale.REGMIX_130M_2P6B,
        name_prefix="unit-test/stratified_130m_0p5x",
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        target_budget_multiplier=0.5,
        tpu_type=spec.tpu_type,
        tpu_regions=spec.tpu_regions,
        tpu_zone=spec.tpu_zone,
        eval_datasets_cache_path=stratified_baseline_launch.EVAL_DATASETS_CACHE_PATH,
        resume_latest_checkpoints=False,
        cohort="unit_test_stratified_130m_0p5x",
    )

    assert artifacts.run_spec.cohort == "unit_test_stratified_130m_0p5x"
    assert artifacts.run_spec.model_family == spec.model_family
    assert artifacts.run_spec.experiment_budget == 1_300_000_000
    assert artifacts.run_spec.target_budget == scaling_study_recipes.scaled_budget(
        scaling_study_recipes.BASE_TARGET_BUDGET,
        0.5,
    )
    assert artifacts.run_spec.target_budget_multiplier == pytest.approx(0.5)
    assert artifacts.run_spec.num_train_steps == 4_959
    assert artifacts.training_step.name.endswith("unit-test/stratified_130m_0p5x/baseline_stratified")


def test_stratified_build_launch_artifacts_uses_scale_specific_batch_size_and_seq_len(monkeypatch):
    scale = stratified_baseline_launch.StratifiedScale.REGMIX_520M_10P4B
    spec = stratified_baseline_launch.resolve_scale_spec(scale)
    captured: dict[str, object] = {}

    class FakeExperiment:
        def __init__(self, *, batch_size: int = 128, seq_len: int = 2048, experiment_budget: int, **_: object) -> None:
            captured["batch_size"] = batch_size
            captured["seq_len"] = seq_len
            self.num_train_steps = experiment_budget // (batch_size * seq_len)

        def create_training_step(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(name="fake_train_step")

    monkeypatch.setattr(
        stratified_baseline_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **kwargs: FakeExperiment(**kwargs),
    )
    monkeypatch.setattr(
        stratified_baseline_launch,
        "create_run_manifest_step",
        lambda **kwargs: SimpleNamespace(name="fake_manifest_step", kwargs=kwargs),
    )

    artifacts = stratified_baseline_launch.build_launch_artifacts(
        scale=scale,
        name_prefix="unit-test/stratified_520m",
        experiment_budget=spec.experiment_budget,
        target_budget=scaling_study_recipes.BASE_TARGET_BUDGET,
        target_budget_multiplier=1.0,
        tpu_type=spec.tpu_type,
        tpu_regions=spec.tpu_regions,
        tpu_zone=spec.tpu_zone,
        eval_datasets_cache_path=stratified_baseline_launch.EVAL_DATASETS_CACHE_PATH,
        resume_latest_checkpoints=False,
        cohort="unit_test_stratified_520m",
    )

    assert captured["batch_size"] == spec.batch_size
    assert captured["seq_len"] == spec.seq_len
    assert artifacts.run_spec.num_train_steps == spec.experiment_budget // (spec.batch_size * spec.seq_len)


def test_stratified_build_launch_artifacts_raises_on_num_train_steps_mismatch(monkeypatch):
    scale = stratified_baseline_launch.StratifiedScale.REGMIX_520M_10P4B
    spec = stratified_baseline_launch.resolve_scale_spec(scale)

    class FakeExperiment:
        num_train_steps = 123

        def create_training_step(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(name="fake_train_step")

    monkeypatch.setattr(
        stratified_baseline_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **kwargs: FakeExperiment(),
    )
    monkeypatch.setattr(
        stratified_baseline_launch,
        "create_run_manifest_step",
        lambda **kwargs: SimpleNamespace(name="fake_manifest_step", kwargs=kwargs),
    )

    with pytest.raises(ValueError, match="num_train_steps"):
        stratified_baseline_launch.build_launch_artifacts(
            scale=scale,
            name_prefix="unit-test/stratified_520m_mismatch",
            experiment_budget=spec.experiment_budget,
            target_budget=scaling_study_recipes.BASE_TARGET_BUDGET,
            target_budget_multiplier=1.0,
            tpu_type=spec.tpu_type,
            tpu_regions=spec.tpu_regions,
            tpu_zone=spec.tpu_zone,
            eval_datasets_cache_path=stratified_baseline_launch.EVAL_DATASETS_CACHE_PATH,
            resume_latest_checkpoints=False,
            cohort="unit_test_stratified_520m_mismatch",
        )


def test_stratified_baseline_main_builds_selected_scale_in_ci_without_launching(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("CI", "1")

    monkeypatch.setattr(
        stratified_baseline_launch,
        "build_launch_artifacts",
        lambda **kwargs: (
            captured.update({"build_kwargs": kwargs})
            or SimpleNamespace(
                run_spec=SimpleNamespace(run_name=STRATIFIED_RUN_NAME),
                run_manifest_step="manifest",
                training_step="train",
                steps=["manifest", "train"],
            )
        ),
    )
    monkeypatch.setattr(
        stratified_baseline_launch.sys,
        "argv",
        [
            "launcher",
            "--scale",
            "520m_10p4b",
            "--tpu-type",
            "v5p-32",
            "--tpu-region",
            "us-central1",
            "--tpu-zone",
            "us-central1-a",
        ],
    )

    stratified_baseline_launch.main()
    captured["run_name"] = STRATIFIED_RUN_NAME

    assert captured["run_name"] == "baseline_stratified"
    assert captured["build_kwargs"]["scale"] == stratified_baseline_launch.StratifiedScale.REGMIX_520M_10P4B
    assert captured["build_kwargs"]["experiment_budget"] == REGMIX_520M_CHINCHILLA_BUDGET
    assert captured["build_kwargs"]["target_budget_multiplier"] == pytest.approx(1.0)


def test_strong_tier_scaling_study_cells_match_expected_reuse_and_run_counts():
    cells = scaling_study_recipes.build_strong_tier_cells()
    qsplit_path = scaling_study_recipes.ScalingStudyPath.QSPLIT_REPRESENTATIVE12
    stratified_path = scaling_study_recipes.ScalingStudyPath.STRATIFIED
    holdout_status = scaling_study_recipes.ScalingStudyCellStatus.HOLDOUT_ONLY
    qsplit_cells = [cell for cell in cells if cell.path == qsplit_path]
    stratified_cells = [cell for cell in cells if cell.path == stratified_path]
    holdout_cells = [cell for cell in cells if cell.status == holdout_status]
    new_cells = scaling_study_recipes.new_submission_cells(cells)

    assert len(qsplit_cells) == 9
    assert len(stratified_cells) == 9
    assert len(holdout_cells) == 2
    assert scaling_study_recipes.count_new_submission_runs(cells) == 91
    assert len(new_cells) == 14
    assert sum(cell.run_count for cell in new_cells if cell.path == qsplit_path) == 84
    assert sum(cell.run_count for cell in new_cells if cell.path == stratified_path) == 7
    assert {
        (cell.scale, cell.target_budget_multiplier)
        for cell in cells
        if cell.status == scaling_study_recipes.ScalingStudyCellStatus.REUSED
    } == {
        (scaling_study_recipes.ScalingStudyScale.REGMIX_300M_6B, 1.0),
        (scaling_study_recipes.ScalingStudyScale.REGMIX_520M_10P4B, 1.0),
    }


def test_strong_tier_scaling_study_main_builds_graph_in_ci_without_launching(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("CI", "1")
    monkeypatch.setattr(
        strong_tier_scaling_study_launch,
        "build_launch_steps",
        lambda **kwargs: (
            [SimpleNamespace(status="new", run_count=91)],
            ["study_manifest", "qsplit_cell", "stratified_cell"],
        ),
    )

    def fail_executor_main(*args, **kwargs):
        raise AssertionError("executor_main should not run in CI for the strong-tier launcher")

    monkeypatch.setattr(strong_tier_scaling_study_launch, "executor_main", fail_executor_main)
    monkeypatch.setattr(
        strong_tier_scaling_study_launch,
        "count_new_submission_runs",
        lambda cells=None: (captured.update({"cells": cells}) or 91),
    )
    monkeypatch.setattr(
        strong_tier_scaling_study_launch.sys,
        "argv",
        ["launcher", "--max-concurrent", "91"],
    )

    strong_tier_scaling_study_launch.main()

    assert captured["cells"] == [SimpleNamespace(status="new", run_count=91)]


def test_qsplit240_mmlu_sl_verb_rerun_builds_expected_manifest():
    run_specs = build_qsplit240_mmlu_sl_verb_rerun_specs()

    assert len(run_specs) == 240
    assert [spec.run_id for spec in run_specs] == list(range(240))
    assert {spec.cohort for spec in run_specs} == {"original_swarm_mmlu_sl_verb_rerun"}
    assert run_specs[0].run_name == "baseline_proportional"
    assert run_specs[1].run_name == "baseline_unimax"
    assert run_specs[2].run_name == "run_00002"
    assert run_specs[-1].run_name == "run_00239"
    assert run_specs[0].source_experiment != ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT
    assert {spec.source_experiment for spec in run_specs[2:]} == {ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT}
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert all(abs(sum(spec.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for spec in run_specs)
    assert all(abs(sum(spec.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for spec in run_specs)


def test_qsplit240_olmo_base_easy_overlap_rerun_builds_expected_manifest():
    run_specs = build_qsplit240_olmo_base_easy_overlap_rerun_specs()

    assert len(run_specs) == 240
    assert [spec.run_id for spec in run_specs] == list(range(240))
    assert {spec.cohort for spec in run_specs} == {"original_swarm_olmo_base_easy_overlap_rerun"}
    assert run_specs[0].run_name == "baseline_proportional"
    assert run_specs[1].run_name == "baseline_unimax"
    assert run_specs[2].run_name == "run_00002"
    assert run_specs[-1].run_name == "run_00239"
    assert run_specs[0].source_experiment != ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT
    assert {spec.source_experiment for spec in run_specs[2:]} == {ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT}
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert all(abs(sum(spec.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for spec in run_specs)
    assert all(abs(sum(spec.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for spec in run_specs)


def test_qsplit240_olmo_base_easy_overlap_rerun_select_run_specs_for_shard():
    run_specs = build_qsplit240_olmo_base_easy_overlap_rerun_specs()
    shard_run_specs = qsplit240_olmo_base_easy_overlap_rerun.select_run_specs_for_shard(
        run_specs,
        shard_index=2,
        shard_count=8,
    )

    assert [spec.run_id for spec in shard_run_specs] == list(range(60, 90))
    assert qsplit240_olmo_base_easy_overlap_rerun.shard_execution_name_prefix(
        name_prefix=qsplit240_olmo_base_easy_overlap_rerun.NAME,
        shard_index=2,
        shard_count=8,
    ).endswith("shard_03of08")


def test_selected_baselines_olmo_base_easy_overlap_rerun_builds_expected_manifest():
    run_specs = build_selected_baselines_olmo_base_easy_overlap_rerun_specs()

    assert len(run_specs) == 3
    assert [spec.run_id for spec in run_specs] == [412, 3, 248]
    assert {spec.cohort for spec in run_specs} == {"selected_baselines_olmo_base_easy_overlap_rerun"}
    assert [spec.run_name for spec in run_specs] == [
        "baseline_genericfamily_power_family_penalty_raw_optimum",
        "baseline_stratified",
        "baseline_olmix_loglinear_uncheatable_bpb",
    ]
    assert [spec.source_experiment for spec in run_specs] == [
        "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_uncheatable_bpb",
        "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b",
        "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_uncheatable_bpb",
    ]
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert all(abs(sum(spec.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for spec in run_specs)
    assert all(abs(sum(spec.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for spec in run_specs)


def test_selected_baselines_olmo_base_easy_overlap_resolve_checkpoint_roots_uses_requested_regions(monkeypatch):
    seen: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        selected_overlap_rerun,
        "resolve_latest_checkpoint_root",
        lambda *, experiment_name_prefix, run_name, checkpoint_regions: (
            seen.append(tuple(checkpoint_regions)) or f"gs://unit-test/{run_name}"
        ),
    )

    resolved = selected_overlap_rerun.resolve_checkpoint_roots(
        build_selected_baselines_olmo_base_easy_overlap_rerun_specs(),
        checkpoint_regions=("us-east5",),
    )

    assert len(resolved) == 3
    assert seen == [("us-east5",), ("us-east5",), ("us-east5",)]


def test_qsplit240_seedpanel_mmlu_sl_verb_rerun_builds_expected_manifest():
    run_specs = build_qsplit240_seedpanel_mmlu_sl_verb_rerun_specs()

    assert len(run_specs) == 720
    assert [spec.run_id for spec in run_specs] == list(range(720))
    assert {spec.cohort for spec in run_specs} == {"replicated_swarm"}
    assert {spec.trainer_seed for spec in run_specs} == set(TRAINER_SEEDS)
    assert all(spec.data_seed is None for spec in run_specs)
    assert all(spec.simulated_epoch_subset_seed == SIMULATED_EPOCH_SUBSET_SEED for spec in run_specs)
    assert len({spec.run_name for spec in run_specs}) == len(run_specs)
    assert all(spec.source_experiment == QSPLIT240_SEEDPANEL_NAME for spec in run_specs)
    assert all(spec.checkpoint_root is None for spec in run_specs)
    assert run_specs[0].run_name == "seed10000_baseline_proportional"
    assert run_specs[1].run_name == "seed10001_baseline_proportional"
    assert run_specs[3].run_name == "seed10000_baseline_unimax"
    assert run_specs[-1].run_name == "seed10002_run_00239"
    assert run_specs[0].candidate_run_name == "baseline_proportional"
    assert run_specs[-1].candidate_run_name == "run_00239"
    assert all(abs(sum(spec.phase_weights["phase_0"].values()) - 1.0) < 1e-12 for spec in run_specs)
    assert all(abs(sum(spec.phase_weights["phase_1"].values()) - 1.0) < 1e-12 for spec in run_specs)


def test_olmix_loglinear_mmlu_sl_verb_rerun_builds_expected_manifest():
    run_specs = build_olmix_loglinear_mmlu_sl_verb_rerun_specs()

    assert len(run_specs) == 1
    assert run_specs[0].run_id == 240
    assert run_specs[0].run_name == OLMIX_LOGLINEAR_RUN_NAME
    assert run_specs[0].cohort == "original_swarm_mmlu_sl_verb_rerun"
    assert run_specs[0].source_experiment == OLMIX_LOGLINEAR_SOURCE_EXPERIMENT
    assert run_specs[0].checkpoint_root is None
    assert run_specs[0].phase_weights == OLMIX_LOGLINEAR_PHASE_WEIGHTS


def test_fitted_olmix_sl_verb_mmlu_sl_verb_rerun_builds_expected_manifest():
    run_specs = build_fitted_olmix_sl_verb_mmlu_sl_verb_rerun_specs(
        phase_weights={
            "phase_0": {"a": 0.25, "b": 0.75},
            "phase_1": {"a": 0.6, "b": 0.4},
        }
    )

    assert len(run_specs) == 1
    assert run_specs[0].run_id == 243
    assert run_specs[0].run_name == "baseline_olmix_loglinear_sl_verb_choice_logprob_norm"
    assert run_specs[0].cohort == "fitted_olmix_sl_verb_mmlu_sl_verb_rerun"
    assert run_specs[0].source_experiment == "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_sl_verb_choice_logprob_norm"
    assert run_specs[0].checkpoint_root is None
    assert run_specs[0].phase_weights == {
        "phase_0": {"a": 0.25, "b": 0.75},
        "phase_1": {"a": 0.6, "b": 0.4},
    }


def test_power_ridge_single_weight_config_uses_constant_mix_realization():
    weight_config = create_power_ridge_single_weight_config()
    summary = power_ridge_single_summary()

    assert weight_config.run_id == 245
    assert set(weight_config.phase_weights) == {"phase_0", "phase_1"}
    assert weight_config.phase_weights["phase_0"] == weight_config.phase_weights["phase_1"]
    assert abs(sum(weight_config.phase_weights["phase_0"].values()) - 1.0) < 1e-12
    assert POWER_RIDGE_SINGLE_OBJECTIVE_METRIC == "eval/uncheatable_eval/bpb"
    assert POWER_RIDGE_SINGLE_SOURCE_EXPERIMENT.endswith("power_ridge_uncheatable_bpb")
    assert POWER_RIDGE_SINGLE_RUN_NAME == "baseline_power_ridge_single_constant_mix"
    assert float(summary["predicted_optimum_value"]) < float(summary["best_observed_value"])


def test_dsre_ceq_predicted_weight_configs_build_expected_variants():
    full = create_dsre_ceq_predicted_weight_config()
    collapsed = create_dsre_ceq_predicted_quality_collapsed_weight_config()
    full_summary = dsre_ceq_predicted_summary()
    collapsed_summary = dsre_ceq_predicted_quality_collapsed_summary()

    assert full.run_id == 246
    assert collapsed.run_id == 247
    assert set(full.phase_weights) == {"phase_0", "phase_1"}
    assert set(collapsed.phase_weights) == {"phase_0", "phase_1"}
    assert set(full.phase_weights["phase_0"]) == set(collapsed.phase_weights["phase_0"])
    assert set(full.phase_weights["phase_1"]) == set(collapsed.phase_weights["phase_1"])

    for phase_name in ("phase_0", "phase_1"):
        assert sum(full.phase_weights[phase_name].values()) == pytest.approx(1.0)
        assert sum(collapsed.phase_weights[phase_name].values()) == pytest.approx(1.0)

        full_food_total = (
            full.phase_weights[phase_name]["dolma3_cc/food_and_dining_high"]
            + full.phase_weights[phase_name]["dolma3_cc/food_and_dining_low"]
        )
        collapsed_food_total = (
            collapsed.phase_weights[phase_name]["dolma3_cc/food_and_dining_high"]
            + collapsed.phase_weights[phase_name]["dolma3_cc/food_and_dining_low"]
        )
        food_high_ratio = TOP_LEVEL_DOMAIN_TOKEN_COUNTS["dolma3_cc/food_and_dining_high"] / (
            TOP_LEVEL_DOMAIN_TOKEN_COUNTS["dolma3_cc/food_and_dining_high"]
            + TOP_LEVEL_DOMAIN_TOKEN_COUNTS["dolma3_cc/food_and_dining_low"]
        )
        assert collapsed_food_total == pytest.approx(full_food_total)
        assert collapsed.phase_weights[phase_name]["dolma3_cc/food_and_dining_high"] == pytest.approx(
            full_food_total * food_high_ratio
        )

    assert full_summary["run_name"] == DSRE_CEQ_PREDICTED_RUN_NAME
    assert collapsed_summary["run_name"] == DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME
    assert full_summary["predicted_bpb"] > 0
    assert collapsed_summary["derived_from_run_name"] == DSRE_CEQ_PREDICTED_RUN_NAME
    assert full_summary["nearest_observed_tv_distance"] >= 0
    assert collapsed_summary["nearest_observed_tv_distance"] >= 0
    assert DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT.startswith("pinlin_calvin_xu/data_mixture/")


def test_build_top_level_domains_with_collapsed_cc_topics_has_expected_shape():
    domains = build_top_level_domains_with_collapsed_cc_topics()

    assert len(domains) == 26
    domain_names = {domain.name for domain in domains}
    assert "dolma3_cc/art_and_design" in domain_names
    art_and_design = next(domain for domain in domains if domain.name == "dolma3_cc/art_and_design")
    assert {component.name for component in art_and_design.components} == {
        "dolma3_cc/art_and_design_high",
        "dolma3_cc/art_and_design_low",
    }


def test_dsre_ceq_predicted_topic_collapsed_matches_realized_flat_quality_collapsed():
    topic_collapsed = create_dsre_ceq_predicted_topic_collapsed_weight_config()
    collapsed = create_dsre_ceq_predicted_quality_collapsed_weight_config()
    summary = dsre_ceq_predicted_topic_collapsed_summary()

    assert topic_collapsed.run_id == 250
    assert set(topic_collapsed.phase_weights) == {"phase_0", "phase_1"}
    assert len(topic_collapsed.phase_weights["phase_0"]) == 26
    assert topic_collapsed.phase_weights["phase_0"]["dolma3_cc/food_and_dining"] == pytest.approx(
        collapsed.phase_weights["phase_0"]["dolma3_cc/food_and_dining_high"]
        + collapsed.phase_weights["phase_0"]["dolma3_cc/food_and_dining_low"]
    )
    assert summary["equivalent_flat_run_name"] == "baseline_dsre_ceq_predicted_quality_collapsed"
    assert summary["equivalent_flat_tv_distance"] == pytest.approx(0.0, abs=1e-12)
    assert summary["n_collapsed_domains"] == 26
    assert summary["nearest_observed_tv_distance"] >= 0.0
    assert DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME == "baseline_dsre_ceq_predicted_topic_collapsed"
    assert DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_SOURCE_EXPERIMENT.endswith("topic_collapsed")


def test_fit_olmix_sl_verb_from_frame_returns_normalized_schedule():
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | str]] = []
    phase_names = ["phase_0", "phase_1"]
    domain_names = ["a", "b", "c"]
    coeffs = np.array([-0.9, 0.4, 0.8, 0.7, -0.6, 0.5], dtype=float)
    for run_id in range(36):
        phase0 = rng.dirichlet(np.ones(3))
        phase1 = rng.dirichlet(np.ones(3))
        flat = np.concatenate([phase0, phase1])
        negated_metric = 0.2 + np.exp(float(flat @ coeffs))
        row: dict[str, float | str] = {
            "run_id": run_id,
            "run_name": f"run_{run_id:05d}",
            OLMIX_SL_VERB_OBJECTIVE_METRIC: -negated_metric,
        }
        for phase_name, phase_weights in zip(phase_names, [phase0, phase1], strict=True):
            for domain_name, weight in zip(domain_names, phase_weights, strict=True):
                row[f"{phase_name}_{domain_name}"] = float(weight)
        rows.append(row)

    fit = fit_olmix_sl_verb_from_frame(
        pd.DataFrame(rows),
        natural_proportions=np.asarray([0.5, 0.3, 0.2], dtype=float),
        phase_fractions=np.asarray([0.8, 0.2], dtype=float),
        source_results_path="gs://unit-test/results.csv",
    )

    assert fit.objective_metric == OLMIX_SL_VERB_OBJECTIVE_METRIC
    assert fit.negated_objective_metric == NEGATED_OBJECTIVE_METRIC
    assert fit.source_results_path == "gs://unit-test/results.csv"
    assert np.isfinite(fit.fit_huber_loss)
    assert np.isfinite(fit.predicted_choice_logprob_norm)
    assert set(fit.phase_weights) == {"phase_0", "phase_1"}
    assert set(fit.phase_weights["phase_0"]) == {"a", "b", "c"}
    assert set(fit.phase_weights["phase_1"]) == {"a", "b", "c"}
    assert sum(fit.phase_weights["phase_0"].values()) == pytest.approx(1.0)
    assert sum(fit.phase_weights["phase_1"].values()) == pytest.approx(1.0)
    assert fit.predicted_choice_logprob_norm <= 0.0


def test_fit_olmix_uncheatable_from_frame_returns_normalized_schedule():
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | str]] = []
    phase_names = ["phase_0", "phase_1"]
    domain_names = ["a", "b", "c"]
    coeffs = np.array([0.3, -0.5, 0.2, 0.6, -0.4, 0.1], dtype=float)
    for run_id in range(36):
        phase0 = rng.dirichlet(np.ones(3))
        phase1 = rng.dirichlet(np.ones(3))
        flat = np.concatenate([phase0, phase1])
        metric = 0.2 + np.exp(float(flat @ coeffs))
        row: dict[str, float | str] = {
            "run_id": run_id,
            "run_name": f"run_{run_id:05d}",
            OLMIX_UNCHEATABLE_OBJECTIVE_METRIC: metric,
        }
        for phase_name, phase_weights in zip(phase_names, [phase0, phase1], strict=True):
            for domain_name, weight in zip(domain_names, phase_weights, strict=True):
                row[f"{phase_name}_{domain_name}"] = float(weight)
        rows.append(row)

    fit = fit_olmix_uncheatable_from_frame(
        pd.DataFrame(rows),
        natural_proportions=np.asarray([0.5, 0.3, 0.2], dtype=float),
        phase_fractions=np.asarray([0.8, 0.2], dtype=float),
        source_results_path="/tmp/unit-test-two-phase-many.csv",
    )

    assert fit.objective_metric == OLMIX_UNCHEATABLE_OBJECTIVE_METRIC
    assert fit.source_results_path == "/tmp/unit-test-two-phase-many.csv"
    assert np.isfinite(fit.fit_huber_loss)
    assert np.isfinite(fit.predicted_objective)
    assert set(fit.phase_weights) == {"phase_0", "phase_1"}
    assert set(fit.phase_weights["phase_0"]) == {"a", "b", "c"}
    assert set(fit.phase_weights["phase_1"]) == {"a", "b", "c"}
    assert sum(fit.phase_weights["phase_0"].values()) == pytest.approx(1.0)
    assert sum(fit.phase_weights["phase_1"].values()) == pytest.approx(1.0)
    assert fit.predicted_objective > 0.0
    assert fit.nearest_observed_tv_distance >= 0.0


def test_thresholdtotal_overfit_weight_config_and_summary():
    weight_config = create_thresholdtotal_overfit_weight_config()
    summary = thresholdtotal_overfit_summary()

    assert weight_config.run_id == 249
    assert set(weight_config.phase_weights) == {"phase_0", "phase_1"}
    assert sum(weight_config.phase_weights["phase_0"].values()) == pytest.approx(1.0)
    assert sum(weight_config.phase_weights["phase_1"].values()) == pytest.approx(1.0)
    assert THRESHOLDTOTAL_OVERFIT_OBJECTIVE_METRIC == "eval/uncheatable_eval/bpb"
    assert THRESHOLDTOTAL_OVERFIT_SOURCE_EXPERIMENT.endswith("thresholdtotal_overfit_uncheatable_bpb")
    assert THRESHOLDTOTAL_OVERFIT_RUN_NAME == "baseline_thresholdtotal_overfit_uncheatable_bpb"
    assert float(summary["predicted_optimum_value"]) < float(summary["observed_best_value"])
    assert float(summary["nearest_observed_tv_distance"]) >= 0.0
    assert summary["model"] == "ThresholdTotal-Overfit"
    assert len(summary["phase0_top_domains"]) == 10
    assert len(summary["phase1_top_domains"]) == 10


def test_ccglobalpremium_weight_configs_and_summaries():
    threshold_config = create_ccglobalpremium_threshold_weight_config()
    retained_config = create_ccglobalpremium_retainedtotal_weight_config()
    threshold_summary = ccglobalpremium_threshold_summary()
    retained_summary = ccglobalpremium_retainedtotal_summary()

    for config in (threshold_config, retained_config):
        for phase_name, domain_weights in config.phase_weights.items():
            assert abs(sum(domain_weights.values()) - 1.0) < 1e-9, phase_name

    threshold_phase0 = pd.Series(threshold_config.phase_weights["phase_0"]).sort_index()
    retained_phase0 = pd.Series(retained_config.phase_weights["phase_0"]).sort_index()
    threshold_phase1 = pd.Series(threshold_config.phase_weights["phase_1"]).sort_index()
    retained_phase1 = pd.Series(retained_config.phase_weights["phase_1"]).sort_index()

    assert threshold_summary["predicted_optimum_value"] < threshold_summary["observed_best_value"]
    assert retained_summary["predicted_optimum_value"] < retained_summary["observed_best_value"]
    assert threshold_summary["nearest_observed_tv_distance"] > 0.3
    assert retained_summary["nearest_observed_tv_distance"] > 0.3
    assert (
        0.5 * (np.abs(threshold_phase0 - retained_phase0).sum() + np.abs(threshold_phase1 - retained_phase1).sum()) > 0.2
    )
    assert CCGLOBALPREMIUM_SOURCE_EXPERIMENT.endswith("ccglobalpremium_uncheatable_bpb")
    assert CCGLOBALPREMIUM_THRESHOLD_RUN_NAME == "baseline_ccglobalpremium_threshold_uncheatable_bpb"
    assert CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME == "baseline_ccglobalpremium_retainedtotal_uncheatable_bpb"


def test_ccpairtotal_retainedtotal_weight_config_and_summary():
    weight_config = create_ccpairtotal_retainedtotal_weight_config()
    summary = ccpairtotal_retainedtotal_summary()

    assert summary["run_name"] == CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME
    assert summary["optimizer_success"] is True
    assert summary["predicted_optimum_value"] == pytest.approx(1.0125046862932436)
    assert summary["nearest_observed_tv_distance"] == pytest.approx(0.45411673092136945)
    assert summary["phase0_support_below_1e4"] == 6
    assert summary["phase1_support_below_1e4"] == 7
    assert summary["phase0_max_weight"] == pytest.approx(0.2468947868933573)
    assert summary["phase1_max_weight"] == pytest.approx(0.17616211468794302)
    assert summary["phase_weights"] == weight_config.phase_weights
    assert CCPAIRTOTAL_RETAINEDTOTAL_SOURCE_EXPERIMENT.endswith("ccpairtotal_retainedtotal_uncheatable_bpb")
    assert CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME == "baseline_ccpairtotal_retainedtotal_uncheatable_bpb"


def test_genericfamily_tuned_weight_config_and_summary():
    weight_config = create_genericfamily_tuned_weight_config()
    summary = genericfamily_tuned_summary()

    assert summary["run_name"] == GENERICFAMILY_TUNED_RUN_NAME
    assert summary["predicted_optimum_value"] == pytest.approx(1.0436372226731572)
    assert summary["anchor_coefficients"]["validated_global"] == pytest.approx(0.9175270003197237)
    assert summary["anchor_coefficients"]["validated_pair"] == pytest.approx(0.0015381110202398686)
    assert summary["phase0_support_below_1e4"] == 0
    assert summary["phase1_support_below_1e4"] == 0
    assert summary["phase0_max_weight"] == pytest.approx(0.15201949193595957)
    assert summary["phase1_max_weight"] == pytest.approx(0.19824977200209287)
    assert summary["phase_weights"] == weight_config.phase_weights
    assert GENERICFAMILY_TUNED_SOURCE_EXPERIMENT.endswith("genericfamily_tuned_uncheatable_bpb")
    assert GENERICFAMILY_TUNED_RUN_NAME == "baseline_genericfamily_retainedtotal_tuned_uncheatable_bpb"


def test_aggregate_candidate_mean_frame_collapses_replicates():
    frame = pd.DataFrame(
        [
            {
                "candidate_run_id": 1,
                "candidate_run_name": "run_a",
                "candidate_source_experiment": "source",
                "trainer_seed": 10000,
                OLMIX_SL_VERB_OBJECTIVE_METRIC: -1.4,
                "phase_0_a": 0.25,
                "phase_0_b": 0.75,
                "phase_1_a": 0.6,
                "phase_1_b": 0.4,
            },
            {
                "candidate_run_id": 1,
                "candidate_run_name": "run_a",
                "candidate_source_experiment": "source",
                "trainer_seed": 10001,
                OLMIX_SL_VERB_OBJECTIVE_METRIC: -1.2,
                "phase_0_a": 0.25,
                "phase_0_b": 0.75,
                "phase_1_a": 0.6,
                "phase_1_b": 0.4,
            },
            {
                "candidate_run_id": 2,
                "candidate_run_name": "run_b",
                "candidate_source_experiment": "source",
                "trainer_seed": 10000,
                OLMIX_SL_VERB_OBJECTIVE_METRIC: -1.1,
                "phase_0_a": 0.1,
                "phase_0_b": 0.9,
                "phase_1_a": 0.2,
                "phase_1_b": 0.8,
            },
            {
                "candidate_run_id": 2,
                "candidate_run_name": "run_b",
                "candidate_source_experiment": "source",
                "trainer_seed": 10001,
                OLMIX_SL_VERB_OBJECTIVE_METRIC: -0.9,
                "phase_0_a": 0.1,
                "phase_0_b": 0.9,
                "phase_1_a": 0.2,
                "phase_1_b": 0.8,
            },
        ]
    )

    aggregated = aggregate_candidate_mean_frame(frame)

    assert list(aggregated["candidate_run_name"]) == ["run_a", "run_b"]
    assert aggregated.loc[aggregated["candidate_run_name"] == "run_a", OLMIX_SL_VERB_OBJECTIVE_METRIC].iloc[
        0
    ] == pytest.approx(-1.3)
    assert aggregated.loc[aggregated["candidate_run_name"] == "run_b", OLMIX_SL_VERB_OBJECTIVE_METRIC].iloc[
        0
    ] == pytest.approx(-1.0)
    assert aggregated.loc[aggregated["candidate_run_name"] == "run_a", "phase_0_a"].iloc[0] == pytest.approx(0.25)
    assert aggregated.loc[aggregated["candidate_run_name"] == "run_b", "phase_1_b"].iloc[0] == pytest.approx(0.8)


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


def test_collect_manifest_results_backfills_eval_metrics_from_checkpoint_when_wandb_row_missing(tmp_path, monkeypatch):
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    output_dir = tmp_path / "analysis"
    checkpoint_root = tmp_path / "checkpoint_root"
    metrics_path = checkpoint_root / determinism_analysis.CHECKPOINT_EVAL_METRICS_PATH
    output_dir.mkdir()
    metrics_path.parent.mkdir(parents=True)

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
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"eval/uncheatable_eval/bpb": 1.7}),
                json.dumps(
                    {
                        "eval/uncheatable_eval/bpb": 1.23,
                        "eval/paloma/c4_en/bpb": 4.56,
                        "lm_eval/mmlu_5shot/bpb": 9.99,
                    }
                ),
            ]
        )
    )

    monkeypatch.setattr("experiments.domain_phase_mix.analysis.query_wandb_runs", lambda **_: [])
    monkeypatch.setattr("experiments.domain_phase_mix.analysis.query_wandb_runs_by_name_substrings", lambda **_: [])
    monkeypatch.setattr(
        determinism_analysis,
        "resolve_unique_checkpoint_root",
        lambda *, source_experiment, run_name: str(checkpoint_root),
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
    assert results_df.loc[0, "status"] == "checkpoint_eval_only"
    assert results_df.loc[0, "checkpoint_root"] == str(checkpoint_root)
    assert results_df.loc[0, "eval/uncheatable_eval/bpb"] == pytest.approx(1.23)
    assert results_df.loc[0, "eval/paloma/c4_en/bpb"] == pytest.approx(4.56)
    assert "lm_eval/mmlu_5shot/bpb" not in results_df.columns


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


def test_qsplit240_seedpanel_mmlu_sl_verb_rerun_main_wires_max_concurrent(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        qsplit240_seedpanel_slverb_rerun,
        "resolve_checkpoint_roots",
        lambda run_specs: [
            replace(spec, checkpoint_root=f"gs://unit-test/checkpoints/{spec.run_name}") for spec in run_specs
        ],
    )
    monkeypatch.setattr(
        qsplit240_seedpanel_slverb_rerun,
        "evaluate_levanter_lm_evaluation_harness",
        lambda **kwargs: SimpleNamespace(name=kwargs["model_name"]),
    )
    monkeypatch.setattr(
        qsplit240_seedpanel_slverb_rerun,
        "output_path_of",
        lambda step, filename: f"gs://unit-test/{step.name}/{filename}",
    )
    monkeypatch.setattr(
        qsplit240_seedpanel_slverb_rerun,
        "create_run_manifest_step",
        lambda **_: SimpleNamespace(name="run_manifest"),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(qsplit240_seedpanel_slverb_rerun, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        qsplit240_seedpanel_slverb_rerun,
        "build_run_specs",
        lambda: build_qsplit240_seedpanel_mmlu_sl_verb_rerun_specs()[:6],
    )
    monkeypatch.setattr(
        qsplit240_seedpanel_slverb_rerun.sys,
        "argv",
        ["launcher", "--max-concurrent", "19"],
    )

    qsplit240_seedpanel_slverb_rerun.main()

    assert captured["config"].max_concurrent == 19
    assert len(captured["steps"]) == 8
    assert "fixed-subset seedpanel" in captured["description"]
    assert qsplit240_seedpanel_slverb_rerun.DEFAULT_MAX_CONCURRENT == 24


def test_qsplit240_520m_pilot_main_builds_graph_in_ci_without_launching(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("CI", "1")
    monkeypatch.setattr(
        qsplit240_520m_pilot,
        "build_launch_artifacts",
        lambda **kwargs: (
            captured.update({"build_kwargs": kwargs})
            or SimpleNamespace(
                run_specs=build_qsplit240_520m_pilot_run_specs(),
                execution_name_prefix=qsplit240_520m_pilot.NAME,
                steps=["manifest", "cache", "train", "results", "fit"],
            )
        ),
    )

    def fail_executor_main(*args, **kwargs):
        raise AssertionError("executor_main should not run in CI for the 520M pilot launcher")

    monkeypatch.setattr(qsplit240_520m_pilot, "executor_main", fail_executor_main)
    monkeypatch.setattr(
        qsplit240_520m_pilot.sys,
        "argv",
        [
            "launcher",
            "--tpu-type",
            "v5p-32",
            "--tpu-regions",
            "us-east5,us-central1",
            "--panel",
            "representative12",
            "--max-concurrent",
            "1",
        ],
    )

    qsplit240_520m_pilot.main()

    assert captured["build_kwargs"]["name_prefix"] == qsplit240_520m_pilot.NAME
    assert captured["build_kwargs"]["panel"] == "representative12"
    assert captured["build_kwargs"]["tpu_type"] == "v5p-32"
    assert captured["build_kwargs"]["tpu_regions"] == ("us-east5", "us-central1")
    assert captured["build_kwargs"]["tpu_zone"] is None
    assert captured["build_kwargs"]["eval_datasets_cache_path"].startswith("gs://marin-us-central1/")
    assert captured["build_kwargs"]["resume_latest_checkpoints"] is True


def test_qsplit240_1_2b_pilot_main_builds_graph_in_ci_without_launching(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("CI", "1")
    monkeypatch.setattr(
        qsplit240_1_2b_pilot,
        "build_launch_artifacts",
        lambda **kwargs: (
            captured.update({"build_kwargs": kwargs})
            or SimpleNamespace(
                run_specs=build_qsplit240_1_2b_pilot_run_specs(),
                execution_name_prefix=qsplit240_1_2b_pilot.NAME,
                steps=["manifest", "cache", "train", "results", "fit"],
            )
        ),
    )

    def fail_executor_main(*args, **kwargs):
        raise AssertionError("executor_main should not run in CI for the 1.2B pilot launcher")

    monkeypatch.setattr(qsplit240_1_2b_pilot, "executor_main", fail_executor_main)
    monkeypatch.setattr(
        qsplit240_1_2b_pilot.sys,
        "argv",
        [
            "launcher",
            "--tpu-type",
            "v5p-64",
            "--tpu-regions",
            "us-east5,us-central1",
            "--panel",
            "baselines3",
            "--max-concurrent",
            "1",
        ],
    )

    qsplit240_1_2b_pilot.main()

    assert captured["build_kwargs"]["name_prefix"] == qsplit240_1_2b_pilot.NAME
    assert captured["build_kwargs"]["panel"] == "baselines3"
    assert captured["build_kwargs"]["tpu_type"] == "v5p-64"
    assert captured["build_kwargs"]["tpu_regions"] == ("us-east5", "us-central1")
    assert captured["build_kwargs"]["tpu_zone"] is None
    assert captured["build_kwargs"]["eval_datasets_cache_path"].startswith("gs://marin-us-central1/")
    assert captured["build_kwargs"]["resume_latest_checkpoints"] is True


def test_resolve_latest_checkpoint_root_picks_highest_step_across_regions(monkeypatch):
    class FakeFs:
        def glob(self, pattern: str) -> list[str]:
            return {
                "gs://marin-us-east5/checkpoints/unit/prefix/run_00125-*/checkpoints/step-*/metadata.json": [
                    "marin-us-east5/checkpoints/unit/prefix/run_00125-east/checkpoints/step-4000/metadata.json",
                ],
                "gs://marin-us-central1/checkpoints/unit/prefix/run_00125-*/checkpoints/step-*/metadata.json": [
                    "gs://marin-us-central1/checkpoints/unit/prefix/run_00125-central/checkpoints/step-6000/metadata.json",
                ],
            }.get(pattern, [])

        def open(self, path: str, mode: str = "rb") -> BytesIO:
            return BytesIO(b'{"is_temporary": false}')

        def exists(self, path: str) -> bool:
            return path.endswith("/manifest.ocdbt") or path.endswith("/d")

    monkeypatch.setattr(qsplit240_replay.fsspec, "get_fs_token_paths", lambda pattern: (FakeFs(), None, None))

    checkpoint_root = qsplit240_replay.resolve_latest_checkpoint_root(
        experiment_name_prefix="unit/prefix",
        run_name="run_00125",
        checkpoint_regions=("us-east5", "us-central1"),
    )

    assert checkpoint_root == "gs://marin-us-central1/checkpoints/unit/prefix/run_00125-central"


def test_resolve_latest_checkpoint_path_picks_concrete_step_across_regions(monkeypatch):
    class FakeFs:
        def glob(self, pattern: str) -> list[str]:
            return {
                "gs://marin-us-east5/checkpoints/unit/prefix/run_00125-*/checkpoints/step-*/metadata.json": [
                    "marin-us-east5/checkpoints/unit/prefix/run_00125-east/checkpoints/step-4000/metadata.json",
                ],
                "gs://marin-us-central1/checkpoints/unit/prefix/run_00125-*/checkpoints/step-*/metadata.json": [
                    "gs://marin-us-central1/checkpoints/unit/prefix/run_00125-central/checkpoints/step-6000/metadata.json",
                ],
            }.get(pattern, [])

        def open(self, path: str, mode: str = "rb") -> BytesIO:
            return BytesIO(b'{"is_temporary": false}')

        def exists(self, path: str) -> bool:
            return path.endswith("/manifest.ocdbt") or path.endswith("/d")

    monkeypatch.setattr(qsplit240_replay.fsspec, "get_fs_token_paths", lambda pattern: (FakeFs(), None, None))

    checkpoint_path = qsplit240_replay.resolve_latest_checkpoint_path(
        experiment_name_prefix="unit/prefix",
        run_name="run_00125",
        checkpoint_regions=("us-east5", "us-central1"),
    )

    assert checkpoint_path == "gs://marin-us-central1/checkpoints/unit/prefix/run_00125-central/checkpoints/step-6000"


def test_resolve_latest_checkpoint_path_uses_newer_temporary_checkpoint(monkeypatch):
    class FakeFs:
        def glob(self, pattern: str) -> list[str]:
            return {
                "gs://marin-us-east5/checkpoints/unit/prefix/run_00125-*/checkpoints/step-*/metadata.json": [
                    "gs://marin-us-east5/checkpoints/unit/prefix/run_00125-committed/checkpoints/step-6000/metadata.json",
                ],
                ("gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/" "run_00125-*/step-*/metadata.json"): [
                    "gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/run_00125-temp/step-8000/metadata.json",
                ],
            }.get(pattern, [])

        def open(self, path: str, mode: str = "rb") -> BytesIO:
            is_temporary = "marin-tmp" in path
            return BytesIO(json.dumps({"is_temporary": is_temporary}).encode())

        def exists(self, path: str) -> bool:
            return path.endswith("/manifest.ocdbt") or path.endswith("/d")

    monkeypatch.setattr(qsplit240_replay.fsspec, "get_fs_token_paths", lambda pattern: (FakeFs(), None, None))

    checkpoint_path = qsplit240_replay.resolve_latest_checkpoint_path(
        experiment_name_prefix="unit/prefix",
        run_name="run_00125",
        checkpoint_regions=("us-east5",),
    )

    assert checkpoint_path == "gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/run_00125-temp/step-8000"


def test_resolve_latest_checkpoint_path_skips_temporary_metadata_without_tensor_data(monkeypatch):
    class FakeFs:
        def glob(self, pattern: str) -> list[str]:
            return {
                "gs://marin-us-east5/checkpoints/unit/prefix/run_00125-*/checkpoints/step-*/metadata.json": [
                    "gs://marin-us-east5/checkpoints/unit/prefix/run_00125-committed/checkpoints/step-6000/metadata.json",
                ],
                ("gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/" "run_00125-*/step-*/metadata.json"): [
                    "gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/run_00125-temp/step-8000/metadata.json",
                ],
            }.get(pattern, [])

        def open(self, path: str, mode: str = "rb") -> BytesIO:
            is_temporary = "marin-tmp" in path
            return BytesIO(json.dumps({"is_temporary": is_temporary}).encode())

        def exists(self, path: str) -> bool:
            return "run_00125-committed" in path and (path.endswith("/manifest.ocdbt") or path.endswith("/d"))

    monkeypatch.setattr(qsplit240_replay.fsspec, "get_fs_token_paths", lambda pattern: (FakeFs(), None, None))

    checkpoint_path = qsplit240_replay.resolve_latest_checkpoint_path(
        experiment_name_prefix="unit/prefix",
        run_name="run_00125",
        checkpoint_regions=("us-east5",),
    )

    assert checkpoint_path == "gs://marin-us-east5/checkpoints/unit/prefix/run_00125-committed/checkpoints/step-6000"


def test_checkpoint_initialization_path_does_not_mirror_temp_bucket_checkpoints():
    assert (
        qsplit240_replay.checkpoint_initialization_path(
            "gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/run_00125-temp/step-8000"
        )
        == "gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/run_00125-temp/step-8000"
    )


def test_mirror_path_rebases_marin_bucket_urls_to_bucket_relative_paths():
    assert (
        qsplit240_replay.mirror_path("gs://marin-us-central1/checkpoints/unit/prefix/run_00125-central")
        == "mirror://checkpoints/unit/prefix/run_00125-central"
    )
    assert (
        qsplit240_replay.resolve_qsplit240_eval_cache_path_for_regions(
            ("us-east5", "us-central1"),
            "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks",
        )
        == "mirror://raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
    )


def test_qsplit240_olmo_base_easy_overlap_rerun_main_wires_cache_and_max_concurrent(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        qsplit240_olmo_base_easy_overlap_rerun,
        "resolve_checkpoint_roots",
        lambda run_specs: [
            replace(spec, checkpoint_root=f"gs://unit-test/checkpoints/{spec.run_name}") for spec in run_specs
        ],
    )
    monkeypatch.setattr(
        qsplit240_olmo_base_easy_overlap_rerun,
        "evaluate_levanter_lm_evaluation_harness",
        lambda **kwargs: SimpleNamespace(name=kwargs["model_name"]),
    )
    monkeypatch.setattr(
        qsplit240_olmo_base_easy_overlap_rerun,
        "output_path_of",
        lambda step, filename=None: (
            f"gs://unit-test/{step.name}" if filename is None else f"gs://unit-test/{step.name}/{filename}"
        ),
    )
    monkeypatch.setattr(
        qsplit240_olmo_base_easy_overlap_rerun,
        "create_run_manifest_step",
        lambda **_: SimpleNamespace(name="run_manifest"),
    )
    monkeypatch.setattr(
        qsplit240_olmo_base_easy_overlap_rerun,
        "create_cache_eval_datasets_step",
        lambda **_: SimpleNamespace(name="cache_eval_datasets"),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(qsplit240_olmo_base_easy_overlap_rerun, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        qsplit240_olmo_base_easy_overlap_rerun,
        "build_run_specs",
        lambda: build_qsplit240_olmo_base_easy_overlap_rerun_specs()[:6],
    )
    monkeypatch.setattr(
        qsplit240_olmo_base_easy_overlap_rerun.sys,
        "argv",
        ["launcher", "--max-concurrent", "11", "--shard-count", "3", "--shard-index", "1"],
    )

    qsplit240_olmo_base_easy_overlap_rerun.main()

    assert captured["config"].max_concurrent == 11
    assert len(captured["steps"]) == 5
    assert "shard_02of03" in captured["description"]
    assert qsplit240_olmo_base_easy_overlap_rerun.DEFAULT_MAX_CONCURRENT == 12


def test_selected_baselines_olmo_base_easy_overlap_rerun_main_wires_cache_and_max_concurrent(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        selected_overlap_rerun,
        "resolve_checkpoint_roots",
        lambda run_specs, checkpoint_regions=selected_overlap_rerun.CHECKPOINT_REGIONS: [
            replace(spec, checkpoint_root=f"gs://unit-test/checkpoints/{spec.run_name}") for spec in run_specs
        ],
    )
    monkeypatch.setattr(
        selected_overlap_rerun,
        "evaluate_levanter_lm_evaluation_harness",
        lambda **kwargs: captured.setdefault("eval_kwargs", []).append(kwargs)
        or SimpleNamespace(name=kwargs["model_name"]),
    )
    monkeypatch.setattr(
        selected_overlap_rerun,
        "output_path_of",
        lambda step, filename=None: (
            f"gs://unit-test/{step.name}" if filename is None else f"gs://unit-test/{step.name}/{filename}"
        ),
    )
    monkeypatch.setattr(
        selected_overlap_rerun,
        "create_run_manifest_step",
        lambda **_: SimpleNamespace(name="run_manifest"),
    )

    def fake_create_cache_eval_datasets_step(**kwargs):
        captured["cache_kwargs"] = kwargs
        return SimpleNamespace(name="cache_eval_datasets")

    monkeypatch.setattr(
        selected_overlap_rerun,
        "create_cache_eval_datasets_step",
        fake_create_cache_eval_datasets_step,
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(selected_overlap_rerun, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        selected_overlap_rerun,
        "build_run_specs",
        build_selected_baselines_olmo_base_easy_overlap_rerun_specs,
    )
    monkeypatch.setattr(
        selected_overlap_rerun.sys,
        "argv",
        ["launcher", "--max-concurrent", "2", "--tpu-region", "us-central1", "--tpu-zone", "us-central1-a"],
    )

    selected_overlap_rerun.main()

    assert captured["config"].max_concurrent == 2
    assert len(captured["steps"]) == 6
    assert "selected-baseline OLMoBaseEval-overlap rerun" in captured["description"]
    assert captured["cache_kwargs"]["gcs_path"].startswith("gs://marin-us-central1/")
    assert all(
        kwargs["eval_datasets_cache_path"].startswith("gs://marin-us-central1/") for kwargs in captured["eval_kwargs"]
    )
    assert selected_overlap_rerun.DEFAULT_MAX_CONCURRENT == 3


def test_run00097_olmo_base_easy_overlap_rerun_main_wires_cache_and_max_concurrent(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        run00097_olmo_base_easy_overlap_rerun,
        "resolve_checkpoint_roots",
        lambda run_specs: [
            replace(spec, checkpoint_root=f"gs://unit-test/checkpoints/{spec.run_name}") for spec in run_specs
        ],
    )
    monkeypatch.setattr(
        run00097_olmo_base_easy_overlap_rerun,
        "evaluate_levanter_lm_evaluation_harness",
        lambda **kwargs: SimpleNamespace(name=kwargs["model_name"]),
    )
    monkeypatch.setattr(
        run00097_olmo_base_easy_overlap_rerun,
        "output_path_of",
        lambda step, filename=None: (
            f"gs://unit-test/{step.name}" if filename is None else f"gs://unit-test/{step.name}/{filename}"
        ),
    )
    monkeypatch.setattr(
        run00097_olmo_base_easy_overlap_rerun,
        "create_run_manifest_step",
        lambda **_: SimpleNamespace(name="run_manifest"),
    )
    monkeypatch.setattr(
        run00097_olmo_base_easy_overlap_rerun,
        "create_cache_eval_datasets_step",
        lambda **_: SimpleNamespace(name="cache_eval_datasets"),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(run00097_olmo_base_easy_overlap_rerun, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        run00097_olmo_base_easy_overlap_rerun,
        "build_run_specs",
        lambda: build_run_00097_olmo_base_easy_overlap_rerun_specs()[:6],
    )
    monkeypatch.setattr(
        run00097_olmo_base_easy_overlap_rerun.sys,
        "argv",
        ["launcher", "--max-concurrent", "7"],
    )

    run00097_olmo_base_easy_overlap_rerun.main()

    assert captured["config"].max_concurrent == 7
    assert len(captured["steps"]) == 9
    assert "run_00097 OLMoBaseEval-overlap rerun" in captured["description"]
    assert run00097_olmo_base_easy_overlap_rerun.DEFAULT_MAX_CONCURRENT == 10


def test_run00097_300m_parity_rerun_main_wires_cache_and_max_concurrent(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        run00097_300m_parity_rerun,
        "resolve_checkpoint_roots",
        lambda run_specs, checkpoint_regions=("us-east5",): [
            replace(spec, checkpoint_root=f"gs://unit-test/checkpoints/{spec.run_name}") for spec in run_specs
        ],
    )
    monkeypatch.setattr(
        run00097_300m_parity_rerun,
        "evaluate_levanter_lm_evaluation_harness",
        lambda **kwargs: SimpleNamespace(name=kwargs["model_name"]),
    )
    monkeypatch.setattr(
        run00097_300m_parity_rerun,
        "output_path_of",
        lambda step, filename=None: (
            f"gs://unit-test/{step.name}" if filename is None else f"gs://unit-test/{step.name}/{filename}"
        ),
    )
    monkeypatch.setattr(
        run00097_300m_parity_rerun,
        "create_run_manifest_step",
        lambda **_: SimpleNamespace(name="run_manifest"),
    )
    monkeypatch.setattr(
        run00097_300m_parity_rerun,
        "create_cache_eval_datasets_step",
        lambda **_: SimpleNamespace(name="cache_eval_datasets"),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(run00097_300m_parity_rerun, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        run00097_300m_parity_rerun,
        "build_run_specs",
        lambda: build_run_00097_300m_parity_rerun_specs()[:6],
    )
    monkeypatch.setattr(
        run00097_300m_parity_rerun.sys,
        "argv",
        ["launcher", "--max-concurrent", "4"],
    )

    run00097_300m_parity_rerun.main()

    assert captured["config"].max_concurrent == 4
    assert len(captured["steps"]) == 9
    assert "run_00097 300M parity rerun" in captured["description"]


def test_genericfamily_subset_optimum_run_name_formats_subset_size():
    assert genericfamily_subset_optimum_run_name(20) == "baseline_genericfamily_k020_uncheatable_bpb"
    assert genericfamily_subset_optimum_run_name(220) == "baseline_genericfamily_k220_uncheatable_bpb"
    assert GENERICFAMILY_SUBSET_OPTIMA_SOURCE_EXPERIMENT.endswith("genericfamily_subset_optima_uncheatable_bpb")


def test_genericfamily_subset_optima_main_wires_max_concurrent_and_training_steps(monkeypatch):
    captured: dict[str, object] = {}

    fake_summaries = [
        SimpleNamespace(
            subset_size=20,
            run_id=320,
            run_name="baseline_genericfamily_k020_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.01,
            nearest_observed_tv_distance=0.4,
            optimum_move_mean_phase_tv_vs_prev=None,
        ),
        SimpleNamespace(
            subset_size=40,
            run_id=321,
            run_name="baseline_genericfamily_k040_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.02,
            nearest_observed_tv_distance=0.3,
            optimum_move_mean_phase_tv_vs_prev=0.2,
        ),
    ]

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        genericfamily_subset_optima_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        genericfamily_subset_optima_launch,
        "genericfamily_subset_optima_summaries",
        lambda: tuple(fake_summaries),
    )
    monkeypatch.setattr(
        genericfamily_subset_optima_launch,
        "genericfamily_subset_optima_summaries_json",
        lambda: "[]",
    )
    monkeypatch.setattr(
        genericfamily_subset_optima_launch,
        "genericfamily_subset_optima_summaries_frame",
        lambda: pd.DataFrame([{"subset_size": 20}, {"subset_size": 40}]),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(genericfamily_subset_optima_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        genericfamily_subset_optima_launch.sys,
        "argv",
        ["launcher", "--max-concurrent", "5"],
    )

    genericfamily_subset_optima_launch.main()

    assert captured["config"].max_concurrent == 5
    assert len(captured["steps"]) == 3
    assert "subset-fit predicted optima" in captured["description"]
    assert genericfamily_subset_optima_launch.DEFAULT_MAX_CONCURRENT == 4


def test_genericfamily_retuned_subset_optimum_run_name_and_parse_subset_sizes():
    assert genericfamily_retuned_subset_optimum_run_name(40) == "baseline_genericfamily_retuned_k040_uncheatable_bpb"
    assert genericfamily_retuned_subset_optimum_run_name(220) == "baseline_genericfamily_retuned_k220_uncheatable_bpb"
    assert parse_retuned_subset_sizes("40,100,180,220") == (40, 100, 180, 220)
    assert parse_retuned_subset_sizes("all") == tuple(range(20, 240, 20))
    assert GENERICFAMILY_RETUNED_SUBSET_OPTIMA_SOURCE_EXPERIMENT.endswith(
        "genericfamily_retuned_subset_optima_rep_uncheatable_bpb"
    )


def test_genericfamily_retuned_subset_optima_main_wires_subset_sizes_and_training_steps(monkeypatch):
    captured: dict[str, object] = {}

    fake_summaries = [
        SimpleNamespace(
            subset_size=40,
            run_id=341,
            run_name="baseline_genericfamily_retuned_k040_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.01,
            tuning_objective=0.02,
            nearest_observed_tv_distance=0.4,
        ),
        SimpleNamespace(
            subset_size=180,
            run_id=348,
            run_name="baseline_genericfamily_retuned_k180_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.02,
            tuning_objective=0.03,
            nearest_observed_tv_distance=0.3,
        ),
    ]

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        genericfamily_retuned_subset_optima_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        genericfamily_retuned_subset_optima_launch,
        "genericfamily_retuned_subset_optima_summaries",
        lambda subset_sizes: tuple(fake_summaries),
    )
    monkeypatch.setattr(
        genericfamily_retuned_subset_optima_launch,
        "genericfamily_retuned_subset_optima_summaries_json",
        lambda subset_sizes: "[]",
    )
    monkeypatch.setattr(
        genericfamily_retuned_subset_optima_launch,
        "genericfamily_retuned_subset_optima_summaries_frame",
        lambda subset_sizes: pd.DataFrame([{"subset_size": 40}, {"subset_size": 180}]),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(genericfamily_retuned_subset_optima_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        genericfamily_retuned_subset_optima_launch.sys,
        "argv",
        ["launcher", "--subset-sizes", "40,180", "--max-concurrent", "3"],
    )

    genericfamily_retuned_subset_optima_launch.main()

    assert captured["config"].max_concurrent == 3
    assert len(captured["steps"]) == 3
    assert "retuned GRP subset-fit predicted optima" in captured["description"]
    assert genericfamily_retuned_subset_optima_launch.DEFAULT_MAX_CONCURRENT == 4


def test_genericfamily_recovered_hull_subset_optimum_run_name_and_source_experiment():
    assert (
        genericfamily_recovered_hull_subset_optimum_run_name(20)
        == "baseline_genericfamily_recovered_hull_k020_uncheatable_bpb"
    )
    assert (
        genericfamily_recovered_hull_subset_optimum_run_name(220)
        == "baseline_genericfamily_recovered_hull_k220_uncheatable_bpb"
    )
    assert GENERICFAMILY_RECOVERED_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT.endswith(
        "genericfamily_recovered_hull_subset_optima_uncheatable_bpb"
    )


def test_genericfamily_recovered_hull_subset_optima_main_wires_subset_sizes_and_training_steps(monkeypatch):
    captured: dict[str, object] = {}

    fake_summaries = [
        SimpleNamespace(
            subset_size=40,
            run_id=361,
            run_name="baseline_genericfamily_recovered_hull_k040_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.04,
            tuning_objective=0.02,
            nearest_observed_tv_distance=0.4,
        ),
        SimpleNamespace(
            subset_size=180,
            run_id=368,
            run_name="baseline_genericfamily_recovered_hull_k180_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.05,
            tuning_objective=0.03,
            nearest_observed_tv_distance=0.3,
        ),
    ]

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        genericfamily_recovered_hull_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        genericfamily_recovered_hull_launch,
        "genericfamily_recovered_hull_subset_optima_summaries",
        lambda subset_sizes: tuple(fake_summaries),
    )
    monkeypatch.setattr(
        genericfamily_recovered_hull_launch,
        "genericfamily_recovered_hull_subset_optima_summaries_json",
        lambda subset_sizes: "[]",
    )
    monkeypatch.setattr(
        genericfamily_recovered_hull_launch,
        "genericfamily_recovered_hull_subset_optima_summaries_frame",
        lambda subset_sizes: pd.DataFrame([{"subset_size": 40}, {"subset_size": 180}]),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(genericfamily_recovered_hull_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        genericfamily_recovered_hull_launch.sys,
        "argv",
        ["launcher", "--subset-sizes", "40,180", "--max-concurrent", "3"],
    )

    genericfamily_recovered_hull_launch.main()

    assert captured["config"].max_concurrent == 3
    assert len(captured["steps"]) == 3
    assert "recovered-hull GRP subset-fit predicted optima" in captured["description"]


def test_genericfamily_top8actual_hull_subset_optimum_run_name_and_source_experiment():
    assert (
        genericfamily_top8actual_hull_subset_optimum_run_name(40)
        == "baseline_genericfamily_top8actual_hull_k040_uncheatable_bpb"
    )
    assert (
        genericfamily_top8actual_hull_subset_optimum_run_name(220)
        == "baseline_genericfamily_top8actual_hull_k220_uncheatable_bpb"
    )
    assert GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT.endswith(
        "genericfamily_top8actual_hull_subset_optima_rep_uncheatable_bpb"
    )


def test_genericfamily_top8actual_hull_subset_optima_main_wires_subset_sizes_and_training_steps(monkeypatch):
    captured: dict[str, object] = {}

    fake_summaries = [
        SimpleNamespace(
            subset_size=40,
            run_id=381,
            run_name="baseline_genericfamily_top8actual_hull_k040_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.04,
            tuning_objective=0.02,
            nearest_observed_tv_distance=0.4,
        ),
        SimpleNamespace(
            subset_size=180,
            run_id=388,
            run_name="baseline_genericfamily_top8actual_hull_k180_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.05,
            tuning_objective=0.03,
            nearest_observed_tv_distance=0.3,
        ),
    ]

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        genericfamily_top8actual_hull_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        genericfamily_top8actual_hull_launch,
        "genericfamily_top8actual_hull_subset_optima_summaries",
        lambda subset_sizes: tuple(fake_summaries),
    )
    monkeypatch.setattr(
        genericfamily_top8actual_hull_launch,
        "genericfamily_top8actual_hull_subset_optima_summaries_json",
        lambda subset_sizes: "[]",
    )
    monkeypatch.setattr(
        genericfamily_top8actual_hull_launch,
        "genericfamily_top8actual_hull_subset_optima_summaries_frame",
        lambda subset_sizes: pd.DataFrame([{"subset_size": 40}, {"subset_size": 180}]),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(genericfamily_top8actual_hull_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        genericfamily_top8actual_hull_launch.sys,
        "argv",
        ["launcher", "--subset-sizes", "40,180", "--max-concurrent", "3"],
    )

    genericfamily_top8actual_hull_launch.main()

    assert captured["config"].max_concurrent == 3
    assert len(captured["steps"]) == 3
    assert "top-8-actual-hull GRP subset-fit predicted optima" in captured["description"]
    assert genericfamily_recovered_hull_launch.DEFAULT_MAX_CONCURRENT == 4


def test_genericfamily_observed_only_trustblend_subset_optimum_run_name_and_source_experiment():
    assert (
        genericfamily_observed_only_trustblend_subset_optimum_run_name(20)
        == "baseline_genericfamily_trustblend_top8actual_cap_k020_uncheatable_bpb"
    )
    assert (
        genericfamily_observed_only_trustblend_subset_optimum_run_name(220)
        == "baseline_genericfamily_trustblend_top8actual_cap_k220_uncheatable_bpb"
    )
    assert GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_SOURCE_EXPERIMENT.endswith(
        "genericfamily_observed_only_trustblend_top8actual_cap_subset_optima_rep_uncheatable_bpb"
    )


def test_genericfamily_observed_only_trustblend_subset_optima_main_wires_subset_sizes_and_training_steps(
    monkeypatch,
):
    captured: dict[str, object] = {}

    fake_summaries = [
        SimpleNamespace(
            subset_size=20,
            run_id=400,
            run_name="baseline_genericfamily_trustblend_top8actual_cap_k020_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.05,
            deployment_delta=0.05,
            deployment_gain_budget=0.01,
            nearest_observed_tv_distance=0.2,
        ),
        SimpleNamespace(
            subset_size=80,
            run_id=403,
            run_name="baseline_genericfamily_trustblend_top8actual_cap_k080_uncheatable_bpb",
            phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}},
            predicted_optimum_value=1.04,
            deployment_delta=0.12,
            deployment_gain_budget=0.02,
            nearest_observed_tv_distance=0.3,
        ),
    ]

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        obs_trustblend_subset_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        obs_trustblend_subset_launch,
        "genericfamily_observed_only_trustblend_subset_optima_summaries",
        lambda subset_sizes: tuple(fake_summaries),
    )
    monkeypatch.setattr(
        obs_trustblend_subset_launch,
        "genericfamily_observed_only_trustblend_subset_optima_summaries_json",
        lambda subset_sizes: "[]",
    )
    monkeypatch.setattr(
        obs_trustblend_subset_launch,
        "genericfamily_observed_only_trustblend_subset_optima_summaries_frame",
        lambda subset_sizes: pd.DataFrame([{"subset_size": 20}, {"subset_size": 80}]),
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(obs_trustblend_subset_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        obs_trustblend_subset_launch.sys,
        "argv",
        ["launcher", "--subset-sizes", "20,80", "--max-concurrent", "3"],
    )

    obs_trustblend_subset_launch.main()

    assert captured["config"].max_concurrent == 3
    assert len(captured["steps"]) == 3
    assert "observed-only trustblend GRP subset-fit predicted optima" in captured["description"]
    assert obs_trustblend_subset_launch.DEFAULT_MAX_CONCURRENT == 4


def test_genericfamily_observed_only_trustblend_baseline_run_name_and_source_experiment():
    assert (
        GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME
        == "baseline_genericfamily_observed_only_trustblend_top8actual_cap"
    )
    assert GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT.endswith(
        "genericfamily_observed_only_trustblend_top8actual_cap_uncheatable_bpb"
    )


def test_genericfamily_observed_only_trustblend_baseline_main_wires_training_step(monkeypatch):
    captured: dict[str, object] = {}

    fake_summary = SimpleNamespace(
        run_name="baseline_genericfamily_observed_only_trustblend_top8actual_cap",
        predicted_optimum_value=1.04,
        deployment_delta=0.12,
        deployment_gain_budget=0.01,
        nearest_observed_tv_distance=0.2,
    )
    fake_weight_config = WeightConfig(run_id=407, phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}})

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        obs_trustblend_baseline_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        obs_trustblend_baseline_launch,
        "genericfamily_observed_only_trustblend_summary",
        lambda: fake_summary,
    )
    monkeypatch.setattr(
        obs_trustblend_baseline_launch,
        "create_genericfamily_observed_only_trustblend_weight_config",
        lambda: fake_weight_config,
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(obs_trustblend_baseline_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        obs_trustblend_baseline_launch.sys,
        "argv",
        ["launcher"],
    )

    obs_trustblend_baseline_launch.main()

    assert captured["config"].max_concurrent == 1
    assert len(captured["steps"]) == 1
    assert "observed-only trustblend GRP baseline" in captured["description"]


def test_genericfamily_power_observed_only_trustblend_baseline_run_name_and_source_experiment():
    assert (
        GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME
        == "baseline_genericfamily_power_observed_only_trustblend_top8actual_cap"
    )
    assert GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT.endswith(
        "genericfamily_power_observed_only_trustblend_top8actual_cap_uncheatable_bpb"
    )


def test_genericfamily_power_observed_only_trustblend_baseline_main_wires_training_step(monkeypatch):
    captured: dict[str, object] = {}

    fake_summary = SimpleNamespace(
        run_name="baseline_genericfamily_power_observed_only_trustblend_top8actual_cap",
        predicted_optimum_value=1.044,
        deployment_delta=0.112,
        deployment_gain_budget=0.012,
        nearest_observed_tv_distance=0.308,
    )
    fake_weight_config = WeightConfig(run_id=408, phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}})

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        power_obs_trustblend_baseline_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        power_obs_trustblend_baseline_launch,
        "genericfamily_power_observed_only_trustblend_summary",
        lambda: fake_summary,
    )
    monkeypatch.setattr(
        power_obs_trustblend_baseline_launch,
        "create_genericfamily_power_observed_only_trustblend_weight_config",
        lambda: fake_weight_config,
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(power_obs_trustblend_baseline_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        power_obs_trustblend_baseline_launch.sys,
        "argv",
        ["launcher"],
    )

    power_obs_trustblend_baseline_launch.main()

    assert captured["config"].max_concurrent == 1
    assert len(captured["steps"]) == 1
    assert "observed-only trustblend power-law GRP baseline" in captured["description"]


def test_genericfamily_no_penalty_baseline_run_name_and_source_experiment():
    assert GENERICFAMILY_NO_PENALTY_RUN_NAME == "baseline_genericfamily_no_penalty_uncheatable_bpb"
    assert GENERICFAMILY_NO_PENALTY_SOURCE_EXPERIMENT.endswith("genericfamily_no_penalty_uncheatable_bpb")


def test_genericfamily_no_penalty_baseline_main_wires_training_step(monkeypatch):
    captured: dict[str, object] = {}

    fake_summary = {
        "run_name": "baseline_genericfamily_no_penalty_uncheatable_bpb",
        "predicted_optimum_value": 1.04,
        "nearest_observed_tv_distance": 0.2,
    }
    fake_weight_config = WeightConfig(run_id=258, phase_weights={"phase_0": {"a": 1.0}, "phase_1": {"a": 1.0}})

    class DummyExperiment:
        def create_training_step(self, **kwargs):
            return SimpleNamespace(name=kwargs["run_name"], kwargs=kwargs)

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        genericfamily_no_penalty_launch,
        "create_two_phase_dolma3_dolmino_top_level_experiment",
        lambda **_: DummyExperiment(),
    )
    monkeypatch.setattr(
        genericfamily_no_penalty_launch,
        "genericfamily_no_penalty_summary",
        lambda: fake_summary,
    )
    monkeypatch.setattr(
        genericfamily_no_penalty_launch,
        "genericfamily_no_penalty_summary_json",
        lambda: "{}",
    )
    monkeypatch.setattr(
        genericfamily_no_penalty_launch,
        "create_genericfamily_no_penalty_weight_config",
        lambda: fake_weight_config,
    )

    def fake_executor_main(config=None, *, steps, description):
        captured["config"] = config
        captured["steps"] = steps
        captured["description"] = description

    monkeypatch.setattr(genericfamily_no_penalty_launch, "executor_main", fake_executor_main)
    monkeypatch.setattr(
        genericfamily_no_penalty_launch.sys,
        "argv",
        ["launcher"],
    )

    genericfamily_no_penalty_launch.main()

    assert captured["config"] is None
    assert len(captured["steps"]) == 2
    assert "genericfamily_no_penalty_uncheatable_bpb" in captured["description"]


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


def test_qsplit240_520m_pilot_analysis_reports_panel_ranks_and_projection(tmp_path, monkeypatch):
    panel_runs = list(REPRESENTATIVE12_PANEL_RUN_NAMES)
    results_csv = tmp_path / "pilot_results.csv"
    source_60m_csv = tmp_path / "two_phase_many.csv"
    source_300m_csv = tmp_path / "completed_300m.csv"

    pd.DataFrame(
        {
            "run_name": panel_runs,
            "eval/uncheatable_eval/bpb": np.linspace(1.00, 1.11, len(panel_runs)),
        }
    ).to_csv(source_60m_csv, index=False)
    pd.DataFrame(
        {
            "run_name": panel_runs[:3],
            "bpb_300m_6b": [0.98, 1.02, 1.00],
        }
    ).to_csv(source_300m_csv, index=False)
    pd.DataFrame(
        {
            "run_name": panel_runs,
            "eval/uncheatable_eval/bpb": np.linspace(0.91, 1.02, len(panel_runs))[::-1],
            "wandb_run_id": [f"pilot-{idx}" for idx in range(len(panel_runs))],
            "status": ["completed"] * len(panel_runs),
        }
    ).to_csv(results_csv, index=False)

    monkeypatch.setattr(
        qsplit240_520m_analysis,
        "active_compute_hours_for_run",
        lambda run_id, **_: float(int(run_id.split("-")[-1]) + 1),
    )

    comparison_frame, summary = qsplit240_520m_analysis.build_pilot_comparison_frame(
        pilot_results_csv=results_csv,
        source_60m_csv=source_60m_csv,
        source_300m_csv=source_300m_csv,
    )

    assert summary["panel_size"] == 12
    assert summary["completed_520m_runs"] == 12
    assert summary["completed_300m_shared_runs"] == 3
    assert summary["best_60m_run_name"] == panel_runs[0]
    assert summary["best_520m_run_name"] == panel_runs[-1]
    assert summary["active_compute_hours_mean_520m"] == pytest.approx(6.5)
    assert summary["projected_serialized_full_240_hours"] == pytest.approx(1560.0)

    best_row = comparison_frame.iloc[0]
    assert best_row["run_name"] == panel_runs[-1]
    assert best_row["rank_520m_in_panel"] == pytest.approx(1.0)
    assert comparison_frame.loc[
        comparison_frame["run_name"] == panel_runs[0], "rank_60m_in_panel"
    ].item() == pytest.approx(1.0)
    assert np.isnan(comparison_frame.loc[comparison_frame["run_name"] == panel_runs[5], "rank_300m_6b_in_panel"].item())
