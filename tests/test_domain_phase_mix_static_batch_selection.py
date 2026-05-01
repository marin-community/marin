# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    KNOWN_STRATIFIED_60M_METRICS,
    append_two_phase_many_stratified_baseline,
    build_two_phase_many_loop_config,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import STRATIFIED_RUN_NAME
from experiments.domain_phase_mix.nextgen.contracts import LoopConfig, LoopState, RunRecord
from experiments.domain_phase_mix.nextgen.design import plan_new_runs
from experiments.domain_phase_mix.nextgen.model_registry import _build_dataset_spec
from experiments.domain_phase_mix.starcoder_metadata import (
    DEFAULT_STARCODER_OBJECTIVE,
    DOMAIN_NAMES,
    NEMOTRON_TOKENS,
    STARCODER_TOKENS,
    TARGET_BUDGET,
    load_three_phase_starcoder_dataset,
    load_two_phase_starcoder_dataset,
)
from experiments.domain_phase_mix.static_batch_selection import (
    build_schedule_feature_matrix,
    build_dataset_spec_from_frame,
    prospective_generic_selection,
    prospective_d_optimal_selection,
    replay_proposals_to_observed,
    retrospective_generic_selection,
    retrospective_d_optimal_selection,
    run_records_to_dataframe,
    sobol_weight_configs,
    weight_configs_to_tensor,
)
from experiments.domain_phase_mix.scaling_study_recipes import (
    BASE_TARGET_BUDGET,
    ScalingStudyScale,
    resolve_scale_spec,
)
from experiments.domain_phase_mix.weight_sampler import DirichletSamplingParams, WeightSampler


class _FakeExperiment:
    def __init__(self, phase_names, domain_names, min_config_distance=0.05):
        self.phase_names = phase_names
        self.domain_names = domain_names
        self.min_config_distance = min_config_distance

    def create_weight_sampler(self, seed=42):
        return WeightSampler(
            domain_names=list(self.domain_names),
            phase_names=list(self.phase_names),
            seed=seed,
            params=DirichletSamplingParams(
                min_weight=0.05,
                min_config_distance=self.min_config_distance,
            ),
        )


def _synthetic_two_phase_spec() -> DatasetSpec:
    weights = np.array(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.8, 0.2], [0.6, 0.4]],
            [[0.7, 0.3], [0.5, 0.5]],
            [[0.6, 0.4], [0.4, 0.6]],
            [[0.5, 0.5], [0.3, 0.7]],
            [[0.4, 0.6], [0.2, 0.8]],
            [[0.3, 0.7], [0.1, 0.9]],
            [[0.2, 0.8], [0.05, 0.95]],
        ]
    )
    return DatasetSpec(
        weights=weights,
        y=np.linspace(0.9, 0.6, len(weights)),
        epoch_multipliers=np.array([[1.0, 10.0], [1.0, 10.0]]),
        domain_names=list(DOMAIN_NAMES),
        phase_names=["phase_0", "phase_1"],
        small_domains=[1],
        name="synthetic",
    )


def _synthetic_three_phase_spec() -> DatasetSpec:
    weights = np.array(
        [
            [[0.95, 0.05], [0.9, 0.1], [0.85, 0.15]],
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
            [[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]],
            [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8]],
            [[0.2, 0.8], [0.1, 0.9], [0.05, 0.95]],
        ]
    )
    return DatasetSpec(
        weights=weights,
        y=np.linspace(0.8, 0.65, len(weights)),
        epoch_multipliers=np.array([[1.0, 9.0], [1.0, 9.5], [1.0, 10.0]]),
        domain_names=list(DOMAIN_NAMES),
        phase_names=["phase_0", "phase_1", "phase_2"],
        small_domains=[1],
        name="synthetic_three_phase",
    )


def _phase_weights_to_run_record(row: pd.Series) -> RunRecord:
    phase_weights = {
        "phase_0": {
            "nemotron_full": float(row["phase_0_nemotron_full"]),
            "starcoder": float(row["phase_0_starcoder"]),
        },
        "phase_1": {
            "nemotron_full": float(row["phase_1_nemotron_full"]),
            "starcoder": float(row["phase_1_starcoder"]),
        },
    }
    return RunRecord(
        wandb_run_id=str(row["run_id"]),
        source_experiment="two_phase_starcoder",
        local_run_id=int(row["run_id"]),
        run_name=f"run_{int(row['run_id']):05d}",
        phase_weights=phase_weights,
        status="completed",
        metrics={DEFAULT_STARCODER_OBJECTIVE: float(row[DEFAULT_STARCODER_OBJECTIVE])},
    )


def test_two_phase_loader_uses_real_epoch_multipliers():
    spec, _ = load_two_phase_starcoder_dataset()

    assert spec.R == 116
    assert spec.phase_names == ["phase_0", "phase_1"]
    assert spec.small_domains == [1]
    np.testing.assert_allclose(
        spec.epoch_multipliers,
        np.array(
            [
                [0.5 * TARGET_BUDGET / NEMOTRON_TOKENS, 0.5 * TARGET_BUDGET / STARCODER_TOKENS],
                [0.5 * TARGET_BUDGET / NEMOTRON_TOKENS, 0.5 * TARGET_BUDGET / STARCODER_TOKENS],
            ]
        ),
    )


def test_append_two_phase_many_stratified_baseline_adds_known_metric_row():
    frame = pd.DataFrame(
        [
            {
                "run_id": 0,
                "run_name": "baseline_proportional",
                "status": "completed",
                "phase_0_dolma3_cc/adult_content_high": 1.0,
                "phase_1_dolma3_cc/adult_content_high": 1.0,
                "eval/uncheatable_eval/bpb": 1.1,
            }
        ]
    )

    augmented = append_two_phase_many_stratified_baseline(frame, objective_metric="eval/uncheatable_eval/bpb")

    assert len(augmented) == 2
    appended = augmented.loc[augmented["run_name"] == STRATIFIED_RUN_NAME].iloc[0]
    assert float(appended["eval/uncheatable_eval/bpb"]) == KNOWN_STRATIFIED_60M_METRICS["eval/uncheatable_eval/bpb"]
    phase0_cols = [col for col in augmented.columns if col.startswith("phase_0_")]
    phase1_cols = [col for col in augmented.columns if col.startswith("phase_1_")]
    assert int(appended[phase0_cols].notna().sum()) == 39
    assert int(appended[phase1_cols].notna().sum()) == 39
    assert np.isclose(appended[phase0_cols].dropna().astype(float).sum(), 1.0)
    assert np.isclose(appended[phase1_cols].dropna().astype(float).sum(), 1.0)


def test_append_two_phase_many_stratified_baseline_skips_unknown_metric():
    frame = pd.DataFrame([{"run_name": "baseline_proportional", "status": "completed"}])

    augmented = append_two_phase_many_stratified_baseline(frame, objective_metric="choice_logprob_norm_mean")

    pd.testing.assert_frame_equal(augmented, frame)


def test_three_phase_loader_uses_real_epoch_multipliers():
    spec, _ = load_three_phase_starcoder_dataset()

    assert spec.R == 160
    assert spec.phase_names == ["phase_0", "phase_1", "phase_2"]
    assert spec.small_domains == [1]
    np.testing.assert_allclose(
        spec.epoch_multipliers[:, 1],
        np.array([0.33, 0.34, 0.33]) * TARGET_BUDGET / STARCODER_TOKENS,
    )


def test_sobol_weight_configs_are_deterministic_and_respect_constraints():
    experiment = _FakeExperiment(["phase_0", "phase_1"], DOMAIN_NAMES, min_config_distance=0.05)

    configs_a = sobol_weight_configs(experiment, n=6, seed=7)
    configs_b = sobol_weight_configs(experiment, n=6, seed=7)

    assert [cfg.phase_weights for cfg in configs_a] == [cfg.phase_weights for cfg in configs_b]

    weights = weight_configs_to_tensor(configs_a, phase_names=["phase_0", "phase_1"], domain_names=DOMAIN_NAMES)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0)

    sampler = experiment.create_weight_sampler(seed=0)
    pairwise = [
        sampler._config_distance(configs_a[i], configs_a[j])
        for i in range(len(configs_a))
        for j in range(i + 1, len(configs_a))
    ]
    assert min(pairwise) >= sampler.params.min_config_distance - 1e-9


def test_sobol_weight_configs_support_large_candidate_pools():
    experiment = _FakeExperiment(["phase_0", "phase_1"], DOMAIN_NAMES, min_config_distance=0.0)

    configs = sobol_weight_configs(experiment, n=300, seed=11)

    assert len(configs) == 300
    weights = weight_configs_to_tensor(configs, phase_names=["phase_0", "phase_1"], domain_names=DOMAIN_NAMES)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0)


def test_sobol_weight_configs_can_return_feasible_partial_pool():
    experiment = _FakeExperiment(["phase_0", "phase_1"], DOMAIN_NAMES, min_config_distance=10.0)

    configs = sobol_weight_configs(experiment, n=32, seed=3, min_accepted=1)

    assert len(configs) >= 1
    assert len(configs) < 32
    weights = weight_configs_to_tensor(configs, phase_names=["phase_0", "phase_1"], domain_names=DOMAIN_NAMES)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0)


def test_replay_proposals_to_observed_uses_unique_matches():
    observed = np.array(
        [
            [[1.0, 0.0]],
            [[0.5, 0.5]],
            [[0.0, 1.0]],
        ]
    )
    proposals = np.array(
        [
            [[0.45, 0.55]],
            [[0.1, 0.9]],
        ]
    )

    replay = replay_proposals_to_observed(proposals, observed)

    assert replay.selected_indices == [1, 2]
    assert replay.mean_distance >= 0.0
    assert replay.max_distance >= replay.mean_distance


def test_retrospective_doptimal_selection_is_seed_stable():
    spec = _synthetic_two_phase_spec()

    selection_a = retrospective_d_optimal_selection(spec, k=3, seed=4)
    selection_b = retrospective_d_optimal_selection(spec, k=3, seed=4)

    assert selection_a.selected_indices == selection_b.selected_indices


def test_build_schedule_feature_matrix_is_deterministic_and_nan_safe():
    spec = _synthetic_three_phase_spec()
    bundle_a = build_schedule_feature_matrix(spec.weights, spec.epoch_multipliers, spec.small_domains)
    bundle_b = build_schedule_feature_matrix(spec.weights, spec.epoch_multipliers, spec.small_domains)

    assert bundle_a.raw_matrix.shape == (5, 30)
    assert bundle_a.standardized_matrix.shape == (5, 30)
    assert bundle_a.feature_names == bundle_b.feature_names
    np.testing.assert_allclose(bundle_a.raw_matrix, bundle_b.raw_matrix)
    np.testing.assert_allclose(bundle_a.standardized_matrix, bundle_b.standardized_matrix)
    assert np.isfinite(bundle_a.standardized_matrix).all()


def test_generic_retrospective_selectors_are_deterministic():
    spec = _synthetic_two_phase_spec()
    methods = ("feature_maximin", "feature_dpp", "feature_bayes_linear")

    for method in methods:
        selection_a = retrospective_generic_selection(spec, method=method, k=3, seed=9)
        selection_b = retrospective_generic_selection(spec, method=method, k=3, seed=9)
        assert selection_a.selected_indices == selection_b.selected_indices
        assert len(selection_a.selected_indices) == 3
        assert len(set(selection_a.selected_indices)) == 3


def test_generic_retrospective_dpp_is_stable_with_duplicate_rows():
    spec = _synthetic_two_phase_spec()
    duplicated = DatasetSpec(
        weights=np.concatenate([spec.weights, spec.weights[[2]]], axis=0),
        y=np.concatenate([spec.y, spec.y[[2]]], axis=0),
        epoch_multipliers=spec.epoch_multipliers,
        domain_names=spec.domain_names,
        phase_names=spec.phase_names,
        small_domains=spec.small_domains,
        name="duplicated",
    )

    selection = retrospective_generic_selection(duplicated, method="feature_dpp", k=4)

    assert len(selection.selected_indices) == 4
    assert len(set(selection.selected_indices)) == 4


def test_generic_retrospective_dpp_is_stable_with_identical_pool():
    spec = _synthetic_two_phase_spec()
    identical = DatasetSpec(
        weights=np.repeat(spec.weights[[0]], repeats=6, axis=0),
        y=np.linspace(0.8, 0.7, 6),
        epoch_multipliers=spec.epoch_multipliers,
        domain_names=spec.domain_names,
        phase_names=spec.phase_names,
        small_domains=spec.small_domains,
        name="identical",
    )

    selection = retrospective_generic_selection(identical, method="feature_dpp", k=4)

    assert len(selection.selected_indices) == 4
    assert len(set(selection.selected_indices)) == 4
    assert selection.diagnostics["dpp_fallback_steps"] >= 0.0


def test_nextgen_design_policy_can_plan_static_doptimal_runs():
    df = pd.read_csv("experiments/domain_phase_mix/exploratory/two_phase_starcoder_combined.csv")
    df = df[df["status"] == "completed"].head(20).reset_index(drop=True)
    state = LoopState(
        loop_name="loop",
        objective_metric=DEFAULT_STARCODER_OBJECTIVE,
        next_local_run_id=20,
        runs=[_phase_weights_to_run_record(row) for _, row in df.iterrows()],
    )
    loop = LoopConfig(
        name="loop",
        objective_metric=DEFAULT_STARCODER_OBJECTIVE,
        model_names=("DS-RE-CEQ",),
        n_new_runs=2,
        design_policy="static_d_optimal",
        design_candidate_pool_size=32,
        candidate_search_seed=3,
    )
    experiment = _FakeExperiment(["phase_0", "phase_1"], DOMAIN_NAMES, min_config_distance=0.01)

    planned = plan_new_runs(loop, experiment, state)

    assert len(planned) == 2
    assert [item.local_run_id for item in planned] == [20, 21]
    for item in planned:
        assert set(item.phase_weights) == {"phase_0", "phase_1"}
        for phase_weights in item.phase_weights.values():
            assert abs(sum(phase_weights.values()) - 1.0) < 1e-9


def test_nextgen_model_registry_uses_starcoder_epoch_metadata():
    df = pd.read_csv("experiments/domain_phase_mix/exploratory/two_phase_starcoder_combined.csv")
    df = df[df["status"] == "completed"].head(8).reset_index(drop=True)
    loop = LoopConfig(name="loop", objective_metric=DEFAULT_STARCODER_OBJECTIVE, model_names=())

    spec = _build_dataset_spec(df, DEFAULT_STARCODER_OBJECTIVE, loop)

    assert spec.domain_names == list(DOMAIN_NAMES)
    assert spec.small_domains == [1]
    assert not np.allclose(spec.epoch_multipliers, 1.0)


def test_build_dataset_spec_from_frame_ignores_epoch_columns():
    df = pd.read_csv("experiments/domain_phase_mix/exploratory/two_phase_starcoder_combined.csv")
    df = df[df["status"] == "completed"].head(8).reset_index(drop=True)

    spec = build_dataset_spec_from_frame(
        df,
        objective_metric=DEFAULT_STARCODER_OBJECTIVE,
        name="two_phase_starcoder_with_epochs",
    )

    assert spec.domain_names == list(DOMAIN_NAMES)
    assert spec.phase_names == ["phase_0", "phase_1"]
    assert spec.small_domains == [1]


def test_build_dataset_spec_from_frame_uses_real_epoch_metadata_for_two_phase_many():
    df = pd.read_csv(
        "experiments/domain_phase_mix/exploratory/two_phase_many/"
        "qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_candidate_summary.csv"
    ).head(4)
    loop = build_two_phase_many_loop_config(
        objective_metric="choice_logprob_norm_mean",
        name="two_phase_many_test",
    )

    spec = build_dataset_spec_from_frame(
        df,
        objective_metric="choice_logprob_norm_mean",
        name="two_phase_many_candidate_summary",
        loop=loop,
    )

    assert spec.phase_names == ["phase_0", "phase_1"]
    assert spec.small_domains == list(range(len(spec.domain_names)))
    assert spec.epoch_multipliers.shape == (2, len(spec.domain_names))
    assert not np.allclose(spec.epoch_multipliers, 1.0)

    wikipedia_idx = spec.domain_names.index("dolma3_wikipedia")
    expected_phase_0 = loop.phase_fractions[0] * loop.target_budget / loop.domain_token_counts["dolma3_wikipedia"]
    expected_phase_1 = loop.phase_fractions[1] * loop.target_budget / loop.domain_token_counts["dolma3_wikipedia"]
    np.testing.assert_allclose(spec.epoch_multipliers[:, wikipedia_idx], [expected_phase_0, expected_phase_1])


def test_scaling_study_1x_target_budgets_match_across_chinchilla_scales():
    specs = [
        resolve_scale_spec(ScalingStudyScale.REGMIX_130M_2P6B),
        resolve_scale_spec(ScalingStudyScale.REGMIX_300M_6B),
        resolve_scale_spec(ScalingStudyScale.REGMIX_520M_10P4B),
        resolve_scale_spec(ScalingStudyScale.REGMIX_1_2B_24B),
    ]
    target_budgets = {spec.target_budget_for_multiplier(1.0) for spec in specs}

    assert target_budgets == {BASE_TARGET_BUDGET}


def test_scaling_study_target_budget_multipliers_only_change_with_multiplier():
    spec_130m = resolve_scale_spec(ScalingStudyScale.REGMIX_130M_2P6B)
    spec_520m = resolve_scale_spec(ScalingStudyScale.REGMIX_520M_10P4B)
    wikipedia_tokens = build_two_phase_many_loop_config(
        objective_metric="eval/uncheatable_eval/bpb",
        name="scaling_study_target_budget_test",
    ).domain_token_counts["dolma3_wikipedia"]
    phase_fractions = build_two_phase_many_loop_config(
        objective_metric="eval/uncheatable_eval/bpb",
        name="scaling_study_target_budget_test",
    ).phase_fractions

    epoch_130m_1x = phase_fractions[0] * spec_130m.target_budget_for_multiplier(1.0) / wikipedia_tokens
    epoch_520m_1x = phase_fractions[0] * spec_520m.target_budget_for_multiplier(1.0) / wikipedia_tokens
    epoch_130m_half = phase_fractions[0] * spec_130m.target_budget_for_multiplier(0.5) / wikipedia_tokens
    epoch_520m_double = phase_fractions[0] * spec_520m.target_budget_for_multiplier(2.0) / wikipedia_tokens

    assert epoch_130m_1x == pytest.approx(epoch_520m_1x)
    assert epoch_130m_half == pytest.approx(epoch_130m_1x * 0.5, abs=1e-9)
    assert epoch_520m_double == pytest.approx(epoch_520m_1x * 2.0)


def test_run_records_to_dataframe_roundtrip_supports_starcoder_metadata():
    df = pd.read_csv("experiments/domain_phase_mix/exploratory/two_phase_starcoder_combined.csv")
    df = df[df["status"] == "completed"].head(4).reset_index(drop=True)
    runs = [_phase_weights_to_run_record(row) for _, row in df.iterrows()]

    frame = run_records_to_dataframe(runs)
    spec = _build_dataset_spec(frame, DEFAULT_STARCODER_OBJECTIVE)

    assert spec.domain_names == list(DOMAIN_NAMES)
    assert spec.phase_names == ["phase_0", "phase_1"]


def test_prospective_doptimal_selection_returns_valid_weight_configs():
    spec = _synthetic_two_phase_spec()
    experiment = _FakeExperiment(["phase_0", "phase_1"], DOMAIN_NAMES, min_config_distance=0.02)

    configs, selection = prospective_d_optimal_selection(
        spec,
        experiment,
        n_select=3,
        seed=5,
        pool_size=24,
    )

    assert len(configs) == 3
    assert len(selection.selected_indices) == 3
    weights = weight_configs_to_tensor(configs, phase_names=["phase_0", "phase_1"], domain_names=DOMAIN_NAMES)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0)


def test_prospective_generic_selection_returns_valid_weight_configs():
    spec = _synthetic_two_phase_spec()
    experiment = _FakeExperiment(["phase_0", "phase_1"], DOMAIN_NAMES, min_config_distance=0.02)

    for method in ("feature_maximin", "feature_dpp", "feature_bayes_linear"):
        configs, selection = prospective_generic_selection(
            spec,
            experiment,
            method=method,
            n_select=3,
            seed=5,
            pool_size=24,
        )
        assert len(configs) == 3
        assert len(selection.selected_indices) == 3
        assert len(set(selection.selected_indices)) == 3
        weights = weight_configs_to_tensor(configs, phase_names=["phase_0", "phase_1"], domain_names=DOMAIN_NAMES)
        np.testing.assert_allclose(weights.sum(axis=2), 1.0)
