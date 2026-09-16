# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_constraint_dashboard_helpers import (
    active_task_thresholds_from_controls,
    add_materialized_epochs,
    candidate_names_satisfying_task_thresholds,
    candidate_names_satisfying_task_thresholds_wide,
    candidate_selector_options,
    centered_task_slider_steps,
    combine_phase_weights,
    epoch_scale_table_from_mixture_weights,
    filter_candidate_summary,
    label_phase_weights_long,
    load_selected_candidate_weights,
    nearest_observed_tv,
    normalize_candidate_cache_summary,
    oriented_task_delta_table,
    phase_weight_long_from_signal,
    ridge_task_delta_prediction_wide,
    sample_frontier_for_plot,
    selected_candidate_for_dashboard,
    selected_task_prediction_long,
)


def test_phase_weight_long_from_signal_extracts_two_phase_weights():
    signal = pd.DataFrame(
        [
            {
                "run_name": "baseline_proportional",
                "phase_0_alpha": 0.25,
                "phase_0_beta": 0.75,
                "phase_1_alpha": 0.40,
                "phase_1_beta": 0.60,
                "metric": 1.0,
            }
        ]
    )

    weights = phase_weight_long_from_signal(signal, "baseline_proportional")

    assert weights.to_dict("records") == [
        {"candidate": "baseline_proportional", "domain": "alpha", "phase": "phase_0", "weight": 0.25},
        {"candidate": "baseline_proportional", "domain": "beta", "phase": "phase_0", "weight": 0.75},
        {"candidate": "baseline_proportional", "domain": "alpha", "phase": "phase_1", "weight": 0.40},
        {"candidate": "baseline_proportional", "domain": "beta", "phase": "phase_1", "weight": 0.60},
    ]


def test_label_phase_weights_long_normalizes_canonical_dsp_labels():
    mixture_weights = pd.DataFrame(
        [
            {"label": "proportional", "domain": "alpha", "phase_0_weight": 0.25, "phase_1_weight": 0.40},
            {"label": "proportional", "domain": "beta", "phase_0_weight": 0.75, "phase_1_weight": 0.60},
            {"label": "raw_dsp_optimum", "domain": "alpha", "phase_0_weight": 0.10, "phase_1_weight": 0.20},
            {"label": "raw_dsp_optimum", "domain": "beta", "phase_0_weight": 0.90, "phase_1_weight": 0.80},
        ]
    )

    weights = label_phase_weights_long(mixture_weights)

    assert set(weights["candidate"]) == {"proportional", "raw_dsp_optimum"}
    phase_sums = weights.groupby(["candidate", "phase"])["weight"].sum()
    assert np.isclose(phase_sums.to_numpy(), 1.0).all()


def test_combine_phase_weights_rejects_bad_phase_sums():
    good = pd.DataFrame(
        [
            {"candidate": "a", "domain": "alpha", "phase": "phase_0", "weight": 1.0},
            {"candidate": "a", "domain": "alpha", "phase": "phase_1", "weight": 1.0},
        ]
    )
    bad = pd.DataFrame(
        [
            {"candidate": "b", "domain": "alpha", "phase": "phase_0", "weight": 0.4},
            {"candidate": "b", "domain": "alpha", "phase": "phase_1", "weight": 1.0},
        ]
    )

    assert combine_phase_weights(good)["candidate"].tolist() == ["a", "a"]
    with pytest.raises(ValueError, match="phase weights do not sum"):
        combine_phase_weights(good, bad)


def test_nearest_observed_tv_uses_average_phase_tv():
    named = pd.DataFrame(
        [
            {"candidate": "candidate", "domain": "alpha", "phase": "phase_0", "weight": 0.8},
            {"candidate": "candidate", "domain": "beta", "phase": "phase_0", "weight": 0.2},
            {"candidate": "candidate", "domain": "alpha", "phase": "phase_1", "weight": 0.5},
            {"candidate": "candidate", "domain": "beta", "phase": "phase_1", "weight": 0.5},
        ]
    )
    observed = pd.DataFrame(
        [
            {"candidate": "near", "domain": "alpha", "phase": "phase_0", "weight": 0.7},
            {"candidate": "near", "domain": "beta", "phase": "phase_0", "weight": 0.3},
            {"candidate": "near", "domain": "alpha", "phase": "phase_1", "weight": 0.5},
            {"candidate": "near", "domain": "beta", "phase": "phase_1", "weight": 0.5},
            {"candidate": "far", "domain": "alpha", "phase": "phase_0", "weight": 0.0},
            {"candidate": "far", "domain": "beta", "phase": "phase_0", "weight": 1.0},
            {"candidate": "far", "domain": "alpha", "phase": "phase_1", "weight": 0.0},
            {"candidate": "far", "domain": "beta", "phase": "phase_1", "weight": 1.0},
        ]
    )

    nearest = nearest_observed_tv(named, observed)

    assert nearest.to_dict("records") == [
        {"candidate": "candidate", "nearest_observed_run": "near", "nearest_observed_tv": pytest.approx(0.05)}
    ]


def test_filter_candidate_summary_applies_clean_slate_constraints():
    summary = pd.DataFrame(
        [
            {"candidate": "safe", "target_gain": 0.8, "nearest_observed_tv": 0.2, "max_phase_weight": 0.3},
            {"candidate": "too_far", "target_gain": 1.1, "nearest_observed_tv": 0.8, "max_phase_weight": 0.3},
            {"candidate": "too_sparse", "target_gain": 1.2, "nearest_observed_tv": 0.1, "max_phase_weight": 0.9},
        ]
    )

    filtered = filter_candidate_summary(
        summary,
        min_target_gain=0.5,
        max_nearest_tv=0.4,
        max_phase_weight=0.5,
    )

    assert filtered["candidate"].tolist() == ["safe"]


def test_oriented_task_delta_table_uses_noise_then_signal_fallback_scales():
    signal = pd.DataFrame(
        [
            {"run_name": "baseline_proportional", "task_higher": 1.0, "task_lower": 5.0},
            {"run_name": "candidate_a", "task_higher": 1.5, "task_lower": 4.0},
            {"run_name": "candidate_b", "task_higher": 0.5, "task_lower": 7.0},
        ]
    )
    selected = pd.DataFrame(
        [
            {"task": "task_higher", "sign": 1.0},
            {"task": "task_lower", "sign": -1.0},
        ]
    )
    noise = pd.DataFrame(
        [
            {"task_higher": 0.0},
            {"task_higher": 0.5},
            {"task_higher": 1.0},
        ]
    )

    deltas = oriented_task_delta_table(signal, selected, noise_frame=noise)

    candidate = deltas.loc[deltas["candidate"].eq("candidate_a")].sort_values("task_column")
    assert candidate["oriented_delta"].tolist() == pytest.approx([0.5, 1.0])
    assert candidate["scale_source"].tolist() == ["noise_std", "signal_std_fallback"]
    assert np.isfinite(candidate["task_delta_standardized"]).all()


def test_candidate_names_satisfying_task_thresholds_requires_all_active_tasks():
    task_deltas = pd.DataFrame(
        [
            {"candidate": "good", "task_column": "task_a", "task_delta_standardized": 1.0},
            {"candidate": "good", "task_column": "task_b", "task_delta_standardized": 0.5},
            {"candidate": "bad_a", "task_column": "task_a", "task_delta_standardized": -0.1},
            {"candidate": "bad_a", "task_column": "task_b", "task_delta_standardized": 2.0},
            {"candidate": "missing_b", "task_column": "task_a", "task_delta_standardized": 2.0},
        ]
    )

    satisfying = candidate_names_satisfying_task_thresholds(
        task_deltas,
        {"task_a": 0.0, "task_b": 0.0},
        keep_missing=False,
        all_candidates={"good", "bad_a", "missing_b", "unknown"},
    )

    assert satisfying == {"good"}

    with_missing = candidate_names_satisfying_task_thresholds(
        task_deltas,
        {"task_a": 0.0, "task_b": 0.0},
        keep_missing=True,
        all_candidates={"good", "bad_a", "missing_b", "unknown"},
    )

    assert with_missing == {"good", "unknown"}


def test_candidate_names_satisfying_task_thresholds_wide_filters_predictions():
    predictions = pd.DataFrame(
        [
            {"candidate": "good", "task_a": 0.2, "task_b": 1.0},
            {"candidate": "bad", "task_a": -0.1, "task_b": 2.0},
            {"candidate": "missing", "task_a": 0.5, "task_b": np.nan},
        ]
    )

    satisfying = candidate_names_satisfying_task_thresholds_wide(
        predictions,
        {"task_a": 0.0, "task_b": 0.5},
        keep_missing=False,
        all_candidates={"good", "bad", "missing", "unknown"},
    )

    assert satisfying == {"good"}

    with_missing = candidate_names_satisfying_task_thresholds_wide(
        predictions,
        {"task_a": 0.0, "task_b": 0.5},
        keep_missing=True,
        all_candidates={"good", "bad", "missing", "unknown"},
    )

    assert with_missing == {"good", "missing", "unknown"}


def test_normalize_candidate_cache_summary_maps_cache_columns_to_dashboard_columns():
    cache = pd.DataFrame(
        [
            {
                "candidate_id": "sobol_000001",
                "candidate_source": "sobol_logit_trust",
                "predicted_y_factor": 0.7,
                "predicted_y_factor_gain_vs_proportional": 0.2,
                "predicted_y_factor_gain_lcb": 0.05,
                "nearest_observed_tv": 0.12,
                "max_phase_weight": 0.3,
                "min_phase_support_gt_1e3": 37,
                "mean_phase_effective_support": 20.0,
                "passes_basic_dashboard_gate": True,
            },
            {
                "candidate_id": "observed_skip",
                "candidate_source": "observed_signal",
                "predicted_y_factor": 0.6,
                "predicted_y_factor_gain_vs_proportional": 0.1,
                "predicted_y_factor_gain_lcb": -0.1,
                "nearest_observed_tv": 0.0,
                "max_phase_weight": 0.2,
                "passes_basic_dashboard_gate": True,
            },
        ]
    )

    normalized = normalize_candidate_cache_summary(cache, include_sources={"sobol_logit_trust"})

    assert normalized["candidate"].tolist() == ["sobol_000001"]
    assert normalized["target_score"].tolist() == pytest.approx([0.7])
    assert normalized["target_gain"].tolist() == pytest.approx([0.2])
    assert normalized["target_gain_lcb"].tolist() == pytest.approx([0.05])
    assert normalized["score_kind"].tolist() == ["dsp_prediction"]
    assert normalized["has_task_deltas"].tolist() == [False]
    assert normalized["deployability_flag"].tolist() == [True]


def test_load_selected_candidate_weights_uses_eager_weights_then_parquet(tmp_path):
    eager = pd.DataFrame(
        [
            {"candidate": "baseline_proportional", "domain": "alpha", "phase": "phase_0", "weight": 0.25},
            {"candidate": "baseline_proportional", "domain": "beta", "phase": "phase_0", "weight": 0.75},
            {"candidate": "baseline_proportional", "domain": "alpha", "phase": "phase_1", "weight": 0.25},
            {"candidate": "baseline_proportional", "domain": "beta", "phase": "phase_1", "weight": 0.75},
        ]
    )
    wide = pd.DataFrame(
        [
            {
                "candidate_id": "cached_candidate",
                "candidate_source": "sobol_logit_trust",
                "phase_0_alpha": 0.8,
                "phase_0_beta": 0.2,
                "phase_1_alpha": 0.6,
                "phase_1_beta": 0.4,
            }
        ]
    )
    parquet_path = tmp_path / "weights.parquet"
    wide.to_parquet(parquet_path, index=False)

    eager_weights = load_selected_candidate_weights(
        "baseline_proportional",
        eager_weights=eager,
        parquet_weight_paths=[parquet_path],
    )
    cached_weights = load_selected_candidate_weights(
        "cached_candidate",
        eager_weights=eager,
        parquet_weight_paths=[parquet_path],
    )

    assert eager_weights["candidate"].unique().tolist() == ["baseline_proportional"]
    assert cached_weights.sort_values(["phase", "domain"])["weight"].tolist() == pytest.approx([0.8, 0.2, 0.6, 0.4])
    with pytest.raises(ValueError, match="candidate weights not found"):
        load_selected_candidate_weights("missing", eager_weights=eager, parquet_weight_paths=[parquet_path])


def test_sample_frontier_for_plot_keeps_top_gain_rows_and_caps_size():
    frame = pd.DataFrame(
        {
            "candidate": [f"candidate_{idx}" for idx in range(20)],
            "target_gain": np.arange(20, dtype=float),
            "nearest_observed_tv": np.linspace(0.0, 1.0, 20),
        }
    )

    sampled = sample_frontier_for_plot(frame, max_rows=8, top_rows=3, seed=1)

    assert len(sampled) == 8
    assert {"candidate_17", "candidate_18", "candidate_19"}.issubset(set(sampled["candidate"]))


def test_candidate_selector_options_limits_to_top_ranked_candidates():
    filtered = pd.DataFrame(
        {
            "candidate": ["a", "b", "c", "d"],
            "target_gain": [0.1, 0.5, 0.2, 0.4],
            "nearest_observed_tv": [0.2, 0.1, 0.3, 0.4],
        }
    )

    options = candidate_selector_options(filtered, preferred="d", limit=3)

    assert options == ["d", "b", "c"]


def test_centered_task_slider_steps_are_symmetric_and_include_zero():
    steps = centered_task_slider_steps(min_value=-10.0, max_value=10.0, step=0.5)

    assert steps[0] == -10.0
    assert steps[-1] == 10.0
    assert 0.0 in steps
    assert steps == [-value for value in reversed(steps)]


def test_active_task_thresholds_use_moved_sliders_or_explicit_locks_not_sentinels():
    active = active_task_thresholds_from_controls(
        slider_values={"task_a": 0.0, "task_b": 1.5, "task_c": -0.5, "task_d": 0.0},
        lock_values={"task_a": False, "task_b": False, "task_c": True, "task_d": True},
    )

    assert active == {"task_b": 1.5, "task_c": -0.5, "task_d": 0.0}


def test_selected_candidate_for_dashboard_follows_recommendation_by_default():
    assert (
        selected_candidate_for_dashboard(
            recommended_candidate="best_feasible",
            manual_candidate="manual_choice",
            follow_recommendation=True,
        )
        == "best_feasible"
    )
    assert (
        selected_candidate_for_dashboard(
            recommended_candidate="best_feasible",
            manual_candidate="manual_choice",
            follow_recommendation=False,
        )
        == "manual_choice"
    )


def test_ridge_task_delta_prediction_wide_uses_centered_phase_weights():
    signal = pd.DataFrame(
        [
            {
                "run_name": "baseline_proportional",
                "phase_0_a": 0.5,
                "phase_0_b": 0.5,
                "phase_1_a": 0.5,
                "phase_1_b": 0.5,
                "task": 0.0,
            },
            {
                "run_name": "candidate_up",
                "phase_0_a": 0.7,
                "phase_0_b": 0.3,
                "phase_1_a": 0.7,
                "phase_1_b": 0.3,
                "task": 0.2,
            },
            {
                "run_name": "candidate_down",
                "phase_0_a": 0.3,
                "phase_0_b": 0.7,
                "phase_1_a": 0.3,
                "phase_1_b": 0.7,
                "task": -0.2,
            },
        ]
    )
    selected = pd.DataFrame([{"task": "task", "sign": 1.0}])
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "baseline_proportional",
                "candidate_source": "observed",
                "phase_0_a": 0.5,
                "phase_0_b": 0.5,
                "phase_1_a": 0.5,
                "phase_1_b": 0.5,
            },
            {
                "candidate_id": "candidate_up",
                "candidate_source": "observed",
                "phase_0_a": 0.7,
                "phase_0_b": 0.3,
                "phase_1_a": 0.7,
                "phase_1_b": 0.3,
            },
            {
                "candidate_id": "new_candidate",
                "candidate_source": "model",
                "phase_0_a": 0.6,
                "phase_0_b": 0.4,
                "phase_1_a": 0.6,
                "phase_1_b": 0.4,
            },
        ]
    )

    predictions, metrics = ridge_task_delta_prediction_wide(
        signal,
        selected,
        candidates,
        noise_frame=pd.DataFrame({"task": [-1.0, 0.0, 1.0]}),
        alpha=0.0,
    )

    by_candidate = predictions.set_index("candidate")
    assert by_candidate.loc["baseline_proportional", "task"] == pytest.approx(0.0)
    assert by_candidate.loc["candidate_up", "task"] == pytest.approx(0.2)
    assert by_candidate.loc["new_candidate", "task"] == pytest.approx(0.1)
    assert metrics.loc[0, "task_column"] == "task"


def test_selected_task_prediction_long_marks_locked_targets():
    predictions = pd.DataFrame(
        [
            {"candidate": "candidate", "task_a": 0.2, "task_b": -0.1},
        ]
    )

    selected = selected_task_prediction_long(
        predictions,
        "candidate",
        thresholds={"task_a": 0.0},
    )

    rows = selected.sort_values("task_column").to_dict("records")
    assert rows[0] == {
        "task_column": "task_a",
        "predicted_task_delta_standardized": 0.2,
        "locked": True,
        "target_threshold": 0.0,
        "meets_target": True,
    }
    assert rows[1]["task_column"] == "task_b"
    assert rows[1]["predicted_task_delta_standardized"] == pytest.approx(-0.1)
    assert rows[1]["locked"] is False
    assert np.isnan(rows[1]["target_threshold"])
    assert rows[1]["meets_target"] is True


def test_materialized_epoch_diagnostics_use_phase_epoch_scales():
    mixture_weights = pd.DataFrame(
        [
            {
                "label": "proportional",
                "domain": "a",
                "phase_0_weight": 0.5,
                "phase_1_weight": 0.5,
                "phase_0_epochs": 2.0,
                "phase_1_epochs": 1.0,
            },
            {
                "label": "proportional",
                "domain": "b",
                "phase_0_weight": 0.5,
                "phase_1_weight": 0.5,
                "phase_0_epochs": 4.0,
                "phase_1_epochs": 3.0,
            },
        ]
    )
    weights = pd.DataFrame(
        [
            {"candidate": "candidate", "domain": "a", "phase": "phase_0", "weight": 0.25},
            {"candidate": "candidate", "domain": "b", "phase": "phase_0", "weight": 0.75},
            {"candidate": "candidate", "domain": "a", "phase": "phase_1", "weight": 0.60},
            {"candidate": "candidate", "domain": "b", "phase": "phase_1", "weight": 0.40},
        ]
    )

    scales = epoch_scale_table_from_mixture_weights(mixture_weights)
    with_epochs = add_materialized_epochs(weights, scales)

    assert with_epochs.sort_values(["phase", "domain"])["materialized_epochs"].tolist() == pytest.approx(
        [1.0, 6.0, 1.2, 2.4]
    )
