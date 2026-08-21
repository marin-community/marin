# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_gradio_dashboard.dashboard_data import (
    DashboardState,
    clear_constraints,
    constraints_table,
    manual_candidate_choices,
    select_best_candidate,
    slider_values_for_candidate,
    slider_values_for_state,
    starting_candidate_dropdown_value,
    update_constraint_from_lock,
    update_constraint_from_slider,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_gradio_dashboard.dashboard_views import (
    matched_candidate_table,
    task_readiness_summary_from_table,
    task_readiness_table_from_metrics,
    task_slider_metadata_from_metrics,
)


def test_update_constraint_from_slider_selects_best_feasible_candidate_and_moves_sliders():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "best",
                "target_gain": 1.5,
                "target_gain_lcb": 1.2,
                "nearest_observed_tv": 0.20,
                "max_phase_weight": 0.30,
                "source": "sobol_logit_trust",
            },
            {
                "candidate": "fails_task",
                "target_gain": 2.0,
                "target_gain_lcb": 1.8,
                "nearest_observed_tv": 0.20,
                "max_phase_weight": 0.30,
                "source": "sobol_logit_trust",
            },
            {
                "candidate": "second",
                "target_gain": 1.0,
                "target_gain_lcb": 0.9,
                "nearest_observed_tv": 0.10,
                "max_phase_weight": 0.20,
                "source": "observed_300m_signal",
            },
        ]
    )
    task_predictions = pd.DataFrame(
        [
            {"candidate": "best", "task_a": 0.7, "task_b": -0.1},
            {"candidate": "fails_task", "task_a": 0.1, "task_b": 1.0},
            {"candidate": "second", "task_a": 0.9, "task_b": 0.2},
        ]
    )
    state = DashboardState(candidate="fails_task", constraints={}, starting_candidate="raw_dsp_optimum")

    next_state, filtered = update_constraint_from_slider(
        state,
        task_name="task_a",
        threshold=0.5,
        candidate_summary=candidate_summary,
        task_predictions=task_predictions,
        min_target_gain=0.0,
        max_nearest_tv=0.45,
        max_phase_weight=0.5,
        include_sources={"sobol_logit_trust", "observed_300m_signal"},
        keep_missing=False,
    )

    assert next_state.candidate == "best"
    assert next_state.starting_candidate == "raw_dsp_optimum"
    assert next_state.constraints == {"task_a": 0.5}
    assert filtered["candidate"].tolist() == ["best", "second"]
    assert slider_values_for_candidate(task_predictions, next_state.candidate, ["task_a", "task_b"]) == {
        "task_a": pytest.approx(0.7),
        "task_b": pytest.approx(-0.1),
    }


def test_task_constraint_update_always_auto_recommends_best_candidate():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "manual",
                "target_gain": 0.5,
                "target_gain_lcb": 0.3,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "observed_300m_signal",
            },
            {
                "candidate": "best",
                "target_gain": 2.0,
                "target_gain_lcb": 1.8,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
        ]
    )
    task_predictions = pd.DataFrame(
        [
            {"candidate": "manual", "task_a": 1.0},
            {"candidate": "best", "task_a": 2.0},
        ]
    )
    state = DashboardState(candidate="manual", constraints={})

    next_state, _ = update_constraint_from_slider(
        state,
        task_name="task_a",
        threshold=0.5,
        candidate_summary=candidate_summary,
        task_predictions=task_predictions,
        min_target_gain=0.0,
        max_nearest_tv=0.45,
        max_phase_weight=0.5,
        include_sources=None,
        keep_missing=False,
    )

    assert next_state.candidate == "best"
    assert next_state.constraints == {"task_a": 0.5}


def test_lock_checkbox_can_create_zero_threshold_constraint():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "best",
                "target_gain": 1.0,
                "target_gain_lcb": 0.8,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
            {
                "candidate": "regresses",
                "target_gain": 2.0,
                "target_gain_lcb": 1.8,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
        ]
    )
    task_predictions = pd.DataFrame(
        [
            {"candidate": "best", "task_a": 0.2},
            {"candidate": "regresses", "task_a": -0.2},
        ]
    )

    next_state, filtered = update_constraint_from_lock(
        DashboardState(candidate="regresses", constraints={}, starting_candidate="raw_dsp_optimum"),
        task_name="task_a",
        threshold=0.0,
        locked=True,
        candidate_summary=candidate_summary,
        task_predictions=task_predictions,
        min_target_gain=0.0,
        max_nearest_tv=0.45,
        max_phase_weight=0.5,
        include_sources=None,
        keep_missing=False,
    )

    assert next_state.candidate == "best"
    assert next_state.starting_candidate == "raw_dsp_optimum"
    assert next_state.constraints == {"task_a": 0.0}
    assert filtered["candidate"].tolist() == ["best"]


def test_lock_reports_no_feasible_instead_of_falling_back_to_regressing_candidate():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "current",
                "target_gain": 1.0,
                "target_gain_lcb": 0.8,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
            {
                "candidate": "fallback_regresses",
                "target_gain": 0.8,
                "target_gain_lcb": 0.7,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
        ]
    )
    task_predictions = pd.DataFrame(
        [
            {"candidate": "current", "task_a": 0.3},
            {"candidate": "fallback_regresses", "task_a": -0.2},
        ]
    )

    next_state, filtered = update_constraint_from_lock(
        DashboardState(candidate="current", constraints={}, starting_candidate="raw_dsp_optimum"),
        task_name="task_a",
        threshold=0.8,
        locked=True,
        candidate_summary=candidate_summary,
        task_predictions=task_predictions,
        min_target_gain=0.0,
        max_nearest_tv=0.45,
        max_phase_weight=0.5,
        include_sources=None,
        keep_missing=False,
    )

    assert filtered.empty
    assert next_state.candidate == "current"
    assert next_state.constraints == {"task_a": 0.8}
    assert next_state.no_feasible is True
    assert slider_values_for_state(task_predictions, next_state, ["task_a"]) == {"task_a": pytest.approx(0.8)}


def test_unlock_checkbox_removes_task_constraint():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "best",
                "target_gain": 1.0,
                "target_gain_lcb": 0.8,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
        ]
    )

    next_state, _ = update_constraint_from_lock(
        DashboardState(candidate="best", constraints={"task_a": 0.0, "task_b": 1.0}),
        task_name="task_a",
        threshold=0.0,
        locked=False,
        candidate_summary=candidate_summary,
        task_predictions=pd.DataFrame(columns=["candidate", "task_a", "task_b"]),
        min_target_gain=0.0,
        max_nearest_tv=0.45,
        max_phase_weight=0.5,
        include_sources=None,
        keep_missing=False,
    )

    assert next_state.constraints == {"task_b": 1.0}


def test_clear_constraints_resets_to_starting_candidate():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "best",
                "target_gain": 1.0,
                "target_gain_lcb": 0.8,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
            {
                "candidate": "bad",
                "target_gain": -1.0,
                "target_gain_lcb": -1.2,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
        ]
    )
    state = DashboardState(candidate="bad", constraints={"task_a": 0.5}, starting_candidate="best")

    next_state, filtered = clear_constraints(
        state,
        candidate_summary=candidate_summary,
        task_predictions=pd.DataFrame(columns=["candidate", "task_a"]),
        min_target_gain=-2.0,
        max_nearest_tv=0.45,
        max_phase_weight=0.5,
        include_sources=None,
        keep_missing=False,
    )

    assert next_state.candidate == "best"
    assert next_state.starting_candidate == "best"
    assert next_state.constraints == {}
    assert filtered["candidate"].tolist() == ["best", "bad"]


def test_dropdown_preserves_starting_candidate_when_recommendation_is_not_curated():
    state = DashboardState(
        candidate="path_endpoint_kl0p5_mw2p0_t1p0",
        constraints={"teacher_forced/humaneval_10shot_canonical_solution/bpb": 0.0},
        starting_candidate="raw_dsp_optimum",
    )

    value = starting_candidate_dropdown_value(
        state,
        choices=["baseline_proportional", "raw_dsp_optimum"],
        fallback="baseline_proportional",
    )

    assert value == "raw_dsp_optimum"


def test_constraints_table_reports_pass_fail_for_current_candidate():
    task_predictions = pd.DataFrame([{"candidate": "candidate", "task_a": 0.7, "task_b": -0.2}])

    table = constraints_table(
        state=DashboardState(candidate="candidate", constraints={"task_a": 0.5, "task_b": 0.0}),
        task_predictions=task_predictions,
    )

    assert table.to_dict("records") == [
        {"task": "task_a", "target_threshold": 0.5, "candidate_prediction": 0.7, "passes": True},
        {"task": "task_b", "target_threshold": 0.0, "candidate_prediction": -0.2, "passes": False},
    ]


def test_select_best_candidate_prefers_lcb_then_gain():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "gain_only",
                "target_gain": 2.0,
                "target_gain_lcb": 0.5,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
            {
                "candidate": "lcb_best",
                "target_gain": 1.5,
                "target_gain_lcb": 1.0,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
        ]
    )

    selected, filtered = select_best_candidate(
        candidate_summary,
        pd.DataFrame(columns=["candidate"]),
        constraints={},
        min_target_gain=0.0,
        max_nearest_tv=0.45,
        max_phase_weight=0.5,
        include_sources=None,
        keep_missing=False,
    )

    assert selected == "lcb_best"
    assert filtered["candidate"].tolist() == ["lcb_best", "gain_only"]


def test_select_best_candidate_rejects_duplicate_task_prediction_candidates():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "candidate",
                "target_gain": 1.0,
                "target_gain_lcb": 0.8,
                "nearest_observed_tv": 0.1,
                "max_phase_weight": 0.2,
                "source": "sobol_logit_trust",
            },
        ]
    )
    task_predictions = pd.DataFrame(
        [
            {"candidate": "candidate", "task_a": 1.0},
            {"candidate": "candidate", "task_a": 0.0},
        ]
    )

    with pytest.raises(ValueError, match="unique candidate rows"):
        select_best_candidate(
            candidate_summary,
            task_predictions,
            constraints={"task_a": 0.5},
            min_target_gain=0.0,
            max_nearest_tv=0.45,
            max_phase_weight=0.5,
            include_sources=None,
            keep_missing=False,
        )


def test_manual_candidate_choices_are_curated_and_exclude_bulk_cache_rows():
    candidate_summary = pd.DataFrame(
        [
            {"candidate": "baseline_proportional"},
            {"candidate": "raw_dsp_optimum"},
            {"candidate": "path_endpoint_kl2p0_mw2p0_t1p0"},
            {"candidate": "sobol_123456"},
            {"candidate": "run_00042"},
        ]
    )

    assert manual_candidate_choices(candidate_summary) == [
        "baseline_proportional",
        "raw_dsp_optimum",
        "path_endpoint_kl2p0_mw2p0_t1p0",
    ]


def test_matched_candidate_table_labels_auto_and_manual_selection():
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": "candidate_a",
                "source": "sobol_logit_trust",
                "score_kind": "cache_prediction",
                "target_gain": 1.2,
                "target_gain_lcb": 0.9,
                "nearest_observed_tv": 0.3,
                "max_phase_weight": 0.2,
                "has_task_predictions": True,
            }
        ]
    )

    auto = matched_candidate_table(
        candidate_summary,
        DashboardState(candidate="candidate_a", constraints={}),
    )

    assert auto.loc[0, "selection"] == "matched precomputed candidate"
    assert auto.loc[0, "candidate"] == "candidate_a"
    assert auto.loc[0, "source"] == "sobol_logit_trust"


def test_dashboard_state_has_no_manual_mode_flag():
    state = DashboardState(candidate="candidate_a", constraints={})

    assert not hasattr(state, "auto_recommend")


def test_task_readiness_table_surfaces_fit_quality_categories():
    metrics = pd.DataFrame(
        [
            {"task_column": "task_ready", "train_pearson": 0.8, "train_rmse": 0.2, "train_n": 20},
            {"task_column": "task_usable", "train_pearson": 0.6, "train_rmse": 0.3, "train_n": 20},
            {"task_column": "task_caution", "train_pearson": 0.4, "train_rmse": 0.4, "train_n": 20},
            {"task_column": "task_weak", "train_pearson": 0.1, "train_rmse": 0.5, "train_n": 20},
            {"task_column": "task_unknown", "train_pearson": None, "train_rmse": None, "train_n": 0},
        ]
    )

    table = task_readiness_table_from_metrics(metrics)
    by_task = table.set_index("task")

    assert by_task.loc["task_ready", "readiness"] == "ready"
    assert by_task.loc["task_usable", "readiness"] == "usable"
    assert by_task.loc["task_caution", "readiness"] == "caution"
    assert by_task.loc["task_weak", "readiness"] == "weak"
    assert by_task.loc["task_unknown", "readiness"] == "unknown"
    assert table["readiness_rank"].is_monotonic_increasing


def test_task_readiness_summary_counts_visible_categories():
    table = task_readiness_table_from_metrics(
        pd.DataFrame(
            [
                {"task_column": "a", "train_pearson": 0.8, "train_rmse": 0.2, "train_n": 20},
                {"task_column": "b", "train_pearson": 0.2, "train_rmse": 0.5, "train_n": 20},
                {"task_column": "c", "train_pearson": None, "train_rmse": None, "train_n": 0},
            ]
        )
    )

    summary = task_readiness_summary_from_table(table)
    counts = dict(zip(summary["readiness"], summary["task_count"], strict=True))

    assert counts == {"ready": 1, "weak": 1, "unknown": 1}


def test_task_slider_metadata_embeds_readiness_on_each_slider():
    metadata = task_slider_metadata_from_metrics(
        pd.DataFrame(
            [
                {
                    "task_column": "eval/example_ready/bpb",
                    "train_pearson": 0.8,
                    "train_rmse": 0.2,
                    "train_n": 242,
                },
                {
                    "task_column": "eval/example_weak/bpb",
                    "train_pearson": 0.1,
                    "train_rmse": 0.7,
                    "train_n": 242,
                },
            ]
        ),
        ["eval/example_ready/bpb", "eval/example_weak/bpb", "eval/missing/bpb"],
    )

    assert metadata["eval/example_ready/bpb"]["label"].startswith("[ready r=0.80]")
    assert "RMSE=0.20" in metadata["eval/example_ready/bpb"]["info"]
    assert metadata["eval/example_weak/bpb"]["label"].startswith("[weak r=0.10]")
    assert "do not trust as steering objective" in metadata["eval/example_weak/bpb"]["info"]
    assert metadata["eval/missing/bpb"]["label"].startswith("[unknown r=n/a]")
