# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from experiments.domain_phase_mix.exploratory.plot_starcoder_optima_validation import (
    build_validation_plot_frame,
    _format_numeric_tuple,
    _load_rl_rollout_summary,
    _select_actual_metric_candidates,
    _summarize_observed_bpb_frame,
)


def test_build_validation_plot_frame_merges_predicted_regret_and_actual_metrics():
    launch_plan = {
        "name_prefix": "pcx/dm/t3s-e32b1ae8",
        "runs": [
            {
                "subset_size": 16,
                "run_name": "feature_bayes_linear_k016_optimum",
                "weight_config": {
                    "phase_weights": {
                        "phase_0": {"nemotron_full": 0.889593, "starcoder": 0.110407},
                        "phase_1": {"nemotron_full": 0.810375, "starcoder": 0.189625},
                        "phase_2": {"nemotron_full": 0.712191, "starcoder": 0.287809},
                    }
                },
            },
            {
                "subset_size": 4,
                "run_name": "feature_bayes_linear_k004_optimum",
                "weight_config": {
                    "phase_weights": {
                        "phase_0": {"nemotron_full": 0.755598, "starcoder": 0.244402},
                        "phase_1": {"nemotron_full": 0.543296, "starcoder": 0.456704},
                        "phase_2": {"nemotron_full": 0.476622, "starcoder": 0.523378},
                    }
                },
            },
        ],
    }
    predicted_optima = pd.DataFrame(
        [
            {"subset_size": 4, "predicted_bpb": 0.91},
            {"subset_size": 16, "predicted_bpb": 0.87},
        ]
    )
    regret_curve = pd.DataFrame(
        [
            {"subset_size": 4, "regret_at_1": 0.046},
            {"subset_size": 16, "regret_at_1": 0.0021},
        ]
    )
    actual_metric_map = {
        "pcx/dm/t3s-e32b1ae8/feature_bayes_linear_k004_optimum": {
            "actual_bpb": 0.94,
            "wandb_state": "finished",
            "wandb_url": "https://wandb.ai/example/k004",
        },
        "pcx/dm/t3s-e32b1ae8/feature_bayes_linear_k016_optimum": {
            "actual_bpb": 0.87,
            "wandb_state": "finished",
            "wandb_url": "https://wandb.ai/example/k016",
        },
    }

    frame = build_validation_plot_frame(
        launch_plan=launch_plan,
        predicted_optima=predicted_optima,
        regret_curve=regret_curve,
        actual_metric_map=actual_metric_map,
    )

    assert frame["subset_size"].tolist() == [4, 16]
    assert frame["predicted_bpb"].tolist() == [0.91, 0.87]
    assert frame["actual_bpb"].tolist() == [0.94, 0.87]
    assert frame["regret_at_1"].tolist() == [0.046, 0.0021]
    assert frame["predicted_optimum_tuple"].tolist() == [
        "(0.2444, 0.4567, 0.5234)",
        "(0.1104, 0.1896, 0.2878)",
    ]
    assert frame["wandb_run_name"].tolist() == [
        "pcx/dm/t3s-e32b1ae8/feature_bayes_linear_k004_optimum",
        "pcx/dm/t3s-e32b1ae8/feature_bayes_linear_k016_optimum",
    ]


def test_load_rl_rollout_summary_parses_report(tmp_path):
    report = tmp_path / "rollout_report.md"
    report.write_text(
        "\n".join(
            [
                "The resulting executed StarCoder schedule for `r00` was:",
                "",
                "- `phase_0 = 0.4550000131`",
                "- `phase_1 = 0.3199999928`",
                "- `phase_2 = 0.2750000060`",
                "",
                "The final programming BPB was:",
                "",
                "- `0.8346716762`",
                "",
                "The earlier rollout documented elsewhere used the schedule:",
                "",
                "- `phase_0 = 0.815`",
                "- `phase_1 = 0.320`",
                "- `phase_2 = 0.365`",
            ]
        )
    )

    summary = _load_rl_rollout_summary(report)

    assert summary["label"] == "RL rollout BPB (outcome_planner)"
    assert summary["validated_bpb"] == 0.8346716762
    assert summary["rollout_tuple"] == (0.4550000131, 0.3199999928, 0.275000006)


def test_load_rl_rollout_summary_parses_two_phase_report(tmp_path):
    report = tmp_path / "two_phase_rollout_report.md"
    report.write_text(
        "\n".join(
            [
                "# Two-Phase Outcome Planner Rollout Report",
                "",
                "The two-phase rollout schedule was:",
                "",
                "- `phase_0 = 0.05`",
                "- `phase_1 = 0.23`",
                "",
                "Per-phase metrics:",
                "",
                "- final programming BPB: `0.9087211489677429`",
            ]
        )
    )

    summary = _load_rl_rollout_summary(report)

    assert summary["label"] == "RL rollout BPB (outcome_planner)"
    assert summary["validated_bpb"] == 0.9087211489677429
    assert summary["rollout_tuple"] == (0.05, 0.23)


def test_summarize_observed_bpb_frame_returns_best_tuple():
    frame = pd.DataFrame(
        [
            {
                "phase_0_starcoder": 0.10,
                "phase_1_starcoder": 0.30,
                "phase_0_starcoder_epochs": 0.2,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.92,
            },
            {
                "phase_0_starcoder": 0.22,
                "phase_1_starcoder": 0.25,
                "phase_0_starcoder_epochs": 0.4,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.90,
            },
            {
                "phase_0_starcoder": 0.05,
                "phase_1_starcoder": 0.20,
                "phase_0_starcoder_epochs": 0.1,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.94,
            },
        ]
    )

    summary = _summarize_observed_bpb_frame(frame)

    assert summary.best_bpb == 0.90
    assert summary.observed_count == 3
    assert summary.best_tuple == (0.22, 0.25)


def test_format_numeric_tuple_can_trim_trailing_zeros():
    assert _format_numeric_tuple((0.3199999928, 0.3199999928, 0.275000006), trim_trailing_zeros=True) == (
        "(0.32, 0.32, 0.275)"
    )


def test_select_actual_metric_candidates_prefers_finished_suffix_match():
    run_name = (
        "pinlin_calvin_xu/data_mixture/two_phase_starcoder_selector_validation/"
        "full_20260308_rerun2/feature_bayes_linear_k004_optimum"
    )

    selected = _select_actual_metric_candidates(
        run_names=[run_name],
        exact_candidates={run_name: []},
        suffix_candidates={
            run_name: [
                {
                    "actual_bpb": 1.3369,
                    "wandb_state": "crashed",
                    "wandb_url": "https://wandb.ai/example/crashed",
                    "wandb_run_name": (
                        "pinlin_calvin_xu/data_mixture/two_phase_starcoder_selec_5f984bcb/"
                        "feature_bayes_linear_k004_optimum"
                    ),
                    "created_at": "2026-03-08T22:51:00",
                },
                {
                    "actual_bpb": 0.9716,
                    "wandb_state": "finished",
                    "wandb_url": "https://wandb.ai/example/finished",
                    "wandb_run_name": "pcx/dm/t2s-5f984bcb/feature_bayes_linear_k004_optimum",
                    "created_at": "2026-03-08T23:00:00",
                },
            ]
        },
    )

    assert selected[run_name]["wandb_run_name"] == "pcx/dm/t2s-5f984bcb/feature_bayes_linear_k004_optimum"
    assert selected[run_name]["actual_bpb"] == 0.9716
