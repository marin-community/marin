# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import build_grug_moe_mix_dashboard as dash
from experiments.domain_phase_mix.exploratory.two_phase_many.build_grug_moe_mix_dashboard import (
    PATH_MIXTURE_COLORS,
    normalize_scale_columns,
    path_task_delta_plot,
    path_task_deltas,
)


def test_normalize_scale_columns_makes_cached_and_fresh_scale_keys_mergeable():
    cached = pd.DataFrame({"hidden_dim": ["512"], "budget": [2.19e17], "scale_label": ["d512\n2.19e+17"]})
    fresh = pd.DataFrame({"hidden_dim": [512], "budget": ["2.19e+17"], "scale_label": ["d512\n2.19e+17"]})

    cached_norm = normalize_scale_columns(cached)
    fresh_norm = normalize_scale_columns(fresh)

    assert cached_norm["hidden_dim"].dtype == fresh_norm["hidden_dim"].dtype
    assert cached_norm["budget"].dtype == fresh_norm["budget"].dtype
    assert cached_norm.loc[0, "budget"] == fresh_norm.loc[0, "budget"]
    assert cached_norm.loc[0, "scale_label"] == fresh_norm.loc[0, "scale_label"]


def test_path_eval_results_paths_includes_nested_retry_outputs(monkeypatch):
    direct = (
        "gs://marin-us-east5/evaluation/grug_logprob/"
        "grug_moe_mix_v4_path_r1_t025_d512-2.19e+17/"
        "arc_easy_5shot-111111/results.json"
    )
    nested_retry = (
        "gs://marin-us-east5/evaluation/grug_logprob/"
        "grug_moe_mix_v4_path_r1_t025_d512-2.19e+17/"
        "logprob_gsm8k_5shot/numeric_retry2-222222/results.json"
    )

    def fake_gcloud_ls(pattern: str, *, allow_failure: bool = False):
        assert allow_failure
        if pattern.endswith("*/*/*/results.json"):
            return [dash.GcsObject(nested_retry), dash.GcsObject(nested_retry)]
        if pattern.endswith("*/*/results.json"):
            return [dash.GcsObject(direct)]
        raise AssertionError(f"unexpected pattern: {pattern}")

    monkeypatch.setattr(dash, "gcloud_ls", fake_gcloud_ls)

    assert dash.path_eval_results_paths() == [direct, nested_retry]


def test_collect_path_eval_metrics_prefers_latest_retry(monkeypatch):
    direct = (
        "gs://marin-us-east5/evaluation/grug_logprob/"
        "grug_moe_mix_v4_path_r1_t025_d512-2.19e+17/"
        "logprob_gsm8k_5shot-111111/results.json"
    )
    retry1 = (
        "gs://marin-us-east5/evaluation/grug_logprob/"
        "grug_moe_mix_v4_path_r1_t025_d512-2.19e+17/"
        "logprob_gsm8k_5shot/numeric_retry1-222222/results.json"
    )
    retry2 = (
        "gs://marin-us-east5/evaluation/grug_logprob/"
        "grug_moe_mix_v4_path_r1_t025_d512-2.19e+17/"
        "logprob_gsm8k_5shot/numeric_retry2-333333/results.json"
    )

    def fake_collect_one_path_eval(path: str):
        value = {direct: 1.0, retry1: 2.0, retry2: 3.0}[path]
        return [
            {
                "track": "grug_moe_mix_v4_path_r1_t025",
                "track_label": "v4 path t=0.25",
                "endpoint_track": "grug_moe_mix_v4",
                "endpoint_label": "v4",
                "path_t_slug": "t025",
                "path_t": 0.25,
                "hidden_dim": 512,
                "budget": "2.19e+17",
                "scale_label": "d512\n2.19e+17",
                "task_alias": "logprob_gsm8k_5shot",
                "task_group": "gsm8k_logprob",
                "result_key": "gsm8k",
                "metric": "bpb",
                "value": value,
                "result_path": path,
                "checkpoint_path": "gs://example/checkpoints",
            }
        ]

    monkeypatch.setattr(dash, "path_eval_results_paths", lambda: [direct, retry1, retry2])
    monkeypatch.setattr(dash, "collect_one_path_eval", fake_collect_one_path_eval)

    rows = dash.collect_path_eval_metrics()

    assert len(rows) == 1
    assert rows.loc[0, "value"] == pytest.approx(3.0)
    assert rows.loc[0, "result_path"] == retry2
    assert rows.loc[0, "result_attempt_rank"] == 2


def test_path_task_delta_plot_uses_oriented_delta_not_absolute_loss():
    preferred = pd.DataFrame(
        [
            {
                "track": "grug_moe_mix",
                "track_label": "Proportional",
                "hidden_dim": 512,
                "budget": 2.19e17,
                "scale_label": "d512\n2.19e+17",
                "task_alias": "arc_easy_5shot",
                "task_group": "arc_openbook_sciq",
                "preferred_metric": "choice_prob_norm",
                "raw_metric": "choice_prob_norm",
                "raw_value": 0.30,
                "oriented_value": 0.30,
                "common_task": True,
                "available_cells": 1,
                "expected_cells": 1,
                "coverage": "1/1",
            },
            {
                "track": "grug_moe_mix_v4",
                "track_label": "v4",
                "hidden_dim": 512,
                "budget": 2.19e17,
                "scale_label": "d512\n2.19e+17",
                "task_alias": "arc_easy_5shot",
                "task_group": "arc_openbook_sciq",
                "preferred_metric": "choice_prob_norm",
                "raw_metric": "choice_prob_norm",
                "raw_value": 0.40,
                "oriented_value": 0.40,
                "common_task": True,
                "available_cells": 1,
                "expected_cells": 1,
                "coverage": "1/1",
            },
        ]
    )
    path_metrics = pd.DataFrame(
        [
            {
                "track": "grug_moe_mix_v4_path_r1_t050",
                "track_label": "v4 path t=0.50",
                "endpoint_track": "grug_moe_mix_v4",
                "endpoint_label": "v4",
                "path_t_slug": "t050",
                "path_t": 0.50,
                "hidden_dim": 512,
                "budget": 2.19e17,
                "scale_label": "d512\n2.19e+17",
                "task_alias": "arc_easy_5shot",
                "task_group": "arc_openbook_sciq",
                "result_key": "arc_easy",
                "metric": "choice_prob_norm",
                "value": 0.35,
                "result_path": "gs://example/results.json",
                "checkpoint_path": "gs://example/checkpoints",
            }
        ]
    )

    deltas = path_task_deltas(preferred, path_metrics)
    fig = path_task_delta_plot(deltas)

    traces = {trace.name: list(trace.y) for trace in fig.data}
    assert traces["Proportional"] == [0.0]
    assert traces["v4 path t=0.50"][0] == pytest.approx(0.05)
    assert traces["v4"][0] == pytest.approx(0.10)
    assert "delta" in fig.layout.title.text.lower()


def test_path_task_delta_plot_uses_interpolation_color_ladder():
    preferred = pd.DataFrame(
        [
            {
                "track": "grug_moe_mix",
                "track_label": "Proportional",
                "hidden_dim": 512,
                "budget": 2.19e17,
                "scale_label": "d512\n2.19e+17",
                "task_alias": "arc_easy_5shot",
                "task_group": "arc_openbook_sciq",
                "preferred_metric": "choice_prob_norm",
                "raw_metric": "choice_prob_norm",
                "raw_value": 0.30,
                "oriented_value": 0.30,
                "common_task": True,
                "available_cells": 1,
                "expected_cells": 1,
                "coverage": "1/1",
            },
            {
                "track": "grug_moe_mix_v4",
                "track_label": "v4",
                "hidden_dim": 512,
                "budget": 2.19e17,
                "scale_label": "d512\n2.19e+17",
                "task_alias": "arc_easy_5shot",
                "task_group": "arc_openbook_sciq",
                "preferred_metric": "choice_prob_norm",
                "raw_metric": "choice_prob_norm",
                "raw_value": 0.40,
                "oriented_value": 0.40,
                "common_task": True,
                "available_cells": 1,
                "expected_cells": 1,
                "coverage": "1/1",
            },
        ]
    )
    path_metrics = pd.DataFrame(
        [
            {
                "track": "grug_moe_mix_v4_path_r1_t050",
                "track_label": "v4 path t=0.50",
                "endpoint_track": "grug_moe_mix_v4",
                "endpoint_label": "v4",
                "path_t_slug": "t050",
                "path_t": 0.50,
                "hidden_dim": 512,
                "budget": 2.19e17,
                "scale_label": "d512\n2.19e+17",
                "task_alias": "arc_easy_5shot",
                "task_group": "arc_openbook_sciq",
                "result_key": "arc_easy",
                "metric": "choice_prob_norm",
                "value": 0.35,
                "result_path": "gs://example/results.json",
                "checkpoint_path": "gs://example/checkpoints",
            }
        ]
    )

    fig = path_task_delta_plot(path_task_deltas(preferred, path_metrics))
    colors = {trace.name: trace.line.color for trace in fig.data}

    assert colors["Proportional"] == PATH_MIXTURE_COLORS["Proportional"]
    assert colors["v4 path t=0.50"] == PATH_MIXTURE_COLORS["v4 path t=0.50"]
    assert colors["v4"] == PATH_MIXTURE_COLORS["v4"]
