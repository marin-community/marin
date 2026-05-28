# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import analyze_grug_moe_path_response as analysis


def test_task_metric_scales_use_path_and_reference_oriented_values():
    path_df = pd.DataFrame(
        [
            {"task_alias": "task_a", "preferred_metric": "negative_bpb", "oriented_value": 1.0},
            {"task_alias": "task_a", "preferred_metric": "negative_bpb", "oriented_value": 3.0},
        ]
    )
    reference_df = pd.DataFrame(
        [
            {"task_alias": "task_a", "preferred_metric": "negative_bpb", "oriented_value": 5.0},
            {"task_alias": "task_b", "preferred_metric": "choice_prob_norm", "oriented_value": 7.0},
        ]
    )

    scales = analysis.task_metric_scales(path_df, reference_df)

    task_a = scales.set_index("task_alias").loc["task_a"]
    assert task_a["metric_scale_n"] == 3
    assert task_a["metric_scale_std"] == pytest.approx(2.0)
    assert task_a["metric_scale_mad"] == pytest.approx(1.4826 * 2.0)
    task_b = scales.set_index("task_alias").loc["task_b"]
    assert np.isnan(task_b["metric_scale_std"])


def test_add_standardized_path_deltas_preserves_sign_and_native_delta():
    path_df = pd.DataFrame(
        [
            {
                "task_alias": "task_a",
                "preferred_metric": "negative_bpb",
                "hidden_dim": 512,
                "t": 1.0,
                "oriented_value": 3.0,
                "delta_oriented": 0.5,
            },
            {
                "task_alias": "task_b",
                "preferred_metric": "choice_prob_norm",
                "hidden_dim": 512,
                "t": 1.0,
                "oriented_value": 10.0,
                "delta_oriented": -1.0,
            },
        ]
    )
    scales = pd.DataFrame(
        [
            {
                "task_alias": "task_a",
                "preferred_metric": "negative_bpb",
                "metric_scale_std": 0.25,
                "metric_scale_mad": 0.5,
                "metric_scale_n": 4,
            },
            {
                "task_alias": "task_b",
                "preferred_metric": "choice_prob_norm",
                "metric_scale_std": np.nan,
                "metric_scale_mad": 2.0,
                "metric_scale_n": 1,
            },
        ]
    )

    standardized = analysis.add_standardized_path_deltas(path_df, scales)

    task_a = standardized[standardized["task_alias"].eq("task_a")].iloc[0]
    assert task_a["delta_oriented"] == pytest.approx(0.5)
    assert task_a["delta_std"] == pytest.approx(2.0)
    assert task_a["delta_mad"] == pytest.approx(1.0)

    task_b = standardized[standardized["task_alias"].eq("task_b")].iloc[0]
    assert task_b["delta_oriented"] == pytest.approx(-1.0)
    assert np.isnan(task_b["delta_std"])
    assert task_b["delta_mad"] == pytest.approx(-0.5)
