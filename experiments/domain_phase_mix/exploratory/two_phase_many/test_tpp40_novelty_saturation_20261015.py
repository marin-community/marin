# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented metadata and native-equation contracts; no scientific panel loading."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_novelty_saturation_20261015 as diagnostic


def toy_frames() -> dict[str, pd.DataFrame]:
    policies = pd.DataFrame(
        {
            "order": [0, 1],
            "run_name": ["one", "two"],
            "source": ["toy", "toy"],
            "tied": [False, True],
            "alpha_nominal": [0.8, 0.8],
            "alpha_realized": [0.75, 0.75],
            "simulated_epoch_target_budget": [100.0, 100.0],
        }
    )
    axis = pd.DataFrame(
        {
            "bucket_index": [0, 1],
            "bucket": ["large", "small"],
            "pool_tokens": [50.0, 10.0],
            "epochs_per_unit_weight": [2.0, 10.0],
        }
    )
    weights = pd.DataFrame(
        {
            "order": [0, 0, 1, 1],
            "bucket_index": [0, 1, 0, 1],
            "bucket": ["large", "small", "large", "small"],
            "phase0_weight": [1.0, 0.0, 0.5, 0.5],
            "phase1_weight": [0.0, 1.0, 0.5, 0.5],
            "exposure_epochs_nominal": [1.6, 2.0, 1.0, 5.0],
        }
    )
    return {"policies.csv": policies, "policy_weights.csv": weights, "bucket_axis.csv": axis}


def test_physical_metadata_and_poisoned_unread_columns(tmp_path: Path) -> None:
    frames = toy_frames()
    for name, frame in frames.items():
        frame.assign(outcome=["poison"] * len(frame)).to_csv(tmp_path / name, index=False)
    loaded = diagnostic.read_metadata(tmp_path)
    assert all("outcome" not in frame for frame in loaded.values())
    rows = diagnostic.coordinates(loaded)
    np.testing.assert_allclose(rows.early_epochs, [1.5, 0.0, 0.75, 3.75])
    np.testing.assert_allclose(rows.late_epochs, [0.0, 2.5, 0.25, 1.25])
    assert rows.support.tolist() == ["early_only", "late_only", "both_positive", "both_positive"]
    for name, frame in frames.items():
        frame.assign(outcome=np.arange(len(frame)) * 1e30).to_csv(tmp_path / name, index=False)
    pd.testing.assert_frame_equal(rows, diagnostic.coordinates(diagnostic.read_metadata(tmp_path)))


def test_native_novelty_survives_subtraction_cancellation() -> None:
    early = np.array([1.0, 0.0, 2.0, 0.0])
    late = np.array([1e-18, 2.0, 0.0, 0.0])
    values = diagnostic.native_features(early, late, 0.5, 1.0)
    np.testing.assert_allclose(values["late_novelty"][0], np.exp(-0.5) * 0.5e-18, rtol=1e-14, atol=0)
    assert values["benefit_total"][0] - values["benefit_early"][0] == 0
    np.testing.assert_allclose(values["late_novelty"][1:], [-np.expm1(-1), 0, 0], rtol=1e-14, atol=0)
    np.testing.assert_allclose(values["positive_headroom"] + values["benefit_total"], 1, rtol=0, atol=2e-16)


def test_entire_grid_summary_retains_roles_buckets_and_support() -> None:
    rows = diagnostic.coordinates(toy_frames())
    arrays, support, buckets = diagnostic.diagnostic_tables(rows)
    assert arrays["late_novelty"].shape == (28, 2, 2)
    selected = support.query("rate == 0.5 and power == 1 and policy_role == 'all' and support == 'all'").iloc[0]
    assert selected.n_coordinates == 4
    assert selected.hazard_early_le_1 == 3
    expected = -np.expm1(-np.array([0.75, 1.25, 0.5, 2.5]))
    np.testing.assert_allclose(selected["benefit_total_q0.5"], np.median(expected), rtol=1e-14, atol=0)
    assert buckets.query("rate == 0.5 and power == 1 and policy_role == 'tied'").n_coordinates.tolist() == [1, 1]
    assert set(support.query("policy_role == 'asymmetric'").support) == {"all", "early_only", "late_only"}
