# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_nextgen_two_phase_20260921 as audit


def test_synthetic_tied_queries_replace_stored_phase_doses() -> None:
    phase0 = np.asarray([[0.8, 0.2], [0.4, 0.6]])
    phase1 = np.asarray([[0.1, 0.9], [0.7, 0.3]])
    c0, c1 = np.asarray([0.8, 8.0]), np.asarray([0.2, 2.0])
    panel = audit.BenchPanel(
        name="synthetic",
        buckets=("a", "b"),
        phase0=phase0,
        phase1=phase1,
        alpha=0.8,
        c0=c0,
        c1=c1,
        targets={},
        rows=pd.DataFrame({"tied": [False, False]}),
        input_hashes={},
        metadata={"realized_phase0_epochs": phase0 * c0, "realized_phase1_epochs": phase1 * c1},
    )
    tied = audit.tied_queries(panel)
    np.testing.assert_allclose(tied.phase0_epochs, np.asarray([[0.528, 2.72], [0.368, 4.32]]))
    np.testing.assert_allclose(tied.phase1_epochs, np.asarray([[0.132, 0.68], [0.092, 1.08]]))
    np.testing.assert_allclose(tied.epochs, panel.epochs)
    np.testing.assert_array_equal(panel.phase0_epochs, phase0 * c0)


def test_projection_does_not_invent_rank_from_roundoff() -> None:
    x = np.linspace(-1, 1, 20)
    nuisance = np.column_stack([np.ones(20), x, 2 * x])
    projected = audit.residualize(np.column_stack([3 * x + 4, x**2]), nuisance)
    assert audit.spectrum(projected)["rank"] == 1
    np.testing.assert_allclose(projected[:, 0], 0.0, atol=1e-13)
    np.testing.assert_allclose(projected[:, 1], x**2 - (x**2).mean(), atol=1e-13)


def test_recovery_scores_synthetic_tied_as_deployable_menu_choice() -> None:
    frame = pd.DataFrame(
        {
            "role": ["test", "test"],
            "row_id": ["a", "b"],
            "tied": [False, False],
            "truth": [1.1, 1.2],
            "prediction": [0.9, 1.2],
            "tied_truth": [1.0, 1.0],
            "tied_prediction": [1.0, 1.0],
        }
    )
    metrics = audit.recovery_metrics(frame)
    assert metrics["held_policy_regret"] == 0
    np.testing.assert_allclose(metrics["joint_with_synthetic_tied_regret"], 0.1)
    np.testing.assert_allclose(metrics["predicted_max_matched_gain"], 0.1)
    np.testing.assert_allclose(metrics["true_max_matched_gain"], -0.1)
    assert not metrics["joint_selected_tied"]
