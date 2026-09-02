# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    prepare_single_phase_heldout_benchmark_20260902 as heldout,
)


def test_nearest_fit_distance_uses_maximum_coordinate_distance() -> None:
    fit = np.asarray([[0.2, 0.8], [0.6, 0.4]])
    candidates = np.asarray([[0.21, 0.79], [0.5, 0.5]])

    observed = heldout.nearest_fit_distance(candidates, fit)

    np.testing.assert_allclose(observed, [0.01, 0.1])


def test_finalize_audit_excludes_nontied_and_fit_coordinates() -> None:
    weights = np.zeros((3, 39))
    weights[:, 0] = [1.0, 0.8, 0.7]
    weights[:, 1] = [0.0, 0.2, 0.3]
    frame = pd.DataFrame(
        {
            "panel": ["test"] * 3,
            "phase_tv": [0.0, 0.1, 0.0],
            "uncheatable_bpb": [1.0, 1.1, np.nan],
            "table9_macro_bpb": [1.2, 1.3, np.nan],
        }
    )

    observed = heldout._finalize_audit(frame, weights, weights[[0]])

    assert observed["exclusion_reason"].tolist() == [
        "fit_coordinate_overlap",
        "not_single_phase",
        "missing_primary_targets",
    ]
    assert not observed["eligible"].any()


def test_coordinate_table_preserves_replicate_noise() -> None:
    weights = {f"weight::{bucket}": [1.0 / 39, 1.0 / 39] for bucket in heldout.domains()}
    runs = pd.DataFrame(
        {
            "panel": ["delphi_3e18_39bucket"] * 2,
            "scale": ["3e18 FLOPs"] * 2,
            "coordinate_id": ["same", "same"],
            "source": ["first", "second"],
            "row_id": ["run0", "run1"],
            "uncheatable_bpb": [1.0, 1.2],
            "table9_macro_bpb": [1.1, 1.3],
            **weights,
        }
    )

    observed = heldout.coordinate_table(runs)

    assert len(observed) == 1
    assert observed.loc[0, "run_count"] == 2
    assert observed.loc[0, "uncheatable_mean_bpb"] == 1.1
    np.testing.assert_allclose(observed.loc[0, "uncheatable_sd_bpb"], np.sqrt(0.02))


def test_table9_summary_fallback_uses_native_component_keys() -> None:
    components = heldout.table9_components()
    summary_keys = heldout.table9_summary_keys()
    summary = {summary_keys[component]: position / 10 for position, component in enumerate(components)}
    request = heldout.ComponentRequest(
        row_id="row",
        panel="delphi_3e18_39bucket",
        target="table9",
        project="project",
        wandb_run_id="run",
        expected_aggregate=0.0,
    )

    observed = heldout._summary_aggregate(summary, request)

    assert observed == pytest.approx(np.mean(list(summary.values())))


def test_300m_audit_keeps_both_external_validation_panels() -> None:
    audited = heldout._audit_300m(heldout.domains())
    eligible = audited[audited["eligible"]]

    assert eligible.groupby("source").size().to_dict() == {
        "extra_300m_diagnostics": 56,
        "proportional_controllability_tilts": 78,
    }
    assert eligible["coordinate_id"].nunique() == 134
    assert eligible["uncheatable_bpb"].notna().sum() == 117
    assert eligible["table9_macro_bpb"].notna().sum() == 134
