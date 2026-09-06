# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json

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


def test_recent_delphi_audit_includes_successor_epoch_cap_validation() -> None:
    audited = heldout._audit_recent_delphi(heldout.domains())
    successor = audited[audited["source"].eq("weibull_softplus_unscaled_epoch_cap")]

    assert len(successor) == 12
    assert successor["eligible"].all()
    assert successor["coordinate_id"].nunique() == 12
    assert successor["uncheatable_bpb"].notna().all()
    assert successor["table9_macro_bpb"].notna().all()


def test_epoch_dose_audit_preserves_all_rows_and_excludes_fit_overlap() -> None:
    audited = heldout._audit_epoch_dose_delphi(heldout.domains())
    eligible = audited[audited["eligible"]]
    materialization = json.loads(heldout.EPOCH_DOSE_MATERIALIZATION_MANIFEST.read_text())

    assert len(audited) == 277
    assert audited["uncheatable_bpb"].notna().sum() == 277
    assert audited["table9_macro_bpb"].notna().sum() == materialization["counts"]["table9_complete"]
    assert audited["exclusion_reason"].value_counts().to_dict() == {"": 237, "fit_coordinate_overlap": 40}
    assert eligible["coordinate_id"].nunique() == 237
    assert eligible["uncheatable_bpb"].notna().all()


def test_epoch_dose_local_components_are_never_partial() -> None:
    audited = heldout._audit_epoch_dose_delphi(heldout.domains())
    eligible = audited[audited["eligible"]]
    components = heldout._local_epoch_dose_components(eligible)
    counts = components.groupby(["row_id", "target"]).size()

    assert counts.xs("uncheatable", level="target").eq(7).all()
    assert counts.xs("table9", level="target").eq(51).all()


def test_recent_table9_components_use_canonical_inventory() -> None:
    audited = heldout._audit_recent_delphi(heldout.domains())
    eligible = audited[audited["eligible"]]
    components = heldout._local_table9_components(eligible)
    groups = components[components["target"].eq("table9")].groupby("row_id")

    assert groups.ngroups == 34
    for _row_id, group in groups:
        observed = tuple(group.sort_values("component_position")["component"])
        assert observed == heldout.table9_components()


def test_component_validation_rejects_noncanonical_table9_names() -> None:
    names = list(heldout.table9_components())
    names[0] = "minerva_math_algebra"
    runs = pd.DataFrame(
        {
            "row_id": ["row"],
            "panel": ["delphi_3e18_39bucket"],
            "uncheatable_bpb": [np.nan],
            "table9_macro_bpb": [1.0],
        }
    )
    components = pd.DataFrame(
        {
            "row_id": ["row"] * 51,
            "target": ["table9"] * 51,
            "component_position": range(51),
            "component": names,
            "bpb": [1.0] * 51,
        }
    )

    with pytest.raises(ValueError, match="canonical component inventory"):
        heldout._validate_components(runs, components)


def test_coordinate_identity_preserves_full_support_and_separates_subsampled_support() -> None:
    weights = np.arange(1, 40, dtype=float)
    weights /= weights.sum()
    old_digest = hashlib.sha256(
        b"delphi_3e18_39bucket\0" + np.round(np.asarray(weights, dtype="<f8"), 12).tobytes()
    ).hexdigest()

    full_support = heldout.coordinate_id("delphi_3e18_39bucket", weights, np.ones(39))
    subsampled = np.ones(39)
    subsampled[-1] = 0.5

    assert full_support == f"delphi_3e18_39bucket:{old_digest}"
    assert heldout.coordinate_id("delphi_3e18_39bucket", weights) == full_support
    assert heldout.coordinate_id("delphi_3e18_39bucket", weights, subsampled) != full_support


def test_subsampled_support_is_not_a_fit_coordinate_overlap() -> None:
    buckets = heldout.domains()
    weights = np.full((2, len(buckets)), 1.0 / len(buckets))
    fractions = np.ones_like(weights)
    fractions[1, -1] = 0.5
    frame = pd.DataFrame(
        {
            "panel": ["delphi_3e18_39bucket"] * 2,
            "phase_tv": [0.0, 0.0],
            "uncheatable_bpb": [1.0, 1.0],
            "table9_macro_bpb": [np.nan, np.nan],
        }
    )

    audited = heldout._finalize_audit(frame, weights, weights[:1], fractions)

    assert audited.loc[0, "exclusion_reason"] == "fit_coordinate_overlap"
    assert audited.loc[1, "eligible"]
    assert audited.loc[1, f"pool_fraction::{buckets[-1]}"] == 0.5
    assert audited.loc[0, "coordinate_id"] != audited.loc[1, "coordinate_id"]


def test_apriori_audit_uses_runtime_pool_fractions_and_seeds(tmp_path, monkeypatch) -> None:
    design = pd.read_csv(heldout.APRIORI_SWARM_DESIGN, low_memory=False).reset_index(names="run_order")
    canary = design[design["run_name"].str.contains("synth_qa_pool0.5_block0")].iloc[0]
    result = {
        "run_order": int(canary["run_order"]),
        "run_name": canary["run_name"],
        "training_wandb_run_id": "train",
        "training_wandb_url": "https://wandb.invalid/train",
        "table9_eval_run_id": "",
        "table9_eval_url": "",
        "data_seed": int(canary["data_seed"]),
        "trainer_seed": int(canary["trainer_seed"]),
        "subset_seed": int(canary["subset_seed"]),
        "uncheatable_bpb": 1.0,
        "table9_macro_bpb": np.nan,
    }
    for bucket in heldout.domains():
        result[f"pool_fraction::{bucket}"] = float(canary[f"pool_fraction_{bucket}"])
    results_path = tmp_path / "heldout_results.csv"
    pd.DataFrame([result]).to_csv(results_path, index=False)
    monkeypatch.setattr(heldout, "APRIORI_SWARM_RESULTS", results_path)

    audited = heldout._audit_apriori_delphi(heldout.domains())

    assert len(audited) == 1 and audited.iloc[0]["eligible"]
    assert audited.iloc[0]["subset_seed"] == 662_009
    assert audited.iloc[0]["pool_fraction::dolmino_synth_qa"] == 0.5
