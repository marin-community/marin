# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as launch
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design,
)


def _cap4_boundary_design() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    panel = design.common_design.load_canonical_panel_geometry()
    candidate_id = "cap4_shared_bounded_ensemble_kl0"
    centers = design.candidate_centers(design.DEFAULT_CANDIDATE_WEIGHTS, (candidate_id,), panel.buckets)
    center = centers[candidate_id]
    anchors = {
        "proportional": design.runtime_weights(panel.proportional),
        "uniform": np.full(len(panel.buckets), 1.0 / len(panel.buckets)),
    }
    pool, anchor_keys = design.generate_pool(center, anchors, center * panel.c0, panel.c1)
    selected, diagnostics = design.select_fit_points(pool, anchor_keys, center)
    referees = design.select_referee_points(pool, selected, center)
    summary_rows, weight_rows = design.design_rows(
        candidate_id,
        center,
        selected,
        referees,
        panel.buckets,
        panel.c0,
        panel.c1,
    )
    manifest = {
        "selected_candidate_ids": [candidate_id],
        "rows": {
            "controls_per_prefix": design.CONTROL_ROWS_PER_PREFIX,
            "fit_per_prefix": design.FIT_ROWS_PER_PREFIX,
            "sealed_referees_per_prefix": design.REFEREE_ROWS_PER_PREFIX,
            "total": design.FIT_ROWS_PER_PREFIX + design.REFEREE_ROWS_PER_PREFIX + design.CONTROL_ROWS_PER_PREFIX,
        },
        "diagnostics": {candidate_id: diagnostics},
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(weight_rows), manifest


def _write_design(tmp_path: Path) -> tuple[Path, Path, Path]:
    summary, weights, manifest = _cap4_boundary_design()
    summary_path = tmp_path / "continuation_summary.csv"
    weights_path = tmp_path / "continuation_weights.csv"
    manifest_path = tmp_path / "manifest.json"
    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return summary_path, weights_path, manifest_path


def test_boundary_design_is_full_rank_and_keeps_referees_sealed() -> None:
    summary, weights, manifest = _cap4_boundary_design()
    candidate_id = "cap4_shared_bounded_ensemble_kl0"

    assert len(summary) == launch.ROWS_PER_PREFIX
    assert summary.fit_budget.sum() == launch.FIT_ROWS_PER_PREFIX
    assert summary.role.value_counts().to_dict() == {
        "fixed_prefix_response_fit": launch.FIT_ROWS_PER_PREFIX,
        "sealed_geometry_referee": launch.REFEREE_ROWS_PER_PREFIX,
        "prefix_state_tied_control": 4,
        "fresh_tied_control": 3,
        "common_random_tied_control": 1,
    }
    fit_ids = set(summary.loc[summary.fit_budget, "continuation_id"])
    referee_ids = set(summary.loc[summary.role.eq("sealed_geometry_referee"), "continuation_id"])
    assert fit_ids.isdisjoint(referee_ids)
    assert weights.total_materialized_epochs.max() <= design.TOTAL_MATERIALIZED_EPOCH_CAP
    assert manifest["diagnostics"][candidate_id]["direct_feature_rank"] == 38
    assert manifest["diagnostics"][candidate_id]["sqrt_feature_rank"] == 39


def test_load_design_preserves_common_random_number_identity(tmp_path: Path) -> None:
    summary_path, weights_path, manifest_path = _write_design(tmp_path)

    rows = launch.load_design(
        summary_path,
        launch.file_sha256(summary_path),
        weights_path,
        launch.file_sha256(weights_path),
        manifest_path,
        launch.file_sha256(manifest_path),
        ("cap4_shared_bounded_ensemble_kl0",),
    )

    fit_rows = [row for row in rows if row["fit_budget"]]
    referees = [row for row in rows if row["role"] == "sealed_geometry_referee"]
    matched = [row for row in rows if row["role"] == "common_random_tied_control"]
    assert {row["data_seed"] for row in [*fit_rows, *referees, *matched]} == {design.FIT_DATA_SEED}
    assert len(fit_rows) == launch.FIT_ROWS_PER_PREFIX
    assert len(referees) == launch.REFEREE_ROWS_PER_PREFIX
    assert len(matched) == 1


def test_selected_prefix_loader_accepts_one_cap(tmp_path: Path) -> None:
    payload = {
        "candidate_weights_sha256": "candidate-hash",
        "prefix_replay_code_commit": "prefix-commit",
        "selected_aliases": [{"canonical_candidate_id": "cap4_shared_bounded_ensemble_kl0"}],
        "prefixes": [
            {
                "canonical_candidate_id": "cap4_shared_bounded_ensemble_kl0",
                "repeat_seed": seed,
                "checkpoint_uri": f"gs://marin-us-east5/prefix-seed-{seed}/checkpoints/step-2399",
                "provenance_sha256": f"provenance-{seed}",
            }
            for seed in (0, 1)
        ],
    }
    path = tmp_path / "selected.json"
    encoded = (json.dumps(payload) + "\n").encode()
    path.write_bytes(encoded)

    rows, loaded = launch.selected_prefixes(
        str(path), hashlib.sha256(encoded).hexdigest(), "candidate-hash", "prefix-commit"
    )

    assert loaded == payload
    assert [asdict(row) for row in rows] == [
        {
            "candidate_id": "cap4_shared_bounded_ensemble_kl0",
            "repeat_seed": seed,
            "checkpoint_uri": f"gs://marin-us-east5/prefix-seed-{seed}/checkpoints/step-2399",
            "provenance_sha256": f"provenance-{seed}",
        }
        for seed in (0, 1)
    ]


def test_manifest_write_is_idempotent(tmp_path: Path) -> None:
    config = launch.SaveManifestConfig(
        experiment_name="experiment",
        output_path=str(tmp_path),
        selected_prefixes_json="[]",
        selected_prefixes_sha256="selected",
        candidate_weights_sha256="candidates",
        candidate_aliases_sha256="aliases",
        continuation_summary_sha256="summary",
        continuation_weights_sha256="weights",
        design_manifest_sha256="design",
        prefix_replay_code_commit="prefix-commit",
        code_commit="branch-commit",
        branch_run_id_base=973_000,
        full_design_rows=1,
        branch_rows_json=json.dumps([{"fit_budget": True, "role": "fixed_prefix_response_fit", "run_order": 0}]),
    )

    launch.save_manifest(config)
    first = (tmp_path / "manifest.json").read_bytes()
    launch.save_manifest(config)

    assert (tmp_path / "manifest.json").read_bytes() == first
