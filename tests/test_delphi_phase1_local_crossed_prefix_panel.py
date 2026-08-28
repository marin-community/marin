# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_crossed_prefix_panel as launch
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_local_crossed_prefix_panel_20260828 as design,
)


def artifact_paths():
    return (
        design.DEFAULT_OUTPUT_DIR / "manifest.json",
        design.DEFAULT_OUTPUT_DIR / "prefix_registry.json",
        design.DEFAULT_OUTPUT_DIR / "panel_rows.csv",
        design.DEFAULT_OUTPUT_DIR / "panel_weights.csv",
    )


def test_local_action_bank_crosses_every_action_with_every_prefix() -> None:
    manifest_path, registry_path, rows_path, weights_path = artifact_paths()
    manifest, prefixes, rows, weights = launch.load_artifacts(
        manifest_path,
        launch.file_sha256(manifest_path),
        design.CONTRACT_VERSION,
        registry_path,
        launch.file_sha256(registry_path),
        rows_path,
        launch.file_sha256(rows_path),
        weights_path,
        launch.file_sha256(weights_path),
    )
    fit_rows = rows.loc[rows.fit_budget]
    rank_audit = cast(dict[str, object], manifest["rank_audit"])

    assert len(prefixes) == 9
    assert fit_rows.groupby("prefix_state_id").size().eq(10).all()
    assert fit_rows.groupby("continuation_id").size().eq(9).all()
    assert rank_audit["anchor_tangent_rank"] == 6
    assert rank_audit["unresolved_tangent_dimensions"] == 32
    assert rank_audit["residual_degrees_of_freedom_per_state"] == 3
    assert rank_audit["maximum_hellinger_to_anchor"] < design.MAX_LOCAL_HELLINGER
    assert manifest["primary_inference"]["confirmatory_state_ids"] == list(design.CONFIRMATORY_STATE_IDS)
    assert manifest["secondary_inference"]["multiplicity"].startswith("Holm-adjust")
    assert manifest["state_analysis_contract"]["required_sensitivities"]["within_10_epoch_confirmatory_subset"] == list(
        design.WITHIN_EXPOSURE_SENSITIVITY_STATE_IDS
    )
    assert len(manifest["secondary_inference"]["interior_odd_contrasts"]) == 3
    assert "descriptive" in manifest["descriptive_analysis"]["boundary_limitation"]

    fit_weights = weights.merge(
        fit_rows[["row_id", "continuation_id"]],
        on="row_id",
        validate="many_to_one",
    )
    unique_counts = fit_weights.groupby(["continuation_id", "bucket"]).phase_1_count.nunique()
    assert unique_counts.eq(1).all()


def test_sparse_local_screen_preserves_paired_interior_and_boundary_directions() -> None:
    manifest_path, _registry_path, rows_path, weights_path = artifact_paths()
    manifest = json.loads(manifest_path.read_text())
    rows = pd.read_csv(rows_path)
    weights = pd.read_csv(weights_path)
    state_id = rows.prefix_state_id.iloc[0]
    state_fit = rows.loc[rows.prefix_state_id.eq(state_id) & rows.fit_budget]
    action_counts = {
        str(row.continuation_id): weights.loc[weights.row_id.eq(row.row_id)].phase_1_count.to_numpy(dtype=int)
        for row in state_fit.itertuples()
    }
    matrix = np.stack(list(action_counts.values()))
    anchor = action_counts["local_anchor_fit079"]
    projection = np.eye(matrix.shape[1]) - np.ones((matrix.shape[1], matrix.shape[1])) / matrix.shape[1]
    basis = np.linalg.svd(projection)[0][:, : matrix.shape[1] - 1]
    offsets = (matrix / design.MIXTURE_BLOCK_SIZE - anchor / design.MIXTURE_BLOCK_SIZE) @ basis
    singular = np.linalg.svd(offsets, compute_uv=False)
    hellinger = np.linalg.norm(
        np.sqrt(matrix / design.MIXTURE_BLOCK_SIZE) - np.sqrt(anchor / design.MIXTURE_BLOCK_SIZE),
        axis=1,
    ) / np.sqrt(2.0)

    assert len({tuple(row) for row in matrix}) == 10
    assert np.all(matrix.sum(axis=1) == design.MIXTURE_BLOCK_SIZE)
    assert int(matrix.min()) >= 0
    assert sum(state_fit.role.eq("local_boundary_activation")) == 3
    assert sum(state_fit.role.eq("local_interior_forward")) == 3
    assert sum(state_fit.role.eq("local_interior_reverse")) == 3
    assert np.linalg.matrix_rank(offsets, tol=1e-12) == 6
    assert singular[0] / singular[5] < design.MAX_SCREEN_CONDITION_NUMBER
    assert np.isclose(singular[0] / singular[5], manifest["rank_audit"]["anchor_condition_number"])
    assert float(hellinger[hellinger > 0.0].min()) >= design.MINIMUM_LOCAL_HELLINGER
    assert float(hellinger.max()) < design.MAX_LOCAL_HELLINGER
    assert manifest["rank_audit"]["boundary_activation_forward_directions"] == 3
    assert manifest["rank_audit"]["interior_forward_directions"] == 3
    assert manifest["rank_audit"]["unresolved_tangent_dimensions"] == 32

    for action_id, counts in action_counts.items():
        if not action_id.startswith("local_minus_"):
            continue
        forward = action_counts[action_id.replace("local_minus_", "local_plus_")]
        np.testing.assert_array_equal(counts - anchor, -(forward - anchor))

    boundary_actions = state_fit.loc[state_fit.role.eq("local_boundary_activation"), "continuation_id"]
    for action_id in boundary_actions:
        target = int(str(action_id).removeprefix("local_plus_"))
        assert anchor[target] == 0

    interior_actions = state_fit.loc[state_fit.role.eq("local_interior_forward"), "continuation_id"]
    for action_id in interior_actions:
        target = int(str(action_id).removeprefix("local_plus_"))
        assert anchor[target] > 0


def test_frozen_action_bank_matches_design_functions() -> None:
    _manifest_path, _registry_path, rows_path, weights_path = artifact_paths()
    geometry = design.common_design.load_canonical_panel_geometry()
    anchor = design.frontier_counts(design.DEFAULT_FRONTIER_CONTRACT, geometry.buckets)
    actions, audit = design.local_action_bank(anchor, geometry.buckets, geometry.c1)
    rows = pd.read_csv(rows_path)
    weights = pd.read_csv(weights_path)
    state_fit = rows.loc[rows.prefix_state_id.eq(rows.prefix_state_id.iloc[0]) & rows.fit_budget]

    assert [action["action_id"] for action in actions] == [
        "local_anchor_fit079",
        "local_plus_37",
        "local_minus_37",
        "local_plus_32",
        "local_minus_32",
        "local_plus_24",
        "local_minus_24",
        "local_plus_30",
        "local_plus_10",
        "local_plus_20",
    ]
    assert audit["paired_interior_eligible_candidate_count"] == 9
    assert audit["minimum_nonzero_hellinger_to_anchor"] >= design.MINIMUM_LOCAL_HELLINGER
    for action in actions:
        row_id = state_fit.loc[state_fit.continuation_id.eq(action["action_id"]), "row_id"].item()
        observed = weights.loc[weights.row_id.eq(row_id), "phase_1_count"].to_numpy(dtype=int)
        np.testing.assert_array_equal(observed, action["counts"])


def test_local_anchor_matches_validated_frontier_contract() -> None:
    _manifest_path, _registry_path, rows_path, weights_path = artifact_paths()
    rows = pd.read_csv(rows_path)
    weights = pd.read_csv(weights_path)
    anchor_row = rows.loc[
        rows.prefix_state_id.eq(rows.prefix_state_id.iloc[0]) & rows.continuation_id.eq("local_anchor_fit079")
    ].iloc[0]
    anchor_weights = weights.loc[weights.row_id.eq(anchor_row.row_id)]
    contract = json.loads(design.DEFAULT_FRONTIER_CONTRACT.read_text())
    expected = np.asarray([contract["runtime_counts"][bucket] for bucket in anchor_weights.bucket], dtype=int)

    np.testing.assert_array_equal(anchor_weights.phase_1_count.to_numpy(dtype=int), expected)


def test_panel_controls_use_matched_and_disjoint_data_streams() -> None:
    _manifest_path, registry_path, rows_path, _weights_path = artifact_paths()
    registry = json.loads(registry_path.read_text())
    rows = pd.read_csv(rows_path)
    prefix_data_seeds = {int(prefix["run_spec"]["data_seed"]) for prefix in registry["prefixes"]}

    assert rows.groupby("prefix_state_id").size().eq(13).all()
    assert rows.row_id.str.startswith("screen10__").all()
    assert rows.loc[rows.fit_budget].data_seed.eq(design.FIT_DATA_SEED).all()
    assert rows.loc[rows.role.eq("prefix_tied_control")].data_seed.eq(design.FIT_DATA_SEED).all()
    assert rows.loc[rows.role.eq("anchor_exposure_noise_control")].data_seed.eq(design.ANCHOR_REPEAT_DATA_SEED).all()
    assert (
        rows.loc[rows.role.eq("local_action_exposure_noise_control")]
        .data_seed.eq(design.SENTINEL_REPEAT_DATA_SEED)
        .all()
    )
    assert not prefix_data_seeds.intersection(
        {design.FIT_DATA_SEED, design.ANCHOR_REPEAT_DATA_SEED, design.SENTINEL_REPEAT_DATA_SEED}
    )
    assert rows.loc[rows.role.eq("local_action_exposure_noise_control"), "action_id"].eq("local_plus_24").all()


def test_completed_bridge_prefix_is_reused_without_commit_rewrite() -> None:
    manifest_path, registry_path, rows_path, weights_path = artifact_paths()
    _manifest, prefixes, _rows, _weights = launch.load_artifacts(
        manifest_path,
        launch.file_sha256(manifest_path),
        design.CONTRACT_VERSION,
        registry_path,
        launch.file_sha256(registry_path),
        rows_path,
        launch.file_sha256(rows_path),
        weights_path,
        launch.file_sha256(weights_path),
    )
    bridge = next(prefix for prefix in prefixes if prefix.state_id == launch.design.BRIDGE_STATE_ID)

    resolved = launch.resolve_prefix(bridge, "a" * 40)

    assert resolved == bridge
    assert resolved.checkpoint_ready_at_design_time is True
    assert resolved.checkpoint_uri == design.BRIDGE_CHECKPOINT_URI
    assert resolved.prefix_replay_code_commit == design.BRIDGE_CODE_COMMIT
