# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_crossed_prefix_panel as launch
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_crossed_prefix_panel_20260827 as design,
)


def test_complete_common_branch_bank_spans_centered_simplex_tangent() -> None:
    weights = pd.read_csv(design.DEFAULT_COMMON_WEIGHTS)
    selected, audit = design.common_branch_ids(weights)

    assert selected == tuple(f"fit_maximin_{position:02d}" for position in range(50))
    assert audit["centered_tangent_rank"] == 38
    assert audit["residual_degrees_of_freedom_per_state"] == 11
    assert audit["centered_condition_number"] < 25.0


def test_frozen_panel_crosses_all_actions_with_all_prefix_states() -> None:
    rows = pd.read_csv(launch.DEFAULT_PANEL_ROWS)
    weights = pd.read_csv(launch.DEFAULT_PANEL_WEIGHTS)
    fit_rows = rows.loc[rows.fit_budget]

    assert fit_rows.groupby("prefix_state_id").size().eq(50).all()
    assert fit_rows.groupby("continuation_id").size().eq(9).all()
    fit_weights = weights.merge(fit_rows[["row_id", "continuation_id"]], on="row_id", validate="many_to_one")
    unique_actions = fit_weights.groupby(["continuation_id", "bucket"]).phase_1_count.nunique()
    assert unique_actions.eq(1).all()
    assert rows.groupby("prefix_state_id").size().eq(53).all()


def test_panel_contains_seed_and_hardware_controls_without_old_cell_reuse() -> None:
    manifest = json.loads(launch.DEFAULT_MANIFEST.read_text())
    registry = json.loads(launch.DEFAULT_PREFIX_REGISTRY.read_text())

    assert design.SEED_REPLICATE_STATE_ID in registry["state_ids"]
    assert design.BRIDGE_STATE_ID in registry["state_ids"]
    assert manifest["reused_fit_rows"] == 0
    assert manifest["controls"]["prefix_seed_single_draw_diagnostic_state"] == design.SEED_REPLICATE_STATE_ID
    assert manifest["controls"]["phase0_hardware_bridge_state"] == design.BRIDGE_STATE_ID
    assert manifest["state_roles"]["observed_cap10_best"].startswith("descriptive-only")
    assert "hardware-by-code" in manifest["controls"]["phase0_hardware_bridge_interpretation"]


def test_fit_seed_is_disjoint_from_every_prefix_stream() -> None:
    registry = json.loads(launch.DEFAULT_PREFIX_REGISTRY.read_text())
    prefix_data_seeds = {int(prefix["run_spec"]["data_seed"]) for prefix in registry["prefixes"]}

    assert design.FIT_DATA_SEED not in prefix_data_seeds
    assert design.FIT_DATA_SEED not in {design.LOW_SENTINEL_DATA_SEED, design.HIGH_SENTINEL_DATA_SEED}


def test_panel_contracts_pin_old_and_sparse_allocations() -> None:
    old = launch.panel_contract(launch.EXPECTED_CONTRACT_VERSION)
    sparse = launch.panel_contract(launch.LOCAL_SCREEN_CONTRACT_VERSION)

    assert (old.fit_branches_per_prefix, old.tangent_rank, old.residual_degrees_of_freedom) == (50, 38, 11)
    assert (sparse.fit_branches_per_prefix, sparse.tangent_rank, sparse.residual_degrees_of_freedom) == (10, 6, 3)
    assert old.experiment_name != sparse.experiment_name
    assert old.panel_source != sparse.panel_source


def test_default_dry_run_output_is_scoped_by_commit_and_selection() -> None:
    first = launch.default_dry_run_output_dir(launch.DEFAULT_DESIGN_DIR, "a" * 64, "b" * 40, (0, 1))

    assert first == launch.default_dry_run_output_dir(launch.DEFAULT_DESIGN_DIR, "a" * 64, "b" * 40, (0, 1))
    assert first != launch.default_dry_run_output_dir(launch.DEFAULT_DESIGN_DIR, "a" * 64, "c" * 40, (0, 1))
    assert first != launch.default_dry_run_output_dir(launch.DEFAULT_DESIGN_DIR, "a" * 64, "b" * 40, (0, 2))


def test_branch_run_spec_preserves_prefix_state_and_changes_only_continuation() -> None:
    manifest_hash = launch.file_sha256(launch.DEFAULT_MANIFEST)
    registry_hash = launch.file_sha256(launch.DEFAULT_PREFIX_REGISTRY)
    rows_hash = launch.file_sha256(launch.DEFAULT_PANEL_ROWS)
    weights_hash = launch.file_sha256(launch.DEFAULT_PANEL_WEIGHTS)
    _manifest, prefixes, rows, weights = launch.load_artifacts(
        launch.DEFAULT_MANIFEST,
        manifest_hash,
        launch.EXPECTED_CONTRACT_VERSION,
        launch.DEFAULT_PREFIX_REGISTRY,
        registry_hash,
        launch.DEFAULT_PANEL_ROWS,
        rows_hash,
        launch.DEFAULT_PANEL_WEIGHTS,
        weights_hash,
    )
    prefix = prefixes[0]
    row = rows.loc[rows.prefix_state_id.eq(prefix.state_id) & rows.continuation_id.eq("fit_maximin_00")].iloc[0]

    result = launch.branch_run_spec(
        prefix,
        row,
        weights,
        launch.EXPERIMENT_NAME,
        "delphi_phase1_crossed_prefix_panel",
    )

    assert result.phase_weights["phase_0"] == prefix.run_spec.phase_weights["phase_0"]
    assert result.phase_weights["phase_1"] != prefix.run_spec.phase_weights["phase_1"]
    assert np.isclose(sum(result.phase_weights["phase_1"].values()), 1.0)
    assert (result.tpu_type, result.tpu_region, result.tpu_zone) == ("v6e-8", "us-east5", "us-east5-b")
    assert result.train_steps == prefix.run_spec.train_steps
    assert result.data_seed == design.FIT_DATA_SEED


def test_bridge_prefix_resolves_to_commit_scoped_output() -> None:
    registry_hash = launch.file_sha256(launch.DEFAULT_PREFIX_REGISTRY)
    _manifest, prefixes, _rows, _weights = launch.load_artifacts(
        launch.DEFAULT_MANIFEST,
        launch.file_sha256(launch.DEFAULT_MANIFEST),
        launch.EXPECTED_CONTRACT_VERSION,
        launch.DEFAULT_PREFIX_REGISTRY,
        registry_hash,
        launch.DEFAULT_PANEL_ROWS,
        launch.file_sha256(launch.DEFAULT_PANEL_ROWS),
        launch.DEFAULT_PANEL_WEIGHTS,
        launch.file_sha256(launch.DEFAULT_PANEL_WEIGHTS),
    )
    bridge = next(prefix for prefix in prefixes if prefix.state_id == design.BRIDGE_STATE_ID)
    resolved = launch.resolve_prefix(bridge, "a" * 40)

    assert "{" not in resolved.checkpoint_uri
    assert "/" + "a" * 40 + "/" in resolved.checkpoint_uri
    assert resolved.prefix_replay_code_commit == "a" * 40
    assert resolved.checkpoint_ready_at_design_time is False
    assert resolved.source_aliases_sha256 is None
    assert launch.expected_prefix_core(resolved)["experiment_name"] == resolved.run_spec.source_experiment
