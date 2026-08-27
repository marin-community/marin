# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as runtime
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_cross_prefix_graft_20260826 as graft,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_graft_panel_crosses_identical_actions_over_prefix_states() -> None:
    manifest = json.loads((graft.DEFAULT_OUTPUT_DIR / "manifest.json").read_text())
    assert manifest["outcome_informed"] is True
    assert manifest["total_rows"] == 54
    assert manifest["prefix_seeds"] == [0, 1, 2]
    assert manifest["data_seeds"] == [973_000, 973_001, 973_002]
    assert "tied" not in manifest["analysis"]["primary_estimand"]
    assert manifest["analysis"]["additive_null"] == "Y(prefix, action) = alpha_prefix + beta_action"

    action_counts: dict[str, dict[str, tuple[int, ...]]] = {}
    for prefix_id in graft.PREFIX_IDS:
        prefix_dir = graft.DEFAULT_OUTPUT_DIR / prefix_id
        prefix_manifest = json.loads((prefix_dir / "manifest.json").read_text())
        summary = pd.read_csv(prefix_dir / "continuation_summary.csv")
        weights = pd.read_csv(prefix_dir / "continuation_weights.csv")

        assert prefix_manifest["rows"] == {
            "controls_per_prefix": 27,
            "fit_per_prefix": 0,
            "sealed_referees_per_prefix": 0,
            "total": 27,
        }
        assert prefix_manifest["analysis_contract"] == manifest["analysis"]
        assert summary.role.value_counts().to_dict() == {
            "paired_tied_control": 9,
            "imported_frontier_graft": 9,
            "novel_model_graft": 9,
        }
        assert set(summary.prefix_repeat_seed) == {0, 1, 2}
        assert set(summary.data_seed) == {973_000, 973_001, 973_002}
        assert float(weights.total_materialized_epochs.max()) <= 10.0
        assert len(weights) == 27 * 39
        for _, rows in weights.groupby("continuation_id", sort=False):
            counts = rows.phase_1_count.to_numpy(dtype=int)
            assert counts.sum() == graft.MIXTURE_BLOCK_SIZE
            assert np.array_equal(rows.phase_1_weight.to_numpy(dtype=float), counts / graft.MIXTURE_BLOCK_SIZE)

        action_counts[prefix_id] = {}
        for action in ("fit079", "novel", "tied"):
            continuation = f"cross_{action}_seed0_data0"
            rows = weights[weights.continuation_id.eq(continuation)]
            action_counts[prefix_id][action] = tuple(rows.phase_1_count.to_numpy(dtype=int))

    proportional = action_counts[graft.PROPORTIONAL_PREFIX]
    cap4 = action_counts[graft.CAP4_PREFIX]
    assert proportional["fit079"] == cap4["fit079"]
    assert proportional["novel"] == cap4["novel"]
    assert proportional["tied"] != cap4["tied"]


def test_runtime_loads_cross_prefix_control_designs() -> None:
    for prefix_id in graft.PREFIX_IDS:
        prefix_dir = graft.DEFAULT_OUTPUT_DIR / prefix_id
        summary = prefix_dir / "continuation_summary.csv"
        weights = prefix_dir / "continuation_weights.csv"
        manifest = prefix_dir / "manifest.json"
        rows = runtime.load_design(
            summary,
            sha256(summary),
            weights,
            sha256(weights),
            manifest,
            sha256(manifest),
            (prefix_id,),
            expected_fit_rows_per_prefix=0,
            expected_referee_rows_per_prefix=0,
        )

        assert len(rows) == 27
        assert len({str(row["continuation_id"]) for row in rows}) == 27
        assert not any(bool(row["fit_budget"]) for row in rows)


def test_selected_prefix_loader_accepts_explicit_three_seed_contract(tmp_path: Path) -> None:
    payload = {
        "candidate_weights_sha256": "candidate-hash",
        "prefix_replay_code_commit": "prefix-commit",
        "selected_aliases": [{"canonical_candidate_id": graft.CAP4_PREFIX}],
        "prefixes": [
            {
                "canonical_candidate_id": graft.CAP4_PREFIX,
                "repeat_seed": seed,
                "checkpoint_uri": f"gs://marin-us-east5/prefix-seed-{seed}/checkpoints/step-2399",
                "provenance_sha256": f"provenance-{seed}",
            }
            for seed in graft.PREFIX_SEEDS
        ],
    }
    path = tmp_path / "selected.json"
    encoded = (json.dumps(payload) + "\n").encode()
    path.write_bytes(encoded)

    rows, loaded = runtime.selected_prefixes(
        str(path),
        hashlib.sha256(encoded).hexdigest(),
        "candidate-hash",
        "prefix-commit",
        graft.PREFIX_SEEDS,
    )

    assert loaded == payload
    assert tuple(row.repeat_seed for row in rows) == graft.PREFIX_SEEDS
