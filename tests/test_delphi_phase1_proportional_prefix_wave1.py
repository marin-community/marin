# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as runtime
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_proportional_prefix_wave1 as launch
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design_base,
)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_proportional_wave1_contract() -> None:
    manifest = json.loads(launch.DEFAULT_DESIGN_MANIFEST.read_text())
    summary = pd.read_csv(launch.DEFAULT_CONTINUATION_SUMMARY)
    weights = pd.read_csv(launch.DEFAULT_CONTINUATION_WEIGHTS)

    assert manifest["contract_version"] == "delphi_phase1_proportional_prefix_wave1_20260825_v2"
    assert manifest["selected_candidate_ids"] == [launch.TARGET_PREFIX]
    assert manifest["rows"] == {
        "controls_per_prefix": 14,
        "fit_per_prefix": 80,
        "sealed_referees_per_prefix": 8,
        "total": 102,
    }
    assert manifest["diagnostics"][launch.TARGET_PREFIX]["direct_feature_rank"] == 38
    assert manifest["diagnostics"][launch.TARGET_PREFIX]["sqrt_feature_rank"] == 39
    assert len(summary) == 102
    assert int(summary.fit_budget.sum()) == 80
    assert tuple(weights.columns) == design_base.WEIGHT_ARTIFACT_COLUMNS
    assert float(weights.total_materialized_epochs.max()) <= design_base.TOTAL_MATERIALIZED_EPOCH_CAP
    for _, rows in weights.groupby("continuation_id", sort=False):
        counts = rows.phase_1_count.to_numpy(dtype=int)
        assert counts.sum() == design_base.MIXTURE_BLOCK_SIZE
        assert np.array_equal(rows.phase_1_weight.to_numpy(dtype=float), counts / design_base.MIXTURE_BLOCK_SIZE)

    assert manifest["artifacts"]["continuation_summary.csv"] == sha256(launch.DEFAULT_CONTINUATION_SUMMARY)
    assert manifest["artifacts"]["continuation_weights.csv"] == sha256(launch.DEFAULT_CONTINUATION_WEIGHTS)

    frontier_id = manifest["diagnostics"][launch.TARGET_PREFIX]["validated_frontier_exact_continuation_id"]
    frontier = weights[weights.continuation_id.eq(frontier_id)].set_index("bucket")
    contract = json.loads((launch.DEFAULT_DESIGN_DIR / "validated_frontier_contract.json").read_text())
    assert frontier.phase_1_count.to_dict() == contract["runtime_counts"]
    assert summary.role.value_counts().to_dict() == manifest["role_counts_per_prefix"]
    assert len(summary[summary.role.eq("validated_frontier_transfer_repeat")]) == 4
    assert len(summary[summary.role.eq("fresh_tied_control")]) == 4
    assert len(summary[summary.role.eq("prefix_state_tied_control")]) == 5


def test_runtime_loads_all_frozen_wave1_rows() -> None:
    rows = runtime.load_design(
        launch.DEFAULT_CONTINUATION_SUMMARY,
        sha256(launch.DEFAULT_CONTINUATION_SUMMARY),
        launch.DEFAULT_CONTINUATION_WEIGHTS,
        sha256(launch.DEFAULT_CONTINUATION_WEIGHTS),
        launch.DEFAULT_DESIGN_MANIFEST,
        sha256(launch.DEFAULT_DESIGN_MANIFEST),
        (launch.TARGET_PREFIX,),
    )

    assert len(rows) == 102
    assert len({str(row["continuation_id"]) for row in rows}) == 102
    assert sum(bool(row["fit_budget"]) for row in rows) == 80


def test_wave2_contract_is_sealed_to_wave1_artifacts() -> None:
    contract_path = (
        launch.REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave2_contract_20260826" / "contract.json"
    )
    contract = json.loads(contract_path.read_text())
    provenance = contract["provenance"]

    assert contract["sealed_before_wave1_outcomes"] is True
    assert provenance == {
        "selected_prefixes_sha256": sha256(launch.DEFAULT_SELECTED_PREFIXES),
        "validated_frontier_contract_sha256": sha256(launch.DEFAULT_DESIGN_DIR / "validated_frontier_contract.json"),
        "wave1_continuation_summary_sha256": sha256(launch.DEFAULT_CONTINUATION_SUMMARY),
        "wave1_continuation_weights_sha256": sha256(launch.DEFAULT_CONTINUATION_WEIGHTS),
        "wave1_design_manifest_sha256": sha256(launch.DEFAULT_DESIGN_MANIFEST),
    }


def test_proportional_prefix_manifest_freezes_two_wave1_states() -> None:
    payload = json.loads(launch.DEFAULT_SELECTED_PREFIXES.read_text())

    assert payload["candidate_weights_sha256"] == sha256(launch.DEFAULT_CANDIDATE_WEIGHTS)
    assert [row["canonical_candidate_id"] for row in payload["selected_aliases"]] == [launch.TARGET_PREFIX]
    assert {(row["candidate_id"], row["repeat_seed"]) for row in payload["prefixes"]} == {
        (launch.TARGET_PREFIX, 0),
        (launch.TARGET_PREFIX, 1),
    }
    assert payload["source_prefix_hardware"] == {
        "evidence": "successful executor records pinned in each prefix row",
        "tensor_parallel_size": 1,
        "tpu_region": "us-east5",
        "tpu_type": "v5p-8",
        "tpu_zone": "us-east5-a",
    }
    assert all(row["executor_info_sha256"] for row in payload["prefixes"])


def test_proportional_prefix_hardware_is_not_conflated_with_continuation_hardware() -> None:
    assert launch.PREFIX_HARDWARE == runtime.TpuHardware("v5p-8", "us-east5", "us-east5-a")
    assert launch.CONTINUATION_HARDWARE == runtime.TpuHardware("v6e-8", "us-east5", "us-east5-b")
    assert runtime.panel_hardware_status(launch.PREFIX_HARDWARE, launch.CONTINUATION_HARDWARE) == (
        "cross_hardware_v5p-8_prefix_to_v6e-8_continuation"
    )
