# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as runtime
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_proportional_prefix_wave1 as launch
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    combine_delphi_phase1_proportional_prefix_waves_20260826 as combine_waves,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design_base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_proportional_prefix_wave2_20260826 as wave2,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_delphi_phase1_harsh_cap_branch_response as response,
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

    assert contract["initial_contract_sealed_before_wave1_outcomes"] is True
    assert contract["sealed_before_wave1_outcomes"] is False
    assert contract["corrective_amendment"]["visible_wave1_rows_when_amended"] == 13
    assert contract["corrective_amendment"]["outcomes_used"] is False
    assert contract["eligibility"]["fit_data_seed"] == design_base.FIT_DATA_SEED
    assert provenance == {
        "selected_prefixes_sha256": sha256(launch.DEFAULT_SELECTED_PREFIXES),
        "validated_frontier_contract_sha256": sha256(launch.DEFAULT_DESIGN_DIR / "validated_frontier_contract.json"),
        "wave1_continuation_summary_sha256": sha256(launch.DEFAULT_CONTINUATION_SUMMARY),
        "wave1_continuation_weights_sha256": sha256(launch.DEFAULT_CONTINUATION_WEIGHTS),
        "wave1_design_manifest_sha256": sha256(launch.DEFAULT_DESIGN_MANIFEST),
    }


def test_one_standard_error_selection_prefers_simpler_eligible_model() -> None:
    metrics = pd.DataFrame(
        [
            {
                "feature_kind": "sqrt",
                "alpha": 1e-2,
                "rmse_bpb": 0.010,
                "fold_rmse_se_bpb": 0.003,
                "gain_sign_reversals": 0,
            },
            {
                "feature_kind": "direct",
                "alpha": 1.0,
                "rmse_bpb": 0.012,
                "fold_rmse_se_bpb": 0.002,
                "gain_sign_reversals": 1,
            },
            {
                "feature_kind": "direct",
                "alpha": 100.0,
                "rmse_bpb": 0.013,
                "fold_rmse_se_bpb": 0.002,
                "gain_sign_reversals": 2,
            },
        ]
    )

    assert response.selected_parameter(metrics) == ("direct", 1.0)
    assert list(response.eligible_parameters(metrics).feature_kind) == ["direct", "sqrt"]


def test_wave2_disagreement_uses_fold_model_spread() -> None:
    fold_predictions = np.zeros((3, 14))
    fold_predictions[:, 2] = (-0.2, 0.2, -0.1)
    fold_predictions[:, 13] = (-1.0, 1.0, 0.0)

    selected = wave2.select_disagreement(fold_predictions, {13})

    assert selected[0] == 2
    assert 13 not in selected


def test_wave2_exploitation_caps_multiple_model_settings() -> None:
    weights = np.column_stack([np.linspace(0.05, 0.95, 30), np.linspace(0.95, 0.05, 30)])
    center = np.asarray([0.5, 0.5])
    models = [response.ResponseModel("direct", alpha, (1.0, -1.0), 0.1) for alpha in (1e-6, 1e-4, 1e-2)]

    selected = wave2.select_exploitation(weights, center, models, (0.0, 0.1, 1.0, 2.5, 5.0, 10.0, 25.0), set())

    assert len(selected) == wave2.EXPLOIT_ROWS
    assert len(set(selected)) == wave2.EXPLOIT_ROWS


def test_model_selection_fails_closed_when_gain_sign_gate_rejects_everything() -> None:
    metrics = pd.DataFrame(
        [
            {
                "feature_kind": "direct",
                "alpha": 1.0,
                "rmse_bpb": 0.010,
                "fold_rmse_se_bpb": 0.001,
                "gain_sign_reversals": 2,
            },
            {
                "feature_kind": "sqrt",
                "alpha": 1.0,
                "rmse_bpb": 0.011,
                "fold_rmse_se_bpb": 0.001,
                "gain_sign_reversals": 3,
            },
        ]
    )

    with np.testing.assert_raises_regex(ValueError, "gain-sign gate"):
        response.eligible_parameters(metrics)


def test_wave2_design_is_full_rank_and_excludes_wave1(tmp_path: Path, monkeypatch) -> None:
    summary = pd.read_csv(launch.DEFAULT_CONTINUATION_SUMMARY)
    weights = pd.read_csv(launch.DEFAULT_CONTINUATION_WEIGHTS)
    visible = summary[~summary.role.eq("sealed_geometry_referee")].copy()
    center_rows = pd.read_csv(launch.DEFAULT_CANDIDATE_WEIGHTS)
    center = center_rows[center_rows.candidate_id.eq(launch.TARGET_PREFIX)].phase_0_weight.to_numpy(dtype=float)
    beta = np.linspace(-0.02, 0.02, len(center))
    outcomes = []
    for run_order, row in visible.iterrows():
        branch = weights[weights.continuation_id.eq(row.continuation_id)].phase_1_weight.to_numpy(dtype=float)
        effect = float((branch - center) @ beta + 0.02 * response.hellinger(branch[None, :], center)[0] ** 2)
        outcomes.append(
            {
                **row.to_dict(),
                "run_order": run_order,
                "run_id": launch.BRANCH_RUN_ID_BASE + int(run_order),
                response.TARGET: 1.0 + effect if bool(row.fit_budget) else 1.0,
            }
        )
    results_path = tmp_path / "branch_results.csv"
    pd.DataFrame(outcomes).to_csv(results_path, index=False)
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_rows": 102,
                "sealed_referee_rows": 8,
                "missing_rows": 0,
                "referee_outcomes_opened": False,
                "manifest_sha256": "wave1",
            }
        )
    )
    fit_rows = pd.DataFrame(outcomes)
    fit_rows = fit_rows[fit_rows.fit_budget.astype(bool)].sort_values("run_order")
    continuation_ids = tuple(fit_rows.continuation_id)
    _, measured = response.load_weights(
        launch.DEFAULT_CONTINUATION_WEIGHTS,
        launch.TARGET_PREFIX,
        continuation_ids,
    )
    effects = fit_rows[response.TARGET].to_numpy(dtype=float) - 1.0
    metrics = response.parameter_cv(
        measured,
        effects,
        center,
        response.geometric_fold_ids(measured, center, response.OUTER_FOLDS, response.CV_SEED),
    )
    feature_kind, alpha = response.selected_parameter(metrics)
    model_contract_path = tmp_path / "model_contract.json"
    model_contract_path.write_text(
        json.dumps(
            {
                "inputs": {
                    "results_sha256": wave2.file_sha256(results_path),
                    "coverage_sha256": wave2.file_sha256(coverage_path),
                    "design_summary_sha256": sha256(launch.DEFAULT_CONTINUATION_SUMMARY),
                    "design_weights_sha256": sha256(launch.DEFAULT_CONTINUATION_WEIGHTS),
                    "candidate_weights_sha256": sha256(launch.DEFAULT_CANDIDATE_WEIGHTS),
                },
                "seal": {"referee_outcomes_present_in_fit_input": False},
                "frozen_candidates": {launch.TARGET_PREFIX: {"feature_kind": feature_kind, "ridge_alpha": alpha}},
            }
        )
    )
    panel = design_base.common_design.load_canonical_panel_geometry()
    monkeypatch.setattr(
        design_base,
        "anchor_mixtures",
        lambda buckets, proportional: {
            "proportional": proportional,
            "uniform": np.full(len(buckets), 1.0 / len(buckets)),
        },
    )
    generated_summary, generated_weights, manifest = wave2.build_design(
        Namespace(
            results=results_path,
            coverage=coverage_path,
            model_contract=model_contract_path,
            wave1_summary=launch.DEFAULT_CONTINUATION_SUMMARY,
            wave1_weights=launch.DEFAULT_CONTINUATION_WEIGHTS,
            wave1_manifest=launch.DEFAULT_DESIGN_MANIFEST,
            validated_frontier_contract=launch.DEFAULT_DESIGN_DIR / "validated_frontier_contract.json",
            candidate_weights=launch.DEFAULT_CANDIDATE_WEIGHTS,
            selected_prefixes=launch.DEFAULT_SELECTED_PREFIXES,
            wave2_contract=wave2.DEFAULT_WAVE2_CONTRACT,
            output_dir=tmp_path / "wave2",
        )
    )

    assert len(generated_summary) == 80
    assert generated_summary.role.value_counts().to_dict() == {
        "adaptive_model_fit": 40,
        "outcome_blind_coverage_fit": 40,
    }
    assert generated_summary.data_seed.unique().tolist() == [design_base.FIT_DATA_SEED]
    assert generated_summary.prefix_repeat_seed.unique().tolist() == [0]
    assert int(generated_summary.source.str.startswith("adaptive_exploit:").sum()) == wave2.EXPLOIT_ROWS
    assert int(generated_summary.source.str.startswith("adaptive_local:").sum()) == wave2.LOCAL_REFINEMENT_ROWS
    assert int(generated_summary.source.str.startswith("adaptive_disagreement:").sum()) == wave2.DISAGREEMENT_ROWS
    assert generated_summary.continuation_id.nunique() == 80
    assert generated_weights.groupby("continuation_id").phase_1_count.sum().eq(design_base.MIXTURE_BLOCK_SIZE).all()
    diagnostics = cast(dict[str, dict[str, object]], manifest["diagnostics"])[launch.TARGET_PREFIX]
    assert diagnostics["direct_feature_rank"] == len(panel.buckets) - 1
    assert diagnostics["sqrt_feature_rank"] == len(panel.buckets)

    generated_summary_path = tmp_path / "wave2_summary.csv"
    generated_weights_path = tmp_path / "wave2_weights.csv"
    generated_manifest_path = tmp_path / "wave2_manifest.json"
    generated_summary.to_csv(generated_summary_path, index=False)
    generated_weights.loc[:, list(design_base.WEIGHT_ARTIFACT_COLUMNS)].to_csv(generated_weights_path, index=False)
    generated_manifest_path.write_text(json.dumps(manifest))
    loaded_rows = runtime.load_design(
        generated_summary_path,
        sha256(generated_summary_path),
        generated_weights_path,
        sha256(generated_weights_path),
        generated_manifest_path,
        sha256(generated_manifest_path),
        (launch.TARGET_PREFIX,),
        expected_fit_rows_per_prefix=80,
        expected_referee_rows_per_prefix=0,
    )
    assert len(loaded_rows) == 80

    wave2_results_path = tmp_path / "wave2_results.csv"
    wave2_coverage_path = tmp_path / "wave2_coverage.json"
    wave2_summary_path = tmp_path / "wave2_summary_for_combine.csv"
    wave2_weights_path = tmp_path / "wave2_weights_for_combine.csv"
    wave2_results = generated_summary.copy()
    wave2_results["run_order"] = np.arange(len(wave2_results))
    wave2_results["run_id"] = 977_000 + wave2_results.run_order
    wave2_effects = []
    for continuation_id in wave2_results.continuation_id:
        branch = generated_weights[generated_weights.continuation_id.eq(continuation_id)].phase_1_weight.to_numpy(
            dtype=float
        )
        wave2_effects.append(
            float((branch - center) @ beta + 0.02 * response.hellinger(branch[None, :], center)[0] ** 2)
        )
    wave2_results[response.TARGET] = 1.0 + np.asarray(wave2_effects)
    wave2_results.to_csv(wave2_results_path, index=False)
    wave2_coverage_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_rows": 80,
                "sealed_referee_rows": 0,
                "missing_rows": 0,
                "referee_outcomes_opened": False,
                "manifest_sha256": "wave2",
            }
        )
    )
    generated_summary.to_csv(wave2_summary_path, index=False)
    generated_weights.loc[:, list(design_base.WEIGHT_ARTIFACT_COLUMNS)].to_csv(wave2_weights_path, index=False)
    combined_results, combined_summary, combined_weights, combined_coverage = combine_waves.combine(
        Namespace(
            wave1_results=results_path,
            wave1_coverage=coverage_path,
            wave1_summary=launch.DEFAULT_CONTINUATION_SUMMARY,
            wave1_weights=launch.DEFAULT_CONTINUATION_WEIGHTS,
            wave2_results=wave2_results_path,
            wave2_coverage=wave2_coverage_path,
            wave2_summary=wave2_summary_path,
            wave2_weights=wave2_weights_path,
        )
    )
    assert len(combined_results) == 174
    assert len(combined_summary) == 182
    assert int(combined_summary.fit_budget.sum()) == 160
    assert combined_coverage["sealed_referee_rows"] == 8
    assert combined_weights.continuation_id.nunique() == 182

    combined_results_path = tmp_path / "combined_results.csv"
    combined_summary_path = tmp_path / "combined_summary.csv"
    combined_weights_path = tmp_path / "combined_weights.csv"
    combined_results.to_csv(combined_results_path, index=False)
    combined_summary.to_csv(combined_summary_path, index=False)
    combined_weights.to_csv(combined_weights_path, index=False)
    candidate, fit_artifacts = response.fit_candidate(
        combined_results,
        combined_summary_path,
        combined_weights_path,
        launch.DEFAULT_CANDIDATE_WEIGHTS,
        launch.TARGET_PREFIX,
        0.9798883332146539,
        160,
        8,
    )
    assert candidate["candidate_id"] == launch.TARGET_PREFIX
    assert len(fit_artifacts["nested_predictions"]) == 160


def test_control_baselines_accepts_five_paired_prefix_states() -> None:
    results = pd.DataFrame(
        [
            {"role": "common_random_tied_control", "prefix_repeat_seed": 0, response.TARGET: 1.00},
            *[
                {"role": "fresh_tied_control", "prefix_repeat_seed": 0, response.TARGET: 1.00 + index / 1000}
                for index in range(4)
            ],
            *[
                {
                    "role": "prefix_state_tied_control",
                    "prefix_repeat_seed": 1,
                    response.TARGET: 1.01 + index / 1000,
                }
                for index in range(5)
            ],
        ]
    )

    baselines = response.control_baselines(results)

    assert baselines["matched_tied_bpb"] == 1.0
    assert baselines["expected_tied_bpb"] == 1.0012
    assert baselines["stability_tied_bpb"] == 1.012


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
