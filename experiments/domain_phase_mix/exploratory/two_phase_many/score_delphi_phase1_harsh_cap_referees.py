# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Score a frozen harsh-cap branch model after explicitly opening its referee outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_delphi_phase1_harsh_cap_branch_response as fitting,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap_referee_results_20260825" / "branch_results.csv"
DEFAULT_COVERAGE = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap_referee_results_20260825" / "coverage.json"
DEFAULT_CONTRACT = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap_branch_fit_20260825" / "frozen_model_contract.json"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap_referee_score_20260825"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--frozen-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--expected-frozen-contract-sha256", required=True)
    parser.add_argument("--design-weights", type=Path, default=fitting.DEFAULT_DESIGN_WEIGHTS)
    parser.add_argument("--candidate-weights", type=Path, default=fitting.DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_opened_inputs(
    results: pd.DataFrame,
    coverage_path: Path,
    contract_path: Path,
    expected_contract_sha256: str,
    design_weights_path: Path,
    candidate_weights_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    actual_contract_sha256 = file_sha256(contract_path)
    if actual_contract_sha256 != expected_contract_sha256:
        raise ValueError(f"Frozen contract changed: {actual_contract_sha256} != {expected_contract_sha256}")
    contract = json.loads(contract_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    if coverage.get("status") != "complete" or int(coverage.get("missing_rows", -1)) != 0:
        raise ValueError("Opened referee materialization is incomplete")
    if coverage.get("referee_outcomes_opened") is not True:
        raise ValueError("Referee outcomes have not been explicitly opened")
    if len(results) != int(coverage.get("expected_rows", -1)):
        raise ValueError("Opened result row count does not match the frozen panel")
    sealed = cast(dict[str, object], contract.get("seal"))
    if sealed.get("referee_outcomes_present_in_fit_input") is not False:
        raise ValueError("Frozen contract does not certify a referee-free fit input")
    inputs = cast(dict[str, object], contract.get("inputs"))
    if inputs.get("design_weights_sha256") != file_sha256(design_weights_path):
        raise ValueError("Design weights changed after model freeze")
    if inputs.get("candidate_weights_sha256") != file_sha256(candidate_weights_path):
        raise ValueError("Candidate weights changed after model freeze")
    return cast(dict[str, object], contract), cast(dict[str, object], coverage)


def score_candidate(
    results: pd.DataFrame,
    frozen: dict[str, object],
    design_weights_path: Path,
    candidate_weights_path: Path,
    candidate_id: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    referee = results[
        results.prefix_candidate_id.eq(candidate_id) & results.role.eq("sealed_geometry_referee")
    ].sort_values("run_order")
    if len(referee) != 8:
        raise ValueError(f"Expected eight opened referees for {candidate_id}, got {len(referee)}")
    continuation_ids = tuple(referee.continuation_id)
    buckets, weights = fitting.load_weights(design_weights_path, candidate_id, continuation_ids)
    center = fitting.tied_center(candidate_weights_path, candidate_id, buckets)
    coefficients = cast(dict[str, object], frozen["coefficients"])
    baselines = cast(dict[str, object], frozen["baselines"])
    optimum_weights = cast(dict[str, object], frozen["weights"])
    model = fitting.ResponseModel(
        feature_kind=str(frozen["feature_kind"]),
        alpha=float(frozen["ridge_alpha"]),
        coefficients=tuple(float(coefficients[bucket]) for bucket in buckets),
        damage=float(frozen["damage_coefficient"]),
    )
    predicted = fitting.predict(model, weights, center)
    baseline = float(baselines["matched_tied_bpb"])
    observed = referee[fitting.TARGET].to_numpy(dtype=float) - baseline
    residual = predicted - observed
    predicted_optimum = np.asarray([float(optimum_weights[bucket]) for bucket in buckets])
    optimum_key = tuple(fitting.design.common_design.runtime_counts(predicted_optimum).tolist())
    referee_keys = {tuple(fitting.design.common_design.runtime_counts(row).tolist()) for row in weights}
    if optimum_key in referee_keys:
        raise ValueError(f"Frozen optimum for {candidate_id} collides with a sealed referee coordinate")
    scores = referee[["run_order", "run_name", "continuation_id", "prefix_candidate_id", fitting.TARGET]].copy()
    scores = scores.assign(
        predicted_effect_bpb=predicted,
        observed_effect_bpb=observed,
        residual_bpb=residual,
    )
    spearman = None
    if np.ptp(predicted) > 0.0 and np.ptp(observed) > 0.0:
        spearman = float(stats.spearmanr(predicted, observed).statistic)
    summary: dict[str, object] = {
        "candidate_id": candidate_id,
        "referee_rows": len(referee),
        "referee_rmse_bpb": float(np.sqrt(np.mean(residual**2))),
        "referee_mae_bpb": float(np.mean(np.abs(residual))),
        "referee_bias_bpb": float(np.mean(residual)),
        "referee_spearman": spearman,
        "predicted_optimum_excluded_from_referees": True,
    }
    return scores, summary


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    contract, coverage = validate_opened_inputs(
        results,
        args.coverage,
        args.frozen_contract,
        args.expected_frozen_contract_sha256,
        args.design_weights,
        args.candidate_weights,
    )
    frozen_candidates = cast(dict[str, dict[str, object]], contract.get("frozen_candidates"))
    if not frozen_candidates:
        raise ValueError("Frozen contract contains no candidate models")
    score_frames = []
    summaries = []
    for candidate_id, frozen in frozen_candidates.items():
        scores, summary = score_candidate(
            results,
            frozen,
            args.design_weights,
            args.candidate_weights,
            candidate_id,
        )
        score_frames.append(scores)
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(score_frames, ignore_index=True).to_csv(args.output_dir / "referee_scores.csv", index=False)
    report = {
        "contract_version": "delphi_phase1_harsh_cap_referee_score_20260825_v1",
        "frozen_contract_sha256": args.expected_frozen_contract_sha256,
        "opened_coverage_sha256": file_sha256(args.coverage),
        "opened_referee_rows": int(coverage["sealed_referee_rows"]),
        "selection_changed_after_opening": False,
        "candidates": summaries,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
