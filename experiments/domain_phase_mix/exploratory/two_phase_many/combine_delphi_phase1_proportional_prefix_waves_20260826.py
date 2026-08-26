# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas==2.2.2"]
# ///
"""Combine sealed proportional-prefix branch waves for the final response fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_WAVE1_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_results_20260826"
DEFAULT_WAVE1_DESIGN = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_20260825"
DEFAULT_WAVE2_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave2_results_20260826"
DEFAULT_WAVE2_DESIGN = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave2_20260826"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_combined_results_20260826"
WAVE1_EXPECTED_ROWS = 102
WAVE1_SEALED_REFEREES = 8
WAVE2_EXPECTED_ROWS = 80
WAVE2_SEALED_REFEREES = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave1-results", type=Path, default=DEFAULT_WAVE1_RESULTS / "branch_results.csv")
    parser.add_argument("--wave1-coverage", type=Path, default=DEFAULT_WAVE1_RESULTS / "coverage.json")
    parser.add_argument("--wave1-summary", type=Path, default=DEFAULT_WAVE1_DESIGN / "continuation_summary.csv")
    parser.add_argument("--wave1-weights", type=Path, default=DEFAULT_WAVE1_DESIGN / "continuation_weights.csv")
    parser.add_argument("--wave2-results", type=Path, default=DEFAULT_WAVE2_RESULTS / "branch_results.csv")
    parser.add_argument("--wave2-coverage", type=Path, default=DEFAULT_WAVE2_RESULTS / "coverage.json")
    parser.add_argument("--wave2-summary", type=Path, default=DEFAULT_WAVE2_DESIGN / "continuation_summary.csv")
    parser.add_argument("--wave2-weights", type=Path, default=DEFAULT_WAVE2_DESIGN / "continuation_weights.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_wave(
    name: str,
    results: pd.DataFrame,
    coverage: dict[str, object],
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    expected_rows: int,
    sealed_referees: int,
) -> None:
    if coverage.get("status") != "complete" or int(cast(int, coverage.get("missing_rows", -1))) != 0:
        raise ValueError(f"{name} materialization is incomplete")
    if coverage.get("referee_outcomes_opened") is not False:
        raise ValueError(f"{name} referee outcomes were opened before the combined fit")
    if int(cast(int, coverage.get("expected_rows", -1))) != expected_rows:
        raise ValueError(f"{name} expected row count changed")
    if int(cast(int, coverage.get("sealed_referee_rows", -1))) != sealed_referees:
        raise ValueError(f"{name} sealed-referee count changed")
    if len(summary) != expected_rows or len(results) != expected_rows - sealed_referees:
        raise ValueError(f"{name} row coverage changed")
    if tuple(weights.columns) != design.WEIGHT_ARTIFACT_COLUMNS:
        raise ValueError(f"{name} weight columns changed")
    visible_ids = set(summary[~summary.role.eq("sealed_geometry_referee")].continuation_id)
    if set(results.continuation_id) != visible_ids:
        raise ValueError(f"{name} visible results do not match the frozen design")
    if set(weights.continuation_id) != set(summary.continuation_id):
        raise ValueError(f"{name} weights do not match the frozen design")


def fit_coordinate_keys(summary: pd.DataFrame, weights: pd.DataFrame) -> set[tuple[int, ...]]:
    fit_ids = set(summary[summary.fit_budget.astype(bool)].continuation_id)
    return {
        tuple(group.phase_1_count.to_numpy(dtype=int))
        for continuation_id, group in weights.groupby("continuation_id", sort=False)
        if continuation_id in fit_ids
    }


def combine(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    wave1_results = pd.read_csv(args.wave1_results)
    wave1_coverage = cast(dict[str, object], json.loads(args.wave1_coverage.read_text()))
    wave1_summary = pd.read_csv(args.wave1_summary)
    wave1_weights = pd.read_csv(args.wave1_weights)
    wave2_results = pd.read_csv(args.wave2_results)
    wave2_coverage = cast(dict[str, object], json.loads(args.wave2_coverage.read_text()))
    wave2_summary = pd.read_csv(args.wave2_summary)
    wave2_weights = pd.read_csv(args.wave2_weights)
    validate_wave(
        "Wave 1",
        wave1_results,
        wave1_coverage,
        wave1_summary,
        wave1_weights,
        expected_rows=WAVE1_EXPECTED_ROWS,
        sealed_referees=WAVE1_SEALED_REFEREES,
    )
    validate_wave(
        "Wave 2",
        wave2_results,
        wave2_coverage,
        wave2_summary,
        wave2_weights,
        expected_rows=WAVE2_EXPECTED_ROWS,
        sealed_referees=WAVE2_SEALED_REFEREES,
    )
    if set(wave1_summary.continuation_id) & set(wave2_summary.continuation_id):
        raise ValueError("Continuation identities overlap across waves")
    wave1_run_ids = set(wave1_results.run_id.astype(int))
    wave2_run_ids = set(wave2_results.run_id.astype(int))
    if len(wave1_run_ids) != len(wave1_results) or len(wave2_run_ids) != len(wave2_results):
        raise ValueError("Run identities repeat within a wave")
    if wave1_run_ids & wave2_run_ids:
        raise ValueError("Run identities overlap across waves")
    if fit_coordinate_keys(wave1_summary, wave1_weights) & fit_coordinate_keys(wave2_summary, wave2_weights):
        raise ValueError("Wave 2 repeats a Wave-1 fit coordinate")

    wave2_results = wave2_results.copy()
    wave2_results["run_order"] = wave2_results.run_order.astype(int) + int(wave1_results.run_order.max()) + 1
    results = pd.concat([wave1_results, wave2_results], ignore_index=True)
    summary = pd.concat([wave1_summary, wave2_summary], ignore_index=True)
    weights = pd.concat([wave1_weights, wave2_weights], ignore_index=True)
    composite_manifest = hashlib.sha256(
        f"{wave1_coverage['manifest_sha256']}:{wave2_coverage['manifest_sha256']}".encode()
    ).hexdigest()
    coverage: dict[str, object] = {
        "contract_version": "delphi_phase1_proportional_prefix_combined_results_20260826_v1",
        "status": "complete",
        "expected_rows": WAVE1_EXPECTED_ROWS + WAVE2_EXPECTED_ROWS,
        "observed_rows": len(results),
        "visible_result_rows": len(results),
        "missing_rows": 0,
        "sealed_referee_rows": WAVE1_SEALED_REFEREES,
        "referee_outcomes_opened": False,
        "manifest_sha256": composite_manifest,
        "wave1_manifest_sha256": wave1_coverage["manifest_sha256"],
        "wave2_manifest_sha256": wave2_coverage["manifest_sha256"],
    }
    return results, summary, weights, coverage


def main() -> None:
    args = parse_args()
    results, summary, weights, coverage = combine(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "branch_results.csv"
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    coverage_path = args.output_dir / "coverage.json"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    report = {
        **coverage,
        "fit_rows": int(summary.fit_budget.sum()),
        "inputs": {
            "wave1_results_sha256": file_sha256(args.wave1_results),
            "wave1_coverage_sha256": file_sha256(args.wave1_coverage),
            "wave1_summary_sha256": file_sha256(args.wave1_summary),
            "wave1_weights_sha256": file_sha256(args.wave1_weights),
            "wave2_results_sha256": file_sha256(args.wave2_results),
            "wave2_coverage_sha256": file_sha256(args.wave2_coverage),
            "wave2_summary_sha256": file_sha256(args.wave2_summary),
            "wave2_weights_sha256": file_sha256(args.wave2_weights),
        },
        "artifacts": {
            results_path.name: file_sha256(results_path),
            summary_path.name: file_sha256(summary_path),
            weights_path.name: file_sha256(weights_path),
            coverage_path.name: file_sha256(coverage_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
