# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
# ruff: noqa: E501

"""Audit partition, coordinate, prediction, and frozen-metric provenance."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)

DEFAULT_DASHBOARD = SCRIPT_DIR.parent / "mixture_fit_debugger" / "src" / "generated" / "dashboard_data.json"
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs" / "mechanistic_surrogate_discovery_20260717" / "provenance_audit"
)
FROZEN_BASELINES = (
    SCRIPT_DIR.parent
    / "reference_outputs"
    / "mechanistic_surrogate_discovery_20260717"
    / "frozen_gate"
    / "baseline_metrics.csv"
)
FROZEN_GATE = (
    SCRIPT_DIR.parent
    / "reference_outputs"
    / "mechanistic_surrogate_discovery_20260717"
    / "frozen_gate"
    / "acceptance_gate.json"
)
KEY_COLUMNS = ("swarm", "target", "policy", "model", "split")
NUMERIC_METRICS = (
    "n",
    "rmse",
    "mae",
    "spearman",
    "bias_predicted_minus_observed",
    "calibration_slope_observed_on_predicted",
    "calibration_intercept_observed_on_predicted",
    "regret_at_1",
    "regret_at_3",
    "regret_at_5",
    "lower_tail_optimism",
    "low_tail_rmse",
    "optimism_gt_0p05_count",
    "worst_optimism",
    "selected_optimism",
    "selected_observed",
    "selected_predicted",
)


def coordinate_fingerprint(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(row["phase0"] + row["phase1"], dtype=float), 12))


def heldout_overlap_kind(heldout_row: dict[str, Any], fit_rows: list[dict[str, Any]]) -> str:
    if heldout_row["isSharedAlias"]:
        return "shared_alias"
    heldout_name = str(heldout_row["name"])
    if heldout_name.startswith("singleavg_"):
        paired_name = heldout_name.removeprefix("singleavg_")
        if any(str(row["name"]) == paired_name for row in fit_rows):
            return "paired_policy_class"
    if str(heldout_row["id"]).startswith("noise_reference:"):
        if any(str(row["name"]) == "baseline_proportional" for row in fit_rows):
            return "independent_repeat"
    return "unexpected"


def audit_swarm(swarm_id: str, swarm: dict[str, Any]) -> dict[str, object]:
    rows = swarm["rows"]
    domains = swarm["domains"]
    phase_fractions = np.asarray(swarm["dataset"]["phaseFractions"], dtype=float)
    if len({row["id"] for row in rows}) != len(rows):
        raise AssertionError(f"Duplicate row id in {swarm_id}")
    if len({row["name"] for row in rows}) != len(rows):
        raise AssertionError(f"Duplicate row name in {swarm_id}")
    if not np.isclose(phase_fractions.sum(), 1.0, atol=1e-12):
        raise AssertionError(f"Phase fractions do not sum to one in {swarm_id}")

    maximum_weight_sum_error = 0.0
    maximum_aggregate_error = 0.0
    for row in rows:
        phase0 = np.asarray(row["phase0"], dtype=float)
        phase1 = np.asarray(row["phase1"], dtype=float)
        aggregate = np.asarray(row["aggregate"], dtype=float)
        if len(phase0) != len(domains) or len(phase1) != len(domains):
            raise AssertionError(f"Policy/domain dimension mismatch in {swarm_id}/{row['name']}")
        maximum_weight_sum_error = max(
            maximum_weight_sum_error,
            abs(float(phase0.sum()) - 1.0),
            abs(float(phase1.sum()) - 1.0),
        )
        expected_aggregate = phase_fractions[0] * phase0 + phase_fractions[1] * phase1
        maximum_aggregate_error = max(maximum_aggregate_error, float(np.max(np.abs(aggregate - expected_aggregate))))
    if maximum_weight_sum_error > 1e-9 or maximum_aggregate_error > 1e-9:
        raise AssertionError(f"Policy normalization failed in {swarm_id}")

    prediction_arrays = 0
    for target, policies in swarm["predictions"].items():
        for policy, models in policies.items():
            for model, values in models.items():
                for field in ("prediction", "fullFitPrediction"):
                    predictions = np.asarray(values[field], dtype=float)
                    if len(predictions) != len(rows):
                        raise AssertionError(f"Prediction length mismatch: {swarm_id}/{target}/{policy}/{model}/{field}")
                prediction_arrays += 2

    fit_coordinates: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    heldout_coordinates: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    repeat_coordinates: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "fit":
            fit_coordinates[coordinate_fingerprint(row)].append(row)
        elif row["split"] == "heldout":
            heldout_coordinates[coordinate_fingerprint(row)].append(row)
        elif row["split"] == "noise_reference":
            repeat_coordinates[coordinate_fingerprint(row)].append(row)
    overlaps = set(fit_coordinates) & set(heldout_coordinates)
    overlap_rows = [
        (coordinate, row, heldout_overlap_kind(row, fit_coordinates[coordinate]))
        for coordinate in overlaps
        for row in heldout_coordinates[coordinate]
    ]
    unexpected_overlap_rows = [(coordinate, row) for coordinate, row, kind in overlap_rows if kind == "unexpected"]
    if unexpected_overlap_rows:
        unexpected_names = sorted(str(row["name"]) for _coordinate, row in unexpected_overlap_rows)
        raise AssertionError(f"Unexpected fit/heldout coordinate overlaps in {swarm_id}: {unexpected_names}")
    repeat_overlaps = set(fit_coordinates) & set(repeat_coordinates)
    unexpected_repeat_rows = [
        row
        for coordinate in repeat_overlaps
        for row in repeat_coordinates[coordinate]
        if heldout_overlap_kind(row, fit_coordinates[coordinate]) != "independent_repeat"
    ]
    if unexpected_repeat_rows:
        unexpected_names = sorted(str(row["name"]) for row in unexpected_repeat_rows)
        raise AssertionError(f"Unexpected fit/repeat coordinate overlaps in {swarm_id}: {unexpected_names}")

    declared_fit = int(swarm["dataset"]["fitDesignCount"])
    actual_fit = sum(row["split"] == "fit" for row in rows)
    if actual_fit != declared_fit:
        raise AssertionError(f"Fit-design count mismatch in {swarm_id}: {actual_fit} != {declared_fit}")
    declared_heldout = int(swarm["dataset"]["heldoutCount"])
    total_heldout = sum(row["split"] == "heldout" for row in rows)
    actual_heldout = sum(row["split"] == "heldout" and not row["isSharedAlias"] for row in rows)
    if declared_heldout == actual_heldout:
        heldout_count_convention = "excludes_shared_aliases"
    elif declared_heldout == total_heldout:
        heldout_count_convention = "includes_shared_aliases"
    else:
        raise AssertionError(
            f"Heldout count mismatch in {swarm_id}: declared={declared_heldout}, "
            f"all={total_heldout}, scored={actual_heldout}"
        )

    return {
        "swarm": swarm_id,
        "rows": len(rows),
        "fit_rows": actual_fit,
        "heldout_rows_excluding_aliases": actual_heldout,
        "declared_heldout_count_convention": heldout_count_convention,
        "shared_alias_rows": sum(bool(row["isSharedAlias"]) for row in rows),
        "fit_heldout_coordinate_overlaps": len(overlaps),
        "fit_repeat_coordinate_overlaps": len(repeat_overlaps),
        "shared_alias_overlap_rows": sum(kind == "shared_alias" for _coordinate, _row, kind in overlap_rows),
        "paired_policy_class_overlap_rows": sum(
            kind == "paired_policy_class" for _coordinate, _row, kind in overlap_rows
        ),
        "independent_repeat_overlap_rows": sum(len(repeat_coordinates[coordinate]) for coordinate in repeat_overlaps),
        "unexpected_overlap_rows": len(unexpected_overlap_rows),
        "prediction_arrays_checked": prediction_arrays,
        "maximum_weight_sum_error": maximum_weight_sum_error,
        "maximum_aggregate_error": maximum_aggregate_error,
    }


def compare_frozen_metrics(bundle: dict[str, Any]) -> pd.DataFrame:
    recomputed_rows, _bins = gate.dashboard_metric_rows(bundle)
    recomputed = pd.DataFrame(recomputed_rows)
    frozen = pd.read_csv(FROZEN_BASELINES)
    frozen = frozen.loc[frozen["source"].eq("dashboard")].copy()
    joined = frozen.merge(recomputed, on=list(KEY_COLUMNS), suffixes=("_frozen", "_recomputed"), validate="one_to_one")
    if len(joined) != len(frozen) or len(joined) != len(recomputed):
        raise AssertionError("Frozen and recomputed dashboard metric keys differ")
    rows: list[dict[str, object]] = []
    for metric in NUMERIC_METRICS:
        frozen_values = pd.to_numeric(joined[f"{metric}_frozen"], errors="coerce").to_numpy(dtype=float)
        recomputed_values = pd.to_numeric(joined[f"{metric}_recomputed"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(frozen_values) & np.isfinite(recomputed_values)
        maximum_difference = (
            float(np.max(np.abs(frozen_values[finite] - recomputed_values[finite]))) if finite.any() else 0.0
        )
        if maximum_difference > 1e-12 or not np.array_equal(np.isfinite(frozen_values), np.isfinite(recomputed_values)):
            raise AssertionError(f"Frozen metric recomputation mismatch for {metric}: {maximum_difference}")
        rows.append(
            {"metric": metric, "rows_compared": int(finite.sum()), "maximum_absolute_difference": maximum_difference}
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    gate.assert_sealed_absent(args.dashboard)

    frozen_gate = json.loads(FROZEN_GATE.read_text())
    expected_dashboard_hash = frozen_gate["dashboard_sha256"]
    if gate.sha256(args.dashboard) != expected_dashboard_hash:
        raise AssertionError("Dashboard changed after the acceptance gate was frozen")
    bundle = json.loads(args.dashboard.read_text())

    source_paths: list[Path] = []
    for swarm in bundle["swarms"].values():
        source_paths.extend(REPO_ROOT / source for source in swarm["provenance"].get("sources", ()))
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise AssertionError(f"Dashboard provenance sources are missing: {missing_sources}")

    swarm_summary = pd.DataFrame([audit_swarm(swarm_id, swarm) for swarm_id, swarm in bundle["swarms"].items()])
    metric_differences = compare_frozen_metrics(bundle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    swarm_summary.to_csv(args.output_dir / "swarm_provenance_summary.csv", index=False)
    metric_differences.to_csv(args.output_dir / "frozen_metric_recomputation.csv", index=False)
    report = [
        "# Data-provenance and partition audit",
        "",
        f"Dashboard SHA-256 matches the frozen gate: `{expected_dashboard_hash}`.",
        "",
        "## Swarm and policy invariants",
        "",
        swarm_summary.to_markdown(index=False, floatfmt=".3e"),
        "",
        "## Frozen metric recomputation",
        "",
        metric_differences.to_markdown(index=False, floatfmt=".3e"),
        "",
        "Every dashboard-derived frozen baseline metric recomputes exactly to floating-point tolerance. Exact-coordinate "
        "overlaps are exhausted by explicit shared aliases, independently trained proportional repeats, or deliberate "
        "row-matched one-phase counterparts of phase-tied two-phase policies. No unexpected overlap remains. Shared aliases "
        "are excluded from heldout scoring; independent repeats and policy-class counterparts remain valid observations. "
        "This rules out stale exports, row-count drift, unmarked exact-policy leakage, or metric-code drift as explanations "
        "for the negative result.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(swarm_summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
