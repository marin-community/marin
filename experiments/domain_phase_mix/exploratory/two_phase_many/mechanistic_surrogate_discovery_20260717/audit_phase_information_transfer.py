# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "tabulate",
# ]
# ///
"""Audit whether phase-label information is a transferable transition debt.

The audit is deliberately stricter than the model screen that discovered the
term.  It freezes each panel's strongest pre-search surrogate and permits only
one nested, nonnegative coefficient multiplying the phase-label mutual
information.  Smoothing is selected by secondary cross-fitting on fit-panel
OOF residuals; Delphi heldouts remain untouched until final evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_BASELINES = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/frozen_gate/baseline_metrics.csv"
)
DEFAULT_OUTPUT = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/phase_information_transfer_audit"
)
SMOOTHING_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
N_SPLITS = 5
SEED = 1707
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite_target(row: dict[str, Any], target: str) -> bool:
    value = row["observed"].get(target)
    return value is not None and np.isfinite(float(value))


def phase_fraction(rows: list[dict[str, Any]]) -> float:
    estimates: list[float] = []
    for row in rows:
        phase0 = np.asarray(row["phase0"], dtype=float)
        phase1 = np.asarray(row["phase1"], dtype=float)
        aggregate = np.asarray(row["aggregate"], dtype=float)
        denominator = phase0 - phase1
        valid = np.abs(denominator) > 1e-8
        if np.any(valid):
            estimates.extend(((aggregate[valid] - phase1[valid]) / denominator[valid]).tolist())
    if not estimates:
        raise ValueError("Cannot infer phase fraction from only phase-tied rows")
    estimate = float(np.median(estimates))
    if not 0.0 < estimate < 1.0:
        raise ValueError(f"Invalid inferred phase fraction {estimate}")
    return estimate


def baseline_row(swarm: dict[str, Any], target: str) -> dict[str, Any]:
    baseline_id = swarm["baselines"][target][0]["id"]
    matches = [row for row in swarm["rows"] if row["id"] == baseline_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one baseline row {baseline_id!r}; found {len(matches)}")
    return matches[0]


def smoothed_distribution(values: np.ndarray, prior: np.ndarray, smoothing: float) -> np.ndarray:
    normalized = np.maximum(np.asarray(values, dtype=float), 0.0)
    normalized /= np.maximum(normalized.sum(axis=1, keepdims=True), 1e-12)
    return (normalized + smoothing * prior[None, :]) / (1.0 + smoothing)


def phase_information(
    phase0: np.ndarray,
    phase1: np.ndarray,
    prior: np.ndarray,
    gamma0: float,
    smoothing: float,
) -> np.ndarray:
    q0 = smoothed_distribution(phase0, prior, smoothing)
    q1 = smoothed_distribution(phase1, prior, smoothing)
    gamma1 = 1.0 - gamma0
    mixture = gamma0 * q0 + gamma1 * q1

    def row_kl(left: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(left)
        positive = left > 0.0
        terms[positive] = left[positive] * (np.log(left[positive]) - np.log(mixture[positive]))
        return np.sum(terms, axis=1)

    kl0 = row_kl(q0)
    kl1 = row_kl(q1)
    return gamma0 * kl0 + gamma1 * kl1


def nonnegative_amplitude(feature: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(feature @ feature)
    if denominator <= 1e-14:
        return 0.0
    return max(0.0, float(feature @ residual) / denominator)


def secondary_oof(
    observed: np.ndarray,
    baseline_prediction: np.ndarray,
    feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full(len(observed), np.nan, dtype=float)
    amplitudes: list[float] = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for train, test in splitter.split(np.arange(len(observed))):
        amplitude = nonnegative_amplitude(feature[train], observed[train] - baseline_prediction[train])
        prediction[test] = baseline_prediction[test] + amplitude * feature[test]
        amplitudes.append(amplitude)
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete secondary OOF prediction")
    return prediction, np.asarray(amplitudes, dtype=float)


def strongest_baseline(metrics: pd.DataFrame, swarm: str, target: str) -> str:
    candidates = metrics.loc[
        metrics["swarm"].eq(swarm)
        & metrics["target"].eq(target)
        & metrics["policy"].eq("two_phase")
        & metrics["split"].eq("fit_oof")
    ].copy()
    candidates["parameter_count_numeric"] = pd.to_numeric(candidates["parameter_count"], errors="coerce")
    candidates = candidates.loc[candidates["parameter_count_numeric"].notna()]
    if candidates.empty:
        raise ValueError(f"No dashboard baseline for {swarm}/{target}")
    return str(candidates.sort_values(["rmse", "parameter_count_numeric", "model"]).iloc[0]["model"])


def panel_audit(
    swarm_id: str,
    target: str,
    swarm: dict[str, Any],
    baseline_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = swarm["rows"]
    prediction_bundle = swarm["predictions"][target]["two_phase"][baseline_model]
    baseline_prediction = np.asarray(prediction_bundle["prediction"], dtype=float)
    phase0 = np.asarray([row["phase0"] for row in rows], dtype=float)
    phase1 = np.asarray([row["phase1"] for row in rows], dtype=float)
    observed = np.asarray(
        [np.nan if row["observed"].get(target) is None else float(row["observed"][target]) for row in rows],
        dtype=float,
    )
    fit_mask = np.asarray(
        [row["split"] == "fit" and "two_phase" in row["fitPolicies"] and finite_target(row, target) for row in rows],
        dtype=bool,
    )
    if int(fit_mask.sum()) < 10:
        raise ValueError(f"Only {fit_mask.sum()} fit rows for {swarm_id}/{target}")
    prior = np.asarray(baseline_row(swarm, target)["aggregate"], dtype=float)
    prior /= prior.sum()
    gamma0 = phase_fraction(rows)

    screen_rows: list[dict[str, Any]] = []
    features: dict[float, np.ndarray] = {}
    amplitudes_by_smoothing: dict[float, np.ndarray] = {}
    predictions: dict[float, np.ndarray] = {}
    for smoothing in SMOOTHING_GRID:
        feature = phase_information(phase0, phase1, prior, gamma0, smoothing)
        candidate_oof, amplitudes = secondary_oof(observed[fit_mask], baseline_prediction[fit_mask], feature[fit_mask])
        summary, _bins = gate.metrics(observed[fit_mask], candidate_oof)
        screen_rows.append(
            {
                "swarm": swarm_id,
                "target": target,
                "baseline_model": baseline_model,
                "smoothing": smoothing,
                "amplitude_median": float(np.median(amplitudes)),
                "amplitude_mad": float(np.median(np.abs(amplitudes - np.median(amplitudes)))),
                "positive_fold_fraction": float(np.mean(amplitudes > 1e-10)),
                **summary,
            }
        )
        features[smoothing] = feature
        amplitudes_by_smoothing[smoothing] = amplitudes
        predictions[smoothing] = candidate_oof

    local_screen = [row for row in screen_rows if row["swarm"] == swarm_id and row["target"] == target]
    selected = min(local_screen, key=lambda row: (float(row["rmse"]), float(row["regret_at_1"]), row["smoothing"]))
    smoothing = float(selected["smoothing"])
    feature = features[smoothing]
    fit_amplitude = nonnegative_amplitude(feature[fit_mask], observed[fit_mask] - baseline_prediction[fit_mask])
    corrected = baseline_prediction + fit_amplitude * feature
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    split_specs = [("fit_secondary_oof", fit_mask, predictions[smoothing])]
    heldout_mask = np.asarray(
        [
            row["split"] == "heldout"
            and not row["isSharedAlias"]
            and row["policyFamily"] == "two_phase"
            and finite_target(row, target)
            for row in rows
        ],
        dtype=bool,
    )
    if int(heldout_mask.sum()) >= 3:
        split_specs.append(("heldout_policy_matched", heldout_mask, corrected[heldout_mask]))
    for split, mask, candidate_prediction in split_specs:
        local_observed = observed[mask]
        local_baseline = baseline_prediction[mask]
        if split == "fit_secondary_oof":
            local_candidate = candidate_prediction
        else:
            local_candidate = candidate_prediction
        for model, values in (("baseline", local_baseline), ("phase_information", local_candidate)):
            summary, bins = gate.metrics(local_observed, values)
            metric_rows.append(
                {
                    "swarm": swarm_id,
                    "target": target,
                    "split": split,
                    "model": model,
                    "baseline_model": baseline_model,
                    "smoothing": smoothing,
                    "fit_amplitude": fit_amplitude if model == "phase_information" else 0.0,
                    **summary,
                }
            )
            bin_rows.extend(
                {
                    "swarm": swarm_id,
                    "target": target,
                    "split": split,
                    "model": model,
                    **record,
                }
                for record in bins
            )
    coefficient_rows = [
        {
            "swarm": swarm_id,
            "target": target,
            "baseline_model": baseline_model,
            "smoothing": smoothing,
            "fold": fold,
            "amplitude": float(amplitude),
            "full_fit_amplitude": fit_amplitude,
            "gamma0": gamma0,
        }
        for fold, amplitude in enumerate(amplitudes_by_smoothing[smoothing])
    ]
    return metric_rows, screen_rows, coefficient_rows + bin_rows


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.dashboard)
    gate.assert_sealed_absent(args.baselines)
    dashboard = json.loads(args.dashboard.read_text())
    baselines = pd.read_csv(args.baselines)
    metrics: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    panels = (
        ("300m", "uncheatable"),
        ("300m", "table9"),
        ("production", "uncheatable"),
        ("delphi_3e18", "uncheatable"),
        ("delphi_3e18", "table9"),
        ("starcoder_cosine", "starcoder_bpb"),
        ("starcoder_wsd80", "starcoder_bpb"),
    )
    for swarm_id, target in panels:
        baseline_model = strongest_baseline(baselines, swarm_id, target)
        panel_metrics, panel_screens, panel_records = panel_audit(
            swarm_id, target, dashboard["swarms"][swarm_id], baseline_model
        )
        metrics.extend(panel_metrics)
        screens.extend(panel_screens)
        records.extend(panel_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame(metrics)
    screen_frame = pd.DataFrame(screens)
    record_frame = pd.DataFrame(records)
    metric_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    screen_frame.to_csv(args.output_dir / "smoothing_screen.csv", index=False)
    record_frame.to_csv(args.output_dir / "coefficient_and_calibration_records.csv", index=False)

    heldout = metric_frame.loc[metric_frame["split"].eq("heldout_policy_matched")].copy()
    if not heldout.empty:
        figure = px.scatter(
            heldout,
            x="calibration_slope_observed_on_predicted",
            y="rmse",
            color="model",
            facet_col="target",
            hover_data=["swarm", "baseline_model", "optimism_gt_0p05_count", "worst_optimism"],
            title="Phase-information transfer: frozen 3e18 heldout calibration",
            color_discrete_map={"baseline": "#607d8b", "phase_information": "#d95f02"},
        )
        figure.add_vline(x=1.0, line_dash="dash", line_color="#263238")
        figure.write_html(args.output_dir / "heldout_calibration.html", config=PLOT_CONFIG, include_plotlyjs="cdn")

    selected = (
        screen_frame.sort_values(["swarm", "target", "rmse", "regret_at_1"])
        .groupby(["swarm", "target"], as_index=False)
        .first()
    )
    report = [
        "# Phase-information transfer audit",
        "",
        "The test freezes each panel's strongest pre-search surrogate and fits only a nonnegative amplitude on the phase-label Jensen-Shannon information. Smoothing is selected by secondary cross-fitting on fit-panel OOF residuals. This is a transfer falsification, not a replacement full-model fit.",
        "",
        selected[
            [
                "swarm",
                "target",
                "baseline_model",
                "smoothing",
                "amplitude_median",
                "amplitude_mad",
                "positive_fold_fraction",
                "rmse",
                "regret_at_1",
            ]
        ].to_markdown(index=False),
        "",
        "## Frozen 3e18 heldouts",
        "",
        heldout[
            [
                "swarm",
                "target",
                "model",
                "rmse",
                "regret_at_1",
                "calibration_slope_observed_on_predicted",
                "optimism_gt_0p05_count",
                "worst_optimism",
            ]
        ].to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
