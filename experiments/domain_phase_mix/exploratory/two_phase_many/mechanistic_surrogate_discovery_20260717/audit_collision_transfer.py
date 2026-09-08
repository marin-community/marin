# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///
"""Cross-swarm transfer audit for a finite-corpus collision mechanism.

The audit freezes each panel's strongest pre-search surrogate. It then
cross-fits one nonnegative amplitude on a dimensionless collision load derived
from mixture weights and realized epochs. Hyperparameters are selected only on
fit-panel grouped OOF predictions; deployment heldouts are read afterward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    audit_phase_information_transfer as phase_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_BASELINES = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/frozen_gate/baseline_metrics.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / ("reference_outputs/mechanistic_surrogate_discovery_20260717/collision_transfer_audit")
BETAS = (0.0, 0.25, 0.5, 1.0)
MECHANISMS = ("aggregate_collision", "within_phase_collision")
PANELS = (
    ("300m", "uncheatable"),
    ("300m", "table9"),
    ("production", "uncheatable"),
    ("delphi_3e18", "uncheatable"),
    ("delphi_3e18", "table9"),
    ("starcoder_cosine", "starcoder_bpb"),
    ("starcoder_wsd80", "starcoder_bpb"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ess_response(load: np.ndarray, beta: float) -> np.ndarray:
    if abs(beta) < 1e-12:
        return np.log1p(load)
    return np.power(1.0 + load, beta)


def collision_load(rows: list[dict[str, Any]], mechanism: str) -> np.ndarray:
    phase0 = np.asarray([row["phase0"] for row in rows], dtype=float)
    phase1 = np.asarray([row["phase1"] for row in rows], dtype=float)
    aggregate = np.asarray([row["aggregate"] for row in rows], dtype=float)
    epochs0 = np.asarray([row["phase0Epochs"] for row in rows], dtype=float)
    epochs1 = np.asarray([row["phase1Epochs"] for row in rows], dtype=float)
    if mechanism == "aggregate_collision":
        return np.sum(aggregate * (epochs0 + epochs1), axis=1)
    if mechanism == "within_phase_collision":
        return np.sum(phase0 * epochs0 + phase1 * epochs1, axis=1)
    raise ValueError(mechanism)


def panel_audit(
    swarm_id: str,
    target: str,
    swarm: dict[str, Any],
    baseline_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = swarm["rows"]
    baseline_prediction = np.asarray(
        swarm["predictions"][target]["two_phase"][baseline_model]["prediction"], dtype=float
    )
    observed = np.asarray(
        [np.nan if row["observed"].get(target) is None else float(row["observed"][target]) for row in rows],
        dtype=float,
    )
    fit_mask = np.asarray(
        [
            row["split"] == "fit" and "two_phase" in row["fitPolicies"] and phase_audit.finite_target(row, target)
            for row in rows
        ],
        dtype=bool,
    )
    if int(fit_mask.sum()) < 10:
        raise ValueError(f"Only {fit_mask.sum()} fit rows for {swarm_id}/{target}")

    reference = phase_audit.baseline_row(swarm, target)
    reference_index = next(index for index, row in enumerate(rows) if row["id"] == reference["id"])
    screen_rows: list[dict[str, Any]] = []
    features: dict[tuple[str, float], np.ndarray] = {}
    amplitudes: dict[tuple[str, float], np.ndarray] = {}
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for mechanism in MECHANISMS:
        load = collision_load(rows, mechanism)
        for beta in BETAS:
            response = ess_response(load, beta)
            feature = response - response[reference_index]
            candidate_oof, fold_amplitudes = phase_audit.secondary_oof(
                observed[fit_mask], baseline_prediction[fit_mask], feature[fit_mask]
            )
            summary, _bins = gate.metrics(observed[fit_mask], candidate_oof)
            key = (mechanism, beta)
            features[key] = feature
            amplitudes[key] = fold_amplitudes
            predictions[key] = candidate_oof
            screen_rows.append(
                {
                    "swarm": swarm_id,
                    "target": target,
                    "baseline_model": baseline_model,
                    "mechanism": mechanism,
                    "beta": beta,
                    "amplitude_median": float(np.median(fold_amplitudes)),
                    "amplitude_mad": float(np.median(np.abs(fold_amplitudes - np.median(fold_amplitudes)))),
                    "positive_fold_fraction": float(np.mean(fold_amplitudes > 1e-10)),
                    **summary,
                }
            )

    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        local = [row for row in screen_rows if row["mechanism"] == mechanism]
        selected = min(local, key=lambda row: (float(row["rmse"]), float(row["regret_at_1"]), row["beta"]))
        beta = float(selected["beta"])
        key = (mechanism, beta)
        feature = features[key]
        amplitude = phase_audit.nonnegative_amplitude(
            feature[fit_mask], observed[fit_mask] - baseline_prediction[fit_mask]
        )
        corrected = baseline_prediction + amplitude * feature
        split_specs: list[tuple[str, np.ndarray, np.ndarray]] = [("fit_secondary_oof", fit_mask, predictions[key])]
        heldout_mask = np.asarray(
            [
                row["split"] == "heldout"
                and not row["isSharedAlias"]
                and row["policyFamily"] == "two_phase"
                and phase_audit.finite_target(row, target)
                for row in rows
            ],
            dtype=bool,
        )
        if int(heldout_mask.sum()) >= 3:
            split_specs.append(("heldout_policy_matched", heldout_mask, corrected[heldout_mask]))
        for split, mask, candidate_prediction in split_specs:
            baseline_values = baseline_prediction[mask]
            for model, values in (("baseline", baseline_values), (mechanism, candidate_prediction)):
                summary, _bins = gate.metrics(observed[mask], values)
                metric_rows.append(
                    {
                        "swarm": swarm_id,
                        "target": target,
                        "split": split,
                        "model": model,
                        "baseline_model": baseline_model,
                        "mechanism": mechanism,
                        "beta": beta,
                        "fit_amplitude": amplitude if model == mechanism else 0.0,
                        **summary,
                    }
                )
        coefficient_rows.extend(
            {
                "swarm": swarm_id,
                "target": target,
                "baseline_model": baseline_model,
                "mechanism": mechanism,
                "beta": beta,
                "fold": fold,
                "amplitude": float(value),
                "full_fit_amplitude": amplitude,
            }
            for fold, value in enumerate(amplitudes[key])
        )
    return metric_rows, screen_rows, coefficient_rows


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.dashboard)
    gate.assert_sealed_absent(args.baselines)
    dashboard = json.loads(args.dashboard.read_text())
    baselines = pd.read_csv(args.baselines)
    metrics: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for swarm_id, target in PANELS:
        baseline_model = phase_audit.strongest_baseline(baselines, swarm_id, target)
        panel_metrics, panel_screens, panel_coefficients = panel_audit(
            swarm_id, target, dashboard["swarms"][swarm_id], baseline_model
        )
        metrics.extend(panel_metrics)
        screens.extend(panel_screens)
        coefficients.extend(panel_coefficients)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame(metrics)
    screen_frame = pd.DataFrame(screens)
    coefficient_frame = pd.DataFrame(coefficients)
    metric_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    screen_frame.to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    coefficient_frame.to_csv(args.output_dir / "coefficient_stability.csv", index=False)

    selected_rows: list[pd.Series] = []
    for (_swarm, _target, mechanism), panel in metric_frame.loc[metric_frame["split"].eq("fit_secondary_oof")].groupby(
        ["swarm", "target", "mechanism"], sort=False
    ):
        baseline = panel.loc[panel["model"].eq("baseline")].iloc[0]
        candidate = panel.loc[panel["model"].eq(mechanism)].iloc[0].copy()
        candidate["relative_rmse"] = candidate["rmse"] / baseline["rmse"]
        candidate["regret_delta"] = candidate["regret_at_1"] - baseline["regret_at_1"]
        selected_rows.append(candidate)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.output_dir / "selected_transfer.csv", index=False)
    (args.output_dir / "report.md").write_text(
        "# Finite-collision cross-swarm transfer audit\n\n"
        "Each panel keeps its strongest frozen pre-search surrogate. Only one nonnegative collision amplitude is added. Hyperparameters and amplitudes are cross-fitted without deployment heldouts.\n\n"
        + selected[
            [
                "swarm",
                "target",
                "mechanism",
                "beta",
                "relative_rmse",
                "regret_delta",
                "fit_amplitude",
            ]
        ].to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )
    print(
        selected[["swarm", "target", "mechanism", "beta", "relative_rmse", "regret_delta", "fit_amplitude"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
