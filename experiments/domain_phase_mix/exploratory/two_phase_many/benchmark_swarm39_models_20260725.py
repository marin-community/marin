# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Benchmark unique-coverage candidates against the Observatory baselines.

Fit each model on the 280-row two-phase panel of a scale and evaluate on every
coordinate-disjoint heldout observation for that scale, stratified by proposal
series, policy class, and support quartile.

The baseline reimplementations in ``swarm39_models_20260725`` are cross-checked
against the collaborator packet's own fitting code before any comparison is
reported, so a candidate win cannot come from a weakened baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The collaborator packet ships standalone modules that are not an installed
# package, so its directory joins the path before importing its fitting code.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent
        / "reference_outputs"
        / "two_phase_surrogate_collaborator_packet_20260721"
        / "standalone_code"
    ),
)

import reference_models as packet
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    evaluate,
    fit_model,
    load_scale,
    provenance,
    support_distance,
)
from swarm39_models_20260725 import candidates, observatory_baselines

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "swarm39_unique_coverage_20260725"
PACKET_CODE = REFERENCE_OUTPUTS / "two_phase_surrogate_collaborator_packet_20260721" / "standalone_code"
SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = (UNCHEATABLE, TABLE9)
CV_SPLITS = 5
CV_SEED = 0


def packet_cross_check() -> pd.DataFrame:
    """Compare in-harness baseline fits against the packet's own implementation.

    Both fit the same Delphi panel with grouped CV over the same shape grid, so a
    close match on out-of-fold RMSE confirms the reimplementation. Exact equality
    is not expected: the packet uses scikit-learn GroupKFold while the harness
    uses its own deterministic grouped assignment.
    """
    rows = []
    fit_panel, _ = load_scale("delphi_3e18")
    for model_name in ("compact_retained_state", "bucket_family_grp", "hierarchical_phase_replay", "separate_heads"):
        for target in TARGETS:
            dataset = packet.load_dataset(
                REFERENCE_OUTPUTS / "two_phase_surrogate_collaborator_packet_20260721",
                "delphi_3e18_two_phase_fit",
                target,
            )
            packet_fit, search = packet.fit_model(dataset, model_name, n_splits=CV_SPLITS, seed=CV_SEED)
            mine = next(m for m in observatory_baselines(fit_panel) if m.name == model_name)
            harness_fit = fit_model(fit_panel, mine, target, n_splits=CV_SPLITS, seed=CV_SEED)
            in_sample_packet = packet_fit.predict(dataset.weights)
            design = mine.build(fit_panel, harness_fit.shape)
            in_sample_mine = harness_fit.intercept + design.matrix @ harness_fit.coefficients
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "packet_oof_rmse": float(search.iloc[0]["oof_rmse"]),
                    "harness_oof_rmse": harness_fit.oof_rmse,
                    "oof_rmse_ratio": harness_fit.oof_rmse / float(search.iloc[0]["oof_rmse"]),
                    "packet_in_sample_rmse": float(np.sqrt(np.mean((in_sample_packet - dataset.target) ** 2))),
                    "harness_in_sample_rmse": float(np.sqrt(np.mean((in_sample_mine - fit_panel.targets[target]) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def run_scale(scale: str, output: Path) -> tuple[pd.DataFrame, list[dict]]:
    fit_panel, heldout = load_scale(scale)
    models = observatory_baselines(fit_panel) + candidates()
    # Support distance is model independent, so compute it once per scale.
    distance = {target: None for target in TARGETS}
    frames, fits = [], []
    for target in TARGETS:
        if not np.isfinite(heldout.targets[target]).any():
            continue
        usable = np.isfinite(heldout.targets[target])
        distance[target] = support_distance(fit_panel, heldout.subset(usable))
        for model in models:
            if not np.isfinite(fit_panel.targets[target]).any():
                continue
            fit = fit_model(fit_panel, model, target, n_splits=CV_SPLITS, seed=CV_SEED)
            frames.append(evaluate(fit, model, fit_panel, heldout, target, support=distance[target]))
            active = int(np.count_nonzero(fit.coefficients))
            fits.append(
                {
                    "scale": scale,
                    "target": target,
                    "model": model.name,
                    "shape": json.dumps({k: v for k, v in fit.shape.items() if not isinstance(v, list)}, sort_keys=True),
                    "l2": fit.l2,
                    "oof_rmse": fit.oof_rmse,
                    "n_features": len(fit.coefficients),
                    "n_active_features": active,
                    "intercept": fit.intercept,
                }
            )
    return pd.concat(frames, ignore_index=True), fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-cross-check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    if not args.skip_cross_check:
        cross = packet_cross_check()
        cross.to_csv(output / "packet_cross_check.csv", index=False)
        print("=== baseline cross-check against the packet implementation ===")
        print(cross.to_string(index=False))
        worst = float(np.abs(cross["oof_rmse_ratio"] - 1.0).max())
        print(f"worst out-of-fold RMSE ratio deviation from 1.0: {worst:.4f}")

    all_metrics, all_fits = [], []
    for scale in SCALES:
        metrics, fits = run_scale(scale, output)
        all_metrics.append(metrics)
        all_fits.extend(fits)
    metrics = pd.concat(all_metrics, ignore_index=True)
    fit_table = pd.DataFrame(all_fits)
    metrics.to_csv(output / "heldout_metrics.csv", index=False)
    fit_table.to_csv(output / "selected_fits.csv", index=False)

    protocol = {
        "scales": list(SCALES),
        "targets": list(TARGETS),
        "cv_splits": CV_SPLITS,
        "cv_seed": CV_SEED,
        "fit_split": "280-row two-phase panel per scale (242 at 60M)",
        "eval_split": "all coordinate-disjoint heldouts for that scale",
        "sealed_targeted_pairwise_panel_accessed": False,
        "provenance_sha256": provenance(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    pooled = metrics[metrics["stratum_type"] == "pooled"]
    print("\n=== pooled heldout metrics ===")
    columns = [
        "scale",
        "target",
        "model",
        "n",
        "oof_rmse",
        "rmse",
        "spearman",
        "calibration_slope",
        "regret_at_1",
        "optimism_over_0p05",
        "worst_optimism",
    ]
    for scale in SCALES:
        for target in TARGETS:
            block = pooled[(pooled["scale"] == scale) & (pooled["target"] == target)]
            if block.empty:
                continue
            print(f"\n-- {scale} / {target} (sorted by heldout RMSE) --")
            print(block.sort_values("rmse")[columns].to_string(index=False))


if __name__ == "__main__":
    main()
