# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Compare frozen baseline forms inside and outside empirical convex support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "support_stratified_baselines"
DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
SUPPORT = ARTIFACT_ROOT / "convex_support_audit/heldout_convex_support.csv"
EXTERNAL_SOURCES = (
    (
        "early_family_asymmetric",
        RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/predictions.csv",
        "inverse_power_deficit_early_family_asymmetric_surplus",
    ),
    (
        "inverse_deficit_log_link",
        RESEARCH_DIR / "reference_outputs/model_improvement_round2_conditioned_replay_link_20260716/predictions.csv",
        "inverse_power_deficit_conditioned_replay",
    ),
)
OPTIMISM_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dashboard_predictions(bundle: dict[str, Any], support: pd.DataFrame) -> pd.DataFrame:
    swarm = bundle["swarms"]["delphi_3e18"]
    rows = swarm["rows"]
    output: list[dict[str, object]] = []
    for target in ("uncheatable", "table9"):
        target_support = support.loc[support["dataset"].eq(f"delphi_3e18_{target}")]
        support_names = set(target_support["row_id"].astype(str))
        for model_name, model in swarm["predictions"][target]["two_phase"].items():
            prediction = np.asarray(model["prediction"], dtype=float)
            if len(prediction) != len(rows):
                raise ValueError(f"Prediction length mismatch for {target}/{model_name}")
            for row, predicted in zip(rows, prediction, strict=True):
                name = str(row["name"])
                if name not in support_names:
                    continue
                observed = row["observed"].get(target)
                if observed is None or not np.isfinite(predicted):
                    continue
                output.append(
                    {
                        "dataset": f"delphi_3e18_{target}",
                        "model": model_name,
                        "row_id": name,
                        "observed": float(observed),
                        "predicted": float(predicted),
                    }
                )
    return pd.DataFrame(output)


def external_predictions(support: pd.DataFrame) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for model_name, path, variant in EXTERNAL_SOURCES:
        gate.assert_sealed_absent(path)
        source = pd.read_csv(path)
        selected = source.loc[source["deficit_variant"].eq(variant) & source["split"].eq("heldout")].copy()
        if model_name == "early_family_asymmetric":
            selected = selected.loc[
                (selected["dataset"].str.contains("uncheatable") & selected["link"].eq("identity_raw_bpb"))
                | (selected["dataset"].str.contains("table9") & selected["link"].eq("log_reducible_bpb"))
            ]
        else:
            selected = selected.loc[selected["link"].eq("log_reducible_bpb")]
        selected["model"] = model_name
        selected = selected.rename(columns={"dataset": "source_dataset"})
        selected["dataset"] = selected["source_dataset"]
        selected = selected[["dataset", "model", "row_id", "observed", "predicted"]]
        selected = selected.merge(
            support[["dataset", "row_id"]],
            on=["dataset", "row_id"],
            how="inner",
            validate="one_to_one",
        )
        output.append(selected)
    return pd.concat(output, ignore_index=True)


def metric_row(dataset: str, model: str, region: str, frame: pd.DataFrame) -> dict[str, object]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    residual = predicted - observed
    centered = predicted - predicted.mean()
    slope = float(centered @ (observed - observed.mean()) / max(centered @ centered, 1e-12))
    return {
        "dataset": dataset,
        "model": model,
        "support_region": region,
        "count": len(frame),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "bias_predicted_minus_observed": float(residual.mean()),
        "observed_on_predicted_slope": slope,
        "optimism_gt_0p05_count": int((-residual > OPTIMISM_THRESHOLD).sum()),
        "worst_optimism": float((-residual).max()),
    }


def summarize(predictions: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        support[["dataset", "row_id", "outside_fit_loo_p95"]],
        on=["dataset", "row_id"],
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (dataset, model), model_frame in merged.groupby(["dataset", "model"], sort=False):
        for outside, region_frame in model_frame.groupby("outside_fit_loo_p95", sort=True):
            rows.append(metric_row(dataset, model, "outside" if outside else "inside", region_frame))
        rows.append(metric_row(dataset, model, "all", model_frame))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    for path in (DASHBOARD, SUPPORT):
        gate.assert_sealed_absent(path)
    bundle = json.loads(DASHBOARD.read_text())
    support = pd.read_csv(SUPPORT)
    predictions = pd.concat(
        [dashboard_predictions(bundle, support), external_predictions(support)],
        ignore_index=True,
    )
    metrics = summarize(predictions, support)
    metrics["rank_within_region"] = metrics.groupby(["dataset", "support_region"])["rmse"].rank(method="min")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "baseline_heldout_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "support_stratified_baseline_metrics.csv", index=False)
    outside = metrics.loc[metrics["support_region"].eq("outside")].sort_values(["dataset", "rmse"])
    inside = metrics.loc[metrics["support_region"].eq("inside")].sort_values(["dataset", "rmse"])
    report = [
        "# Frozen baseline forms by convex-support region",
        "",
        "The support partition is fixed by the strongest baseline's mechanistic design and is not redefined per "
        "model. No response calibration is fitted here.",
        "",
        "## Outside empirical support",
        "",
        outside.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Inside empirical support",
        "",
        inside.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(outside.to_string(index=False))


if __name__ == "__main__":
    main()
