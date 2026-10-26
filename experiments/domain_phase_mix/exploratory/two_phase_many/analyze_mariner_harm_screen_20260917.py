# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired diagnostics for the archived harm screen; does not select fit parameters."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)


def analyze(report: Path) -> None:
    metrics = pd.read_csv(report / "component_metrics.csv")
    keys = ["panel", "target", "component", "repeat"]
    baseline = metrics[metrics.model.eq("hs17_mariner")].set_index(keys)
    metrics = metrics.join(baseline[["spearman", "rmse"]].add_prefix("baseline_"), on=keys)
    metrics["delta_spearman"] = metrics.spearman - metrics.baseline_spearman
    metrics["rmse_ratio"] = metrics.rmse / metrics.baseline_rmse
    metrics.to_csv(report / "paired_component_metrics.csv", index=False)
    # First balance the two objective groups within each of the three panels;
    # six anchors are not six equally weighted objective measurements.
    balanced = metrics.groupby(["model", "panel", "target"])[["delta_spearman", "rmse_ratio"]].mean()
    summary = balanced.groupby("model").mean()
    summary["component_wins"] = metrics.groupby("model").delta_spearman.apply(lambda values: int((values > 0.001).sum()))
    summary["component_losses"] = metrics.groupby("model").delta_spearman.apply(
        lambda values: int((values < -0.001).sum())
    )
    summary = summary.sort_values("delta_spearman", ascending=False)
    summary.to_csv(report / "screen_ranking.csv")
    balanced.to_csv(report / "screen_by_panel_objective.csv")
    predictions = pd.read_csv(report / "component_predictions.csv")
    high_rows = []
    for (model, panel_name, target, component, repeat, fold), group in predictions.groupby(
        ["model", "panel", "target", "component", "repeat", "fold"]
    ):
        panel = bench.load_panel(panel_name)
        split = next(
            item for item in bench.panel_splits(panel, int(repeat) + 1) if item.repeat == repeat and item.fold == fold
        )
        severity = np.max(panel.features.exposures, axis=1)
        cutoff = float(np.quantile(severity[split.train], 0.8))
        selected = group[severity[group.row_index.to_numpy(int)] >= cutoff]
        residual = selected.prediction - selected.observed
        high_rows.append(
            {
                "model": model,
                "panel": panel_name,
                "target": target,
                "component": component,
                "repeat": repeat,
                "fold": fold,
                "training_exposure_p80": cutoff,
                "n": len(selected),
                "bias": float(residual.mean()),
                "rmse": float(np.sqrt((residual**2).mean())),
            }
        )
    pd.DataFrame(high_rows).to_csv(report / "high_repetition_diagnostics.csv", index=False)
    print(summary.round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    if args.raw:
        analyze_raw(args.report)
    else:
        analyze(args.report)


def analyze_raw(report: Path) -> None:
    metrics = pd.read_csv(report / "component_metrics.csv")
    folds = pd.read_csv(report / "component_fold_metrics.csv")
    rows = []
    contrast_tables = []
    for harm in ("squared_log", "unsquared_log", "convex_linear", "bounded_rational_c1"):
        inside_id, outside_id = f"hs17_log_deficit_{harm}", f"hs17_raw_loss_{harm}"
        keys = ["panel", "target", "component", "repeat"]
        inside = metrics[metrics.model.eq(inside_id)].set_index(keys)
        outside = metrics[metrics.model.eq(outside_id)].set_index(keys)
        if not inside.index.equals(outside.index):
            raise ValueError(f"Placement coverage differs for {harm}")
        paired = outside[["spearman", "rmse", "regret_at_1"]].join(
            inside[["spearman", "rmse", "regret_at_1"]].add_prefix("inside_")
        )
        paired["delta_spearman"] = paired.spearman - paired.inside_spearman
        paired["rmse_ratio"] = paired.rmse / paired.inside_rmse
        paired["delta_pooled_regret"] = paired.regret_at_1 - paired.inside_regret_at_1
        paired["harm"] = harm
        contrast_tables.append(paired.reset_index())
        balanced = (
            paired.groupby(["panel", "target"])[["delta_spearman", "rmse_ratio", "delta_pooled_regret"]].mean().mean()
        )
        fold_keys = [*keys, "fold"]
        inside_fold = folds[folds.model.eq(inside_id)].set_index(fold_keys)
        outside_fold = folds[folds.model.eq(outside_id)].set_index(fold_keys)
        regret_delta = outside_fold.regret_at_1 - inside_fold.regret_at_1
        rows.append(
            {
                "harm": harm,
                **balanced.to_dict(),
                "delta_mean_fold_regret": float(regret_delta.groupby(["panel", "target"]).mean().mean()),
                "component_rank_wins": int((paired.delta_spearman > 0.001).sum()),
                "component_rank_losses": int((paired.delta_spearman < -0.001).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(report / "placement_summary.csv", index=False)
    pd.concat(contrast_tables, ignore_index=True).to_csv(report / "placement_contrasts.csv", index=False)
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
