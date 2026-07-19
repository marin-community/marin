# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Describe frozen 3e18 optimism failures through policy/exposure invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    screen_portfolio as portfolio,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
PREDICTIONS = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round3_nested_support/predictions.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/failure_atlas"
SUPPORT_FLOOR = 0.01


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(values[order[np.searchsorted(cumulative, quantile, side="left")]])


def row_features(
    row: dict[str, Any],
    proportional: np.ndarray,
    reference_epochs: np.ndarray,
    family_members: tuple[np.ndarray, ...],
    domains: tuple[str, ...],
) -> dict[str, Any]:
    total_epochs = np.asarray(row["totalEpochs"], dtype=float)
    ratio = total_epochs / np.maximum(reference_epochs, 1e-12)
    safe = ratio + SUPPORT_FLOOR
    normalized = proportional * safe
    normalized /= normalized.sum()
    reverse_kl = float(np.sum(proportional * (np.log(proportional) - np.log(normalized))))
    importance_variance = float(np.sum(proportional * (1.0 / safe - 1.0) ** 2))
    family_ratios = np.asarray(
        [
            np.sum(total_epochs[members] * proportional[members])
            / max(np.sum(reference_epochs[members] * proportional[members]), 1e-12)
            for members in family_members
        ]
    )
    replay = np.maximum(total_epochs + np.expm1(-total_epochs), 0.0)
    top = np.argsort(total_epochs)[-5:][::-1]
    return {
        "phase_tv": float(row["diagnostics"]["phaseTv"]),
        "aggregate_tv_to_proportional": float(row["diagnostics"]["aggregateTvToProportional"]),
        "aggregate_kl_to_proportional": float(row["diagnostics"]["aggregateKlToProportional"]),
        "support_distance": float(row["diagnostics"]["supportDistance"]),
        "max_epoch": float(total_epochs.max()),
        "mean_literal_replay": float(proportional @ replay),
        "max_literal_replay": float(replay.max()),
        "count_ratio_lt_0p1": int(np.sum(ratio < 0.1)),
        "count_ratio_lt_0p25": int(np.sum(ratio < 0.25)),
        "count_ratio_lt_0p5": int(np.sum(ratio < 0.5)),
        "mass_ratio_lt_0p1": float(proportional[ratio < 0.1].sum()),
        "mass_ratio_lt_0p25": float(proportional[ratio < 0.25].sum()),
        "mass_ratio_lt_0p5": float(proportional[ratio < 0.5].sum()),
        "weighted_ratio_q10": weighted_quantile(ratio, proportional, 0.1),
        "weighted_ratio_q25": weighted_quantile(ratio, proportional, 0.25),
        "min_family_ratio": float(family_ratios.min()),
        "family_log_ratio_variance": float(np.var(np.log(family_ratios + SUPPORT_FLOOR))),
        "bucket_reverse_kl": reverse_kl,
        "bucket_importance_variance": importance_variance,
        "top_epoch_buckets": "; ".join(f"{domains[index]}={total_epochs[index]:.2f}" for index in top),
    }


def main() -> None:
    bundle = json.loads(DASHBOARD.read_text())
    swarm = bundle["swarms"]["delphi_3e18"]
    domains = tuple(item["id"] for item in swarm["domains"])
    proportional = np.asarray([item["proportionalWeight"] for item in swarm["domains"]])
    reference_epochs = np.asarray(
        [
            item["proportionalWeight"] * (item["phase0EpochFactor"] + item["phase1EpochFactor"])
            for item in swarm["domains"]
        ]
    )
    family_names, family_members, _group_names, _group_members, _indices = portfolio.hierarchical_partition(domains)
    rows_by_name = {row["name"]: row for row in swarm["rows"] if row["split"] == "heldout" and not row["isSharedAlias"]}
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[
        predictions["split"].eq("heldout_policy_matched")
        & predictions["mechanism"].isin(["baseline", "bucket_reverse_kl"])
    ].copy()

    records: list[dict[str, Any]] = []
    for item in predictions.to_dict("records"):
        row = rows_by_name[str(item["row_id"])]
        records.append(
            {
                **item,
                "optimism": float(item["observed"] - item["predicted"]),
                "panel": row["panel"],
                "method": row["method"],
                "candidate_target": row["candidateTarget"],
                **row_features(
                    row,
                    proportional,
                    reference_epochs,
                    family_members,
                    domains,
                ),
            }
        )
    atlas = pd.DataFrame(records)
    feature_columns = [
        "phase_tv",
        "aggregate_tv_to_proportional",
        "aggregate_kl_to_proportional",
        "support_distance",
        "max_epoch",
        "mean_literal_replay",
        "max_literal_replay",
        "count_ratio_lt_0p1",
        "count_ratio_lt_0p25",
        "count_ratio_lt_0p5",
        "mass_ratio_lt_0p1",
        "mass_ratio_lt_0p25",
        "mass_ratio_lt_0p5",
        "weighted_ratio_q10",
        "weighted_ratio_q25",
        "min_family_ratio",
        "family_log_ratio_variance",
        "bucket_reverse_kl",
        "bucket_importance_variance",
    ]
    correlations: list[dict[str, Any]] = []
    for (dataset, mechanism), group in atlas.groupby(["dataset", "mechanism"]):
        for feature in feature_columns:
            result = spearmanr(group[feature], group["optimism"])
            correlations.append(
                {
                    "dataset": dataset,
                    "mechanism": mechanism,
                    "feature": feature,
                    "spearman_with_optimism": float(result.statistic),
                    "pvalue": float(result.pvalue),
                }
            )

    worst = (
        atlas.sort_values(["dataset", "mechanism", "optimism"], ascending=[True, True, False])
        .groupby(["dataset", "mechanism"], as_index=False)
        .head(10)
    )
    calibration_rows: list[dict[str, Any]] = []
    for (dataset, mechanism), group in atlas.groupby(["dataset", "mechanism"]):
        bins = pd.qcut(group["bucket_reverse_kl"], q=5, duplicates="drop")
        for interval, local in group.groupby(bins, observed=True):
            calibration_rows.append(
                {
                    "dataset": dataset,
                    "mechanism": mechanism,
                    "reverse_kl_bin": str(interval),
                    "n": len(local),
                    "mean_reverse_kl": float(local["bucket_reverse_kl"].mean()),
                    "mean_observed": float(local["observed"].mean()),
                    "mean_predicted": float(local["predicted"].mean()),
                    "mean_optimism": float(local["optimism"].mean()),
                    "optimism_gt_0p05_count": int((local["optimism"] > 0.05).sum()),
                }
            )

    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(DEFAULT_OUTPUT / "heldout_failure_atlas.csv", index=False)
    worst.to_csv(DEFAULT_OUTPUT / "worst_predictions.csv", index=False)
    pd.DataFrame(correlations).to_csv(DEFAULT_OUTPUT / "feature_correlations.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(DEFAULT_OUTPUT / "reverse_kl_binned_calibration.csv", index=False)

    report = [
        "# Frozen 3e18 heldout failure atlas",
        "",
        "This analysis uses only the pre-existing development heldouts. The sealed adversarial panel is absent.",
        "",
        f"Semantic families: {', '.join(family_names)}.",
    ]
    for (dataset, mechanism), group in atlas.groupby(["dataset", "mechanism"]):
        top_corr = (
            pd.DataFrame(correlations)
            .query("dataset == @dataset and mechanism == @mechanism")
            .assign(abs_rho=lambda frame: frame["spearman_with_optimism"].abs())
            .nlargest(5, "abs_rho")
        )
        report.extend(
            [
                "",
                f"## {dataset} / {mechanism}",
                "",
                f"- Heldouts: {len(group)}; optimism > 0.05: {int((group['optimism'] > 0.05).sum())}; worst optimism: {group['optimism'].max():.5f} BPB.",
                "- Strongest exposure-invariant associations with optimism:",
            ]
        )
        report.extend(
            f"  - `{row.feature}`: Spearman {row.spearman_with_optimism:+.3f}" for row in top_corr.itertuples()
        )
    (DEFAULT_OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
