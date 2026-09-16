# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Test whether OOF-equivalent model disagreement warns of heldout failure.

This is an uncertainty diagnostic, not an ensemble or deployment correction.
The equivalent-model sets were frozen from fit-panel OOF performance before
heldout predictions were inspected by ``audit_oof_identification.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import rankdata, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_PREDICTIONS = ARTIFACT_ROOT / "identification_audit/predictions.csv"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "model_disagreement_warning_audit"
OPTIMISM_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    label = np.asarray(label, dtype=bool)
    positives = int(label.sum())
    negatives = int((~label).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = rankdata(np.asarray(score, dtype=float), method="average")
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def row_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    heldout = frame.loc[frame["split"].eq("heldout_policy_matched")].copy()
    outputs: list[pd.DataFrame] = []
    for target, target_rows in heldout.groupby("target", sort=True):
        matrix = target_rows.pivot(index="row_id", columns="model", values="predicted").sort_index()
        observed = target_rows.groupby("row_id", sort=True)["observed"].first().reindex(matrix.index)
        if matrix.isna().any().any():
            raise ValueError(f"Equivalent-model prediction matrix is incomplete for {target}")
        output = pd.DataFrame({"target": target, "row_id": matrix.index, "observed": observed}).reset_index(drop=True)
        predictions = matrix.to_numpy(dtype=float)
        output["n_models"] = predictions.shape[1]
        output["prediction_mean"] = predictions.mean(axis=1)
        output["prediction_min"] = predictions.min(axis=1)
        output["prediction_max"] = predictions.max(axis=1)
        output["prediction_range"] = predictions.max(axis=1) - predictions.min(axis=1)
        output["prediction_std"] = predictions.std(axis=1, ddof=0)
        output["consensus_error"] = output["prediction_mean"] - output["observed"]
        output["consensus_abs_error"] = output["consensus_error"].abs()
        output["consensus_optimism"] = output["observed"] - output["prediction_mean"]
        output["any_model_optimism"] = output["observed"] - output["prediction_min"]
        output["all_model_optimism"] = output["observed"] - output["prediction_max"]
        output["any_model_optimism_gt_0p05"] = output["any_model_optimism"].gt(OPTIMISM_THRESHOLD)
        output["all_models_optimistic_gt_0p05"] = output["all_model_optimism"].gt(OPTIMISM_THRESHOLD)
        output["consensus_optimism_gt_0p05"] = output["consensus_optimism"].gt(OPTIMISM_THRESHOLD)
        metadata = target_rows.drop_duplicates("row_id").set_index("row_id")
        for column in ("panel", "method", "max_epoch", "max_weight", "phase_tv", "aggregate_kl"):
            output[column] = output["row_id"].map(metadata[column])
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for target, group in rows.groupby("target", sort=True):
        disagreement = group["prediction_range"].to_numpy(dtype=float)
        abs_error = group["consensus_abs_error"].to_numpy(dtype=float)
        optimism = group["consensus_optimism"].to_numpy(dtype=float)
        low_cutoff = float(group["prediction_range"].quantile(0.25))
        low_disagreement = group["prediction_range"].le(low_cutoff)
        summary_rows.append(
            {
                "target": target,
                "n_rows": len(group),
                "n_models": int(group["n_models"].iloc[0]),
                "spearman_disagreement_vs_abs_error": float(spearmanr(disagreement, abs_error).statistic),
                "spearman_disagreement_vs_optimism": float(spearmanr(disagreement, optimism).statistic),
                "auc_disagreement_detects_any_model_optimism_gt_0p05": roc_auc(
                    disagreement, group["any_model_optimism_gt_0p05"].to_numpy(dtype=bool)
                ),
                "auc_disagreement_detects_consensus_optimism_gt_0p05": roc_auc(
                    disagreement, group["consensus_optimism_gt_0p05"].to_numpy(dtype=bool)
                ),
                "any_model_optimism_gt_0p05_count": int(group["any_model_optimism_gt_0p05"].sum()),
                "all_models_optimistic_gt_0p05_count": int(group["all_models_optimistic_gt_0p05"].sum()),
                "consensus_optimism_gt_0p05_count": int(group["consensus_optimism_gt_0p05"].sum()),
                "low_disagreement_consensus_failure_count": int(
                    (low_disagreement & group["consensus_optimism_gt_0p05"]).sum()
                ),
                "low_disagreement_all_models_failure_count": int(
                    (low_disagreement & group["all_models_optimistic_gt_0p05"]).sum()
                ),
                "low_disagreement_cutoff": low_cutoff,
                "worst_consensus_optimism": float(group["consensus_optimism"].max()),
            }
        )
        ranked = group.copy()
        ranked["disagreement_bin"] = pd.qcut(
            ranked["prediction_range"].rank(method="first"),
            q=4,
            labels=("Q1 low", "Q2", "Q3", "Q4 high"),
        )
        for bin_name, subset in ranked.groupby("disagreement_bin", observed=True, sort=True):
            bin_rows.append(
                {
                    "target": target,
                    "disagreement_bin": str(bin_name),
                    "n": len(subset),
                    "mean_prediction_range": float(subset["prediction_range"].mean()),
                    "rmse_consensus": float(np.sqrt(np.mean(subset["consensus_error"] ** 2))),
                    "mean_consensus_optimism": float(subset["consensus_optimism"].mean()),
                    "consensus_optimism_gt_0p05_count": int(subset["consensus_optimism_gt_0p05"].sum()),
                    "worst_consensus_optimism": float(subset["consensus_optimism"].max()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def write_artifacts(
    output_dir: Path,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    bins: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "row_level_disagreement.csv", index=False)
    summary.to_csv(output_dir / "disagreement_warning_summary.csv", index=False)
    bins.to_csv(output_dir / "disagreement_bins.csv", index=False)

    figure = px.scatter(
        rows,
        x="prediction_range",
        y="consensus_optimism",
        facet_col="target",
        color="consensus_optimism_gt_0p05",
        hover_name="row_id",
        hover_data=["panel", "method", "observed", "prediction_mean", "max_epoch", "phase_tv"],
        color_discrete_map={False: "#1a9850", True: "#d73027"},
        labels={
            "prediction_range": "Prediction range among OOF-equivalent models (BPB)",
            "consensus_optimism": "Observed minus mean prediction (BPB)",
            "consensus_optimism_gt_0p05": "Consensus optimism > 0.05",
        },
        title="Does model-family disagreement warn of heldout optimism?",
    )
    figure.add_hline(y=OPTIMISM_THRESHOLD, line_dash="dash", line_color="#d73027")
    figure.update_layout(template="plotly_white", width=1450, height=690)
    figure.write_html(
        output_dir / "model_disagreement_warning.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )

    false_reassurance = rows.loc[
        rows["consensus_optimism_gt_0p05"]
        & rows.groupby("target")["prediction_range"].transform(lambda values: values <= values.quantile(0.25))
    ].sort_values("consensus_optimism", ascending=False)
    report = [
        "# OOF-equivalent model-disagreement warning audit",
        "",
        "This is a diagnostic only. Predictions are not averaged for deployment, and no heldout-derived correction is fitted.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Disagreement bins",
        "",
        bins.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Low-disagreement failures",
        "",
        (
            false_reassurance[
                [
                    "target",
                    "row_id",
                    "panel",
                    "observed",
                    "prediction_mean",
                    "prediction_range",
                    "consensus_optimism",
                    "max_epoch",
                    "phase_tv",
                ]
            ].to_markdown(index=False, floatfmt=".5f")
            if len(false_reassurance)
            else "None."
        ),
        "",
        "## Interpretation",
        "",
        "A useful epistemic warning should rank severe optimism above ordinary rows (AUC materially above 0.5) "
        "and should not leave consensus failures in the lowest-disagreement quartile. Failure of either test "
        "means OOF-equivalent mechanistic laws can be confidently wrong together; disagreement cannot rescue "
        "raw surrogate optimization.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.predictions)
    predictions = pd.read_csv(args.predictions)
    rows = row_metrics(predictions)
    summary, bins = summarize(rows)
    write_artifacts(args.output_dir, rows, summary, bins)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
