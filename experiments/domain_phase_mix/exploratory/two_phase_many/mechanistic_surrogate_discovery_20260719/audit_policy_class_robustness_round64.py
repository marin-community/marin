# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "plotly>=6.0", "tabulate>=0.9"]
# ///
"""Audit frozen baseline behavior separately by heldout policy class."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
FROZEN_METRICS = OUTPUT_ROOT / "frozen_gate/baseline_metrics.csv"
ROUND_DIR = OUTPUT_ROOT / "round64_policy_class_robustness"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def segment_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    parts = []
    fixed_segments = {
        "historical_one_phase": ("historical", "one_phase", "historical_352__single_phase"),
        "historical_two_phase": ("historical", "two_phase", "historical_352__two_phase"),
        "new_matched_one_phase": (
            "new_matched_one_phase",
            "one_phase",
            "matched_one_phase_238__single_phase",
        ),
    }
    for segment, (panel, policy_class, split) in fixed_segments.items():
        frame = metrics.loc[metrics["split"].eq(split)].copy()
        frame["segment"] = segment
        frame["source_panel"] = panel
        frame["heldout_policy_class"] = policy_class
        parts.append(frame)

    for target in ("uncheatable", "table9"):
        for policy_class, split_suffix in (
            ("one_phase", "single_phase"),
            ("two_phase", "two_phase"),
        ):
            split = f"adversarial_candidate_{target}__{split_suffix}"
            frame = metrics.loc[metrics["split"].eq(split) & metrics["target"].eq(target)].copy()
            frame["segment"] = f"adversarial_target_matched_{policy_class}"
            frame["source_panel"] = "adversarial_target_matched"
            frame["heldout_policy_class"] = policy_class
            parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    result["calibration_error"] = (result["calibration_slope_observed_on_predicted"].astype(float) - 1.0).abs()
    return result


def winner_rows(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (segment, target), group in segments.groupby(["segment", "target"]):
        if group.empty:
            continue
        rmse = group.loc[group["rmse"].astype(float).idxmin()]
        regret = group.loc[group["regret_at_1"].astype(float).idxmin()]
        calibration = group.loc[group["calibration_error"].astype(float).idxmin()]
        rows.append(
            {
                "segment": segment,
                "target": target,
                "n": int(group["n"].max()),
                "model_count": len(group),
                "best_rmse_model": rmse["model"],
                "best_rmse": float(rmse["rmse"]),
                "best_regret_model": regret["model"],
                "best_regret_at_1": float(regret["regret_at_1"]),
                "best_calibration_model": calibration["model"],
                "best_calibration_slope": float(calibration["calibration_slope_observed_on_predicted"]),
            }
        )
    return pd.DataFrame(rows)


def rank_transfer(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel in ("historical", "adversarial_target_matched"):
        frame = segments.loc[segments["source_panel"].eq(panel)]
        for target, group in frame.groupby("target"):
            for metric in ("rmse", "regret_at_1", "calibration_error"):
                pivot = group.pivot_table(
                    index="model",
                    columns="heldout_policy_class",
                    values=metric,
                    aggfunc="first",
                ).dropna(subset=["one_phase", "two_phase"])
                one_rank = pivot["one_phase"].rank(method="average")
                two_rank = pivot["two_phase"].rank(method="average")
                rows.append(
                    {
                        "source_panel": panel,
                        "target": target,
                        "metric": metric,
                        "common_models": len(pivot),
                        "one_vs_two_model_rank_spearman": float(one_rank.corr(two_rank)),
                        "one_phase_best_model": pivot["one_phase"].idxmin(),
                        "two_phase_best_model": pivot["two_phase"].idxmin(),
                        "same_best_model": pivot["one_phase"].idxmin() == pivot["two_phase"].idxmin(),
                    }
                )
    return pd.DataFrame(rows)


def write_plot(segments: pd.DataFrame) -> None:
    paired = (
        segments.loc[segments["source_panel"].isin(["historical", "adversarial_target_matched"])]
        .pivot_table(
            index=["source_panel", "target", "model"],
            columns="heldout_policy_class",
            values="rmse",
            aggfunc="first",
        )
        .dropna(subset=["one_phase", "two_phase"])
        .reset_index()
    )
    paired.columns.name = None
    figure = px.scatter(
        paired,
        x="one_phase",
        y="two_phase",
        color="model",
        facet_row="source_panel",
        facet_col="target",
        hover_name="model",
        title="Frozen baseline RMSE does not transfer uniformly between one-phase and two-phase heldouts",
        labels={"one_phase": "One-phase heldout RMSE", "two_phase": "Two-phase heldout RMSE"},
    )
    bound = 1.05 * float(paired[["one_phase", "two_phase"]].to_numpy().max())
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=bound,
        y1=bound,
        line_dash="dash",
        line_color="#59636c",
        row="all",
        col="all",
    )
    figure.update_xaxes(range=[0, bound])
    figure.update_yaxes(range=[0, bound])
    figure.update_layout(template="plotly_white", width=1200, height=850)
    figure.write_html(
        ROUND_DIR / "policy_class_rmse_transfer.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(FROZEN_METRICS)
    metrics = metrics.loc[metrics["swarm"].eq("delphi_3e18") & metrics["policy"].eq("two_phase")].copy()
    segments = segment_rows(metrics)
    winners = winner_rows(segments)
    transfer = rank_transfer(segments)
    segments.to_csv(ROUND_DIR / "policy_class_metrics.csv", index=False)
    winners.to_csv(ROUND_DIR / "policy_class_winners.csv", index=False)
    transfer.to_csv(ROUND_DIR / "policy_class_rank_transfer.csv", index=False)
    write_plot(segments)

    historical_rmse = transfer.loc[transfer["source_panel"].eq("historical") & transfer["metric"].eq("rmse")]
    adversarial_rmse = transfer.loc[
        transfer["source_panel"].eq("adversarial_target_matched") & transfer["metric"].eq("rmse")
    ]
    report = "\n".join(
        [
            "# Round 64: heldout policy-class robustness",
            "",
            "This is a frozen-baseline diagnostic. It fits no model and selects no hyperparameter. Historical and exposed target-matched adversarial heldouts are split into one-phase and two-phase policy classes before metrics are compared.",
            "",
            "## Diagnostic winners",
            "",
            winners.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Model-rank transfer",
            "",
            transfer.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Conclusion",
            "",
            f"Historical one-phase versus two-phase RMSE rank correlations span {historical_rmse['one_vs_two_model_rank_spearman'].min():.3f}--{historical_rmse['one_vs_two_model_rank_spearman'].max():.3f}; exposed target-matched correlations span {adversarial_rmse['one_vs_two_model_rank_spearman'].min():.3f}--{adversarial_rmse['one_vs_two_model_rank_spearman'].max():.3f}.",
            "The best-RMSE, best-calibration, and best-selection models need not agree across policy classes. Pooled archive metrics therefore cannot establish a policy-class-robust incumbent, and the algebraically tied restriction must remain distinct from an independently fitted one-phase law.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
