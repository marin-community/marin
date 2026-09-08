# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "plotly>=6.0", "tabulate>=0.9"]
# ///
"""Summarize worst-stratum behavior on the exposed adversarial panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round62_adversarial_strata_robustness"
STRATA = OUTPUT_ROOT / "frozen_gate" / "adversarial_strata_metrics.csv"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    strata = pd.read_csv(STRATA)
    expected_types = {"origin", "policy_class", "proposal_models", "selection_stratum"}
    if set(strata["stratum_type"]) != expected_types:
        raise ValueError("Adversarial stratum taxonomy has drifted")

    rows = []
    for (target, model, stratum_type), group in strata.groupby(["target", "model", "stratum_type"], sort=True):
        worst_rmse = group.loc[group["rmse"].idxmax()]
        worst_slope = group.loc[group["calibration_slope_observed_on_predicted"].idxmin()]
        worst_rank = group.loc[group["spearman"].idxmin()]
        worst_regret = group.loc[group["regret_at_1"].idxmax()]
        rows.append(
            {
                "target": target,
                "model": model,
                "stratum_type": stratum_type,
                "stratum_count": group["stratum_value"].nunique(),
                "worst_rmse": float(worst_rmse["rmse"]),
                "worst_rmse_stratum": worst_rmse["stratum_value"],
                "minimum_calibration_slope": float(worst_slope["calibration_slope_observed_on_predicted"]),
                "minimum_slope_stratum": worst_slope["stratum_value"],
                "minimum_spearman": float(worst_rank["spearman"]),
                "minimum_spearman_stratum": worst_rank["stratum_value"],
                "maximum_regret_at_1": float(worst_regret["regret_at_1"]),
                "maximum_regret_stratum": worst_regret["stratum_value"],
                "total_optimism_gt_0p05": int(group["optimism_gt_0p05_count"].sum()),
            }
        )
    robustness = pd.DataFrame(rows)
    robustness.to_csv(ROUND_DIR / "stratum_robustness_summary.csv", index=False)

    selection = robustness.loc[robustness["stratum_type"].eq("selection_stratum")].copy()
    selection["negative_rank"] = selection["minimum_spearman"].lt(0)
    selection["compressed_below_half"] = selection["minimum_calibration_slope"].lt(0.5)
    if not selection["compressed_below_half"].all():
        raise ValueError("Expected every baseline to fail the half-slope worst-selection-stratum gate")

    figure = px.scatter(
        robustness,
        x="worst_rmse",
        y="minimum_calibration_slope",
        color="maximum_regret_at_1",
        symbol="stratum_type",
        facet_col="target",
        hover_name="model",
        hover_data=[
            "worst_rmse_stratum",
            "minimum_slope_stratum",
            "minimum_spearman",
            "minimum_spearman_stratum",
        ],
        color_continuous_scale="RdYlGn_r",
        title="Exposed adversarial panel: no baseline is calibrated across all proposal strata",
        labels={
            "worst_rmse": "Worst stratum RMSE",
            "minimum_calibration_slope": "Minimum observed-on-predicted slope",
            "maximum_regret_at_1": "Worst stratum Regret@1",
        },
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#4d5963")
    figure.add_hline(y=0.5, line_dash="dot", line_color="#a33b20")
    figure.update_layout(template="plotly_white", height=620, width=1250)
    figure.write_html(
        ROUND_DIR / "stratum_robustness.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    winners = (
        robustness.sort_values("worst_rmse")
        .groupby(["target", "stratum_type"], as_index=False)
        .first()[
            [
                "target",
                "stratum_type",
                "model",
                "worst_rmse",
                "minimum_calibration_slope",
                "minimum_spearman",
                "maximum_regret_at_1",
            ]
        ]
    )
    winners.to_csv(ROUND_DIR / "worst_case_winners.csv", index=False)
    negative_counts = (
        selection.groupby("target")["negative_rank"]
        .agg([("negative_rank_models", "sum"), ("model_count", "size")])
        .reset_index()
    )
    report = "\n".join(
        [
            "# Round 62: adversarial proposal-stratum robustness",
            "",
            "This audit summarizes the already-exposed, frozen baseline predictions. It proposes or tunes no model and reads no sealed confirmation outcome.",
            "",
            "## Worst-case winners",
            "",
            winners.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Selection-stratum failure counts",
            "",
            negative_counts.to_markdown(index=False),
            "",
            "Every one of the 11 baselines has at least one selection stratum with observed-on-predicted calibration slope below 0.5 on each target. Rank correlation becomes negative in 6/11 Uncheatable models and 8/11 Table-9 models. The pooled target-matched winner is therefore not uniformly useful across baseline-ranked, challenger-ranked, and high-disagreement candidates.",
            "",
            "Origin strata are less pathological: inverse-deficit log link has the best Uncheatable worst-origin RMSE and bucket-family GRP has the best Table-9 worst-origin RMSE. But the selection-stratum reversal means proposal geometry, not only proposer identity, controls whether the response range is resolved.",
            "",
            "This blocks choosing a headline baseline from pooled RMSE. A future candidate must preserve calibration and ranking inside each major selection stratum, exactly as required by the frozen gate.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
