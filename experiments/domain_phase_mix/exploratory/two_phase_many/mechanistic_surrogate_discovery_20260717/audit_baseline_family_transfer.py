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
"""Audit whether frozen baseline-family rankings transfer across panels.

This is a model-selection audit, not a candidate screen. It uses only the
baseline table frozen before model generation and never reads sealed evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_BASELINES = ARTIFACT_ROOT / "frozen_gate/baseline_metrics.csv"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "baseline_family_transfer_audit"
FIT_SPLIT = "fit_oof"
HELDOUT_SPLIT = "heldout_policy_matched"
POLICY = "two_phase"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def panel_label(swarm: str, target: str) -> str:
    labels = {
        "300m": "300M",
        "delphi_3e18": "Delphi 3e18",
        "production": "Production",
        "starcoder_cosine": "StarCoder cosine",
        "starcoder_wsd80": "StarCoder WSD",
        "uncheatable": "Uncheatable",
        "table9": "Table-9",
        "starcoder_bpb": "StarCoder BPB",
    }
    return f"{labels.get(swarm, swarm)} / {labels.get(target, target)}"


def fit_panel_ranks(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = metrics.loc[metrics["policy"].eq(POLICY) & metrics["split"].eq(FIT_SPLIT)].copy()
    fit["parameter_count"] = pd.to_numeric(fit["parameter_count"], errors="coerce")
    fit["panel"] = [panel_label(swarm, target) for swarm, target in zip(fit["swarm"], fit["target"], strict=False)]
    panel_count = fit["panel"].nunique()
    common_models = fit.groupby("model")["panel"].nunique().loc[lambda values: values.eq(panel_count)].index
    fit = fit.loc[fit["model"].isin(common_models)].copy()
    fit["panel_best_rmse"] = fit.groupby("panel")["rmse"].transform("min")
    fit["relative_rmse"] = fit["rmse"] / fit["panel_best_rmse"]
    fit["rmse_rank"] = fit.groupby("panel")["rmse"].rank(method="min")

    rank_matrix = fit.pivot(index="model", columns="panel", values="rmse_rank")
    fit.pivot(index="model", columns="panel", values="relative_rmse")
    summaries = (
        fit.groupby("model", as_index=False)
        .agg(
            panel_count=("panel", "nunique"),
            mean_rmse_rank=("rmse_rank", "mean"),
            median_rmse_rank=("rmse_rank", "median"),
            worst_rmse_rank=("rmse_rank", "max"),
            best_panel_count=("rmse_rank", lambda values: int(np.sum(values.eq(1.0)))),
            mean_relative_rmse=("relative_rmse", "mean"),
            max_relative_rmse=("relative_rmse", "max"),
            median_parameter_count=("parameter_count", "median"),
        )
        .sort_values(["mean_rmse_rank", "mean_relative_rmse"])
    )

    pairwise_rows: list[dict[str, float | str | int]] = []
    for left_index, left in enumerate(rank_matrix.columns):
        for right in rank_matrix.columns[left_index + 1 :]:
            result = spearmanr(rank_matrix[left], rank_matrix[right])
            pairwise_rows.append(
                {
                    "left_panel": left,
                    "right_panel": right,
                    "n_models": len(rank_matrix),
                    "rmse_rank_spearman": float(result.statistic),
                }
            )
    return fit, summaries, pd.DataFrame(pairwise_rows)


def heldout_rank_transfer(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str | int]] = []
    panels = metrics.loc[
        metrics["policy"].eq(POLICY) & metrics["split"].eq(HELDOUT_SPLIT),
        ["swarm", "target"],
    ].drop_duplicates()
    for panel in panels.itertuples(index=False):
        local = metrics.loc[
            metrics["policy"].eq(POLICY)
            & metrics["swarm"].eq(panel.swarm)
            & metrics["target"].eq(panel.target)
            & metrics["split"].isin((FIT_SPLIT, HELDOUT_SPLIT))
        ].copy()
        wide = local.pivot(index="model", columns="split", values="rmse").dropna()
        wide["fit_rank"] = wide[FIT_SPLIT].rank(method="min")
        wide["heldout_rank"] = wide[HELDOUT_SPLIT].rank(method="min")
        wide["rank_change_heldout_minus_fit"] = wide["heldout_rank"] - wide["fit_rank"]
        wide["heldout_to_fit_rmse"] = wide[HELDOUT_SPLIT] / wide[FIT_SPLIT]
        wide = wide.reset_index()
        wide.insert(0, "panel", panel_label(panel.swarm, panel.target))
        wide.insert(1, "swarm", panel.swarm)
        wide.insert(2, "target", panel.target)
        rows.append(wide)
        result = spearmanr(wide["fit_rank"], wide["heldout_rank"])
        fit_winner = wide.sort_values(FIT_SPLIT).iloc[0]
        heldout_winner = wide.sort_values(HELDOUT_SPLIT).iloc[0]
        summaries.append(
            {
                "panel": panel_label(panel.swarm, panel.target),
                "swarm": panel.swarm,
                "target": panel.target,
                "n_models": len(wide),
                "fit_to_heldout_rank_spearman": float(result.statistic),
                "fit_winner": fit_winner["model"],
                "fit_winner_fit_rmse": float(fit_winner[FIT_SPLIT]),
                "fit_winner_heldout_rmse": float(fit_winner[HELDOUT_SPLIT]),
                "heldout_winner": heldout_winner["model"],
                "heldout_winner_fit_rmse": float(heldout_winner[FIT_SPLIT]),
                "heldout_winner_heldout_rmse": float(heldout_winner[HELDOUT_SPLIT]),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def write_dashboard(
    fit_ranks: pd.DataFrame,
    model_summary: pd.DataFrame,
    heldout_ranks: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    path: Path,
) -> None:
    model_order = model_summary["model"].tolist()
    panel_order = fit_ranks["panel"].drop_duplicates().tolist()
    relative = fit_ranks.pivot(index="model", columns="panel", values="relative_rmse").reindex(
        index=model_order, columns=panel_order
    )
    heatmap = px.imshow(
        relative,
        color_continuous_scale="RdYlGn_r",
        zmin=1.0,
        zmax=max(1.05, float(np.nanquantile(relative.to_numpy(), 0.95))),
        text_auto=".3f",
        labels={"color": "RMSE / panel best"},
        aspect="auto",
    )
    heatmap.update_layout(
        title="Frozen baseline-family fit-OOF transfer",
        xaxis_title="Swarm / target",
        yaxis_title="Surrogate",
        height=540,
        width=1400,
        margin=dict(l=280, r=100, t=90, b=180),
    )
    heatmap.update_xaxes(tickangle=-35)

    panels = heldout_summary["panel"].tolist()
    rank_figure = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=panels,
        horizontal_spacing=0.08,
    )
    for column, panel in enumerate(panels, start=1):
        local = heldout_ranks.loc[heldout_ranks["panel"].eq(panel)].copy()
        rank_figure.add_trace(
            go.Scatter(
                x=local["fit_rank"],
                y=local["heldout_rank"],
                mode="markers+text",
                text=local["model"],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=local["heldout_to_fit_rmse"],
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="Heldout / fit RMSE") if column == len(panels) else None,
                ),
                customdata=np.stack(
                    [local[FIT_SPLIT], local[HELDOUT_SPLIT], local["rank_change_heldout_minus_fit"]], axis=1
                ),
                hovertemplate=(
                    "%{text}<br>fit rank=%{x:.0f}<br>heldout rank=%{y:.0f}"
                    "<br>fit RMSE=%{customdata[0]:.5f}<br>heldout RMSE=%{customdata[1]:.5f}"
                    "<br>rank change=%{customdata[2]:+.0f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        max_rank = len(local)
        rank_figure.add_trace(
            go.Scatter(
                x=[1, max_rank],
                y=[1, max_rank],
                mode="lines",
                line=dict(color="#52636d", dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        rank_figure.update_xaxes(title_text="Fit-OOF RMSE rank", row=1, col=column)
        rank_figure.update_yaxes(title_text="Heldout RMSE rank" if column == 1 else "", row=1, col=column)
    rank_figure.update_layout(
        title="Fit-OOF model selection does not reliably transfer to heldouts",
        height=600,
        width=max(1200, 470 * len(panels)),
        margin=dict(l=80, r=120, t=100, b=80),
    )

    html = "\n".join(
        (
            "<!doctype html><html><head><meta charset='utf-8'><title>Baseline family transfer audit</title></head><body>",
            heatmap.to_html(full_html=False, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}}),
            rank_figure.to_html(full_html=False, include_plotlyjs=False, config={"toImageButtonOptions": {"scale": 4}}),
            "</body></html>",
        )
    )
    path.write_text(html)


def write_report(
    model_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    heldout_ranks: pd.DataFrame,
    path: Path,
) -> None:
    worst_pair = pairwise.sort_values("rmse_rank_spearman").iloc[0]
    rank_inversions = heldout_ranks.reindex(
        heldout_ranks["rank_change_heldout_minus_fit"].abs().sort_values(ascending=False).index
    ).head(12)
    lines = [
        "# Frozen baseline-family transfer audit",
        "",
        "This audit uses only the baseline table frozen before candidate generation. It asks whether choosing a "
        "surrogate by ordinary fit-panel grouped OOF RMSE identifies the same family across swarms or on development "
        "heldouts.",
        "",
        "## Cross-panel fit-OOF ranking",
        "",
        model_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"The weakest pairwise fit-OOF rank agreement is `{worst_pair['left_panel']}` versus "
        f"`{worst_pair['right_panel']}` (Spearman `{worst_pair['rmse_rank_spearman']:.3f}` across "
        f"{int(worst_pair['n_models'])} common forms). No common form wins every panel.",
        "",
        "## Fit-OOF to heldout rank transfer",
        "",
        heldout_summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "Largest rank inversions:",
        "",
        rank_inversions[
            [
                "panel",
                "model",
                FIT_SPLIT,
                HELDOUT_SPLIT,
                "fit_rank",
                "heldout_rank",
                "rank_change_heldout_minus_fit",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Interpretation",
        "",
        "- Fit-panel OOF error is necessary but not sufficient for choosing the structural family. The ranking is "
        "panel-dependent and can invert under optimizer-derived heldout policies.",
        "- The Delphi inversion is not a small tie: effective-exposure is the fit-OOF winner on both targets, yet its "
        "heldout rank drops substantially. Early-family asymmetric is not the fit-OOF winner, but is the heldout winner.",
        "- This supports the frozen gate's independent-heldout and raw-optimum requirements. It does not justify "
        "selecting early-family asymmetric as a transferable headline model, because that form has not improved a "
        "second independent panel and its raw optimum remains unsupported.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.baselines)
    metrics = pd.read_csv(args.baselines)
    fit_ranks, model_summary, pairwise = fit_panel_ranks(metrics)
    heldout_ranks, heldout_summary = heldout_rank_transfer(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_ranks.to_csv(args.output_dir / "fit_panel_ranks.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_transfer_summary.csv", index=False)
    pairwise.to_csv(args.output_dir / "fit_panel_pairwise_rank_correlations.csv", index=False)
    heldout_ranks.to_csv(args.output_dir / "fit_to_heldout_rank_transfer.csv", index=False)
    heldout_summary.to_csv(args.output_dir / "fit_to_heldout_rank_summary.csv", index=False)
    write_dashboard(
        fit_ranks,
        model_summary,
        heldout_ranks,
        heldout_summary,
        args.output_dir / "baseline_family_transfer.html",
    )
    write_report(
        model_summary,
        pairwise,
        heldout_summary,
        heldout_ranks,
        args.output_dir / "report.md",
    )


if __name__ == "__main__":
    main()
