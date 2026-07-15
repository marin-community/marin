# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Collect and analyze the compact retained-state 3e18 validation panel."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "compact_retained_state_validation_panel_20260713"
DEFAULT_NOISE_PANEL = SCRIPT_DIR / "reference_outputs/delphi_3e18_proportional_noise_floor_20260703/noise_panel.csv"
DEFAULT_PRIOR_RESULTS = SCRIPT_DIR / (
    "reference_outputs/decoupled_phase_information_validation_results_20260712/observed_results.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "compact_retained_state_validation_results_20260713"

WANDB_PROJECT = "marin-community/marin"
WANDB_TAG = "delphi-compact-retained-state-ablation"
TARGET_METRIC = "eval/uncheatable_eval/bpb"
EXPECTED_CANDIDATES = 4
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

PRIOR_FRONTIERS = {
    "Effective-exposure DSP, 2p": "dphase_unch05_eff_e0p005",
    "Separate heads, 2p": "dphase_unch05_sep_e0p005",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--noise-panel", type=Path, default=DEFAULT_NOISE_PANEL)
    parser.add_argument("--prior-results", type=Path, default=DEFAULT_PRIOR_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def collect_observed(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    api = wandb.Api(timeout=180)
    attempts = list(api.runs(WANDB_PROJECT, filters={"tags": {"$in": [WANDB_TAG]}}, per_page=100))
    rows: list[dict[str, object]] = []
    for candidate in manifest["candidate"]:
        matches = [
            run
            for run in attempts
            if run.state == "finished"
            and run.name.startswith(f"{candidate}_3e18-")
            and run.summary.get(TARGET_METRIC) is not None
        ]
        if len(matches) != 1:
            states = [(run.name, run.state) for run in attempts if run.name.startswith(f"{candidate}_3e18-")]
            raise ValueError(f"Expected one finished W&B run for {candidate}, got {states}")
        run = matches[0]
        rows.append(
            {
                "candidate": candidate,
                "observed_bpb": float(run.summary[TARGET_METRIC]),
                "eval_bpb": float(run.summary["eval/bpb"]),
                "eval_macro_bpb": float(run.summary["eval/macro_bpb"]),
                "train_loss": float(run.summary["train/loss"]),
                "final_step": int(run.summary["_step"]),
                "data_seed": int(run.config["data_seed"]),
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "wandb_url": run.url,
                "wandb_state": run.state,
            }
        )
    observed = pd.DataFrame(rows)
    audit = {
        "queried_at": datetime.now(UTC).isoformat(),
        "wandb_project": WANDB_PROJECT,
        "wandb_tag": WANDB_TAG,
        "wandb_attempt_count": len(attempts),
        "selected_run_count": len(observed),
        "selected_run_ids": observed["wandb_run_id"].tolist(),
        "non_finished_attempts": [run.name for run in attempts if run.state != "finished"],
    }
    return observed, audit


def add_prediction_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    output["prediction_error"] = output["observed_bpb"] - output["predicted_bpb"]
    output["predicted_phase_gain_vs_tied"] = output["predicted_tied_bpb"] - output["predicted_bpb"]
    return output.sort_values(["grouping", "policy"]).reset_index(drop=True)


def noise_statistics(noise_panel: pd.DataFrame) -> dict[str, float]:
    repeat_sd = float(noise_panel["uncheatable_bpb"].std(ddof=1))
    return {
        "proportional_repeat_mean": float(noise_panel["uncheatable_bpb"].mean()),
        "proportional_repeat_sd": repeat_sd,
        "independent_two_run_difference_sd": float(np.sqrt(2.0) * repeat_sd),
        "repeat_count": len(noise_panel),
    }


def factorial_contrasts(results: pd.DataFrame, difference_sd: float) -> pd.DataFrame:
    indexed = results.set_index(["grouping", "policy"])

    def observed(grouping: str, policy: str) -> float:
        return float(indexed.loc[(grouping, policy), "observed_bpb"])

    def predicted(grouping: str, policy: str) -> float:
        return float(indexed.loc[(grouping, policy), "predicted_bpb"])

    rows: list[dict[str, object]] = []
    phase_gains: dict[str, float] = {}
    predicted_phase_gains: dict[str, float] = {}
    for grouping in ("nogroup", "grouped"):
        gain = observed(grouping, "1p") - observed(grouping, "2p")
        predicted_gain = float(indexed.loc[(grouping, "2p"), "predicted_phase_gain_vs_tied"])
        phase_gains[grouping] = gain
        predicted_phase_gains[grouping] = predicted_gain
        rows.append(
            {
                "contrast": f"phase_gain_{grouping}",
                "interpretation": f"{grouping}: 1p BPB - 2p BPB",
                "observed_gain": gain,
                "predicted_gain": predicted_gain,
                "gain_realization_fraction": gain / predicted_gain,
                "independent_difference_sd_units": gain / difference_sd,
            }
        )
    for policy in ("1p", "2p"):
        gain = observed("nogroup", policy) - observed("grouped", policy)
        predicted_gain = predicted("nogroup", policy) - predicted("grouped", policy)
        rows.append(
            {
                "contrast": f"grouping_gain_{policy}",
                "interpretation": f"{policy}: no-group BPB - grouped BPB",
                "observed_gain": gain,
                "predicted_gain": predicted_gain,
                "gain_realization_fraction": gain / predicted_gain,
                "independent_difference_sd_units": gain / difference_sd,
            }
        )
    observed_interaction = phase_gains["grouped"] - phase_gains["nogroup"]
    predicted_interaction = predicted_phase_gains["grouped"] - predicted_phase_gains["nogroup"]
    rows.append(
        {
            "contrast": "phase_by_grouping_interaction",
            "interpretation": "grouped phase gain - no-group phase gain",
            "observed_gain": observed_interaction,
            "predicted_gain": predicted_interaction,
            "gain_realization_fraction": observed_interaction / predicted_interaction,
            "independent_difference_sd_units": observed_interaction / difference_sd,
        }
    )
    best = float(results["observed_bpb"].min())
    worst = float(results["observed_bpb"].max())
    rows.append(
        {
            "contrast": "best_vs_worst_panel",
            "interpretation": "worst panel BPB - best panel BPB",
            "observed_gain": worst - best,
            "predicted_gain": np.nan,
            "gain_realization_fraction": np.nan,
            "independent_difference_sd_units": (worst - best) / difference_sd,
        }
    )
    return pd.DataFrame(rows)


def frontier_comparison(results: pd.DataFrame, prior_results: pd.DataFrame, difference_sd: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, candidate in PRIOR_FRONTIERS.items():
        match = prior_results[prior_results["candidate"].eq(candidate)]
        if len(match) != 1:
            raise ValueError(f"Expected one prior result for {candidate}, got {len(match)}")
        row = match.iloc[0]
        rows.append(
            {
                "method": label,
                "candidate": candidate,
                "observed_bpb": float(row["observed_uncheatable_bpb"]),
                "wandb_url": row["training_wandb_url"],
                "source": "prior_frontier",
            }
        )
    for row in results.itertuples(index=False):
        rows.append(
            {
                "method": f"Compact retained-state, {row.grouping}, {row.policy}",
                "candidate": row.candidate,
                "observed_bpb": row.observed_bpb,
                "wandb_url": row.wandb_url,
                "source": "current_panel",
            }
        )
    comparison = pd.DataFrame(rows).sort_values("observed_bpb").reset_index(drop=True)
    best_prior = float(comparison.loc[comparison["source"].eq("prior_frontier"), "observed_bpb"].min())
    comparison["gap_vs_best_prior"] = comparison["observed_bpb"] - best_prior
    comparison["gap_vs_best_prior_difference_sd_units"] = comparison["gap_vs_best_prior"] / difference_sd
    return comparison


def plot_results(results: pd.DataFrame, contrasts: pd.DataFrame, frontiers: pd.DataFrame, output_path: Path) -> None:
    colors = sample_colorscale("RdYlGn_r", [0.15, 0.85])
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Observed 3e18 factorial panel", "Predicted vs observed phase benefit"),
        horizontal_spacing=0.16,
    )
    for color, grouping in zip(colors, ("nogroup", "grouped"), strict=True):
        group = results[results["grouping"].eq(grouping)].set_index("policy").loc[["1p", "2p"]]
        figure.add_trace(
            go.Scatter(
                x=["One phase", "Two phases"],
                y=group["observed_bpb"],
                mode="lines+markers+text",
                text=[f"{value:.6f}" for value in group["observed_bpb"]],
                textposition="top center",
                name="Family coverage" if grouping == "grouped" else "No grouping prior",
                line={"color": color, "width": 3},
                marker={"size": 10},
                hovertemplate="%{fullData.name}<br>%{x}<br>BPB=%{y:.7f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    frontier_labels = {
        "Effective-exposure DSP, 2p": "Eff-exp 2p frontier",
        "Separate heads, 2p": "Separate-heads 2p",
    }
    for row in frontiers[frontiers["source"].eq("prior_frontier")].itertuples(index=False):
        figure.add_hline(
            y=row.observed_bpb,
            line_dash="dot",
            line_color="#6b7280",
            annotation_text=frontier_labels[row.method],
            annotation_position="bottom right",
            row=1,
            col=1,
        )
    phase = contrasts[contrasts["contrast"].str.startswith("phase_gain_")]
    for name, column, color in (
        ("Predicted", "predicted_gain", colors[1]),
        ("Observed", "observed_gain", colors[0]),
    ):
        figure.add_bar(
            x=phase["contrast"].map(
                {"phase_gain_nogroup": "No grouping prior", "phase_gain_grouped": "Family coverage"}
            ),
            y=phase[column],
            name=name,
            marker_color=color,
            text=[f"{value:.5f}" for value in phase[column]],
            textposition="outside",
            hovertemplate=f"{name} phase benefit=%{{y:.7f}} BPB<extra></extra>",
            row=1,
            col=2,
        )
    figure.update_yaxes(title_text="Uncheatable BPB (lower is better)", row=1, col=1)
    figure.update_yaxes(title_text="1p BPB - 2p BPB", rangemode="tozero", row=1, col=2)
    figure.update_layout(
        title="Compact retained-state 3e18 validation: directional effects, no frontier advance",
        template="plotly_white",
        barmode="group",
        height=620,
        width=1250,
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
        margin={"t": 100, "b": 130},
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    results: pd.DataFrame,
    contrasts: pd.DataFrame,
    frontiers: pd.DataFrame,
    noise: dict[str, float],
    output_dir: Path,
) -> None:
    best = results.loc[results["observed_bpb"].idxmin()]
    best_prior = frontiers.loc[frontiers["source"].eq("prior_frontier")].iloc[0]
    phase = contrasts[contrasts["contrast"].str.startswith("phase_gain_")]
    max_panel_signal = float(contrasts.loc[contrasts["contrast"].eq("best_vs_worst_panel"), "observed_gain"].iloc[0])
    lines = [
        "# Compact retained-state 3e18 validation results",
        "",
        "## Verdict",
        "",
        f"All four runs completed. The nominal winner is `{best['candidate']}` at "
        f"{best['observed_bpb']:.7f} Uncheatable BPB. It is {best['observed_bpb'] - best_prior['observed_bpb']:.7f} "
        f"worse than the established {best_prior['method']} frontier ({best_prior['observed_bpb']:.7f}).",
        "",
        "Family coverage and two phases both move in the expected direction, but this one-seed panel does not resolve "
        "either effect. The largest panel separation is "
        f"{max_panel_signal:.7f} BPB, or {max_panel_signal / noise['independent_two_run_difference_sd']:.2f} times "
        "the independent two-run difference SD estimated from proportional repeats.",
        "",
        "The central failure is transfer calibration. The two-phase fits predict "
        f"{phase['predicted_gain'].min():.5f}-{phase['predicted_gain'].max():.5f} BPB of phase benefit, but only "
        f"{phase['observed_gain'].min():.5f}-{phase['observed_gain'].max():.5f} materializes. The retained-state "
        "mechanism therefore remains too optimistic about phase ordering at its proposed optima.",
        "",
        "## Observed candidates",
        "",
        results[
            [
                "candidate",
                "grouping",
                "policy",
                "observed_bpb",
                "predicted_bpb",
                "prediction_error",
                "predicted_phase_gain_vs_tied",
                "wandb_url",
            ]
        ].to_markdown(index=False),
        "",
        "## Factorial contrasts",
        "",
        contrasts.to_markdown(index=False),
        "",
        "Positive phase or grouping gains mean the more structured arm is better. Noise units use the proportional "
        "repeat SD as a reference magnitude only; the four candidates share `data_seed=713300`, no paired repeats "
        "were run, and mixture-dependent heteroskedasticity prevents interpreting these units as formal z-scores.",
        "",
        "## Existing-frontier comparison",
        "",
        frontiers.to_markdown(index=False),
        "",
        "## Noise reference",
        "",
        "```json",
        json.dumps(noise, indent=2),
        "```",
        "",
        "## Decision",
        "",
        "Do not scale this surrogate or treat family coverage as established. Keep the compact retained-state form as "
        "an interpretable negative-result ablation. Further modeling should target its overprediction of phase-order "
        "benefit rather than adding more deployment regularization.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.panel_dir / "candidate_manifest.csv")
    if len(manifest) != EXPECTED_CANDIDATES or manifest["candidate"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} unique candidates, got {len(manifest)}")
    observed, audit = collect_observed(manifest)
    results = add_prediction_diagnostics(manifest.merge(observed, on="candidate", validate="one_to_one"))
    noise = noise_statistics(pd.read_csv(args.noise_panel))
    contrasts = factorial_contrasts(results, noise["independent_two_run_difference_sd"])
    frontiers = frontier_comparison(results, pd.read_csv(args.prior_results), noise["independent_two_run_difference_sd"])

    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    contrasts.to_csv(args.output_dir / "factorial_contrasts.csv", index=False)
    frontiers.to_csv(args.output_dir / "frontier_comparison.csv", index=False)
    audit["noise_reference"] = noise
    (args.output_dir / "collection_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    plot_results(results, contrasts, frontiers, args.output_dir / "compact_retained_state_validation.html")
    write_report(results, contrasts, frontiers, noise, args.output_dir)
    print(results[["candidate", "observed_bpb", "predicted_bpb"]].to_string(index=False))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
