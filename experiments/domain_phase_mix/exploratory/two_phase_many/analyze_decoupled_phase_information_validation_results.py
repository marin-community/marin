# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "wandb",
# ]
# ///
"""Collect and analyze the decoupled phase-information 3e18 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_panel_20260712"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_results_20260712"
TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-decoupled-phase-information"
EVAL_GROUP = "olmo_base_eval_table9_decoupled_phase_information"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
RUN_SUFFIX = "_3e18"
EXPECTED_CANDIDATES = 79
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

# Ten independent proportional runs at 3e18. These are deliberately reported as
# independent-run difference scales, not as standard errors for this shared-seed panel.
NOISE = {
    "uncheatable": {"run_sd": 0.00091299968961728, "difference_sd": 0.001291176543909418},
    "table9": {"run_sd": 0.003771768091801164, "difference_sd": 0.0053340855895512955},
}
PRIOR_FRONTIERS = {
    "uncheatable": {
        "matched one-phase control": 0.9851201772689819,
        "prior controlled two-phase": 0.985661,
        "prior separate-heads two-phase": 0.9887123108,
    },
    "table9": {
        "matched observed-best one-phase control": 1.0575300915544252,
        "prior separate-heads two-phase": 1.066538,
    },
}
FAMILY_COLORS = {
    "canonical": "#D73027",
    "effective_exposure": "#FC8D59",
    "effective_exposure_geometry": "#1A9850",
    "separate_heads": "#4575B4",
    "control": "#4D5562",
}
FAMILY_LABELS = {
    "canonical": "Canonical DSP",
    "effective_exposure": "Effective-exposure DSP",
    "effective_exposure_geometry": "Effective-exposure + geometry",
    "separate_heads": "Separate heads",
    "control": "Aggregate-matched tied control",
}
ANCHOR_LABELS = {
    "unch05": "Uncheatable: one-phase anchor KL=0.05",
    "t9s05": "Table-9: stable one-phase anchor KL=0.05",
    "t9b075": "Table-9: observed-best one-phase anchor KL=0.075",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def finished_training_runs(runs: list[wandb.apis.public.Run], candidates: set[str]) -> dict[str, wandb.apis.public.Run]:
    matches: dict[str, list[wandb.apis.public.Run]] = {candidate: [] for candidate in candidates}
    for run in runs:
        for candidate in candidates:
            if run.name.startswith(f"{candidate}{RUN_SUFFIX}-") and run.state == "finished":
                matches[candidate].append(run)
                break
    bad = {candidate: len(found) for candidate, found in matches.items() if len(found) != 1}
    if bad:
        raise ValueError(f"Expected one finished training run per candidate: {bad}")
    return {candidate: found[0] for candidate, found in matches.items()}


def finished_eval_runs(runs: list[wandb.apis.public.Run], candidates: set[str]) -> dict[str, wandb.apis.public.Run]:
    matches: dict[str, list[wandb.apis.public.Run]] = {candidate: [] for candidate in candidates}
    for run in runs:
        if run.state != "finished" or not run.name.startswith("t9_") or not run.name.endswith(RUN_SUFFIX):
            continue
        candidate = run.name[len("t9_") : -len(RUN_SUFFIX)]
        if candidate in candidates and run.summary.get(TABLE9_METRIC) is not None:
            matches[candidate].append(run)
    bad = {candidate: len(found) for candidate, found in matches.items() if len(found) != 1}
    if bad:
        raise ValueError(f"Expected one finished native Table-9 eval per candidate: {bad}")
    return {candidate: found[0] for candidate, found in matches.items()}


def collect_results(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    candidates = set(manifest["candidate"].astype(str))
    api = wandb.Api(timeout=180)
    training_api_rows = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=200))
    eval_api_rows = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=200))
    training_runs = finished_training_runs(training_api_rows, candidates)
    eval_runs = finished_eval_runs(eval_api_rows, candidates)

    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        candidate = str(record["candidate"])
        training_run = training_runs[candidate]
        eval_run = eval_runs[candidate]
        uncheatable = training_run.summary.get(UNCHEATABLE_METRIC)
        table9 = eval_run.summary.get(TABLE9_METRIC)
        if uncheatable is None or table9 is None:
            raise ValueError(f"Missing metric for {candidate}: uncheatable={uncheatable}, table9={table9}")
        observed_target = float(uncheatable if record["objective"] == "uncheatable" else table9)
        rows.append(
            {
                **record,
                "observed_uncheatable_bpb": float(uncheatable),
                "observed_table9_macro_bpb": float(table9),
                "observed_target_bpb": observed_target,
                "training_wandb_name": training_run.name,
                "training_wandb_url": training_run.url,
                "eval_wandb_name": eval_run.name,
                "eval_wandb_url": eval_run.url,
            }
        )
    results = pd.DataFrame(rows)
    if len(results) != EXPECTED_CANDIDATES or results["candidate"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_CANDIDATES} unique candidates, got {len(results)}")

    results["tied_observed_target_bpb"] = np.nan
    for anchor_tag, anchor_rows in results.groupby("anchor_tag"):
        controls = anchor_rows[anchor_rows["family"].eq("control")]
        if len(controls) != 1:
            raise ValueError(f"Expected one tied control for {anchor_tag}, got {len(controls)}")
        results.loc[results["anchor_tag"].eq(anchor_tag), "tied_observed_target_bpb"] = float(
            controls.iloc[0]["observed_target_bpb"]
        )
    results["observed_gain_vs_tied"] = results["tied_observed_target_bpb"] - results["observed_target_bpb"]
    results["independent_difference_sd_units"] = results.apply(
        lambda row: row["observed_gain_vs_tied"] / NOISE[str(row["objective"])]["difference_sd"], axis=1
    )
    results["gain_realization_fraction"] = np.where(
        results["predicted_gain_vs_tied"].abs() > 1e-12,
        results["observed_gain_vs_tied"] / results["predicted_gain_vs_tied"],
        np.nan,
    )
    results = results.sort_values(["objective", "anchor_tag", "family", "phase_information_budget"]).reset_index(
        drop=True
    )
    audit = {
        "training_api_rows": len(training_api_rows),
        "finished_training_rows": len(training_runs),
        "eval_api_rows": len(eval_api_rows),
        "finished_eval_rows": len(eval_runs),
        "non_finished_eval_attempts": len(eval_api_rows) - len(eval_runs),
    }
    return results, audit


def path_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (objective, anchor_tag, family), path in results[~results["family"].eq("control")].groupby(
        ["objective", "anchor_tag", "family"]
    ):
        path = path.sort_values("phase_information_budget")
        best = path.loc[path["observed_target_bpb"].idxmin()]
        predicted_vs_observed_spearman = (
            path[["predicted_bpb", "observed_target_bpb"]].corr(method="spearman").iloc[0, 1]
        )
        rows.append(
            {
                "objective": objective,
                "anchor_tag": anchor_tag,
                "family": family,
                "n": len(path),
                "best_candidate": best["candidate"],
                "best_phase_information_budget": float(best["phase_information_budget"]),
                "best_observed_target_bpb": float(best["observed_target_bpb"]),
                "best_observed_gain_vs_tied": float(best["observed_gain_vs_tied"]),
                "best_independent_difference_sd_units": float(best["independent_difference_sd_units"]),
                "best_predicted_gain_vs_tied": float(best["predicted_gain_vs_tied"]),
                "best_gain_realization_fraction": float(best["gain_realization_fraction"]),
                "best_phase_tv": float(best["phase_tv"]),
                "predicted_vs_observed_spearman": float(predicted_vs_observed_spearman),
                "best_is_smallest_epsilon": bool(
                    best["phase_information_budget"] == path["phase_information_budget"].min()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["objective", "anchor_tag", "best_observed_target_bpb"]).reset_index(drop=True)


def frontier_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for objective, objective_rows in results.groupby("objective"):
        best = objective_rows.loc[objective_rows["observed_target_bpb"].idxmin()]
        rows.append(
            {
                "objective": objective,
                "candidate": best["candidate"],
                "anchor_tag": best["anchor_tag"],
                "family": best["family"],
                "phase_information_budget": float(best["phase_information_budget"]),
                "phase_tv": float(best["phase_tv"]),
                "observed_target_bpb": float(best["observed_target_bpb"]),
                "tied_observed_target_bpb": float(best["tied_observed_target_bpb"]),
                "observed_gain_vs_tied": float(best["observed_gain_vs_tied"]),
                "independent_difference_sd_units": float(best["independent_difference_sd_units"]),
                "predicted_gain_vs_tied": float(best["predicted_gain_vs_tied"]),
                "gain_realization_fraction": float(best["gain_realization_fraction"]),
            }
        )
    return pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)


def render_paths(results: pd.DataFrame, output_dir: Path) -> None:
    anchors = ["unch05", "t9s05", "t9b075"]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[ANCHOR_LABELS[anchor] for anchor in anchors],
        horizontal_spacing=0.07,
    )
    legend_seen: set[str] = set()
    for col, anchor_tag in enumerate(anchors, start=1):
        anchor_rows = results[results["anchor_tag"].eq(anchor_tag)]
        control = anchor_rows[anchor_rows["family"].eq("control")].iloc[0]
        fig.add_hline(
            y=float(control["observed_target_bpb"]),
            line={"color": FAMILY_COLORS["control"], "dash": "dash", "width": 1.5},
            row=1,
            col=col,
        )
        for family, path in anchor_rows[~anchor_rows["family"].eq("control")].groupby("family"):
            path = path.sort_values("phase_information_budget")
            label = FAMILY_LABELS[str(family)]
            fig.add_trace(
                go.Scatter(
                    x=path["phase_information_budget"],
                    y=path["observed_target_bpb"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=label not in legend_seen,
                    line={"color": FAMILY_COLORS[str(family)], "width": 2},
                    marker={"size": 8},
                    customdata=np.stack(
                        [
                            path["candidate"],
                            path["observed_gain_vs_tied"],
                            path["predicted_gain_vs_tied"],
                            path["phase_tv"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>epsilon=%{x:.3f}<br>observed BPB=%{y:.6f}"
                        "<br>observed gain vs tied=%{customdata[1]:+.6f}"
                        "<br>predicted gain vs tied=%{customdata[2]:+.6f}"
                        "<br>phase TV=%{customdata[3]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            legend_seen.add(label)
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[control["observed_target_bpb"]],
                mode="markers",
                name=FAMILY_LABELS["control"],
                legendgroup="control",
                showlegend=col == 1,
                marker={"color": FAMILY_COLORS["control"], "size": 10, "symbol": "diamond"},
                customdata=[[control["candidate"]]],
                hovertemplate="%{customdata[0]}<br>epsilon=0<br>observed BPB=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Phase-information budget", row=1, col=col)
        fig.update_yaxes(title_text="Observed target BPB" if col == 1 else None, row=1, col=col)
    fig.update_layout(
        title={"text": "Fixed-aggregate phase-ordering paths at 3e18", "x": 0.5},
        template="plotly_white",
        width=1700,
        height=700,
        margin={"l": 80, "r": 40, "t": 170, "b": 90},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.14, "xanchor": "center", "x": 0.5},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.write_html(output_dir / "observed_phase_information_paths.html", include_plotlyjs=True, config=EXPORT_CONFIG)
    fig.write_image(output_dir / "observed_phase_information_paths.png", scale=2)


def render_gain_calibration(results: pd.DataFrame, output_dir: Path) -> None:
    candidates = results[~results["family"].eq("control")].copy()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[ANCHOR_LABELS[anchor] for anchor in ("unch05", "t9s05", "t9b075")],
    )
    legend_seen: set[str] = set()
    for col, anchor_tag in enumerate(("unch05", "t9s05", "t9b075"), start=1):
        anchor_rows = candidates[candidates["anchor_tag"].eq(anchor_tag)]
        for family, family_rows in anchor_rows.groupby("family"):
            label = FAMILY_LABELS[str(family)]
            fig.add_trace(
                go.Scatter(
                    x=family_rows["predicted_gain_vs_tied"],
                    y=family_rows["observed_gain_vs_tied"],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=label not in legend_seen,
                    marker={
                        "size": 7 + 18 * family_rows["phase_information_budget"] / 0.2,
                        "color": FAMILY_COLORS[str(family)],
                        "opacity": 0.85,
                    },
                    customdata=np.stack(
                        [
                            family_rows["candidate"],
                            family_rows["phase_information_budget"],
                            family_rows["phase_tv"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>epsilon=%{customdata[1]:.3f}<br>phase TV=%{customdata[2]:.3f}"
                        "<br>predicted gain=%{x:+.6f}<br>observed gain=%{y:+.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            legend_seen.add(label)
        fig.add_hline(y=0.0, line={"color": "#4D5562", "dash": "dash", "width": 1}, row=1, col=col)
        fig.update_xaxes(title_text="Predicted gain vs tied (BPB)", row=1, col=col)
        fig.update_yaxes(title_text="Observed gain vs tied (BPB)" if col == 1 else None, row=1, col=col)
    fig.update_layout(
        title={"text": "Surrogate phase gains are strongly overpredicted", "x": 0.5},
        template="plotly_white",
        width=1700,
        height=700,
        margin={"l": 80, "r": 40, "t": 170, "b": 90},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.14, "xanchor": "center", "x": 0.5},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.write_html(output_dir / "predicted_vs_observed_phase_gain.html", include_plotlyjs=True, config=EXPORT_CONFIG)
    fig.write_image(output_dir / "predicted_vs_observed_phase_gain.png", scale=2)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    rows = []
    headers = [column.replace("_", " ") for column in columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(":" + "-" * max(3, len(header) - 1) for header in headers) + "|")
    for record in frame[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(formats[column].format(value) if column in formats else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    results: pd.DataFrame,
    paths: pd.DataFrame,
    frontiers: pd.DataFrame,
    audit: dict[str, int],
    output_dir: Path,
) -> None:
    best_uncheatable = frontiers[frontiers["objective"].eq("uncheatable")].iloc[0]
    best_table9 = frontiers[frontiers["objective"].eq("table9")].iloc[0]
    prior_controlled_uncheatable = PRIOR_FRONTIERS["uncheatable"]["prior controlled two-phase"]
    stable_table9 = paths[paths["anchor_tag"].eq("t9s05") & paths["family"].eq("effective_exposure")].iloc[0]
    uncheatable_headline = (
        f"1. **Uncheatable has a new single-seed frontier.** `{best_uncheatable['candidate']}` reaches "
        f"**{best_uncheatable['observed_target_bpb']:.6f} BPB** at phase-information budget "
        f"{best_uncheatable['phase_information_budget']:.3f}. It improves its exact aggregate-matched tied control "
        f"({best_uncheatable['tied_observed_target_bpb']:.6f}) by "
        f"**{best_uncheatable['observed_gain_vs_tied']:.6f} BPB**, or "
        f"{best_uncheatable['independent_difference_sd_units']:.2f} independent-run difference SDs. It also beats "
        f"the prior controlled two-phase value {prior_controlled_uncheatable:.6f} by "
        f"{prior_controlled_uncheatable - best_uncheatable['observed_target_bpb']:.6f} BPB."
    )
    table9_headline = (
        f"2. **Table-9 does not establish a new two-phase frontier.** The best candidate "
        f"`{best_table9['candidate']}` is {best_table9['observed_target_bpb']:.6f}, only "
        f"{best_table9['observed_gain_vs_tied']:.6f} BPB below its observed-best tied control "
        f"({best_table9['tied_observed_target_bpb']:.6f}); this is "
        f"{best_table9['independent_difference_sd_units']:.2f} independent-run difference SDs and is not resolved "
        "by this one-seed panel."
    )
    stable_table9_headline = (
        "3. **A weaker Table-9 aggregate can be repaired by a small phase split.** On the stable KL=0.05 anchor, "
        f"effective exposure at epsilon={stable_table9['best_phase_information_budget']:.3f} improves the exact "
        f"tied control by {stable_table9['best_observed_gain_vs_tied']:.6f} BPB "
        f"({stable_table9['best_independent_difference_sd_units']:.2f} independent-run difference SDs), reaching "
        f"{stable_table9['best_observed_target_bpb']:.6f}. This is suggestive phase-ordering signal, but it merely "
        "ties the observed-best one-phase frontier and needs repeats before a causal claim."
    )
    magnitude_headline = (
        "4. **The phase-magnitude model remains wrong.** Every surrogate predicts larger gains as phase information "
        "increases, but observed paths generally worsen; pathwise predicted-vs-observed Spearman ranges from "
        f"{paths['predicted_vs_observed_spearman'].min():.3f} to "
        f"{paths['predicted_vs_observed_spearman'].max():.3f}. The best point is the smallest tested epsilon in "
        f"{int(paths['best_is_smallest_epsilon'].sum())}/{len(paths)} paths."
    )
    completion_line = (
        f"- All {audit['finished_training_rows']}/{EXPECTED_CANDIDATES} training runs and "
        f"{audit['finished_eval_rows']}/{EXPECTED_CANDIDATES} native Table-9 evals finished."
    )
    retry_line = (
        f"- W&B contained {audit['non_finished_eval_attempts']} non-finished eval attempts; collection uses only the "
        "unique finished eval for each manifest candidate."
    )
    interpretation_one = (
        "Decoupling aggregate specialization from phase information fixes the earlier structural regularization error: "
        "the two-phase policy is no longer charged extra merely for differing across phases, and aggregate exposure is "
        "identical to the one-phase anchor. The panel therefore isolates phase order cleanly."
    )
    interpretation_two = (
        "The result is not that phase order is useless. Uncheatable shows a small, useful split at epsilon=0.005, and "
        "the stable Table-9 anchor shows a suggestive improvement at epsilon=0.01. The result is that all tested "
        "surrogates overstate how far to move along their phase direction. Canonical DSP, effective-exposure DSP, "
        "geometry, and separate heads all become worse when allowed moderate or large phase information, despite "
        "monotonically improving surrogate predictions."
    )
    interpretation_three = (
        "The next modeling target should therefore be **phase-gain magnitude and saturation**, not a richer aggregate "
        "model. For deployment, epsilon should be selected as a real hyperparameter from validation rather than "
        "inferred from unconstrained surrogate gain. A focused repeat panel on the Uncheatable winner and its tied "
        "control is warranted before scaling; Table-9 should not be scaled as a two-phase improvement from this "
        "evidence."
    )
    statistical_boundary = (
        "The reported SD units divide a same-seed observed difference by the independent-run difference SD from the "
        "proportional noise panel. This is a reference scale, not a z-test: the covariance of these exact paired "
        "schedules is unknown, and there are no repeat seeds in this panel. Values below roughly one Table-9 difference "
        "SD (0.00533 BPB) should not be interpreted as resolved improvements."
    )
    path_columns = [
        "objective",
        "anchor_tag",
        "family",
        "best_phase_information_budget",
        "best_observed_target_bpb",
        "best_observed_gain_vs_tied",
        "best_independent_difference_sd_units",
        "best_predicted_gain_vs_tied",
        "predicted_vs_observed_spearman",
    ]
    path_formats = {
        "best_phase_information_budget": "{:.3f}",
        "best_observed_target_bpb": "{:.6f}",
        "best_observed_gain_vs_tied": "{:+.6f}",
        "best_independent_difference_sd_units": "{:+.2f}",
        "best_predicted_gain_vs_tied": "{:+.6f}",
        "predicted_vs_observed_spearman": "{:+.3f}",
    }
    report = f"""# Decoupled phase-information panel: 3e18 results

## Completion and provenance

{completion_line}
{retry_line}
- Every candidate uses data seed 690300. The aggregate mixture is exactly fixed within each anchor; only phase ordering
  changes.

## Headline findings

{uncheatable_headline}
{table9_headline}
{stable_table9_headline}
{magnitude_headline}

## Best point on each path

{markdown_table(paths, path_columns, path_formats)}

## Interpretation

{interpretation_one}

{interpretation_two}

{interpretation_three}

## Statistical boundary

{statistical_boundary}
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.panel_dir / "selected_candidate_manifest.csv")
    results, audit = collect_results(manifest)
    paths = path_summary(results)
    frontiers = frontier_summary(results)
    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    paths.to_csv(args.output_dir / "path_summary.csv", index=False)
    frontiers.to_csv(args.output_dir / "frontier_summary.csv", index=False)
    (args.output_dir / "collection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    render_paths(results, args.output_dir)
    render_gain_calibration(results, args.output_dir)
    write_report(results, paths, frontiers, audit, args.output_dir)
    print(frontiers.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
