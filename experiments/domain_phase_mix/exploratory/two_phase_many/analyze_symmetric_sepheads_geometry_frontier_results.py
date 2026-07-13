# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "wandb",
# ]
# ///
"""Collect and analyze the symmetric separate-heads and geometry 3e18 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "symmetric_sepheads_geometry_frontier_panel_20260711"
TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-symmetric-sepheads-geometry-frontier"
EVAL_GROUP = "olmo_base_eval_table9_scaling_validation"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
RUN_SUFFIX = "_3e18"

PRIOR_FRONTIERS = {
    "uncheatable": {
        "separate-heads KL=0.1": 0.9887123108,
        "best controlled two-phase": 0.985661,
    },
    "table9": {
        "separate-heads KL=0.1": 1.0676900654,
        "one-phase effective-exposure DSP": 1.070728,
    },
}

COLORS = {
    ("separate_heads", "1p"): "#E67E22",
    ("separate_heads", "2p"): "#C0392B",
    ("effective_exposure_geometry", "1p"): "#2E8B57",
    ("effective_exposure_geometry", "2p"): "#2C7FB8",
    ("effective_exposure_geometry", "tied"): "#7F8C8D",
}
DISPLAY_NAMES = {
    ("separate_heads", "1p"): "Separate heads: fitted one-phase",
    ("separate_heads", "2p"): "Separate heads: fitted two-phase",
    ("effective_exposure_geometry", "1p"): "Eff-exp + geometry: one-phase",
    ("effective_exposure_geometry", "2p"): "Eff-exp + geometry: two-phase",
    ("effective_exposure_geometry", "tied"): "Eff-exp + geometry: aggregate-matched tied",
}
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    return parser.parse_args()


def candidate_run(runs: list[wandb.apis.public.Run], candidate: str) -> wandb.apis.public.Run:
    prefix = f"{candidate}{RUN_SUFFIX}-"
    matches = [run for run in runs if run.name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one training run with prefix {prefix!r}, got {len(matches)}")
    return matches[0]


def candidate_eval_run(runs: list[wandb.apis.public.Run], candidate: str) -> wandb.apis.public.Run:
    expected_name = f"t9_{candidate}{RUN_SUFFIX}"
    matches = [run for run in runs if run.name == expected_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one eval run named {expected_name!r}, got {len(matches)}")
    return matches[0]


def collect_results(manifest: pd.DataFrame) -> pd.DataFrame:
    api = wandb.Api(timeout=120)
    training_runs = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=200))
    eval_runs = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=1000))
    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        candidate = str(record["candidate"])
        train_run = candidate_run(training_runs, candidate)
        eval_run = candidate_eval_run(eval_runs, candidate)
        uncheatable = train_run.summary.get(UNCHEATABLE_METRIC)
        table9 = eval_run.summary.get(TABLE9_METRIC)
        if train_run.state != "finished" or eval_run.state != "finished" or uncheatable is None or table9 is None:
            raise ValueError(
                f"Incomplete candidate {candidate}: train={train_run.state}/{uncheatable}, "
                f"eval={eval_run.state}/{table9}"
            )
        objective = str(record["objective"])
        target_observed = float(table9 if objective == "table9" else uncheatable)
        rows.append(
            {
                **record,
                "observed_uncheatable_bpb": float(uncheatable),
                "observed_table9_macro_bpb": float(table9),
                "observed_target_bpb": target_observed,
                "prediction_error_target": target_observed - float(record["predicted_bpb_300m"]),
                "training_wandb_state": train_run.state,
                "training_wandb_name": train_run.name,
                "training_wandb_url": train_run.url,
                "eval_wandb_state": eval_run.state,
                "eval_wandb_name": eval_run.name,
                "eval_wandb_url": eval_run.url,
            }
        )
    results = pd.DataFrame(rows)
    if len(results) != 30 or results["candidate"].duplicated().any():
        raise ValueError(f"Expected 30 unique completed candidates, got {len(results)}")
    return results.sort_values(["objective", "family", "policy", "kl_reg"]).reset_index(drop=True)


def rank_diagnostics(results: pd.DataFrame) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for (family, objective, policy), rows in results.groupby(["family", "objective", "policy"]):
        if len(rows) < 3:
            continue
        spearman = stats.spearmanr(rows["predicted_bpb_300m"], rows["observed_target_bpb"])
        diagnostics.append(
            {
                "family": family,
                "objective": objective,
                "policy": policy,
                "n": len(rows),
                "spearman": float(spearman.statistic),
                "mean_optimism": float((rows["predicted_bpb_300m"] - rows["observed_target_bpb"]).mean()),
            }
        )
    return diagnostics


def best_rows(results: pd.DataFrame) -> list[dict[str, object]]:
    best: list[dict[str, object]] = []
    for (objective, family, policy), rows in results.groupby(["objective", "family", "policy"]):
        row = rows.loc[rows["observed_target_bpb"].idxmin()]
        best.append(
            {
                "objective": objective,
                "family": family,
                "policy": policy,
                "candidate": row["candidate"],
                "kl_reg": float(row["kl_reg"]),
                "observed_target_bpb": float(row["observed_target_bpb"]),
                "predicted_bpb_300m": float(row["predicted_bpb_300m"]),
                "max_simulated_epoch": float(row["max_simulated_epoch"]),
            }
        )
    return best


def summary_payload(results: pd.DataFrame) -> dict[str, object]:
    best = best_rows(results)
    best_panel = {
        objective: min(
            (row for row in best if row["objective"] == objective),
            key=lambda row: row["observed_target_bpb"],
        )
        for objective in ("uncheatable", "table9")
    }
    return {
        "completed_training_runs": int(results["training_wandb_state"].eq("finished").sum()),
        "completed_native_table9_evals": int(results["eval_wandb_state"].eq("finished").sum()),
        "best_by_family_policy": best,
        "best_panel": best_panel,
        "prior_frontiers": PRIOR_FRONTIERS,
        "rank_diagnostics": rank_diagnostics(results),
    }


def add_reference_lines(fig: go.Figure, objective: str, col: int) -> None:
    dash_styles = ["dash", "dot"]
    for index, (label, value) in enumerate(PRIOR_FRONTIERS[objective].items()):
        fig.add_hline(
            y=value,
            line={"color": "#23395D", "width": 1.5, "dash": dash_styles[index]},
            annotation_text=label,
            annotation_position="top left" if index == 0 else "bottom right",
            annotation_font_size=12,
            row=1,
            col=col,
        )


def render_plot(results: pd.DataFrame, output_dir: Path) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Uncheatable BPB", "Table-9 51-component macro BPB"],
        horizontal_spacing=0.1,
    )
    for col, objective in enumerate(("uncheatable", "table9"), start=1):
        objective_rows = results[results["objective"].eq(objective)]
        for (family, policy), rows in objective_rows.groupby(["family", "policy"]):
            rows = rows.sort_values("kl_reg")
            fig.add_trace(
                go.Scatter(
                    x=rows["kl_reg"],
                    y=rows["observed_target_bpb"],
                    mode="lines+markers",
                    name=DISPLAY_NAMES[(family, policy)],
                    legendgroup=f"{family}-{policy}",
                    showlegend=col == 1,
                    marker={"size": 9, "color": COLORS[(family, policy)]},
                    line={"width": 2.2, "color": COLORS[(family, policy)]},
                    customdata=np.column_stack(
                        [
                            rows["candidate"],
                            rows["predicted_bpb_300m"],
                            rows["max_simulated_epoch"],
                        ]
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>KL=%{x:g}<br>observed=%{y:.6f}<br>"
                        "predicted@300M=%{customdata[1]:.6f}<br>max epoch=%{customdata[2]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        add_reference_lines(fig, objective, col)
        fig.update_xaxes(title_text="Deployment KL coefficient", row=1, col=col)
        fig.update_yaxes(title_text="BPB (lower is better)", row=1, col=col)
        if objective == "uncheatable":
            fig.update_yaxes(range=[0.984, 1.02], row=1, col=col)
        else:
            fig.update_yaxes(range=[1.06, 1.13], row=1, col=col)
            fig.add_annotation(
                x=0.055,
                y=1.126,
                text="Two fitted-1p low-KL points are off-scale: 1.165 and 1.311",
                showarrow=False,
                xanchor="left",
                font={"size": 12, "color": "#E67E22"},
                row=1,
                col=col,
            )
    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=650,
        title={"text": "Symmetric separate-heads and effective-exposure + geometry at 3e18", "x": 0.5},
        font={"family": "Times New Roman, Times, serif", "size": 16, "color": "#23395D"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.2},
        margin={"l": 80, "r": 45, "t": 90, "b": 130},
    )
    stem = output_dir / "observed_frontier_comparison"
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn", config=EXPORT_CONFIG)
    fig.write_image(stem.with_suffix(".png"), scale=2)


def report_table_rows(results: pd.DataFrame, objective: str) -> list[str]:
    rows = results[results["objective"].eq(objective)].sort_values("observed_target_bpb")
    return [
        "| "
        + " | ".join(
            [
                str(row.candidate),
                str(row.family),
                str(row.policy),
                f"{row.kl_reg:g}",
                f"{row.observed_target_bpb:.6f}",
                f"{row.predicted_bpb_300m:.6f}",
                f"{row.max_simulated_epoch:.3f}",
            ]
        )
        + " |"
        for row in rows.itertuples()
    ]


def write_report(results: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> None:
    best_uncheatable = summary["best_panel"]["uncheatable"]
    best_table9 = summary["best_panel"]["table9"]
    uncheatable_gap_separate = (
        best_uncheatable["observed_target_bpb"] - PRIOR_FRONTIERS["uncheatable"]["separate-heads KL=0.1"]
    )
    uncheatable_gap_controlled = (
        best_uncheatable["observed_target_bpb"] - PRIOR_FRONTIERS["uncheatable"]["best controlled two-phase"]
    )
    lines = [
        "# Symmetric separate-heads and geometry frontier: 3e18 results",
        "",
        "## Coverage",
        "",
        "- 30/30 training runs finished.",
        "- 30/30 Marin-native Table-9 evals finished.",
        "- Every observed row joins one-to-one to the reviewed candidate manifest.",
        "- Runs use distinct data seeds, so policy contrasts are not paired causal estimates.",
        "",
        "## Verdict",
        "",
        (
            f"The best new Table-9 result is `{best_table9['candidate']}` at "
            f"{best_table9['observed_target_bpb']:.6f}. It is only "
            f"{best_table9['observed_target_bpb'] - PRIOR_FRONTIERS['table9']['separate-heads KL=0.1']:+.6f} "
            "from the prior separate-heads frontier, far below the 3e18 Table-9 noise resolution. "
            "It is a one-phase geometry candidate, so it does not validate a stronger two-phase model."
        ),
        (
            f"The best new Uncheatable result is `{best_uncheatable['candidate']}` at "
            f"{best_uncheatable['observed_target_bpb']:.6f}, "
            f"{uncheatable_gap_separate:+.6f} "
            "worse than the prior separate-heads result and "
            f"{uncheatable_gap_controlled:+.6f} "
            "worse than the controlled frontier."
        ),
        (
            "Within the symmetric separate-heads subpanel, Table-9 favors the best fitted two-phase "
            "candidate over the best fitted one-phase candidate by 0.010255 BPB, while Uncheatable "
            "favors one-phase by 0.000675 BPB. These are best-of-sweep, different-seed comparisons, "
            "not paired estimates of a phase-ordering effect."
        ),
        "",
        "The symmetric separately fitted one-phase/two-phase procedure therefore does not establish a new frontier. "
        "The lower-KL effective-exposure + geometry sweep also shows that excessive deployment KL was not the main "
        "reason its two-phase candidates underperformed: relaxing KL improves some candidates, but the best Table-9 "
        "point is one-phase and the best Uncheatable two-phase point remains behind the incumbent.",
        "",
        "## Uncheatable objective-matched ranking",
        "",
        "| Candidate | Family | Policy | KL | Observed BPB | Predicted@300M | Max epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
        *report_table_rows(results, "uncheatable"),
        "",
        "## Table-9 objective-matched ranking",
        "",
        "| Candidate | Family | Policy | KL | Observed BPB | Predicted@300M | Max epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
        *report_table_rows(results, "table9"),
        "",
        "## Modeling implications",
        "",
        (
            "1. The old separate-heads KL=0.1 result remains the Table-9 modeling frontier; "
            "the new nominal tie is one-phase."
        ),
        (
            "2. Symmetric one-phase fitting is unstable for Table-9 at low KL: the 300M model predicts "
            "KL=0.05 as best, but it is the worst deployed point by a large margin."
        ),
        (
            "3. Effective-exposure + geometry still overstates two-phase gains on Table-9. At KL=0.15 "
            "it predicts two-phase well ahead of one-phase, while deployment reverses the ordering by "
            "about 0.025 BPB."
        ),
        (
            "4. Uncheatable retains some local two-phase ordering signal within the geometry family, "
            "but not enough to beat the established frontier."
        ),
        (
            "5. Further scaling of these candidates is not justified. If we want to close the local "
            "regularization question, only two narrow 3e18 probes remain informative: lower geometry "
            "KL for Uncheatable, whose observed trend still improves toward KL=0.2, and lower two-phase "
            "symmetric-separate-heads KL for Table-9. Keep separate heads as the candidate generator and "
            "treat the geometry form as a diagnostic, not a paper-form successor."
        ),
        "",
    ]
    (output_dir / "observed_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.panel_dir / "candidate_manifest.csv")
    results = collect_results(manifest)
    summary = summary_payload(results)
    results.to_csv(args.panel_dir / "observed_results.csv", index=False)
    pd.DataFrame(summary["best_by_family_policy"]).to_csv(
        args.panel_dir / "observed_best_by_family_policy.csv", index=False
    )
    (args.panel_dir / "observed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_plot(results, args.panel_dir)
    write_report(results, summary, args.panel_dir)
    print(args.panel_dir)


if __name__ == "__main__":
    main()
