# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec",
#   "gcsfs",
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Collect the low-epsilon decoupled phase-information validation panel."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.two_phase_many.interactive_mixture_inspector import (
    mixture_inspector_payload,
    mixture_inspector_script,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_PANEL_DIR = REFERENCE_OUTPUTS / "decoupled_phase_information_low_epsilon_validation_panel_20260712"
DEFAULT_BASE_MIXTURE_DIR = REFERENCE_OUTPUTS / "decoupled_phase_information_validation_panel_20260712" / "mixtures"
DEFAULT_LOW_EPSILON_MIXTURE_DIR = DEFAULT_PANEL_DIR / "mixtures"
DEFAULT_PRIOR_RESULTS = (
    REFERENCE_OUTPUTS / "decoupled_phase_information_validation_results_20260712" / "observed_results.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "decoupled_phase_information_low_epsilon_validation_results_20260712"
TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-decoupled-phase-information"
EVAL_GROUP = "olmo_base_eval_table9_decoupled_phase_information"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
RUN_SUFFIX = "_3e18"
NOISE = {"uncheatable": {"difference_sd": 0.001291176543909418}}
FAMILY_COLORS = {
    "effective_exposure": "#FC8D59",
    "separate_heads": "#4575B4",
    "control": "#4D5562",
}
FAMILY_LABELS = {
    "effective_exposure": "Effective-exposure DSP",
    "separate_heads": "Separate heads",
    "control": "Aggregate-matched tied control",
}
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
ANCHOR_LABELS = {
    "unch05": "Uncheatable, aggregate KL=0.05",
    "t9s05": "Table-9, stable aggregate KL=0.05",
    "t9b075": "Table-9, observed-best aggregate KL=0.075",
}
RESULT_URI_PATTERN = re.compile(r"wrote results to (gs://\S+/olmo_base_eval_table9_results\.json)")
PARAMETER_COUNT = 358_306_688
TRAIN_TOKENS = 1_576_534_016
VALIDATION_TPP = TRAIN_TOKENS / PARAMETER_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--prior-results", type=Path, default=DEFAULT_PRIOR_RESULTS)
    parser.add_argument("--base-mixture-dir", type=Path, default=DEFAULT_BASE_MIXTURE_DIR)
    parser.add_argument("--low-epsilon-mixture-dir", type=Path, default=DEFAULT_LOW_EPSILON_MIXTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def unique_finished_run(
    runs: list[wandb.apis.public.Run],
    candidate: str,
    *,
    kind: str,
) -> wandb.apis.public.Run | None:
    if kind == "training":
        matches = [run for run in runs if run.state == "finished" and run.name.startswith(f"{candidate}{RUN_SUFFIX}-")]
    elif kind == "eval":
        expected_name = f"t9_{candidate}{RUN_SUFFIX}"
        matches = [run for run in runs if run.state == "finished" and run.name == expected_name]
    else:
        raise ValueError(f"Unknown run kind: {kind}")
    if len(matches) > 1:
        raise ValueError(f"Expected at most one finished {kind} run for {candidate}, got {len(matches)}")
    return matches[0] if matches else None


def table9_metric_from_run(run: wandb.apis.public.Run | None) -> tuple[float | None, str, str]:
    if run is None:
        return None, "missing", ""
    summary_value = run.summary.get(TABLE9_METRIC)
    if summary_value is not None:
        return float(summary_value), "wandb_summary", ""

    with tempfile.TemporaryDirectory(prefix=f"table9-{run.id}-") as temporary_dir:
        output_log = run.file("output.log").download(root=temporary_dir, replace=True)
        log_text = Path(output_log.name).read_text(errors="replace")
    matches = RESULT_URI_PATTERN.findall(log_text)
    if len(matches) != 1:
        raise ValueError(f"Expected one result URI in output log for {run.name}, got {matches}")
    result_uri = matches[0]
    with fsspec.open(result_uri) as result_file:
        result = json.load(result_file)
    return float(result["table9_macro_bpb"]), "gcs_result_json", result_uri


def collect_results(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    api = wandb.Api(timeout=180)
    training_api_rows = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=200))
    eval_api_rows = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=200))
    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        candidate = str(record["candidate"])
        training_run = unique_finished_run(training_api_rows, candidate, kind="training")
        eval_run = unique_finished_run(eval_api_rows, candidate, kind="eval")
        if eval_run is None:
            expected_name = f"t9_{candidate}{RUN_SUFFIX}"
            direct_eval_rows = list(api.runs(EVAL_PROJECT, filters={"display_name": expected_name}, per_page=20))
            eval_run = unique_finished_run(direct_eval_rows, candidate, kind="eval")
        uncheatable = training_run.summary.get(UNCHEATABLE_METRIC) if training_run is not None else None
        table9, table9_source, table9_result_uri = table9_metric_from_run(eval_run)
        objective = str(record["objective"])
        observed_target = uncheatable if objective == "uncheatable" else table9
        rows.append(
            {
                **record,
                "observed_uncheatable_bpb": float(uncheatable) if uncheatable is not None else np.nan,
                "observed_table9_macro_bpb": float(table9) if table9 is not None else np.nan,
                "observed_target_bpb": float(observed_target) if observed_target is not None else np.nan,
                "training_wandb_name": training_run.name if training_run is not None else "",
                "training_wandb_url": training_run.url if training_run is not None else "",
                "eval_wandb_name": eval_run.name if eval_run is not None else "",
                "eval_wandb_url": eval_run.url if eval_run is not None else "",
                "table9_metric_source": table9_source,
                "table9_result_uri": table9_result_uri,
            }
        )
    results = pd.DataFrame(rows)
    audit = {
        "manifest_rows": len(manifest),
        "finished_training_rows": int(results["training_wandb_name"].ne("").sum()),
        "finished_eval_rows": int(results["eval_wandb_name"].ne("").sum()),
        "uncheatable_target_rows": int(
            (results["objective"].eq("uncheatable") & results["observed_target_bpb"].notna()).sum()
        ),
        "table9_target_rows": int((results["objective"].eq("table9") & results["observed_target_bpb"].notna()).sum()),
        "missing_target_candidates": results.loc[results["observed_target_bpb"].isna(), "candidate"].tolist(),
        "table9_metric_sources": results["table9_metric_source"].value_counts().sort_index().to_dict(),
    }
    return results, audit


def combined_paths(current: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    current = current[current["observed_target_bpb"].notna()].copy()
    prior = prior[
        prior["anchor_tag"].isin(ANCHOR_LABELS)
        & prior["family"].isin(["control", "effective_exposure", "separate_heads"])
    ].copy()
    columns = [
        "candidate",
        "objective",
        "anchor_tag",
        "family",
        "phase_information_budget",
        "phase_tv",
        "predicted_bpb",
        "predicted_gain_vs_tied",
        "observed_target_bpb",
    ]
    combined = pd.concat([prior[columns], current[columns]], ignore_index=True)
    combined = combined.drop_duplicates("candidate", keep="last")
    combined["tied_observed_target_bpb"] = np.nan
    for anchor_tag, anchor_rows in combined.groupby("anchor_tag"):
        tied = anchor_rows[anchor_rows["family"].eq("control")]
        if len(tied) != 1:
            raise ValueError(f"Expected one tied control for {anchor_tag}, got {len(tied)}")
        combined.loc[combined["anchor_tag"].eq(anchor_tag), "tied_observed_target_bpb"] = float(
            tied.iloc[0]["observed_target_bpb"]
        )
    combined["observed_gain_vs_tied"] = combined["tied_observed_target_bpb"] - combined["observed_target_bpb"]
    combined["anchored_predicted_target_bpb"] = combined["tied_observed_target_bpb"] - combined["predicted_gain_vs_tied"]
    combined["prediction_residual"] = combined["observed_target_bpb"] - combined["anchored_predicted_target_bpb"]
    combined["independent_difference_sd_units"] = np.where(
        combined["objective"].eq("uncheatable"),
        combined["observed_gain_vs_tied"] / NOISE["uncheatable"]["difference_sd"],
        np.nan,
    )
    return combined.sort_values(["objective", "anchor_tag", "family", "phase_information_budget"]).reset_index(drop=True)


def path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = paths[paths["anchor_tag"].eq("unch05")]
    tied_bpb = float(paths.loc[paths["family"].eq("control"), "observed_target_bpb"].iloc[0])
    for family, family_rows in paths[~paths["family"].eq("control")].groupby("family"):
        family_rows = family_rows.sort_values("phase_information_budget")
        best = family_rows.loc[family_rows["observed_target_bpb"].idxmin()]
        low_epsilon_rows = family_rows[family_rows["phase_information_budget"].le(0.01)]
        fit_x = np.concatenate([[0.0], low_epsilon_rows["phase_information_budget"].to_numpy(dtype=float)])
        fit_y = np.concatenate([[tied_bpb], low_epsilon_rows["observed_target_bpb"].to_numpy(dtype=float)])
        quadratic = np.polyfit(fit_x, fit_y, 2)
        quadratic_vertex = float(-quadratic[1] / (2 * quadratic[0])) if quadratic[0] > 0 else np.nan
        quadratic_rmse = float(np.sqrt(np.mean(np.square(fit_y - np.polyval(quadratic, fit_x)))))
        lower_rows = low_epsilon_rows[low_epsilon_rows["phase_information_budget"].lt(best["phase_information_budget"])]
        upper_rows = low_epsilon_rows[low_epsilon_rows["phase_information_budget"].gt(best["phase_information_budget"])]
        lower_neighbor = lower_rows.iloc[-1] if not lower_rows.empty else None
        upper_neighbor = upper_rows.iloc[0] if not upper_rows.empty else None
        rows.append(
            {
                "family": family,
                "candidate_count": len(family_rows),
                "best_candidate": best["candidate"],
                "best_epsilon": float(best["phase_information_budget"]),
                "best_phase_tv": float(best["phase_tv"]),
                "best_observed_bpb": float(best["observed_target_bpb"]),
                "best_gain_vs_tied": float(best["observed_gain_vs_tied"]),
                "best_difference_sd_units": float(best["independent_difference_sd_units"]),
                "lower_neighbor_margin": (
                    float(lower_neighbor["observed_target_bpb"] - best["observed_target_bpb"])
                    if lower_neighbor is not None
                    else np.nan
                ),
                "upper_neighbor_margin": (
                    float(upper_neighbor["observed_target_bpb"] - best["observed_target_bpb"])
                    if upper_neighbor is not None
                    else np.nan
                ),
                "descriptive_quadratic_vertex": quadratic_vertex,
                "descriptive_quadratic_rmse": quadratic_rmse,
                "best_is_smallest_positive_epsilon": bool(
                    best["phase_information_budget"]
                    == family_rows.loc[family_rows["phase_information_budget"].gt(0), "phase_information_budget"].min()
                ),
                "best_is_largest_epsilon": bool(
                    best["phase_information_budget"] == family_rows["phase_information_budget"].max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("best_observed_bpb").reset_index(drop=True)


def complete_path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    low_epsilon = paths[paths["phase_information_budget"].le(0.01)]
    for (objective, anchor_tag, family), family_rows in low_epsilon[~low_epsilon["family"].eq("control")].groupby(
        ["objective", "anchor_tag", "family"]
    ):
        family_rows = family_rows.sort_values("phase_information_budget")
        observed_best = family_rows.loc[family_rows["observed_target_bpb"].idxmin()]
        predicted_best = family_rows.loc[family_rows["anchored_predicted_target_bpb"].idxmin()]
        rows.append(
            {
                "objective": objective,
                "anchor_tag": anchor_tag,
                "family": family,
                "n": len(family_rows),
                "observed_best_epsilon": float(observed_best["phase_information_budget"]),
                "observed_best_bpb": float(observed_best["observed_target_bpb"]),
                "observed_best_gain_vs_tied": float(observed_best["observed_gain_vs_tied"]),
                "predicted_best_epsilon": float(predicted_best["phase_information_budget"]),
                "predicted_best_anchored_bpb": float(predicted_best["anchored_predicted_target_bpb"]),
                "predicted_best_gain_vs_tied": float(predicted_best["predicted_gain_vs_tied"]),
                "predicted_vs_observed_spearman": float(
                    family_rows[["anchored_predicted_target_bpb", "observed_target_bpb"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "anchored_prediction_rmse": float(np.sqrt(np.mean(np.square(family_rows["prediction_residual"])))),
            }
        )
    return pd.DataFrame(rows).sort_values(["objective", "anchor_tag", "family"]).reset_index(drop=True)


def add_anchor_traces(
    figure: go.Figure,
    paths: pd.DataFrame,
    anchor_tag: str,
    *,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool,
    max_epsilon: float | None = None,
) -> None:
    anchor_rows = paths[paths["anchor_tag"].eq(anchor_tag)]
    if max_epsilon is not None:
        anchor_rows = anchor_rows[anchor_rows["phase_information_budget"].le(max_epsilon)]
    control = anchor_rows[anchor_rows["family"].eq("control")].iloc[0]

    def add(trace: go.Scatter) -> None:
        if row is None or col is None:
            figure.add_trace(trace)
        else:
            figure.add_trace(trace, row=row, col=col)

    for family in ("effective_exposure", "separate_heads"):
        family_rows = anchor_rows[anchor_rows["family"].eq(family)].sort_values("phase_information_budget")
        x = np.concatenate([[0.0], family_rows["phase_information_budget"].to_numpy(float)])
        observed = np.concatenate([[float(control["observed_target_bpb"])], family_rows["observed_target_bpb"]])
        predicted = np.concatenate(
            [[float(control["observed_target_bpb"])], family_rows["anchored_predicted_target_bpb"]]
        )
        candidates = np.concatenate([[str(control["candidate"])], family_rows["candidate"].astype(str)])
        observed_gain = np.concatenate([[0.0], family_rows["observed_gain_vs_tied"]])
        predicted_gain = np.concatenate([[0.0], family_rows["predicted_gain_vs_tied"]])
        phase_tv = np.concatenate([[0.0], family_rows["phase_tv"]])
        raw_predicted = np.concatenate([[np.nan], family_rows["predicted_bpb"]])
        add(
            go.Scatter(
                x=x,
                y=observed,
                mode="lines+markers",
                name=f"{FAMILY_LABELS[family]} observed",
                legendgroup=family,
                showlegend=show_legend,
                line={"color": FAMILY_COLORS[family], "width": 3},
                marker={"size": 10},
                customdata=np.stack(
                    [
                        candidates,
                        np.full(len(candidates), VALIDATION_TPP),
                        observed_gain,
                        predicted_gain,
                        phase_tv,
                        raw_predicted,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>epsilon=%{x:.4f}<br>observed BPB=%{y:.6f}"
                    "<br>observed gain vs tied=%{customdata[2]:+.6f}"
                    "<br>predicted gain vs tied=%{customdata[3]:+.6f}"
                    "<br>phase TV=%{customdata[4]:.4f}<br>raw 300M prediction=%{customdata[5]:.6f}<extra></extra>"
                ),
            )
        )
        add(
            go.Scatter(
                x=x,
                y=predicted,
                mode="lines+markers",
                name=f"{FAMILY_LABELS[family]} predicted",
                legendgroup=family,
                showlegend=show_legend,
                line={"color": FAMILY_COLORS[family], "width": 2, "dash": "dash"},
                marker={"size": 9, "symbol": "circle-open"},
                customdata=np.stack(
                    [
                        candidates,
                        np.full(len(candidates), VALIDATION_TPP),
                        observed,
                        observed_gain,
                        predicted_gain,
                        phase_tv,
                        raw_predicted,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>epsilon=%{x:.4f}<br>anchored predicted BPB=%{y:.6f}"
                    "<br>observed BPB=%{customdata[2]:.6f}<br>observed gain vs tied=%{customdata[3]:+.6f}"
                    "<br>predicted gain vs tied=%{customdata[4]:+.6f}"
                    "<br>phase TV=%{customdata[5]:.4f}<br>raw 300M prediction=%{customdata[6]:.6f}<extra></extra>"
                ),
            )
        )
    add(
        go.Scatter(
            x=[0.0],
            y=[control["observed_target_bpb"]],
            mode="markers",
            name=FAMILY_LABELS["control"],
            legendgroup="control",
            showlegend=show_legend,
            marker={"color": FAMILY_COLORS["control"], "size": 12, "symbol": "diamond"},
            customdata=[[control["candidate"], VALIDATION_TPP]],
            hovertemplate="%{customdata[0]}<br>epsilon=0<br>observed BPB=%{y:.6f}<extra></extra>",
        )
    )


def add_fact_sheet(figure: go.Figure) -> None:
    columns = (
        (
            ("Surrogate fit panel", "300M / 6B-token Dolma 3 + Dolmino swarm"),
            ("Fit rows", "280 two-phase schedules; 39 top-level buckets"),
            ("Models", "effective-exposure DSP and separate phase heads"),
        ),
        (
            ("Intervention", "hold aggregate mixture fixed; vary phase-information budget"),
            ("Phase-information grid", "0.001-0.2; sampled most densely near zero"),
            ("Aggregate anchors", "KL 0.05; plus Table-9 observed-best KL 0.075"),
        ),
        (
            ("Architecture", "Qwen3: 10 layers, d=896, FFN=3,584, 7 Q/KV heads"),
            ("Parameters / tokens", "358.3M trainable; 1.576B materialized tokens"),
            ("Optimization", "AdamH; sequence 4,096; batch 128; 3,007 steps; seed 690300"),
        ),
        (
            ("Phases / LR", "80% / 20% boundary at step 2,406; 10% warmup, 20% linear decay"),
            ("Simulated epoch target", "6.325T tokens; no fixed subset seed"),
            ("Evidence", "one training seed per candidate; no candidate repeats"),
            ("Prediction display", "300M curves offset to the observed tied 3e18 control"),
        ),
    )
    figure.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.0,
        x1=1.0,
        y0=-0.46,
        y1=-0.18,
        fillcolor="#F5F1E8",
        line={"color": "#C7BFB0", "width": 1},
        layer="below",
    )
    figure.add_annotation(
        x=0.01,
        y=-0.22,
        xref="paper",
        yref="paper",
        text="<b>EXPERIMENT FACT SHEET</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"family": "Arial, sans-serif", "size": 14, "color": "#C94F2D"},
    )
    for index, facts in enumerate(columns):
        text = "<br>".join(f"<b>{label}</b>  {value}" for label, value in facts)
        figure.add_annotation(
            x=0.01 + index * 0.247,
            y=-0.29,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"family": "Arial, sans-serif", "size": 11, "color": "#23395D"},
        )


def mixture_inspector_post_script(
    paths: pd.DataFrame,
    base_mixture_dir: Path,
    low_epsilon_mixture_dir: Path,
) -> str:
    displayed = paths[paths["phase_information_budget"].le(0.2)].drop_duplicates("candidate")
    mixture_paths: dict[str, Path] = {}
    labels: dict[str, str] = {}
    for record in displayed.to_dict(orient="records"):
        candidate = str(record["candidate"])
        matches = [
            path
            for path in (base_mixture_dir / f"{candidate}.csv", low_epsilon_mixture_dir / f"{candidate}.csv")
            if path.exists()
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one local mixture CSV for {candidate}, got {matches}")
        mixture_paths[candidate] = matches[0]

        anchor = ANCHOR_LABELS[str(record["anchor_tag"])]
        family = FAMILY_LABELS[str(record["family"])]
        if record["family"] == "control":
            labels[candidate] = f"{anchor} · {family}"
        else:
            epsilon = float(record["phase_information_budget"])
            labels[candidate] = f"{anchor} · {family} · <i>ε</i><sub>phase</sub> = {epsilon:g}"

    payload = mixture_inspector_payload(mixture_paths, labels)
    return mixture_inspector_script(payload, parameter_count=PARAMETER_COUNT)


def render_paths(
    paths: pd.DataFrame,
    base_mixture_dir: Path,
    low_epsilon_mixture_dir: Path,
    output_dir: Path,
) -> None:
    figure = go.Figure()
    add_anchor_traces(figure, paths, "unch05", show_legend=True, max_epsilon=0.01)
    figure.update_layout(
        title={"text": "Uncheatable BPB: predicted versus observed low-epsilon path", "x": 0.5},
        template="plotly_white",
        width=1100,
        height=700,
        xaxis_title="Phase-information budget epsilon_phase",
        yaxis_title="eval/uncheatable_eval/bpb (lower is better)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "center", "x": 0.5},
        margin={"l": 90, "r": 40, "t": 130, "b": 90},
    )
    figure.write_html(
        output_dir / "uncheatable_low_epsilon_observed_paths.html", include_plotlyjs=True, config=EXPORT_CONFIG
    )
    figure.write_image(output_dir / "uncheatable_low_epsilon_observed_paths.png", scale=2)

    comparison = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[ANCHOR_LABELS[anchor] for anchor in ANCHOR_LABELS],
        horizontal_spacing=0.07,
    )
    for col, anchor_tag in enumerate(ANCHOR_LABELS, start=1):
        add_anchor_traces(
            comparison,
            paths,
            anchor_tag,
            row=1,
            col=col,
            show_legend=col == 1,
        )
        comparison.update_xaxes(
            title_text="Phase-information budget",
            tickmode="array",
            tickvals=[0.0, 0.05, 0.1, 0.15, 0.2],
            row=1,
            col=col,
        )
        comparison.update_yaxes(title_text="Target BPB" if col == 1 else None, row=1, col=col)
    add_fact_sheet(comparison)
    comparison.update_layout(
        title={
            "text": (
                "Phase-information paths: predicted versus observed at 3e18"
                "<br><sup>Separate-heads epsilon=0.2 duplicates epsilon=0.15 for Uncheatable and stable Table-9; "
                "it was not rerun.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        width=1800,
        height=900,
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "center", "x": 0.5},
        margin={"l": 80, "r": 40, "t": 135, "b": 310},
    )
    comparison.write_html(
        output_dir / "low_epsilon_predicted_vs_observed_paths.html",
        include_plotlyjs=True,
        config=EXPORT_CONFIG,
        post_script=mixture_inspector_post_script(paths, base_mixture_dir, low_epsilon_mixture_dir),
    )
    comparison.write_image(output_dir / "low_epsilon_predicted_vs_observed_paths.png", scale=2)

    full_comparison = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[ANCHOR_LABELS[anchor] for anchor in ANCHOR_LABELS],
        horizontal_spacing=0.07,
    )
    for col, anchor_tag in enumerate(ANCHOR_LABELS, start=1):
        add_anchor_traces(full_comparison, paths, anchor_tag, row=1, col=col, show_legend=col == 1)
        full_comparison.update_xaxes(title_text="Phase-information budget", row=1, col=col)
        full_comparison.update_yaxes(title_text="Target BPB" if col == 1 else None, row=1, col=col)
    full_comparison.update_layout(
        title={
            "text": (
                "Full phase-information paths: predicted versus observed at 3e18"
                "<br><sup>Separate-heads epsilon=0.2 duplicates epsilon=0.15 for Uncheatable and stable Table-9; "
                "it was not rerun.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        width=1800,
        height=720,
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "center", "x": 0.5},
        margin={"l": 80, "r": 40, "t": 135, "b": 175},
    )
    full_comparison.write_html(
        output_dir / "predicted_vs_observed_paths.html", include_plotlyjs=True, config=EXPORT_CONFIG
    )
    full_comparison.write_image(output_dir / "predicted_vs_observed_paths.png", scale=2)


def write_report(
    results: pd.DataFrame,
    paths: pd.DataFrame,
    summary: pd.DataFrame,
    complete_summary: pd.DataFrame,
    audit: dict[str, object],
    output_dir: Path,
) -> None:
    best = summary.iloc[0]
    lower_margin_sd = best["lower_neighbor_margin"] / NOISE["uncheatable"]["difference_sd"]
    upper_margin_sd = best["upper_neighbor_margin"] / NOISE["uncheatable"]["difference_sd"]
    report = f"""# Low-epsilon decoupled phase-information results

## Current coverage

- Finished training rows: {audit['finished_training_rows']}/{audit['manifest_rows']}.
- Uncheatable target rows: {audit['uncheatable_target_rows']}/6.
- Finished native Table-9 eval rows: {audit['finished_eval_rows']}/{audit['manifest_rows']}.
- Table-9 target rows: {audit['table9_target_rows']}/12.
- Table-9 metric sources: {audit['table9_metric_sources']}.

## Uncheatable result

The best observed path point is `{best['best_candidate']}` from `{best['family']}` at
epsilon_phase={best['best_epsilon']:.4f}: BPB={best['best_observed_bpb']:.6f}. It improves the exact
aggregate-matched tied control by {best['best_gain_vs_tied']:.6f} BPB, or
{best['best_difference_sd_units']:.2f} independent-run difference SDs.

The discrete minimum is bracketed by epsilon=0.0025 and 0.0075. Their BPBs are respectively
{best['lower_neighbor_margin']:.6f} and {best['upper_neighbor_margin']:.6f} above the epsilon=0.005 point, or
{lower_margin_sd:.2f} and {upper_margin_sd:.2f} independent-run difference SDs. A descriptive quadratic fit over
epsilon in [0,0.01] has its vertex at {best['descriptive_quadratic_vertex']:.6f}, but this is not an inferential optimum:
the fit residual RMSE is {best['descriptive_quadratic_rmse']:.6f} BPB and all candidates have only one training seed.

Therefore epsilon=0.005 is the best tested deployment value and the low-epsilon sweep brackets it geometrically, but
the continuous optimum is not statistically identified. Resolve it with paired repeats of the tied control and
epsilon in {{0.001, 0.005, 0.0075}} before treating the location as fixed.

The 300M surrogate predicts monotone improvement as epsilon grows over this range. The 3e18 path instead turns back
up after epsilon=0.005, so the surrogate identifies a useful phase direction but overstates how far to move along it.

## Table-9 result

The stable aggregate (KL=0.05) has a noisy interior improvement: effective exposure is best at epsilon=0.01
(1.056877 BPB), while separate heads is best at epsilon=0.005 (1.060932 BPB). The aggregate selected from the earlier
observed-best path (KL=0.075) does not benefit from phase asymmetry: every tested effective-exposure and separate-heads
point through epsilon=0.01 is worse than its tied control (1.057530 BPB). These observations contradict the surrogate's
monotone predicted gains and show that the aggregate anchor and phase-ordering move cannot be optimized independently.

Because the 300M and 3e18 intercepts differ, the plot offsets each predicted path to the observed tied-control BPB and
compares predicted versus observed gains. It does not treat the raw 300M predicted BPB as a calibrated 3e18 value.

## Family summaries

{summary.to_markdown(index=False, floatfmt='.6f')}

## Predicted-versus-observed path summary

{complete_summary.to_markdown(index=False, floatfmt='.6f')}
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.panel_dir / "selected_candidate_manifest.csv")
    prior = pd.read_csv(args.prior_results)
    results, audit = collect_results(manifest)
    paths = combined_paths(results, prior)
    summary = path_summary(paths)
    complete_summary = complete_path_summary(paths)
    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    paths.to_csv(args.output_dir / "combined_uncheatable_paths.csv", index=False)
    summary.to_csv(args.output_dir / "uncheatable_path_summary.csv", index=False)
    complete_summary.to_csv(args.output_dir / "predicted_vs_observed_path_summary.csv", index=False)
    (args.output_dir / "collection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    render_paths(paths, args.base_mixture_dir, args.low_epsilon_mixture_dir, args.output_dir)
    write_report(results, paths, summary, complete_summary, audit, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
