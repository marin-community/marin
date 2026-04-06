# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEED_PATTERN = re.compile(r"_seed(\d+)")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    row: int
    col: int


@dataclass(frozen=True)
class RunRecord:
    name: str
    beta: float
    learning_rate: float
    seed: int
    history_path: Path
    history_format: str
    summary: dict[str, object]


METRICS = (
    MetricSpec("train/dpo_loss", "Train DPO Loss · Log Scale", 1, 1),
    MetricSpec("eval/bloom_speceval_v2_val/dpo_loss", "Eval DPO Loss · Log Scale", 1, 2),
    MetricSpec("eval/bloom_speceval_v2_val/dpo_accuracy", "Eval DPO Accuracy", 2, 1),
    MetricSpec("eval/bloom_speceval_v2_val/dpo_margin_policy", "Eval Policy Margin", 2, 2),
)
FINAL_EVAL_ACCURACY = "eval/bloom_speceval_v2_val/dpo_accuracy"
FINAL_EVAL_LOSS = "eval/bloom_speceval_v2_val/dpo_loss"
FINAL_EVAL_MARGIN = "eval/bloom_speceval_v2_val/dpo_margin_policy"
GROUP_COLORS = {
    "OG No LoRA": "#5B6577",
    "Tinker Recommended": "#E07A26",
    "Best Eval": "#1C8C7C",
}
TRAIN_SMOOTHING_WINDOW = 31


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_output_dir = root / "scratch" / "wandb_dpo_data" / "plots"
    parser = argparse.ArgumentParser(
        description="Plot beta=0.1 OG no-LoRA vs tune-LoRA DPO comparisons from local W&B exports."
    )
    parser.add_argument(
        "--og-dir",
        type=Path,
        default=root / "scratch" / "wandb_dpo_data" / "og_no_lora",
        help="Directory containing the archived OG no-LoRA runs.",
    )
    parser.add_argument(
        "--tune-dir",
        type=Path,
        default=root / "scratch" / "wandb_dpo_data" / "tune_lora",
        help="Directory containing the archived tune-LoRA runs.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=default_output_dir / "beta0p1_no_lora_vs_tune_lora.html",
        help="Path for the interactive HTML plot.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_output_dir / "beta0p1_no_lora_vs_tune_lora_selection.json",
        help="Path for the selection summary JSON.",
    )
    return parser.parse_args()


def parse_seed(run_name: str) -> int:
    match = SEED_PATTERN.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse seed from run name: {run_name}")
    return int(match.group(1))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def load_old_runs(base_dir: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config = load_json(run_dir / "config.json")
        summary = load_json(run_dir / "summary.json")
        learning_rate = float(config["optimizer"]["learning_rate"])
        runs.append(
            RunRecord(
                name=run_dir.name,
                beta=float(config["beta"]),
                learning_rate=learning_rate,
                seed=parse_seed(run_dir.name),
                history_path=run_dir / "history.csv",
                history_format="csv",
                summary=summary,
            )
        )
    return runs


def load_tune_runs(base_dir: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "history.jsonl.gz"
        if not config_path.exists() or not summary_path.exists() or not history_path.exists():
            continue
        config = load_json(config_path)
        summary = load_json(summary_path)
        learning_rate = float(config["optimizer"]["learning_rate"])
        runs.append(
            RunRecord(
                name=run_dir.name,
                beta=float(config["beta"]),
                learning_rate=learning_rate,
                seed=parse_seed(run_dir.name),
                history_path=history_path,
                history_format="jsonl.gz",
                summary=summary,
            )
        )
    return runs


def load_history(run: RunRecord) -> pd.DataFrame:
    columns = ["_step", *[metric.key for metric in METRICS]]
    if run.history_format == "csv":
        frame = pd.read_csv(run.history_path, usecols=lambda column: column in columns)
    elif run.history_format == "jsonl.gz":
        rows = []
        with gzip.open(run.history_path, "rt") as handle:
            for line in handle:
                payload = json.loads(line)
                rows.append({column: payload.get(column) for column in columns})
        frame = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported history format: {run.history_format}")

    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.rename(columns={"_step": "step"}).sort_values("step")
    frame["seed"] = run.seed
    return frame


def aggregate_group(label: str, runs: list[RunRecord]) -> pd.DataFrame:
    history = pd.concat([load_history(run) for run in runs], ignore_index=True)
    aggregations = {metric.key: ["mean", "std", "count"] for metric in METRICS}
    grouped = history.groupby("step", sort=True).agg(aggregations)
    grouped.columns = [f"{metric}__{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped["group"] = label
    return grouped


def select_runs(
    old_runs: list[RunRecord], tune_runs: list[RunRecord]
) -> tuple[list[tuple[str, list[RunRecord]]], pd.DataFrame, float, float, float]:
    og_beta_runs = [run for run in old_runs if math.isclose(run.beta, 0.1)]
    if not og_beta_runs:
        raise ValueError("No OG no-LoRA beta=0.1 runs found.")

    og_learning_rates = sorted({run.learning_rate for run in og_beta_runs})
    if len(og_learning_rates) != 1:
        raise ValueError(f"Expected one OG baseline learning rate, found {og_learning_rates}")

    baseline_lr = og_learning_rates[0]
    tinker_lr = baseline_lr * 10
    tune_beta_runs = [run for run in tune_runs if math.isclose(run.beta, 0.1)]
    if not tune_beta_runs:
        raise ValueError("No tune-LoRA beta=0.1 runs found.")

    ranking = pd.DataFrame(
        [
            {
                "learning_rate": run.learning_rate,
                "seed": run.seed,
                "final_eval_accuracy": run.summary.get(FINAL_EVAL_ACCURACY),
                "final_eval_loss": run.summary.get(FINAL_EVAL_LOSS),
                "final_eval_margin": run.summary.get(FINAL_EVAL_MARGIN),
            }
            for run in tune_beta_runs
        ]
    )
    ranking = (
        ranking.groupby("learning_rate", as_index=False)
        .agg(
            mean_final_eval_accuracy=("final_eval_accuracy", "mean"),
            mean_final_eval_loss=("final_eval_loss", "mean"),
            mean_final_eval_margin=("final_eval_margin", "mean"),
            num_seeds=("seed", "nunique"),
        )
        .sort_values(
            ["mean_final_eval_accuracy", "mean_final_eval_loss", "mean_final_eval_margin"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )

    tinker_runs = [run for run in tune_beta_runs if math.isclose(run.learning_rate, tinker_lr)]
    if not tinker_runs:
        raise ValueError(f"No tune-LoRA runs found for 10x baseline lr={tinker_lr:.2e}.")

    best_lr = float(ranking.iloc[0]["learning_rate"])
    best_runs = [run for run in tune_beta_runs if math.isclose(run.learning_rate, best_lr)]

    selections = [
        ("OG No LoRA", sorted(og_beta_runs, key=lambda run: run.seed)),
        ("Tinker Recommended", sorted(tinker_runs, key=lambda run: run.seed)),
        ("Best Eval", sorted(best_runs, key=lambda run: run.seed)),
    ]
    return selections, ranking, baseline_lr, tinker_lr, best_lr


def hex_to_rgba(color: str, alpha: float) -> str:
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def display_label(label: str, learning_rate: float) -> str:
    return f"{label} · lr {learning_rate:.1e}"


def plot_groups(
    aggregated: list[pd.DataFrame],
    baseline_lr: float,
    tinker_lr: float,
    best_lr: float,
    output_html: Path,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[metric.title for metric in METRICS],
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )
    learning_rates = {
        "OG No LoRA": baseline_lr,
        "Tinker Recommended": tinker_lr,
        "Best Eval": best_lr,
    }

    for metric_index, metric in enumerate(METRICS):
        for group in aggregated:
            label = group["group"].iloc[0]
            color = GROUP_COLORS[label]
            legend_label = display_label(label, learning_rates[label])
            mean_key = f"{metric.key}__mean"
            std_key = f"{metric.key}__std"
            count_key = f"{metric.key}__count"
            subset = group[group[count_key] > 0][["step", mean_key, std_key]].copy()
            subset[std_key] = subset[std_key].fillna(0.0)
            if metric.key == FINAL_EVAL_ACCURACY:
                subset = subset[subset["step"] > 0]
            subset = subset.dropna(subset=[mean_key])
            if subset.empty:
                continue
            if metric.key == "train/dpo_loss":
                subset[mean_key] = (
                    subset[mean_key].rolling(window=TRAIN_SMOOTHING_WINDOW, center=True, min_periods=1).mean()
                )
                subset[std_key] = (
                    subset[std_key].rolling(window=TRAIN_SMOOTHING_WINDOW, center=True, min_periods=1).mean()
                )
            is_eval_metric = metric.key.startswith("eval/")
            line_style = {"color": color, "width": 3}
            mode = "lines"
            marker = None
            band_alpha = 0.12
            if is_eval_metric:
                mode = "lines+markers"
                line_style = {"color": color, "width": 2.5}
                marker = {
                    "size": 8,
                    "color": color,
                    "line": {"color": "#FFFDF8", "width": 1.5},
                }
                band_alpha = 0.08
            upper_band = subset[mean_key] + subset[std_key]
            lower_band = subset[mean_key] - subset[std_key]
            if metric.row == 1:
                upper_band = upper_band.clip(lower=1e-6)
                lower_band = lower_band.clip(lower=1e-6)

            figure.add_trace(
                go.Scatter(
                    x=subset["step"],
                    y=upper_band,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=metric.row,
                col=metric.col,
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["step"],
                    y=lower_band,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=hex_to_rgba(color, band_alpha),
                    hoverinfo="skip",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=metric.row,
                col=metric.col,
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["step"],
                    y=subset[mean_key],
                    mode=mode,
                    line=line_style,
                    marker=marker,
                    name=legend_label,
                    legendgroup=label,
                    showlegend=metric_index == 0,
                    hovertemplate=f"{legend_label}<br>step=%{{x}}<br>{metric.title}=%{{y:.6f}}<extra></extra>",
                ),
                row=metric.row,
                col=metric.col,
            )

    figure.add_vline(x=849, line_dash="dot", line_color="#A0A8B8", row="all", col="all")
    figure.update_layout(
        title={
            "text": (
                "<span style='font-size:13px; letter-spacing:0.16em; color:#8B6B4A;'>LOCAL W&B ARCHIVE</span><br>"
                "<span style='font-size:31px;'>Bloom SpecEval v2 DPO Curves</span><br>"
                "<span style='font-size:18px; color:#4B5563;'>beta=0.1 baseline against two tune-LoRA selections</span>"
            ),
            "x": 0.02,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        template="plotly_white",
        paper_bgcolor="#F7F3EA",
        plot_bgcolor="#FFFDF8",
        font={"family": "Georgia, Times New Roman, serif", "size": 15, "color": "#1F2937"},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.045,
            "yanchor": "bottom",
            "bgcolor": "rgba(255, 253, 248, 0.82)",
            "bordercolor": "#E5DED0",
            "borderwidth": 1,
        },
        hovermode="x unified",
        margin={"l": 70, "r": 40, "t": 180, "b": 105},
        height=900,
        width=1400,
    )
    figure.add_annotation(
        x=0,
        y=-0.14,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            f"OG baseline: 3 seeds at {baseline_lr:.1e}. "
            f"Tinker Recommended: 10x baseline at {tinker_lr:.1e}. "
            f"Best Eval: top final beta=0.1 tune-LoRA setting at {best_lr:.1e}. "
            "Dotted line marks the OG stop at step 849."
        ),
        font={"size": 13, "color": "#6B7280"},
    )
    figure.update_xaxes(
        title_text="Train Step",
        gridcolor="#E5E7EB",
        zeroline=False,
        showline=True,
        linecolor="#D6D3D1",
    )
    figure.update_yaxes(gridcolor="#E5E7EB", zeroline=False, showline=True, linecolor="#D6D3D1")
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_yaxes(range=[-4, 0], row=1, col=1)
    figure.update_yaxes(range=[-2.4, 0], row=1, col=2)
    figure.update_yaxes(tickformat=".2%", row=2, col=1)
    figure.update_yaxes(range=[0.988, 1.0005], row=2, col=1)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_html, include_plotlyjs="cdn")


def write_selection_summary(
    output_json: Path,
    selections: list[tuple[str, list[RunRecord]]],
    ranking: pd.DataFrame,
    baseline_lr: float,
    tinker_lr: float,
    best_lr: float,
) -> None:
    payload = {
        "baseline_learning_rate": baseline_lr,
        "tinker_recommended_learning_rate": tinker_lr,
        "best_eval_learning_rate": best_lr,
        "selection_rule": {
            "tinker_recommended": "10x the OG no-LoRA beta=0.1 learning rate",
            "best_eval": "highest mean final eval accuracy, tie-break lowest mean final eval loss",
        },
        "groups": [
            {
                "label": label,
                "runs": [
                    {
                        "name": run.name,
                        "beta": run.beta,
                        "learning_rate": run.learning_rate,
                        "seed": run.seed,
                    }
                    for run in runs
                ],
            }
            for label, runs in selections
        ],
        "tune_lora_ranking": ranking.to_dict(orient="records"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    old_runs = load_old_runs(args.og_dir)
    tune_runs = load_tune_runs(args.tune_dir)
    selections, ranking, baseline_lr, tinker_lr, best_lr = select_runs(old_runs, tune_runs)
    aggregated = [aggregate_group(label, runs) for label, runs in selections]
    plot_groups(aggregated, baseline_lr, tinker_lr, best_lr, args.output_html)
    write_selection_summary(
        args.output_json,
        selections,
        ranking,
        baseline_lr,
        tinker_lr,
        best_lr,
    )
    print(f"Wrote plot to {args.output_html}")
    print(f"Wrote selection summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
