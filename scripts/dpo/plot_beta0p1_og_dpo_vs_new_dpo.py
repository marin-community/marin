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
TRAIN_LOSS_KEY = "train/dpo_loss"
EVAL_ACCURACY_KEY = "eval/bloom_speceval_v2_val/dpo_accuracy"
PROGRESS_DECIMALS = 4
TRAIN_SMOOTHING_FRACTION = 0.02


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
    train_batch_size: int
    num_train_steps: int


@dataclass(frozen=True)
class GroupSelection:
    label: str
    runs: list[RunRecord]
    learning_rate: float
    train_batch_size: int
    num_train_steps: int


METRICS = (
    MetricSpec(TRAIN_LOSS_KEY, "Train DPO Loss · Log Scale", 1, 1),
    MetricSpec("eval/bloom_speceval_v2_val/dpo_loss", "Eval DPO Loss · Log Scale", 1, 2),
    MetricSpec(EVAL_ACCURACY_KEY, "Eval DPO Accuracy", 2, 1),
    MetricSpec("eval/bloom_speceval_v2_val/dpo_margin_policy", "Eval Policy Margin", 2, 2),
)

GROUP_COLORS = {
    "OG DPO": "#5B6577",
    "New DPO": "#1C8C7C",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_output_dir = root / "scratch" / "wandb_dpo_data" / "plots"
    parser = argparse.ArgumentParser(
        description="Plot beta=0.1 OG full-DPO vs new full-DPO comparisons from local W&B exports."
    )
    parser.add_argument(
        "--og-dir",
        type=Path,
        default=root / "scratch" / "wandb_dpo_data" / "og_no_lora",
        help="Directory containing the archived OG DPO runs.",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=root / "scratch" / "wandb_dpo_data" / "new_dpo",
        help="Directory containing the archived new DPO runs.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=default_output_dir / "beta0p1_og_dpo_vs_new_dpo.html",
        help="Path for the interactive HTML plot.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_output_dir / "beta0p1_og_dpo_vs_new_dpo_selection.json",
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


def load_runs(base_dir: Path, history_format: str) -> list[RunRecord]:
    runs: list[RunRecord] = []
    history_name = "history.csv" if history_format == "csv" else "history.jsonl.gz"
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir() or "_seed" not in run_dir.name:
            continue
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        history_path = run_dir / history_name
        if not config_path.exists() or not summary_path.exists() or not history_path.exists():
            continue
        config = load_json(config_path)
        beta = float(config["beta"])
        if not math.isclose(beta, 0.1):
            continue
        trainer = config["trainer"]
        optimizer = config["optimizer"]
        runs.append(
            RunRecord(
                name=run_dir.name,
                beta=beta,
                learning_rate=float(optimizer["learning_rate"]),
                seed=parse_seed(run_dir.name),
                history_path=history_path,
                history_format=history_format,
                summary=load_json(summary_path),
                train_batch_size=int(trainer["train_batch_size"]),
                num_train_steps=int(trainer["num_train_steps"]),
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


def build_group(label: str, runs: list[RunRecord]) -> GroupSelection:
    if not runs:
        raise ValueError(f"No runs found for {label}.")

    learning_rates = {run.learning_rate for run in runs}
    train_batch_sizes = {run.train_batch_size for run in runs}
    num_train_steps = {run.num_train_steps for run in runs}
    seeds = sorted(run.seed for run in runs)

    if len(learning_rates) != 1:
        raise ValueError(f"Expected one learning rate for {label}, found {sorted(learning_rates)}")
    if len(train_batch_sizes) != 1:
        raise ValueError(f"Expected one batch size for {label}, found {sorted(train_batch_sizes)}")
    if len(num_train_steps) != 1:
        raise ValueError(f"Expected one step count for {label}, found {sorted(num_train_steps)}")
    if seeds != [0, 1, 2]:
        raise ValueError(f"Expected seeds [0, 1, 2] for {label}, found {seeds}")

    return GroupSelection(
        label=label,
        runs=sorted(runs, key=lambda run: run.seed),
        learning_rate=next(iter(learning_rates)),
        train_batch_size=next(iter(train_batch_sizes)),
        num_train_steps=next(iter(num_train_steps)),
    )


def progress_percent(frame: pd.DataFrame) -> pd.Series:
    max_step = int(frame["step"].max())
    if max_step <= 0:
        raise ValueError("Expected positive max step in run history.")
    return (100.0 * frame["step"] / max_step).round(PROGRESS_DECIMALS)


def aggregate_group(group: GroupSelection) -> pd.DataFrame:
    run_frames: list[pd.DataFrame] = []
    for run in group.runs:
        frame = load_history(run).copy()
        frame["progress_pct"] = progress_percent(frame)
        run_frames.append(frame)

    history = pd.concat(run_frames, ignore_index=True)
    aggregations: dict[str, list[str]] = {"step": ["mean", "min", "max", "count"]}
    aggregations.update({metric.key: ["mean", "std", "count"] for metric in METRICS})
    grouped = history.groupby("progress_pct", sort=True).agg(aggregations)
    grouped.columns = [f"{metric}__{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped["group"] = group.label
    return grouped


def hex_to_rgba(color: str, alpha: float) -> str:
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def legend_label(group: GroupSelection) -> str:
    return f"{group.label} · b{group.train_batch_size} · {group.num_train_steps} steps"


def smoothing_window(num_points: int) -> int:
    if num_points <= 2:
        return 1
    window = max(5, round(num_points * TRAIN_SMOOTHING_FRACTION))
    if window % 2 == 0:
        window += 1
    if window > num_points:
        window = num_points if num_points % 2 == 1 else num_points - 1
    return max(window, 1)


def accuracy_axis_range(aggregated: list[pd.DataFrame]) -> list[float]:
    mean_key = f"{EVAL_ACCURACY_KEY}__mean"
    values = []
    for group in aggregated:
        subset = group[group["step__mean"] > 0][mean_key].dropna()
        values.extend(subset.tolist())

    if not values:
        return [0.0, 1.0]

    min_value = min(values)
    max_value = max(values)
    padding = max(0.002, (max_value - min_value) * 0.2)
    lower = max(0.0, min_value - padding)
    upper = min(1.0, max_value + padding)
    if upper <= lower:
        upper = min(1.0, lower + 0.01)
    return [lower, upper]


def plot_groups(groups: list[GroupSelection], aggregated: list[pd.DataFrame], output_html: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[metric.title for metric in METRICS],
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )
    group_lookup = {group.label: group for group in groups}

    for metric_index, metric in enumerate(METRICS):
        for group_frame in aggregated:
            label = group_frame["group"].iloc[0]
            color = GROUP_COLORS[label]
            group = group_lookup[label]
            mean_key = f"{metric.key}__mean"
            std_key = f"{metric.key}__std"
            count_key = f"{metric.key}__count"
            subset = group_frame[group_frame[count_key] > 0][["progress_pct", "step__mean", mean_key, std_key]].copy()
            subset[std_key] = subset[std_key].fillna(0.0)
            if metric.key == EVAL_ACCURACY_KEY:
                subset = subset[subset["step__mean"] > 0]
            subset = subset.dropna(subset=[mean_key])
            if subset.empty:
                continue

            if metric.key == TRAIN_LOSS_KEY:
                window = smoothing_window(len(subset))
                subset[mean_key] = subset[mean_key].rolling(window=window, center=True, min_periods=1).mean()
                subset[std_key] = subset[std_key].rolling(window=window, center=True, min_periods=1).mean()

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

            customdata = subset[["step__mean"]].to_numpy()
            trace_name = legend_label(group)
            hovertemplate = (
                f"{trace_name}<br>"
                "progress=%{x:.1f}%<br>"
                "mean step=%{customdata[0]:.0f}<br>"
                f"{metric.title}=%{{y:.6f}}<extra></extra>"
            )

            figure.add_trace(
                go.Scatter(
                    x=subset["progress_pct"],
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
                    x=subset["progress_pct"],
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
                    x=subset["progress_pct"],
                    y=subset[mean_key],
                    mode=mode,
                    line=line_style,
                    marker=marker,
                    name=trace_name,
                    legendgroup=label,
                    showlegend=metric_index == 0,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                row=metric.row,
                col=metric.col,
            )

    og_group = next(group for group in groups if group.label == "OG DPO")
    new_group = next(group for group in groups if group.label == "New DPO")
    figure.update_layout(
        title={
            "text": (
                "<span style='font-size:13px; letter-spacing:0.16em; color:#8B6B4A;'>LOCAL W&B ARCHIVE</span><br>"
                "<span style='font-size:31px;'>Bloom SpecEval v2 Full DPO</span><br>"
                "<span style='font-size:18px; color:#4B5563;'>"
                "beta=0.1 February baseline vs April batch-64 rerun, averaged across seeds"
                "</span>"
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
        margin={"l": 70, "r": 40, "t": 180, "b": 110},
        height=900,
        width=1400,
    )
    figure.add_annotation(
        x=0,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            f"OG DPO: Feb 17 runs, batch {og_group.train_batch_size}, {og_group.num_train_steps} steps. "
            f"New DPO: Apr 3 runs, batch {new_group.train_batch_size}, {new_group.num_train_steps} steps. "
            f"Both use beta=0.1 and lr {og_group.learning_rate:.1e}. "
            "The x-axis is normalized to percent of the run so one epoch lines up across batch sizes. "
            "The 10-step dummy probe is excluded."
        ),
        font={"size": 13, "color": "#6B7280"},
    )
    figure.update_xaxes(
        title_text="Progress Through Run (%)",
        range=[0, 100],
        tickmode="array",
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["0%", "25%", "50%", "75%", "100%"],
        gridcolor="#E5E7EB",
        zeroline=False,
        showline=True,
        linecolor="#D6D3D1",
    )
    figure.update_yaxes(gridcolor="#E5E7EB", zeroline=False, showline=True, linecolor="#D6D3D1")
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_yaxes(tickformat=".2%", row=2, col=1)
    figure.update_yaxes(range=accuracy_axis_range(aggregated), row=2, col=1)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_html, include_plotlyjs="cdn")


def write_selection_summary(output_json: Path, groups: list[GroupSelection]) -> None:
    payload = {
        "selection_rule": "beta=0.1 runs averaged across seeds; excludes the 10-step dummy probe",
        "groups": [
            {
                "label": group.label,
                "learning_rate": group.learning_rate,
                "num_train_steps": group.num_train_steps,
                "train_batch_size": group.train_batch_size,
                "runs": [
                    {
                        "name": run.name,
                        "seed": run.seed,
                        "history_path": str(run.history_path),
                    }
                    for run in group.runs
                ],
            }
            for group in groups
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    og_group = build_group("OG DPO", load_runs(args.og_dir, history_format="csv"))
    new_group = build_group("New DPO", load_runs(args.new_dir, history_format="jsonl.gz"))
    groups = [og_group, new_group]
    aggregated = [aggregate_group(group) for group in groups]
    plot_groups(groups, aggregated, args.output_html)
    write_selection_summary(args.output_json, groups)
    print(f"Wrote plot to {args.output_html}")
    print(f"Wrote selection summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
