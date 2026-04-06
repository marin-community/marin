# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRAIN_LOSS_KEY = "train/dpo_loss"
EVAL_LOSS_KEY = "eval/bloom_speceval_v2_val/dpo_loss"
EVAL_ACCURACY_KEY = "eval/bloom_speceval_v2_val/dpo_accuracy"
EVAL_MARGIN_KEY = "eval/bloom_speceval_v2_val/dpo_margin_policy"
METRIC_TITLES = {
    TRAIN_LOSS_KEY: "Train DPO Loss · Log Scale",
    EVAL_LOSS_KEY: "Eval DPO Loss · Log Scale",
    EVAL_ACCURACY_KEY: "Eval DPO Accuracy",
    EVAL_MARGIN_KEY: "Eval Policy Margin",
}
METRIC_LAYOUT = {
    TRAIN_LOSS_KEY: (1, 1),
    EVAL_LOSS_KEY: (1, 2),
    EVAL_ACCURACY_KEY: (2, 1),
    EVAL_MARGIN_KEY: (2, 2),
}
METRICS = tuple(METRIC_TITLES)
TRAIN_SMOOTHING_FRACTION = 0.02
RUN_COLORS = {
    "Reference Full Val": "#5B6577",
    "Regression Deduped Val": "#1C8C7C",
}


@dataclass(frozen=True)
class RunArchive:
    label: str
    short_id: str
    run_dir: Path
    color: str
    validation_note: str
    history_path: Path
    config: dict[str, object]
    summary: dict[str, object]
    metadata: dict[str, object]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    output_dir = root / "scratch" / "wandb_dpo_data" / "plots"
    parser = argparse.ArgumentParser(
        description="Compare the new_dpo_v2 947c5d baseline against the deduped-val regression test."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=(
            root
            / "scratch"
            / "wandb_dpo_data"
            / "reference_dpo"
            / "new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_-7_seed2-947c5d"
        ),
        help="Archived W&B export directory for the full-val baseline run.",
    )
    parser.add_argument(
        "--regression-dir",
        type=Path,
        default=(
            root
            / "scratch"
            / "wandb_dpo_data"
            / "new_dpo"
            / "regression_test_dpo_bloom_lr7.5e-7_seed2_deduped_val-1e4e93"
        ),
        help="Archived W&B export directory for the deduped-val regression run.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=output_dir / "regression_test_vs_new_dpo_v2_seed2.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=output_dir / "regression_test_vs_new_dpo_v2_seed2_selection.json",
        help="Selection summary output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def load_run(run_dir: Path, label: str, color: str, validation_note: str) -> RunArchive:
    history_path = run_dir / "history.jsonl.gz"
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    metadata = load_json(run_dir / "run_metadata.json")
    return RunArchive(
        label=label,
        short_id=run_dir.name.rsplit("-", 1)[-1],
        run_dir=run_dir,
        color=color,
        validation_note=validation_note,
        history_path=history_path,
        config=config,
        summary=summary,
        metadata=metadata,
    )


def last_non_null(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return float("nan")
    return float(non_null.iloc[-1])


def load_history(run: RunArchive) -> pd.DataFrame:
    rows = []
    with gzip.open(run.history_path, "rt") as handle:
        for line in handle:
            payload = json.loads(line)
            row = {"step": payload.get("_step")}
            for metric in METRICS:
                row[metric] = payload.get(metric)
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No history rows found in {run.history_path}")

    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["step"]).sort_values("step")
    collapsed = frame.groupby("step", sort=True).agg({metric: last_non_null for metric in METRICS}).reset_index()
    collapsed["step"] = collapsed["step"].astype(int)
    return collapsed


def smoothing_window(num_points: int) -> int:
    if num_points <= 2:
        return 1
    window = max(5, round(num_points * TRAIN_SMOOTHING_FRACTION))
    if window % 2 == 0:
        window += 1
    if window > num_points:
        window = num_points if num_points % 2 == 1 else num_points - 1
    return max(window, 1)


def smooth_train_loss(frame: pd.DataFrame) -> pd.Series:
    window = smoothing_window(len(frame))
    return frame[TRAIN_LOSS_KEY].rolling(window=window, center=True, min_periods=1).mean()


def accuracy_axis_range(frames: list[pd.DataFrame]) -> list[float]:
    values = []
    for frame in frames:
        subset = frame[EVAL_ACCURACY_KEY].dropna()
        values.extend(subset.tolist())
    if not values:
        return [0.0, 1.0]
    lower = max(0.0, min(values) - 0.03)
    upper = min(1.0, max(values) + 0.01)
    if upper <= lower:
        upper = min(1.0, lower + 0.05)
    return [lower, upper]


def summary_value(run: RunArchive, key: str) -> float:
    return float(run.summary[key])


def final_metrics_annotation(baseline: RunArchive, regression: RunArchive) -> str:
    baseline_eval_loss = summary_value(baseline, EVAL_LOSS_KEY)
    regression_eval_loss = summary_value(regression, EVAL_LOSS_KEY)
    loss_ratio = regression_eval_loss / baseline_eval_loss
    baseline_train_loss = summary_value(baseline, TRAIN_LOSS_KEY)
    regression_train_loss = summary_value(regression, TRAIN_LOSS_KEY)
    baseline_eval_accuracy = summary_value(baseline, EVAL_ACCURACY_KEY)
    regression_eval_accuracy = summary_value(regression, EVAL_ACCURACY_KEY)
    return (
        f"<b>Final Step 849</b><br>"
        f"train loss: {baseline_train_loss:.5f} vs {regression_train_loss:.5f}<br>"
        f"eval loss: {baseline_eval_loss:.5f} vs {regression_eval_loss:.5f} ({loss_ratio:.1f}x)<br>"
        f"eval acc: {baseline_eval_accuracy:.2%} vs {regression_eval_accuracy:.2%}"
    )


def legend_label(run: RunArchive) -> str:
    trainer = run.config["trainer"]
    optimizer = run.config["optimizer"]
    batch = int(trainer["train_batch_size"])
    steps = int(trainer["num_train_steps"])
    lr = float(optimizer["learning_rate"])
    return f"{run.label} · b{batch} · {steps} steps · lr={lr:.1e}"


def add_metric_trace(figure: go.Figure, run: RunArchive, frame: pd.DataFrame, metric: str, showlegend: bool) -> None:
    subset = frame[["step", metric]].dropna().copy()
    if subset.empty:
        return

    y_values = subset[metric]
    mode = "lines"
    line = {"color": run.color, "width": 3}
    marker = None

    if metric == TRAIN_LOSS_KEY:
        y_values = smooth_train_loss(frame.loc[subset.index])
    else:
        mode = "lines+markers"
        line = {"color": run.color, "width": 2.5}
        marker = {
            "size": 8,
            "color": run.color,
            "line": {"color": "#FFFDF8", "width": 1.5},
        }

    if metric in (TRAIN_LOSS_KEY, EVAL_LOSS_KEY):
        y_values = y_values.clip(lower=1e-6)

    row, col = METRIC_LAYOUT[metric]
    figure.add_trace(
        go.Scatter(
            x=subset["step"],
            y=y_values,
            mode=mode,
            line=line,
            marker=marker,
            name=legend_label(run),
            legendgroup=run.label,
            showlegend=showlegend,
            customdata=[[run.short_id, run.validation_note] for _ in range(len(subset))],
            hovertemplate=(
                "%{fullData.name}<br>"
                "step=%{x}<br>"
                f"{METRIC_TITLES[metric]}=%{{y:.6f}}<br>"
                "run=%{customdata[0]}<br>"
                "validation=%{customdata[1]}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def plot_runs(baseline: RunArchive, regression: RunArchive, output_html: Path) -> None:
    baseline_history = load_history(baseline)
    regression_history = load_history(regression)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[METRIC_TITLES[metric] for metric in METRICS],
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )

    for index, metric in enumerate(METRICS):
        add_metric_trace(figure, baseline, baseline_history, metric, showlegend=index == 0)
        add_metric_trace(figure, regression, regression_history, metric, showlegend=index == 0)

    trainer = baseline.config["trainer"]
    optimizer = baseline.config["optimizer"]
    figure.update_layout(
        title={
            "text": (
                "<span style='font-size:13px; letter-spacing:0.16em; color:#8B6B4A;'>LOCAL W&B ARCHIVE</span><br>"
                "<span style='font-size:31px;'>Regression Test vs new_dpo_v2 Baseline</span><br>"
                "<span style='font-size:18px; color:#4B5563;'>"
                "Same seed-2 training shape, exact step overlay, validation stack changed"
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
        margin={"l": 70, "r": 40, "t": 190, "b": 120},
        height=920,
        width=1440,
    )
    figure.add_annotation(
        x=0.985,
        y=1.16,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 253, 248, 0.92)",
        bordercolor="#E5DED0",
        borderwidth=1,
        text=final_metrics_annotation(baseline, regression),
        font={"size": 13, "color": "#374151"},
    )
    figure.add_annotation(
        x=0,
        y=-0.16,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            f"Both runs use beta={baseline.config['beta']}, lr={float(optimizer['learning_rate']):.1e}, "
            f"batch {int(trainer['train_batch_size'])}, {int(trainer['num_train_steps'])} steps, "
            f"steps_per_eval={int(trainer['steps_per_eval'])}, data_seed={baseline.config['data_seed']}, "
            f"trainer.seed={int(trainer['seed'])}. "
            f"Reference ({baseline.short_id}) used the older full validation set. "
            f"Regression ({regression.short_id}) used the deduped validation set plus current validation callbacks."
        ),
        font={"size": 13, "color": "#6B7280"},
    )
    figure.update_xaxes(
        title_text="Train Step",
        range=[0, int(trainer["num_train_steps"])],
        gridcolor="#E5E7EB",
        zeroline=False,
        showline=True,
        linecolor="#D6D3D1",
    )
    figure.update_yaxes(gridcolor="#E5E7EB", zeroline=False, showline=True, linecolor="#D6D3D1")
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_yaxes(
        tickformat=".1%",
        range=accuracy_axis_range([baseline_history, regression_history]),
        row=2,
        col=1,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_html, include_plotlyjs="cdn")


def write_selection_summary(baseline: RunArchive, regression: RunArchive, output_json: Path) -> None:
    payload = {
        "selection_rule": "Single-run exact-step comparison of baseline 947c5d vs regression 1e4e93.",
        "runs": [
            {
                "label": baseline.label,
                "history_path": str(baseline.history_path),
                "id": baseline.metadata["id"],
                "name": baseline.metadata["name"],
                "run_dir": str(baseline.run_dir),
                "url": baseline.metadata["url"],
                "validation_note": baseline.validation_note,
            },
            {
                "label": regression.label,
                "history_path": str(regression.history_path),
                "id": regression.metadata["id"],
                "name": regression.metadata["name"],
                "run_dir": str(regression.run_dir),
                "url": regression.metadata["url"],
                "validation_note": regression.validation_note,
            },
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    baseline = load_run(
        args.baseline_dir,
        label="Reference Full Val",
        color=RUN_COLORS["Reference Full Val"],
        validation_note="older full validation set",
    )
    regression = load_run(
        args.regression_dir,
        label="Regression Deduped Val",
        color=RUN_COLORS["Regression Deduped Val"],
        validation_note="deduped validation set + current validation callbacks",
    )

    plot_runs(baseline, regression, args.output_html)
    write_selection_summary(baseline, regression, args.output_json)
    print(f"Wrote plot to {args.output_html}")
    print(f"Wrote selection summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
