# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from math import ceil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

SEED_PATTERN = re.compile(r"_seed(\d+)")
TRAIN_LOSS_KEY = "train/dpo_loss"
EVAL_LOSS_KEY = "eval/bloom_speceval_v2_val/dpo_loss"
EVAL_ACCURACY_KEY = "eval/bloom_speceval_v2_val/dpo_accuracy"
EVAL_MARGIN_KEY = "eval/bloom_speceval_v2_val/dpo_margin_policy"
TRAIN_SMOOTHING_FRACTION = 0.02
PROGRESS_DECIMALS = 4
NUM_COLUMNS = 4
PREFERRED_METRIC_ORDER = (
    "train/loss",
    TRAIN_LOSS_KEY,
    "train/dpo_accuracy",
    "train/dpo_chosen_reward",
    "train/dpo_rejected_reward",
    "train/dpo_margin_policy",
    "train/dpo_margin_ref",
    "eval/bloom_speceval_v2_val/loss",
    EVAL_LOSS_KEY,
    EVAL_ACCURACY_KEY,
    "eval/bloom_speceval_v2_val/dpo_chosen_reward",
    "eval/bloom_speceval_v2_val/dpo_rejected_reward",
    EVAL_MARGIN_KEY,
    "eval/bloom_speceval_v2_val/dpo_margin_ref",
    "eval/bloom_speceval_v2_val/timing/load_time",
    "eval/bloom_speceval_v2_val/timing/loss_time",
    "eval/bloom_speceval_v2_val/timing/num_batches",
    "lm_eval/loss",
    "lm_eval/bpb",
    "lm_eval/macro_loss",
    "lm_eval/macro_bpb",
    "lm_eval/loading_time",
    "lm_eval/total_time",
    "lm_eval/paloma/loss",
    "lm_eval/paloma/bpb",
    "lm_eval/paloma/macro_loss",
    "lm_eval/paloma/macro_bpb",
    "lm_eval/paloma/micro_loss",
    "lm_eval/uncheatable_eval/bpb",
    "lm_eval/uncheatable_eval/macro_loss",
    "lm_eval/uncheatable_eval/macro_bpb",
    "lm_eval/uncheatable_eval/micro_loss",
)
METRIC_TITLES = {
    "train/loss": "Train Loss · Log Scale",
    TRAIN_LOSS_KEY: "Train DPO Loss · Log Scale",
    "train/dpo_accuracy": "Train DPO Accuracy",
    "train/dpo_chosen_reward": "Train Chosen Reward",
    "train/dpo_rejected_reward": "Train Rejected Reward",
    "train/dpo_margin_policy": "Train Policy Margin",
    "train/dpo_margin_ref": "Train Reference Margin",
    "eval/bloom_speceval_v2_val/loss": "Eval Loss · Log Scale",
    EVAL_LOSS_KEY: "Eval DPO Loss · Log Scale",
    EVAL_ACCURACY_KEY: "Eval DPO Accuracy",
    "eval/bloom_speceval_v2_val/dpo_chosen_reward": "Eval Chosen Reward",
    "eval/bloom_speceval_v2_val/dpo_rejected_reward": "Eval Rejected Reward",
    EVAL_MARGIN_KEY: "Eval Policy Margin",
    "eval/bloom_speceval_v2_val/dpo_margin_ref": "Eval Reference Margin",
    "eval/bloom_speceval_v2_val/timing/load_time": "Eval Load Time",
    "eval/bloom_speceval_v2_val/timing/loss_time": "Eval Loss Time",
    "eval/bloom_speceval_v2_val/timing/num_batches": "Eval Num Batches",
    "lm_eval/loss": "LM Eval Loss · Log Scale",
    "lm_eval/bpb": "LM Eval BPB",
    "lm_eval/macro_loss": "LM Eval Macro Loss · Log Scale",
    "lm_eval/macro_bpb": "LM Eval Macro BPB",
    "lm_eval/loading_time": "LM Eval Loading Time",
    "lm_eval/total_time": "LM Eval Total Time",
    "lm_eval/paloma/loss": "LM Eval Paloma Loss · Log Scale",
    "lm_eval/paloma/bpb": "LM Eval Paloma BPB",
    "lm_eval/paloma/macro_loss": "LM Eval Paloma Macro Loss · Log Scale",
    "lm_eval/paloma/macro_bpb": "LM Eval Paloma Macro BPB",
    "lm_eval/paloma/micro_loss": "LM Eval Paloma Micro Loss · Log Scale",
    "lm_eval/uncheatable_eval/bpb": "LM Eval Uncheatable BPB",
    "lm_eval/uncheatable_eval/macro_loss": "LM Eval Uncheatable Macro Loss · Log Scale",
    "lm_eval/uncheatable_eval/macro_bpb": "LM Eval Uncheatable Macro BPB",
    "lm_eval/uncheatable_eval/micro_loss": "LM Eval Uncheatable Micro Loss · Log Scale",
}
TRAIN_SMOOTHING_KEYS = {
    "train/loss",
    TRAIN_LOSS_KEY,
    "train/dpo_accuracy",
    "train/dpo_chosen_reward",
    "train/dpo_rejected_reward",
    "train/dpo_margin_policy",
    "train/dpo_margin_ref",
}
ACCURACY_KEYS = {
    "train/dpo_accuracy",
    EVAL_ACCURACY_KEY,
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    row: int
    col: int


@dataclass
class TabSpec:
    tab_id: str
    label: str
    description: str
    metric_keys: list[str]


SUMMARY_METRIC_KEYS = tuple(METRIC_TITLES)

FINAL_RANKING_KEYS = (
    EVAL_ACCURACY_KEY,
    EVAL_LOSS_KEY,
    EVAL_MARGIN_KEY,
)
GROUP_COLORS = {
    "Full DPO": "#5B6577",
    "LoRA 10x LR": "#E07A26",
    "LoRA Best Eval": "#1C8C7C",
}


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
    seed_count: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    output_dir = root / "scratch" / "wandb_dpo_data" / "plots"
    parser = argparse.ArgumentParser(
        description="Plot batch-64 full DPO against selected tune-LoRA groups from local W&B archives."
    )
    parser.add_argument(
        "--full-dpo-dir",
        type=Path,
        default=root / "scratch" / "wandb_dpo_data" / "new_dpo",
        help="Directory containing archived full-DPO runs.",
    )
    parser.add_argument(
        "--tune-dir",
        type=Path,
        default=root / "scratch" / "wandb_dpo_data" / "tune_lora",
        help="Directory containing archived tune-LoRA runs.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=output_dir / "beta0p1_full_dpo_b64_vs_tune_lora.html",
        help="Path for the interactive HTML plot.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=output_dir / "beta0p1_full_dpo_b64_vs_tune_lora_selection.json",
        help="Path for the selection summary JSON.",
    )
    return parser.parse_args()


def parse_seed(run_name: str, config: dict[str, object]) -> int:
    data_seed = config.get("data_seed")
    if data_seed is not None:
        return int(data_seed)
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
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        history_path = run_dir / history_name
        if not config_path.exists() or not summary_path.exists() or not history_path.exists():
            continue
        config = load_json(config_path)
        trainer = config["trainer"]
        optimizer = config["optimizer"]
        runs.append(
            RunRecord(
                name=run_dir.name,
                beta=float(config["beta"]),
                learning_rate=float(optimizer["learning_rate"]),
                seed=parse_seed(run_dir.name, config),
                history_path=history_path,
                history_format=history_format,
                summary=load_json(summary_path),
                train_batch_size=int(trainer["train_batch_size"]),
                num_train_steps=int(trainer["num_train_steps"]),
            )
        )
    return runs


def load_history(run: RunRecord, metric_keys: list[str]) -> pd.DataFrame:
    columns = ["_step", *metric_keys]
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


def build_group(label: str, runs: list[RunRecord], expected_seeds: list[int] | None) -> GroupSelection:
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
    if expected_seeds is not None and seeds != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds} for {label}, found {seeds}")

    return GroupSelection(
        label=label,
        runs=sorted(runs, key=lambda run: run.seed),
        learning_rate=next(iter(learning_rates)),
        train_batch_size=next(iter(train_batch_sizes)),
        num_train_steps=next(iter(num_train_steps)),
        seed_count=len(runs),
    )


def progress_percent(frame: pd.DataFrame) -> pd.Series:
    max_step = int(frame["step"].max())
    if max_step <= 0:
        raise ValueError("Expected positive max step in run history.")
    return (100.0 * frame["step"] / max_step).round(PROGRESS_DECIMALS)


def aggregate_group(group: GroupSelection, metric_keys: list[str]) -> pd.DataFrame:
    run_frames: list[pd.DataFrame] = []
    for run in group.runs:
        frame = load_history(run, metric_keys).copy()
        frame["progress_pct"] = progress_percent(frame)
        run_frames.append(frame)

    history = pd.concat(run_frames, ignore_index=True)
    aggregations: dict[str, list[str]] = {"step": ["mean", "min", "max", "count"]}
    aggregations.update({metric: ["mean", "std", "count"] for metric in metric_keys})
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
    values = []
    mean_key = f"{EVAL_ACCURACY_KEY}__mean"
    for group in aggregated:
        subset = group[group["step__mean"] > 0][mean_key].dropna()
        values.extend(subset.tolist())
    if not values:
        return [0.0, 1.0]
    min_value = min(values)
    max_value = max(values)
    padding = max(0.01, (max_value - min_value) * 0.2)
    lower = max(0.0, min_value - padding)
    upper = min(1.0, max_value + padding)
    if upper <= lower:
        upper = min(1.0, lower + 0.05)
    return [lower, upper]


def is_metric_key(key: str) -> bool:
    return key.startswith(("train/", "eval/", "lm_eval/"))


def default_metric_title(key: str) -> str:
    pieces = key.split("/")
    if pieces[0] == "train":
        prefix = "Train"
        suffix_parts = pieces[1:]
    elif pieces[0] == "eval":
        prefix = f"Eval {pieces[1].replace('_', ' ').title()}"
        suffix_parts = pieces[2:]
    else:
        prefix = "LM Eval"
        suffix_parts = pieces[1:]

    formatted_suffix = " ".join(part.replace("_", " ").title() for part in suffix_parts)
    title = f"{prefix} {formatted_suffix}".strip()
    if key.endswith("/loss") or key in {"train/loss", TRAIN_LOSS_KEY, EVAL_LOSS_KEY, "lm_eval/loss"}:
        title += " · Log Scale"
    return title


def is_log_scale_key(key: str) -> bool:
    return key.endswith("/loss") or key in {
        "train/loss",
        TRAIN_LOSS_KEY,
        "eval/bloom_speceval_v2_val/loss",
        EVAL_LOSS_KEY,
        "lm_eval/loss",
        "lm_eval/macro_loss",
        "lm_eval/paloma/loss",
        "lm_eval/paloma/macro_loss",
        "lm_eval/paloma/micro_loss",
        "lm_eval/uncheatable_eval/macro_loss",
        "lm_eval/uncheatable_eval/ao3_english/loss",
        "lm_eval/uncheatable_eval/arxiv_computer_science/loss",
        "lm_eval/uncheatable_eval/arxiv_physics/loss",
        "lm_eval/uncheatable_eval/bbc_news/loss",
        "lm_eval/uncheatable_eval/github_cpp/loss",
        "lm_eval/uncheatable_eval/github_python/loss",
        "lm_eval/uncheatable_eval/wikipedia_english/loss",
    }


def metric_specs_for_groups(groups: list[GroupSelection]) -> list[MetricSpec]:
    all_common_keys = {
        key
        for key in groups[0].runs[0].summary
        if is_metric_key(key) and all(key in run.summary for group in groups for run in group.runs)
    }
    metric_keys = [key for key in PREFERRED_METRIC_ORDER if key in all_common_keys]
    metric_keys.extend(sorted(all_common_keys - set(metric_keys)))
    if not metric_keys:
        raise ValueError("No common train/eval metrics found across selected groups.")

    specs = []
    for index, key in enumerate(metric_keys):
        specs.append(
            MetricSpec(
                key=key,
                title=METRIC_TITLES.get(key, default_metric_title(key)),
                row=(index // NUM_COLUMNS) + 1,
                col=(index % NUM_COLUMNS) + 1,
            )
        )
    return specs


def build_tabs(metric_specs: list[MetricSpec]) -> list[TabSpec]:
    tabs = [
        TabSpec("core-dpo", "Core DPO", "Train and DPO validation metrics.", []),
        TabSpec("lm-summary", "LM Summary", "Top-level LM-eval aggregates and timing.", []),
        TabSpec("paloma-summary", "Paloma Summary", "Paloma aggregate metrics.", []),
        TabSpec("paloma-bpb", "Paloma BPB", "Per-slice Paloma bits-per-byte metrics.", []),
        TabSpec("paloma-loss", "Paloma Loss", "Per-slice Paloma loss metrics.", []),
        TabSpec("uncheatable-summary", "Uncheatable Summary", "Uncheatable aggregate metrics.", []),
        TabSpec("uncheatable-bpb", "Uncheatable BPB", "Per-slice uncheatable bits-per-byte metrics.", []),
        TabSpec("uncheatable-loss", "Uncheatable Loss", "Per-slice uncheatable loss metrics.", []),
        TabSpec("other", "Other", "Shared metrics that did not fit the named buckets.", []),
    ]
    tab_lookup = {tab.tab_id: tab for tab in tabs}

    for spec in metric_specs:
        key = spec.key
        if key.startswith(("train/", "eval/")):
            tab_lookup["core-dpo"].metric_keys.append(key)
            continue

        if key.startswith("lm_eval/paloma/"):
            suffix = key.removeprefix("lm_eval/paloma/")
            if suffix in {"loss", "bpb", "macro_loss", "macro_bpb", "micro_loss"}:
                tab_lookup["paloma-summary"].metric_keys.append(key)
            elif suffix.endswith("/bpb"):
                tab_lookup["paloma-bpb"].metric_keys.append(key)
            elif suffix.endswith("/loss"):
                tab_lookup["paloma-loss"].metric_keys.append(key)
            else:
                tab_lookup["other"].metric_keys.append(key)
            continue

        if key.startswith("lm_eval/uncheatable_eval/"):
            suffix = key.removeprefix("lm_eval/uncheatable_eval/")
            if suffix in {"bpb", "macro_loss", "macro_bpb", "micro_loss"}:
                tab_lookup["uncheatable-summary"].metric_keys.append(key)
            elif suffix.endswith("/bpb"):
                tab_lookup["uncheatable-bpb"].metric_keys.append(key)
            elif suffix.endswith("/loss"):
                tab_lookup["uncheatable-loss"].metric_keys.append(key)
            else:
                tab_lookup["other"].metric_keys.append(key)
            continue

        if key.startswith("lm_eval/"):
            tab_lookup["lm-summary"].metric_keys.append(key)
            continue

        tab_lookup["other"].metric_keys.append(key)

    return [tab for tab in tabs if tab.metric_keys]


def reindex_metric_specs(metric_specs: list[MetricSpec]) -> list[MetricSpec]:
    return [
        MetricSpec(
            key=spec.key,
            title=spec.title,
            row=(index // NUM_COLUMNS) + 1,
            col=(index % NUM_COLUMNS) + 1,
        )
        for index, spec in enumerate(metric_specs)
    ]


def display_label(group: GroupSelection) -> str:
    return f"{group.label} · {group.seed_count} seeds · lr {group.learning_rate:.1e}"


def build_figure(
    groups: list[GroupSelection],
    aggregated: list[pd.DataFrame],
    metric_specs: list[MetricSpec],
    panel_title: str,
    panel_description: str,
) -> go.Figure:
    num_rows = ceil(len(metric_specs) / NUM_COLUMNS)
    vertical_spacing = min(0.06, 0.8 / max(1, num_rows - 1))
    subplot_titles = [""] * (num_rows * NUM_COLUMNS)
    for spec in metric_specs:
        subplot_titles[(spec.row - 1) * NUM_COLUMNS + (spec.col - 1)] = spec.title

    figure = make_subplots(
        rows=num_rows,
        cols=NUM_COLUMNS,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=vertical_spacing,
    )
    group_lookup = {group.label: group for group in groups}

    for metric_index, spec in enumerate(metric_specs):
        metric_key = spec.key
        metric_title = spec.title
        row = spec.row
        col = spec.col
        for group_frame in aggregated:
            label = group_frame["group"].iloc[0]
            color = GROUP_COLORS[label]
            group = group_lookup[label]
            mean_key = f"{metric_key}__mean"
            std_key = f"{metric_key}__std"
            count_key = f"{metric_key}__count"
            subset = group_frame[group_frame[count_key] > 0][["progress_pct", "step__mean", mean_key, std_key]].copy()
            subset[std_key] = subset[std_key].fillna(0.0)
            if metric_key in ACCURACY_KEYS:
                subset = subset[subset["step__mean"] > 0]
            subset = subset.dropna(subset=[mean_key])
            if subset.empty:
                continue

            if metric_key in TRAIN_SMOOTHING_KEYS:
                window = smoothing_window(len(subset))
                subset[mean_key] = subset[mean_key].rolling(window=window, center=True, min_periods=1).mean()
                subset[std_key] = subset[std_key].rolling(window=window, center=True, min_periods=1).mean()

            is_eval_metric = metric_key.startswith("eval/")
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
            if is_log_scale_key(metric_key):
                upper_band = upper_band.clip(lower=1e-6)
                lower_band = lower_band.clip(lower=1e-6)

            customdata = subset[["step__mean"]].to_numpy()
            trace_name = display_label(group)
            hovertemplate = (
                f"{trace_name}<br>"
                "progress=%{x:.1f}%<br>"
                "mean step=%{customdata[0]:.0f}<br>"
                f"{metric_title}=%{{y:.6f}}<extra></extra>"
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
                row=row,
                col=col,
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
                row=row,
                col=col,
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
                row=row,
                col=col,
            )

    full_group = next(group for group in groups if group.label == "Full DPO")
    tinker_group = next(group for group in groups if group.label == "LoRA 10x LR")
    best_group = next(group for group in groups if group.label == "LoRA Best Eval")
    figure.update_layout(
        title={
            "text": (
                "<span style='font-size:13px; letter-spacing:0.16em; color:#8B6B4A;'>LOCAL W&B ARCHIVE</span><br>"
                "<span style='font-size:31px;'>Batch-64 Full DPO vs Tune-LoRA</span><br>"
                "<span style='font-size:18px; color:#4B5563;'>"
                f"{panel_title}: {panel_description}"
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
        margin={"l": 70, "r": 40, "t": 180, "b": 118},
        height=max(1050, 250 * num_rows + 220),
        width=1800,
    )
    figure.add_annotation(
        x=0,
        y=-0.16,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            f"Full DPO: {full_group.seed_count} seeds (0/1/2), batch {full_group.train_batch_size}, "
            f"{full_group.num_train_steps} steps, lr {full_group.learning_rate:.1e}. "
            f"LoRA 10x LR: {tinker_group.seed_count} archived seeds (0/2), lr {tinker_group.learning_rate:.1e}. "
            f"LoRA Best Eval: {best_group.seed_count} archived seeds (0/2), lr {best_group.learning_rate:.1e}. "
            "This HTML now includes every shared `train/`, `eval/`, and `lm_eval/` metric across all three groups. "
            "The LoRA sweep archive has no seed-1 run, "
            "so those groups are true 2-seed means."
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
    for spec in metric_specs:
        if is_log_scale_key(spec.key):
            figure.update_yaxes(type="log", row=spec.row, col=spec.col)
        if spec.key in ACCURACY_KEYS:
            figure.update_yaxes(
                tickformat=".1%",
                range=accuracy_axis_range(aggregated),
                row=spec.row,
                col=spec.col,
            )

    return figure


def write_tabbed_html(
    groups: list[GroupSelection],
    aggregated: list[pd.DataFrame],
    metric_specs: list[MetricSpec],
    output_html: Path,
) -> None:
    metric_lookup = {spec.key: spec for spec in metric_specs}
    tabs = build_tabs(metric_specs)
    tab_buttons: list[str] = []
    tab_panels: list[str] = []

    for index, tab in enumerate(tabs):
        tab_metric_specs = reindex_metric_specs([metric_lookup[key] for key in tab.metric_keys])
        figure = build_figure(groups, aggregated, tab_metric_specs, tab.label, tab.description)
        figure_html = pio.to_html(figure, full_html=False, include_plotlyjs=False)
        active_class = " active" if index == 0 else ""
        panel_style = "display:block;" if index == 0 else "display:none;"
        tab_buttons.append(f'<button class="tab-button{active_class}" data-tab="{tab.tab_id}">{tab.label}</button>')
        tab_panels.append(
            f'<section class="tab-panel" id="{tab.tab_id}" style="{panel_style}">'
            f'<div class="tab-description">{tab.description}</div>{figure_html}</section>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Batch-64 Full DPO vs Tune-LoRA</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      body {{
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background: #f7f3ea;
        color: #1f2937;
      }}
      .page {{
        padding: 20px 24px 32px;
      }}
      .tabs {{
        position: sticky;
        top: 0;
        z-index: 10;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 12px 0 16px;
        background: rgba(247, 243, 234, 0.96);
        backdrop-filter: blur(6px);
      }}
      .tab-button {{
        border: 1px solid #d8cdb9;
        background: #fffdf8;
        color: #5b6577;
        border-radius: 999px;
        padding: 10px 16px;
        cursor: pointer;
        font-size: 15px;
      }}
      .tab-button.active {{
        background: #1f2937;
        color: #fffdf8;
        border-color: #1f2937;
      }}
      .tab-panel {{
        margin-top: 8px;
      }}
      .tab-description {{
        margin: 0 0 8px;
        color: #6b7280;
        font-size: 15px;
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <div class="tabs">
        {''.join(tab_buttons)}
      </div>
      {''.join(tab_panels)}
    </main>
    <script>
      const buttons = Array.from(document.querySelectorAll('.tab-button'));
      const panels = Array.from(document.querySelectorAll('.tab-panel'));
      function showTab(tabId) {{
        for (const button of buttons) {{
          button.classList.toggle('active', button.dataset.tab === tabId);
        }}
        for (const panel of panels) {{
          panel.style.display = panel.id === tabId ? 'block' : 'none';
        }}
        window.dispatchEvent(new Event('resize'));
      }}
      for (const button of buttons) {{
        button.addEventListener('click', () => showTab(button.dataset.tab));
      }}
    </script>
  </body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html)


def select_runs(
    full_dpo_runs: list[RunRecord], tune_runs: list[RunRecord]
) -> tuple[list[GroupSelection], pd.DataFrame, float, float]:
    full_beta_runs = [
        run
        for run in full_dpo_runs
        if math.isclose(run.beta, 0.1) and run.train_batch_size == 64 and run.num_train_steps == 1700
    ]
    full_group = build_group("Full DPO", full_beta_runs, expected_seeds=[0, 1, 2])

    tune_beta_runs = [
        run
        for run in tune_runs
        if math.isclose(run.beta, 0.1) and run.train_batch_size == 64 and run.num_train_steps == 1700
    ]
    if not tune_beta_runs:
        raise ValueError("No tune-LoRA beta=0.1 batch-64 runs found.")

    ranking = pd.DataFrame(
        [
            {
                "learning_rate": run.learning_rate,
                "seed": run.seed,
                "final_eval_accuracy": run.summary.get(FINAL_RANKING_KEYS[0]),
                "final_eval_loss": run.summary.get(FINAL_RANKING_KEYS[1]),
                "final_eval_margin": run.summary.get(FINAL_RANKING_KEYS[2]),
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

    tinker_lr = full_group.learning_rate * 10
    tinker_runs = [run for run in tune_beta_runs if math.isclose(run.learning_rate, tinker_lr)]
    if not tinker_runs:
        raise ValueError(f"No tune-LoRA runs found for 10x full-DPO lr={tinker_lr:.2e}.")

    best_lr = float(ranking.iloc[0]["learning_rate"])
    best_runs = [run for run in tune_beta_runs if math.isclose(run.learning_rate, best_lr)]

    tinker_group = build_group("LoRA 10x LR", tinker_runs, expected_seeds=[0, 2])
    best_group = build_group("LoRA Best Eval", best_runs, expected_seeds=[0, 2])
    return [full_group, tinker_group, best_group], ranking, tinker_lr, best_lr


def write_selection_summary(
    output_json: Path,
    groups: list[GroupSelection],
    ranking: pd.DataFrame,
    metric_specs: list[MetricSpec],
    tinker_lr: float,
    best_lr: float,
) -> None:
    tabs = build_tabs(metric_specs)
    payload = {
        "best_eval_learning_rate": best_lr,
        "common_metric_keys": [spec.key for spec in metric_specs],
        "tabs": [
            {
                "description": tab.description,
                "label": tab.label,
                "metric_keys": tab.metric_keys,
                "tab_id": tab.tab_id,
            }
            for tab in tabs
        ],
        "selection_rule": {
            "best_eval": "highest mean final eval accuracy, tie-break lowest mean final eval loss",
            "common_metrics": "all shared train/eval metrics present in the archived histories for all three groups",
            "full_dpo": "beta=0.1, batch=64, steps=1700, seeds 0/1/2",
            "seed_note": "LoRA archive contains seeds 0 and 2 only for each learning rate",
            "tinker_recommended": "10x the full-DPO batch-64 learning rate",
        },
        "tinker_recommended_learning_rate": tinker_lr,
        "groups": [
            {
                "label": group.label,
                "learning_rate": group.learning_rate,
                "num_train_steps": group.num_train_steps,
                "seed_count": group.seed_count,
                "train_batch_size": group.train_batch_size,
                "runs": [
                    {
                        "name": run.name,
                        "beta": run.beta,
                        "learning_rate": run.learning_rate,
                        "seed": run.seed,
                    }
                    for run in group.runs
                ],
            }
            for group in groups
        ],
        "tune_lora_ranking": ranking.to_dict(orient="records"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    full_dpo_runs = load_runs(args.full_dpo_dir, history_format="jsonl.gz")
    tune_runs = load_runs(args.tune_dir, history_format="jsonl.gz")
    groups, ranking, tinker_lr, best_lr = select_runs(full_dpo_runs, tune_runs)
    metric_specs = metric_specs_for_groups(groups)
    metric_keys = [spec.key for spec in metric_specs]
    aggregated = [aggregate_group(group, metric_keys) for group in groups]
    write_tabbed_html(groups, aggregated, metric_specs, args.output_html)
    write_selection_summary(args.output_json, groups, ranking, metric_specs, tinker_lr, best_lr)
    print(f"Wrote plot to {args.output_html}")
    print(f"Wrote selection summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
