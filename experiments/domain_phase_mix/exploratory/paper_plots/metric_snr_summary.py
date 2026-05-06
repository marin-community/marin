# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["kaleido==0.2.1", "pandas", "plotly"]
# ///
"""Render a compact SNR summary for the 300M metric matrix.

The plot ranks one best-SNR smooth/proxy metric per eval/task slice. This keeps
duplicate columns such as BPB/loss or choice-prob/logprob variants from
dominating the visual, while still showing which metric was selected for each
slice. Hard accuracy-like metrics are intentionally excluded from selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    PAPER_AXIS,
    PAPER_MUTED,
    configure_interactive_layout,
    configure_static_layout,
    write_static_images,
)

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent / "two_phase_many"
DEFAULT_SNR_CSV = TWO_PHASE_MANY_DIR / "eval_signal_to_noise_all_metrics_300m_current.csv"
OUTPUT_STEM = IMG_DIR / "metric_snr_summary"

SMOOTH_METRIC_KINDS = {
    "bpb",
    "choice_logprob",
    "choice_logprob_norm",
    "choice_prob_norm",
    "logprob",
    "loss",
    "nll",
    "perplexity",
}

METRIC_KIND_SUFFIXES = (
    "_choice_logprob_norm",
    "_choice_prob_norm",
    "_choice_logprob",
    "_exact_match",
    "_pass_at_1",
    "_perplexity",
    "_acc_norm",
    "_logprob",
    "_loss",
    "_bpb",
    "_nll",
    "_acc",
)

FAMILY_COLORS = {
    "Uncheatable BPB": "#4C78A8",
    "Paloma BPB": "#59A14F",
    "Agentic coding BPB": "#E15759",
    "Generic eval BPB": "#B07AA1",
    "MMLU": "#F28E2B",
    "MMLU SL-Verb": "#FFBE7D",
    "MMLU-Pro": "#9C755F",
    "English MCQ/cloze": "#EDC948",
    "Generation proxies": "#76B7B2",
    "Other lm-eval": "#BAB0AC",
}

SOURCE_SYMBOLS = {
    "eval BPB/loss": "circle",
    "lm-eval task": "diamond",
    "custom task proxy": "square",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr-csv", type=Path, default=DEFAULT_SNR_CSV)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    return parser.parse_args()


def _metric_prefix(metric: str) -> str:
    return metric.split("/", maxsplit=1)[0]


def _source_class(metric: str) -> str:
    prefix = _metric_prefix(metric)
    if prefix == "eval":
        return "eval BPB/loss"
    if prefix == "lm_eval":
        return "lm-eval task"
    if prefix in {"teacher_forced", "mcq_smooth"}:
        return "custom task proxy"
    return "other"


def _strip_suffix_metric_kind(path: str) -> str:
    for suffix in METRIC_KIND_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def _metric_item(metric: str, metric_kind: str) -> str:
    """Return the eval/task slice key used before best-metric selection."""
    if "/" in metric:
        parts = metric.split("/")
        if parts[-1] == metric_kind:
            return "/".join(parts[:-1])
    return _strip_suffix_metric_kind(metric)


def _metric_family(metric: str) -> str:
    parts = metric.split("/")
    if metric.startswith("eval/uncheatable_eval"):
        return "Uncheatable BPB"
    if metric.startswith("eval/paloma"):
        return "Paloma BPB"
    if metric.startswith("eval/agentic_coding"):
        return "Agentic coding BPB"
    if metric.startswith("eval/"):
        return "Generic eval BPB"
    if len(parts) >= 2 and parts[0] == "lm_eval":
        task = parts[1]
        if "mmlu_pro" in task:
            return "MMLU-Pro"
        if task == "mmlu_sl_verb_5shot" or (task.startswith("mmlu_") and "_sl_verb_" in task):
            return "MMLU SL-Verb"
        if task == "mmlu_5shot" or task.startswith("mmlu_") or task.startswith("mmlu"):
            return "MMLU"
        if task.startswith(("gsm8k", "humaneval")):
            return "Generation proxies"
        if task.startswith(
            (
                "arc_",
                "boolq",
                "copa",
                "csqa",
                "hellaswag",
                "lambada",
                "medmcqa",
                "openbookqa",
                "piqa",
                "sciq",
                "socialiqa",
                "swag",
                "truthfulqa",
                "winogrande",
                "wsc",
            )
        ):
            return "English MCQ/cloze"
        return "Other lm-eval"
    if metric.startswith(("teacher_forced/gsm8k", "teacher_forced/humaneval")):
        return "Generation proxies"
    if metric.startswith("mcq_smooth/"):
        return "English MCQ/cloze"
    return "Other lm-eval"


def _short_item_label(item: str) -> str:
    if item.startswith("eval/"):
        return item.removeprefix("eval/")
    if item.startswith("lm_eval/"):
        return item.removeprefix("lm_eval/")
    return item


def _best_metric_rows(snr: pd.DataFrame) -> pd.DataFrame:
    frame = snr[snr["primary_metric_kind"].isin(SMOOTH_METRIC_KINDS)].copy()
    frame = frame[frame["signal_n"].eq(242) & frame["noise_n"].eq(10)].copy()
    frame["item"] = [
        _metric_item(metric, metric_kind)
        for metric, metric_kind in zip(frame["metric"], frame["primary_metric_kind"], strict=True)
    ]
    frame["family"] = frame["metric"].map(_metric_family)
    frame["source_class"] = frame["metric"].map(_source_class)
    frame["item_label"] = frame["item"].map(_short_item_label)
    frame = frame.sort_values("signal_to_noise", ascending=False)
    best = frame.groupby("item", as_index=False, sort=False).head(1).copy()
    best = best.sort_values("signal_to_noise", ascending=False).reset_index(drop=True)
    best["rank"] = best.index + 1
    return best


def _hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "family=%{customdata[1]}<br>"
        "source=%{customdata[2]}<br>"
        "selected metric=%{customdata[3]}<br>"
        "metric kind=%{customdata[4]}<br>"
        "signal mean=%{customdata[5]:.4f}<br>"
        "signal range=%{customdata[6]:.4f}-%{customdata[7]:.4f}<br>"
        "noise std=%{customdata[8]:.4f}<br>"
        "SNR=%{y:.2f}<extra></extra>"
    )


def _build_figure(points: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    max_rank = int(points["rank"].max())
    for (family, source_class), group in points.groupby(["family", "source_class"], sort=False):
        customdata = group[
            [
                "item_label",
                "family",
                "source_class",
                "metric",
                "primary_metric_kind",
                "signal_mean",
                "signal_min",
                "signal_max",
                "noise_scale",
            ]
        ].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=group["rank"],
                y=group["signal_to_noise"],
                customdata=customdata,
                mode="markers",
                name=f"{family} · {source_class}",
                marker={
                    "color": FAMILY_COLORS.get(family, "#BAB0AC"),
                    "symbol": SOURCE_SYMBOLS.get(source_class, "circle"),
                    "size": 9 if source_class != "eval BPB/loss" else 8,
                    "line": {"color": "white", "width": 0.6},
                    "opacity": 0.86,
                },
                hovertemplate=_hover_template(),
            )
        )
    for threshold, label in ((1, "SNR=1"), (2, "SNR=2"), (5, "SNR=5"), (10, "SNR=10")):
        fig.add_trace(
            go.Scatter(
                x=[1, max_rank],
                y=[threshold, threshold],
                mode="lines",
                name=label,
                showlegend=False,
                line={"color": PAPER_AXIS, "dash": "dash", "width": 1},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[max_rank],
                y=[threshold],
                mode="text",
                text=[label],
                textposition="middle right",
                textfont={"size": 13, "color": PAPER_MUTED},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return fig


def _summary(points: pd.DataFrame) -> dict[str, object]:
    thresholds = {
        f"snr_ge_{threshold:g}": int((points["signal_to_noise"] >= threshold).sum())
        for threshold in (1, 2, 5, 10)
    }
    return {
        "num_ranked_items": len(points),
        "threshold_counts": thresholds,
        "family_counts": points["family"].value_counts().to_dict(),
        "source_class_counts": points["source_class"].value_counts().to_dict(),
        "top_20": points.head(20)[
            ["rank", "item_label", "family", "source_class", "metric", "primary_metric_kind", "signal_to_noise"]
        ].to_dict(orient="records"),
    }


def main() -> None:
    args = _parse_args()
    points = _best_metric_rows(pd.read_csv(args.snr_csv, low_memory=False))
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output_stem.with_name(args.output_stem.name + "_points.csv"), index=False)
    args.output_stem.with_name(args.output_stem.name + "_summary.json").write_text(
        json.dumps(_summary(points), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fig = _build_figure(points)
    configure_interactive_layout(
        fig,
        title="300M metric signal-to-noise by best available metric per eval slice",
        x_title="Eval/task slice rank by SNR",
        y_title="Signal-to-noise ratio, variable-subset noise",
    )
    fig.update_yaxes(type="log")
    fig.update_yaxes(range=[math.log10(0.25), math.log10(60)])
    fig.update_yaxes(tickvals=[0.3, 0.5, 1, 2, 5, 10, 20, 50], ticktext=["0.3", "0.5", "1", "2", "5", "10", "20", "50"])
    fig.write_html(args.output_stem.with_suffix(".html"), include_plotlyjs="cdn")

    static_fig = _build_figure(points)
    configure_static_layout(
        static_fig,
        x_title="Eval/task slice rank by SNR",
        y_title="Signal-to-noise ratio",
    )
    static_fig.update_yaxes(type="log")
    static_fig.update_yaxes(range=[math.log10(0.25), math.log10(60)])
    static_fig.update_yaxes(
        tickvals=[0.3, 0.5, 1, 2, 5, 10, 20, 50],
        ticktext=["0.3", "0.5", "1", "2", "5", "10", "20", "50"],
    )
    static_fig.update_layout(legend={"font": {"size": 13}, "orientation": "v", "x": 1.01, "y": 0.98})
    static_fig.update_layout(margin={"l": 82, "r": 260, "t": 54, "b": 92})
    write_static_images(static_fig, args.output_stem)
    print(json.dumps(_summary(points), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
