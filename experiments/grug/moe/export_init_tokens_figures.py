#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull eval/train history from V1 (baseline) and V4 (positional tanh) MoE runs.

Writes JSON cache + plotly figures (HTML for local inspection, PNG for posting
to GitHub). Mirrors the style of `~/open-athena.github.io/scripts/delphi/
export_figures.py` — Open Athena palette, transparent backgrounds, no template.
"""

import argparse
import json
import os
from pathlib import Path

import plotly.graph_objects as go
import wandb

OUT_DIR = Path(__file__).parent / "figures" / "init_tokens"
CACHE_PATH = OUT_DIR / "runs.json"

RUNS = {
    "V1 (baseline)": "marin-community/marin/4_10_test_moe",
    "V4 (tanh(pos/10))": "marin-community/marin/4_10_positional_weight_moe",
}

# Open Athena palette (excerpt — Blue + Burnt Umber for two-series figs).
BRAND_BLUE = "#385C8F"
BRAND_GOLD = "#8F6B38"
BRAND_TEAL = "#388F8D"
BRAND_BLACK = "#1F1E1B"
SERIES_COLORS = {"V1 (baseline)": BRAND_BLUE, "V4 (tanh(pos/10))": BRAND_GOLD}

EVAL_METRICS = [
    "eval/loss",
    "eval/loss_last500",
    "eval/macro_loss",
    "eval/macro_loss_last500",
    "eval/bpb",
    "eval/bpb_last500",
    "eval/macro_bpb",
    "eval/macro_bpb_last500",
    "eval/paloma/macro_loss",
    "eval/paloma/macro_loss_last500",
    "eval/uncheatable_eval/macro_loss",
    "eval/uncheatable_eval/macro_loss_last500",
]
TRAIN_METRICS = ["train/loss", "train/cross_entropy_loss"]


def fetch(out: Path) -> dict:
    api = wandb.Api()
    cache: dict = {}
    for label, path in RUNS.items():
        run = api.run(path)
        keys = [k for k in EVAL_METRICS + TRAIN_METRICS if k in run.summary]
        # Always fetch _step for x-axis alignment.
        history = run.history(keys=keys + ["_step"], pandas=False, samples=10000)
        cache[label] = {
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "summary": {k: float(run.summary[k]) for k in keys},
            "history": history,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=2, default=float))
    return cache


def _series(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in history:
        v = row.get(key)
        if v is None:
            continue
        xs.append(row["_step"])
        ys.append(v)
    return xs, ys


def _strip_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=None, margin=dict(t=50, r=40, b=60, l=70), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def fig_train_loss(cache: dict) -> go.Figure:
    fig = go.Figure()
    for label, run in cache.items():
        xs, ys = _series(run["history"], "train/loss")
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", name=label, line=dict(width=2, color=SERIES_COLORS[label]),
                hovertemplate="step=%{x}<br>train/loss=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Training step")
    fig.update_yaxes(title_text="train/loss")
    fig.update_layout(title="Train loss — V1 baseline vs V4 tanh(pos/10)")
    return _strip_layout(fig)


def fig_eval_macro(cache: dict, *, paloma: bool) -> go.Figure:
    base_key = "eval/paloma/macro_loss" if paloma else "eval/uncheatable_eval/macro_loss"
    last_key = base_key + "_last500"
    title_tag = "paloma" if paloma else "uncheatable_eval"
    fig = go.Figure()
    for label, run in cache.items():
        color = SERIES_COLORS[label]
        x_full, y_full = _series(run["history"], base_key)
        x_last, y_last = _series(run["history"], last_key)
        fig.add_trace(
            go.Scatter(
                x=x_full, y=y_full, mode="lines+markers", name=f"{label} — full",
                line=dict(width=2, color=color),
                marker=dict(size=7, color=color),
                hovertemplate=f"step=%{{x}}<br>{base_key}=%{{y:.3f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_last, y=y_last, mode="lines+markers", name=f"{label} — last 500",
                line=dict(width=2, color=color, dash="dash"),
                marker=dict(size=7, color=color, symbol="diamond"),
                hovertemplate=f"step=%{{x}}<br>{last_key}=%{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Training step")
    fig.update_yaxes(title_text=f"{title_tag} macro_loss")
    fig.update_layout(title=f"Eval macro_loss ({title_tag}) — full vs last 500 tokens")
    return _strip_layout(fig)


def fig_per_tag_final(cache: dict) -> go.Figure:
    """Final per-tag macro_loss for paloma + uncheatable, V1 vs V4, full + last500."""
    api = wandb.Api()
    rows = []
    for label, path in RUNS.items():
        run = api.run(path)
        for k, v in run.summary.items():
            if not isinstance(v, (int, float)):
                continue
            for prefix in ("eval/paloma/", "eval/uncheatable_eval/"):
                if k.startswith(prefix) and (k.endswith("/macro_loss") or k.endswith("/macro_loss_last500")):
                    tag = k[len(prefix) : -len("/macro_loss_last500" if k.endswith("/macro_loss_last500") else "/macro_loss")]
                    is_last = k.endswith("_last500")
                    rows.append({"label": label, "tag": f"{prefix.split('/')[1]}/{tag}", "metric": "last500" if is_last else "full", "value": float(v)})
    fig = go.Figure()
    tags = sorted({r["tag"] for r in rows})
    for label in cache:
        for metric, dash in [("full", "solid"), ("last500", "dot")]:
            ys = [next((r["value"] for r in rows if r["label"] == label and r["tag"] == t and r["metric"] == metric), None) for t in tags]
            fig.add_trace(
                go.Bar(
                    x=tags, y=ys,
                    name=f"{label} — {metric}",
                    marker_color=SERIES_COLORS[label],
                    marker_pattern_shape="" if metric == "full" else "/",
                    opacity=1.0 if metric == "full" else 0.85,
                    hovertemplate="%{x}<br>" + f"{label} {metric}=%{{y:.3f}}<extra></extra>",
                )
            )
    fig.update_xaxes(title_text="dataset tag", tickangle=-45)
    fig.update_yaxes(title_text="final macro_loss")
    fig.update_layout(title="Final per-dataset macro_loss — V1 vs V4 (full vs last 500)", barmode="group")
    return _strip_layout(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true", help="Force refetch from W&B.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refetch or not CACHE_PATH.exists():
        if "WANDB_API_KEY" not in os.environ:
            raise SystemExit("Set WANDB_API_KEY before refetching.")
        cache = fetch(CACHE_PATH)
    else:
        cache = json.loads(CACHE_PATH.read_text())

    figs = {
        "train_loss": fig_train_loss(cache),
        "paloma_macro_loss": fig_eval_macro(cache, paloma=True),
        "uncheatable_macro_loss": fig_eval_macro(cache, paloma=False),
        "per_tag_final": fig_per_tag_final(cache),
    }

    for name, fig in figs.items():
        html_path = OUT_DIR / f"{name}.html"
        png_path = OUT_DIR / f"{name}.png"
        fig.write_html(html_path, include_plotlyjs="cdn")
        fig.write_image(png_path, width=1100, height=600, scale=2)
        print(f"  wrote {html_path}")
        print(f"  wrote {png_path}")


if __name__ == "__main__":
    main()
